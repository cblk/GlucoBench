import numpy as np
from scipy import signal
import json
import traceback

class TensorEngine:
    def __init__(self):
        self.event_log = []
    
    def log_event(self, msg):
        self.event_log.append(msg)

    def extract_tau(self, values_json):
        # Concurrency/Lifecycle Audit fix: self.event_log is a singleton accumulator on this
        # long-lived engine instance. Without a per-call reset, every response's "events" field
        # would leak the ENTIRE session's history (all prior calls, all prior epochs) into the
        # current call's result, which JS then appends onto _eventSequence at every one of the
        # 5 call sites per epoch -- causing unbounded log growth and burying the current epoch's
        # forensic signal under stale, unrelated history. Reset here so each call returns ONLY
        # the log lines it itself generated.
        self.event_log = []
        try:
            values_list = json.loads(values_json)
            self.log_event("[Python Engine L1] Starting ACF Tau Extraction.")
            chunk_indices = []
            cur_indices = []
            for i, v in enumerate(values_list):
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    cur_indices.append(i)
                else:
                    if len(cur_indices) > 0:
                        chunk_indices.append(cur_indices)
                        cur_indices = []
            if len(cur_indices) > 0:
                chunk_indices.append(cur_indices)
                
            chunks = [np.array([values_list[i] for i in idxs], dtype=np.float64) for idxs in chunk_indices if len(idxs) >= 30]
            
            if not chunks:
                self.log_event("[L1 ERROR] No valid chunks >= 30 points for ACF. ABORTING Tau extraction.")
                return json.dumps({"result": None, "events": self.event_log})

            # v9.1 / Blueprint v3.4: raised from 20 to 60 (180min) after the 2026-08-15 Hall
            # wind-tunnel run showed 42/57 subjects pinned at the old 20-step ceiling -- a
            # measurement-ceiling artifact, not a real cross-subject invariant. See
            # reports/wind_tunnel_hall_20260815_2149.md and Blueprint v3.3 Sec 3.1 revision note.
            max_lag = 60
            total_len = sum(len(c) for c in chunks)
            avg_acf = np.zeros(max_lag + 1)
            
            for c in chunks:
                n = len(c)
                mean = np.mean(c)
                var = np.var(c)
                if var < 1e-9:
                    continue
                acf = np.zeros(max_lag + 1)
                acf[0] = 1.0
                for lag in range(1, max_lag + 1):
                    if n <= lag:
                        break
                    acf[lag] = np.sum((c[:-lag] - mean) * (c[lag:] - mean)) / ((n - lag) * var)
                avg_acf += acf * (n / total_len)

            tau = max_lag
            decayed = False
            for i in range(1, max_lag):
                if avg_acf[i] < 0.7:
                    decayed = True
                if avg_acf[i] < 0.3678:
                    tau = i
                    break
                if decayed and avg_acf[i] < avg_acf[i-1] and avg_acf[i] < avg_acf[i+1]:
                    tau = i
                    break

            self.log_event(f"[Python Engine L1] Weighted ACF calculated, locked tau = {tau}")
            return json.dumps({"result": int(tau), "events": self.event_log})
        except Exception as e:
            err = traceback.format_exc()
            self.log_event(f"[Python L1 Error] Tau Extraction: {err}")
            return json.dumps({"error": str(e), "events": self.event_log})

    def estimate_dimension(self, values_json, tau):
        self.event_log = []  # Per-call reset; see extract_tau for rationale.
        try:
            values_list = json.loads(values_json)
            self.log_event(f"[Python Engine L1] Starting FNN & Jacobi Razor with tau={tau}.")
            
            chunk_indices = []
            cur_indices = []
            for i, v in enumerate(values_list):
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    cur_indices.append(i)
                else:
                    if len(cur_indices) > 0:
                        chunk_indices.append(cur_indices)
                        cur_indices = []
            if len(cur_indices) > 0:
                chunk_indices.append(cur_indices)
                
            chunks = [np.array([values_list[i] for i in idxs], dtype=np.float64) for idxs in chunk_indices if len(idxs) >= 30]
            
            if not chunks:
                self.log_event("[L1 ERROR] No valid chunks for Dimension Estimation. ABORTING.")
                return json.dumps({"result": None, "events": self.event_log})
                
            all_vals = np.concatenate(chunks)
            R_A = np.std(all_vals)
            if R_A < 1e-3: R_A = 1e-3
            
            R_tol = 15.0
            A_tol = 2.0
            
            def build_phase_space(c, m, t):
                n = len(c)
                if n <= (m-1)*t:
                    return np.empty((0, m))
                return np.array([c[i : i + m*t : t] for i in range(n - (m-1)*t)])
                
            m_fnn = 10
            prev_fnn_ratio = 1.0
            from scipy.spatial import cKDTree
            
            for m in range(1, 11):
                pts_m = []
                pts_m_plus_1 = []
                for c in chunks:
                    p_m = build_phase_space(c, m, tau)
                    p_m1 = build_phase_space(c, m+1, tau)
                    if len(p_m1) > 0:
                        pts_m.append(p_m[:len(p_m1)])
                        pts_m_plus_1.append(p_m1)
                
                if not pts_m:
                    m_fnn = max(1, m-1)
                    break
                    
                pts_m = np.concatenate(pts_m)
                pts_m_plus_1 = np.concatenate(pts_m_plus_1)
                
                if len(pts_m) < 10:
                    m_fnn = max(1, m-1)
                    break
                    
                tree = cKDTree(pts_m)
                # k=2 because nearest neighbor to itself is distance 0 (index 0)
                distances, indices = tree.query(pts_m, k=2)
                if distances.shape[1] < 2:
                    m_fnn = max(1, m-1)
                    break
                    
                nn_dists = distances[:, 1]
                nn_indices = indices[:, 1]
                
                fnn_count = 0
                for i in range(len(pts_m)):
                    R_d = nn_dists[i]
                    if R_d < 1e-6:
                        R_d = 1e-6
                    
                    coord_diff = np.abs(pts_m_plus_1[i, -1] - pts_m_plus_1[nn_indices[i], -1])
                    
                    if coord_diff / R_d > R_tol:
                        fnn_count += 1
                    elif np.sqrt(R_d**2 + coord_diff**2) / R_A > A_tol:
                        fnn_count += 1
                        
                fnn_ratio = fnn_count / len(pts_m)
                self.log_event(f"  [FNN] m={m}, ratio={fnn_ratio:.4f}")
                
                if fnn_ratio < 0.05 or fnn_ratio > prev_fnn_ratio:
                    m_fnn = m
                    break
                prev_fnn_ratio = fnn_ratio
                
            m_fnn = int(max(2, min(m_fnn, 10)))
            self.log_event(f"[Python Engine L1] FNN locked m_fnn={m_fnn}.")
            
            # Jacobi Razor
            pts_fnn = []
            for c in chunks:
                p = build_phase_space(c, m_fnn, tau)
                if len(p) > 0:
                    pts_fnn.append(p)
                    
            if not pts_fnn:
                self.log_event("[L1 ERROR] FNN points empty. ABORTING.")
                return json.dumps({"result": None, "events": self.event_log})

            pts_fnn = np.concatenate(pts_fnn)
            if len(pts_fnn) < m_fnn:
                self.log_event(f"[L1 ERROR] FNN points ({len(pts_fnn)}) < m_fnn ({m_fnn}). ABORTING.")
                return json.dumps({"result": None, "events": self.event_log})

            cov = np.cov(pts_fnn, rowvar=False)
            if m_fnn == 1:
                self.log_event("[L1 ERROR] Covariance matrix is 1D (m_fnn=1), cannot calculate eigenvalues. ABORTING.")
                return json.dumps({"result": None, "events": self.event_log})

            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1]

            total_var = np.sum(eigvals)
            if total_var < 1e-9:
                self.log_event("[L1 ERROR] Total variance < 1e-9. System is frozen. ABORTING.")
                return json.dumps({"result": None, "events": self.event_log})
                
            cum_var = 0.0
            m_eff = m_fnn
            for i, val in enumerate(eigvals):
                cum_var += max(0, val)
                if cum_var / total_var >= 0.99:
                    m_eff = i + 1
                    break
                    
            m_eff = int(max(2, min(m_eff, m_fnn)))
            self.log_event(f"[Python Engine L1] Jacobi Razor effective dim m_eff={m_eff} (from {m_fnn}).")
            return json.dumps({"result": m_eff, "events": self.event_log})
        except Exception as e:
            err = traceback.format_exc()
            self.log_event(f"[Python L1 Error] Dimension Estimation: {err}")
            return json.dumps({"error": str(e), "events": self.event_log})
            
    def compute_rqa(self, points_json, tau):
        self.event_log = []  # Per-call reset; see extract_tau for rationale.
        try:
            # Contract v1.3 4.4 Spatiotemporal Alignment: the caller MUST pass the FULL point
            # array including None gap markers. Squashing nulls before this call would collapse
            # the array index onto the compacted position, silently converting the Theiler window
            # from a "physical time" filter into a meaningless "array position" filter, and would
            # let diagonal-line scans splice together two chunks separated by a real data dropout.
            pts = json.loads(points_json)
            orig_idx = []
            valid_pts = []
            for i, p in enumerate(pts):
                if p is not None and len(p) >= 2:
                    orig_idx.append(i)
                    valid_pts.append(p)
            n = len(valid_pts)
            if n < 10:
                self.log_event(f"[L2 ERROR] Insufficient points for RQA ({n} < 10). ABORTING RQA.")
                return json.dumps({"result": None, "events": self.event_log})

            orig_idx = np.array(orig_idx, dtype=np.int64)
            arr = np.array(valid_pts, dtype=np.float64)
            T = int(max(5, tau))
            target_rr = 0.05

            # Using scipy.spatial.distance.pdist for efficient pair-wise distance
            from scipy.spatial.distance import pdist, squareform
            condensed_dists = pdist(arr, metric='euclidean')

            # Max expected points is ~4000 (14 days = 6720 / chunking).
            # memory for 4000x4000 float64 is ~128MB, very safe in Pyodide.
            dist_mat = squareform(condensed_dists)

            # Theiler mask keyed on TRUE physical-time separation (orig_idx), not compacted
            # array position, so a gap does not falsely shrink or inflate the exclusion band.
            idx_gap = np.abs(orig_idx[:, None] - orig_idx[None, :])
            mask = np.triu(idx_gap > T)
            valid_dists = dist_mat[mask]

            if len(valid_dists) == 0:
                self.log_event("[L2 ERROR] Theiler window excluded all points (T >= N/2). ABORTING RQA.")
                return json.dumps({"result": None, "events": self.event_log})

            # Find epsilon to achieve 5% RR (5th percentile of the valid distances)
            epsilon = float(np.percentile(valid_dists, target_rr * 100))
            if epsilon < 1e-9: epsilon = 1e-9

            # Build recurrence matrix (upper triangle only for calculation)
            R = (dist_mat < epsilon) & mask

            rr = float(np.sum(R)) / len(valid_dists)

            # Chunk-boundary marker: True where compacted positions m, m+1 are ALSO physically
            # adjacent (no dropout between them). A diagonal "line" may only extend across a
            # step where this holds for BOTH tracks of the pair; otherwise it is a seam artifact.
            contiguous = (np.diff(orig_idx) == 1) if n > 1 else np.array([], dtype=bool)

            diag_lengths = []
            l_min = 2
            total_recurrent_pts = int(np.sum(R))

            for k in range(T + 1, n):
                diag = np.diag(R, k=k)
                L = len(diag)
                if L < l_min:
                    continue
                if L > 1:
                    can_extend = contiguous[0:L-1] & contiguous[k:k+L-1]
                    barrier_idx = np.where(~can_extend)[0] + 1
                    segments = np.split(diag, barrier_idx) if len(barrier_idx) > 0 else [diag]
                else:
                    segments = [diag]
                for seg in segments:
                    if len(seg) == 0:
                        continue
                    padded = np.concatenate(([False], seg, [False]))
                    diffs = np.diff(padded.astype(int))
                    starts = np.where(diffs == 1)[0]
                    ends = np.where(diffs == -1)[0]
                    lengths = ends - starts
                    for l in lengths:
                        if l >= l_min:
                            diag_lengths.append(int(l))

            det = 0.0
            entr = 0.0

            if len(diag_lengths) > 0 and total_recurrent_pts > 0:
                det = sum(diag_lengths) / total_recurrent_pts
                lengths_arr = np.array(diag_lengths)
                unique_lengths, counts = np.unique(lengths_arr, return_counts=True)
                probs = counts / np.sum(counts)
                entr = float(-np.sum(probs * np.log(probs)))

            # Pack recurrence plot for JS (sparse representation, compacted-index space; purely
            # a rendering convenience and does not feed back into DET/ENTR/RR math above).
            rp_y, rp_x = np.where(dist_mat < epsilon)

            rp_coords = []
            if len(rp_x) < 50000:
                rp_coords = [int(i) for i in rp_x]
                rp_coords_y = [int(i) for i in rp_y]
            else:
                stride = len(rp_x) // 50000 + 1
                rp_coords = [int(i) for i in rp_x[::stride]]
                rp_coords_y = [int(i) for i in rp_y[::stride]]

            res = {
                "rr": rr,
                "det": det,
                "entr": entr,
                "rp_x": rp_coords,
                "rp_y": rp_coords_y
            }
            self.log_event(f"[Python Engine L2] RQA computed. T={T} (physical steps), eps={epsilon:.4f}, DET={det:.3f}, ENTR={entr:.3f}, chunk_seams={int(np.sum(~contiguous)) if n > 1 else 0}")
            return json.dumps({"result": res, "events": self.event_log})
        except Exception as e:
            err = traceback.format_exc()
            self.log_event(f"[Python L2 Error] RQA Calculation: {err}")
            return json.dumps({"error": str(e), "events": self.event_log})

    def filter_chunks(self, values_json, order, Wn):
        self.event_log = []  # Per-call reset; see extract_tau for rationale.
        try:
            values = json.loads(values_json)
            out = list(values)
            chunk_indices = []
            cur_indices = []
            for i, v in enumerate(values):
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    cur_indices.append(i)
                else:
                    if len(cur_indices) > 0:
                        chunk_indices.append(cur_indices)
                        cur_indices = []
            if len(cur_indices) > 0:
                chunk_indices.append(cur_indices)

            b, a = signal.butter(int(order), float(Wn), btype='low')
            for idxs in chunk_indices:
                chunk = np.array([values[i] for i in idxs], dtype=np.float64)
                if len(chunk) >= 12:
                    padlen = min(3 * max(len(a), len(b)), len(chunk) - 1)
                    filtered = signal.filtfilt(b, a, chunk, padlen=padlen)
                    for k, i in enumerate(idxs):
                        out[i] = float(filtered[k])
                else:
                    for k, i in enumerate(idxs):
                        out[i] = float(chunk[k])
            self.log_event(f"[Python Engine L0] Zero-phase Butterworth (order={order}, Wn={Wn}) applied on {len(chunk_indices)} chunks.")
            return json.dumps({"result": out, "events": self.event_log})
        except Exception as e:
            err = traceback.format_exc()
            self.log_event(f"[Python Filter Error] {err}")
            return json.dumps({"error": str(e), "events": self.event_log})

    def compute_work_integral(self, points_json):
        self.event_log = []  # Per-call reset; see extract_tau for rationale.
        try:
            # Contract v1.3 4.1 Spatiotemporal Alignment: caller MUST pass the FULL array
            # (nulls included, on the fixed-step resampled grid). We never pre-filter to a
            # compacted array here -- distance is only accumulated between two ADJACENT grid
            # slots that are BOTH valid, which is exactly the Chunking Engine boundary rule:
            # a null gap between them silently breaks the sum instead of teleporting across it.
            pts = json.loads(points_json)
            n = len(pts)
            valid_count = sum(1 for p in pts if p is not None and len(p) >= 2)
            if valid_count < 2:
                self.log_event("[L0 ERROR] Insufficient points for Work Integral (<2). ABORTING calculation.")
                return json.dumps({"result": None, "events": self.event_log})

            xs = np.full(n, np.nan, dtype=np.float64)
            ys = np.full(n, np.nan, dtype=np.float64)
            for i, p in enumerate(pts):
                if p is not None and len(p) >= 2:
                    xs[i] = p[0]
                    ys[i] = p[1]

            dx = np.diff(xs)
            dy = np.diff(ys)
            dist = np.sqrt(dx * dx + dy * dy)
            # NaN propagates from either endpoint being invalid/missing, so this automatically
            # drops any step that would have crossed a chunk boundary.
            raw_work = float(np.nansum(dist))

            # 24h Time-Normalization Mandate (Contract v1.3 4.1 / Blueprint v3.3 L2):
            # normalize to a 480-point (3-min step) equivalent basis so the thermodynamic bill
            # stays comparable across epochs with different data-loss rates.
            val = raw_work / (valid_count / 480.0)
            self.log_event(f"[Python Engine L2] Work Integral computed. raw={raw_work:.4f}, valid_pts={valid_count}, normalized={val:.4f}")
            return json.dumps({"result": val, "events": self.event_log})
        except Exception as e:
            err = traceback.format_exc()
            self.log_event(f"[Python L2 Error] Work Integral: {err}")
            return json.dumps({"error": str(e), "events": self.event_log})

engine = TensorEngine()
