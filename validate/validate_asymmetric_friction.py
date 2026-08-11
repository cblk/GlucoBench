import json
from pathlib import Path
import numpy as np
import pandas as pd

def rankdata(values):
    return pd.Series(np.asarray(values, dtype=float)).rank(method="average").to_numpy()

def roc_auc(y, scores):
    y = np.asarray(y, dtype=int)
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    ranks = rankdata(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

def load_metrics(cohort):
    path = Path("output") / f"phase_screening_metrics_{cohort}.json"
    frame = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    return frame

def evaluate_friction():
    hall = load_metrics("hall")
    colas = load_metrics("colas")

    print("=== Asymmetric Friction (Aether Protocol) ===")
    for name, df in [("Hall (Untreated)", hall), ("Colas (Treated)", colas)]:
        y = df["y"].to_numpy(int)
        friction = df["asymFriction"].to_numpy(float)
        
        valid = ~np.isnan(friction)
        y_valid = y[valid]
        friction_valid = friction[valid]

        auc = roc_auc(y_valid, friction_valid)
        auc_inverse = roc_auc(y_valid, -friction_valid)
        best_auc = max(auc, auc_inverse)
        direction = "Positive" if auc > auc_inverse else "Negative"
        
        mean_healthy = np.mean(friction_valid[y_valid == 0])
        mean_abnormal = np.mean(friction_valid[y_valid == 1])

        print(f"\nCohort: {name}")
        print(f"Valid subjects: {len(y_valid)}/{len(y)} (Positives: {np.sum(y_valid)})")
        print(f"Healthy Friction: {mean_healthy:.4f}")
        print(f"Abnormal Friction: {mean_abnormal:.4f}")
        print(f"ROC AUC: {best_auc:.4f} (Direction: {direction})")

if __name__ == "__main__":
    evaluate_friction()
