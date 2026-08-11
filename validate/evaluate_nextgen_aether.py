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
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = rankdata(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

def load_metrics(cohort):
    path = Path("output") / f"phase_screening_metrics_{cohort}.json"
    frame = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    return frame

def evaluate_features():
    hall = load_metrics("hall")
    colas = load_metrics("colas")

    features = [
        "asymFriction",
        "workIntegral"
    ]

    print("=== Next-Gen Aether Metrics Evaluation ===")
    for name, df in [("Hall (Untreated)", hall), ("Colas (Treated)", colas)]:
        y = df["y"].to_numpy(int)
        
        print(f"\nCohort: {name} (N={len(y)}, Positives: {np.sum(y)})")
        print("-" * 40)
        
        for feature in features:
            if feature not in df.columns:
                print(f"{feature}: Not found in dataframe")
                continue
                
            val = df[feature].to_numpy(float)
            
            valid = ~np.isnan(val)
            y_valid = y[valid]
            val_valid = val[valid]

            if len(y_valid) == 0:
                continue
                
            auc = roc_auc(y_valid, val_valid)
            auc_inverse = roc_auc(y_valid, -val_valid)
            best_auc = max(auc, auc_inverse)
            direction = "Positive" if auc > auc_inverse else "Negative"
            
            mean_healthy = np.mean(val_valid[y_valid == 0])
            mean_abnormal = np.mean(val_valid[y_valid == 1])

            print(f"Feature: {feature}")
            print(f"  Valid: {len(y_valid)}")
            print(f"  Healthy Mean: {mean_healthy:.4f}")
            print(f"  Abnormal Mean: {mean_abnormal:.4f}")
            print(f"  ROC AUC: {best_auc:.4f} ({direction})")

if __name__ == "__main__":
    evaluate_features()
