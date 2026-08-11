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

import sys
import os

# Add local path to import internal logistic fit
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validate.validate_context_consensus import fit_logistic, stratified_folds, roc_auc

def evaluate_cv(X, y, splits=5, repeats=20):
    aucs = []
    for r in range(repeats):
        folds = stratified_folds(y, n_splits=splits, seed=42+r)
        for train_idx, test_idx in folds:
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            model = fit_logistic(X_train, y_train, monotonic=False)
            
            # predict test
            z = np.zeros(len(X_test))
            for i in range(X.shape[1]):
                z += model["beta"][i+1] * (X_test[:, i] - model["mean"][i]) / model["scale"][i]
            z += model["beta"][0]
            scores = 1 / (1 + np.exp(-z))
            
            try:
                aucs.append(roc_auc(y_test, scores))
            except:
                pass
    return aucs

def evaluate_features():
    hall = load_metrics("hall")
    colas = load_metrics("colas")

    features = [
        "asymFriction",
        "workIntegral",
        "dayFriction",
        "nightFriction",
        "ascendFriction",
        "frictionGradient",
        "earlyDelay",
        "relaxationTime",
        "nightAR1"
    ]

    print("=== Next-Gen Aether Metrics Evaluation ===")
    for name, df in [("Hall (Untreated)", hall), ("Colas (Treated)", colas)]:
        y = df["y"].to_numpy(int)
        
        # Derived feature: Friction Ratio
        if "dayFriction" in df.columns and "nightFriction" in df.columns:
            df["frictionRatio"] = df["dayFriction"] / df["nightFriction"]
            features.append("frictionRatio")
            
        print(f"\nCohort: {name} (N={len(y)}, Positives: {np.sum(y)})")
        print("-" * 40)
        
        for feature in set(features):
            if feature not in df.columns:
                continue
                
            val = df[feature].to_numpy(float)
            valid = ~np.isnan(val) & ~np.isinf(val)
            y_valid = y[valid]
            val_valid = val[valid]

            if len(y_valid) == 0:
                continue
                
            h_mean = np.mean(val_valid[y_valid == 0])
            a_mean = np.mean(val_valid[y_valid == 1])
            
            # Use custom roc_auc which handles ties appropriately
            try:
                auc = roc_auc(y_valid, val_valid)
                if auc < 0.5:
                    auc = 1 - auc
                    direction = "Negative"
                else:
                    direction = "Positive"
                    
                print(f"Feature: {feature}")
                print(f"  Valid: {len(y_valid)}")
                print(f"  Healthy Mean: {h_mean:.4f}")
                print(f"  Abnormal Mean: {a_mean:.4f}")
                print(f"  ROC AUC: {auc:.4f} ({direction})")
            except Exception as e:
                print(f"Feature: {feature} - Error: {e}")
        
    print("\n--- Logistic Regression Combination Tests ---")
    
    # Hall untreated combinations
    X_hall1 = hall[["nightMean", "workIntegral"]].to_numpy(float)
    y_h = hall["y"].to_numpy(int)
    valid_h1 = ~np.isnan(X_hall1).any(axis=1)
    
    print("Hall: nightMean + workIntegral AUC:", np.mean(evaluate_cv(X_hall1[valid_h1], y_h[valid_h1])))
    
    X_hall2 = hall[["nightMean", "ascendFriction"]].to_numpy(float)
    valid_h2 = ~np.isnan(X_hall2).any(axis=1)
    print("Hall: nightMean + ascendFriction AUC:", np.mean(evaluate_cv(X_hall2[valid_h2], y_h[valid_h2])))
    
    # Try all three

    # Evaluate combined models with the new metrics
    print("\n--- Logistic Regression Combination Tests ---")
    
    # Hall untreated combinations
    X_hall1 = hall[["nightMean", "workIntegral"]].to_numpy(float)
    y_h = hall["y"].to_numpy(int)
    valid_h1 = ~np.isnan(X_hall1).any(axis=1) & ~np.isinf(X_hall1).any(axis=1)
    
    print("Hall: nightMean + workIntegral AUC:", np.mean(evaluate_cv(X_hall1[valid_h1], y_h[valid_h1])))
    
    if "ascendFriction" in hall.columns:
        X_hall2 = hall[["nightMean", "ascendFriction"]].to_numpy(float)
        valid_h2 = ~np.isnan(X_hall2).any(axis=1) & ~np.isinf(X_hall2).any(axis=1)
        print("Hall: nightMean + ascendFriction AUC:", np.mean(evaluate_cv(X_hall2[valid_h2], y_h[valid_h2])))
        
        X_hall3 = hall[["nightMean", "workIntegral", "ascendFriction"]].to_numpy(float)
        valid_h3 = ~np.isnan(X_hall3).any(axis=1) & ~np.isinf(X_hall3).any(axis=1)
        print("Hall: nightMean + workIntegral + ascendFriction AUC:", np.mean(evaluate_cv(X_hall3[valid_h3], y_h[valid_h3])))

    # Colas combinations
    print(f"Hall: nightMean + relaxationTime + nightAR1 AUC: {np.mean(evaluate_cv(hall[['nightMean', 'relaxationTime', 'nightAR1']].fillna(hall[['nightMean', 'relaxationTime', 'nightAR1']].median()).to_numpy(), hall['y'].to_numpy())):.4f}")
    
    print("")
    y_c = colas["y"].to_numpy(int)
    if "nightFriction" in colas.columns:
        X_c1 = colas[["nightMean", "nightFriction"]].to_numpy(float)
        valid_c1 = ~np.isnan(X_c1).any(axis=1) & ~np.isinf(X_c1).any(axis=1)
        print("Colas: nightMean + nightFriction AUC:", np.mean(evaluate_cv(X_c1[valid_c1], y_c[valid_c1])))
        
    if "ascendFriction" in colas.columns:
        X_c2 = colas[["nightMean", "ascendFriction"]].to_numpy(float)
        valid_c2 = ~np.isnan(X_c2).any(axis=1) & ~np.isinf(X_c2).any(axis=1)
        print("Colas: nightMean + ascendFriction AUC:", np.mean(evaluate_cv(X_c2[valid_c2], y_c[valid_c2])))
        
        if "nightFriction" in colas.columns:
            X_c3 = colas[["nightMean", "nightFriction", "ascendFriction"]].to_numpy(float)
            valid_c3 = ~np.isnan(X_c3).any(axis=1) & ~np.isinf(X_c3).any(axis=1)
            print("Colas: nightMean + nightFriction + ascendFriction AUC:", np.mean(evaluate_cv(X_c3[valid_c3], y_c[valid_c3])))

if __name__ == "__main__":
    evaluate_features()
