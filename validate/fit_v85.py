import json
from pathlib import Path
import numpy as np
import pandas as pd

from validate_context_consensus import roc_auc, stratified_folds, fit_logistic, predict_logistic

def load_metrics(cohort):
    path = Path("output") / f"phase_screening_metrics_{cohort}.json"
    frame = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    return frame

def get_model():
    hall = load_metrics("hall")
    
    y = hall["y"].to_numpy(int)
    X = hall[["nightMean", "asymFriction"]].to_numpy(float)
    
    valid = ~np.isnan(X).any(axis=1)
    y_valid = y[valid]
    X_valid = X[valid]

    # Full fit
    model = fit_logistic(X_valid, y_valid, monotonic=False)
    
    # Evaluate 20x5 CV
    aucs = []
    for seed in range(20):
        preds = np.zeros_like(y_valid, dtype=float)
        for train, test in stratified_folds(y_valid, 5, 20260810 + seed):
            m = fit_logistic(X_valid[train], y_valid[train], monotonic=False)
            preds[test] = predict_logistic(m, X_valid[test])
        aucs.append(roc_auc(y_valid, preds))
        
    print(f"Hall (Untreated) - nightMean + asymFriction")
    print(f"20x5 CV AUC: {np.mean(aucs):.4f}")
    
    # Recover original coefficients
    # z = (X - mean)/scale
    # logit = beta[0] + beta[1]*z1 + beta[2]*z2
    # logit = beta[0] + beta[1]*(X1 - m1)/s1 + beta[2]*(X2 - m2)/s2
    # logit = (beta[0] - beta[1]*m1/s1 - beta[2]*m2/s2) + (beta[1]/s1)*X1 + (beta[2]/s2)*X2
    b0 = model["beta"][0] - model["beta"][1]*model["mean"][0]/model["scale"][0] - model["beta"][2]*model["mean"][1]/model["scale"][1]
    b1 = model["beta"][1]/model["scale"][0]
    b2 = model["beta"][2]/model["scale"][1]
    
    print(f"Intercept: {b0:.6f}")
    print(f"Coef nightMean: {b1:.6f}")
    print(f"Coef asymFriction: {b2:.6f}")

    # For Colas (Treated)
    colas = load_metrics("colas")
    y_c = colas["y"].to_numpy(int)
    X_c = colas[["nightMean", "asymFriction"]].to_numpy(float)
    valid_c = ~np.isnan(X_c).any(axis=1)
    y_c = y_c[valid_c]
    X_c = X_c[valid_c]
    
    # Fit separate model for treated
    model_c = fit_logistic(X_c, y_c, monotonic=False)
    aucs_c = []
    for seed in range(20):
        preds_c = np.zeros_like(y_c, dtype=float)
        for train, test in stratified_folds(y_c, 5, 20260810 + seed):
            m = fit_logistic(X_c[train], y_c[train], monotonic=False)
            preds_c[test] = predict_logistic(m, X_c[test])
        aucs_c.append(roc_auc(y_c, preds_c))
        
    print(f"\nColas (Treated) - nightMean + asymFriction")
    print(f"20x5 CV AUC: {np.mean(aucs_c):.4f}")
    
    bc0 = model_c["beta"][0] - model_c["beta"][1]*model_c["mean"][0]/model_c["scale"][0] - model_c["beta"][2]*model_c["mean"][1]/model_c["scale"][1]
    bc1 = model_c["beta"][1]/model_c["scale"][0]
    bc2 = model_c["beta"][2]/model_c["scale"][1]
    
    print(f"Intercept: {bc0:.6f}")
    print(f"Coef nightMean: {bc1:.6f}")
    print(f"Coef asymFriction: {bc2:.6f}")

if __name__ == "__main__":
    get_model()