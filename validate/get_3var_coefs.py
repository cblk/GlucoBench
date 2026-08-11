import numpy as np
import pandas as pd
from validate_context_consensus import load_metrics, fit_logistic

hall = load_metrics("hall")
colas = load_metrics("colas")

print("--- 3-Variable Model for Hall (Untreated) ---")
X_hall = hall[["nightMean", "workIntegral", "ascendFriction"]].to_numpy(float)
y_hall = hall["y"].to_numpy(int)
valid_hall = ~np.isnan(X_hall).any(axis=1) & ~np.isinf(X_hall).any(axis=1)
model_hall = fit_logistic(X_hall[valid_hall], y_hall[valid_hall], monotonic=False)

# logit = beta[0] - sum(beta[i]*mean[i]/scale[i]) + sum(beta[i]/scale[i] * X[i])
b0_h = model_hall["beta"][0] - sum(model_hall["beta"][i+1]*model_hall["mean"][i]/model_hall["scale"][i] for i in range(3))
b1_h = model_hall["beta"][1]/model_hall["scale"][0]
b2_h = model_hall["beta"][2]/model_hall["scale"][1]
b3_h = model_hall["beta"][3]/model_hall["scale"][2]
print(f"Intercept: {b0_h:.6f}")
print(f"Coef nightMean: {b1_h:.6f}")
print(f"Coef workIntegral: {b2_h:.6f}")
print(f"Coef ascendFriction: {b3_h:.6f}")

print("\n--- 3-Variable Model for Colas (Treated) ---")
X_colas = colas[["nightMean", "nightFriction", "ascendFriction"]].to_numpy(float)
y_colas = colas["y"].to_numpy(int)
valid_colas = ~np.isnan(X_colas).any(axis=1) & ~np.isinf(X_colas).any(axis=1)
model_colas = fit_logistic(X_colas[valid_colas], y_colas[valid_colas], monotonic=False)

b0_c = model_colas["beta"][0] - sum(model_colas["beta"][i+1]*model_colas["mean"][i]/model_colas["scale"][i] for i in range(3))
b1_c = model_colas["beta"][1]/model_colas["scale"][0]
b2_c = model_colas["beta"][2]/model_colas["scale"][1]
b3_c = model_colas["beta"][3]/model_colas["scale"][2]
print(f"Intercept: {b0_c:.6f}")
print(f"Coef nightMean: {b1_c:.6f}")
print(f"Coef nightFriction: {b2_c:.6f}")
print(f"Coef ascendFriction: {b3_c:.6f}")
