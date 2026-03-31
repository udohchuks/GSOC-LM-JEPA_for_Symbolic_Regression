import torch
import numpy as np
import sys
import os

sys.path.append(os.getcwd())
from data.aif_dataset import AIFDataset
from evaluation.evaluate import _reconstruct_X
from inference.beam_search import fit_and_score
import sympy

equations = torch.load('C:/Users/chukw/GSOC-LM-JEPA_for_Symbolic_Regression/cache/aif_preprocessed.pt', map_location='cpu', weights_only=False)
dataset = AIFDataset(equations)

target_eq = None
for eq in dataset.equations:
    if "I.15.10" in eq.eq_id or "I.12.4" in eq.eq_id or eq.eq_id == "III.10.19":
        target_eq = eq
        if eq.eq_id == "I.15.10":
            break

if target_eq is None:
    print("Could not find equation")
    sys.exit()

print(f"Testing Eq: {target_eq.eq_id}")
print(f"True Formula: {target_eq.formula_str}")
print(f"Variables: {target_eq.var_names}")

X_raw = _reconstruct_X(target_eq)
y_true = target_eq.y

# Ground truth expression!
test_expr = "m_0/sqrt(1 - v**2/c**2)" if target_eq.eq_id == "I.15.10" else target_eq.formula_str

res = fit_and_score(test_expr, ["mock"], X_raw, y_true, target_eq.var_names, max_iter=10)
print(f"fit_and_score Result:")
print(res)

import sklearn.metrics
f = sympy.lambdify([sympy.Symbol(v) for v in target_eq.var_names], sympy.sympify(test_expr), 'numpy')
y_pred = f(*X_raw.T)

print("y_pred stats:")
print(f"NaNs: {np.isnan(y_pred).sum()}")
print(f"Infs: {np.isinf(y_pred).sum()}")

mask = np.isfinite(y_pred) & np.isfinite(y_true)
print(f"Valid points: {np.sum(mask)} out of {len(y_pred)}")

r2 = sklearn.metrics.r2_score(y_true[mask], y_pred[mask])
print(f"Manual R2: {r2}")
