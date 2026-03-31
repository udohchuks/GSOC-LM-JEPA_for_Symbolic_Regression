import torch
import numpy as np
import sys
import os

sys.path.append(os.getcwd())
from data.aif_dataset import AIFDataset
from evaluation.evaluate import _reconstruct_X
import sympy

try:
    equations = torch.load('C:/Users/chukw/GSOC-LM-JEPA_for_Symbolic_Regression/cache/aif_preprocessed.pt', map_location='cpu', weights_only=False)
except FileNotFoundError:
    print("Cannot find cached data locally.")
    sys.exit()
    
dataset = AIFDataset(equations)
eq = dataset.equations[0]
print(f"Eq: {eq.eq_id}")
X_raw = _reconstruct_X(eq)
print(f"X_raw shape: {X_raw.shape}, dtype: {X_raw.dtype}")
print(f"X_raw sample:\n{X_raw[:5]}")
print(f"y_true sample:\n{eq.y[:5]}")

f_true = sympy.lambdify([sympy.Symbol(v) for v in eq.var_names], sympy.sympify(eq.formula_str), 'numpy')
y_pred = f_true(*X_raw.T)
print(f"y_pred sample (from X_raw):\n{y_pred[:5]}")
