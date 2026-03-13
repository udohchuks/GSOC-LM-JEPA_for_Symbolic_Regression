import pickle
import numpy as np
import os

pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed', 'records.pkl')
with open(pkl_path, 'rb') as f:
    splits = pickle.load(f)

train = splits['train']
val   = splits['val']
test  = splits['test']

print(f"Train: {len(train)} equations")
print(f"Val:   {len(val)} equations")
print(f"Test:  {len(test)} equations")
print()

# Check one record has everything we need
r = train[0]
print("Keys in record:", list(r.keys()))
print()
print(f"Formula    : {r['formula']}")
print(f"n_vars     : {r['n_vars']}")
print(f"tokens     : {r['tokens']}")
print(f"token_ids  : {r['token_ids']}")
print(f"X_norm     : {r['X_norm'].shape}  dtype={r['X_norm'].dtype}")
print(f"y_norm     : {r['y_norm'].shape}  dtype={r['y_norm'].dtype}")
print(f"X_orig     : {r['X_orig'].shape}  dtype={r['X_orig'].dtype}")
print(f"token_len  : {r['token_len']}")
print()

# Check no NaN or Inf in any record
print("Checking for NaN/Inf across all records...")
issues = []
for split_name, records in splits.items():
    for rec in records:
        if np.any(np.isnan(rec['X_norm'])) or np.any(np.isinf(rec['X_norm'])):
            issues.append(f"{split_name}: {rec['formula']} — X_norm has NaN/Inf")
        if np.any(np.isnan(rec['y_norm'])) or np.any(np.isinf(rec['y_norm'])):
            issues.append(f"{split_name}: {rec['formula']} — y_norm has NaN/Inf")

if issues:
    print("ISSUES FOUND:")
    for i in issues: print(f"  {i}")
else:
    print("All clean — no NaN or Inf found")