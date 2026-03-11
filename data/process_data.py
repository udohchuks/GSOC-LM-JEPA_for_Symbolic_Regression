"""
Builds processed_records.pkl from raw Feynman dataset.
Each record contains everything needed for training:
  - tokens (View B)
  - sampled normalised X (View A input)
  - original X (for R² evaluation)
  - metadata (var_map, n_vars, formula etc.)

"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer.run_tokenizer import formula_to_tokens, test_roundtrip
from tokenizer.vocab import build_vocab, save_vocab, load_vocab, PAD_ID, SOS_ID, EOS_ID

#Config

RAW_DIR      = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed')
CSV_PATH     = os.path.join(RAW_DIR, 'raw', 'FeynmanEquations.csv')

MAX_EQ_LEN   = 40       # covers all equations (max was 38)
N_SAMPLES    = 5000      # rows to sample per equation for training
RANDOM_SEED  = 42

def load_raw_data(filename: str) -> np.ndarray:
    """Load a raw Feynman data file."""
    path = os.path.join(RAW_DIR, 'raw', 'Feynman_with_units', filename)
    return np.loadtxt(path)

def process_one_equation(row: pd.Series, tok2id: dict) -> dict | None:
    """
    Process one equation row from the CSV into a training record.

    Returns None if the equation should be skipped.
    """
    formula = str(row['Formula'])
    filename = str(row['Filename'])
    n_vars   = int(row['# variables'])

    var_names = []

    for i in range(1, n_vars + 1):
        name = row.get(f"v{i}_name", None)
        if pd.notna(name):
            var_names.append(str(name))
    
    try:
        tokens, var_map, var_map_inv, num_values = formula_to_tokens(formula)
    except Exception as e:
        print(f"  SKIP (tokenize error): {formula} — {e}")
        return None
    
    # Verify round-trip
    if not test_roundtrip(formula, verbose=False):
        print(f"  SKIP (round-trip fail): {formula}")
        return None
    
    token_ids = [SOS_ID] + \
     [tok2id.get(t, PAD_ID) for t in tokens] + [EOS_ID]
    
    if len(token_ids) > MAX_EQ_LEN:
        token_ids = token_ids[:MAX_EQ_LEN -1] + [EOS_ID]
    else:
        token_ids = token_ids + [PAD_ID] * (MAX_EQ_LEN - len(token_ids))
    
    # Load raw data file
    try:
        data = load_raw_data(filename)
    except Exception as e:
        print(f"  SKIP (data load error): {filename} — {e}")
        return None
    
    X_orig = data[:, :-1].astype(np.float32)
    y = data[:, -1].astype(np.float32)

    rng     = np.random.RandomState(RANDOM_SEED)
    indices = rng.choice(len(data), size=N_SAMPLES, replace=False)

    X_sample = X_orig[indices]
    y_sample = y[indices]

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_sample).astype(np.float32)
    X_norm = np.clip(X_norm, -5, 5)

    y_mean = y_sample.mean()
    y_std = y_sample.std() + 1e-6
    y_norm = ((y_sample - y_mean) / y_std).astype(np.float32)
    y_norm = np.clip(y_norm, -5, 5)

    return {
        # Identity
        'filename':    filename,
        'formula':     formula,
        'n_vars':      n_vars,
        'var_names':   var_names,

        # Tokenization
        'tokens':      tokens, 
        'token_ids':   token_ids, 
        'var_map':     var_map,        
        'var_map_inv': var_map_inv,      
        'num_values':  num_values,      

        # Numerical data — normalised (fed to model)
        'X_norm':      X_norm,  
        'y_norm':      y_norm,       
        'scaler':      scaler,   

        # Original data — for R² evaluation only, never fed to model
        'X_orig':      X_sample, 
        'y_orig':      y_sample,   

        # Sequence info
        'token_len':   len(tokens) + 2,  # +2 for SOS and EOS
    }


def build_splits(records: list) -> tuple[list, list, list]:
    """
    Split records into train/val/test at the equation level.
    80% train, 10% val, 10% test.
    Split at equation level — not row level — so the model
    is evaluated on equations it has never seen.
    """
    n = len(records)
    rng = np.random.RandomState(RANDOM_SEED)
    indices = rng.permutation(n)

    train_end = int(0.8 * n)
    val_end   = int(0.9 * n)

    train_idx = indices[:train_end]
    val_idx   = indices[train_end:val_end]
    test_idx  = indices[val_end:]

    train = [records[i] for i in train_idx]
    val   = [records[i] for i in val_idx]
    test  = [records[i] for i in test_idx]

    return train, val, test


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    # ── Step 1: Build vocabulary ──────────────────────────────────────────
    print("Building vocabulary...")
    tok2id, id2tok = build_vocab(CSV_PATH)
    save_vocab(tok2id, id2tok, PROCESSED_DIR)
    print(f"Vocabulary size: {len(tok2id)}")
    print()


    print("Processing equations...")
    df      = pd.read_csv(CSV_PATH)
    records = []

    for _, row in df.iterrows():
        formula = str(row['Formula'])
        record = process_one_equation(row, tok2id)
        if record is not None:
            records.append(record)

    print(f"\nProcessed : {len(records)} equations")
    print(f"Skipped   : {len(df) - len(records)} equations")



      # ── Step 3: Split into train/val/test ─────────────────────────────────
    print("\nSplitting into train/val/test...")
    train, val, test = build_splits(records)
    print(f"Train : {len(train)} equations")
    print(f"Val   : {len(val)} equations")
    print(f"Test  : {len(test)} equations")


    # ── Step 4: Save ──────────────────────────────────────────────────────
    print("\nSaving...")
    splits = {'train': train, 'val': val, 'test': test}
    save_path = os.path.join(PROCESSED_DIR, 'records.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(splits, f)
    print(f"Saved to {save_path}")

    # ── Step 5: Verify ────────────────────────────────────────────────────
    print("\nVerifying load...")
    with open(save_path, 'rb') as f:
        loaded = pickle.load(f)
    assert len(loaded['train']) == len(train)
    assert len(loaded['val'])   == len(val)
    assert len(loaded['test'])  == len(test)

    # Print one example record
    ex = loaded['train'][0]
    print("\nExample record:")
    print(f"  formula    : {ex['formula']}")
    print(f"  n_vars     : {ex['n_vars']}")
    print(f"  tokens     : {ex['tokens']}")
    print(f"  token_ids  : {ex['token_ids']}")
    print(f"  X_norm shape: {ex['X_norm'].shape}")
    print(f"  y_norm shape: {ex['y_norm'].shape}")
    print(f"  X_orig shape: {ex['X_orig'].shape}")
    print(f"  token_len  : {ex['token_len']}")
    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()