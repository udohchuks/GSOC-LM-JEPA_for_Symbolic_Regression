"""
AIF Dataset for LLM-JEPA Symbolic Regression.

Loads the Feynman Symbolic Regression Database for evaluation.
AIF is used ONLY as the evaluation set.
Pretraining uses physics-informed synthetic data (synthetic_dataset.py).

Design decisions:
    - No normalisation: IEEE-754 handles raw floats directly.
      Normalisation was rejected because it alters the symbolic target.
    - Preprocessing cached to disk: SymPy parsing + IEEE-754 + unit
      targets computed once, saved as .pt file, loaded on subsequent runs.
    - Row subsampling at __getitem__ time: 100k rows per equation is too
      many to hold in GPU memory for a full batch. Subsample n_rows at
      training time for memory efficiency.
    - Variable padding to max_n_vars=9: different equations have 1-9
      variables. Pad to fixed size so tensors can be stacked into batches.
"""

from __future__ import annotations
import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, List, Dict
import sympy

from data.tokenizer import (
    formula_string_to_rpn, encode_formula,
    MAX_SEQ_LEN, PAD_IDX
)
from data.unit_table import get_unit_matrix, unit_to_class_indices
from data.utils import (
    to_ieee754_16bit,
    compute_unit_targets,
    unit_targets_to_class_indices,
)


# ── Equation metadata ─────────────────────────────────────────────────────────

class EquationMeta:
    """
    Metadata for one Feynman equation parsed from the CSV.

    This is a plain data container — no methods, no logic.
    Keeping metadata separate from preprocessed data means you can
    inspect what an equation is without loading its 100k data points.
    """

    def __init__(
        self,
        eq_id:       str,
        n_vars:      int,
        output_name: str,
        formula_str: str,
        var_names:   List[str],
        var_lows:    List[float],
        var_highs:   List[float],
    ):
        self.eq_id       = eq_id
        self.n_vars      = n_vars
        self.output_name = output_name
        self.formula_str = formula_str
        self.var_names   = var_names
        self.var_lows    = var_lows
        self.var_highs   = var_highs

    def __repr__(self) -> str:
        return (f"EquationMeta(id={self.eq_id!r}, "
                f"n_vars={self.n_vars}, "
                f"formula={self.formula_str!r})")


def parse_equations_csv(csv_path: str | Path) -> List[EquationMeta]:
    """
    Parse the Feynman equations CSV file.

    Actual CSV structure:
        Col 0: Filename (eq_id)
        Col 1: Number (ignored)
        Col 2: Output name
        Col 3: Formula string
        Col 4: # variables (n_vars)
        Col 5,6,7:   v1_name, v1_low, v1_high
        Col 8,9,10:  v2_name, v2_low, v2_high
        ...  (3 columns per variable)
        Up to 10 variables (columns go to col 34)

    First row is a header — skipped automatically.
    File may have a UTF-8 BOM — handled by encoding='utf-8-sig'.
    """
    equations = []

    # utf-8-sig handles the   character at the start of the file
    with open(csv_path, 'r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        # DictReader uses the header row as keys automatically

        for row in reader:
            try:
                eq_id       = row['Filename'].strip()
                n_vars      = int(row['# variables'].strip())
                output_name = row['Output'].strip()
                formula_str = row['Formula'].strip()

                var_names: List[str]   = []
                var_lows:  List[float] = []
                var_highs: List[float] = []

                # Read variable columns: v1_name, v1_low, v1_high, ...
                for i in range(1, n_vars + 1):
                    name_key = f'v{i}_name'
                    low_key  = f'v{i}_low'
                    high_key = f'v{i}_high'

                    name = row.get(name_key, '').strip()
                    low  = row.get(low_key,  '').strip()
                    high = row.get(high_key, '').strip()

                    if not name:
                        break

                    var_names.append(name)
                    var_lows.append(float(low))
                    var_highs.append(float(high))
                

                if len(var_names) != n_vars:
                    print(f"Warning: {eq_id} expected {n_vars} vars, "
                          f"parsed {len(var_names)} — skipping")
                    continue

                equations.append(EquationMeta(
                    eq_id=eq_id,
                    n_vars=n_vars,
                    output_name=output_name,
                    formula_str=formula_str,
                    var_names=var_names,
                    var_lows=var_lows,
                    var_highs=var_highs,
                ))
            except (ValueError, IndexError) as e:
                print(f"Warning: skipping malformed row "
                      f"{row.get('Filename', 'Unknown')!r}: {e}")
                continue
    return equations    


# ── Preprocessed equation ─────────────────────────────────────────────────────

class PreprocessedEquation:
    """
    Fully preprocessed equation ready for the Dataset.

    All expensive computation (SymPy parsing, IEEE-754 conversion,
    unit target computation) is done once and stored here.
    Instances are saved to disk via torch.save and loaded on
    subsequent runs to avoid recomputation.
    """

    def __init__(
        self,
        eq_id:             str,
        X_bits:            np.ndarray,   # [N, n_vars, 16] IEEE-754
        y:                 np.ndarray,   # [N] raw floats
        unit_matrix_idx:   np.ndarray,   # [n_vars, 5] class indices
        token_ids:         np.ndarray,   # [MAX_SEQ_LEN] padded
        unit_targets_idx:  np.ndarray,   # [MAX_SEQ_LEN, 5] class indices
        var_names:         List[str],
        formula_str:       str,
        n_vars:            int,
    ):
        self.eq_id            = eq_id
        self.X_bits           = X_bits
        self.y                = y
        self.unit_matrix_idx  = unit_matrix_idx
        self.token_ids        = token_ids
        self.unit_targets_idx = unit_targets_idx
        self.var_names        = var_names
        self.formula_str      = formula_str
        self.n_vars           = n_vars



def preprocess_equation(
    meta:     EquationMeta,
    data_dir: str | Path,
    max_rows: int = 100_000,
) -> Optional[PreprocessedEquation]:
    """
    Preprocess one AIF equation: load data, encode, tokenise.

    Expected data file location:
        {data_dir}/{meta.eq_id}

    Steps:
        1. Find and load the data file
        2. IEEE-754 encode X
        3. Look up unit vectors
        4. Parse formula to RPN and encode to token indices
        5. Compute unit targets for every token position
        6. Pack everything into PreprocessedEquation

    Args:
        meta:     equation metadata
        data_dir: directory containing data files
        max_rows: rows to load (default 100k = full dataset)

    Returns:
        PreprocessedEquation or None if any step failed.
    """
    # ── Step 1: Find and load data file ──────────────────────────────────
    
    data_dir = Path(data_dir)
    data_path = data_dir / meta.eq_id

    if not data_path.exists():
        print(
            f"Data file not found: {data_path}\n"
            f"  Expected a file named exactly '{meta.eq_id}' "
            f"with no extension in {data_dir}"
        )
        return None

    try:
        data = np.loadtxt(str(data_path), max_rows=max_rows)
    except Exception as e:
        print(f"Warning: could not load {data_path}: {e}")
        return None

    # Handle edge case: single-row file gives 1D array
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    # Validate column count
    expected_cols = meta.n_vars + 1
    if data.shape[1] != expected_cols:
        print(
            f"Column mismatch in {meta.eq_id}: "
            f"expected {expected_cols} columns "
            f"(n_vars={meta.n_vars} + 1 output), "
            f"got {data.shape[1]}"
        )
        return None
    
    X = data[:, :meta.n_vars].astype(np.float32)  # [N, n_vars]
    y = data[:,  meta.n_vars].astype(np.float32)  # [N]

    # ── Step 2: IEEE-754 encode ───────────────────────────────────────────
    # Shape: [N, n_vars, 16]
    # Pre-computed here so __getitem__ only needs to index, not convert
    X_bits = to_ieee754_16bit(X)

    # ── Step 3: Unit vectors ──────────────────────────────────────────────
    unit_matrix     = get_unit_matrix(meta.var_names)   # [n_vars, 5]
    unit_matrix_idx = unit_to_class_indices(unit_matrix) # [n_vars, 5]

    # ── Step 4: Tokenise formula ──────────────────────────────────────────
    try:
        rpn_tokens = formula_string_to_rpn(
            meta.formula_str, meta.var_names
        )
    except ValueError as e:
        print(f"Warning: could not tokenise {meta.eq_id}: {e}")
        return None

    token_ids = np.array(
        encode_formula(
            rpn_tokens,
            add_bos=True,
            add_eos=True,
            pad_to=MAX_SEQ_LEN
        ),
        dtype=np.int64
    )  # shape: [MAX_SEQ_LEN]

    # ── Step 5: Unit targets ──────────────────────────────────────────────
    # compute_unit_targets returns one unit vector per RPN token
    # We need to pad this to MAX_SEQ_LEN to match token_ids
    raw_targets = compute_unit_targets(rpn_tokens, meta.var_names)

    # Pad to match token_ids:
    #   [BOS_units, token_units..., EOS_units, PAD_units...]
    # BOS, EOS, and PAD all get dimensionless [0,0,0,0,0]
    padded = (
        [[0] * 5]          # BOS
        + raw_targets
        + [[0] * 5]        # EOS
    )
    # Truncate or pad to MAX_SEQ_LEN
    while len(padded) < MAX_SEQ_LEN:
        padded.append([0]*5)
    padded = padded[:MAX_SEQ_LEN]

    unit_targets_idx = unit_targets_to_class_indices(padded) # shape: [MAX_SEQ_LEN, 5]

    pad_mask = (token_ids == PAD_IDX)
    unit_targets_idx[pad_mask] = -100

    return PreprocessedEquation(
        eq_id=meta.eq_id,
        X_bits=X_bits,
        y=y,
        unit_matrix_idx=unit_matrix_idx,
        token_ids=token_ids,
        unit_targets_idx=unit_targets_idx,
        var_names=meta.var_names,
        formula_str=meta.formula_str,
        n_vars=meta.n_vars,
    )

# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class AIFDataset(Dataset):
    """
    PyTorch Dataset for AIF evaluation equations.

    Each __getitem__ call:
        1. Subsamples n_rows from the equation's 100k rows
        2. Pads variable dimension from n_vars to max_n_vars
        3. Returns a dict of tensors ready for the model

    Args:
        equations:   list of PreprocessedEquation objects
        n_rows:      rows to subsample per equation per step
        max_n_vars:  pad variable dimension to this size (9 for Feynman)
    """
    def __init__(
        self,
        equations:  List[PreprocessedEquation],
        n_rows:     int = 200,
        max_n_vars: int = 9,
    ):
        self.equations  = equations
        self.n_rows     = n_rows
        self.max_n_vars = max_n_vars
    
    def __len__(self) -> int:
        return len(self.equations)

    def __getitem__(self, idx: int) -> Dict:
        eq     = self.equations[idx]
        n_vars = eq.n_vars
        N      = eq.X_bits.shape[0]   # total available rows

        # ── Subsample rows ────────────────────────────────────────────────
        if N > self.n_rows:
            row_idx  = np.random.choice(N, self.n_rows, replace=False)
            X_bits   = eq.X_bits[row_idx]   # [n_rows, n_vars, 16] or [n_rows, n_vars]
        else:
            X_bits   = eq.X_bits
        
        # ── UNPACK BITS (Compact uint16 format) ─────────────────────────
        # Each uint16 maps to 16 bits. This reduces RAM usage by 8x.
        nr, nv = X_bits.shape
        X_bits = X_bits.view(np.uint8).reshape(nr, nv, 2)
        X_bits = np.unpackbits(X_bits, axis=-1, bitorder='big')
        X_bits = X_bits.reshape(nr, nv, 16)
        # Result is now consistently [n_rows, n_vars, 16]
        
        # ── Pad variable dimension to max_n_vars ──────────────────────────
        # Why pad? Tensors in a batch must have identical shapes.
        # Padding with zeros is safe: the var_mask tells the model
        # which positions are real vs padding.
        pad_vars = self.max_n_vars - n_vars

        if pad_vars > 0:
            # Pad X_bits with zeros along variable axis
            pad_x = np.zeros(
                (X_bits.shape[0], pad_vars, 16), dtype=np.float32
            )
            X_bits = np.concatenate([X_bits, pad_x], axis=1)
            # shape: [n_rows, max_n_vars, 16]

            # Pad unit indices with 4 (= offset for exponent 0 = dimensionless)
            pad_u = np.full((pad_vars, 5), 4, dtype=np.int64)
            unit_idx = np.concatenate([eq.unit_matrix_idx, pad_u], axis=0)
            # shape: [max_n_vars, 5]
        else:
            unit_idx = eq.unit_matrix_idx
        
        # ── Variable mask ─────────────────────────────────────────────────
        # 1.0 for real variables, 0.0 for padding
        # Used in the encoder to ignore padded variable slots
        var_mask = np.zeros(self.max_n_vars, dtype=np.float32)
        var_mask[:n_vars] = 1.0

        return {
            # Encoder inputs
            'X_bits':           torch.from_numpy(X_bits).float(),
            # [n_rows, max_n_vars, 16]

            'unit_idx':         torch.from_numpy(unit_idx).long(),
            # [max_n_vars, 5]

            'var_mask':         torch.from_numpy(var_mask).float(),
            # [max_n_vars]  — 1 for real, 0 for padding

            'n_vars':           torch.tensor(n_vars, dtype=torch.long),
            # scalar

            # Decoder targets
            'token_ids':        torch.from_numpy(eq.token_ids).long(),
            # [MAX_SEQ_LEN]

            'unit_targets_idx': torch.from_numpy(eq.unit_targets_idx).long(),
            # [MAX_SEQ_LEN, 5]

            # Metadata — strings, not tensors
            'eq_id':            eq.eq_id,
            'formula_str':      eq.formula_str,
            'var_names':        eq.var_names,
        }


# ── Collate function ──────────────────────────────────────────────────────────

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for DataLoader.

    Stacks tensor fields normally.
    Collects string fields (eq_id, formula_str, var_names) into lists
    because torch.stack cannot handle strings.

    This function is passed to DataLoader as collate_fn=collate_fn.
    """
    tensor_keys = [
        'X_bits', 'unit_idx', 'var_mask', 'n_vars',
        'token_ids', 'unit_targets_idx',
    ]
    string_keys = ['eq_id', 'formula_str', 'var_names']

    result = {}
    for key in tensor_keys:
        result[key] = torch.stack([item[key] for item in batch])
    for key in string_keys:
        result[key] = [item[key] for item in batch]

    return result

# ── Factory functions ─────────────────────────────────────────────────────────

def build_aif_dataset(
    csv_path:        str | Path,
    data_dir:        str | Path,
    cache_dir:       Optional[str | Path] = None,
    n_rows:          int = 200,
    max_n_vars:      int = 9,
    max_rows_per_eq: int = 10000,
) -> AIFDataset:
    """
    Build the AIF dataset from CSV and data files.

    On first call: parses CSV, preprocesses all equations, optionally
    saves to cache_dir.
    On subsequent calls: loads from cache_dir if available.

    Args:
        csv_path:        path to FeynmanEquations.csv
        data_dir:        directory containing equation data files
        cache_dir:       directory to cache preprocessed data
                         (saves time on subsequent runs)
        n_rows:          rows to subsample per equation per step
        max_n_vars:      pad variable dimension to this
        max_rows_per_eq: rows to load per equation (default: all 100k)

    Returns:
        AIFDataset ready for DataLoader.
    """
    # Check cache first
    if cache_dir is not None:
        cache_path = Path(cache_dir) / 'aif_preprocessed.pt'
        if cache_path.exists():
            print(f"Loading cached AIF data from {cache_path}")
            equations = torch.load(str(cache_path), weights_only=False)
            return AIFDataset(equations, n_rows=n_rows,
                              max_n_vars=max_n_vars)
    # Parse CSV
    print(f"Parsing AIF equations from {csv_path}")
    metas = parse_equations_csv(csv_path)
    print(f"Found {len(metas)} equations")

    # Preprocess each equation
    equations: List[PreprocessedEquation] = []
    n_failed = 0

    for i, meta in enumerate(metas):
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(metas)}: {meta.eq_id}")
        eq = preprocess_equation(meta, data_dir,
                                  max_rows=max_rows_per_eq)
        if eq is not None:
            equations.append(eq)
        else:
            n_failed += 1

    print(f"Preprocessed {len(equations)}/{len(metas)} equations "
          f"({n_failed} failed)")
    
    # Save cache
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache_path = Path(cache_dir) / 'aif_preprocessed.pt'
        torch.save(equations, str(cache_path))
        print(f"Cached to {cache_path}")

    return AIFDataset(equations, n_rows=n_rows, max_n_vars=max_n_vars)

def build_aif_dataloader(
    csv_path:   str | Path,
    data_dir:   str | Path,
    batch_size: int = 32,
    n_rows:     int = 200,
    max_n_vars: int = 9,
    cache_dir:  Optional[str | Path] = None,
    num_workers: int = 2,
    shuffle:    bool = True,
) -> DataLoader:
    """
    Build a DataLoader for AIF data.

    Args:
        csv_path:    path to FeynmanEquations.csv
        data_dir:    directory containing data files
        batch_size:  equations per batch
        n_rows:      data rows subsampled per equation
        max_n_vars:  variable padding size
        cache_dir:   preprocessing cache directory
        num_workers: parallel data loading workers
        shuffle:     randomise equation order each epoch

    Returns:
        DataLoader.
    """
    dataset = build_aif_dataset(
        csv_path=csv_path,
        data_dir=data_dir,
        cache_dir=cache_dir,
        n_rows=n_rows,
        max_n_vars=max_n_vars,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(num_workers > 0),
        persistent_workers=(num_workers > 0),
    )


if __name__ == '__main__':
    import tempfile

    # ── Test 1: CSV parser ────────────────────────────────────────────────
    # Create a minimal test CSV
    test_csv = (
        "Filename,Number,Output,Formula,# variables,"
        "v1_name,v1_low,v1_high,v2_name,v2_low,v2_high\n"
        "I.6.2a,1,f,exp(-theta**2/2)/sqrt(2*pi),1,"
        "theta,1,3\n"
        "I.12.1,2,F,mu*Nn,2,"
        "mu,1,5,Nn,1,5\n"
    )

    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.csv', delete=False
    ) as f:
        f.write(test_csv)
        csv_path = f.name

    metas = parse_equations_csv(csv_path)
    assert len(metas) == 2
    assert metas[0].eq_id == 'I.6.2a'
    assert metas[0].n_vars == 1
    assert metas[0].var_names == ['theta']
    assert metas[0].var_lows == [1.0]
    assert metas[0].var_highs == [3.0]
    assert metas[1].n_vars == 2
    assert metas[1].var_names == ['mu', 'Nn']
    print('CSV parser: OK')

    # ── Test 2: Preprocessing without real data files ─────────────────────
    # Create a minimal synthetic data file
    import tempfile, os
    data_dir = tempfile.mkdtemp()

    # I.6.2a: theta, f(theta) = exp(-theta^2/2)/sqrt(2*pi)
    theta = np.random.uniform(1, 3, 500)
    f_vals = np.exp(-theta**2 / 2) / np.sqrt(2 * np.pi)
    data = np.column_stack([theta, f_vals])
    np.savetxt(os.path.join(data_dir, 'I.6.2a'), data)

    eq = preprocess_equation(metas[0], data_dir, max_rows=500)
    assert eq is not None
    assert eq.X_bits.shape == (500, 1, 16)
    assert eq.unit_matrix_idx.shape == (1, 5)
    assert eq.token_ids.shape == (MAX_SEQ_LEN,)
    assert eq.unit_targets_idx.shape == (MAX_SEQ_LEN, 5)
    assert eq.token_ids[0] == 1   # BOS_IDX
    print(f'Preprocessing: OK — RPN: {eq.token_ids[:10].tolist()}')

    # ── Test 3: Dataset __getitem__ ───────────────────────────────────────
    dataset = AIFDataset([eq], n_rows=50, max_n_vars=9)
    item = dataset[0]

    assert item['X_bits'].shape    == (50, 9, 16)
    assert item['unit_idx'].shape  == (9, 5)
    assert item['var_mask'].shape  == (9,)
    assert item['var_mask'][0]     == 1.0   # real variable
    assert item['var_mask'][1]     == 0.0   # padding
    assert item['token_ids'].shape == (MAX_SEQ_LEN,)
    print('Dataset __getitem__: OK')

    # ── Test 4: DataLoader batch ──────────────────────────────────────────
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    batch = next(iter(loader))

    assert batch['X_bits'].shape    == (1, 50, 9, 16)
    assert batch['unit_idx'].shape  == (1, 9, 5)
    assert batch['var_mask'].shape  == (1, 9)
    assert batch['token_ids'].shape == (1, MAX_SEQ_LEN)
    assert isinstance(batch['eq_id'], list)
    assert batch['eq_id'][0] == 'I.6.2a'
    print('DataLoader batch shapes: OK')
    print(f'  X_bits:    {batch["X_bits"].shape}')
    print(f'  unit_idx:  {batch["unit_idx"].shape}')
    print(f'  token_ids: {batch["token_ids"].shape}')

    # Cleanup
    import shutil
    os.unlink(csv_path)
    shutil.rmtree(data_dir)

    print('\nAll AIF dataset tests passed.')