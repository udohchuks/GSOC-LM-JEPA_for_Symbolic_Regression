"""
PyTorch Dataset class for LM-JEPA symbolic regression.

Each item returned is a tuple of three tensors:
  view_a   : [seq_len_a]  numerical context tokens (binned X, y values)
  view_b   : [MAX_EQ_LEN] equation token IDs (padded)
  n_vars   : int          number of variables (used for masking)

View A is freshly sampled every time __getitem__ is called.
This means each epoch the model sees different numerical examples
for the same equation — effectively infinite augmentation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ── Config ────────────────────────────────────────────────────────────────
N_VIEW_A   = 20     # number of (X, y) rows to pack into View A
N_BINS     = 64     # number of bins for numerical value quantisation
BIN_MIN    = -5.0   # min value after normalisation (we clipped to this)
BIN_MAX    =  5.0   # max value after normalisation


def quantise(values: np.ndarray, n_bins: int = N_BINS,
    vmin: float = BIN_MIN, vmax: float = BIN_MAX) -> np.ndarray:
    """
    Convert continuous normalised values into discrete bin indices.

    Example with N_BINS=4, range [-5, 5]:
      Bucket 0: [-5.0, -2.5)
      Bucket 1: [-2.5,  0.0)
      Bucket 2: [ 0.0,  2.5)
      Bucket 3: [ 2.5,  5.0]

    A value of 1.3 falls in bucket 2.
    A value of -3.1 falls in bucket 0
    """
    clipped = np.clip(values, vmin, vmax)

    normalised = (clipped - vmin) / (vmax - vmin)
    bins = (normalised * n_bins).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)

    return bins.astype(np.int64)



class FeynmanDataset(Dataset):
    def __init__(
        self,
        records: list,
        tok2id:     dict,
        vocab_size: int,
        n_view_a:   int  = N_VIEW_A,
        n_bins:     int  = N_BINS,
        max_eq_len: int  = 40
    ):
        """
        Args:
            records:    list of processed record dicts from records.pkl
            tok2id:     token → ID mapping from vocab
            vocab_size: total vocabulary size (equation tokens)
            n_view_a:   number of (X,y) rows to sample for View A
            n_bins:     number of bins for numerical quantisation
            max_eq_len: maximum equation token sequence length
        """
        self.records    = records
        self.tok2id     = tok2id
        self.vocab_size = vocab_size
        self.n_view_a   = n_view_a
        self.n_bins     = n_bins
        self.max_eq_len = max_eq_len

        self.bin_offset = vocab_size

        # Special token IDs
        self.pad_id  = tok2id['[PAD]']   # 0
        self.sos_id  = tok2id['[SOS]']   # 1
        self.eos_id  = tok2id['[EOS]']   # 2
        self.pred_id = tok2id['[PRED]']  # 3
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        rec = self.records[idx]
        n_vars = rec['n_vars']
        # This gives different examples each epoch — free augmentation
        n_available = len(rec['X_norm'])
        sel         = np.random.choice(n_available, size=self.n_view_a,
                                       replace=False)
        X_sel = rec['X_norm'][sel]   # [n_view_a, n_vars]
        y_sel = rec['y_norm'][sel]   # [n_view_a]

        view_a_tokens = []

        for i in range(self.n_view_a):
            view_a_tokens.append(self.sos_id)
            x_bins = quantise(X_sel[i], self.n_bins)

            for b in x_bins:
                view_a_tokens.append(self.bin_offset + int(b))
            y_bin = quantise(np.array([y_sel[i]]), self.n_bins)[0]
            view_a_tokens.append(self.bin_offset + int(y_bin))
        
        view_b_tokens = rec['token_ids']
        # ── Convert to tensors ────────────────────────────────────────────
        view_a = torch.tensor(view_a_tokens, dtype=torch.long)
        view_b = torch.tensor(view_b_tokens, dtype=torch.long)

        return view_a, view_b, n_vars

def collate_fn(batch: list) -> tuple:
    """
    Collate a list of (view_a, view_b, n_vars) tuples into batched tensors.

    view_a sequences should all be the same length since we always
    sample exactly n_view_a rows with the same number of vars... 
    except n_vars differs per equation. So we pad view_a too.
    """
    view_a_list = [item[0] for item in batch]
    view_b_list = [item[1] for item in batch]
    n_vars_list = [item[2] for item in batch]

    max_a = max(v.shape[0] for v in view_a_list)
    max_b = max(v.shape[0] for v in view_b_list)
    pad_id = 0

    def pad_sequence(seq: torch.Tensor, target_len: int) -> torch.Tensor:
        if seq.shape[0] < target_len:
            padding  = torch.full((target_len - seq.shape[0],), pad_id, dtype=torch.long)
            return torch.cat([seq, padding])
        return seq[:target_len]
    
    view_a_padded = torch.stack([pad_sequence(v, max_a) for v in view_a_list])
    view_b_padded = torch.stack([pad_sequence(v, max_b) for v in view_b_list])
    n_vars_tensor = torch.tensor(n_vars_list, dtype=torch.long)

    return view_a_padded, view_b_padded, n_vars_tensor

