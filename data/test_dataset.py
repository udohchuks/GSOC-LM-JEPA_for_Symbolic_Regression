import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import FeynmanDataset, collate_fn, N_VIEW_A
from tokenizer.vocab import load_vocab
from torch.utils.data import DataLoader

# Load data
processed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed')
with open(os.path.join(processed_dir, 'records.pkl'), 'rb') as f:
    splits = pickle.load(f)

tok2id, id2tok = load_vocab(processed_dir)
VOCAB_SIZE = len(tok2id)
N_BINS     = 64
TOTAL_VOCAB = VOCAB_SIZE + N_BINS   # equation tokens + bin tokens

# Build dataset
train_ds = FeynmanDataset(splits['train'], tok2id, VOCAB_SIZE)
val_ds   = FeynmanDataset(splits['val'],   tok2id, VOCAB_SIZE)
test_ds  = FeynmanDataset(splits['test'],  tok2id, VOCAB_SIZE)

print(f"Train: {len(train_ds)} equations")
print(f"Val:   {len(val_ds)} equations")
print(f"Test:  {len(test_ds)} equations")
print(f"Vocab size (equation): {VOCAB_SIZE}")
print(f"Vocab size (total):    {TOTAL_VOCAB}")
print()

# # Test single item
# view_a, view_b, n_vars = train_ds[0]
# print(f"Single item:")
# print(f"  view_a shape : {view_a.shape}")
# print(f"  view_b shape : {view_b.shape}")
# print(f"  n_vars       : {n_vars}")
# print(f"  view_a[:10]  : {view_a[:10].tolist()}")
# print(f"  view_b[:10]  : {view_b[:10].tolist()}")
# print()

tok2id, _ = load_vocab(processed_dir)
all_records = splits['train'] + splits['val'] + splits['test']

max_vars = max(r['n_vars'] for r in all_records)
max_view_a_len = N_VIEW_A * (1 + max_vars + 1)
print(f"Max n_vars     : {max_vars}")
print(f"Max view_a len : {max_view_a_len}")
print(f"Max view_b len : 40")
print(f"Total max ctx  : {max_view_a_len + 40 + 1} (view_a + view_b + PRED)")

# Test DataLoader with batch
loader = DataLoader(train_ds, batch_size=8,
                    shuffle=True, collate_fn=collate_fn)
batch  = next(iter(loader))
va, vb, nv = batch

print(f"Batch:")
print(f"  view_a : {va.shape}  min={va.min()} max={va.max()}")
print(f"  view_b : {vb.shape}  min={vb.min()} max={vb.max()}")
print(f"  n_vars : {nv.tolist()}")
print()

# Sanity check — bin tokens should be >= VOCAB_SIZE
assert va.max() < TOTAL_VOCAB, "Bin token exceeds total vocab!"
assert vb.max() < VOCAB_SIZE,  "Equation token exceeds vocab!"
assert va.min() >= 0,          "Negative token ID found!"
print("All assertions passed.")