import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from model.vanilla_transformer import (
    CustomTransformer, CONFIG,
    build_causal_mask, build_jepa_mask
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = CustomTransformer(CONFIG).to(device)
print(f"Parameters: {model.count_parameters():,}")

B      = 4
len_a  = CONFIG['len_a']
len_b  = CONFIG['len_b']

# ── Test 1: Baseline causal LM ────────────────────────────────────────────
print("\nTest 1: Baseline causal LM")
ids  = torch.randint(0, CONFIG['vocab_size'], (B, len_b)).to(device)
mask = build_causal_mask(len_b, device)
h    = model(ids, mask)
logits = model.get_lm_logits(h)
print(f"  hidden:  {h.shape}")
print(f"  logits:  {logits.shape}")
assert h.shape     == (B, len_b, 512)
assert logits.shape == (B, len_b, 110)
print("  PASSED")

# ── Test 2: JEPA forward pass ─────────────────────────────────────────────
print("\nTest 2: JEPA forward pass")
view_a   = torch.randint(0, 110, (B, len_a)).to(device)
view_b   = torch.randint(0, 46,  (B, len_b)).to(device)
pred_tok = torch.full((B, 1), 3, dtype=torch.long).to(device)
full_seq = torch.cat([view_a, view_b, pred_tok], dim=1)

mask   = build_jepa_mask(len_a, len_b, device)
hidden = model(full_seq, mask)

z_pred   = hidden[:, -1, :]
pad_mask = (view_b != 0).float().unsqueeze(-1)
hidden_b = hidden[:, len_a:-1, :]
z_target = (hidden_b * pad_mask).sum(1) / pad_mask.sum(1).clamp(min=1e-8)

loss_jepa = 1 - F.cosine_similarity(z_pred, z_target, dim=-1).mean()
print(f"  z_pred:    {z_pred.shape}")
print(f"  z_target:  {z_target.shape}")
print(f"  loss_jepa: {loss_jepa.item():.4f}")
print("  PASSED")

# ── Test 3: LM loss ───────────────────────────────────────────────────────
print("\nTest 3: LM loss")
lm_logits = model.get_lm_logits(hidden[:, len_a:-1, :])
loss_lm   = F.cross_entropy(
    lm_logits[:, :-1].reshape(-1, 110),
    view_b[:, 1:].reshape(-1),
    ignore_index=0
)
print(f"  loss_lm: {loss_lm.item():.4f}")
print("  PASSED")

# ── Test 4: Combined loss + backward ─────────────────────────────────────
print("\nTest 4: Combined loss + backward")
loss = loss_lm + 1.0 * loss_jepa
loss.backward()
print(f"  total loss: {loss.item():.4f}")
print("  PASSED")

print("\nAll tests passed.")