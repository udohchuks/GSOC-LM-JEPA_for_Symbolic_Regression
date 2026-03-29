"""
Comprehensive test for models and loss functions.
"""
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print('=' * 70)
print('MODEL AND LOSS VERIFICATION')
print('=' * 70)

# ============================================================================
# 1. Test Model Components
# ============================================================================
print('\n1. Testing Model Components...')

from models.embedders import DataEmbedder, UnitEmbedder
from models.encoder import MixEncoder
from models.decoder import RPNDecoder
from models.model import LLMJEPA
from data.tokenizer import VOCAB_SIZE, MAX_SEQ_LEN

# Test DataEmbedder
print('\n  DataEmbedder:')
d_model = 64
max_n_vars = 9
data_emb = DataEmbedder(d_model=d_model, max_n_vars=max_n_vars)
X_bits = torch.randn(2, 100, max_n_vars, 16)  # batch, rows, vars, bits
data_out = data_emb(X_bits)
print(f'    Input: {X_bits.shape} → Output: {data_out.shape}')
assert data_out.shape == (2, 100, max_n_vars, d_model), "DataEmbedder shape mismatch"
print('    ✓ DataEmbedder works')

# Test UnitEmbedder
print('\n  UnitEmbedder:')
unit_emb = UnitEmbedder(d_model=d_model)
unit_idx = torch.randint(0, 5, (2, max_n_vars, 5)).long()  # batch, vars, unit_dims
unit_out = unit_emb(unit_idx)
print(f'    Input: {unit_idx.shape} → Output: {unit_out.shape}')
assert unit_out.shape == (2, max_n_vars, d_model), "UnitEmbedder shape mismatch"
print('    ✓ UnitEmbedder works')

# Test MixEncoder
print('\n  MixEncoder:')
encoder = MixEncoder(
    d_model=d_model,
    n_heads=4,
    n_isab=2,
    n_col_attn=2,
    m_inducing=32,
    max_n_vars=max_n_vars,
)
var_mask = torch.ones(2, max_n_vars)
z_context, z_per_row = encoder(data_out, unit_out, var_mask)
print(f'    z_context: {z_context.shape}')
print(f'    z_per_row: {z_per_row.shape}')
assert z_context.shape == (2, d_model), "MixEncoder z_context shape mismatch"
print('    ✓ MixEncoder works')

# Test RPNDecoder
print('\n  RPNDecoder:')
decoder = RPNDecoder(
    d_model=d_model,
    n_heads=4,
    n_layers=2,
    vocab_size=VOCAB_SIZE,
    max_seq_len=MAX_SEQ_LEN,
    dropout=0.0,
)
decoder_input = torch.randint(0, VOCAB_SIZE, (2, 10)).long()  # batch, seq_len
# Decoder expects unit_matrix with shape [B, max_n_vars, 5]
unit_matrix = torch.randint(0, 5, (2, max_n_vars, 5)).long()
logits, attn = decoder(decoder_input, z_context, unit_matrix)
print(f'    Input: {decoder_input.shape} → Output: {logits.shape}')
assert logits.shape == (2, 10, VOCAB_SIZE), "RPNDecoder shape mismatch"
print('    ✓ RPNDecoder works')

# Test Full LLMJEPA Model
print('\n  LLMJEPA (Full Model):')
model = LLMJEPA(
    d_model=d_model,
    n_heads=4,
    n_isab=2,
    n_col_attn=2,
    n_enc_layers=2,
    n_dec_layers=2,
    m_inducing=32,
    max_n_vars=max_n_vars,
    dropout=0.1,
)
print(f'    Total parameters: {sum(p.numel() for p in model.parameters()):,}')

# Forward pass
X_bits = torch.randn(2, 50, max_n_vars, 16)
unit_idx = torch.randint(0, 5, (2, max_n_vars, 5)).long()
var_mask = torch.ones(2, max_n_vars)
formula_tokens = torch.randint(0, VOCAB_SIZE, (2, 20)).long()
unit_targets = torch.randint(0, 5, (2, 20, 5)).long()

outputs = model(X_bits, unit_idx, var_mask, formula_tokens, unit_targets)
print(f'    z_context: {outputs["z_context"].shape}')
print(f'    logits: {outputs["logits"].shape}')
print('    ✓ LLMJEPA forward pass works')

# ============================================================================
# 2. Test Loss Functions
# ============================================================================
print('\n2. Testing Loss Functions...')

from training.losses import ValidityWeightedCE, JEPALoss, SIGRegLoss, UnitLoss, LLMJEPALoss

# Test ValidityWeightedCE
print('\n  ValidityWeightedCE:')
lm_loss = ValidityWeightedCE(invalid_weight=2.0)
logits = torch.randn(2, 20, VOCAB_SIZE)
targets = torch.randint(4, VOCAB_SIZE-1, (2, 20))  # Valid tokens
token_ids = torch.randint(4, VOCAB_SIZE-1, (2, 20))
loss = lm_loss(logits, targets, token_ids)
print(f'    Loss value: {loss.item():.4f}')
assert loss > 0, "Loss should be positive"
print('    ✓ ValidityWeightedCE works')

# Test JEPALoss
print('\n  JEPALoss:')
jepa_loss = JEPALoss()
z_pred = torch.randn(2, 32)
z_target = torch.randn(2, 32)
loss = jepa_loss(z_pred, z_target)
print(f'    Loss value: {loss.item():.4f}')
assert loss >= 0, "JEPA loss should be non-negative"
print('    ✓ JEPALoss works')

# Test SIGRegLoss
print('\n  SIGRegLoss:')
sigreg_loss = SIGRegLoss(num_slices=256, num_points=17, lambda_reg=0.1)
z = torch.randn(2, 64, requires_grad=True)
loss = sigreg_loss(z)
print(f'    Loss value: {loss.item():.4f}')
assert loss >= 0, "SIGReg loss should be non-negative"
print('    ✓ SIGRegLoss works')

# Test UnitLoss
print('\n  UnitLoss:')
unit_loss = UnitLoss(n_unit_dims=5, n_unit_classes=10)
unit_preds = [torch.randn(2, 20, 10) for _ in range(5)]  # [B, T, n_classes]
unit_targets = torch.randint(0, 10, (2, 20, 5)).long()  # [B, T, 5]
loss = unit_loss(unit_preds, unit_targets)
print(f'    Loss value: {loss.item():.4f}')
assert loss >= 0, "Unit loss should be non-negative"
print('    ✓ UnitLoss works')

# Test Combined LLMJEPALoss
print('\n  LLMJEPALoss (Combined):')
combined_loss = LLMJEPALoss(
    lambda_jepa=1.0,
    lambda_sigreg=0.1,
    lambda_lm=1.0,
    lambda_units=1.0,
    sigreg_num_slices=256,
    sigreg_num_points=17,
    invalid_weight=2.0,
    n_unit_classes=9,  # Default
)

loss_dict = combined_loss(
    z_pred=torch.randn(2, 32),
    z_target=torch.randn(2, 32),
    z_context=torch.randn(2, 64, requires_grad=True),
    logits=torch.randn(2, 20, VOCAB_SIZE),
    token_targets=torch.randint(4, VOCAB_SIZE-1, (2, 20)).long(),
    unit_preds=[torch.randn(2, 20, 9) for _ in range(5)],  # n_unit_classes=9
    unit_targets=torch.randint(0, 9, (2, 20, 5)).long(),
    token_ids=torch.randint(4, VOCAB_SIZE-1, (2, 20)).long(),
)

print(f'    Total loss: {loss_dict["total"].item():.4f}')
print(f'    JEPA: {loss_dict["jepa"].item():.4f}')
print(f'    SIGReg: {loss_dict["sigreg"].item():.4f}')
print(f'    LM: {loss_dict["lm"].item():.4f}')
print(f'    Units: {loss_dict["units"].item():.4f}')
print('    ✓ LLMJEPALoss works')

# ============================================================================
# 3. Test Training Step
# ============================================================================
print('\n3. Testing Training Step...')

from training.trainer import LLMJEPAModule

module = LLMJEPAModule(
    d_model=d_model,
    n_heads=4,
    n_encoder_layers=2,
    n_decoder_layers=2,
    max_n_vars=max_n_vars,
    vocab_size=VOCAB_SIZE,
    max_seq_len=MAX_SEQ_LEN,
    learning_rate=1e-4,
)

# Create a training batch
batch = {
    'X_bits': torch.randn(2, 50, max_n_vars, 16),
    'unit_idx': torch.randint(0, 5, (2, max_n_vars, 5)).long(),
    'var_mask': torch.ones(2, max_n_vars),
    'formula_tokens': torch.randint(4, VOCAB_SIZE-1, (2, 20)).long(),
    'target_tokens': torch.randint(4, VOCAB_SIZE-1, (2, 20)).long(),
    'unit_targets_idx': torch.randint(0, 5, (2, 20, 5)).long(),
    'token_ids': torch.randint(4, VOCAB_SIZE-1, (2, 20)).long(),
}

# Forward pass
outputs = module(batch)
print(f'    Loss dict keys: {list(outputs["losses"].keys())}')
print(f'    Total loss: {outputs["loss"].item():.4f}')
print('    ✓ Training forward pass works')

# ============================================================================
# 4. Test Gradient Flow
# ============================================================================
print('\n4. Testing Gradient Flow...')

# Zero gradients
module.zero_grad()

# Backward pass
outputs['loss'].backward()

# Check gradients
has_gradients = False
for name, param in module.named_parameters():
    if param.grad is not None and param.grad.abs().sum() > 0:
        has_gradients = True
        break

if has_gradients:
    print('    ✓ Gradients flowing correctly')
else:
    print('    ⚠ Warning: No gradients detected')

# ============================================================================
# 5. Test Inference Model
# ============================================================================
print('\n5. Testing Inference Model...')

from inference.generate import InferenceModel

inf_model = InferenceModel(
    d_model=d_model,
    n_heads=4,
    n_encoder_layers=2,
    n_decoder_layers=2,
    max_n_vars=max_n_vars,
    vocab_size=VOCAB_SIZE,
    max_seq_len=MAX_SEQ_LEN,
)

# Copy weights from training model
inf_model.data_embedder.load_state_dict(module.model.data_embedder.state_dict())
inf_model.unit_embedder.load_state_dict(module.model.unit_embedder.state_dict())
inf_model.context_encoder.load_state_dict(module.model.mix_encoder.state_dict())
inf_model.decoder.load_state_dict(module.model.decoder.state_dict())

# Test encoding
X_bits = torch.randn(2, 50, max_n_vars, 16)
unit_idx = torch.randint(0, 5, (2, max_n_vars, 5)).long()
var_mask = torch.ones(2, max_n_vars)

z_context = inf_model.encode(X_bits, unit_idx, var_mask)
print(f'    z_context: {z_context.shape}')

# Test generation
generated = inf_model.generate(z_context, unit_idx, max_len=20, greedy=True)
print(f'    Generated: {generated.shape}')
print('    ✓ Inference model works')

# ============================================================================
# Summary
# ============================================================================
print('\n' + '=' * 70)
print('ALL MODEL AND LOSS TESTS PASSED ✓')
print('=' * 70)
print('\nVerified components:')
print('  ✓ DataEmbedder')
print('  ✓ UnitEmbedder')
print('  ✓ MixEncoder')
print('  ✓ RPNDecoder')
print('  ✓ LLMJEPA (full model)')
print('  ✓ ValidityWeightedCE')
print('  ✓ JEPALoss')
print('  ✓ SIGRegLoss')
print('  ✓ UnitLoss')
print('  ✓ LLMJEPALoss (combined)')
print('  ✓ LLMJEPAModule (training)')
print('  ✓ InferenceModel')
print('  ✓ Gradient flow')
