"""Simple parameter counter for small model."""
import yaml

# Load small config
with open('configs/small.yaml', 'r') as f:
    cfg = yaml.safe_load(f)['model']

print('='*60)
print('TINY MODEL CONFIG (small.yaml)')
print('='*60)
print()
print('Architecture:')
print(f"  d_model:      {cfg['d_model']}")
print(f"  n_enc_layers: {cfg['n_enc_layers']}")
print(f"  n_dec_layers: {cfg['n_dec_layers']}")
print(f"  n_isab:       {cfg.get('n_isab', 2)}")
print(f"  n_col_attn:   {cfg.get('n_col_attn', 2)}")
print(f"  m_inducing:   {cfg.get('m_inducing', 32)}")
print(f"  max_n_vars:   {cfg['max_n_vars']}")
print()
print('Predictor (KEPT SMALL):')
pred = cfg['predictor']
print(f"  bottleneck_ratio: {pred['pred_bottleneck_ratio']}")
print(f"  pred_n_heads:     {pred['pred_n_heads']}")
print(f"  pred_dropout:     {pred['pred_dropout']}")
print()

# Rough estimate based on architecture
d = cfg['d_model']
enc = cfg['n_enc_layers']
dec = cfg['n_dec_layers']
isab = cfg.get('n_isab', 2)
col = cfg.get('n_col_attn', 2)
ind = cfg.get('m_inducing', 32)
vars = cfg['max_n_vars']
bottleneck = pred['pred_bottleneck_ratio']

# Estimate components
embedders = 2 * (vars * 16 * d)
encoder = 4 * (d**2 * 4) * isab * col + 2 * (d**2 * 4) * enc
decoder = 8 * (d**2 * 4) * dec
predictor = 2 * (d * int(d * bottleneck)) * pred['pred_n_heads']
heads = 5 * d * 44
other = 500000  # embeddings, layernorm, etc.

total = embedders + encoder + decoder + predictor + heads + other

print('Estimated Parameter Count:')
print('-'*60)
print(f"  Embedders:     {embedders:8,} ({embedders/total*100:5.1f}%)")
print(f"  MixEncoder:    {encoder:8,} ({encoder/total*100:5.1f}%)")
print(f"  RPNDecoder:    {decoder:8,} ({decoder/total*100:5.1f}%)")
print(f"  Predictor:     {predictor:8,} ({predictor/total*100:5.1f}%)  <- SMALL!")
print(f"  Heads:         {heads:8,} ({heads/total*100:5.1f}%)")
print(f"  Other:         {other:8,} ({other/total*100:5.1f}%)")
print('-'*60)
print(f"  TOTAL:         {total:8,} ({total/1e6:.2f}M)")
print()
print('='*60)
print(f"Target: ~1M params, Predictor <100K")
print(f"Status: {'✅ ON TARGET' if total < 1.2e6 else '⚠️ NEEDS ADJUSTMENT'}")
print('='*60)
