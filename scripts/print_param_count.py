"""Analyze model parameter distribution."""
import sys
import yaml
import torch
from models.model import LLMJEPAModule

def count_params(config_name, config_path):
    print(f"\n{'='*60}")
    print(f"{config_name}")
    print('='*60)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    cfg = config['model']
    
    print(f"\nArchitecture:")
    print(f"  d_model: {cfg['d_model']}")
    print(f"  n_enc_layers: {cfg['n_enc_layers']}")
    print(f"  n_dec_layers: {cfg['n_dec_layers']}")
    print(f"  n_isab: {cfg.get('n_isab', 2)}")
    print(f"  n_col_attn: {cfg.get('n_col_attn', 2)}")
    print(f"  m_inducing: {cfg.get('m_inducing', 32)}")
    
    model = LLMJEPAModule(
        d_model=cfg['d_model'],
        n_heads=cfg['n_heads'],
        n_encoder_layers=cfg['n_enc_layers'],
        n_decoder_layers=cfg['n_dec_layers'],
        max_n_vars=cfg['max_n_vars'],
        n_isab=cfg.get('n_isab', 2),
        n_col_attn=cfg.get('n_col_attn', 2),
        m_inducing=cfg.get('m_inducing', 32),
        dropout=cfg.get('dropout', 0.1),
        pred_n_heads=cfg['predictor']['pred_n_heads'],
        pred_bottleneck_ratio=cfg['predictor']['pred_bottleneck_ratio'],
        pred_dropout=cfg['predictor']['pred_dropout'],
    )
    
    # Count by component
    components = {}
    components['Data Embedder'] = sum(p.numel() for p in model.model.data_embedder.parameters())
    components['Unit Embedder'] = sum(p.numel() for p in model.model.unit_embedder.parameters())
    components['MixEncoder'] = sum(p.numel() for p in model.model.mix_encoder.parameters())
    components['RPNDecoder'] = sum(p.numel() for p in model.model.decoder.parameters())
    components['Target Encoder'] = sum(p.numel() for p in model.model.target_encoder.parameters())
    components['Predictor'] = sum(p.numel() for p in model.model.predictor.parameters())
    components['Unit Head'] = sum(p.numel() for p in model.model.unit_head.parameters()) if model.model.unit_head else 0
    
    total = sum(components.values())
    
    print(f"\nTotal Parameters: {total:,} ({total/1e6:.2f}M)\n")
    print("Parameter Distribution:")
    print("-" * 60)
    for name, count in sorted(components.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        bar_len = int(pct / 2)
        bar = '#' * bar_len
        print(f"  {name:20s} {count:8,} ({pct:5.1f}%) {bar}")
    
    return total

if __name__ == '__main__':
    print("Loading models...")
    sys.stdout.flush()
    
    # Compare configs
    base_params = count_params('BASE MODEL (3.4M)', 'configs/base_config.yaml')
    print("\n")
    small_params = count_params('TINY MODEL (~1M)', 'configs/small.yaml')
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Base Model:  {base_params:,} params ({base_params/1e6:.2f}M)")
    print(f"Tiny Model:  {small_params:,} params ({small_params/1e6:.2f}M)")
    print(f"Reduction:   {(1 - small_params/base_params)*100:.1f}% smaller")
    print(f"Predictor:   {small_params/base_params*100:.0f}% the size")
    print('='*60)
