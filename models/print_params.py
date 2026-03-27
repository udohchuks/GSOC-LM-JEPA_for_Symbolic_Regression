import torch
import argparse
import yaml
from pathlib import Path

from models.embedders import DataEmbedder, UnitEmbedder
from models.encoder import MixEncoder
from models.target_encoder import TargetEncoder
from models.predictor import JEPAPredictor
from models.decoder import RPNDecoder, UnitPredictionHead
from models.model import LLMJEPA

def count_parameters(model):
    """Count trainable parameters of a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description="Print Model Parameters")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    m = config['model']
    data = config['data']
    
    d_model = m['d_model']
    max_n_vars = data['max_n_vars']
    
    # Instantiate models
    data_embedder = DataEmbedder(d_model=d_model, max_n_vars=max_n_vars)
    unit_embedder = UnitEmbedder(d_model=d_model)
    mix_encoder = MixEncoder(
        d_model=d_model, 
        n_heads=m['n_heads'], 
        n_isab=m['n_isab'], 
        n_col_attn=m['n_col_attn'], 
        m_inducing=m['m_inducing'], 
        max_n_vars=max_n_vars,
        dropout=m['dropout']
    )
    target_encoder = TargetEncoder(d_model=d_model, n_heads=m['n_heads'], n_layers=m['n_enc_layers'], dropout=m['dropout'])
    # Enable target encoder gradients (like how SIGReg trains it)
    for param in target_encoder.parameters():
        param.requires_grad = True

    pred_cfg = m['predictor']
    predictor = JEPAPredictor(
        d_model=d_model, 
        n_heads=pred_cfg['pred_n_heads'], 
        bottleneck_ratio=pred_cfg['pred_bottleneck_ratio'], 
        dropout=pred_cfg['pred_dropout']
    )
    decoder = RPNDecoder(d_model=d_model, n_heads=m['n_heads'], n_layers=m['n_dec_layers'], dropout=m['dropout'])
    unit_head = UnitPredictionHead(d_model=d_model)
    
    unified_model = LLMJEPA(
        d_model=d_model, 
        n_heads=m['n_heads'], 
        n_isab=m['n_isab'],
        n_col_attn=m['n_col_attn'], 
        n_enc_layers=m['n_enc_layers'], 
        n_dec_layers=m['n_dec_layers'],
        m_inducing=m['m_inducing'], 
        max_n_vars=max_n_vars,
        dropout=m['dropout'],
        pred_n_heads=pred_cfg['pred_n_heads'], 
        pred_bottleneck_ratio=pred_cfg['pred_bottleneck_ratio'], 
        pred_dropout=pred_cfg['pred_dropout']
    )
    
    print("="*55)
    print(f"🧠 MODEL PARAMETER COUNTS (Config: {args.config})")
    print("="*55)
    
    components = [
        ("Data Embedder", data_embedder),
        ("Unit Embedder", unit_embedder),
        ("Context Encoder (Mix)", mix_encoder),
        ("Target Encoder", target_encoder),
        ("JEPA Predictor", predictor),
        ("RPN Decoder", decoder),
        ("Unit Prediction Head", unit_head)
    ]
    
    total_components = 0
    for name, model in components:
        params = count_parameters(model)
        total_components += params
        print(f"{name:<25}: {params:>15,} params")
        
    print("-" * 55)
    print(f"{'Sum of Components':<25}: {total_components:>15,} params")
    
    unified_params = count_parameters(unified_model)
    print(f"{'Unified LLMJEPA Model':<25}: {unified_params:>15,} params")
    print("="*55)

if __name__ == '__main__':
    main()
