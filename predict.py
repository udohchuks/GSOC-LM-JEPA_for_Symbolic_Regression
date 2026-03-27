import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from training.trainer import LLMJEPAModule
from inference.generate import InferenceModel
from data.tokenizer import VOCAB_SIZE, MAX_SEQ_LEN, rpn_to_sympy, decode_formula
from data.aif_dataset import build_aif_dataloader
from data.utils import to_ieee754_16bit
from data.unit_table import get_unit_matrix, unit_to_class_indices

def load_inference_model(config_path: str, ckpt_path: str, device: str) -> InferenceModel:
    """Load trained checkpoint into the InferenceModel for generation."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    pl_module = LLMJEPAModule.load_from_checkpoint(ckpt_path, map_location=device)
    base_model = pl_module.model
    
    inf_model = InferenceModel(
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_encoder_layers=config['model']['n_enc_layers'],
        n_decoder_layers=config['model']['n_dec_layers'],
        max_n_vars=config['data']['max_n_vars'],
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
    ).to(device)
    
    # Copy weights
    inf_model.data_embedder.load_state_dict(base_model.data_embedder.state_dict())
    inf_model.unit_embedder.load_state_dict(base_model.unit_embedder.state_dict())
    inf_model.context_encoder.load_state_dict(base_model.mix_encoder.state_dict())
    inf_model.decoder.load_state_dict(base_model.decoder.state_dict())
    
    inf_model.max_n_vars = config['data']['max_n_vars']
    inf_model.eval()
    return inf_model

def predict_from_csv(model, csv_path, device):
    """Run inference on a custom user-provided CSV file."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Assume last column is target y, others are inputs X
    var_names = list(df.columns)[:-1]
    n_vars = len(var_names)
    print(f"Found {n_vars} variables: {var_names}")
    
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.float32)
    
    # Subsample to 200 rows if it's too large to fit in Context Encoder memory
    N = min(X.shape[0], 200) 
    if X.shape[0] > N:
        idx = np.random.choice(X.shape[0], N, replace=False)
        X = X[idx]
        y = y[idx]
        
    X_bits = to_ieee754_16bit(X) # [N, n_vars, 16]
    
    unit_matrix = get_unit_matrix(var_names)
    unit_matrix_idx = unit_to_class_indices(unit_matrix) # [n_vars, 5]
    
    max_n = model.max_n_vars
    pad = max_n - n_vars
    
    X_bits = torch.from_numpy(X_bits).unsqueeze(0).to(device) # [1, N, n_vars, 16]
    unit_idx = torch.from_numpy(unit_matrix_idx).unsqueeze(0).to(device) # [1, n_vars, 5]
    
    var_mask = torch.zeros((1, max_n), device=device)
    var_mask[:, :n_vars] = 1.0
    
    # Pad to max variables
    if pad > 0:
        pad_x = torch.zeros(1, X_bits.shape[1], pad, 16, dtype=torch.uint8, device=device)
        X_bits = torch.cat([X_bits, pad_x], dim=2)
        
        pad_u = torch.full((1, pad, 5), 4, dtype=torch.long, device=device)
        unit_idx = torch.cat([unit_idx, pad_u], dim=1)
        
    # Convert bits to float for the embedder
    X_t = X_bits.float()
    
    with torch.no_grad():
        z_context = model(X_t, unit_idx, var_mask)   # encode first
        generated = model.generate(z_context, unit_idx)
        
    tokens = generated[0].cpu().tolist()
    return tokens, var_names

def predict_from_aif(model, config, eq_id, device):
    """Run inference on a specific equation from the AI Feynman dataset."""
    print(f"Loading AI Feynman dataset...")
    full_loader = build_aif_dataloader(
        csv_path=config['data']['csv_path'],
        data_dir=config['data']['data_dir'],
        batch_size=1,
        n_rows=200,
        max_n_vars=config['data']['max_n_vars'],
        cache_dir=config['data']['cache_dir'] + "aif_preprocessed.pt",
        shuffle=False
    )
    
    for batch in full_loader:
        if batch['eq_id'][0] == eq_id:
            print(f"Found equation {eq_id}: Ground truth is {batch['formula_str'][0]}")
            X_bits = batch['X_bits'].to(device)
            unit_idx = batch['unit_idx'].to(device)
            var_mask = batch['var_mask'].to(device)
            
            with torch.no_grad():
                z_context = model(X_bits, unit_idx, var_mask)  # encode first
                generated = model.generate(z_context, unit_idx)
            
            tokens = generated[0].cpu().tolist()
            return tokens, batch['var_names'][0]
            
    raise ValueError(f"Equation {eq_id} not found in dataset.")

def main():
    parser = argparse.ArgumentParser(description="Run Inference on LLM-JEPA")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to config file")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained checkpoint (.ckpt)")
    parser.add_argument("--csv", type=str, default=None, help="Path to a custom CSV file for inference")
    parser.add_argument("--eq_id", type=str, default=None, help="Equation ID to evaluate from AI Feynman (e.g., I.6.2a)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {args.ckpt} on {device}...")
    model = load_inference_model(args.config, args.ckpt, device)
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    if args.csv:
        tokens, var_names = predict_from_csv(model, args.csv, device)
    elif args.eq_id:
        tokens, var_names = predict_from_aif(model, config, args.eq_id, device)
    else:
        print("Please provide either --csv or --eq_id to run inference.")
        return

    rpn_tokens = decode_formula(tokens, strip_special=True)
    print(f"\nGenerated RPN Tokens: {' '.join(rpn_tokens)}")
    
    try:
        expr = rpn_to_sympy(rpn_tokens)
        print(f"SymPy Expression: {expr}")
    except Exception as e:
        print(f"Could not parse as SymPy: {e}")

if __name__ == '__main__':
    main()
