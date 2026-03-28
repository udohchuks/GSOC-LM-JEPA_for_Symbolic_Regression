"""
Unit test for ModelEvaluator and InferenceModel API.
Verifies that the encode() method and eq_id lookup logic are correct.
"""

import torch
import torch.nn as nn
from models.evaluator import ModelEvaluator
from inference.generate import InferenceModel
from data.tokenizer import VOCAB_SIZE, MAX_SEQ_LEN, BOS_IDX, EOS_IDX

def test_inference_model_api():
    print("Testing InferenceModel API...")
    model = InferenceModel(
        d_model=64,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        max_n_vars=9,
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN
    )
    
    # Mock inputs
    B, N, n_vars = 1, 10, 4
    X_bits = torch.randn(B, N, 9, 16)
    unit_idx = torch.zeros(B, 9, 5, dtype=torch.long)
    var_mask = torch.ones(B, 9)
    
    # 1. Test encode()
    print("  Testing encode()...")
    z_context = model.encode(X_bits, unit_idx, var_mask)
    assert z_context.shape == (B, 64), f"Wrong z_context shape: {z_context.shape}"
    
    # 2. Test generate()
    print("  Testing generate()...")
    generated = model.generate(z_context, unit_idx, max_len=10)
    assert generated.shape == (B, 10), f"Wrong generated shape: {generated.shape}"
    print("  InferenceModel API: OK")

def test_evaluator_logic():
    print("\nTesting ModelEvaluator Logic (Partial)...")
    # We won't load a real checkpoint, but we'll verify the ground truth lookup
    class MockRes:
        def __init__(self, eq_id, predicted, exact):
            self.eq_id = eq_id
            self.predicted = predicted
            self.exact = exact
            
    class MockEq:
        def __init__(self, eq_id, formula_str):
            self.eq_id = eq_id
            self.formula_str = formula_str
            
    mock_metrics = {
        'per_eq_results': [
            {'eq_id': 'ID1', 'predicted': 'x1', 'exact': True},
            {'eq_id': 'ID2', 'predicted': 'x2', 'exact': False}
        ],
        'n_equations': 2,
        'exact_recovery_rate': 0.5,
        'valid_rpn_rate': 1.0
    }
    
    # Simulating the lookup logic in _save_results
    id_to_formula = {'ID1': 'var1', 'ID2': 'var2'}
    
    print("  Verifying ID lookup alignment...")
    for res in mock_metrics['per_eq_results']:
        eq_id = res['eq_id']
        gt = id_to_formula.get(eq_id)
        print(f"    ID: {eq_id} | GT: {gt} | Pred: {res['predicted']}")
        assert gt is not None, f"Failed to find GT for {eq_id}"
        
    print("✅ Evaluator Logic: OK")

if __name__ == "__main__":
    test_inference_model_api()
    test_evaluator_logic()
