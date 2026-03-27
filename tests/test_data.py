import unittest
import torch
import numpy as np

from data.tokenizer import is_valid_rpn, get_valid_next_tokens, ARITY, IDX2TOKEN
from data.synthetic_dataset import build_synthetic_dataloader

class TestDataPipeline(unittest.TestCase):
    def test_tokenizer_valid_rpn(self):
        """Test RPN validation logic"""
        self.assertTrue(is_valid_rpn(['x1', 'x2', '+']))
        self.assertFalse(is_valid_rpn(['+', 'x1', 'x2']))
        self.assertFalse(is_valid_rpn(['x1', 'x2', '+', 'x3']))
        
    def test_tokenizer_validity_mask(self):
        """Test validity masking at step 0"""
        valid = get_valid_next_tokens(stack_depth=0, seq_len=0, max_len=25)
        for idx in valid:
            self.assertEqual(ARITY.get(IDX2TOKEN[idx], 0), 0)

    def test_synthetic_dataloader(self):
        """Test synthetic data loader shapes"""
        loader = build_synthetic_dataloader(n_equations=10, batch_size=2, cache_path=None)
        batch = next(iter(loader))
        
        # Verify expected keys (same interface as AIFDataset)
        for key in ('X_bits', 'unit_idx', 'var_mask', 'token_ids', 'unit_targets_idx'):
            self.assertIn(key, batch, f"Missing key: {key}")
        
        # Verify batch size
        self.assertEqual(batch['X_bits'].shape[0], 2)
        # Verify IEEE-754 bits encoding dimension
        self.assertEqual(batch['X_bits'].shape[-1], 16)
        # Verify variable padding to max_n_vars=9
        self.assertEqual(batch['X_bits'].shape[2], 9)


if __name__ == '__main__':
    unittest.main()
