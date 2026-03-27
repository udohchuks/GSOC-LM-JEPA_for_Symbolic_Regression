import unittest
import torch
from models.model import LLMJEPA
from data.tokenizer import MAX_SEQ_LEN, BOS_IDX, EOS_IDX, VOCAB_SIZE

class TestLLMJEPA(unittest.TestCase):
    def test_model_forward_pass(self):
        """Test the forward pass of the full LLMJEPA model"""
        B, N, n_vars, d_model = 2, 50, 4, 64
        T = MAX_SEQ_LEN

        model = LLMJEPA(
            d_model=d_model, n_heads=4, n_isab=1,
            n_col_attn=1, n_enc_layers=2, n_dec_layers=2,
            m_inducing=8, max_n_vars=9,
        )

        X_bits   = torch.randint(0, 2, (B, N, 9, 16)).float()
        unit_idx = torch.randint(0, 9, (B, 9, 5))
        var_mask = torch.ones(B, 9)
        var_mask[:, n_vars:] = 0.0

        token_ids        = torch.zeros(B, T, dtype=torch.long)
        token_ids[:, 0]  = BOS_IDX
        token_ids[:, 1]  = 4   # x1
        token_ids[:, 2]  = 12  # sin
        token_ids[:, 3]  = EOS_IDX

        unit_targets = torch.randint(0, 9, (B, T, 5))

        model.train()
        out = model(X_bits, unit_idx, var_mask, token_ids, unit_targets)

        self.assertEqual(out['z_context'].shape, (B, d_model))
        self.assertEqual(out['z_target'].shape, (B, d_model))
        self.assertEqual(out['z_hat'].shape, (B, d_model))
        self.assertEqual(out['logits'].shape, (B, T - 1, VOCAB_SIZE))

if __name__ == '__main__':
    unittest.main()
