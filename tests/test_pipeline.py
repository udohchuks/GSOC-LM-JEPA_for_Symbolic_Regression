import unittest
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from training.trainer import LLMJEPAModule
from data.synthetic_dataset import build_synthetic_dataloader

class TestTrainingPipeline(unittest.TestCase):
    def test_dummy_training_step(self):
        """Run 1 training step using synthetic data to verify lightning module"""
        B, d_model = 2, 64
        loader = build_synthetic_dataloader(n_equations=4, batch_size=B, cache_path=None)
        
        model = LLMJEPAModule(
            d_model=d_model,
            n_heads=2,
            n_encoder_layers=1,
            n_decoder_layers=1,
            learning_rate=1e-4,
        )
        
        trainer = pl.Trainer(
            max_steps=1,
            accelerator="cpu",
            enable_checkpointing=False,
            logger=False,
        )
        
        try:
            trainer.fit(model, loader)
            success = True
        except Exception as e:
            print(f"Training step failed with exception: {e}")
            success = False
            
        self.assertTrue(success)

if __name__ == '__main__':
    unittest.main()
