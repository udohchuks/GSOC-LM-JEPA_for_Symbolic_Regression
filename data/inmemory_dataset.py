#!/usr/bin/env python
"""
In-Memory Synthetic Dataset for Maximum Training Speed.

Loads ALL synthetic equations into RAM at startup (~12-15GB for 1M equations).
Eliminates Google Drive I/O bottlenecks completely.

Usage:
    from data.synthetic_dataset import InMemorySyntheticDataset
    
    dataset = InMemorySyntheticDataset(
        cache_dir='cache/synthetic_1M',
        max_n_vars=10,
        n_rows=200
    )
    
    # Dataset is now fully loaded in RAM - no more Drive I/O!
    loader = DataLoader(dataset, batch_size=2048, num_workers=16, ...)
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Optional
import time

from data.synthetic_dataset import SyntheticEquation, SyntheticDataset
from data.aif_dataset import collate_fn


class InMemorySyntheticDataset(Dataset):
    """
    Full-RAM Dataset for synthetic equations.
    
    Loads ALL equations into memory at startup (~12-15GB for 1M equations).
    This eliminates Google Drive I/O bottlenecks during training.
    
    Memory usage:
        - 100k equations: ~1.5GB RAM
        - 500k equations: ~7.5GB RAM
        - 1M equations: ~15GB RAM
    
    Args:
        cache_dir: Directory containing .pt chunk files
        max_n_vars: Maximum number of variables (including Y)
        n_rows: Number of data points per equation
    """
    
    def __init__(
        self,
        cache_dir: str,
        max_n_vars: int = 10,
        n_rows: int = 200,
    ):
        super().__init__()
        self.max_n_vars = max_n_vars
        self.n_rows = n_rows
        
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache directory not found: {cache_path}")
        
        # Find all .pt chunk files
        chunk_files = sorted(cache_path.glob("part_*.pt"))
        if not chunk_files:
            raise ValueError(f"No .pt chunk files found in {cache_path}")
        
        print(f"📦 Loading {len(chunk_files)} chunk files into RAM...")
        t0 = time.perf_counter()
        
        # Load ALL chunks into a single list
        self.equations: List[SyntheticEquation] = []
        
        # Use parallel loading for large datasets (>1000 chunks)
        if len(chunk_files) > 1000:
            print(f"   Using parallel loading for {len(chunk_files):,} chunks...")
            import concurrent.futures
            
            def load_chunk(chunk_file):
                try:
                    chunk_data = torch.load(chunk_file, map_location='cpu', weights_only=False)
                    return chunk_data if isinstance(chunk_data, list) else [chunk_data]
                except Exception as e:
                    print(f"   ⚠️  Warning: Failed to load {chunk_file.name}: {e}")
                    return []
            
            # Load in parallel with progress tracking
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                for i, chunk_data in enumerate(executor.map(load_chunk, chunk_files)):
                    self.equations.extend(chunk_data)
                    if (i + 1) % 500 == 0:
                        elapsed = time.perf_counter() - t0
                        rate = (i + 1) / elapsed
                        eta = (len(chunk_files) - i - 1) / rate
                        print(f"   Loaded {i+1}/{len(chunk_files)} chunks ({len(self.equations):,} equations) - ETA: {eta:.0f}s")
        else:
            # Sequential loading for smaller datasets
            for i, chunk_file in enumerate(chunk_files):
                try:
                    chunk_data = torch.load(chunk_file, map_location='cpu', weights_only=False)
                    if isinstance(chunk_data, list):
                        self.equations.extend(chunk_data)
                    else:
                        self.equations.append(chunk_data)
                    
                    if (i + 1) % 10 == 0:
                        print(f"   Loaded {i+1}/{len(chunk_files)} chunks ({len(self.equations):,} equations)")
                        
                except Exception as e:
                    print(f"   ⚠️  Warning: Failed to load {chunk_file.name}: {e}")
        
        elapsed = time.perf_counter() - t0
        
        # Estimate RAM usage
        eq_size = len(self.equations) * 0.015  # ~15KB per equation average
        print(f"✅ Loaded {len(self.equations):,} equations in {elapsed:.1f}s")
        print(f"   Estimated RAM usage: ~{eq_size:.1f}GB")
        print(f"   Equations per chunk: ~{len(self.equations)/len(chunk_files):.0f}")
    
    def __len__(self) -> int:
        return len(self.equations)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get one equation with preprocessing applied.
        
        Returns dict with:
            - X_bits: [n_rows, max_n_vars, 16] float16 (includes Y)
            - unit_idx: [max_n_vars, 5] int64
            - var_mask: [max_n_vars] float32
            - token_ids: [MAX_SEQ_LEN] int64
            - unit_targets_idx: [MAX_SEQ_LEN, 5] int64
            - n_vars: int (original variable count, not including Y)
            - eq_id, formula_str, var_names (metadata)
        """
        eq = self.equations[idx]
        n_vars = eq.n_vars
        
        X_bits = eq.X_bits   # [N, n_vars+1, 16] (includes Y)
        N = X_bits.shape[0]
        
        # ── Subsample/pad rows ────────────────────────────────────────
        if N > self.n_rows:
            row_idx = np.random.choice(N, self.n_rows, replace=False)
            X_bits = X_bits[row_idx]
        elif N < self.n_rows:
            pad_rows = self.n_rows - N
            pad_shape = (pad_rows,) + X_bits.shape[1:]
            padding = np.zeros(pad_shape, dtype=X_bits.dtype)
            X_bits = np.concatenate([X_bits, padding], axis=0)
        
        # ── Pad/truncate variables to max_n_vars ──────────────────────
        current_vars = X_bits.shape[1]  # Should be n_vars + 1 (includes Y)
        desired_vars = self.max_n_vars
        
        if current_vars < desired_vars:
            pad_w = desired_vars - current_vars
            pad_x = np.zeros((X_bits.shape[0], pad_w, 16), dtype=np.float16)
            X_bits = np.concatenate([X_bits, pad_x], axis=1)
        else:
            X_bits = X_bits[:, :desired_vars, :]
        
        # ── Handle units (X from metadata, Y is dimensionless) ────────
        y_unit = np.full((1, 5), 4, dtype=np.int64)  # dimensionless
        x_units = eq.unit_matrix_idx  # [n_vars, 5]
        combined_units = np.concatenate([x_units, y_unit], axis=0)
        
        # Pad/truncate units
        current_u = combined_units.shape[0]
        if current_u < desired_vars:
            pad_u = np.full((desired_vars - current_u, 5), 4, dtype=np.int64)
            unit_idx = np.concatenate([combined_units, pad_u], axis=0)
        else:
            unit_idx = combined_units[:desired_vars, :]
        
        # ── Variable mask ─────────────────────────────────────────────
        var_mask = np.zeros(self.max_n_vars, dtype=np.float32)
        valid_len = min(current_vars, self.max_n_vars)
        var_mask[:valid_len] = 1.0
        
        return {
            'X_bits': torch.from_numpy(X_bits),              # [n_rows, max_n_vars, 16]
            'unit_idx': torch.from_numpy(unit_idx).long(),   # [max_n_vars, 5]
            'var_mask': torch.from_numpy(var_mask).float(),  # [max_n_vars]
            'n_vars': torch.tensor(n_vars, dtype=torch.long),
            'token_ids': torch.from_numpy(eq.token_ids).long(),
            'unit_targets_idx': torch.from_numpy(eq.unit_targets_idx).long(),
            'eq_id': f'syn_{idx}',
            'formula_str': eq.expr_str,
            'var_names': eq.var_names,
        }


def create_inmemory_dataloader(
    cache_dir: str,
    max_n_vars: int = 10,
    n_rows: int = 200,
    batch_size: int = 2048,
    num_workers: int = 16,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> DataLoader:
    """
    Create a DataLoader with InMemorySyntheticDataset.
    
    Optimized for maximum GPU utilization:
        - batch_size: 2048 (keeps GPU at 100%)
        - num_workers: 16 (CPU stays ahead of GPU)
        - pin_memory: True (faster CPU→GPU transfer)
        - persistent_workers: True (no worker restart overhead)
    
    Args:
        cache_dir: Path to synthetic data chunks
        max_n_vars: Max variables including Y (default: 10)
        n_rows: Data points per equation (default: 200)
        batch_size: Batch size (default: 2048)
        num_workers: CPU workers (default: 16)
        pin_memory: Pin memory for faster GPU transfer
        persistent_workers: Keep workers alive between epochs
    
    Returns:
        DataLoader ready for training
    """
    print(f"\n{'='*70}")
    print(f"🚀 CREATING IN-MEMORY DATALOADER (GPU REDLINE MODE)")
    print(f"{'='*70}")
    
    # Load dataset into RAM
    dataset = InMemorySyntheticDataset(
        cache_dir=cache_dir,
        max_n_vars=max_n_vars,
        n_rows=n_rows,
    )
    
    # Create DataLoader with optimal settings
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    print(f"\n✅ DataLoader ready:")
    print(f"   Dataset size: {len(dataset):,} equations")
    print(f"   Batch size: {batch_size}")
    print(f"   Batches per epoch: {len(loader):,}")
    print(f"   Workers: {num_workers}")
    print(f"   Pin memory: {pin_memory}")
    print(f"   Persistent workers: {persistent_workers}")
    print(f"{'='*70}\n")
    
    return loader


if __name__ == "__main__":
    # Test the InMemory dataset
    print("Testing InMemorySyntheticDataset...")
    
    test_cache = "cache/synthetic_smoke_test"
    if Path(test_cache).exists():
        dataset = InMemorySyntheticDataset(
            cache_dir=test_cache,
            max_n_vars=10,
            n_rows=200,
        )
        
        print(f"\nTesting __getitem__...")
        item = dataset[0]
        print(f"  X_bits shape: {item['X_bits'].shape}")
        print(f"  unit_idx shape: {item['unit_idx'].shape}")
        print(f"  var_mask sum: {item['var_mask'].sum().item()}")
        print(f"  n_vars: {item['n_vars'].item()}")
        
        print("\n✅ Test passed!")
    else:
        print(f"  Test cache not found: {test_cache}")
        print("  Run data generation first: python -m data.generate_data")
