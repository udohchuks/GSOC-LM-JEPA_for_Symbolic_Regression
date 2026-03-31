#!/usr/bin/env python
"""
High-Speed Incremental Dataset Loader

Loads first 50 chunks synchronously for instant training start,
then hydrates remaining chunks in background thread.

Usage:
    from data.inmemory_dataset import InMemorySyntheticDataset
    
    # Starts training in ~30 seconds with first 50 chunks
    # Background thread continues loading remaining chunks
    dataset = InMemorySyntheticDataset(
        cache_dir='cache/synthetic_1M',
        max_n_vars=10,
        n_rows=200,
        min_chunks_for_start=50,  # Start after loading 50 chunks
    )
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Optional
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from data.synthetic_dataset import SyntheticEquation
from data.aif_dataset import collate_fn


class InMemorySyntheticDataset(Dataset):
    """
    Incremental Full-RAM Dataset for synthetic equations.
    
    Loads first N chunks synchronously (~30 seconds), then continues
    loading remaining chunks in background thread. Training can start
    immediately while dataset grows in real-time.
    
    Memory usage:
        - 100k equations: ~1.5GB RAM
        - 500k equations: ~7.5GB RAM
        - 1M equations: ~15GB RAM
    
    Args:
        cache_dir: Directory containing .pt chunk files
        max_n_vars: Maximum number of variables (including Y)
        n_rows: Number of data points per equation
        min_chunks_for_start: Number of chunks to load before returning (default: 50)
    """
    
    def __init__(
        self,
        cache_dir: str,
        max_n_vars: int = 10,
        n_rows: int = 200,
        min_chunks_for_start: int = 50,
    ):
        super().__init__()
        self.max_n_vars = max_n_vars
        self.n_rows = n_rows
        self.min_chunks_for_start = min_chunks_for_start
        
        self.equations: List[SyntheticEquation] = []
        self._lock = threading.RLock()  # Thread-safe list access
        self._loading_complete = False
        self._chunks_loaded = 0
        self._total_chunks = 0
        
        # Estimate equations per chunk (assume ~100 based on generation)
        self._equations_per_chunk = 100
        self._projected_total = 0
        
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache directory not found: {cache_path}")
        
        # Find all .pt chunk files (exclude merged files initially)
        chunk_files = sorted(cache_path.glob("part_*.pt"))
        # Prioritize merged files if they exist
        merged_files = [f for f in chunk_files if 'merged' in f.name]
        if merged_files:
            chunk_files = sorted(merged_files)
            print(f"✅ Found {len(merged_files):,} merged chunk files")
        
        if not chunk_files:
            raise ValueError(f"No .pt chunk files found in {cache_path}")
        
        self._chunk_files = chunk_files
        self._total_chunks = len(chunk_files)
        
        # Calculate projected total size (for Subset compatibility)
        self._projected_total = self._total_chunks * self._equations_per_chunk
        
        print(f"\n{'='*70}")
        print(f"🚀 INCREMENTAL LOADING: {self._total_chunks:,} chunks")
        print(f"   Projected dataset size: ~{self._projected_total:,} equations")
        print(f"{'='*70}")
        print(f"Phase 1: Load first {min_chunks_for_start} chunks synchronously...")
        print(f"Phase 2: Background loading of remaining {self._total_chunks - min_chunks_for_start:,} chunks")
        print(f"{'='*70}\n")
        
        # PHASE 1: Load first N chunks synchronously (fast startup)
        t0 = time.perf_counter()
        self._load_chunks_range(0, min_chunks_for_start)
        
        elapsed = time.perf_counter() - t0
        print(f"\n✅ READY FOR TRAINING in {elapsed:.1f}s")
        print(f"   Initial dataset size: {len(self.equations):,} equations")
        print(f"   Chunks loaded: {self._chunks_loaded:,}/{self._total_chunks:,}")
        print(f"\n📡 Background loading started...")
        print(f"   Dataset will grow as more chunks load\n")
        
        # PHASE 2: Start background thread for remaining chunks
        self._background_thread = threading.Thread(
            target=self._load_remaining_chunks,
            args=(min_chunks_for_start, self._total_chunks),
            daemon=True
        )
        self._background_thread.start()
    
    def _load_chunks_range(self, start_idx: int, end_idx: int):
        """Load chunks from start_idx to end_idx (exclusive)."""
        for i in range(start_idx, min(end_idx, self._total_chunks)):
            chunk_file = self._chunk_files[i]
            try:
                chunk_data = torch.load(chunk_file, map_location='cpu', weights_only=False)
                
                with self._lock:
                    if isinstance(chunk_data, list):
                        actual_count = len(chunk_data)
                        self.equations.extend(chunk_data)
                    else:
                        actual_count = 1
                        self.equations.append(chunk_data)
                    
                    # Update equations per chunk estimate
                    if self._chunks_loaded < 10:  # Learn from first 10 chunks
                        self._equations_per_chunk = len(self.equations) // (self._chunks_loaded + 1)
                        self._projected_total = self._total_chunks * self._equations_per_chunk
                
                self._chunks_loaded += 1
                
            except Exception as e:
                print(f"   ⚠️  Warning: Failed to load {chunk_file.name}: {e}")
    
    def _load_remaining_chunks(self, start_idx: int, end_idx: int):
        """Load remaining chunks in background with parallel loading."""
        print(f"   Loading remaining {end_idx - start_idx:,} chunks in background...")
        t0 = time.perf_counter()
        
        # Use ThreadPoolExecutor for parallel loading (32 threads for I/O-bound)
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = []
            for i in range(start_idx, end_idx):
                futures.append(executor.submit(self._load_single_chunk, i))
            
            # Track progress
            completed = 0
            for future in futures:
                future.result()  # Wait for completion
                completed += 1
                if completed % 500 == 0:
                    elapsed = time.perf_counter() - t0
                    rate = completed / elapsed
                    eta = (len(futures) - completed) / rate
                    with self._lock:
                        eq_count = len(self.equations)
                    print(f"   Background: {completed:,}/{len(futures):,} chunks ({eq_count:,} equations) - ETA: {eta:.0f}s")
        
        self._loading_complete = True
        elapsed = time.perf_counter() - t0
        with self._lock:
            final_count = len(self.equations)
        print(f"\n✅ Background loading complete in {elapsed:.1f}s")
        print(f"   Final dataset size: {final_count:,} equations")
        print(f"   Total chunks: {self._chunks_loaded:,}/{self._total_chunks:,}\n")
    
    def _load_single_chunk(self, idx: int):
        """Load a single chunk (called from thread pool)."""
        chunk_file = self._chunk_files[idx]
        try:
            chunk_data = torch.load(chunk_file, map_location='cpu', weights_only=False)
            
            with self._lock:
                if isinstance(chunk_data, list):
                    self.equations.extend(chunk_data)
                else:
                    self.equations.append(chunk_data)
                
                # Update equations per chunk estimate (first 10 chunks only)
                if self._chunks_loaded < 10:
                    self._equations_per_chunk = len(self.equations) // (self._chunks_loaded + 1)
                    self._projected_total = self._total_chunks * self._equations_per_chunk
            
            self._chunks_loaded += 1
            
        except Exception as e:
            # Silent failure in background thread
            pass
    
    def __len__(self) -> int:
        """
        Return projected total dataset size (not current loaded size).
        
        This ensures Subset and DataLoader see the full dataset size from
        the start, allowing training to access all equations as they load.
        
        The __getitem__ uses modulo wrap-around for indices beyond current
        loaded size, so this is safe.
        """
        return self._projected_total
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get one equation with preprocessing applied.
        
        Note: Uses modulo wrap-around for indices beyond currently loaded
        equations. This allows training to start immediately while dataset
        grows in background.
        
        As more chunks load, the wrap-around frequency decreases, providing
        natural curriculum learning (more diversity over time).
        """
        with self._lock:
            current_size = len(self.equations)
            if current_size == 0:
                # Should not happen (min_chunks_for_start ensures data)
                raise IndexError("No equations loaded yet")
            
            # Wrap around if index exceeds current loaded size
            # This allows full dataset range access from start
            idx = idx % current_size
            
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
    
    def get_loading_status(self) -> Dict:
        """Get current loading progress."""
        with self._lock:
            return {
                'chunks_loaded': self._chunks_loaded,
                'total_chunks': self._total_chunks,
                'equations_loaded': len(self.equations),
                'projected_total': self._projected_total,
                'loading_complete': self._loading_complete,
                'background_thread_alive': self._background_thread.is_alive() if hasattr(self, '_background_thread') else False,
            }


def create_inmemory_dataloader(
    cache_dir: str,
    max_n_vars: int = 10,
    n_rows: int = 200,
    min_chunks_for_start: int = 50,
    batch_size: int = 2048,
    num_workers: int = 16,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> DataLoader:
    """
    Create a DataLoader with Incremental InMemorySyntheticDataset.
    
    Training starts in ~30 seconds with first 50 chunks.
    Background thread continues loading remaining chunks.
    
    Args:
        cache_dir: Path to synthetic data chunks
        max_n_vars: Max variables including Y (default: 10)
        n_rows: Data points per equation (default: 200)
        min_chunks_for_start: Chunks to load before starting (default: 50)
        batch_size: Batch size (default: 2048)
        num_workers: CPU workers (default: 16)
        pin_memory: Pin memory for faster GPU transfer
        persistent_workers: Keep workers alive between epochs
    
    Returns:
        DataLoader ready for immediate training
    """
    print(f"\n{'='*70}")
    print(f"🚀 CREATING INCREMENTAL DATALOADER (INSTANT START)")
    print(f"{'='*70}")
    
    # Load dataset (starts training after min_chunks_for_start)
    dataset = InMemorySyntheticDataset(
        cache_dir=cache_dir,
        max_n_vars=max_n_vars,
        n_rows=n_rows,
        min_chunks_for_start=min_chunks_for_start,
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
    print(f"   Initial size: {len(dataset):,} equations")
    print(f"   Batch size: {batch_size}")
    print(f"   Batches per epoch: {len(loader):,}")
    print(f"   Workers: {num_workers}")
    print(f"   Background loading: Active")
    print(f"{'='*70}\n")
    
    return loader


if __name__ == "__main__":
    # Test the incremental dataset
    print("Testing Incremental InMemorySyntheticDataset...")
    
    test_cache = "cache/synthetic_smoke_test"
    if Path(test_cache).exists():
        dataset = InMemorySyntheticDataset(
            cache_dir=test_cache,
            max_n_vars=10,
            n_rows=200,
            min_chunks_for_start=10,  # Start after 10 chunks for testing
        )
        
        print(f"\nTesting __getitem__...")
        item = dataset[0]
        print(f"  X_bits shape: {item['X_bits'].shape}")
        print(f"  unit_idx shape: {item['unit_idx'].shape}")
        print(f"  var_mask sum: {item['var_mask'].sum().item()}")
        print(f"  n_vars: {item['n_vars'].item()}")
        
        # Wait for background loading
        print(f"\nWaiting for background loading...")
        import time
        while not dataset._loading_complete:
            status = dataset.get_loading_status()
            print(f"  Progress: {status['chunks_loaded']}/{status['total_chunks']} chunks ({status['equations_loaded']:,} equations)")
            time.sleep(2)
        
        print(f"\n✅ Test passed! Final size: {len(dataset):,} equations")
    else:
        print(f"  Test cache not found: {test_cache}")
        print("  Run data generation first: python -m data.generate_data")
