#!/usr/bin/env python
"""
Merge small chunk files into larger ones for faster In-Memory loading.

Usage:
    python merge_chunks.py cache/synthetic_small 1000
    
This will merge chunks into files with 1000 equations each.
"""

import sys
import torch
from pathlib import Path
import time

def merge_chunks(cache_dir: str, new_chunk_size: int = 1000):
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"❌ Cache directory not found: {cache_path}")
        return
    
    chunk_files = sorted(cache_path.glob("part_*.pt"))
    # Exclude already merged files
    chunk_files = [f for f in chunk_files if 'merged' not in f.name]
    
    if not chunk_files:
        print(f"❌ No chunk files found in {cache_path}")
        return
    
    print(f"{'='*70}")
    print(f"CHUNK MERGER: {len(chunk_files):,} files → ~{len(chunk_files)*100//new_chunk_size:,} files")
    print(f"{'='*70}")
    print(f"Source: {cache_path}")
    print(f"Target chunk size: {new_chunk_size} equations")
    print()
    
    # Load all equations
    print("Step 1: Loading all equations...")
    t0 = time.perf_counter()
    all_equations = []
    
    for i, chunk_file in enumerate(chunk_files):
        try:
            data = torch.load(chunk_file, map_location='cpu', weights_only=False)
            if isinstance(data, list):
                all_equations.extend(data)
            else:
                all_equations.append(data)
            
            if (i + 1) % 500 == 0:
                elapsed = time.perf_counter() - t0
                print(f"   Loaded {i+1:,}/{len(chunk_files):,} chunks ({len(all_equations):,} equations) - {elapsed:.1f}s")
                
        except Exception as e:
            print(f"   ⚠️  Warning: Failed to load {chunk_file.name}: {e}")
    
    elapsed = time.perf_counter() - t0
    print(f"✅ Loaded {len(all_equations):,} equations in {elapsed:.1f}s")
    print()
    
    # Save as larger chunks
    print(f"Step 2: Saving as {new_chunk_size} equations per chunk...")
    t0 = time.perf_counter()
    
    new_chunks = 0
    for i in range(0, len(all_equations), new_chunk_size):
        chunk = all_equations[i:i+new_chunk_size]
        part_num = i // new_chunk_size
        output_file = cache_path / f"part_merged_{part_num:04d}.pt"
        
        torch.save(chunk, output_file)
        new_chunks += 1
        
        if new_chunks % 10 == 0:
            print(f"   Saved {new_chunks} chunks...")
    
    elapsed = time.perf_counter() - t0
    print(f"✅ Saved {new_chunks} chunks in {elapsed:.1f}s")
    print()
    
    # Summary
    print(f"{'='*70}")
    print(f"MERGE COMPLETE")
    print(f"{'='*70}")
    print(f"Before: {len(chunk_files):,} chunk files")
    print(f"After:  {new_chunks:,} chunk files (part_merged_*.pt)")
    print(f"Reduction: {len(chunk_files)/new_chunks:.1f}x fewer files")
    print()
    print(f"Next steps:")
    print(f"  1. Test the merged files work:")
    print(f"     python -c \"from data.inmemory_dataset import InMemorySyntheticDataset; d = InMemorySyntheticDataset('{cache_path}'); print(f'Loaded {len(d):,} equations')\"")
    print(f"  2. Delete old chunks:")
    print(f"     rm {cache_path}/part_[0-9]*.pt  (keeps only part_merged_*.pt)")
    print(f"  3. Run training - should load in ~{new_chunks/100:.0f}-{new_chunks/50:.0f} seconds!")
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merge_chunks.py <cache_dir> [chunk_size]")
        print("Example: python merge_chunks.py cache/synthetic_small 1000")
        sys.exit(1)
    
    cache_dir = sys.argv[1]
    chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    
    merge_chunks(cache_dir, chunk_size)
