import argparse
import torch
import os
from pathlib import Path
from tqdm import tqdm

def verify_cache(cache_dir: str, min_file_size_kb: int = 1):
    """
    Scans the cache directory to ensure all part files are present,
    readable, and not corrupted (size check).
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"Error: Cache path {cache_dir} not found.")
        return

    if cache_path.is_file():
        part_files = [cache_path]
    else:
        part_files = sorted(list(cache_path.glob("part_*.pt")), 
                            key=lambda x: int(x.stem.split('_')[1]) if '_' in x.stem else 0)
    
    if not part_files:
        print(f"No valid data files found at {cache_dir}")
        return

    print(f"Found {len(part_files)} part files. Starting integrity check...")
    
    corrupted = []
    valid_files = []
    valid_sizes = []
    total_equations = 0
    
    for pf in tqdm(part_files, desc="Verifying parts"):
        # 1. Size check
        size_kb = os.path.getsize(pf) / 1024
        if size_kb < min_file_size_kb:
            print(f"\n  [!] {pf.name} is too small ({size_kb:.2f} KB) - likely corrupted.")
            corrupted.append(pf)
            continue
            
        # 2. Load check
        try:
            data = torch.load(pf, weights_only=False)
            size = len(data)
            total_equations += size
            valid_files.append(pf)
            valid_sizes.append(size)
            del data
        except Exception as e:
            print(f"\n  [!] Failed to load {pf.name}: {e}")
            corrupted.append(pf)

    print("\n--- Verification Report ---")
    print(f"Total parts checked: {len(part_files)}")
    print(f"Total equations:    {total_equations:,}")
    print(f"Corrupted files:    {len(corrupted)}")
    
    if not corrupted:
        print("\nSUCCESS: All files passed integrity check. Ready for training!")
        # Save manifest for LazySyntheticDataset instant startup
        manifest_path = (cache_path if cache_path.is_dir() else cache_path.parent) / "metadata_manifest.pt"
        try:
            torch.save({
                'files': [f.name for f in valid_files],
                'sizes': valid_sizes,
                'total_size': total_equations,
                'offsets': [sum(valid_sizes[:i]) for i in range(len(valid_sizes))]
            }, manifest_path)
            print(f"Manifest saved to {manifest_path}")
            print(f"Total equations indexed: {total_equations:,}")
            print("Startup for this cache will now be near-instant.")
        except Exception as e:
            print(f"Warning: Could not save manifest: {e}")
    else:
        print("\nACTION REQUIRED: Delete the following files and re-run generation:")
        for cf in corrupted:
            print(f"  rm {cf}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Synthetic Data Cache Integrity")
    parser.add_argument("--cache_dir", type=str, default="cache/synthetic_1M", help="Path to cache directory")
    parser.add_argument("--min_size", type=int, default=1, help="Minimum file size in KB")
    args = parser.parse_args()
    
    verify_cache(args.cache_dir, args.min_size)
