"""Test that config values are being read correctly."""
import yaml
from pathlib import Path

# Check small.yaml config
with open('configs/small.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("="*70)
print("CONFIG VERIFICATION: configs/small.yaml")
print("="*70)
print()
print("Data generation settings:")
print(f"  n_synthetic:      {config['data']['n_synthetic']:,}")
print(f"  n_data_points:    {config['data']['n_data_points']:,}")
print(f"  num_workers:      {config['data']['num_workers']}")
print(f"  chunk_size:       {config['data']['chunk_size']}")
print(f"  synthetic_cache:  {config['data']['synthetic_cache']}")
print()

# Verify generate_data.py reads these
print("generate_data.py will use:")
cfg_data = config['data']
n_equations = cfg_data.get('n_synthetic', 1000000)
n_data_points = cfg_data.get('n_data_points', 1000)
num_workers = cfg_data.get('num_workers', 4)
chunk_size = cfg_data.get('chunk_size', 1000)
cache_path = cfg_data.get('synthetic_cache', 'cache/synthetic_1M')

print(f"  n_equations:      {n_equations:,}")
print(f"  n_data_points:    {n_data_points:,}")
print(f"  num_workers:      {num_workers}")
print(f"  chunk_size:       {chunk_size} ← Should be 100")
print(f"  cache_path:       {cache_path}")
print()

if chunk_size == 100:
    print("✅ Chunk size is correct (100)")
else:
    print(f"❌ Chunk size is WRONG! Expected 100, got {chunk_size}")

print("="*70)
