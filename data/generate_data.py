"""
Standalone Synthetic Data Generation Script.

Reads parameters from the project's YAML config to ensure consistency 
with the training pipeline.

Usage:
    python -m data.generate_data --config configs/base_config.yaml
"""
import argparse
import yaml
from pathlib import Path
from data.synthetic_dataset import _generate_corpus

def main():
    parser = argparse.ArgumentParser(description="Generate Synthetic symbolic data")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", 
                        help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    cfg_data = config['data']
    cfg_model = config['model']
    
    n_equations = cfg_data.get('n_synthetic', 1000000)
    n_data_points = cfg_data.get('n_data_points', 1000)
    num_workers = cfg_data.get('num_workers', 4)
    chunk_size = 10000 # Optimized chunk size
    cache_path = cfg_data.get('synthetic_cache', 'cache/synthetic_1M')
    
    # Large scale check (consistent with build_synthetic_dataloader)
    IS_LARGE_SCALE = (n_equations >= 100000)
    
    target_dir = cache_path
    if IS_LARGE_SCALE and cache_path.endswith('.pt'):
        target_dir = cache_path.replace('.pt', '')
        
    print(f"--- Synthetic Data Generation Started ---")
    print(f"Goal: {n_equations} equations")
    print(f"Chunk size: {chunk_size}")
    print(f"Workers: {num_workers}")
    print(f"Cache Directory: {target_dir}")
    print(f"------------------------------------------")

    _generate_corpus(
        n_equations=n_equations,
        n_data_points=n_data_points,
        num_workers=num_workers,
        cache_dir=target_dir,
        chunk_size=chunk_size
    )
    
    print("\nGeneration complete!")

if __name__ == '__main__':
    main()
