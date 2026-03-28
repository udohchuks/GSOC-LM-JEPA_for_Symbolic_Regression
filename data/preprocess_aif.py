"""
Standalone AI Feynman (AIF) Preprocessing Script.

Parses the AIF dataset (FeynmanEquations.csv + data files) and saves 
the preprocessed tensors to a cache file for fast loading during training.

Usage:
    python -m data.preprocess_aif --config configs/base_config.yaml
"""
import argparse
import yaml
from pathlib import Path
from data.aif_dataset import build_aif_dataset

def main():
    parser = argparse.ArgumentParser(description="Preprocess AIF data")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", 
                        help="Path to config file")
    parser.add_argument("--rows", type=int, default=10000, 
                        help="Max rows per equation to cache (default 10k)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    cfg_data = config['data']
    
    csv_path = cfg_data.get('csv_path', 'data/FeynmanEquations.csv')
    data_dir = cfg_data.get('data_dir', 'data/Feynman_with_units/')
    cache_dir = cfg_data.get('cache_dir', 'cache/')
    
    print(f"--- AIF Data Preprocessing Started ---")
    print(f"CSV Path: {csv_path}")
    print(f"Data Dir: {data_dir}")
    print(f"Cache Dir: {cache_dir}")
    print(f"Rows per eq: {args.rows}")
    print(f"--------------------------------------")

    build_aif_dataset(
        csv_path=csv_path,
        data_dir=data_dir,
        cache_dir=cache_dir,
        max_rows_per_eq=args.rows
    )
    
    print("\nPreprocessing complete!")

if __name__ == '__main__':
    main()
