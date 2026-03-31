import yaml
import torch
from models.model import LLMJEPA

def test():
    with open('configs/small.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    print(f"Config loaded: d_model={cfg['model']['d_model']}, n_heads={cfg['model']['n_heads']}")
    
    try:
        model = LLMJEPA(**cfg['model'])
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Success! Parameter count: {total_params:,}")
        
        # Check divisibility explicitly for safety
        d_model = cfg['model']['d_model']
        n_heads = cfg['model']['n_heads']
        if d_model % n_heads != 0:
            print(f"❌ Error: d_model ({d_model}) not divisible by n_heads ({n_heads})")
            
        bottleneck = int(d_model * cfg['model']['predictor']['pred_bottleneck_ratio'])
        p_heads = cfg['model']['predictor']['pred_n_heads']
        if bottleneck % p_heads != 0:
            print(f"❌ Error: predictor bottleneck ({bottleneck}) not divisible by pred_n_heads ({p_heads})")
            
    except Exception as e:
        print(f"❌ Error instantiating model: {e}")

if __name__ == '__main__':
    test()
