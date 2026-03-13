import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("__file___"))))
from model.sigreg import sigreg_loss, monitor_collapse

# Test 1: random embeddings — should be low loss (already roughly Gaussian)
z_random = torch.randn(32, 512)
loss = sigreg_loss(z_random)
print(f'Random embeddings loss: {loss.item():.4f}  (expect low ~0.0)')

# Test 2: collapsed embeddings — all same vector
z_collapsed = torch.ones(32, 512)
loss = sigreg_loss(z_collapsed)
print(f'Collapsed embeddings loss: {loss.item():.4f}  (expect high)')

# Test 3: monitor collapse detection
health = monitor_collapse(z_random)
print(f'Random health: {health}')

health = monitor_collapse(z_collapsed)
print(f'Collapsed health: {health}')