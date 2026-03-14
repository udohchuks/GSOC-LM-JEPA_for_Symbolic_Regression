"""
SIGReg: Spectral Isotropic Gaussian Regularisation.
From Balestriero & LeCun (2025) — LeJEPA paper.

Purpose:
    Prevent representation collapse in JEPA training.
    Collapse = all equations map to the same embedding vector.
    If collapse happens, JEPA loss goes to zero but the model
    learns nothing — every equation looks the same.

How it works:
    Force the distribution of embeddings across a batch to be
    approximately isotropic Gaussian (mean=0, variance=1 in all directions).

   Uses Epps-Pulley test via random 1D projections.
   Real/imaginary parts computed separately for numerical stability.

Why this works:
    An isotropic Gaussian cannot collapse — by definition it has
    spread in all directions. Forcing embeddings toward this distribution
    prevents the model from taking the easy path of mapping everything
    to one point.

Why better than EMA teacher:
    EMA requires a second copy of the model (~2x memory).
    SIGReg is O(N) in batch size, zero extra parameters,
    works with a single model copy.
"""

import torch
import torch.nn.functional as F


def sigreg_loss(z: torch.Tensor, global_step: int = 0, 
                    num_projections: int = 512, num_integration_points: int = 17, 
                    integration_range: float = 5.0) -> torch.Tensor:
    """
    SIGReg loss using Epps-Pulley statistic (Algorithm 1, LeJEPA paper).

    Args:
        z:             [B, d_model]  batch of embedding vectors

        n_projections: number of random 1D projection directions (M in paper)
                       Paper recommends 1024 for best results
                       default. Even 64 works due to SGD resampling effect.

        num_intergration_points :  number of integration points for trapezoidal rule
                       Paper uses 17 — sufficient for accurate integration

        integration_range:       integration domain [-t_range, t_range]
                       Paper uses [-5, 5]

        global_step:   used to seed random projection sampling
                       ensures different projections each step (key for
                       SGD to cover the full space over training)

    Returns:
        scalar SIGReg loss
    """
    B, dim = z.shape
    device = z.device
    dtype = z.dtype

    g = torch.Generator(device=device)
    g.manual_seed(global_step)

    # Step 2: Generate random projection directions -> shape (dim x num_projections)
    A = torch.randn(dim, num_projections, generator=g,  device=device, dtype=dtype)
    A = F.normalize(A, p=2, dim=0) # unit vector

    # Shape: (num_integration_points,)
    t = torch.linspace(-integration_range, integration_range, 
                       num_integration_points, device=device, dtype=z.dtype)
    

    #  Theoretical CF for N(0,1) -> exp(-0.5 * t^2)
    # Shape: (num_integration_points,)
    target_cf = torch.exp(-0.5 * t ** 2)

    # Project embeddings onto each direction
    # (B, K) * (K, S) -> (B, S)
    z_proj = z @ A
    
    # Multiply by t: (B, S, 1) * (1, 1, T) -> (B, S, T)
    x_t = z_proj.unsqueeze(-1) * t.unsqueeze(0).unsqueeze(0)


    ecf_real = torch.cos(x_t).mean(dim=0) # (S, T)
    ecf_imag = torch.sin(x_t).mean(dim=0) # (S, T)

    err = ((ecf_real - target_cf.unsqueeze(0)) ** 2 +
            ecf_imag ** 2) * target_cf.unsqueeze(0) # (S, T)
    
    loss = torch.trapezoid(err, t, dim=1).mean()  
    
    return loss


def monitor_collapse(z: torch.Tensor) -> dict:
    """
    Monitor embedding health during training.
    Call this every N steps and log the results.

    Returns dict of metrics — if variance drops below 0.1,
    collapse is happening and you need to increase lambda_sig.

    Args:
        z: [B, d_model] batch of embeddings
    """
    with torch.no_grad():
        # Mean variance across all dimensions
        # Healthy: ~1.0   Collapsing: approaching 0

        variance = z.var(dim=0).mean().item()
        mean_norm = z.mean(dim=0).norm().item()

        # Standard deviation of norms
        # Healthy: embeddings have varied magnitudes
        # Collapsing: all norms become equal (or zero)
        norms       = z.norm(dim=-1)
        norm_std    = norms.std().item()
        norm_mean   = norms.mean().item()

        # Cosine similarity between random pairs
        # Healthy: close to 0 (embeddings point in different directions)
        # Collapsing: approaching 1 (all embeddings point same direction)

        if z.shape[0] >= 2:
            z_norm    = F.normalize(z, dim=-1)
            sim_matrix = z_norm @ z_norm.T
            # Exclude diagonal (self-similarity = 1 always)
            mask       = ~torch.eye(z.shape[0], dtype=torch.bool,
                                    device=z.device)
            mean_cosine = sim_matrix[mask].mean().item()
        else:
            mean_cosine = 0.0
    
    return {
        'variance':     variance,
        'mean_norm':    mean_norm, 
        'norm_mean':    norm_mean,
        'norm_std':     norm_std,
        'mean_cosine':  mean_cosine,
        'collapsed':    variance < 0.1 or mean_cosine > 0.9,
    }

