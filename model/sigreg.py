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

    Uses random 1D projections + characteristic function matching:
    1. Project embeddings onto random unit vectors
    2. Compare the projected distribution to N(0,1)
       using characteristic function (cos and sin moments)
    3. Penalise deviation from Gaussian

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
import torch.nn as nn


def sigreg_loss(z: torch.Tensor, n_projections: int = 64, lambda_sig: float = 1.0) -> torch.Tensor:
    """
    Compute SIGReg loss for a batch of embeddings.

    Args:
        z:             [B, d_model]  batch of embedding vectors
        n_projections: number of random projection directions
                       more projections = better Gaussian approximation
                       but slower. 64 is a good default.
        lambda_sig:    weight of the SIGReg loss

    Returns:
        scalar loss — zero when z is perfectly isotropic Gaussian
    """
    B, dim = z.shape

    # Step 1: Normalise embeddings to zero mean
    z = z - z.mean(dim=0, keepdim=True) # shape z -> B x dim

    # Step 2: Generate random projection directions
    directions = torch.randn(n_projections, dim, device=z.device)
    directions = F.normalize(directions, dim=-1) # unit vector

    # Step 3: Project embeddings onto each direction
    projections = z @ directions.T

    # Step 4: Match characteristic function of N(0,1)
    # The characteristic function of N(0,1) at frequency t is:
    #   phi(t) = E[e^(itX)] = e^(-t^2/2)
    # Which gives:
    #   E[cos(tX)] = e^(-t^2/2)    (real part)
    #   E[sin(tX)] = 0              (imaginary part — N(0,1) is symmetric)
    #
    # We evaluate at t=1 (standard frequency):
    #   Target cos moment: e^(-0.5) ≈ 0.6065
    #   Target sin moment: 0.0

    cos_emperical = torch.cos(projections).mean(dim=0)
    sin_emperical = torch.sin(projections).mean(dim=0)

    cos_target = torch.exp(torch.tensor(-0.5, device=z.device))

    sin_target = torch.tensor(0.0, device=z.device)

    loss_cos = (cos_emperical - cos_target).pow(2).mean()
    loss_sin = (sin_emperical - sin_target).pow(2).mean()

    return lambda_sig * (loss_cos + loss_sin)


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
        'norm_mean':    norm_mean,
        'norm_std':     norm_std,
        'mean_cosine':  mean_cosine,
        'collapsed':    variance < 0.1 or mean_cosine > 0.9,
    }

