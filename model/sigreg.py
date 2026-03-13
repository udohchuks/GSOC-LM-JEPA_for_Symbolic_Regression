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