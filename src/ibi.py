# src/ibi.py

import numpy as np

from src.potential import fit_mask


def rdf_error(g_sim, g_target, mask, sigma=None, weighted=False):
    """
    Compute RMSD between simulated and target RDFs over mask.

    If weighted=True and sigma is provided, compute weighted RMSD:

        sqrt(mean[((g_sim - g_target) / sigma)^2])

    Otherwise compute ordinary RMSD:

        sqrt(mean[(g_sim - g_target)^2])
    """
    g_sim = np.asarray(g_sim, dtype=np.float64)
    g_target = np.asarray(g_target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    if not np.any(mask):
        return float("inf")

    diff = g_sim[mask] - g_target[mask]

    if weighted and sigma is not None:
        sigma = np.asarray(sigma, dtype=np.float64)[mask]

        # Avoid division by zero.
        sigma_safe = np.where(sigma > 1.0e-12, sigma, 1.0e-12)

        return float(np.sqrt(np.mean((diff / sigma_safe) ** 2)))

    return float(np.sqrt(np.mean(diff * diff)))