# src/potential.py

import numpy as np
import pandas as pd


def fit_mask(r, cfg):
    """
    Boolean mask for the radial range where the potential is allowed
    to be fitted/updated.
    """
    r = np.asarray(r, dtype=np.float64)
    p = cfg.get("potential", {})

    r_min = float(p.get("fit_r_min", r[0]))
    r_max = float(p.get("fit_r_max", r[-1]))

    return (r >= r_min) & (r <= r_max)


def _tail_mask(r, cfg):
    """
    Mask for the tail region used to shift U(r) toward zero.
    """
    r = np.asarray(r, dtype=np.float64)
    p = cfg.get("potential", {})

    r_min = float(p.get("fit_r_min", r[0]))
    r_max = float(p.get("fit_r_max", r[-1]))
    frac = float(p.get("tail_fraction", 0.2))

    if frac <= 0.0:
        return np.zeros_like(r, dtype=bool)

    start = r_max - frac * max(0.0, r_max - r_min)
    return (r >= start) & (r <= r_max)


def smooth_potential(U, mask, sigma_bins):
    """
    Simple Gaussian smoothing of U(r) inside the fitting mask.
    """
    U = np.asarray(U, dtype=np.float64)

    if sigma_bins is None:
        return U

    sigma_bins = float(sigma_bins)

    if sigma_bins <= 0.0:
        return U

    if not np.any(mask):
        return U

    radius = max(1, int(4.0 * sigma_bins + 0.5))
    x = np.arange(-radius, radius + 1, dtype=np.float64)

    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= np.sum(kernel)

    U_smooth = np.convolve(U, kernel, mode="same")

    out = U.copy()
    out[mask] = U_smooth[mask]

    return out


def _shift_tail_to_zero(U, r, cfg):
    """
    Subtract the average tail value in the fitting region so that
    the potential approaches zero near the end of the range.
    """
    U = np.asarray(U, dtype=np.float64).copy()

    mask = fit_mask(r, cfg)
    tail = _tail_mask(r, cfg) & mask

    if np.any(tail):
        shift = np.mean(U[tail])
        U[mask] -= shift

    return U


def boltzmann_inversion(r, g, cfg):
    """
    Low-density / potential-of-mean-force initial guess:

        U_0(r) = -kBT ln g(r)

    Parameters
    ----------
    r : np.ndarray
        RDF bin centers.
    g : np.ndarray
        Target RDF.
    cfg : dict
        Configuration dictionary.

    Returns
    -------
    U : np.ndarray
        Initial tabulated potential.
    """
    r = np.asarray(r, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)

    p = cfg.get("potential", {})
    kBT = float(cfg["system"]["temperature"])

    g_floor = float(p.get("g_floor", 1.0e-12))
    cap = float(p.get("potential_cap", 50.0))

    r_min = float(p.get("fit_r_min", r[0]))
    r_max = float(p.get("fit_r_max", r[-1]))

    g_safe = np.maximum(g, g_floor)

    U_full = -kBT * np.log(g_safe)
    U_full = np.clip(U_full, -cap, cap)

    U = np.zeros_like(r)

    mask = fit_mask(r, cfg)

    # Hard-core-like region below fit_r_min:
    # make it strongly repulsive rather than undefined.
    U[r < r_min] = cap

    # Fit region: use Boltzmann inversion.
    U[mask] = U_full[mask]

    # Beyond fitting range: set potential to zero.
    U[r > r_max] = 0.0

    # Optional smoothing.
    sigma_bins = float(p.get("smooth_sigma_bins", 0.0))
    U = smooth_potential(U, mask, sigma_bins)

    # Make tail go to zero.
    U = _shift_tail_to_zero(U, r, cfg)

    # Enforce boundary conditions again after shifting/smoothing.
    U[r > r_max] = 0.0
    U[r < r_min] = cap

    return np.clip(U, -cap, cap)


def ibi_update(r, U, g_sim, g_target, cfg):
    """
    IBI potential update:

        U_{n+1}(r) = U_n(r)
            + alpha * kBT * ln[g_n(r) / g_target(r)]

    Parameters
    ----------
    r : np.ndarray
        RDF bin centers.
    U : np.ndarray
        Current potential.
    g_sim : np.ndarray
        RDF simulated using current potential.
    g_target : np.ndarray
        Target RDF from input snapshots.
    cfg : dict
        Configuration dictionary.

    Returns
    -------
    U_new : np.ndarray
        Updated potential.
    """
    r = np.asarray(r, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    g_sim = np.asarray(g_sim, dtype=np.float64)
    g_target = np.asarray(g_target, dtype=np.float64)

    p = cfg.get("potential", {})
    ibi_cfg = cfg.get("ibi", {})

    kBT = float(cfg["system"]["temperature"])
    alpha = float(ibi_cfg.get("alpha", 0.2))

    g_floor = float(p.get("g_floor", 1.0e-12))
    cap = float(p.get("potential_cap", 50.0))

    r_min = float(p.get("fit_r_min", r[0]))
    r_max = float(p.get("fit_r_max", r[-1]))

    mask = fit_mask(r, cfg)

    g_sim_safe = np.maximum(g_sim, g_floor)
    g_target_safe = np.maximum(g_target, g_floor)

    delta_U = alpha * kBT * np.log(g_sim_safe / g_target_safe)

    U_new = U.copy()
    U_new[mask] += delta_U[mask]

    U_new = np.clip(U_new, -cap, cap)

    # Maintain boundary behavior.
    U_new[r < r_min] = cap
    U_new[r > r_max] = 0.0

    # Smooth updated potential.
    sigma_bins = float(p.get("smooth_sigma_bins", 0.0))
    U_new = smooth_potential(U_new, mask, sigma_bins)

    # Keep tail near zero.
    U_new = _shift_tail_to_zero(U_new, r, cfg)

    # Enforce boundary conditions again.
    U_new[r > r_max] = 0.0
    U_new[r < r_min] = cap

    return np.clip(U_new, -cap, cap)


def save_potential_csv(path, r, U):
    """Save tabulated potential as CSV with columns r,U."""
    df = pd.DataFrame(
        {
            "r": np.asarray(r, dtype=np.float64),
            "U": np.asarray(U, dtype=np.float64),
        }
    )
    df.to_csv(path, index=False)


def load_potential_csv(path):
    """Load tabulated potential CSV with columns r,U."""
    df = pd.read_csv(path)

    if "r" not in df.columns or "U" not in df.columns:
        raise ValueError(f"Potential file {path} must contain columns 'r' and 'U'.")

    r = df["r"].to_numpy(dtype=np.float64)
    U = df["U"].to_numpy(dtype=np.float64)

    return r, U