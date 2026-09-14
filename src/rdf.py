# src/rdf.py

import numpy as np


def radial_bins(r_min, r_max, dr):
    """
    Create radial bin edges and centers.

    Returns
    -------
    edges : np.ndarray
        Bin edges, length n_bins + 1.
    centers : np.ndarray
        Bin centers, length n_bins.
    """
    r_min = float(r_min)
    r_max = float(r_max)
    dr = float(dr)

    if dr <= 0:
        raise ValueError("rdf.dr must be positive.")
    if r_max <= r_min:
        raise ValueError("rdf.r_max must be larger than rdf.r_min.")

    n_bins = int(np.ceil((r_max - r_min) / dr))
    edges = r_min + dr * np.arange(n_bins + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])

    return edges, centers


def shell_measure(edges, dim):
    """
    Area/volume of each radial shell.

    2D: A(r) = pi (r_outer^2 - r_inner^2)
    3D: V(r) = 4/3 pi (r_outer^3 - r_inner^3)
    """
    if dim == 2:
        return np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    elif dim == 3:
        return (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    else:
        raise ValueError("Only dimension=2 or dimension=3 is supported.")


def histogram_pair_distances(snapshot, box_length, dim, edges):
    """
    Count pair distances i < j in radial bins using the minimum image convention.
    """
    snapshot = np.asarray(snapshot, dtype=np.float64)
    N = snapshot.shape[0]
    L = float(box_length)

    H = np.zeros(len(edges) - 1, dtype=np.float64)

    if N < 2:
        return H

    for i in range(N - 1):
        # Pair particle i with particles j > i.
        delta = snapshot[i + 1:, :dim] - snapshot[i, :dim]

        # Minimum image convention.
        delta -= L * np.round(delta / L)

        r = np.sqrt(np.sum(delta * delta, axis=1))

        # Keep only distances inside the RDF range.
        r = r[(r >= edges[0]) & (r < edges[-1])]

        if r.size > 0:
            H += np.histogram(r, bins=edges)[0]

    return H


def normalize_histogram(H, N, box_length, dim, edges):
    """
    Normalize pair histogram to obtain g(r).

    For i < j counting, the ideal-gas expected count in a shell is:

        H_ideal(r) = 1/2 * N * (N - 1) * shell_volume / box_volume

    Therefore:

        g(r) = H(r) / H_ideal(r)
    """
    H = np.asarray(H, dtype=np.float64)

    if N < 2:
        return np.zeros_like(H)

    shell = shell_measure(edges, dim)
    box_volume = float(box_length) ** dim

    ideal_counts = 0.5 * float(N) * float(N - 1) * shell / box_volume

    g = np.zeros_like(H)
    mask = ideal_counts > 0.0
    g[mask] = H[mask] / ideal_counts[mask]

    return g


def compute_rdf_snapshots(snapshots, cfg):
    """
    Compute g_k(r) for each snapshot, then average.

    Returns
    -------
    centers : np.ndarray
        Radial bin centers.
    g_target : np.ndarray
        Ensemble-averaged RDF.
    sigma_g : np.ndarray
        Sample standard deviation over snapshots.
    g_all : np.ndarray
        Array of shape (M, n_bins) with individual snapshot RDFs.
    """
    if len(snapshots) == 0:
        raise RuntimeError("No snapshots provided to compute_rdf_snapshots().")

    system_cfg = cfg["system"]
    rdf_cfg = cfg["rdf"]

    dim = int(system_cfg["dimension"])
    box_length = float(system_cfg["box_length"])

    r_min = float(rdf_cfg.get("r_min", 0.0))
    r_max = float(rdf_cfg.get("r_max", box_length / 2.0))
    dr = float(rdf_cfg.get("dr", 0.01))

    edges, centers = radial_bins(r_min, r_max, dr)

    g_list = []

    for snap in snapshots:
        N = snap.shape[0]
        H = histogram_pair_distances(snap, box_length, dim, edges)
        g = normalize_histogram(H, N, box_length, dim, edges)
        g_list.append(g)

    g_all = np.asarray(g_list, dtype=np.float64)
    g_target = np.mean(g_all, axis=0)

    if g_all.shape[0] > 1:
        sigma_g = np.std(g_all, axis=0, ddof=1)
    else:
        sigma_g = np.zeros_like(g_target)

    return centers, g_target, sigma_g, g_all


def floor_rdf(g, g_floor):
    """Replace values below g_floor by g_floor."""
    return np.maximum(np.asarray(g, dtype=np.float64), float(g_floor))