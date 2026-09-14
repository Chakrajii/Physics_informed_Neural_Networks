# src/io.py

import glob
import os

import numpy as np
import pandas as pd
import yaml


def load_config(path):
    """Load YAML configuration file."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_snapshots(cfg):
    """
    Load snapshots according to cfg['snapshots'].

    Returns
    -------
    snapshots : list of np.ndarray
        Each snapshot has shape (N, dimension).
    """
    snap_cfg = cfg["snapshots"]
    fmt = str(snap_cfg.get("format", "csv")).lower()

    if fmt == "csv":
        return _load_csv_snapshots(cfg)
    elif fmt == "xyz":
        return _load_xyz_snapshots(cfg)
    else:
        raise ValueError(f"Unknown snapshot format: {fmt}")


def _load_csv_snapshots(cfg):
    """
    Load snapshots from a long-format CSV file:
        mc_iteration, particle_id, x, y [, z]

    This matches the output of lj_mc_simulation.py.
    """
    snap_cfg = cfg["snapshots"]
    system_cfg = cfg["system"]

    path = snap_cfg["path"]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Snapshot file not found: {path}\n"
            "Please set snapshots.path in configs/parameters.yaml "
            "to the trajectory CSV produced by lj_mc_simulation.py."
        )

    dim = int(system_cfg["dimension"])
    box_length = float(system_cfg["box_length"])

    mc_col = snap_cfg.get("mc_column", "mc_iteration")
    particle_col = snap_cfg.get("particle_column", "particle_id")
    x_col = snap_cfg.get("x_column", "x")
    y_col = snap_cfg.get("y_column", "y")
    z_col = snap_cfg.get("z_column", None)

    df = pd.read_csv(path)

    if mc_col not in df.columns:
        # If there is no snapshot label, treat the whole file as one snapshot.
        df[mc_col] = 0

    required = [x_col, y_col]
    if dim == 3:
        if z_col is None:
            raise ValueError("For dimension=3, snapshots.z_column must be set.")
        required.append(z_col)

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {path}")

    snapshots = []

    for _, group in df.groupby(mc_col, sort=True):
        if particle_col is not None and particle_col in group.columns:
            group = group.sort_values(particle_col)

        coords = group[required].to_numpy(dtype=np.float64)

        # Wrap coordinates into [0, L). The IBI/RDF code assumes periodic images.
        coords = np.mod(coords, box_length)

        snapshots.append(coords)

    if len(snapshots) == 0:
        raise RuntimeError(f"No snapshots were loaded from {path}")

    return snapshots


def _load_xyz_snapshots(cfg):
    """
    Load snapshots from .xyz files.

    Expected standard XYZ format:
        N
        comment
        x y [z]
        x y [z]
        ...

    If the coordinate line contains non-numeric tokens, such as atom names,
    those tokens are ignored.
    """
    snap_cfg = cfg["snapshots"]
    system_cfg = cfg["system"]

    path = snap_cfg["path"]
    dim = int(system_cfg["dimension"])
    box_length = float(system_cfg["box_length"])

    files = sorted(glob.glob(path))
    if len(files) == 0 and os.path.isfile(path):
        files = [path]

    if len(files) == 0:
        raise FileNotFoundError(f"No XYZ snapshots found for path: {path}")

    snapshots = []

    for file in files:
        with open(file, "r") as f:
            lines = f.readlines()

        if len(lines) < 2:
            raise ValueError(f"XYZ file {file} is too short.")

        n_expected = int(lines[0].strip())
        coords = []

        for line in lines[2:2 + n_expected]:
            vals = []
            for token in line.split():
                try:
                    vals.append(float(token))
                except ValueError:
                    # Ignore non-numeric tokens such as element names.
                    pass

            if len(vals) >= dim:
                coords.append(vals[:dim])

        if len(coords) != n_expected:
            raise ValueError(
                f"XYZ file {file} says {n_expected} particles, "
                f"but {len(coords)} valid coordinate lines were parsed."
            )

        coords = np.asarray(coords, dtype=np.float64)
        coords = np.mod(coords, box_length)

        snapshots.append(coords)

    return snapshots


def validate_snapshots(snapshots, cfg):
    """
    Basic consistency checks:
      - all snapshots have same N
      - coordinates are finite
      - dimension is correct
      - coordinates are inside the periodic box after wrapping

    Returns
    -------
    N : int
        Number of particles per snapshot.
    """
    if len(snapshots) == 0:
        raise RuntimeError("Snapshot list is empty.")

    dim = int(cfg["system"]["dimension"])
    box_length = float(cfg["system"]["box_length"])

    if box_length <= 0:
        raise ValueError("system.box_length must be positive.")

    N = snapshots[0].shape[0]

    for k, snap in enumerate(snapshots):
        if snap.shape[0] != N:
            raise ValueError(
                f"Snapshot {k} has {snap.shape[0]} particles, "
                f"but snapshot 0 has {N}."
            )

        if snap.shape[1] != dim:
            raise ValueError(
                f"Snapshot {k} has dimension {snap.shape[1]}, "
                f"but system.dimension is {dim}."
            )

        if not np.all(np.isfinite(snap)):
            raise ValueError(f"Snapshot {k} contains non-finite coordinates.")

        if np.any(snap < 0.0) or np.any(snap >= box_length):
            print(
                f"Warning: snapshot {k} has coordinates outside [0, L). "
                "They will be interpreted with periodic wrapping."
            )

    return N