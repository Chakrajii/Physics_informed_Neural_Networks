# run_ibi.py

import os
import sys

import numpy as np
import pandas as pd

from src.io import load_config, load_snapshots, validate_snapshots
from src.rdf import compute_rdf_snapshots, radial_bins
from src.potential import (
    boltzmann_inversion,
    ibi_update,
    save_potential_csv,
    load_potential_csv,
    fit_mask,
)
from src.monte_carlo import TabulatedMC
from src.ibi import rdf_error


def _load_or_build_target(cfg):
    """
    Load target RDF if it exists and matches the current RDF grid.
    Otherwise compute it from snapshots.
    """
    target_path = os.path.join("results", "target", "rdf_target.csv")

    rdf_cfg = cfg["rdf"]
    _, expected_centers = radial_bins(
        rdf_cfg.get("r_min", 0.0),
        rdf_cfg.get("r_max", float(cfg["system"]["box_length"]) / 2.0),
        rdf_cfg.get("dr", 0.01),
    )

    if os.path.exists(target_path):
        df = pd.read_csv(target_path)

        r = df["r"].to_numpy(dtype=np.float64)
        g = df["g_target"].to_numpy(dtype=np.float64)

        if "uncertainty" in df.columns:
            sigma = df["uncertainty"].to_numpy(dtype=np.float64)
        else:
            sigma = np.zeros_like(g)

        if len(r) == len(expected_centers) and np.allclose(r, expected_centers):
            print(f"Loaded existing target RDF from {target_path}")
            return r, g, sigma

        print("Existing rdf_target.csv does not match current RDF grid.")
        print("Recomputing target RDF from snapshots.")

    print("Computing target RDF from snapshots...")
    snapshots = load_snapshots(cfg)
    validate_snapshots(snapshots, cfg)

    centers, g_target, sigma_g, _ = compute_rdf_snapshots(snapshots, cfg)

    os.makedirs("results/target", exist_ok=True)
    pd.DataFrame(
        {
            "r": centers,
            "g_target": g_target,
            "uncertainty": sigma_g,
        }
    ).to_csv(target_path, index=False)

    print(f"Saved target RDF to {target_path}")

    return centers, g_target, sigma_g


def _load_or_build_initial_potential(r, g_target, cfg):
    """
    Load potential_000.csv if present, otherwise create it by Boltzmann inversion.
    """
    pot_path = os.path.join("results", "potentials", "potential_000.csv")

    if os.path.exists(pot_path):
        r_pot, U_pot = load_potential_csv(pot_path)

        if len(r_pot) == len(r) and np.allclose(r_pot, r):
            print(f"Loaded existing initial potential from {pot_path}")
            return U_pot

        print("Existing potential_000.csv grid does not match target RDF grid.")
        print("Interpolating initial potential onto current grid.")
        U = np.interp(r, r_pot, U_pot, left=U_pot[0], right=0.0)
        return U

    print("Computing initial potential by Boltzmann inversion...")
    U0 = boltzmann_inversion(r, g_target, cfg)

    os.makedirs("results/potentials", exist_ok=True)
    save_potential_csv(pot_path, r, U0)

    print(f"Saved initial potential to {pot_path}")

    return U0


def _load_initial_positions_if_requested(cfg):
    """
    Optionally start IBI MC from the last input snapshot.
    """
    ibi_cfg = cfg.get("ibi", {})
    use_prev = bool(ibi_cfg.get("use_previous_positions", True))

    if not use_prev:
        return None

    try:
        snapshots = load_snapshots(cfg)
        validate_snapshots(snapshots, cfg)
        print("Starting IBI MC from the last input snapshot.")
        return snapshots[-1]
    except Exception as e:
        print("Could not load input snapshots for initial positions:")
        print(f"  {e}")
        print("Using random initial positions instead.")
        return None


def main(config_path="configs/parameters.yaml"):
    cfg = load_config(config_path)

    os.makedirs("results/potentials", exist_ok=True)
    os.makedirs("results/rdf", exist_ok=True)
    os.makedirs("results/convergence", exist_ok=True)

    # Load target RDF.
    r, g_target, sigma_target = _load_or_build_target(cfg)

    # Load or construct initial potential.
    current_U = _load_or_build_initial_potential(r, g_target, cfg)

    # Optional starting configuration.
    initial_positions = _load_initial_positions_if_requested(cfg)

    ibi_cfg = cfg.get("ibi", {})
    mc_cfg = cfg.get("monte_carlo", {})

    max_iterations = int(ibi_cfg.get("max_iterations", 20))
    tolerance = float(ibi_cfg.get("tolerance", 1.0e-2))
    weighted_error = bool(ibi_cfg.get("weighted_error", False))
    use_prev = bool(ibi_cfg.get("use_previous_positions", True))

    equilibration_sweeps = int(mc_cfg.get("equilibration_sweeps", 1000))
    production_sweeps = int(mc_cfg.get("production_sweeps", 5000))
    sample_every_sweeps = int(mc_cfg.get("sample_every_sweeps", 50))
    adapt_displacement = bool(mc_cfg.get("adapt_displacement", True))
    seed = int(mc_cfg.get("seed", 1234))

    current_positions = initial_positions

    records = []
    effective_U = current_U.copy()

    print()
    print("Starting IBI loop.")
    print(f"  max_iterations = {max_iterations}")
    print(f"  tolerance      = {tolerance}")
    print(f"  alpha          = {ibi_cfg.get('alpha', 0.2)}")
    print()

    converged = False

    for iteration in range(max_iterations):
        print(f"IBI iteration {iteration:03d}")

        # Run MC with current potential.
        mc = TabulatedMC(
            r,
            current_U,
            cfg,
            positions=current_positions,
            seed=seed + iteration,
        )

        print("  Equilibrating MC...")
        acceptance = mc.equilibrate(
            equilibration_sweeps,
            adapt=adapt_displacement,
        )
        print(
            f"  Equilibration acceptance = {acceptance:.4f}, "
            f"max_disp = {mc.max_disp:.4g}"
        )

        print("  Production MC...")
        sim_snapshots = mc.sample_production(
            production_sweeps,
            sample_every_sweeps,
        )

        if use_prev:
            current_positions = mc.pos.copy()

        # Compute simulated RDF.
        centers, g_sim, sigma_sim, _ = compute_rdf_snapshots(sim_snapshots, cfg)

        if len(centers) != len(r):
            raise RuntimeError(
                "Simulated RDF grid does not match target RDF grid. "
                "Do not change rdf.r_min/rdf.r_max/rdf.dr during an IBI run."
            )

        rdf_path = os.path.join("results", "rdf", f"rdf_{iteration:03d}.csv")
        pd.DataFrame(
            {
                "r": centers,
                "g_sim": g_sim,
                "uncertainty": sigma_sim,
            }
        ).to_csv(rdf_path, index=False)

        print(f"  Saved simulated RDF to {rdf_path}")

        # Error metric.
        mask = fit_mask(r, cfg)
        err = rdf_error(
            g_sim,
            g_target,
            mask,
            sigma=sigma_target,
            weighted=weighted_error,
        )

        print(f"  RDF error = {err:.6g}")

        records.append(
            {
                "iteration": iteration,
                "rmsd": err,
                "acceptance": acceptance,
                "max_displacement": mc.max_disp,
            }
        )

        if err < tolerance:
            print("  Converged.")
            effective_U = current_U.copy()
            converged = True
            break

        # IBI update.
        print("  Updating potential with IBI...")
        current_U = ibi_update(r, current_U, g_sim, g_target, cfg)

        pot_path = os.path.join(
            "results",
            "potentials",
            f"potential_{iteration + 1:03d}.csv",
        )
        save_potential_csv(pot_path, r, current_U)
        print(f"  Saved updated potential to {pot_path}")

        effective_U = current_U.copy()

        print()

    # Save final effective potential.
    eff_path = os.path.join("results", "potentials", "effective_potential.csv")
    save_potential_csv(eff_path, r, effective_U)
    print(f"Saved final effective potential to {eff_path}")

    # Save convergence history.
    conv_path = os.path.join("results", "convergence", "convergence.csv")
    pd.DataFrame(records).to_csv(conv_path, index=False)
    print(f"Saved convergence history to {conv_path}")

    if not converged:
        print()
        print("IBI did not reach the requested tolerance within max_iterations.")
        print("You can increase ibi.max_iterations, decrease ibi.alpha,")
        print("increase MC sampling, or adjust the RDF fitting range.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()