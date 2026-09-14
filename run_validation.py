# run_validation.py

import os
import sys

import numpy as np
import pandas as pd

from src.io import load_config
from src.rdf import compute_rdf_snapshots
from src.potential import load_potential_csv, fit_mask
from src.monte_carlo import TabulatedMC
from src.ibi import rdf_error


def main(config_path="configs/parameters.yaml"):
    cfg = load_config(config_path)

    target_path = os.path.join("results", "target", "rdf_target.csv")
    if not os.path.exists(target_path):
        raise FileNotFoundError(
            f"Target RDF not found at {target_path}. Run run_rdf.py first."
        )

    target_df = pd.read_csv(target_path)
    r_target = target_df["r"].to_numpy(dtype=np.float64)
    g_target = target_df["g_target"].to_numpy(dtype=np.float64)

    if "uncertainty" in target_df.columns:
        sigma_target = target_df["uncertainty"].to_numpy(dtype=np.float64)
    else:
        sigma_target = np.zeros_like(g_target)

    potential_path = os.path.join(
        "results",
        "potentials",
        "effective_potential.csv",
    )

    if not os.path.exists(potential_path):
        fallback = os.path.join("results", "potentials", "potential_000.csv")
        if os.path.exists(fallback):
            print("effective_potential.csv not found.")
            print(f"Falling back to {fallback}.")
            potential_path = fallback
        else:
            raise FileNotFoundError(
                "No effective potential found. Run run_ibi.py first."
            )

    print(f"Loading potential from {potential_path}")
    r_pot, U_pot = load_potential_csv(potential_path)

    # Interpolate potential onto target RDF grid if necessary.
    if len(r_pot) == len(r_target) and np.allclose(r_pot, r_target):
        U = U_pot
        r = r_target
    else:
        print("Interpolating potential onto target RDF grid.")
        U = np.interp(r_target, r_pot, U_pot, left=U_pot[0], right=0.0)
        r = r_target

    mc_cfg = cfg.get("monte_carlo", {})
    val_cfg = cfg.get("validation", {})

    equilibration_sweeps = int(
        val_cfg.get(
            "equilibration_sweeps",
            mc_cfg.get("equilibration_sweeps", 5000),
        )
    )

    production_sweeps = int(
        val_cfg.get(
            "production_sweeps",
            mc_cfg.get("production_sweeps", 10000),
        )
    )

    sample_every_sweeps = int(
        val_cfg.get(
            "sample_every_sweeps",
            mc_cfg.get("sample_every_sweeps", 50),
        )
    )

    seed = int(val_cfg.get("seed", mc_cfg.get("seed", 9999)))
    adapt_displacement = bool(mc_cfg.get("adapt_displacement", True))

    print("Starting independent validation MC simulation.")
    print(f"  equilibration sweeps = {equilibration_sweeps}")
    print(f"  production sweeps    = {production_sweeps}")
    print(f"  sample every         = {sample_every_sweeps}")

    # Independent validation: random initial positions.
    mc = TabulatedMC(
        r,
        U,
        cfg,
        positions=None,
        seed=seed,
    )

    acceptance = mc.equilibrate(
        equilibration_sweeps,
        adapt=adapt_displacement,
    )

    print(f"Validation equilibration acceptance = {acceptance:.4f}")
    print(f"Final max_displacement = {mc.max_disp:.4g}")

    snapshots = mc.sample_production(
        production_sweeps,
        sample_every_sweeps,
    )

    print("Computing validation RDF...")
    centers, g_val, sigma_val, _ = compute_rdf_snapshots(snapshots, cfg)

    # If grids differ slightly, interpolate target onto validation grid.
    if len(centers) == len(r_target) and np.allclose(centers, r_target):
        r_use = r_target
        g_target_use = g_target
        sigma_target_use = sigma_target
    else:
        print("Interpolating target RDF onto validation RDF grid.")
        r_use = centers
        g_target_use = np.interp(centers, r_target, g_target, left=1.0, right=1.0)
        sigma_target_use = np.interp(
            centers,
            r_target,
            sigma_target,
            left=1.0e-12,
            right=1.0e-12,
        )

    os.makedirs("results/validation", exist_ok=True)

    val_rdf_path = os.path.join("results", "validation", "validation_rdf.csv")
    pd.DataFrame(
        {
            "r": centers,
            "g_validation": g_val,
            "uncertainty": sigma_val,
            "g_target": g_target_use,
        }
    ).to_csv(val_rdf_path, index=False)

    print(f"Saved validation RDF to {val_rdf_path}")

    mask = fit_mask(r_use, cfg)

    err_unweighted = rdf_error(
        g_val,
        g_target_use,
        mask,
        sigma=sigma_target_use,
        weighted=False,
    )

    err_weighted = rdf_error(
        g_val,
        g_target_use,
        mask,
        sigma=sigma_target_use,
        weighted=True,
    )

    summary_path = os.path.join("results", "validation", "validation_summary.csv")
    pd.DataFrame(
        [
            {
                "acceptance": acceptance,
                "max_displacement": mc.max_disp,
                "rmsd_unweighted": err_unweighted,
                "rmsd_weighted": err_weighted,
            }
        ]
    ).to_csv(summary_path, index=False)

    print()
    print("Validation summary")
    print("------------------")
    print(f"Unweighted RMSD = {err_unweighted:.6g}")
    print(f"Weighted RMSD   = {err_weighted:.6g}")
    print()
    print("If the validation RDF matches the target RDF within statistical")
    print("uncertainty, the inferred effective potential is successful.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()