# run_rdf.py

import os
import sys

import pandas as pd

from src.io import load_config, load_snapshots, validate_snapshots
from src.rdf import compute_rdf_snapshots
from src.potential import boltzmann_inversion, save_potential_csv


def main(config_path="configs/parameters.yaml"):
    cfg = load_config(config_path)

    print("Loading snapshots...")
    snapshots = load_snapshots(cfg)

    print("Validating snapshots...")
    N = validate_snapshots(snapshots, cfg)
    print(f"Loaded {len(snapshots)} snapshots with N={N} particles.")

    print("Computing RDF for each snapshot and averaging...")
    centers, g_target, sigma_g, _ = compute_rdf_snapshots(snapshots, cfg)

    os.makedirs("results/target", exist_ok=True)
    os.makedirs("results/potentials", exist_ok=True)

    target_path = os.path.join("results", "target", "rdf_target.csv")
    pd.DataFrame(
        {
            "r": centers,
            "g_target": g_target,
            "uncertainty": sigma_g,
        }
    ).to_csv(target_path, index=False)

    print(f"Saved target RDF to {target_path}")

    print("Performing Boltzmann inversion: U0(r) = -kBT ln g_target(r)")
    U0 = boltzmann_inversion(centers, g_target, cfg)

    potential_path = os.path.join("results", "potentials", "potential_000.csv")
    save_potential_csv(potential_path, centers, U0)

    print(f"Saved initial potential to {potential_path}")

    print()
    print("Low-density / potential-of-mean-force approximation complete.")
    print("If your system is truly very dilute, potential_000.csv may already")
    print("be a good approximation to the effective pair potential.")
    print()
    print("To refine with IBI, run:")
    print("    python run_ibi.py")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()