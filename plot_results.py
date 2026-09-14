# plot_results.py

import os

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError(
        "matplotlib is required for plot_results.py. "
        "Install it with: pip install matplotlib"
    )


def plot_rdf_comparison():
    target_path = os.path.join("results", "target", "rdf_target.csv")
    validation_path = os.path.join(
        "results",
        "validation",
        "validation_rdf.csv",
    )

    if not os.path.exists(target_path):
        print("No target RDF found. Run run_rdf.py first.")
        return

    os.makedirs("results/figures", exist_ok=True)

    target = pd.read_csv(target_path)

    plt.figure(figsize=(7, 5))
    plt.plot(
        target["r"],
        target["g_target"],
        label="Target",
        color="black",
        lw=2,
    )

    if "uncertainty" in target.columns:
        plt.fill_between(
            target["r"],
            target["g_target"] - target["uncertainty"],
            target["g_target"] + target["uncertainty"],
            color="gray",
            alpha=0.35,
            label="Target uncertainty",
        )

    if os.path.exists(validation_path):
        val = pd.read_csv(validation_path)
        plt.plot(
            val["r"],
            val["g_validation"],
            label="Validation",
            color="tab:red",
            lw=2,
            alpha=0.8,
        )

        if "uncertainty" in val.columns:
            plt.fill_between(
                val["r"],
                val["g_validation"] - val["uncertainty"],
                val["g_validation"] + val["uncertainty"],
                color="tab:red",
                alpha=0.25,
                label="Validation uncertainty",
            )
    else:
        print("No validation RDF found. Run run_validation.py to add it.")

    plt.xlabel("r")
    plt.ylabel("g(r)")
    plt.title("Target RDF vs validation RDF")
    plt.legend()
    plt.tight_layout()

    out = os.path.join("results", "figures", "rdf_comparison.png")
    plt.savefig(out, dpi=200)
    print(f"Saved {out}")


def plot_effective_potential():
    potential_path = os.path.join(
        "results",
        "potentials",
        "effective_potential.csv",
    )

    if not os.path.exists(potential_path):
        print("No effective potential found. Run run_ibi.py first.")
        return

    os.makedirs("results/figures", exist_ok=True)

    df = pd.read_csv(potential_path)

    plt.figure(figsize=(7, 5))
    plt.plot(df["r"], df["U"], color="tab:blue", lw=2)
    plt.axhline(0.0, color="black", lw=0.8, ls="--")
    plt.xlabel("r")
    plt.ylabel("U(r)")
    plt.title("Effective pair potential")
    plt.tight_layout()

    out = os.path.join("results", "figures", "effective_potential.png")
    plt.savefig(out, dpi=200)
    print(f"Saved {out}")


def plot_convergence():
    conv_path = os.path.join("results", "convergence", "convergence.csv")

    if not os.path.exists(conv_path):
        print("No convergence file found. Run run_ibi.py first.")
        return

    os.makedirs("results/figures", exist_ok=True)

    df = pd.read_csv(conv_path)

    plt.figure(figsize=(7, 5))
    plt.plot(df["iteration"], df["rmsd"], marker="o", color="tab:green")
    plt.xlabel("IBI iteration")
    plt.ylabel("RMSD")
    plt.yscale("log")
    plt.title("IBI convergence")
    plt.tight_layout()

    out = os.path.join("results", "figures", "convergence.png")
    plt.savefig(out, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    plot_rdf_comparison()
    plot_effective_potential()
    plot_convergence()