"""
2D Monte Carlo particle simulation on an 800x800 lattice, driven by a
tabulated Lennard-Jones potential.

Implements the spec exactly as given, with two flagged interpretive choices
(search "ASSUMPTION" below) where the original spec was ambiguous or, in one
case, likely contains a typo. Both are exposed as toggles so you can switch
behavior with one flag.

------------------------------------------------------------------------
HOW TO RUN AT FULL SPEC SCALE
------------------------------------------------------------------------
Set BURNIN_MC = 50_000 and MOVES_PER_MC = 5000 (both already the defaults
below). At that scale this is ~250 MILLION single-particle update moves for
burn-in alone (each move costs O(N)=499 pairwise distance evaluations against
the other particles -> ~1.25e11 elementary operations total). Even fully
JIT-compiled with Numba, on a single CPU core this is roughly:

    ~25 microseconds/move x 250,000,000 moves =~ 1.5-2 hours for burn-in
    + ~25 microseconds/move x 5000 moves x N_SNAPSHOTS for the sampling phase

Plan accordingly -- run it as a background/batch job, not interactively.
Progress is printed every 10% of burn-in so you can monitor it.

To go faster: increase N_SNAPSHOTS/BURNIN_MC as needed, or parallelize the
inner pairwise-gradient loop with `numba.prange` + `@njit(parallel=True)`
(not done here by default, to keep the script maximally portable).
------------------------------------------------------------------------
"""

import os
import time
import numpy as np
import pandas as pd
from numba import njit
import math

# ============================================================
# PARAMETERS  (edit these)
# ============================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
LJ_CSV_PATH = os.path.join(_HERE, "lj_potential.csv")  # columns: r, U (r = 0.0,0.1,...,800.0)
                                                        # <-- point this at YOUR real CSV
OUTPUT_DIR  = os.path.join(_HERE, "output")            # where snapshot CSVs get written

BOX          = 800     # grid is BOX x BOX integer lattice sites: (0..BOX-1, 0..BOX-1)
N_PARTICLES  = 500
MASS         = 1.0     # m
KB           = 1.0     # Boltzmann constant (reduced units)
TEMPERATURE  = 1.0     # T
DT           = 1.0     # delta t (per spec, item 6)

MOVES_PER_MC   = 500      # spec item 8: 5000 single-particle updates = 1 MC iteration
BURNIN_MC      = 1000    # spec item 9: initial equilibration length, in MC iterations
N_SNAPSHOTS    = 20       # how many post-burn-in snapshots to write (extend/re-run as needed)
SNAPSHOT_EVERY_MC = 1       # ASSUMPTION #2 below: save every 1 MC iteration (=5000 moves) after burn-in

TRAJECTORY_FILE = "positions_trajectory.csv"  # all snapshots appended into ONE file
                                               # (long format: one row per particle per snapshot)

SEED = 42

# ASSUMPTION #1 (position update formula, spec item 6):
# The spec literally states  r_(n+1) = r_n * v_n + 1/2*(-1/m * dU/dr)^2
# This is NOT standard kinematics (position*velocity, and the acceleration
# term is squared so it's always >= 0). Implemented literally when True.
# Standard kinematics fallback (r_(n+1) = r_n + v_(n+1)*dt + 1/2*a*dt^2)
# is used when False. Default: True (literal, as specified).
LITERAL_POSITION_UPDATE = True


# ============================================================
# Load tabulated potential
# ============================================================
def load_potential_table(csv_path):
    df = pd.read_csv(csv_path)
    r = df.iloc[:, 0].to_numpy(dtype=np.float64)
    U = df.iloc[:, 1].to_numpy(dtype=np.float64)
    bin_width = round(float(r[1] - r[0]), 6)
    assert np.isclose(r[0], 0.0), "potential table must start at r=0"
    return U, bin_width


# ============================================================
# Numba-accelerated core
# ============================================================
@njit(cache=True)
def lookup_dUdr(r, U_table, bin_width):
    # "calculate the derivative manually from the next points and taking
    # average" -> central difference using neighboring bins (spec item 5)
    n = U_table.shape[0]
    idx = int(round(r / bin_width))
    if idx <= 0:
        idx = 1
    if idx >= n - 1:
        idx = n - 2
    return (U_table[idx + 1] - U_table[idx - 1]) / (2.0 * bin_width)


@njit(cache=True)
def compute_gradient(i, x, y, U_table, bin_width, n_particles):
    # Pairwise LJ force on particle i from all other particles (standard
    # decomposition of the radial derivative into x/y components).
    fx = 0.0
    fy = 0.0
    xi = x[i]
    yi = y[i]
    for j in range(n_particles):
        if j == i:
            continue
        dx = xi - x[j]
        dy = yi - y[j]
        r = math.sqrt(dx * dx + dy * dy)
        if r < 1e-9:
            continue
        dUdr = lookup_dUdr(r, U_table, bin_width)
        fx += dUdr * dx / r
        fy += dUdr * dy / r
    return fx, fy


@njit(cache=True)
def find_nearest_free(occ, x0, y0, box):
    # spec item 7: "if the place it is going is already taken, move it to
    # the next nearest place" -> expanding ring search for a free site
    if occ[x0, y0] == 0:
        return x0, y0
    for radius in range(1, box):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if -radius < dx < radius and -radius < dy < radius:
                    continue  # interior already covered by smaller radius
                x = x0 + dx
                y = y0 + dy
                if 0 <= x < box and 0 <= y < box and occ[x, y] == 0:
                    return x, y
    return x0, y0  # unreachable at this density (500 / 640000 sites)


@njit(cache=True)
def single_particle_move(i, x, y, vx, vy, occ, U_table, bin_width, mass,
                          box, n_particles, literal_update, dt):
    xi = x[i]
    yi = y[i]
    dUdx, dUdy = compute_gradient(i, x, y, U_table, bin_width, n_particles)

    vx_old = vx[i]
    vy_old = vy[i]

    # --- spec item 5: velocity update ---
    vx_new = vx_old - (1.0 / mass) * dUdx
    vy_new = vy_old - (1.0 / mass) * dUdy

    # --- spec item 6: position update ---
    if literal_update:
        x_new_f = xi * vx_old + 0.5 * (-(1.0 / mass) * dUdx) ** 2
        y_new_f = yi * vy_old + 0.5 * (-(1.0 / mass) * dUdy) ** 2
    else:
        ax = -(1.0 / mass) * dUdx
        ay = -(1.0 / mass) * dUdy
        x_new_f = xi + vx_new * dt + 0.5 * ax * dt * dt
        y_new_f = yi + vy_new * dt + 0.5 * ay * dt * dt

    x_new = int(round(x_new_f))
    y_new = int(round(y_new_f))
    if x_new < 0:
        x_new = 0
    if x_new > box - 1:
        x_new = box - 1
    if y_new < 0:
        y_new = 0
    if y_new > box - 1:
        y_new = box - 1

    occ[xi, yi] = 0  # vacate old site

    if occ[x_new, y_new] != 0:
        x_new, y_new = find_nearest_free(occ, x_new, y_new, box)

    occ[x_new, y_new] = 1
    x[i] = x_new
    y[i] = y_new
    vx[i] = vx_new
    vy[i] = vy_new


@njit(cache=True)
def run_mc_iteration(x, y, vx, vy, occ, U_table, bin_width, mass, box,
                      n_particles, moves_per_mc, literal_update, dt):
    for _ in range(moves_per_mc):
        i = np.random.randint(0, n_particles)
        single_particle_move(i, x, y, vx, vy, occ, U_table, bin_width,
                              mass, box, n_particles, literal_update, dt)


# ============================================================
# Initialization
# ============================================================
def initialize(box, n_particles, mass, kT, rng):
    occ = np.zeros((box, box), dtype=np.int8)
    x = np.zeros(n_particles, dtype=np.int64)
    y = np.zeros(n_particles, dtype=np.int64)
    count = 0
    while count < n_particles:
        xi = rng.integers(0, box)
        yi = rng.integers(0, box)
        if occ[xi, yi] == 0:
            occ[xi, yi] = 1
            x[count] = xi
            y[count] = yi
            count += 1

    # spec item 3: Boltzmann/Maxwell distribution P(v) ~ exp(-m v^2 / 2kT)
    # -> Gaussian with mean 0, std = sqrt(kT/m). Applied independently to
    # vx and vy (2D velocity vector).
    std = np.sqrt(kT / mass)
    vx = rng.normal(0.0, std, n_particles)
    vy = rng.normal(0.0, std, n_particles)
    return x, y, vx, vy, occ


def append_snapshot(path, mc_iter, x, y, write_header):
    # positions only (no velocities) -- one row per particle, appended to a
    # single long-format trajectory file: mc_iteration, particle_id, x, y
    df = pd.DataFrame({
        "mc_iteration": mc_iter,
        "particle_id": np.arange(len(x)),
        "x": x, "y": y,
    })
    df.to_csv(path, mode="a", header=write_header, index=False)


# ============================================================
# Main driver
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    U_table, bin_width = load_potential_table(LJ_CSV_PATH)
    print(f"Loaded potential table: {len(U_table)} bins, width={bin_width}")

    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)  # numba's np.random inside njit uses this global state

    x, y, vx, vy, occ = initialize(BOX, N_PARTICLES, MASS, KB * TEMPERATURE, rng)

    # warm up JIT compilation on a throwaway single move
    run_mc_iteration(x.copy(), y.copy(), vx.copy(), vy.copy(), occ.copy(),
                      U_table, bin_width, MASS, BOX, N_PARTICLES, 1,
                      LITERAL_POSITION_UPDATE, DT)

    print(f"Starting burn-in: {BURNIN_MC} MC iterations x {MOVES_PER_MC} moves "
          f"= {BURNIN_MC * MOVES_PER_MC:,} total moves")
    t0 = time.time()
    checkpoint = max(1, BURNIN_MC // 10)
    for mc in range(BURNIN_MC):
        run_mc_iteration(x, y, vx, vy, occ, U_table, bin_width, MASS, BOX,
                          N_PARTICLES, MOVES_PER_MC, LITERAL_POSITION_UPDATE, DT)
        if (mc + 1) % checkpoint == 0:
            print(f"  burn-in {mc + 1}/{BURNIN_MC}  elapsed {time.time() - t0:.1f}s")
    print(f"Burn-in complete in {time.time() - t0:.1f}s")

    print(f"Sampling: {N_SNAPSHOTS} snapshots, every {SNAPSHOT_EVERY_MC} MC "
          f"iteration(s) ({SNAPSHOT_EVERY_MC * MOVES_PER_MC:,} moves apart)")
    traj_path = os.path.join(OUTPUT_DIR, TRAJECTORY_FILE)
    if os.path.exists(traj_path):
        os.remove(traj_path)  # start fresh each run

    t1 = time.time()
    total_mc = BURNIN_MC
    for snap in range(N_SNAPSHOTS):
        for _ in range(SNAPSHOT_EVERY_MC):
            run_mc_iteration(x, y, vx, vy, occ, U_table, bin_width, MASS, BOX,
                              N_PARTICLES, MOVES_PER_MC, LITERAL_POSITION_UPDATE, DT)
            total_mc += 1
        append_snapshot(traj_path, total_mc, x, y, write_header=(snap == 0))
        if (snap + 1) % max(1, N_SNAPSHOTS // 10) == 0 or snap == N_SNAPSHOTS - 1:
            print(f"  snapshot {snap + 1}/{N_SNAPSHOTS} appended "
                  f"(elapsed {time.time() - t1:.1f}s)")

    print(f"Sampling phase complete in {time.time() - t1:.1f}s")
    print(f"All {N_SNAPSHOTS} snapshots written to: {traj_path}")
    print("Columns: mc_iteration, particle_id, x, y")
    return traj_path


if __name__ == "__main__":
    main()
