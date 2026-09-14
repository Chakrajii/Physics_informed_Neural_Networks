# src/monte_carlo.py

import math

import numpy as np
from numba import njit


@njit(cache=True)
def _potential_at_r(r, U_table, dr, r0, cutoff):
    """
    Linear interpolation of tabulated U(r).

    Parameters
    ----------
    r : float
        Pair distance.
    U_table : np.ndarray
        Potential values on a uniform grid.
    dr : float
        Grid spacing.
    r0 : float
        First grid point.
    cutoff : float
        Distances larger than cutoff have U=0.
    """
    if r >= cutoff:
        return 0.0

    if r <= r0:
        return U_table[0]

    x = (r - r0) / dr
    i = int(x)

    n = U_table.shape[0]

    if i >= n - 1:
        return U_table[-1]

    frac = x - float(i)

    return (1.0 - frac) * U_table[i] + frac * U_table[i + 1]


@njit(cache=True)
def _particle_energy(i, pos, U_table, dr, r0, cutoff, L, dim):
    """
    Energy of particle i due to all other particles.
    """
    e = 0.0

    xi = pos[i, 0]
    yi = pos[i, 1]

    zi = 0.0
    if dim > 2:
        zi = pos[i, 2]

    N = pos.shape[0]

    for j in range(N):
        if j == i:
            continue

        dx = xi - pos[j, 0]
        dx -= L * math.floor(dx / L + 0.5)

        dy = yi - pos[j, 1]
        dy -= L * math.floor(dy / L + 0.5)

        if dim == 2:
            r = math.sqrt(dx * dx + dy * dy)
        else:
            dz = zi - pos[j, 2]
            dz -= L * math.floor(dz / L + 0.5)
            r = math.sqrt(dx * dx + dy * dy + dz * dz)

        if r < cutoff:
            e += _potential_at_r(r, U_table, dr, r0, cutoff)

    return e


@njit(cache=True)
def _run_mc_sweeps(
    pos,
    U_table,
    dr,
    r0,
    cutoff,
    L,
    beta,
    max_disp,
    dim,
    sweeps,
):
    """
    Perform a number of MC sweeps.

    One sweep = N particle move attempts.

    Returns
    -------
    acceptance : float
        Fraction of accepted moves.
    """
    N = pos.shape[0]

    accepted = 0
    total = 0

    for _ in range(sweeps):
        for _ in range(N):
            i = int(np.random.random() * N)

            old_e = _particle_energy(
                i, pos, U_table, dr, r0, cutoff, L, dim
            )

            oldx = pos[i, 0]
            oldy = pos[i, 1]
            oldz = pos[i, 2]

            newx = oldx + (2.0 * np.random.random() - 1.0) * max_disp
            newy = oldy + (2.0 * np.random.random() - 1.0) * max_disp

            newx -= L * math.floor(newx / L)
            newy -= L * math.floor(newy / L)

            pos[i, 0] = newx
            pos[i, 1] = newy

            if dim > 2:
                newz = oldz + (2.0 * np.random.random() - 1.0) * max_disp
                newz -= L * math.floor(newz / L)
                pos[i, 2] = newz
            else:
                pos[i, 2] = 0.0

            new_e = _particle_energy(
                i, pos, U_table, dr, r0, cutoff, L, dim
            )

            delta = new_e - old_e

            accept = False

            if delta != delta:
                # NaN guard.
                accept = False
            elif delta <= 0.0:
                accept = True
            else:
                arg = beta * delta
                if arg > 700.0:
                    prob = 0.0
                else:
                    prob = math.exp(-arg)

                if np.random.random() < prob:
                    accept = True

            if accept:
                accepted += 1
            else:
                pos[i, 0] = oldx
                pos[i, 1] = oldy
                pos[i, 2] = oldz

            total += 1

    if total == 0:
        return 0.0

    return float(accepted) / float(total)


class TabulatedMC:
    """
    Metropolis Monte Carlo simulation using a tabulated pair potential U(r).
    """

    def __init__(self, r, U, cfg, positions=None, seed=None):
        """
        Parameters
        ----------
        r : np.ndarray
            Radial grid for potential.
        U : np.ndarray
            Potential values on r.
        cfg : dict
            Configuration dictionary.
        positions : np.ndarray or None
            Initial positions, shape (N, dim). If None, random positions.
        seed : int or None
            Random seed.
        """
        self.cfg = cfg

        self.dim = int(cfg["system"]["dimension"])
        self.L = float(cfg["system"]["box_length"])

        kBT = float(cfg["system"]["temperature"])
        if kBT <= 0:
            raise ValueError("system.temperature must be positive.")

        self.beta = 1.0 / kBT

        r = np.asarray(r, dtype=np.float64)
        U = np.asarray(U, dtype=np.float64)

        if len(r) != len(U):
            raise ValueError("r and U must have the same length.")

        if len(r) < 2:
            raise ValueError("Potential table must contain at least two points.")

        self.r_table = r
        self.U_table = np.ascontiguousarray(U)

        self.dr = float(np.mean(np.diff(r)))
        self.r0 = float(r[0])

        # The MC cutoff is normally the fitted potential cutoff.
        p_cfg = cfg.get("potential", {})
        table_cutoff = float(r[-1] + 0.5 * self.dr)
        self.cutoff = min(
            float(p_cfg.get("fit_r_max", table_cutoff)),
            table_cutoff,
        )

        mc_cfg = cfg.get("monte_carlo", {})
        self.max_disp = float(mc_cfg.get("max_displacement", 0.1))
        self.target_acc = float(mc_cfg.get("target_acceptance", 0.4))

        if seed is not None:
            np.random.seed(int(seed) % 2147483647)

        self.pos = self._make_positions(positions)

    def _make_positions(self, positions):
        """
        Initialize positions.
        """
        mc_cfg = self.cfg.get("monte_carlo", {})
        N_cfg = int(mc_cfg.get("n_particles", 0) or 0)

        if positions is not None:
            arr = np.asarray(positions, dtype=np.float64)

            if arr.ndim != 2:
                raise ValueError("positions must be a 2D array.")

            if arr.shape[1] < self.dim:
                raise ValueError(
                    f"positions has {arr.shape[1]} coordinates, "
                    f"but system.dimension is {self.dim}."
                )

            N = arr.shape[0]

            if N_cfg > 0 and N_cfg != N:
                print(
                    f"Warning: monte_carlo.n_particles={N_cfg} but "
                    f"provided positions have N={N}. Using N={N}."
                )

            pos = np.zeros((N, 3), dtype=np.float64)
            cols = min(3, arr.shape[1])
            pos[:, :cols] = arr[:, :cols]

        else:
            if N_cfg <= 0:
                raise ValueError(
                    "monte_carlo.n_particles must be positive when "
                    "positions are not provided."
                )

            N = N_cfg
            pos = np.zeros((N, 3), dtype=np.float64)

            for d in range(self.dim):
                pos[:, d] = np.random.random(N) * self.L

        # Wrap into periodic box.
        pos[:, :self.dim] = np.mod(pos[:, :self.dim], self.L)

        return np.ascontiguousarray(pos)

    def run_sweeps(self, sweeps):
        """
        Run MC sweeps and return acceptance.
        """
        if sweeps <= 0:
            return 1.0

        return _run_mc_sweeps(
            self.pos,
            self.U_table,
            self.dr,
            self.r0,
            self.cutoff,
            self.L,
            self.beta,
            self.max_disp,
            self.dim,
            int(sweeps),
        )

    def equilibrate(self, sweeps, adapt=True):
        """
        Equilibrate the system.

        If adapt=True, adjust max_displacement to approach target_acceptance.
        """
        sweeps = int(sweeps)

        if sweeps <= 0:
            return 1.0

        if not adapt:
            return self.run_sweeps(sweeps)

        blocks = min(10, max(1, sweeps))
        block = sweeps // blocks
        remaining = sweeps - block * blocks

        if block <= 0:
            block = 1
            blocks = sweeps
            remaining = 0

        acc = 1.0

        for _ in range(blocks):
            acc = self.run_sweeps(block)

            if acc > self.target_acc:
                self.max_disp *= 1.05
            else:
                self.max_disp *= 0.95

            # Keep displacement reasonable.
            self.max_disp = max(self.max_disp, 1.0e-4)
            self.max_disp = min(self.max_disp, 0.5 * self.L)

        if remaining > 0:
            acc = self.run_sweeps(remaining)

        return acc

    def sample_production(self, production_sweeps, sample_every_sweeps):
        """
        Run production MC and collect snapshots.

        Parameters
        ----------
        production_sweeps : int
            Total number of production sweeps.
        sample_every_sweeps : int
            Save a snapshot every this many sweeps.

        Returns
        -------
        snapshots : list of np.ndarray
            Each snapshot has shape (N, dim).
        """
        production_sweeps = int(production_sweeps)
        sample_every_sweeps = max(1, int(sample_every_sweeps))

        snapshots = []
        done = 0

        while done < production_sweeps:
            steps = min(sample_every_sweeps, production_sweeps - done)
            self.run_sweeps(steps)
            done += steps

            if done % sample_every_sweeps == 0 or done == production_sweeps:
                snapshots.append(self.pos[:, :self.dim].copy())

        return snapshots