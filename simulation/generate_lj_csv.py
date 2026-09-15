"""
Generates a placeholder Lennard-Jones potential lookup table CSV.
Format: r, U(r)  for r = 0.0, 0.1, 0.2, ..., 800.0  (matches the 800x800 box spec)

Standard 12-6 LJ form: U(r) = 4*epsilon*[(sigma/r)^12 - (sigma/r)^6]
sigma = 1.0, epsilon = 1.0 (reduced units)

Replace this file's output (lj_potential.csv) with your real data --
just keep the same two-column [r, U] format and 0.1 spacing from 0 to 800.
"""
import numpy as np
import pandas as pd

sigma = 2.0   # chosen so the repulsive wall sits well below the lattice
              # spacing of 1 -- with sigma=1 the wall sits AT the lattice
              # spacing and every adjacent pair gets hit with a huge force,
              # causing runaway velocities regardless of the update rule.
              # Your real CSV should have a lengthscale appropriate to your
              # 800-unit box; if it's similar to sigma=1 here you'll see the
              # same runaway and will want to rescale distances or damp dt.
epsilon = 1.0



r = np.round(np.arange(0.0, 800.0 + 1e-9, 0.1), 1)
U = np.empty_like(r)

# r=0 is a singularity for LJ -> cap it at a large finite repulsive value
# instead of inf, so downstream finite-difference derivatives stay well-defined.
nonzero = r > 0
U[nonzero] = 4 * epsilon * ((sigma / r[nonzero])**12 - (sigma / r[nonzero])**6)
U[~nonzero] = U[nonzero][0] * 2  # arbitrary large finite cap for r=0 bin

df = pd.DataFrame({"r": r, "U": U})
df.to_csv("lj_potential.csv", index=False)
print(df.head())
print("...")
print(df.iloc[10:15])
print("rows:", len(df), " min U:", U.min(), " at r=", r[np.argmin(U)])
