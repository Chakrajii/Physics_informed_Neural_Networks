# Equilibrium Snapshots → RDF → IBI

## Overview

This project starts with a collection of particle configurations sampled from an equilibrium simulation and attempts to reconstruct an effective pair interaction using the **Iterative Boltzmann Inversion (IBI)** method.

The complete workflow is:

```text
Equilibrium Particle Snapshots
            │
            ▼
   Validate Input Snapshots
            │
            ▼
   Apply Periodic Boundary
        Conditions
            │
            ▼
   Calculate Pair Distances
            │
            ▼
      Calculate RDF
     for Each Snapshot
            │
            ▼
   Average RDFs Over All
      Equilibrium Snapshots
            │
            ▼
       Target RDF
      g_target(r)
            │
            ▼
    Initial Potential
   U₀(r) = -kBT ln g(r)
            │
            ▼
       Run MC/MD
   using current potential
            │
            ▼
   Calculate Simulated RDF
       g_sim(r)
            │
            ▼
       Compare RDFs
 g_sim(r) vs g_target(r)
            │
            ▼
      Calculate Error
            │
            ▼
     IBI Potential Update
            │
            ▼
       Uₙ₊₁(r)
            │
            ▼
     Convergence?
        /       \
      NO         YES
      │           │
      └───────────┘
                  ▼
        Final Effective
          Potential
          U_eff(r)
                  │
                  ▼
        Independent Long
             Simulation
                  │
                  ▼
       Final RDF Validation
                  │
                  ▼
             Results
```

---

# 1. Problem Definition

Assume we are given \(M\) equilibrium snapshots.

Each snapshot contains \(N\) particles:

$$
S_k =\left\{\mathbf r_1^{(k)},\mathbf r_2^{(k)},\ldots,\mathbf r_N^{(k)}\right\}
$$

where

$$
k=1,2,\ldots,M.
$$

The goal is to determine an effective pair potential

$$
U_{\mathrm{eff}}(r)
$$

such that a simulation using this potential reproduces the structural information contained in the original equilibrium snapshots.

The central pipeline is

$$
\boxed{
\text{Snapshots}
\rightarrow
g_{\mathrm{target}}(r)
\rightarrow
U_{\mathrm{eff}}(r)
}
$$

using IBI.

---

# 2. Input Data

The project assumes that the input contains particle positions.

For example:

```text
snapshot_0001.xyz
snapshot_0002.xyz
snapshot_0003.xyz
...
snapshot_0100.xyz
```

A snapshot could look like:

```text
100
LJ equilibrium snapshot
0.153 1.827
1.442 0.923
2.731 3.152
...
```

For a 3D system:

```text
x y z
```

For a 2D system:

```text
x y
```

The simulation box dimensions must also be known:

$$
L_x,L_y,L_z
$$

or, for a square 2D box,

$$
L_x=L_y=L.
$$

---

# 3. Step 0 — Validate the Snapshots

Before calculating anything, verify:

* number of particles is consistent
* box size is known
* particle coordinates are valid
* snapshots correspond to equilibrium
* boundary conditions are known
* snapshots are sufficiently separated in simulation time

The last point is important.

If snapshots are saved after every MC step,

```text
S₁
S₂
S₃
S₄
...
```

they may be strongly correlated.

Instead, we want approximately independent equilibrium samples:

```text
S₁ ---------------- S₂ ---------------- S₃
       equilibration/sample interval
```

---

# 4. Step 1 — Calculate Pair Distances

For every snapshot, calculate the distance between particle pairs.

For particles \(i\) and \(j\),

$$
r_{ij}=|\mathbf r_i-\mathbf r_j|.
$$

For periodic systems, use the **minimum image convention**.

For example:

$$
\Delta x
=
x_i-x_j
$$

then

$$
\Delta x
\leftarrow
\Delta x
-
L_x
\operatorname{round}\left(\frac{\Delta x}{L_x}\right).
$$

Similarly for \(y\) and \(z\).

Then

$$
r_{ij}
=
\sqrt{
\Delta x^2+
\Delta y^2+
\Delta z^2
}.
$$

Only distances up to approximately

$$
r_{\max}\leq \frac{L}{2}
$$

should normally be used for a cubic box.

---

# 5. Step 2 — Construct the RDF for Each Snapshot

Divide the radial coordinate into bins:

$$
r_0,r_1,r_2,\ldots,r_{N_b}.
$$

For each pair distance, determine which radial bin it belongs to.

This produces a histogram:

$$
H_k(r).
$$

However, the raw histogram is **not yet the RDF**.

It must be normalized by the expected number of particles in the corresponding shell.

---

# 6. RDF Normalization

For a homogeneous 3D system, the volume of a spherical shell is

$$
\Delta V(r)
=
4\pi r^2\Delta r.
$$

For a system with number density

$$
\rho=\frac{N}{V},
$$

the RDF is approximately

$$
\boxed{
g(r)
=
\frac{H(r)}
{
N\rho\,4\pi r^2\Delta r
}
}
$$

with the precise prefactor depending on whether pairs are counted as \(i<j\) or \(i\neq j\).

The important physical property is:

$$
\boxed{
g(r)\rightarrow1
}
$$

at sufficiently large \(r\) for a homogeneous fluid.

---

# 7. 2D RDF

If the system is two-dimensional, the shell area is

$$
\Delta A(r)
=
2\pi r\Delta r.
$$

Therefore,

$$
\boxed{
g(r)
=
\frac{H(r)}
{
N\rho\,2\pi r\Delta r
}
}
$$

again with the appropriate pair-counting normalization.

The dimensionality of the system must therefore be specified before calculating the RDF.

---

# 8. Step 3 — Calculate \(g_k(r)\) for Every Snapshot

For each equilibrium snapshot:

```text
snapshot_1
     ↓
pair distances
     ↓
histogram
     ↓
normalization
     ↓
g₁(r)

snapshot_2
     ↓
pair distances
     ↓
histogram
     ↓
normalization
     ↓
g₂(r)

...

snapshot_M
     ↓
pair distances
     ↓
histogram
     ↓
normalization
     ↓
g_M(r)
```

At this point we have

$$
g_1(r),g_2(r),\ldots,g_M(r).
$$

---

# 9. Step 4 — Ensemble-Average the RDF

Because all snapshots represent the same equilibrium ensemble, calculate

$$
\boxed{
g_{\mathrm{target}}(r)
=
\frac{1}{M}
\sum_{k=1}^{M}g_k(r)
}
$$

This becomes the target RDF for IBI.

The important point is:

> **Average the RDFs before performing IBI, rather than performing IBI separately and averaging the potentials.**

The resulting quantity is our best estimate of the equilibrium RDF:

$$
\boxed{
\hat g(r)\approx g_{\mathrm{true}}(r)
}
$$

subject to finite-size and sampling errors.

---

# 10. Estimate RDF Uncertainty

Because we have multiple snapshots, we can also calculate the statistical variation.

At each \(r\),

$$
\sigma_g(r)
=
\sqrt{
\frac{1}{M-1}
\sum_{k=1}^{M}
\left[
g_k(r)-g_{\mathrm{target}}(r)
\right]^2
}.
$$

Thus we obtain:

```text
r       g_target(r)       uncertainty
--------------------------------------
0.1        0.00              0.00
0.2        0.02              0.01
0.3        0.31              0.04
0.4        1.82              0.12
...
```

This is useful for diagnosing whether apparent features are genuine or simply statistical fluctuations.

---

# 11. Step 5 — Prepare the Target RDF for IBI

Before applying IBI, check the target RDF.

Important conditions:

### At small \(r\)

If particles cannot overlap,

$$
g(r)\rightarrow0.
$$

### At large \(r\)

For a homogeneous fluid,

$$
g(r)\rightarrow1.
$$

### Avoid exact zeros in the logarithm

The initial Boltzmann inversion contains

$$
\ln g(r).
$$

Therefore,

$$
g(r)=0
$$

causes a numerical problem.

A small floor can be used:

$$
g(r)\rightarrow\max[g(r),g_{\min}].
$$

The treatment of the hard-core region should be handled carefully rather than blindly clipping physical structure.

---

# 12. Step 6 — Initial Guess for the Potential

The simplest starting potential is obtained using Boltzmann inversion:

$$
\boxed{
U_0(r)
=
-k_BT\ln g_{\mathrm{target}}(r)
}
$$

or in reduced units,

$$
\boxed{
\beta U_0(r)
=
-\ln g_{\mathrm{target}}(r)
}
$$

where

$$
\beta=\frac{1}{k_BT}.
$$

This is the **Potential of Mean Force**.

Important:

$$
U_0(r)\neq U_{\mathrm{true}}(r)
$$

in general.

It is an initial guess for IBI.

---

# 13. Step 7 — Represent the Potential Numerically

Instead of assuming an analytical form, store the potential as a lookup table:

```text
r       U(r)
----------------
0.10    12.34
0.11    11.52
0.12    10.81
...
4.00    -0.01
```

Thus,

$$
U(r_i)=U_i.
$$

During the simulation, interpolation is used to obtain \(U(r)\) between tabulated points.

This makes the method flexible.

We do **not** need to assume that the interaction is Lennard-Jones.

---

# 14. Step 8 — Run a Monte Carlo Simulation

Now use \(U_0(r)\) in a fresh Monte Carlo simulation.

For a particle:

$$
\mathbf r_i\rightarrow\mathbf r_i'
$$

is proposed.

Calculate the energy difference:

$$
\Delta E
=
E_{\mathrm{new}}-E_{\mathrm{old}}.
$$

Accept the move if

$$
\Delta E\leq0.
$$

Otherwise accept it with probability

$$
\boxed{
P_{\mathrm{accept}}
=
e^{-\beta\Delta E}
}
$$

using a random number \(q\in[0,1]\).

Accept if

$$
q<P_{\mathrm{accept}}.
$$

---

# 15. Step 9 — Equilibrate the IBI Simulation

The newly started simulation must itself reach equilibrium.

Therefore:

```text
Initial configuration
        ↓
MC equilibration
        ↓
Equilibrium
        ↓
Start collecting RDF
```

Do not calculate the IBI RDF primarily from the initial transient.

After equilibration, collect configurations separated sufficiently in MC time.

---

# 16. Step 10 — Calculate the Simulated RDF

Using the equilibrated MC configurations, calculate

$$
\boxed{
g_n(r)
}
$$

where \(n\) represents the current IBI iteration.

Now we have:

$$
g_{\mathrm{target}}(r)
$$

and

$$
g_n(r).
$$

Compare them.

---

# 17. Step 11 — Calculate the RDF Error

A simple error metric is

$$
\boxed{
\chi^2_n
=
\sum_i
\left[
g_n(r_i)-g_{\mathrm{target}}(r_i)
\right]^2
}
$$

A statistically weighted version is

$$
\boxed{
\chi^2_n
=
\sum_i
\frac{
[g_n(r_i)-g_{\mathrm{target}}(r_i)]^2
}{
\sigma_g^2(r_i)
}
}
$$

This allows us to quantify convergence.

---

# 18. Step 12 — IBI Update

The fundamental IBI update is

$$
\boxed{
U_{n+1}(r)
=
U_n(r)
+
\alpha k_BT
\ln
\left[
\frac{g_n(r)}
{g_{\mathrm{target}}(r)}
\right]
}
$$

where

$$
0<\alpha\leq1
$$

is the damping parameter.

Typical values might be something like

$$
\alpha=0.1-0.5.
$$

A smaller \(\alpha\) generally gives slower but more stable updates.

---

# 19. Understanding the Update

Suppose at some \(r\),

$$
g_n(r)>g_{\mathrm{target}}(r).
$$

There are too many particle pairs at that separation.

Then

$$
\ln
\frac{g_n(r)}
{g_{\mathrm{target}}(r)}>0
$$

so

$$
U_{n+1}(r)>U_n(r).
$$

The interaction becomes more repulsive.

This should reduce the number of particles found at that separation.

Conversely, if

$$
g_n(r)<g_{\mathrm{target}}(r),
$$

then

$$
U_{n+1}(r)<U_n(r),
$$

making that separation relatively more favorable.

Therefore IBI is essentially a feedback loop:

```text
Too many particles at r
        ↓
Increase U(r)
        ↓
Fewer particles at r

Too few particles at r
        ↓
Decrease U(r)
        ↓
More particles at r
```

---

# 20. Step 13 — Repeat

After updating the potential:

$$
U_n(r)
\rightarrow
U_{n+1}(r)
$$

run another MC simulation.

Then:

$$
U_{n+1}(r)
\rightarrow
g_{n+1}(r).
$$

Repeat:

```text
U₀
 ↓
MC
 ↓
g₀
 ↓
IBI update
 ↓
U₁
 ↓
MC
 ↓
g₁
 ↓
IBI update
 ↓
U₂
 ↓
...
```

---

# 21. Step 14 — Convergence

The IBI loop is stopped when the simulated RDF is sufficiently close to the target RDF.

For example,

$$
\mathrm{RMSD}
=
\sqrt{
\frac{1}{N_b}
\sum_i
[g_n(r_i)-g_{\mathrm{target}}(r_i)]^2
}
$$

becomes smaller than a chosen tolerance.

For example:

$$
\boxed{
\mathrm{RMSD}<10^{-2}
}
$$

could be used as one possible criterion, although the appropriate threshold depends on your system and statistical noise.

The convergence condition should not be chosen blindly.

---

# 22. Complete IBI Loop

The core algorithm is therefore:

```text
              ┌──────────────────────┐
              │   Target RDF         │
              │ g_target(r)          │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Initial potential    │
              │ U₀=-kBT ln(g_target)│
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │      MC Simulation   │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Calculate g_n(r)     │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Compare with target  │
              └──────────┬───────────┘
                         │
                    Converged?
                    /          \
                  YES           NO
                   │             │
                   │             ▼
                   │    ┌────────────────────┐
                   │    │ IBI update         │
                   │    │                    │
                   │    │ Uₙ₊₁ = Uₙ +       │
                   │    │ αkBT ln(gₙ/g_tgt) │
                   │    └─────────┬──────────┘
                   │              │
                   │              └───────┐
                   │                      │
                   │                      ▼
                   │                 New MC run
                   │
                   ▼
          ┌─────────────────┐
          │ Final U_eff(r)  │
          └─────────────────┘
```

---

# 23. Step 15 — Independent Validation

Do **not** consider the potential validated merely because the final IBI simulation matches the target RDF.

Take the final potential

$$
U_{\mathrm{eff}}(r)
$$

and perform a **fresh, independent, long simulation**.

Calculate:

$$
g_{\mathrm{validation}}(r).
$$

Then compare:

$$
\boxed{
g_{\mathrm{validation}}(r)
\approx
g_{\mathrm{target}}(r)
}
$$

If they agree within statistical uncertainty, the inferred potential successfully reproduces the target structure.

---

# 24. What the Final Result Means

The final result is:

$$
\boxed{
U_{\mathrm{eff}}(r)
}
$$

such that

$$
\boxed{
g_{U_{\mathrm{eff}}}(r)
\approx
g_{\mathrm{target}}(r).
}
$$

It is important to call this an **effective potential**.

It is not automatically guaranteed to be the microscopic potential that generated the original snapshots.

This distinction becomes particularly important for systems containing many-body interactions.

---

# 25. Full Project Pipeline

The entire project can now be summarized as:

```text
                    INPUT
                      │
                      ▼
       ┌─────────────────────────────┐
       │ Equilibrium particle       │
       │ snapshots S₁...S_M         │
       └──────────────┬──────────────┘
                      │
                      ▼
       ┌─────────────────────────────┐
       │ Validate coordinates,       │
       │ box, dimensionality, PBC    │
       └──────────────┬──────────────┘
                      │
                      ▼
       ┌─────────────────────────────┐
       │ Pair-distance calculation   │
       │ + minimum image convention  │
       └──────────────┬──────────────┘
                      │
                      ▼
       ┌─────────────────────────────┐
       │ RDF for each snapshot       │
       │ g₁(r), g₂(r), ..., g_M(r)  │
       └──────────────┬──────────────┘
                      │
                      ▼
       ┌─────────────────────────────┐
       │ Ensemble average            │
       │                             │
       │ g_target(r) = <g_k(r)>     │
       └──────────────┬──────────────┘
                      │
                      ▼
       ┌─────────────────────────────┐
       │ Estimate RDF uncertainty    │
       └──────────────┬──────────────┘
                      │
                      ▼
       ┌─────────────────────────────┐
       │ Boltzmann inversion         │
       │                             │
       │ U₀(r)=-kBT ln g_target(r)  │
       └──────────────┬──────────────┘
                      │
                      ▼
       ┌─────────────────────────────┐
       │ Run MC simulation           │
       │ using Uₙ(r)                 │
       └──────────────┬──────────────┘
                      │
                      ▼
       ┌─────────────────────────────┐
       │ Calculate simulated RDF     │
       │ gₙ(r)                       │
       └──────────────┬──────────────┘
                      │
                      ▼
       ┌─────────────────────────────┐
       │ Calculate RDF error         │
       └──────────────┬──────────────┘
                      │
                      ▼
                 Converged?
                  /       \
                NO         YES
                │           │
                ▼           ▼
       ┌────────────────┐   │
       │ IBI update     │   │
       │                │   │
       │ Uₙ₊₁ = Uₙ +    │   │
       │ αkBT ln(gₙ/g_t)│   │
       └───────┬────────┘   │
               │            │
               └───► MC ◄───┘
                            │
                            ▼
                 ┌──────────────────┐
                 │ U_eff(r)         │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Independent      │
                 │ validation MC    │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Final RDF        │
                 │ comparison       │
                 └────────┬─────────┘
                          │
                          ▼
                       RESULTS
```

---

# 26. Recommended Project Structure

A clean implementation could look like:

```text
RDF_IBI/
│
├── data/
│   ├── snapshots/
│   │   ├── snapshot_001.xyz
│   │   ├── snapshot_002.xyz
│   │   └── ...
│   │
│   └── target/
│       └── rdf_target.csv
│
├── src/
│   ├── io.py
│   ├── distances.py
│   ├── rdf.py
│   ├── potential.py
│   ├── monte_carlo.py
│   ├── ibi.py
│   └── validation.py
│
├── results/
│   ├── rdf/
│   ├── potentials/
│   ├── convergence/
│   └── validation/
│
├── configs/
│   └── parameters.yaml
│
├── run_rdf.py
├── run_ibi.py
├── run_validation.py
│
└── README.md
```

---

# 27. Important Parameters

A configuration file could contain:

```yaml
system:
    dimension: 2
    N: 100
    box_length: 10.0
    temperature: 1.0

rdf:
    r_min: 0.0
    r_max: 5.0
    dr: 0.02

ibi:
    alpha: 0.2
    max_iterations: 100
    tolerance: 0.01

monte_carlo:
    equilibration_steps: 100000
    production_steps: 500000
    max_displacement: 0.2
```

These values are illustrative; they should be chosen based on the actual system.

---

# 28. Files Produced by the Pipeline

The project can generate:

### Target RDF

```text
target_rdf.csv
```

containing

```text
r,g_target,error
```

### Initial potential

```text
potential_000.csv
```

### Potential after each IBI iteration

```text
potential_001.csv
potential_002.csv
...
```

### Simulated RDF

```text
rdf_000.csv
rdf_001.csv
...
```

### Final potential

```text
effective_potential.csv
```

### Validation RDF

```text
validation_rdf.csv
```

---

# 29. Final Scientific Comparison

The most important final plot should be:

```text
        g(r)
         │
         │       target
         │      /\
         │     /  \      /\
         │    /    \____/  \____
         │   /               ...
         │  /  validation
         │ /  /\
         │/  /  \____/\________
         └──────────────────────── r
```

The target RDF and validation RDF should overlap within their statistical uncertainties.

Another important plot is:

$$
U_{\mathrm{eff}}(r)
$$

versus \(r\).

Finally, plot the convergence:

$$
\mathrm{RMSD}_n
$$

versus IBI iteration:

```text
RMSD
 │\
 │ \
 │  \
 │   \
 │    \____
 │         \____
 └──────────────── iteration
```

---

# 30. The Core Idea in One Equation Chain

The entire project can ultimately be reduced to:

$$
\boxed{
\{S_1,S_2,\ldots,S_M\}
\rightarrow
\{g_1,g_2,\ldots,g_M\}
\rightarrow
g_{\mathrm{target}}
\rightarrow
U_0
\rightarrow
\mathrm{MC}
\rightarrow
g_0
\rightarrow
U_1
\rightarrow
\mathrm{MC}
\rightarrow\cdots
\rightarrow
U_{\mathrm{eff}}
}
$$

followed by

$$
\boxed{
U_{\mathrm{eff}}
\rightarrow
\text{independent MC}
\rightarrow
g_{\mathrm{validation}}
\approx
g_{\mathrm{target}}.
}
$$

---

# 31. What We Should Implement First

Do **not** start by coding IBI.

Build the project in stages:

### Stage 1 — RDF

```text
Snapshots
   ↓
Pair distances
   ↓
RDF
   ↓
target_rdf.csv
```

First verify that your RDF calculation works.

### Stage 2 — Monte Carlo

Implement a simple MC simulation with a **known potential**, e.g. Lennard-Jones.

Verify that MC produces a sensible RDF.

### Stage 3 — Boltzmann inversion

Take the target RDF and construct

$$
U_0(r)=-k_BT\ln g_{\rm target}(r).
$$

### Stage 4 — IBI

Implement

$$
U_{n+1}
=
U_n+
\alpha k_BT
\ln\left(
\frac{g_n}{g_{\rm target}}
\right).
$$

### Stage 5 — Validation

Compare the final RDF with the original target RDF.

### Stage 6 — Error analysis

Study:

* number of snapshots
* number of particles
* snapshot correlation
* bin width
* MC length
* IBI damping \(\alpha\)
* finite-size effects

This turns the project from simply **"I wrote an IBI code"** into a proper computational statistical-mechanics study.
