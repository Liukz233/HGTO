# Reproducing the paper

Run commands from the repository root after installation. Configurations are
complete experiment recipes; command-line overrides are recorded with each run.
All timed experiments should run serially. The paper uses two CPU threads,
an RTX 6000 Ada, and a Threadripper PRO 5995WX. Figure and table numbers
below follow the final manuscript.

## Experiment map

| Paper | Configuration under `configs/` | Target volume |
|---|---|---:|
| Fig. 3: cantilever | `linear2d/cantilever_120x40.yaml` | 0.50 |
| Fig. 3: half-MBB | `linear2d/half_mbb.yaml` | 0.50 |
| Fig. 3: inclined load | `linear2d/inclined_cantilever.yaml` | 0.50 |
| Fig. 4, Table B.1: intermediate cantilever | `linear2d/cantilever_240x80.yaml` | 0.50 |
| Fig. 4, Table B.1: fine cantilever | `linear2d/cantilever_480x160.yaml` | 0.50 |
| Fig. 4, Table B.1: largest cantilever | `linear2d/cantilever_960x320.yaml` | 0.50 |
| Fig. 5a: L-bracket | `domains/l_bracket.yaml` | 0.40 |
| Fig. 5b: perforated bracket | `domains/perforated_bracket.yaml` | 0.40 |
| Fig. 5c: beam with reinforced openings | `domains/ring_beam.yaml` | 0.40 |
| Fig. 6: cantilever | `linear3d/cantilever.yaml` | 0.30 |
| Fig. 6: four-foot support | `linear3d/four_foot_support.yaml` | 0.18 |
| Fig. 6: torsion | `linear3d/torsion.yaml` | 0.20 |
| Fig. 7: cantilever, $P_0=0.00125$ | `nonlinear/cantilever_weak.yaml` | 0.45 |
| Fig. 7: cantilever, $10P_0$ | `nonlinear/cantilever_strong.yaml` | 0.45 |
| Fig. 8: doubly fixed bridge | `nonlinear/bridge.yaml` | 0.40 |
| Figs. 9, 10: L-bracket, $P=0.04$ | `nonlinear/l_bracket_p0.04.yaml` | 0.40 |
| Table D.2: L-bracket, $P=0.02$ | `nonlinear/l_bracket_p0.02.yaml` | 0.40 |
| Table D.2: L-bracket, $P=0.03$ | `nonlinear/l_bracket_p0.03.yaml` | 0.40 |

Each configuration runs with:

```bash
hgto run --config configs/domains/perforated_bracket.yaml --output runs/perforated/hgto
hgto run --config configs/domains/perforated_bracket.yaml --method oc --output runs/perforated/oc
```

SIMP--OC configurations select the PARDISO solver (`pypardiso-spd` for linear
states, `pypardiso` for nonlinear tangents); install `.[fast-cpu]`. HGTO's
CPU option uses the independent SciPy/scikit-fem solver. This is useful for
portability, but does not reproduce GPU timings. For a SciPy-only OC run, copy
a configuration and set `oc.solver: scipy` (linear) or
`oc.tangent_backend: scipy` (nonlinear).

The saved meshes in `meshes/` are the meshes used in the paper. Gmsh is not
needed to run them. The largest cantilever has 307,200 design elements. Its
six-cell filter radius allows finer physical features as resolution increases.

To run a group sequentially:

```bash
python scripts/reproduce.py --suite linear2d --method hgto --output runs/paper --dry-run
python scripts/reproduce.py --suite linear2d --method hgto --output runs/paper
```

Available suites are `linear2d`, `domains`, `linear3d`, `nonlinear`, and `all`.

## Reference metrics

Objective and total time $t$ as reported in the paper (Tables 4, 5 and B.1).
Full-precision values, volumes and termination records are in
[`benchmarks/reference/`](../benchmarks/reference). $\dagger$ update limit
reached.

| Case | SIMP--OC $C$ | $t$ (s) | NTopo $C$ | $t$ (s) | HGTO $C$ | $t$ (s) |
|---|---:|---:|---:|---:|---:|---:|
| Cantilever $120\times40$ | 175.08 | 9.0 | 177.93 | 862.2 | 176.49 | 8.3 |
| Half-MBB | 191.49 | 9.0 | 194.79 | 942.5 | 195.47 | 7.4 |
| Inclined load | 132.72 | 26.9$^\dagger$ | 133.81 | 906.6 | 133.48 | 8.5 |
| Cantilever $240\times80$ | 175.77 | 30.9 | 178.03 | 862.1 | 176.85 | 11.7 |
| Cantilever $480\times160$ | 171.58 | 197.5 | 178.45 | 863.7 | 172.44 | 47.4 |
| Cantilever $960\times320$ | 168.56 | 1880.1 | 178.92 | 867.6 | 171.00 | 195.7 |
| L-bracket | 96.06 | 9.7 | 99.43 | 952.3 | 96.95 | 10.3 |
| Perforated bracket | 49.32 | 6.3 | 49.78 | 1001.1 | 48.71 | 14.2 |
| Beam with reinforced openings | 19.86 | 27.7 | 20.86 | 1150.3 | 20.23 | 4.9 |
| 3D cantilever | 15.07 | 188.3 | 16.17 | 3886.0 | 15.23 | 24.3 |
| Four-foot support | 0.5374 | 377.2 | 0.5810 | 3781.3 | 0.5347 | 39.3 |
| Torsion ($C\times10^3$) | 8.135 | 337.0 | 10.148 | 3564.1 | 8.144 | 48.1 |

Finite-deformation problems (Table 5 and Table D.2). $J$ is twice the
complementary work; $N$ design updates; $n_\mathrm{N}$ Newton iterations of
all completed state solves; FB updates whose continued solve fell back to
incremental loading; Rej. rejected trial steps.

| Case | Method | $J$ | $t$ (s) | $N$ | $n_\mathrm{N}$ | FB | Rej. |
|---|---|---:|---:|---:|---:|---:|---:|
| Cantilever $P_0$ | SIMP--OC | 0.000649 | 63.7 | 562 | 1334 | 0 | 0 |
| | NTopo | 0.000660 | 859.7 | | | | |
| | HGTO | 0.000651 | 14.6 | 322 | 704 | 0 | 0 |
| Cantilever $10P_0$ | SIMP--OC | 0.06374 | 283.2 | 403 | 3849 | 36 | 6 |
| | NTopo | training failed | | | | | |
| | HGTO | 0.06557 | 41.8 | 324 | 2112 | 14 | 0 |
| Bridge | SIMP--OC | 0.10352 | 35.6 | 257 | 589 | 0 | 0 |
| | NTopo | training failed | | | | | |
| | HGTO | 0.10341 | 13.3 | 285 | 631 | 0 | 0 |
| L-bracket $P=0.02$ | SIMP--OC | 0.03622 | 43.5 | 265 | 768 | 0 | 0 |
| | HGTO | 0.03642 | 14.6 | 269 | 648 | 0 | 0 |
| L-bracket $P=0.03$ | SIMP--OC | 0.08085 | 319.2 | 299 | 2890 | 23 | 2 |
| | HGTO | 0.08149 | 21.5 | 306 | 978 | 2 | 0 |
| L-bracket $P=0.04$ | SIMP--OC | 0.14334 | 1757.1 | 284 | 5786 | 54 | 89 |
| | NTopo | 0.18736 | 1141.5 | | | | |
| | HGTO | 0.14355 | 38.6 | 338 | 1323 | 6 | 1 |

These are historical measurements, not a promise of identical runtime on
other hardware. SIMP--OC in the finite-deformation problems is sensitive to
round-off in the sparse factorization: repeated $P_0$, $10P_0$ and L-bracket
runs end after different update counts (439--582, 403--569 and 284--384), with
objectives within 0.4%. Repeated HGTO runs reproduce the update counts,
designs and objectives to round-off.

## Beam with reinforced openings

The mesh in `meshes/ring_beam/` has 17,043 linear triangles and 8,751 nodes.
`fixed_density` in `geometry.npz` holds the physical density 1 for the 1,868
ring triangles and NaN for design cells. Both methods keep the rings at full
density and count them toward $V_f = 0.40$; the remaining cells carry a volume
fraction of 0.357. HGTO assembles the triangle stiffness on the GPU and
factorizes it with cuDSS at every update (requires `.[gpu-sparse]`); SIMP--OC
uses PARDISO on the CPU. The final designs of both methods are evaluated with
scikit-fem linear triangles.

## Nonlinear mechanics

```bash
hgto run --config configs/nonlinear/cantilever_strong.yaml --output runs/strong/hgto
hgto run --config configs/nonlinear/cantilever_strong.yaml --method oc --output runs/strong/oc
hgto run --config configs/nonlinear/l_bracket_p0.04.yaml --output runs/l_bracket/hgto
```

HGTO uses GPU density learning and GPU cuDSS mechanics. SIMP--OC uses the CPU
with two threads and the PARDISO sparse direct solver for the tangent. To run
HGTO mechanics on the CPU while retaining the GPU density network:

```bash
hgto run --config configs/nonlinear/bridge.yaml --state-device cpu --output runs/bridge/hgto_cpu_mechanics
```

`--device cpu` runs both density learning and mechanics on the CPU, unless
`--state-device` is explicitly set. `--device cuda:0 --method oc` runs the
SIMP--OC mechanics on the GPU with cuDSS instead of the paper's CPU setting.
cuDSS is optional for CPU runs and mandatory for CUDA nonlinear states; there
is no silent CPU fallback.

Both methods share the settings of Appendix C, recorded in each configuration:

- Compressible Neo-Hookean material with the energy interpolation of Wang et al.
  ($\beta_0=500$, $\eta_0=0.01$), filter radius 1, density floor 0.001.
- $p$ rises from 1 to 3 by update 125. The projection sharpness rises
  geometrically as $8^{k/144}$ and continues at the same rate to $\beta=32$ at
  update 240 (`beta_final: 32`, `beta_ramp_base: 8`).
- Twelve load increments. A failed increment is bisected from its last
  converged state (`adaptive_load`), with at most 40 Newton iterations per
  increment, six subdivisions, four failed attempts and 32 extra increments
  (`case.nh_solver`).
- State continuation (`state_continuation: true`): each analysis starts Newton
  at the full load from the equilibrium of the previous accepted design and
  returns to incremental loading from the undeformed state only if Newton
  fails. For HGTO this is the evolution of its physics field. All final
  designs are re-analyzed from the undeformed state.
- HGTO: Adam, learning rate from 0.01 to 0.001 over 180 updates, then
  exponential decay with half-life 60 and floor $10^{-5}$; a candidate whose
  equilibrium cannot be computed is retried with half the step, up to ten
  times. The bridge averages the logits with their spanwise reflection.
- SIMP--OC: move limit $0.2/\beta$; at unchanged parameters a step that raises
  the objective by more than a relative $10^{-8}$ is halved, up to ten times.
  In the final stage, the move limit of an element is halved when its update
  changes sign and otherwise grows by 20%, up to $0.2/\beta$ (`adaptive_move`).
- Stopping: at the final $(p, \beta)$, relative objective range below
  $10^{-3}$ over ten checks and maximum density change below $5\times10^{-3}$,
  both for five consecutive checks, after at least ten fixed-parameter
  updates; at most 1,000 updates.

The Neo-Hookean objective is twice the complementary work,
`J = 2 * (f.T @ u - U)`.

Timing and counters: HGTO's $t$ is `wall_s` in `result.json` (construction,
optimization and the common final evaluation). SIMP--OC's $t$ is `wall_s` in
`record.json`, whose final check is already a re-analysis from the undeformed
state; its `result.json` adds the common evaluation. $n_\mathrm{N}$ is the sum
of the `newton` column of `history.csv` for HGTO and
`total_completed_newton_iterations` in `record.json` for SIMP--OC. FB counts
rows after the first whose `state_start` is `undeformed`; Rej. is the length of
`candidate_rejections.json`.

## Stopping and reference values

Linear HGTO uses two first-degree Chebyshev layers, 64 Fourier frequencies
and width 64 in 2D, and 128 frequencies and width 128 in 3D. The trainable
parameter counts are 16,577 and 65,921. Uniform initialization uses seed 0.
Continuation is `(p, beta) = (1,1), (2,2), (3,4), (3,8)`, with 50 updates in
each initial stage. The final stage ends when the physical stopping condition
is satisfied or the total 1,000-update limit is reached.

At fixed final parameters, the stopping rule requires an objective range
below `1e-3` over ten checks and maximum density change below `5e-3` for five
consecutive checks, alongside volume and equilibrium tolerances. Outputs
separate `converged`, an update limit, and nonlinear OC stagnation.
The inclined-load linear OC endpoint in the paper is not a converged control.
NTopo uses its native fixed training budget.

NTopo's high-resolution rows sample one trained field and retain its shared
training cost; they are not independently trained fine-grid models.

A complete HGTO run records case construction, network/solver setup,
optimization, and final evaluation. NTopo additionally includes process startup
and output. Offline plots and trajectory evaluation are excluded. The renamed
release preserves the numerical algorithms; source hashes necessarily differ
from the original run snapshots because packaging and identifiers changed.
