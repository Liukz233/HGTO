# Reproducing the paper

Run commands from the repository root after installation. Configurations are
complete experiment recipes; command-line overrides are recorded with each run.
All timed experiments should run serially. The paper uses two CPU threads,
an RTX 6000 Ada, and a Threadripper PRO 5995WX.

## Experiment map

| Paper | Configuration under `configs/` | Target volume |
|---|---|---:|
| Fig. 3: cantilever | `linear2d/cantilever_120x40.yaml` | 0.50 |
| Fig. 3: half-MBB | `linear2d/half_mbb.yaml` | 0.50 |
| Fig. 3: inclined load | `linear2d/inclined_cantilever.yaml` | 0.50 |
| Fig. 4/S1: intermediate cantilever | `linear2d/cantilever_240x80.yaml` | 0.50 |
| Fig. 4: fine cantilever | `linear2d/cantilever_480x160.yaml` | 0.50 |
| Fig. 4: largest cantilever | `linear2d/cantilever_960x320.yaml` | 0.50 |
| Fig. 5: L bracket | `domains/l_bracket.yaml` | 0.40 |
| Fig. 5: perforated bracket | `domains/perforated_bracket.yaml` | 0.40 |
| Fig. 6: cantilever | `linear3d/cantilever.yaml` | 0.30 |
| Fig. 6: four-foot support | `linear3d/four_foot_support.yaml` | 0.18 |
| Fig. 6: torsion | `linear3d/torsion.yaml` | 0.20 |
| Fig. 7: weak load | `nonlinear/cantilever_weak.yaml` | 0.45 |
| Fig. 7: strong load | `nonlinear/cantilever_strong.yaml` | 0.45 |
| Fig. 8: doubly fixed bridge | `nonlinear/bridge.yaml` | 0.40 |
| Fig. 9: elastic feedback | `nonlinear/connection_elastic.yaml` | 0.40 |
| Fig. 9: plastic feedback | `nonlinear/connection_plastic.yaml` | 0.40 |

Each configuration runs with:

```bash
hgto run --config configs/domains/perforated_bracket.yaml --output runs/perforated/hgto
hgto run --config configs/domains/perforated_bracket.yaml --method oc --output runs/perforated/oc
```

Linear OC configurations explicitly select `pypardiso-spd`; install `.[fast-cpu]`.
HGTO's CPU option uses the independent SciPy/scikit-fem solver. This is useful
for portability, but does not reproduce GPU timings. For a SciPy-only linear
OC run, copy a configuration and set `oc.solver: scipy`.

The saved meshes in `meshes/` are the meshes used in the paper. Gmsh is not
needed to run them. The largest cantilever has 307,200 design elements. Its
six-cell filter radius allows finer physical features as resolution increases.

To run a group sequentially:

```bash
python scripts/reproduce.py --suite linear2d --method hgto --output runs/paper --dry-run
python scripts/reproduce.py --suite linear2d --method hgto --output runs/paper
```

Available suites are `linear2d`, `domains`, `linear3d`, `nonlinear`, and `all`.
The plastic configurations contain only HGTO comparisons, so they are omitted
when selecting OC.

## Nonlinear mechanics

```bash
hgto run --config configs/nonlinear/cantilever_strong.yaml --output runs/strong/hgto
hgto run --config configs/nonlinear/cantilever_strong.yaml --method oc --output runs/strong/oc
hgto run --config configs/nonlinear/bridge.yaml --output runs/bridge/hgto
```

HGTO uses GPU density learning and GPU cuDSS mechanics by default. To reproduce
the additional CPU-mechanics comparison while retaining the GPU density network:

```bash
hgto run --config configs/nonlinear/bridge.yaml --state-device cpu --output runs/bridge/hgto_cpu_mechanics
```

`--device cpu` runs both density learning and mechanics on the CPU, unless
`--state-device` is explicitly set. Nonlinear OC uses CPU SciPy mechanics.
cuDSS is optional for CPU runs and mandatory for CUDA nonlinear states;
there is no silent CPU fallback.

The Neo-Hookean objective is twice complementary work,
`J = 2 * (f.T @ u - U)`. The plastic objective is peak-load external work.
Both use incremental equilibrium and the paper's material interpolation.
The bridge imposes reflection symmetry on HGTO density outputs.

The connection's unloading comparison is a separate evaluation:

```bash
hgto run --config configs/nonlinear/connection_elastic.yaml --output runs/connection/elastic
hgto run --config configs/nonlinear/connection_plastic.yaml --output runs/connection/plastic
hgto compare-plastic --elastic runs/connection/elastic --plastic runs/connection/plastic --output runs/connection/response
```

Residual displacement is an evaluation metric, not the optimized objective.
The paper compares these two HGTO designs, not a plastic OC/NTopo baseline.

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
The inclined-load linear OC and strong-load nonlinear OC endpoints in the
paper are not converged controls. NTopo uses its native fixed training budget.

Recorded paper metrics are in [`benchmarks/reference/`](../benchmarks/reference).
They are historical measurements, not a promise of identical runtime on other
hardware. NTopo's high-resolution rows sample one trained field and retain its
shared training cost; they are not independently trained fine-grid models.

A complete HGTO run records case construction, network/solver setup,
optimization, and final evaluation. NTopo additionally includes process startup
and output. Offline plots and trajectory evaluation are excluded. The renamed
release preserves the numerical algorithms; source hashes necessarily differ
from the original run snapshots because packaging and identifiers changed.
