# HGTO

**A Unified Graph-Based Physics-Informed Formulation for Structural Topology Optimization**

Kangzheng Liu · Uday Kumar Punna · Leixin Ma

**Paper:** [arXiv:2609.15001](https://arxiv.org/abs/2609.15001)

HGTO couples a graph neural network for material density with finite-element-consistent hypergraph mechanics. Each design is optimized from a uniform density field without labeled topology data. The same formulation supports planar and spatial elasticity, irregular domains on quadrilateral and triangular meshes, and finite deformation.

![HGTO topology optimization: cantilever, half-MBB beam, and inclined load](docs/assets/linear_designs.png)

## Installation

Use Python 3.11 or newer. Install the appropriate [PyTorch distribution](https://pytorch.org/get-started/locally/) for your CPU or CUDA environment, then:

```bash
git clone https://github.com/Liukz233/HGTO.git
cd HGTO
python -m pip install -e ".[test,plot]"
```

Optional components:

| Extra | Purpose |
|---|---|
| `fast-cpu` | PARDISO solver used by the paper's SIMP--OC configurations, linear and nonlinear |
| `gpu-sparse` | cuDSS 0.8 for GPU nonlinear mechanics and the triangular mesh; requires CUDA 12 |
| `mesh` | Regenerate the perforated-bracket mesh with Gmsh; all paper meshes are included |
| `dev` | Tests and source formatting checks |

For the GPU nonlinear examples and all paper baselines:

```bash
python -m pip install -e ".[fast-cpu,gpu-sparse]"
```

NTopo runs in a separate TensorFlow environment; see [baseline setup](docs/baselines.md).

## Quick start

Run a small CPU example, plot its density, and independently evaluate it:

```bash
hgto run --config configs/smoke/cantilever_cpu.yaml --output runs/demo
hgto plot runs/demo --output runs/demo/topology.png
hgto evaluate runs/demo
```

This four-update example checks installation; it is not a converged paper result. Output directories must be new. `python -m hgto` provides the same commands as `hgto`.

Run the paper's standard cantilever:

```bash
hgto run --config configs/linear2d/cantilever_120x40.yaml --output runs/cantilever/hgto
hgto run --config configs/linear2d/cantilever_120x40.yaml --method oc --output runs/cantilever/oc
```

HGTO paper configurations use `cuda:0`; SIMP--OC runs on the CPU. Add `--device cpu` for CPU execution of HGTO. To inspect a configuration without computing or writing results, add `--dry-run`.

## Paper experiments

[Reproduction instructions](docs/reproduction.md) map every reported experiment to its configuration, baseline, and reference metrics.

| Study | Configurations |
|---|---|
| Standard beams and high resolution | [`configs/linear2d`](configs/linear2d) |
| L-bracket, perforated bracket, beam with reinforced openings | [`configs/domains`](configs/domains) |
| 3D cantilever, four-foot support, torsion | [`configs/linear3d`](configs/linear3d) |
| Finite-deformation cantilever ($P_0$, $10P_0$), bridge, L-bracket | [`configs/nonlinear`](configs/nonlinear) |

The repository also contains a small-strain J2 plasticity model with two example configurations in [`configs/additional`](configs/additional); it is not part of the paper's experiments.

![Finite-deformation cantilever designs under weak and strong loading](docs/assets/large_deformation.png)

Finite-deformation cantilever under $P_0 = 0.00125$ and $10P_0$ (paper Fig. 7). (a, b) SIMP–OC, NTopo and HGTO designs; (c, d) HGTO loaded shapes at the actual displacement scale over the undeformed design in gray; (e) change of the secant stiffness along the load path.

## Repository layout

```text
src/hgto/
  cli.py                 Public commands
  linear.py              Linear design and independent evaluation
  optimization.py        Graph density optimization with Adam
  topopt/                Graph layers, features, density filtering and projection
  fem/                   Mesh operators, constitutive laws and equilibrium solvers
  reference/             Independent elasticity and SIMP--OC
  linear2d/, linear3d/    Planar and spatial problem/physics utilities
  domains/, case_studies/ Irregular meshes and paper problem definitions
  nonlinear/             Incremental mechanics, design, and response evaluation
  baselines/ntopo/       Isolated NTopo adapter
configs/                 Named, reproducible experiment settings
meshes/                  The three paper irregular-domain meshes
scripts/                 Batch reproduction and NTopo launchers
benchmarks/reference/    Recorded paper metrics, not outputs of a fresh run
tests/                   Mechanics, gradients, constraints, solvers and CLI checks
third_party/ntopo/       Pinned upstream source and its original MIT license
```

See [architecture](docs/architecture.md), [output files](docs/outputs.md), and [numerical settings](docs/reproduction.md). Generated histories, checkpoints, plots, and local experiment logs belong under the ignored `runs/` directory.

## Validation

```bash
python -m pytest
```

The default suite uses small problems. CUDA/cuDSS and PARDISO checks run when those optional components are available. Tests cover physical-volume gradients, independent state/sensitivity agreement, triangular elements and passive regions, continuation and stopping, nonlinear adjoints and state continuation, and the installed command-line workflow.

## Citation and license

Please cite the accompanying manuscript:

> Kangzheng Liu, Uday Kumar Punna, and Leixin Ma. *HGTO: A Unified Graph-Based Physics-Informed Formulation for Structural Topology Optimization*. arXiv:2609.15001, 2026. [Paper](https://arxiv.org/abs/2609.15001).

```bibtex
@misc{liu2026hgto,
  title = {HGTO: A Unified Graph-Based Physics-Informed Formulation for Structural Topology Optimization},
  author = {Kangzheng Liu and Uday Kumar Punna and Leixin Ma},
  year = {2026},
  eprint = {2609.15001},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  doi = {10.48550/arXiv.2609.15001},
  url = {https://arxiv.org/abs/2609.15001}
}
```

Machine-readable citation metadata is in [CITATION.cff](CITATION.cff). HGTO is distributed under the [MIT license](LICENSE). NTopo retains its original license; see [third-party notices](THIRD_PARTY.md).
