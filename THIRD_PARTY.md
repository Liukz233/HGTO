# Third-party notices

HGTO's source is distributed under the repository's MIT license.
PyTorch, PyTorch Geometric, NumPy, SciPy, scikit-fem, PyAMG, Matplotlib,
PyYAML, and optional runtime packages are imported dependencies and retain
their own licenses. Chebyshev convolutions use PyTorch Geometric's `ChebConv`.

## NTopo

Jonas Zehnder, Yue Li, Stelian Coros, and Bernhard Thomaszewski.
*NTopo: Mesh-free Topology Optimization using Implicit Neural Representations*,
NeurIPS 2021.

- Upstream: https://github.com/JonasZehn/ntopo
- Commit: `d3e17ca4cfb1d7a71c4c4f0c965cfcdc67d53fa9`
- License: MIT, Copyright (c) 2021 JonasZehn
- Original source and license: `third_party/ntopo/upstream/`
- File fingerprints: `third_party/ntopo/PINNED_SOURCE.json`

The upstream files are unmodified. HGTO's linear adapters live in
`src/hgto/baselines/ntopo/`. The separately named `nonlinear_worker.py`
implements this paper's Neo-Hookean adaptation, described in
`third_party/ntopo/NONLINEAR_ADAPTATION.md`.
