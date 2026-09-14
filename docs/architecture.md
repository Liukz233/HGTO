# Architecture

The material network and the equilibrium state share one mesh. Elements are
nodes in the density graph and hyperedges in the mechanics representation.
The density network has trainable weights. Mechanical message operators are
prescribed by element geometry and constitutive laws; they are not a second
trained displacement network.

```text
centroids + element adjacency
           |
    graph density network
           |
 weighted filtering + volume-preserving projection
           |
   physical element densities
           |
 node -> element mechanics -> node force assembly
           |
 equilibrium state + density-space sensitivities
           |
   backpropagation to graph weights
```

- `topopt/parameterize/chebnet.py`: Chebyshev graph convolutions and cached
  fixed features. `paper_K=1` means degree one, or two polynomial terms.
- `linear2d/design.py`: implicit derivative of the physical-volume constraint.
- `optimization.py`: the single linear Adam/continuation optimizer.
- `linear.py`: common problem construction, HGTO/OC dispatch, and independent
  final linear evaluation.
- `fem/operator.py`: gather/element/scatter operators and state solution.
- `fem/solvers/`: geometric, masked and algebraic multigrid; sparse tangent
  solvers; optional CUDA actions and cuDSS bindings.
- `fem/physics/`: Neo-Hookean and J2 constitutive responses, incremental state
  calculations, and corresponding sensitivities.
- `reference/`: separately assembled scikit-fem elasticity and linear OC.
- `nonlinear/optimization.py`: coupled incremental mechanics and density updates.
- `nonlinear/oc.py`: the conventional nonlinear density control.
- `nonlinear/evaluation.py`, `response.py`: independent final response and
  common loading/unloading evaluations.

Planar geometry uses `(x,y)` and spatial geometry `(x,y,z)`. Nodal degrees of
freedom are interleaved by node. Mesh ordering, force quadrature and volumes
are shared by the design and independent checks.

The public configuration names describe problems rather than development
rounds. `identifier_map.json` maps the original experiment identifiers to the
release names. The historical metric tables are normalized to configuration
paths; no experimental values were recomputed when creating those tables.
