# Baselines

## SIMP--OC

Use `hgto run --method oc` with a configuration containing an `oc` section.
Linear OC uses independent scikit-fem elasticity with the PARDISO solver.
Nonlinear OC uses the same incremental material model, load bisection and
state continuation as HGTO, with the declared mixed-sign update, step
acceptance and final-stage adaptive move limits; its tangents are factorized
by PARDISO on two CPU threads. Passive elements, such as the rings of the beam
with reinforced openings, keep their prescribed density in both methods. Both
methods retain their own optimization trajectory.

## NTopo

The repository includes the unmodified MIT-licensed NTopo author source at
commit `d3e17ca4cfb1d7a71c4c4f0c965cfcdc67d53fa9` under
`third_party/ntopo/upstream/`. Its TensorFlow runtime is isolated from HGTO's
PyTorch runtime.

Create the separate environment:

```bash
python3.11 -m venv .venv-ntopo
.venv-ntopo/bin/python -m pip install -r third_party/ntopo/requirements-runtime.txt
export HGTO_NTOPO_PYTHON="$PWD/.venv-ntopo/bin/python"
```

Run the launcher with the HGTO environment, not the TensorFlow environment:

```bash
python scripts/run_ntopo.py --config configs/linear2d/cantilever_120x40.yaml --output runs/cantilever/ntopo --device 0 --fem-threads 2
```

`--device 0` selects CUDA device 0 in the isolated TensorFlow process;
`--device cpu` disables its GPU. Linear budgets are 200 outer iterations in
2D and 100 in 3D, with 1,000 state updates and 50 density-fitting batches per
outer iteration. `--dry-run` inspects the setup. Reduced budgets are smoke
checks, not paper results.

The beam with reinforced openings uses the same launcher:

```bash
python scripts/run_ntopo.py --config configs/domains/ring_beam.yaml --output runs/ring_beam/ntopo --device 0 --fem-threads 2
```

Its rings and openings are NTopo's native hard density constraints on the
exact ring and opening polygons of the mesh. Ring samples enter the OC volume
sum at their constrained value 1, so the rings count toward the volume
fraction as in HGTO and SIMP--OC.

To obtain a finer readout of the same cantilever field, add
`--high-resolution 960 320` to its coarse-grid training command. This does not
train a separate high-resolution field. Final raw densities are evaluated by
the independent HGTO-side evaluator without volume correction.

The nonlinear launcher uses the paper's explicit Neo-Hookean adaptation:

```bash
python scripts/run_ntopo_nonlinear.py --case cantilever_nh --load 0.00125 --output runs/nonlinear/ntopo_weak --device 0
python scripts/run_ntopo_nonlinear.py --case cantilever_nh --load 0.0125 --output runs/nonlinear/ntopo_strong --device 0
python scripts/run_ntopo_nonlinear.py --case bridge_nh --load 0.1 --output runs/nonlinear/ntopo_bridge --device 0
python scripts/run_ntopo_nonlinear.py --case lbracket_nh --load 0.04 --output runs/nonlinear/ntopo_l_bracket --device 0
```

In the recorded experiments, the strong cantilever and bridge runs stop during
displacement training when the deformation determinant becomes nonpositive.
A failed process has no completed-design metric. The L-bracket run completes
its budget; a repeated run with the same seed stops during state training,
because the upstream GPU training is not bit-reproducible. The removed
quadrant is imposed as the native zero-density constraint of the linear
L-bracket mapping.
The exact material, derivative, and mixed-sign OC changes are described in
[`NONLINEAR_ADAPTATION.md`](../third_party/ntopo/NONLINEAR_ADAPTATION.md).
