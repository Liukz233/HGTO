# NTopo adaptation to the common nonlinear objective

`nonlinear_worker.py` adapts the vendored author's implementation to the
project's Wang-regularized compressible Neo-Hookean material. This is a local
nonlinear adaptation, **not native upstream nonlinear functionality**.

The displacement and density SIRENs, sine features, continuous spatial AD,
stratified sampling, sensitivity filtering, `train_mmse` and MSE density fitting
remain the author's implementations. The all-negative-gradient OC update is
also unchanged; mixed-sign nonlinear sensitivities use the explicit extension
described below. The 2D architecture has
15,430 displacement parameters and 15,305 density parameters. The nominal
sampling budget is 7,500; aspect-ratio rounding determines the actual count.

The common objective is

\[
J=-2\min_u\Pi(u,\rho)=2\,[f^T u-U(u,\rho)].
\]

At exact equilibrium its density derivative is `-2 * partial U / partial rho`.
The upstream negative partial-energy derivative differs by a constant factor
two, which the OC multiplier absorbs. Automatic differentiation includes both
the SIMP stiffness factor and the Wang kinematic factor `gamma(rho)`. This
objective reduces to compliance in linear elasticity. It is **not** terminal
`f^T u` for a nonlinear material. An incompletely trained neural displacement
is not an exact equilibrium state; its reported training energy is diagnostic.

Unlike linear compliance, the Wang-interpolated nonlinear objective can have
positive as well as negative density sensitivities. The original square-root
OC rule is undefined for a positive sensitivity. Formal nonlinear runs select
`mixed_sign_oc="reciprocal_linear"`: a negative-gradient component retains the
author's reciprocal square-root update, while a nonnegative-gradient component
uses a linear local approximation. The volume-equality multiplier may be
negative when additional mass must be assigned to those linear branches; a
shared helper resolves the tied branch while respecting the move bounds.
Only genuine move-bound infeasibility is rejected. The all-negative
case calls the original OC function directly and has been checked for bitwise
agreement. This extension is disclosed as part of the NH adaptation. It does
not modify the sampled density values, energy gradients or final volume after
training.

Only the network's input coordinates are normalized (`q=s*x`, native height
0.5). Its output remains physical displacement. Spatial derivatives are
multiplied by `s` before constitutive evaluation, the integral uses the physical
domain area, and the physical force vector is retained. The author force class
averages over points, so its input vectors receive the exact point-count
compensation. There is no arbitrary load rescaling. The adapter supports a left-clamped rectangular cantilever and a doubly fixed
bridge, with geometry-specific boundary constraints.

The material uses `E=1`, `nu=0.3`, `p=3`, `Emin/E=1e-6`, and the same sharp
Wang factor (`beta0=500`, `eta0=0.01`). Formal revised cases explicitly select
`gamma_mode="simp_heaviside"`, applying the switch to `rho**p` and retaining
its complete density derivative. The legacy direct-density `heaviside` mode
is kept as the default for archived screening compatibility. Its two-dimensional Lamé constants match
the linear plane-stress law; this is not a finite-strain plane-stress
elimination. The physical density is `0.001 + 0.999 * sigmoid(...)`, with its
initial bias adjusted to the target volume. Raw final cell-center densities
are saved without filtering, thresholding or volume repair. The actual final
volume must be reported.

Continuous AD and sampling remain different from Q4 finite-element
reanalysis, as in the linear NTopo comparison. The native sensitivity filter
also remains distinct from the HGTO/OC physical-density filter. Final density
arrays must be evaluated with the common nonlinear FEM before comparison;
training energies cannot replace that evaluation. Invalid `det(F_gamma)` is
an explicit error, not a silently clipped constitutive value.

## Input and execution

From the public project root, use the HGTO interpreter and the separate
TensorFlow environment configured in the baseline README:

```bash
python scripts/run_ntopo_nonlinear.py --case cantilever_nh --load 0.00125 --output runs/ntopo_weak
python scripts/run_ntopo_nonlinear.py --case cantilever_nh --load 0.0125 --output runs/ntopo_strong
```

These commands select both revised material/OC modes explicitly, archive the
executed worker, and request the full 200/1000/50 budget. When training
completes, the launcher evaluates the unchanged final density through the
shared nonlinear benchmark and writes the displacement path, physical volume,
objective and complete process plus evaluation time. A training failure
instead retains its original inputs, log, executed source and failure/process
records; no final density or physical evaluation is manufactured. Each output
directory must be new. `--device cpu` is useful for diagnostics; formal
comparisons use the recorded GPU configuration.

The recorded weak-load run completed 200 outer iterations. The strong-load
run failed the positive `det(F_gamma)` check during displacement training in
outer iteration 4, after three completed iterations. Its only saved density
snapshot is iteration 0, with no final network checkpoint. The canonical
strong record therefore reports `training_failed`, null final metrics and
solution time, and a separate attempted-process duration. This describes the
outcome of the recorded configuration and load, not every possible NTopo
nonlinear configuration.

For a custom launcher, the worker input contract is as follows.

An output directory must first contain:

- `input.npz`: physical `coords`, Q4 `cells`, physical `centroids`, flattened
  `fixed` DOFs, and physical `forces` (one simultaneous load vector).
- `config.json`: `case`, `family` (`cantilever` or `bridge`), `volume_fraction`, `outer`,
  `inner`, and `batches`. Optional keys are `seed` (42), `rho_min` (0.001),
  `penalty` (3), `emin_fraction` (1e-6), `gamma_mode` (`heaviside` for legacy
  inputs; explicitly use `simp_heaviside` for the revised study), `mixed_sign_oc`
  (`native` for legacy inputs; explicitly use `reciprocal_linear` for the revised study),
  `snapshot_interval` (10), and
  `sample_budget` (7500).

Use the documented isolated TensorFlow environment:

```bash
TF_USE_LEGACY_KERAS=1 python third_party/ntopo/nonlinear_worker.py --out OUTPUT
```

When executing an archived copy of the worker, set `HGTO_PUBLIC_PATH` to this
package and `HGTO_NTOPO_UPSTREAM_PATH` to the author's source directory so the
audited compatibility layer and pinned source are located independently. The process
launcher should record its own wall time, including interpreter startup; the
worker reports its internal wall time and all training counts. Low budgets
are labeled `smoke-or-development`, never substituted for formal results.

`--verify-reference REFERENCE.npz --verification-output REPORT.json` compares
the float64 TensorFlow material with supplied canonical energy, stress and
density-gradient arrays. The reference uses `F` shaped `(elements,4,2,2)`,
one `rho` per element, per-point `energy` and `stress`, and the integrated
unweighted `rho_gradient` sum. It also runs a centered density-difference check.

The current mixed-sign extension also permits a negative volume-equality multiplier when the nonnegative branch cannot reach the required mass. It uses the same NumPy helper as the FEM OC adapter, leaves gradients unchanged, and preserves the original negative-gradient path. Tied linear branches share material symmetrically. An earlier pilot failure caused by the restricted multiplier is a superseded implementation result and must not be cited as neural-state instability. The executed helper is archived with each new nonlinear NTopo run.
