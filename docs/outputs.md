# Run outputs

Every run writes to a new directory. Existing results are never overwritten
by `hgto run`. Use a different directory for each method or configuration.

| File | Contents |
|---|---|
| `config.yaml` | Configuration with command-line overrides |
| `rho.npy` | Complete physical density in element order |
| `result.json` | Final objective, volume, timing and termination metadata |
| `history.csv` / `history.json` | Available optimization observations |
| `snapshots.npz` | Saved physical-density snapshots and update indices |
| `source_manifest.json` | SHA-256 hashes of the executing HGTO modules |
| `network.pt` / `resume.pt` | Saved weights or optimizer checkpoint when available |

Linear runs also contain `geometry.npz` (`coords`, `cells`, `fixed`, `forces`)
and `state.npy`. Nonlinear runs contain `protocol.json`, `record.json`,
`coords.npy`, `econn.npy`, and `unit_force.npy`, together with material/state
history fields. Final Neo-Hookean evaluation writes a `verification/` folder
and `response.npz`. Plastic loading/unloading comparisons are written to the
separate directory passed to `compare-plastic`.

`result.json` for a Neo-Hookean run reports construction + optimization +
independent final evaluation. Plastic runs report `optimization_s`; common
loading/unloading evaluation is separate. Detailed nonlinear timing remains
available in `record.json`. Do not compare these different intervals as if
they were the same runtime measure.

`converged` describes the declared physical stopping rule. Reaching an update
limit or an OC line-search stall does not establish convergence. The saved
physical density is the field used for evaluation. Plotting does not sharpen,
rescale volume, or otherwise repair it; 3D views display its 0.5 isosurface.
