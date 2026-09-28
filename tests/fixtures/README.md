# Nonlinear mechanics fixture

`bridge_density.npy` is the unchanged 3,072-element HGTO density from the paper’s doubly fixed Neo-Hookean bridge. It tests CPU tangent solvers on an optimized, high-contrast field.

SHA-256: `fe35f2be4ad623ffd0899241610a5027504e90c5aba4e4bc8741b3e01ae5105e`.

`perforated_volume_root.npz` contains logits, nonuniform element areas and a
row-normalized sparse density filter from a failed exact-volume projection in
the boundary-fitted, 1,536-element connection benchmark (HGTO elastic feedback,
seed 0, beta 8, volume fraction 0.4). The former Newton iteration cycled near
both bracket endpoints for 100 steps. The regression checks the safeguarded
root, mass conservation and the implicit gradient on CPU and CUDA. These are
actual captured optimizer inputs, with no generated mechanics results.
