"""Calls the unmodified NTopo train_mmse; compatibility and case adapters only."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

UPSTREAM = (
    Path(os.environ["HGTO_NTOPO_UPSTREAM_PATH"])
    if os.environ.get("HGTO_NTOPO_UPSTREAM_PATH")
    else Path(__file__).resolve().parents[4] / "third_party" / "ntopo" / "upstream"
)
COMMIT = "d3e17ca4cfb1d7a71c4c4f0c965cfcdc67d53fa9"


def install_shims():
    # Plotting is intentionally replaced by numeric snapshots. These packages
    # are not needed by any mechanics/training path and may not exist in TF env.
    def stub(name, attrs=()):
        module = types.ModuleType(name)
        for attr in attrs:
            setattr(module, attr, lambda *a, **k: None)
        sys.modules[name] = module
        return module

    mpl = stub("matplotlib", ("use", "reload"))
    mpl.pyplot = stub("matplotlib.pyplot")
    mpl.cm = stub("matplotlib.cm")
    sk = stub("skimage")
    sk.measure = stub("skimage.measure")
    stub("jsonpickle", ("encode", "decode"))

    class QuietProgress:
        def __init__(self, iterable):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_description(self, *a, **k):
            pass

        def refresh(self):
            pass

    progress = stub("tqdm")
    progress.tqdm = lambda it, *a, **k: QuietProgress(it)
    import tensorflow as tf
    import tf_keras

    pil = stub("PIL")
    pil.Image = stub("PIL.Image", ("fromarray",))
    legacy = stub("tensorflow.python.keras")
    legacy.__version__ = tf_keras.__version__
    stub("tensorflow.python.keras.saving")
    saved = stub("tensorflow.python.keras.saving.saved_model")
    saved.json_utils = types.SimpleNamespace()
    # TF 2.3 uses stateful random_uniform for an unseeded initializer. Modern
    # tf_keras reuses identical random arrays when one initializer is reused.
    # Restore the published-version semantics before NTopo constructs layers.
    base_initializer = tf.keras.initializers.RandomUniform

    class TF23RandomUniform(base_initializer):
        def __call__(self, shape, dtype=None, **kwargs):
            dtype = dtype or tf.keras.backend.floatx()
            if self.seed is None:
                return tf.random.uniform(shape, self.minval, self.maxval, dtype=dtype)
            return super().__call__(shape, dtype=dtype, **kwargs)

    tf.keras.initializers.RandomUniform = TF23RandomUniform
    sys.path.insert(0, str(UPSTREAM))
    sys.dont_write_bytecode = True
    return tf


def build_problem(config, data, tf):
    import functools
    import numpy as np
    from ntopo.physics import DiracForce
    from ntopo.problems import Problem2D, fix_left, LShapedBeam2D_bc
    from ntopo.constraints import (
        DensityConstraint,
        DisplacementConstraint,
        DisplacementPoint,
        zero_densities_function,
    )
    from ntopo.sdf import SDFRectangle

    p = Problem2D()
    coords = data["coords"]
    lo = coords.min(axis=0)
    extent = coords.max(axis=0) - lo
    # Match upstream Beam2D's 0.5 height and preserve canonical aspect ratio.
    scale = 0.5 / extent[1]
    p.domain = np.array([0, extent[0] * scale, 0, 0.5], dtype=np.float32)
    mapped = ((coords - lo) * scale).astype(np.float32)
    centers = ((data["centroids"] - lo) * scale).astype(np.float32)
    family = config["family"]
    if family in ("cantilever", "inclined"):
        p.bc = functools.partial(fix_left, dim=2)
    elif family == "mbb":
        right_roller = DisplacementConstraint(
            p.domain, [DisplacementPoint([float(p.domain[1]), 0.0])]
        )
        p.bc = lambda x: tf.concat(
            [x[0], right_roller.compute_length_factor(tf.concat(x, axis=1))], axis=1
        )
    elif family == "bridge":
        left_pin = DisplacementConstraint(p.domain, [DisplacementPoint([0.0, 0.0])])
        both = DisplacementConstraint(
            p.domain, [DisplacementPoint([0.0, 0.0]), DisplacementPoint([float(p.domain[1]), 0.0])]
        )
        p.bc = lambda x: tf.concat(
            [
                left_pin.compute_length_factor(tf.concat(x, axis=1)),
                both.compute_length_factor(tf.concat(x, axis=1)),
            ],
            axis=1,
        )
    elif family == "l_bracket":
        nx, ny = config["shape"][1], config["shape"][0]
        cutx, cuty = nx // 2 * scale, ny // 2 * scale
        p.density_constraint = DensityConstraint(
            SDFRectangle([cutx, p.domain[1], cuty, 0.5]), zero_densities_function
        )
        p.bc = lambda x: tf.tile(0.5 - x[1], [1, 2])
    else:
        raise ValueError(family)
    f = data["forces"].reshape(-1, 2)
    nodes = np.flatnonzero(np.any(f != 0, axis=1))
    # Author DiracForce averages over force points. Supply quadrature weights
    # so its mean is exactly the canonical sum of nodal external work.
    point_count = len(nodes)
    p.forcing = DiracForce(
        position=mapped[nodes].tolist(), force=(point_count * 0.0025 * f[nodes]).tolist()
    )
    p.init()
    # Verify BC ansatz zero set on *all* canonical nodal DOFs and verify forces
    # use exactly the supplied nodal locations/components, without load snapping.
    factors = np.asarray(p.bc([tf.constant(mapped[:, 0:1]), tf.constant(mapped[:, 1:2])])).reshape(
        -1
    )
    zeros = np.flatnonzero(np.abs(factors) < 1e-12)
    assert np.array_equal(zeros, np.sort(data["fixed"])), (zeros, data["fixed"])
    assert np.allclose(
        np.asarray(p.forcing.force) / point_count, 0.0025 * f[nodes], rtol=1e-6, atol=1e-12
    )
    assert np.allclose(np.asarray(p.forcing.force_position), mapped[nodes], rtol=0, atol=1e-7)
    return (
        p,
        centers,
        dict(
            coordinate_scale=scale,
            force_scale=0.0025,
            canonical_fixed_dof_count=len(zeros),
            boundary_zero_set_verified=True,
            load_nodes=nodes.tolist(),
            load_positions=mapped[nodes].tolist(),
            load_vectors=(0.0025 * f[nodes]).tolist(),
            upstream_force_vectors=p.forcing.force.tolist(),
            force_point_mean_compensation=point_count,
        ),
    )


def build_problem_irregular(config, data, tf):
    """Use native zero-density constraints on the exact canonical cutout.

    The hole follows the polygonal boundary of the released mesh, avoiding
    an analytic-circle/mesh-domain mismatch. Training keeps the author's
    rectangular stratified sampling, energy, filter and OC/MSE updates.
    """
    import numpy as np
    from ntopo.problems import Problem2D, fix_left
    from ntopo.physics import DiracForce
    from ntopo.constraints import DensityConstraint, zero_densities_function
    from ntopo.sdf import SDFRectangle
    import functools

    coords, cells = data["coords"], data["cells"]
    lo = coords.min(0)
    extent = coords.max(0) - lo
    scale = 0.5 / extent[1]
    mapped = ((coords - lo) * scale).astype(np.float32)
    centers = ((data["centroids"] - lo) * scale).astype(np.float32)
    p = Problem2D()
    p.domain = np.array([0.0, extent[0] * scale, 0.0, 0.5], dtype=np.float32)
    metadata = config["extra"]
    polygon = None
    if config["family"] == "l_bracket_domain":
        x0, y0, x1, y1 = metadata["cutout"]
        sdf = SDFRectangle(
            [(x0 - lo[0]) * scale, (x1 - lo[0]) * scale, (y0 - lo[1]) * scale, (y1 - lo[1]) * scale]
        )
        p.bc = lambda x: tf.tile(0.5 - x[1], [1, 2])
        excluded_area = (x1 - x0) * (y1 - y0)
    elif config["family"] == "perforated_bracket":
        edges = np.sort(np.concatenate([cells[:, [i, (i + 1) % 4]] for i in range(4)]), axis=1)
        edges, counts = np.unique(edges, axis=0, return_counts=True)
        boundary_nodes = np.unique(edges[counts == 1])
        xy = coords[boundary_nodes]
        outer = np.any(
            np.isclose(xy, lo, atol=1e-10) | np.isclose(xy, lo + extent, atol=1e-10), axis=1
        )
        polygon = xy[~outer]
        center = np.asarray(metadata["hole_center"])
        polygon = polygon[np.argsort(np.arctan2(*(polygon - center)[:, ::-1].T))]
        following = np.roll(polygon, -1, axis=0)
        excluded_area = 0.5 * np.sum(
            polygon[:, 0] * following[:, 1] - following[:, 0] * polygon[:, 1]
        )
        if len(polygon) < 3 or excluded_area <= 0:
            raise ValueError("Expected one counterclockwise convex clearance boundary")
        edges = following - polygon
        normals = np.c_[edges[:, 1], -edges[:, 0]] / np.linalg.norm(edges, axis=1)[:, None]
        signed = np.einsum("ijk,jk->ij", polygon[:, None, :] - polygon[None, :, :], normals)
        if np.max(signed) > 1e-8:
            raise ValueError("The polygonal clearance must be convex")

        class ConvexHole:
            def __init__(self):
                self.normals = tf.constant(normals.astype(np.float32))
                self.offsets = tf.constant(
                    np.sum(((polygon - lo) * scale) * normals, axis=1).astype(np.float32)
                )

            def eval_distance(self, positions):
                return tf.reduce_max(
                    tf.matmul(positions, self.normals, transpose_b=True) - self.offsets,
                    axis=1,
                    keepdims=True,
                )

        sdf = ConvexHole()
        p.bc = functools.partial(fix_left, dim=2)
    else:
        raise ValueError(config["family"])
    vertices = coords[cells]
    following = np.roll(vertices, -1, axis=1)
    area = 0.5 * np.sum(
        vertices[:, :, 0] * following[:, :, 1] - following[:, :, 0] * vertices[:, :, 1], axis=1
    )
    if np.any(area <= 0) or not np.isclose(area.sum(), extent.prod() - excluded_area, rtol=1e-10):
        raise ValueError("Neural domain area differs from the canonical mesh")
    if not np.all(sdf.eval_distance(tf.constant(centers)).numpy() > 0):
        raise ValueError("A canonical cell centroid lies in the neural cutout")
    p.density_constraint = DensityConstraint(sdf, zero_densities_function)
    force = data["forces"].reshape(-1, 2)
    nodes = np.flatnonzero(np.any(force != 0, axis=1))
    point_count = len(nodes)
    force_scale = 0.0025
    p.forcing = DiracForce(
        position=mapped[nodes].tolist(), force=(point_count * force_scale * force[nodes]).tolist()
    )
    p.init()
    estimated_free_volume = float(p.free_volume)
    p.free_volume = float(area.sum() * scale**2)
    p.constraint_volume = 0.0
    factors = np.asarray(p.bc([tf.constant(mapped[:, 0:1]), tf.constant(mapped[:, 1:2])])).reshape(
        -1
    )
    zeros = np.flatnonzero(np.abs(factors) < 1e-12)
    assert np.array_equal(zeros, np.sort(data["fixed"]))
    probe = np.random.default_rng(709).normal(size=(point_count, 2)).astype(np.float32)
    work = -float(p.forcing.compute_force_loss(lambda _: tf.constant(probe), None).numpy().item())
    expected = float(np.sum(probe * force_scale * force[nodes]))
    assert np.isclose(work, expected, rtol=2e-6, atol=1e-12)
    assert np.allclose(
        np.asarray(p.forcing.force) / point_count, force_scale * force[nodes], rtol=1e-6, atol=1e-12
    )
    assert np.allclose(np.asarray(p.forcing.force_position), mapped[nodes], rtol=0, atol=1e-7)
    return (
        p,
        centers,
        dict(
            coordinate_scale=scale,
            force_scale=force_scale,
            canonical_fixed_dof_count=len(zeros),
            boundary_zero_set_verified=True,
            load_nodes=nodes.tolist(),
            load_positions=mapped[nodes].tolist(),
            load_vectors=(force_scale * force[nodes]).tolist(),
            upstream_force_vectors=p.forcing.force.tolist(),
            force_point_mean_compensation=point_count,
            force_work_quadrature_verified=True,
            canonical_area=float(area.sum()),
            canonical_excluded_area=float(excluded_area),
            native_free_volume=p.free_volume,
            upstream_grid_estimated_free_volume=estimated_free_volume,
            volume_mapping="Exact canonical polygon area; OC targets native_free_volume times prescribed volume fraction",
            excluded_polygon=None if polygon is None else polygon.tolist(),
            domain_mapping="Native hard zero-density constraint on the exact canonical cutout; no energy in the excluded domain",
            readout="Raw continuous density at canonical Q4 centroids; no rescaling or repair",
        ),
    )


def build_problem_3d(config, data, tf):
    import numpy as np
    from ntopo.physics import DiracForce
    from ntopo.problems import Problem3D

    p = Problem3D()
    coords = data["coords"]
    lo = coords.min(0)
    extent = coords.max(0) - lo
    scale = 1.0 / extent.max()
    p.domain = np.array(
        [0, extent[0] * scale, 0, extent[1] * scale, 0, extent[2] * scale], dtype=np.float32
    )
    mapped = ((coords - lo) * scale).astype(np.float32)
    centers = ((data["centroids"] - lo) * scale).astype(np.float32)
    if config["family"] in ("cantilever_3d", "torsion_member"):
        p.bc = lambda x: tf.tile(x[0], [1, 3])
    else:
        patches = config["extra"]["support_patches"]

        def patch_distance_bc(x):
            distances = []
            for patch in patches:
                a = patch["axis"]
                tangent = [i for i in range(3) if i != a]
                d2 = tf.square(x[a] - patch["position"] * scale)
                for t, bounds in zip(tangent, patch["limits"]):
                    low, high = bounds[0] * scale, bounds[1] * scale
                    d2 += tf.square(tf.maximum(low - x[t], 0.0)) + tf.square(
                        tf.maximum(x[t] - high, 0.0)
                    )
                distances.append(d2)
            # Exact union of finite support patches; zero nowhere else. This
            # extends the author's distance-factor BC prescription to3D.
            dist2 = tf.reduce_min(tf.stack(distances, axis=0), axis=0)
            return tf.tile(2.0 * tf.sqrt(dist2 + 1e-35), [1, 3])

        p.bc = patch_distance_bc
    f = data["forces"].reshape(-1, 3)
    nodes = np.flatnonzero(np.any(f != 0, axis=1))
    force_scale = 0.0025 / float(np.linalg.norm(f[nodes], axis=1).sum())
    # DiracForce computes mean_i(f_i dot u_i), not sum_i. Multiplication by
    # node count preserves the exact canonical nodal-work quadrature.
    point_count = len(nodes)
    p.forcing = DiracForce(
        position=mapped[nodes].tolist(), force=(point_count * force_scale * f[nodes]).tolist()
    )
    p.init()
    factors = np.asarray(p.bc([tf.constant(mapped[:, i : i + 1]) for i in range(3)])).reshape(-1)
    zeros = np.flatnonzero(np.abs(factors) < 1e-12)
    assert np.array_equal(zeros, np.sort(data["fixed"]))
    assert np.allclose(
        np.asarray(p.forcing.force) / point_count, force_scale * f[nodes], rtol=1e-6, atol=1e-12
    )
    assert np.allclose(np.asarray(p.forcing.force_position), mapped[nodes], rtol=0, atol=1e-7)
    probe = np.random.default_rng(709).normal(size=(point_count, 3)).astype(np.float32)
    actual_work = -float(
        p.forcing.compute_force_loss(lambda _: tf.constant(probe), None).numpy().item()
    )
    expected_work = float(np.sum(probe * force_scale * f[nodes]))
    assert np.isclose(actual_work, expected_work, rtol=2e-6, atol=1e-12), (
        actual_work,
        expected_work,
    )
    return (
        p,
        centers,
        dict(
            coordinate_scale=scale,
            force_scale=force_scale,
            force_normalization="sum of nodal force magnitudes =0.0025 in native units",
            canonical_fixed_dof_count=len(zeros),
            boundary_zero_set_verified=True,
            load_nodes=nodes.tolist(),
            load_positions=mapped[nodes].tolist(),
            load_vectors=(force_scale * f[nodes]).tolist(),
            upstream_force_vectors=p.forcing.force.tolist(),
            force_point_mean_compensation=point_count,
            force_work_quadrature_verified=True,
            force_discretization="Exact canonical consistent nodal forces used as DiracForce quadrature; no load relocation",
            support_constraint="distance to finite rectangular support-patch union"
            if config["family"] == "four_foot_support"
            else "x=0 full clamp",
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    config = json.loads((out / "config.json").read_text())
    started = time.perf_counter()
    tf = install_shims()
    import numpy as np
    from ntopo.models import DispModel, DensityModel
    from ntopo.utils import (
        get_default_sample_counts,
        get_sample_generator,
        set_random_seed,
        get_grid_centers,
    )
    import ntopo.train as train
    from ntopo.monitors import SimulationMonitor

    gpus = tf.config.list_physical_devices("GPU")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") and not gpus:
        raise RuntimeError(
            "A GPU was requested but TensorFlow found no GPU; refusing silent CPU fallback."
        )
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    set_random_seed(config["seed"])
    dimension = config.get("dimension", 2)
    build = (
        build_problem_3d
        if dimension == 3
        else build_problem_irregular
        if config["family"] in ("l_bracket_domain", "perforated_bracket")
        else build_problem
    )
    problem, centers, mapping = build(config, np.load(out / "input.npz"), tf)
    problem.plot_displacement = lambda *a, **k: None
    problem.plot_densities = lambda *a, **k: None
    train.save_densities_to_file = lambda *a, **k: None
    train.save_model_configs = lambda *a, **k: None
    SimulationMonitor.save_plot = lambda *a, **k: None
    features = dict(class_name="ConcatSineFeatures", config=dict(n_input=dimension))

    def model_config(n_output):
        return dict(
            class_name="DenseSIRENModel",
            config=dict(
                n_input=2 * dimension,
                n_output=n_output,
                n_hidden=180 if dimension == 3 else 60,
                last_layer_init_scale=1e-3,
                omega0=60.0,
            ),
        )

    disp = DispModel(
        problem.domain, dimension, problem.bc, features=features, model=model_config(dimension)
    )
    density = DensityModel(
        problem.domain,
        dimension,
        config["volume_fraction"],
        constraint=problem.density_constraint,
        features=features,
        model=model_config(1),
    )
    # Reused random initializers must produce independent draws as in TF 2.3.
    assert not np.array_equal(disp.model.dense1.kernel.numpy(), disp.model.dense2.kernel.numpy())
    opt_disp = tf.keras.optimizers.Adam(learning_rate=3e-4, beta_2=0.99)
    opt_density = tf.keras.optimizers.Adam(learning_rate=3e-4, beta_1=0.8, beta_2=0.9)
    samples = get_default_sample_counts(
        problem.domain, 80 * 40 * 20 if dimension == 3 else 150 * 50
    )
    sim_gen = get_sample_generator(problem.domain, samples)
    opt_gen = get_sample_generator(problem.domain, samples)
    snapshots = out / "snapshots"
    snapshots.mkdir(exist_ok=True)
    history = []

    class Converged(Exception):
        pass

    stopped = False
    train_start = time.perf_counter()

    def snapshot(disp_model, density_model, save_path, save_postfix):
        iteration = int(save_postfix.lstrip("-"))
        rho = np.asarray(density_model(tf.constant(centers), training=False)).reshape(-1)
        assert np.isfinite(rho).all() and (rho >= 0).all() and (rho <= 1).all(), (
            "Invalid emitted density"
        )
        np.save(snapshots / f"rho_{iteration:04d}.npy", rho)
        if config.get("stopping") and config.get("high_resolution"):
            high_centers = get_grid_centers(
                problem.domain, config["high_resolution"], dtype=np.float32
            )
            high = np.asarray(density_model(tf.constant(high_centers), training=False)).reshape(-1)
            np.save(snapshots / f"rho_high_{iteration:04d}.npy", high)
        external_work = -float(
            problem.compute_force_loss(disp_model, tf.constant(centers)).numpy().reshape(-1)[0]
        )
        row = dict(
            iteration=iteration,
            wall_s_train=time.perf_counter() - train_start,
            mean_density=float(rho.mean()),
            grayness=float((4 * rho * (1 - rho)).mean()),
            neural_external_work=external_work,
        )
        history.append(row)
        (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print("SNAPSHOT " + json.dumps(row), flush=True)
        if config.get("stopping"):
            print("STOP_REQUEST " + json.dumps(dict(iteration=iteration)), flush=True)
            answer = json.loads(sys.stdin.readline())
            if answer["converged"]:
                raise Converged()

    train.save_model_weights = snapshot
    displacement_block_seconds = []
    original_simulation = train.run_simulation

    def timed_simulation(*args, **kwargs):
        block_start = time.perf_counter()
        answer = original_simulation(*args, **kwargs)
        seconds = time.perf_counter() - block_start
        displacement_block_seconds.append(seconds)
        if dimension == 3 or len(displacement_block_seconds) <= 2:
            print(
                "SIMULATION_BLOCK "
                + json.dumps(
                    dict(
                        block=len(displacement_block_seconds) - 1,
                        steps=config["inner"],
                        seconds=seconds,
                    )
                ),
                flush=True,
            )
        return answer

    train.run_simulation = timed_simulation
    try:
        train.train_mmse(
            problem,
            disp,
            density,
            opt_disp,
            opt_density,
            n_opt_samples=samples,
            opt_sample_generator=opt_gen,
            sim_sample_generator=sim_gen,
            vol_penalty_strength=10.0,
            target_volume_ratio=config["volume_fraction"],
            save_path=str(out),
            filter="sensitivity",
            filter_radius=2.0,
            use_oc=True,
            save_interval=1 if config.get("stopping") else config["snapshot_interval"],
            n_opt_iterations=config["outer"],
            n_sim_iterations=config["inner"],
            n_opt_batches=config["batches"],
            oc_config=dict(max_move=0.2, damping_parameter=0.5),
        )
    except Converged:
        stopped = True
    train_end = time.perf_counter()
    if not stopped and history[-1]["iteration"] != config["outer"]:
        snapshot(disp, density, str(out), f"-{config['outer']:06d}")
    completed = history[-1]["iteration"]
    rho = np.asarray(density(tf.constant(centers), training=False)).reshape(-1)
    np.save(out / "rho.npy", rho)
    raw_coords = np.load(out / "input.npz")["coords"]
    neural_nodes = ((raw_coords - raw_coords.min(0)) * mapping["coordinate_scale"]).astype(
        np.float32
    )
    native_displacement = np.asarray(disp(tf.constant(neural_nodes), training=False))
    canonical_displacement = (
        native_displacement
        * (mapping["coordinate_scale"] if dimension == 3 else 1.0)
        / mapping["force_scale"]
    )
    np.save(out / "neural_displacement_canonical.npy", canonical_displacement)
    high_time = None
    if config["high_resolution"]:
        t = time.perf_counter()
        high_centers = get_grid_centers(problem.domain, config["high_resolution"], dtype=np.float32)
        high = np.asarray(density(tf.constant(high_centers), training=False)).reshape(-1)
        np.save(out / "rho_high_resolution.npy", high)
        high_time = time.perf_counter() - t
    disp.save_weights(str(out / "displacement.weights.h5"))
    density.save_weights(str(out / "density.weights.h5"))
    tf.train.Checkpoint(
        displacement=disp, density=density, opt_disp=opt_disp, opt_density=opt_density
    ).write(str(out / "checkpoint"))
    hash_manifest = {
        str(p.relative_to(UPSTREAM)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in UPSTREAM.rglob("*")
        if p.is_file() and ".git" not in p.parts
    }
    result = dict(
        method="NTopo",
        case=config["case"],
        upstream_url="https://github.com/JonasZehn/ntopo",
        upstream_commit=COMMIT,
        upstream_sha256=hash_manifest,
        tensorflow_version=tf.__version__,
        keras_version=__import__("tf_keras").__version__,
        devices=[gpu.name for gpu in gpus],
        seed=config["seed"],
        mapping=mapping,
        wall_s_train=train_end - train_start,
        wall_s_worker=time.perf_counter() - started,
        mean_density=float(rho.mean()),
        grayness=float(np.mean(4 * rho * (1 - rho))),
        n_parameters_displacement=disp.count_params(),
        n_parameters_density=density.count_params(),
        train_samples=[int(s) for s in samples],
        readout_shape=config["shape"],
        high_resolution_shape=list(reversed(config["high_resolution"]))
        if config["high_resolution"]
        else None,
        high_resolution_grid_nx_ny=config["high_resolution"],
        high_resolution_readout_s=high_time,
        displacement_block_seconds=displacement_block_seconds,
        outer_iterations=completed,
        maximum_outer_iterations=config["outer"],
        converged=stopped,
        termination="converged" if stopped else "maximum_outer_iterations",
        stopping=config.get("stopping"),
        displacement_steps=(completed + 1) * config["inner"],
        density_steps=completed * config["batches"],
        inner_iterations=config["inner"],
        optimization_batches=config["batches"],
        native_material="E(rho)=rho^3; Esolid=1; nu=.3; "
        + ("3D linear elasticity" if dimension == 3 else "plane stress"),
        native_filter="sensitivity radius=2 sample cells; no density filtering or final projection",
        canonical_filter_radius=config["canonical_filter_radius"],
        native_filter_radius_in_canonical_coordinates=[
            2.0
            * float(problem.domain[2 * i + 1] - problem.domain[2 * i])
            / int(samples[i])
            / mapping["coordinate_scale"]
            for i in range(dimension)
        ],
        recipe_status="paper-scale"
        if config["inner"] == 1000
        and config["batches"] == 50
        and config["outer"] >= (100 if dimension == 3 else 200)
        else "smoke-or-development",
        modifications=[
            "case geometry/BC/force adapter",
            "TF 2.19 legacy Keras compatibility",
            "restore TF 2.3 stateful unseeded RandomUniform",
            "headless numeric snapshots",
        ],
    )
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "RESULT " + json.dumps({k: v for k, v in result.items() if k != "upstream_sha256"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
