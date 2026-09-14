#!/usr/bin/env python
"""Wang-regularized NH adaptation of the author's NTopo dual-network code.

This is an explicitly adapted nonlinear baseline, not an upstream capability.
The vendored SIRENs, continuous AD kinematics, stratified sample generators,
train_mmse loop, sensitivity filter and MSE fitting are retained. The problem
mapping, material energy and physical density floor change; an explicitly
selected reciprocal/linear OC extension handles mixed-sign NH sensitivities.

The design objective is J=-2 min_u Pi = 2(f.u-U), not terminal f.u.  The
upstream negative partial-energy density derivative is its envelope gradient
up to the constant factor two, absorbed by the OC multiplier.  As in upstream
NTopo, an approximate neural state need not satisfy the envelope assumptions;
all final designs must therefore be reanalysed by the common nonlinear FEM.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

PUBLIC = Path(os.environ.get('HGTO_PUBLIC_PATH', Path(__file__).resolve().parents[2])).resolve()
if not (PUBLIC/'src/hgto/baselines/ntopo/worker.py').is_file() and os.environ.get('HGTO_NTOPO_UPSTREAM_PATH'):
    PUBLIC = Path(os.environ['HGTO_NTOPO_UPSTREAM_PATH']).resolve().parents[2]


def runtime():
    """Reuse only the already audited TF-version compatibility shims."""
    os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
    source = PUBLIC / 'src/hgto/baselines/ntopo/worker.py'
    spec = importlib.util.spec_from_file_location('ntopo_linear_compatibility', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.install_shims(), module


def wang_energy(tf, grad_u, rho, *, young=1., nu=.3, penalty=3.,
                emin_fraction=1e-6, beta0=500., eta0=.01, gamma_mode='heaviside'):
    """Physical energy density with full rho -> gamma -> F_gamma chain.

    grad_u has shape (..., 2, 2), and rho broadcasts to its leading shape.
    An invalid determinant is an explicit error, never clamped to a positive
    value. This function also supports float64 constitutive verification.
    """
    dtype = grad_u.dtype
    scalar = lambda value: tf.convert_to_tensor(value, dtype=dtype)
    rho = tf.cast(rho, dtype)
    beta, eta = scalar(beta0), scalar(eta0)
    low = tf.math.tanh(beta * eta)
    if gamma_mode=='simp_heaviside':gamma_input=tf.pow(rho,scalar(penalty))
    elif gamma_mode=='heaviside':gamma_input=rho
    else:raise ValueError(f'Unsupported Wang gamma mode: {gamma_mode}')
    gamma = (low + tf.math.tanh(beta * (gamma_input - eta))) / (
        low + tf.math.tanh(beta * (scalar(1.) - eta)))
    mu = scalar(young / (2. * (1. + nu)))
    lam = scalar(young * nu / (1. - nu * nu))
    scaled_h = gamma[..., None, None] * grad_u
    trace_h = tf.linalg.trace(scaled_h)
    det_h = (scaled_h[..., 0, 0] * scaled_h[..., 1, 1]
             - scaled_h[..., 0, 1] * scaled_h[..., 1, 0])
    det_increment = trace_h + det_h
    determinant = scalar(1.) + det_increment
    tf.debugging.assert_positive(determinant, message='Invalid det(F_gamma) in NH neural state')
    log_j = tf.math.log1p(det_increment)
    # Algebraically equivalent NH energy, with less cancellation near F=I.
    nonlinear = (scalar(.5) * mu * (tf.reduce_sum(scaled_h * scaled_h, axis=(-2, -1))
                 + scalar(2.) * (trace_h - log_j)) + scalar(.5) * lam * log_j * log_j)

    def linear(h):
        strain = scalar(.5) * (h + tf.linalg.matrix_transpose(h))
        trace = tf.linalg.trace(strain)
        return mu * tf.reduce_sum(strain * strain, axis=(-2, -1)) + scalar(.5) * lam * trace * trace

    full_linear = linear(grad_u)
    bracket = nonlinear - linear(scaled_h) + full_linear
    bracket = tf.where(gamma == scalar(1.), nonlinear, bracket)
    bracket = tf.where(gamma == scalar(0.), full_linear, bracket)
    floor = scalar(emin_fraction)
    scale = floor + (scalar(1.) - floor) * tf.pow(rho, scalar(penalty))
    # Preserve the full-solid endpoint without suppressing its density slope.
    return scale * bracket, determinant


def reciprocal_linear_oc(tf, native, *, old_densities, sensitivities,
                         sample_volume, target_volume, max_move,
                         damping_parameter):
    """Native OC for negative gradients; an explicit NH mixed-sign extension.

    The reciprocal surrogate for g<0 yields the original square-root update.
    A linear surrogate for g>=0 minimizes at its move lower bound when the
    volume multiplier is nonnegative. If those branches cannot reach the
    volume target, a signed equality multiplier is used. No gradient is negated or
    silently clipped, and no emitted neural density is modified here.
    """
    import numpy as np
    shapes=[tuple(value.shape) for value in old_densities]
    sizes=[int(np.prod(shape)) for shape in shapes]
    old=np.concatenate([value.numpy().reshape(-1) for value in old_densities]).astype(np.float64)
    gradient=np.concatenate([value.numpy().reshape(-1) for value in sensitivities]).astype(np.float64)
    if old.shape!=gradient.shape or not np.isfinite(old).all() or not np.isfinite(gradient).all():
        raise ValueError('Nonfinite or mismatched OC inputs')
    if np.any(old<0.) or np.any(old>1.) or sample_volume<=0.:
        raise ValueError('Invalid OC density or sample-domain volume')
    negative=gradient<0.
    record=dict(positive_gradient_fraction=float(np.mean(gradient>0.)),
                nonnegative_gradient_fraction=float(np.mean(~negative)),
                gradient_min=float(gradient.min()),gradient_max=float(gradient.max()),
                old_volume_fraction=float(old.mean()),target_volume_fraction=float(target_volume/sample_volume))
    if negative.all():
        result=native(old_densities=old_densities,sensitivities=sensitivities,
            sample_volume=sample_volume,target_volume=target_volume,max_move=max_move,
            damping_parameter=damping_parameter)
        record['path']='unchanged author OC (all gradients negative)'
    else:
        if damping_parameter != .5: raise ValueError('Reciprocal/linear NH OC requires damping=.5')
        lower=np.maximum(0.,old-max_move);upper=np.minimum(1.,old+max_move)
        branch_upper=np.where(negative,upper,lower)
        target=float(target_volume/sample_volume)
        minimum=float(lower.mean());maximum=float(branch_upper.mean())
        record.update(path='reciprocal/linear mixed-sign NH extension',
                      feasible_volume_min=minimum,feasible_volume_max=maximum)
        dv=float(sample_volume/len(old))
        if not minimum-1e-10 <= target <= maximum+1e-10:
            source=globals().get('SIGNED_OC_SOURCE',PUBLIC/'src/hgto/nonlinear/signed_oc.py')
            module_spec=importlib.util.spec_from_file_location('ntopo_signed_oc',source)
            module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
            value,details=module.negative_multiplier_update(old,gradient,np.full_like(old,dv),
                lower,upper,target,lambda design:float(design.mean()),tolerance=1e-10)
            record.update(details,path='reciprocal/linear signed equality multiplier')
            pieces=np.split(value,np.cumsum(sizes)[:-1])
            result=[tf.convert_to_tensor(piece.reshape(shape),dtype=old_tensor.dtype)
                    for piece,shape,old_tensor in zip(pieces,shapes,old_densities)]
            record['actual_target_volume_fraction']=float(np.concatenate([x.numpy().reshape(-1) for x in result]).astype(np.float64).mean())
            return result,record
        factor=np.zeros_like(gradient)
        factor[negative]=np.sqrt(-gradient[negative]/dv)
        left,right=0.,max(1.,float(np.max(factor*factor)))
        def candidate(multiplier):
            value=np.clip(old*factor/np.sqrt(multiplier),lower,upper)
            value[~negative]=lower[~negative]
            return value
        for _ in range(100):
            if candidate(right).mean() <= target:break
            right*=2.
        else:raise RuntimeError('OC multiplier upper bound failed')
        for _ in range(140):
            multiplier=.5*(left+right)
            value=candidate(multiplier)
            error=float(value.mean())-target
            if abs(error)<1e-10:break
            if error>0:left=multiplier
            else:right=multiplier
        else:raise RuntimeError('OC mixed-sign volume bisection failed')
        pieces=np.split(value,np.cumsum(sizes)[:-1])
        result=[tf.convert_to_tensor(piece.reshape(shape),dtype=old_tensor.dtype)
                for piece,shape,old_tensor in zip(pieces,shapes,old_densities)]
    values=np.concatenate([value.numpy().reshape(-1) for value in result])
    if not np.isfinite(values).all():raise RuntimeError('OC returned nonfinite targets')
    record['actual_target_volume_fraction']=float(values.astype(np.float64).mean())
    return result,record


def verify_mixed_oc(tf, destination):
    """Check native equivalence, constrained descent, and infeasible branches."""
    import numpy as np
    from ntopo.oc import compute_oc_multi_batch as native
    old=[tf.constant([[.30],[.45],[.55],[.70]],dtype=tf.float32)]
    negative=[tf.constant([[-1.],[-.3],[-.8],[-.2]],dtype=tf.float32)]
    args=dict(old_densities=old,sensitivities=negative,sample_volume=1.,
              target_volume=.5,max_move=.2,damping_parameter=.5)
    expected=native(**args)
    actual,negative_record=reciprocal_linear_oc(tf,native,**args)
    exact=all(np.array_equal(a.numpy(),b.numpy()) for a,b in zip(actual,expected))
    mixed=[tf.constant([[.2],[-.3],[-.8],[-.2]],dtype=tf.float32)]
    actual,mixed_record=reciprocal_linear_oc(tf,native,**(args|dict(sensitivities=mixed)))
    x=old[0].numpy().reshape(-1);y=actual[0].numpy().reshape(-1);g=mixed[0].numpy().reshape(-1)
    volume_error=abs(float(y.astype(np.float64).mean())-.5)
    linearized_change=float(g@(y-x))
    positive_at_lower=bool(np.allclose(y[g>=0],np.maximum(0.,x[g>=0]-.2),atol=1e-7,rtol=0))
    lower=np.maximum(0.,x-.2);upper=np.minimum(1.,x+.2)
    bounds=bool(np.all(y>=lower-1e-7) and np.all(y<=upper+1e-7))
    # A volume equality can require a negative multiplier even when every
    # gradient is positive. This is feasible inside the actual move bounds.
    positive,positive_record=reciprocal_linear_oc(tf,native,
        **(args|dict(sensitivities=[tf.ones_like(old[0])])))
    positive_volume_error=abs(float(positive[0].numpy().astype(np.float64).mean())-.5)
    refused=False
    try:
        reciprocal_linear_oc(tf,native,**(args|dict(target_volume=.99,
            sensitivities=[tf.ones_like(old[0])])))
    except (ValueError,RuntimeError):refused=True
    report=dict(all_negative_bitwise_equal=exact,mixed_volume_error=volume_error,
        mixed_linearized_objective_change=linearized_change,positive_at_move_lower=positive_at_lower,
        move_bounds_respected=bounds,infeasible_branch_refused=refused,
        positive_volume_error=positive_volume_error,positive_case=positive_record,
        negative_case=negative_record,mixed_case=mixed_record)
    report['passed']=exact and volume_error<1e-7 and positive_volume_error<1e-7 and linearized_change<0 and positive_at_lower and bounds and refused
    Path(destination).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report),flush=True)
    if not report['passed']:raise RuntimeError('NH mixed-sign OC verification failed')


def verify_constitutive(tf, source, destination):
    """Compare energy, stress and rho derivative with canonical saved values."""
    import numpy as np
    data = dict(np.load(source))
    gamma_mode=str(data['gamma_mode'].item()) if 'gamma_mode' in data else 'heaviside'
    material=dict(gamma_mode=gamma_mode,beta0=float(data.get('beta0',500.)),eta0=float(data.get('eta0',.01)))
    h = tf.Variable(data['F'] - np.eye(2), dtype=tf.float64)
    rho = tf.Variable(data['rho'][:, None], dtype=tf.float64)
    with tf.GradientTape(persistent=True) as tape:
        energy, determinant = wang_energy(tf, h, rho,**material)
        total = tf.reduce_sum(energy)
    stress = tape.gradient(total, h).numpy()
    gradient = tape.gradient(total, rho).numpy().reshape(-1)
    actual = {'energy': energy.numpy(), 'stress': stress, 'rho_gradient': gradient}
    checks = {}
    for key, values in actual.items():
        expected = data[key]
        absolute = float(np.max(np.abs(values - expected)))
        scale = max(float(np.max(np.abs(expected))), 1e-12)
        checks[key] = dict(max_abs_error=absolute, max_scaled_error=absolute/scale,
                           passed=bool(np.allclose(values, expected, rtol=2e-8, atol=2e-12)))
    # Centered density finite difference checks the implemented AD chain too.
    eps = 1e-7
    interior = (data['rho'] > eps) & (data['rho'] < 1. - eps)
    perturbed = []
    for sign in [-1., 1.]:
        value, _ = wang_energy(tf, h, rho + tf.cast(sign*eps, tf.float64),**material)
        perturbed.append(value.numpy().sum(axis=-1))
    finite_difference = (perturbed[1] - perturbed[0]) / (2*eps)
    fd_error = float(np.max(np.abs(finite_difference[interior] - gradient[interior])))
    fd_scale = max(float(np.max(np.abs(gradient[interior]))), 1e-12)
    report = dict(checks=checks, density_fd_scaled_error=fd_error/fd_scale,
                  gamma_mode=gamma_mode,
                  min_det_F_gamma=float(tf.reduce_min(determinant)),
                  samples=int(data['rho'].size), gauss_points=int(data['F'].shape[1]),
                  passed=all(item['passed'] for item in checks.values()) and fd_error/fd_scale < 2e-7)
    Path(destination).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, allow_nan=False), flush=True)
    if not report['passed']:
        raise RuntimeError('Wang material verification failed')


def run(tf, compatibility, out):
    import numpy as np
    from ntopo.models import DispModel, DensityModel
    from ntopo.problems import Problem2D, fix_left
    from ntopo.physics import DiracForce, compute_elasticity_energies
    from ntopo.utils import get_default_sample_counts, get_sample_generator, set_random_seed
    from ntopo.monitors import SimulationMonitor
    import ntopo.train as train
    import functools

    config = json.loads((out/'config.json').read_text())
    global SIGNED_OC_SOURCE
    SIGNED_OC_SOURCE=out/'signed_oc_executed.py'
    SIGNED_OC_SOURCE.write_bytes((PUBLIC/'src/hgto/nonlinear/signed_oc.py').read_bytes())
    data = dict(np.load(out/'input.npz'))
    if config.get('family', 'cantilever') not in ('cantilever','bridge'):
        raise ValueError('This nonlinear adaptation currently validates a rectangular cantilever only')
    gpus = tf.config.list_physical_devices('GPU')
    if os.environ.get('CUDA_VISIBLE_DEVICES', '') and not gpus:
        raise RuntimeError('GPU requested but TensorFlow found no GPU')
    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
    set_random_seed(config.get('seed', 42))
    coords = data['coords']; extent = coords.max(0)-coords.min(0)
    coordinate_scale = .5 / extent[1]
    mapped = ((coords-coords.min(0))*coordinate_scale).astype(np.float32)
    centers = ((data['centroids']-coords.min(0))*coordinate_scale).astype(np.float32)
    physical_area = float(np.prod(extent))
    forces = data['forces'].reshape(-1, 2)
    nodes = np.flatnonzero(np.any(forces != 0, axis=1))
    rho_min = float(config.get('rho_min', .001))
    vf = float(config['volume_fraction'])
    if not 0 <= rho_min < vf < 1: raise ValueError('Require 0 <= rho_min < volume_fraction < 1')

    problem = Problem2D()
    problem.domain = np.array([0, extent[0]*coordinate_scale, 0, .5], dtype=np.float32)
    if config.get('family')=='bridge':
        span=float(problem.domain[1])
        def fixed_both(inputs):
            x=inputs[0]
            factor=4.*x*(span-x)/span
            return tf.concat([factor,factor],axis=1)
        problem.bc=fixed_both
    else:
        problem.bc = functools.partial(fix_left, dim=2)
    # The upstream force object takes a mean, hence the point-count factor.
    # Forces themselves retain canonical physical units, with no load scaling.
    problem.forcing = DiracForce(mapped[nodes].tolist(), (len(nodes)*forces[nodes]).tolist())
    problem.init()
    problem.domain_volume = physical_area
    problem.free_volume = physical_area
    factors = np.asarray(problem.bc([tf.constant(mapped[:, :1]), tf.constant(mapped[:, 1:])])).reshape(-1)
    assert np.array_equal(np.flatnonzero(factors == 0.), np.sort(data['fixed']))
    probe = np.random.default_rng(440).normal(size=(len(nodes), 2)).astype(np.float32)
    actual_work = -float(problem.forcing.compute_force_loss(lambda _: tf.constant(probe), None).numpy().item())
    expected_work = float(np.sum(probe*forces[nodes]))
    assert np.isclose(actual_work, expected_work, rtol=2e-6, atol=1e-10)

    class WangEnergy(tf.keras.layers.Layer):
        def call(self, inputs, training=None):
            densities, derivatives = inputs
            # AD differentiates with respect to normalized sample coordinates;
            # restore the physical spatial gradient before evaluating energy.
            h = tf.stack(derivatives, axis=1) * coordinate_scale
            energy, _ = wang_energy(tf, h, tf.reshape(densities, [-1]),
                                    penalty=float(config.get('penalty', 3.)),
                                    emin_fraction=float(config.get('emin_fraction', 1e-6)),
                                    gamma_mode=config.get('gamma_mode','heaviside'),
                                    beta0=float(config.get('beta0',500.)),eta0=float(config.get('eta0',.01)))
            return energy[:, None]

    class DensityFloor:
        def apply(self, inputs, densities): return rho_min + (1.-rho_min)*densities

    problem.energy_model = WangEnergy()
    problem.plot_displacement = lambda *a, **k: None
    problem.plot_densities = lambda *a, **k: None
    train.save_densities_to_file = lambda *a, **k: None
    train.save_model_configs = lambda *a, **k: None
    SimulationMonitor.save_plot = lambda *a, **k: None
    features = dict(class_name='ConcatSineFeatures', config=dict(n_input=2))
    def model_config(n_output):
        return dict(class_name='DenseSIRENModel', config=dict(n_input=4, n_output=n_output,
                    n_hidden=60, last_layer_init_scale=1e-3, omega0=60.))
    disp = DispModel(problem.domain, 2, problem.bc, features=features, model=model_config(2))
    density = DensityModel(problem.domain, 2, (vf-rho_min)/(1.-rho_min),
                           constraint=DensityFloor(), features=features, model=model_config(1))
    assert not np.array_equal(disp.model.dense1.kernel.numpy(), disp.model.dense2.kernel.numpy())
    opt_disp = tf.keras.optimizers.Adam(learning_rate=3e-4, beta_2=.99)
    opt_density = tf.keras.optimizers.Adam(learning_rate=3e-4, beta_1=.8, beta_2=.9)
    samples = get_default_sample_counts(problem.domain, config.get('sample_budget', 7500))
    sim_generator = get_sample_generator(problem.domain, samples)
    opt_generator = get_sample_generator(problem.domain, samples)
    (out/'snapshots').mkdir(exist_ok=True)
    rows = []
    train_started = time.perf_counter()

    def snapshot(disp_model, density_model, save_path, save_postfix):
        iteration = int(save_postfix.lstrip('-'))
        rho = np.asarray(density_model(tf.constant(centers), training=False)).reshape(-1)
        assert np.isfinite(rho).all() and (rho >= rho_min).all() and (rho <= 1.).all(), 'Invalid emitted physical density'
        np.save(out/'snapshots'/f'rho_{iteration:04d}.npy', rho)
        energy, force_loss = compute_elasticity_energies(problem, disp_model, density_model,
                                                       tf.constant(centers), training=False)
        internal = float(energy.numpy().item()); external = -float(force_loss.numpy().item())
        row = dict(iteration=iteration, wall_s_train=time.perf_counter()-train_started,
                   mean_density=float(rho.mean()), neural_internal_energy=internal,
                   neural_terminal_work=external, neural_J=2.*(external-internal),
                   diagnostic_scope='Approximate current neural state; center-sampled energy, not certified equilibrium')
        rows.append(row)
        (out/'history.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
        print('SNAPSHOT '+json.dumps(row, allow_nan=False), flush=True)

    if config.get('mixed_sign_oc') == 'reciprocal_linear':
        original_oc=train.compute_oc_multi_batch
        oc_history=[]
        def adapted_oc(**kwargs):
            try:
                targets,diagnostic=reciprocal_linear_oc(tf,original_oc,**kwargs)
            except Exception as error:
                (out/'oc_failure.json').write_text(json.dumps(dict(outer=len(oc_history)+1,
                    error=str(error),mode='reciprocal_linear'),indent=2,allow_nan=False)+'\n')
                raise
            diagnostic['outer']=len(oc_history)+1
            oc_history.append(diagnostic)
            (out/'oc_history.json').write_text(json.dumps(oc_history,indent=2,allow_nan=False)+'\n')
            return targets
        train.compute_oc_multi_batch=adapted_oc
    elif config.get('mixed_sign_oc','native')!='native':
        raise ValueError('Unknown mixed-sign OC mode')
    train.save_model_weights = snapshot
    try:
        train.train_mmse(problem, disp, density, opt_disp, opt_density,
            n_opt_samples=samples, opt_sample_generator=opt_generator, sim_sample_generator=sim_generator,
            vol_penalty_strength=10., target_volume_ratio=vf, save_path=str(out),
            filter='sensitivity', filter_radius=2., use_oc=True,
            save_interval=config.get('snapshot_interval', 10), n_opt_iterations=config['outer'],
            n_sim_iterations=config['inner'], n_opt_batches=config['batches'],
            oc_config=dict(max_move=.2, damping_parameter=.5))
    except Exception:
        captured=np.asarray(density(tf.constant(centers),training=False)).reshape(-1)
        np.save(out/'failed_density.npy',captured)
        (out/'failure_capture.json').write_text(json.dumps(dict(
            completed_density_updates=len(oc_history),last_saved_snapshot=rows[-1]['iteration'] if rows else None,
            signed_oc_helper_sha256=hashlib.sha256(SIGNED_OC_SOURCE.read_bytes()).hexdigest(),
            density_sha256=hashlib.sha256((out/'failed_density.npy').read_bytes()).hexdigest(),
            finite_density=bool(np.isfinite(captured).all()),scope='Actual density at training failure; no final optimization metrics.'),indent=2)+'\n')
        raise
    if rows[-1]['iteration'] != config['outer']: snapshot(disp, density, str(out), f"-{config['outer']:06d}")
    train_seconds = time.perf_counter()-train_started
    rho = np.asarray(density(tf.constant(centers), training=False)).reshape(-1)
    assert np.isfinite(rho).all() and (rho >= rho_min).all() and (rho <= 1.).all(), 'Invalid final physical density'
    np.save(out/'rho.npy', rho)
    np.save(out/'neural_displacement_canonical.npy', np.asarray(disp(tf.constant(mapped), training=False)))
    disp.save_weights(str(out/'displacement.weights.h5'))
    density.save_weights(str(out/'density.weights.h5'))
    tf.train.Checkpoint(displacement=disp, density=density, opt_disp=opt_disp,
                        opt_density=opt_density).write(str(out/'checkpoint'))
    hashes = {str(path.relative_to(compatibility.UPSTREAM)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in compatibility.UPSTREAM.rglob('*') if path.is_file() and '.git' not in path.parts}
    result = dict(method='NTopo (NH adaptation)', case=config['case'], upstream_commit=compatibility.COMMIT,
        signed_oc_helper_sha256=hashlib.sha256(SIGNED_OC_SOURCE.read_bytes()).hexdigest(),
        upstream_url='https://github.com/JonasZehn/ntopo', upstream_sha256=hashes,
        compatibility_source_sha256=hashlib.sha256(Path(compatibility.__file__).read_bytes()).hexdigest(),
        objective='J=-2 min_u Pi=2(f.u-U); density OC uses -partial U/partial rho',
        mixed_sign_oc=config.get('mixed_sign_oc','native'),
        wall_s_train=train_seconds, volume=float(rho.mean()), target_volume=vf,
        seed=config.get('seed', 42), train_samples=[int(n) for n in samples],
        outer_iterations=config['outer'], inner_iterations=config['inner'], optimization_batches=config['batches'],
        displacement_steps=(config['outer']+1)*config['inner'], density_steps=config['outer']*config['batches'],
        n_parameters_displacement=disp.count_params(), n_parameters_density=density.count_params(),
        tensorflow_version=tf.__version__, devices=[gpu.name for gpu in gpus],
        mapping=dict(coordinate_scale=coordinate_scale, displacement_scale=1., force_scale=1.,
                     physical_area=physical_area, force_point_mean_compensation=len(nodes),
                     boundary_zero_set_verified=True, force_work_verified=True),
        material=dict(young=1., nu=.3, penalty=config.get('penalty', 3.), emin_fraction=config.get('emin_fraction', 1e-6),
                      gamma=config.get('gamma_mode','heaviside'), beta0=config.get('beta0',500.), eta0=config.get('eta0',.01), rho_min=rho_min,
                      calibration='2D compressible NH with small-strain plane-stress-matched Lame constants'),
        modifications=['Wang NH energy and full gamma density chain', 'physical density lower bound',
                       'Explicit reciprocal/linear OC extension for mixed-sign NH sensitivities' if config.get('mixed_sign_oc')=='reciprocal_linear' else 'Unchanged author OC',
                       'physical spatial-gradient restoration after input-coordinate normalization',
                       'physical force and domain measure adapter', 'existing TF compatibility and numeric snapshots'],
        final_density_processing='None; raw continuous-network density at FE cell centers',
        native_filter='Author sensitivity filter, radius 2 sample cells; not a physical-density filter',
        native_filter_radius_physical=2.*float(extent[0])/int(samples[0]),
        comparison_scope='Same material, objective, load and domain; continuous AD training differs from Q4 reanalysis',
        recipe_status='paper-scale NH adaptation' if config['inner']==1000 and config['batches']==50 and config['outer']>=200 else 'smoke-or-development')
    (out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('RESULT '+json.dumps({key: value for key, value in result.items() if key != 'upstream_sha256'}, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--verify-reference', type=Path)
    parser.add_argument('--verify-oc', action='store_true')
    parser.add_argument('--verification-output', type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    tf, compatibility = runtime()
    if args.verify_oc:
        verify_mixed_oc(tf,args.verification_output)
    elif args.verify_reference:
        verify_constitutive(tf, args.verify_reference, args.verification_output)
    else:
        if args.out is None: parser.error('--out is required for training')
        if (args.out/'result.json').exists(): raise FileExistsError('Refusing to overwrite a finished NTopo run')
        try:
            run(tf, compatibility, args.out)
        except Exception as exc:
            import traceback
            (args.out/'failure.json').write_text(json.dumps(dict(message=str(exc),
                traceback=traceback.format_exc(), wall_s_worker=time.perf_counter()-started), indent=2, allow_nan=False)+'\n')
            raise
        record = json.loads((args.out/'result.json').read_text())
        record['wall_s_worker'] = time.perf_counter()-started
        record['executed_worker_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        (args.out/'result.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__': main()
