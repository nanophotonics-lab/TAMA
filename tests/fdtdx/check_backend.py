"""Executable end-to-end GPU validation; requires the documented CUDA environment."""
import sys
import argparse
import json
import importlib.metadata
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from tama.backends.fdtdx.backend import MultiTDAObjective, TDAObjective
from tama.backends.fdtdx import (PointTarget, FieldRegionTarget,
    FluxTarget, EigenmodeCoefficientTarget, Band, bandpass_kernel, DesignGrid,
    periodic_conic_filter, tanh_projection, verify_fdtdx_sources)
from _scene import make_scene


def relative(a, b):
    a, b = np.asarray(a), np.asarray(b)
    difference, norm = float(np.linalg.norm(a - b)), float(np.linalg.norm(b))
    return difference / norm if norm else (0. if difference == 0 else float('inf'))


def native_reference(objective, scene):
    import fdtdx
    spec, design = objective.spec, objective.design
    start = np.asarray(scene['detector_start'])

    def reference(rho):
        arrays = spec.arrays.aset('inv_permittivities', spec.arrays.inv_permittivities.at[design.region].set(design.inverse_permittivity(rho)))
        _, final = fdtdx.run_fdtd(arrays=arrays, objects=spec.objects, config=spec.config,
                                 key=spec.key, show_progress=False)
        raw = final.detector_states[scene['detector_name']]['fields']
        fields = jnp.transpose(raw, (0, 2, 3, 4, 1))
        histories = []
        for target in objective.targets:
            ix = np.asarray(target.indices) - start
            histories.append(fields[:, ix[:, 0], ix[:, 1], ix[:, 2], :])
        return objective._loss(tuple(histories))

    return jax.jit(jax.value_and_grad(reference)), jax.jit(reference)


def targets_and_bands(scene):
    spec = scene['spec']
    start, shape = scene['detector_start'], scene['detector_shape']
    indices = np.array([[start[0], y, z] for y in range(start[1], start[1] + shape[1])
                        for z in range(start[2], start[2] + shape[2])])
    n = len(indices)
    area = spec.config.resolution ** 2
    E, H = np.zeros((n, 3)), np.zeros((n, 3))
    E[:, 2], H[:, 1] = 1, -1
    targets = [PointTarget(scene['target'], 'Ez'), PointTarget(scene['target'], 'Hy'),
               FieldRegionTarget(indices, 'Ez', np.ones(n) / n),
               FluxTarget(indices, (1, 0, 0), np.ones(n) * area),
               EigenmodeCoefficientTarget(indices, (1, 0, 0), np.ones(n) * area, E, H)]
    kernels = [bandpass_kernel(.65 * scene['frequency'], 1.05 * scene['frequency'], spec.dt, 41),
               bandpass_kernel(.95 * scene['frequency'], 1.35 * scene['frequency'], spec.dt, 40)]
    bands = [Band(0, kernels[0], weight=1 / spec.dt), Band(0, kernels[1], weight=.5 / spec.dt),
             Band(1, kernels[1], weight=.2 / spec.dt), Band(2, kernels[0], weight=.3 / spec.dt),
             Band(3, kernels[1], weight=.1 / (spec.dt * n * area)),
             Band(4, kernels[0], weight=.1 / spec.dt)]
    return targets, bands


def scalarize(values):
    return jnp.sum(values) + .01 * jnp.sum(values ** 2)


def run_case(name, *, dtype=jnp.float64, periodic_z=False, source_kind='uniform', steps=257, fabrication=False,
             boundary_kind='pml', bloch_vector=(0., 0., 0.)):
    scene = make_scene(dtype=dtype, steps=steps, periodic_z=periodic_z, source_kind=source_kind,
                       boundary_kind=boundary_kind, bloch_vector=bloch_vector)
    spec, design, rho = scene['spec'], scene['design'], scene['rho']
    if fabrication:
        transform = lambda r: tanh_projection(periodic_conic_filter(r, radius=1.8, spacing=1.), beta=3.)
        design = DesignGrid(design.start, design.shape, design.parameter_shape, transform=transform)
    targets, bands = targets_and_bands(scene)
    objective = MultiTDAObjective(spec, design, targets, bands, scalarize)
    native, native_value = native_reference(objective, scene)
    v, g, info = objective.evaluate(rho, return_info=True)
    vn, gn = native(rho)
    vr, gr = objective.reference_value_and_grad(rho)
    tolerance = 3e-5 if dtype == jnp.float32 else 2e-10
    row = dict(name=name, value=float(v), native_value_error=relative(v, vn),
               native_gradient_error=relative(g, gn), step_ad_gradient_error=relative(g, gr),
               steps=steps, dtype=str(dtype), info=info)
    assert row['native_value_error'] < tolerance, row
    assert row['native_gradient_error'] < tolerance, row
    assert row['step_ad_gradient_error'] < tolerance, row
    assert relative(vr, vn) < tolerance
    directions = np.random.default_rng(19).normal(size=(3, *rho.shape))
    fd_errors = []
    step_size = 2e-3 if dtype == jnp.float32 else 2e-5
    for direction in directions:
        direction /= np.linalg.norm(direction)
        direction = jnp.asarray(direction, dtype=rho.dtype)
        fd = (native_value(rho + step_size * direction) - native_value(rho - step_size * direction)) / (2 * step_size)
        predicted = jnp.sum(g * direction)
        fd_errors.append(relative(predicted, fd))
    row['finite_difference_relative_errors'] = fd_errors
    assert max(fd_errors) < (1e-2 if dtype == jnp.float32 else 3e-5), row
    vo, go, host_info = objective.evaluate(rho, cpu_offload=True, return_info=True)
    row['offload_value_error'] = relative(v, vo)
    row['offload_gradient_error'] = relative(g, go)
    assert row['offload_value_error'] < tolerance and row['offload_gradient_error'] < tolerance, row
    assert info['history_memory_kind'] == 'device' and host_info['history_memory_kind'] == 'pinned_host'
    assert info['d2h_bytes'] == info['h2d_bytes'] == 0
    assert host_info['d2h_bytes'] == host_info['h2d_bytes'] == host_info['history_bytes']
    changed = rho + .015
    vc, gc = objective.evaluate(changed)
    vcn, gcn = native(changed)
    assert float(vc) != float(v) and relative(gc, gcn) < tolerance
    row.update(offload_passed=True, changed_rho_gradient_error=relative(gc, gcn))
    print(json.dumps(row), flush=True)
    return row, objective, rho, scene


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', default='main', choices=['main', 'periodic', 'gaussian', 'dipole', 'float32', 'fabrication', 'sparse', 'pec', 'pmc', 'bloch', 'bloch32'])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    jax.config.update('jax_enable_x64', args.case not in ('float32', 'bloch32'))
    assert jax.devices()[0].platform == 'gpu'
    options = dict(periodic_z=args.case in ('periodic', 'bloch', 'bloch32'), fabrication=args.case == 'fabrication',
                   source_kind=('dipole' if args.case in ('bloch', 'bloch32') else args.case)
                     if args.case in ('gaussian', 'dipole', 'bloch', 'bloch32') else 'uniform',
                   dtype=jnp.float32 if args.case in ('float32', 'bloch32') else jnp.float64,
                   steps=512 if args.case == 'sparse' else 257,
                   boundary_kind=args.case if args.case in ('pec', 'pmc') else 'pml',
                   bloch_vector=(0., 0., 1e6) if args.case in ('bloch', 'bloch32') else (0., 0., 0.))
    row, objective, rho, scene = run_case(args.case, **options)
    if args.case == 'main':
        from scipy.optimize import minimize
        initial = float(objective.value(rho))
        result = minimize(objective.scipy_fun, np.asarray(rho).ravel(), jac=True,
                          bounds=[(0., 1.)] * rho.size, method='L-BFGS-B', options=dict(maxiter=3))
        assert result.fun < initial and result.nit > 0 and not np.array_equal(result.x, np.asarray(rho).ravel())
        row['optimization'] = dict(initial=initial, final=float(result.fun), iterations=int(result.nit),
                                   evaluations=int(result.nfev), stop_message=str(result.message))
    if args.case == 'sparse':
        rows = []
        _, dense_gradient = objective.reference_value_and_grad(rho)
        for stride in (2, 4, 8):
            sparse = MultiTDAObjective(objective.spec, objective.design, objective.targets,
                                      objective.bands, scalarize, stride=stride)
            value, gradient = sparse.evaluate(rho)
            error = relative(gradient, dense_gradient)
            assert relative(value, row['value']) < 2e-10
            assert np.isfinite(error)
            assert sparse.last_info['history_memory_kind'] == 'device'
            assert sparse.last_info['d2h_bytes'] == sparse.last_info['h2d_bytes'] == 0
            rows.append(dict(relative_gradient_error=error, **sparse.last_info))
        assert max(r['relative_gradient_error'] for r in rows[:2]) < 1e-3
        assert rows[-1]['relative_gradient_error'] > 1e-3
        row['sparse'] = rows
    verify_fdtdx_sources.cache_clear()
    assert verify_fdtdx_sources()
    output = dict(device=jax.devices()[0].device_kind, versions={n: importlib.metadata.version(n)
                  for n in ('fdtdx','jax','jaxlib','equinox','numpy')}, result=row,
                  fdtdx_sources_unchanged=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(dict(saved=str(args.output), passed=True)), flush=True)


if __name__ == '__main__':
    main()
