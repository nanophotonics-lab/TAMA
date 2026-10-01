"""Installed-wheel CUDA check; no Meep import or engine source edits required."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

import tama
from tama.backends.fdtdx import backend as native

from _scene import make_scene


def relative(a, b):
    a, b = np.asarray(a), np.asarray(b)
    error, scale = float(np.linalg.norm(a - b)), float(np.linalg.norm(b))
    return error / scale if scale else (0. if error == 0 else float('inf'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--expect-install', type=Path)
    args = parser.parse_args()
    if args.expect_install:
        assert Path(tama.__file__).resolve().is_relative_to(args.expect_install.resolve()), tama.__file__
        assert Path(native.__file__).resolve().is_relative_to(args.expect_install.resolve()), native.__file__
    assert 'meep' not in sys.modules
    assert tama.__version__ == importlib.metadata.version('tama')
    tm = tama.get_backend('fdtdx')
    assert tm.__name__ == 'tama.backends.fdtdx'
    assert 'tama_fdtdx' not in sys.modules
    assert tm.verify_fdtdx_sources()
    jax.config.update('jax_enable_x64', True)
    assert jax.default_backend() == 'gpu', jax.devices()
    scene = make_scene(steps=512)
    spec, design, rho = scene['spec'], scene['design'], scene['rho']
    x = np.asarray(rho).ravel()
    targets = [tm.PointTarget(scene['target'], 'Ez'), tm.PointTarget(scene['target'], 'Hy')]
    f0 = scene['frequency']
    bands = [tm.Band(0, tm.bandpass_kernel(.65*f0, 1.05*f0, spec.dt, 41), weight=1/spec.dt),
             tm.Band(0, tm.bandpass_kernel(.95*f0, 1.35*f0, spec.dt, 40), weight=.5/spec.dt),
             tm.Band(1, tm.bandpass_kernel(.95*f0, 1.35*f0, spec.dt, 40), weight=.2/spec.dt)]
    scalarize = lambda values: jnp.sum(values) + .01*jnp.sum(values**2)
    objective = tm.MultiTDAObjective(simulation=spec, design=design, targets=targets,
                                    bands=bands, scalarization_fn=scalarize)
    direct = native.MultiTDAObjective(spec, design, targets, bands, scalarize)
    value, gradient = objective.evaluate(x)
    assert type(value) is float and isinstance(gradient, np.ndarray) and gradient.shape == x.shape
    direct_value, direct_gradient = direct.evaluate(rho)
    parity = dict(value_error=relative(value, direct_value), gradient_error=relative(gradient, np.asarray(direct_gradient).ravel()))
    assert max(parity.values()) < 2e-10, parity
    device_info = objective.last_info
    value_host, gradient_host = objective.evaluate(x, cpu_offload=True)
    host_info = objective.last_info
    assert host_info['history_memory_kind'] == 'pinned_host'
    offload = dict(value_error=relative(value_host, value), gradient_error=relative(gradient_host, gradient),
                   off=device_info, on=host_info)
    assert max(offload['value_error'], offload['gradient_error']) < 2e-10, offload
    assert objective.evaluate(x, False)[1] is None and objective.last_info is None
    assert relative(objective.fom(x), value) < 2e-10
    changed = np.clip(x + .02, 0., 1.)
    assert not np.isclose(objective.fom(changed), value, rtol=1e-8, atol=0.)
    for method in (objective, objective.fom_and_grad, objective.scipy_fun):
        v, g = method(x)
        assert relative(v, value) < 2e-10 and relative(g, gradient) < 2e-10
    validation = objective.validate(x, directions=3)
    reference_value, reference_gradient = objective.reference_value_and_grad(x)
    reference = dict(value_error=relative(value, reference_value), gradient_error=relative(gradient, reference_gradient))
    assert max(reference.values()) < 2e-10, reference
    single = tm.TDAObjective(simulation=spec, design=design, target=targets[0],
                            fom_fn=lambda h, dt: jnp.sum(jnp.abs(h[..., 2])**2))
    direct_single = native.TDAObjective(spec, design, targets[0],
                                        fom_fn=lambda h, dt: jnp.sum(jnp.abs(h[..., 2])**2))
    sv, sg = single.fom_and_grad(x)
    nv, ng = direct_single.evaluate(rho)
    single_parity = dict(value_error=relative(sv, nv), gradient_error=relative(sg, np.asarray(ng).ravel()))
    assert max(single_parity.values()) < 2e-10
    sparse = []
    for stride in (2, 8):
        obj = tm.MultiTDAObjective(simulation=spec, design=design, targets=targets, bands=bands,
            scalarization_fn=scalarize, sampling_interval=stride, cpu_offload=True)
        sparse_value, sparse_gradient = obj.evaluate(x)
        error = relative(sparse_gradient, gradient)
        assert relative(sparse_value, value) < 2e-10
        assert np.isfinite(error)
        assert error < 1e-3 if stride == 2 else error > 1e-3
        assert obj.last_info['history_memory_kind'] == 'pinned_host'
        sparse.append(dict(stride=stride, relative_gradient_error=error, info=obj.last_info))
    assert 'meep' not in sys.modules
    result = dict(passed=True, tama_version=tama.__version__, tama_path=tama.__file__,
        backend_path=native.__file__, device=str(jax.devices()[0]),
        versions={name: importlib.metadata.version(name) for name in ('fdtdx', 'jax', 'jaxlib', 'numpy')},
        direct_backend_parity=parity, offload=offload, single_target_parity=single_parity,
        full_step_ad=reference, validation=validation, sparse=sparse,
        meep_imported=False, fdtdx_sources_unchanged=tm.verify_fdtdx_sources())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
