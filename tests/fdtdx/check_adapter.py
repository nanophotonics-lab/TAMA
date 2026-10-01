"""Runnable CPU check: common API, native numerical parity and input guards."""

import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from _scene import make_scene
import tama
from tama.backends.fdtdx import specs, targets, filters, backend


def rejected(fn, exception, phrase):
    try:
        fn()
    except exception as error:
        assert phrase in str(error), str(error)
        return
    raise AssertionError(f"Expected {exception.__name__} containing {phrase!r}")


def main():
    started = time.perf_counter()
    tm = tama.get_backend("fdtdx")
    assert "meep" not in sys.modules
    for name in ("SimulationSpec", "DesignGrid", "PointTarget", "FieldRegionTarget",
                 "FluxTarget", "EigenmodeCoefficientTarget", "Band", "AccuracyError",
                 "verify_fdtdx_sources", "periodic_conic_filter", "tanh_projection",
                 "bandpass_kernel", "power_complementary_kernels"):
        origin = next(module for module in (specs, targets, filters, backend) if hasattr(module, name))
        assert getattr(tm, name) is getattr(origin, name), name
    scene = make_scene(steps=128, periodic_z=True, backend="cpu")
    rho = np.asarray(scene["rho"])
    x = rho.ravel()
    target = tm.PointTarget(scene["target"])
    common = dict(simulation=scene["spec"], design=scene["design"], block_steps=64)
    callback = lambda h, dt: dt * 1e15 * jnp.sum(jnp.abs(h[..., 2]) ** 2)
    objective = tm.TDAObjective(**common, target=target, fom_fn=callback)
    native = objective.native_objective
    assert native.stride == 1 and native.half_width == 64
    rejected(lambda: setattr(objective, "native_objective", None), AttributeError, "no setter")
    rejected(lambda: setattr(native, "stride", 2), AttributeError, "immutable")
    rejected(lambda: tm.TDAObjective(**common, target=target, wavelength_bands=[(1., 2.)]),
             TypeError, "wavelength_bands")
    rejected(lambda: tm.TDAObjective(**common, target=target, cpu_offload="on"), TypeError, "Python bool")
    sparse = tm.TDAObjective(**common, target=target, sampling_interval=2, reconstruction_half_width=32)
    assert sparse.native_objective.stride == 2 and sparse.native_objective.half_width == 32
    for invalid in (False, np.ones(x.shape, dtype=int), np.full(x.shape, np.nan),
                    np.full(x.shape, 1.1), np.ones(x.shape, dtype=complex),
                    x.reshape(2, -1), x[:-1]):
        rejected(lambda invalid=invalid: objective.evaluate(invalid), ValueError,
                 "shape" if np.shape(invalid) not in (rho.shape, x.shape) else
                 "[0, 1]" if np.asarray(invalid).dtype.kind == "f" and np.isfinite(invalid).all() else "finite real")
    for invalid in (0, 1, None, "yes", np.bool_(True)):
        rejected(lambda invalid=invalid: objective.evaluate(x, need_gradient=invalid), TypeError, "Python bool")
    for invalid in (0, 1, "on", np.bool_(False)):
        rejected(lambda invalid=invalid: objective.evaluate(x, False, cpu_offload=invalid), TypeError, "Python bool")
    assert not native._kernels, "Invalid inputs should not construct FDTD kernels"

    value, gradient = objective(x)
    assert type(value) is float and isinstance(gradient, np.ndarray)
    assert gradient.shape == x.shape and gradient.dtype == rho.dtype
    assert np.linalg.norm(gradient) > 0 and objective.last_info["stride"] == 1
    native_value, native_gradient = native.evaluate(rho)
    np.testing.assert_allclose(value, native_value, rtol=1e-13, atol=0)
    np.testing.assert_allclose(gradient, np.asarray(native_gradient).ravel(), rtol=1e-13, atol=0)
    for evaluate in (objective.fom_and_grad, objective.scipy_fun):
        shaped_value, shaped_gradient = evaluate(rho)
        np.testing.assert_allclose(shaped_value, value, rtol=1e-13, atol=0)
        np.testing.assert_allclose(shaped_gradient, gradient, rtol=1e-13, atol=0)

    # This catches accidentally routing need_gradient=False through the adjoint.
    with patch.object(type(native), "evaluate", side_effect=AssertionError("Unexpected adjoint")):
        forward, no_gradient = objective.evaluate(x, False)
        assert no_gradient is None and objective.last_info is None
        np.testing.assert_allclose(forward, value, rtol=1e-12, atol=0)
        np.testing.assert_allclose(objective.fom(x), value, rtol=1e-12, atol=0)
        assert objective.last_info is None
    with patch.object(type(native), "value", return_value=jnp.nan):
        rejected(lambda: objective.fom(x), FloatingPointError, "Nonfinite")
    with patch.object(type(native), "reference_value_and_grad", return_value=(1., np.full(rho.shape, np.nan))):
        rejected(lambda: objective.reference_value_and_grad(x), FloatingPointError, "Nonfinite")
    reference, reference_gradient = objective.reference_value_and_grad(x)
    assert reference_gradient.shape == x.shape and objective.last_info is None
    np.testing.assert_allclose(reference, value, rtol=1e-12, atol=0)
    np.testing.assert_allclose(reference_gradient, gradient, rtol=1e-10, atol=1e-14)
    validation = objective.validate(x, directions=1)
    assert validation["passed"] and objective.last_info is None

    # Sparse evaluation must not allocate or execute a dense history run.
    run = backend.MultiTDAObjective._run
    calls = []

    def sparse_run(self, rho, inv, stride, cpu_offload):
        calls.append((stride, cpu_offload))
        assert stride == 8, "Sparse evaluation started a dense run"
        return run(self, rho, inv, stride, cpu_offload)

    for targets_arg in ({"target": target}, {"targets": [target]}):
        constructor = tm.TDAObjective if "target" in targets_arg else tm.MultiTDAObjective
        sparse = constructor(**common, **targets_arg, sampling_interval=8)
        calls.clear()
        with patch.object(backend.MultiTDAObjective, "_run", sparse_run):
            sparse_value, sparse_gradient = sparse(x)
        assert calls == [(8, False)]
        assert np.isfinite(sparse_value) and np.isfinite(sparse_gradient).all()
        assert np.linalg.norm(sparse_gradient) > 0
        assert sparse.last_info["stride"] == 8
        assert sparse.last_info["d2h_bytes"] == sparse.last_info["h2d_bytes"] == 0

    scaled = tm.TDAObjective(**common, target=target, fom_fn=callback,
                             scalarization_fn=lambda values: 2 * jnp.sum(values))
    scaled_value, scaled_gradient = scaled(x)
    np.testing.assert_allclose(scaled_value, 2 * value, rtol=1e-13, atol=0)
    np.testing.assert_allclose(scaled_gradient, 2 * gradient, rtol=1e-13, atol=0)

    bands = [tm.Band(0, kernel=np.array([.25, .5, .25]), fom_fn=callback),
             tm.Band(0, weight=.3, fom_fn=callback)]
    multi = tm.MultiTDAObjective(**common, targets=[target], bands=bands,
                                scalarization_fn=lambda values: jnp.sum(values))
    multi_value, multi_gradient = multi(x)
    expected_value, expected_gradient = multi.native_objective.evaluate(rho)
    np.testing.assert_allclose(multi_value, expected_value, rtol=1e-13, atol=0)
    np.testing.assert_allclose(multi_gradient, np.asarray(expected_gradient).ravel(), rtol=1e-13, atol=0)
    result = dict(passed=True, backend=jax.default_backend(), no_meep_import=True,
                  flat_and_shaped_designs=True, gradient_dtype=str(gradient.dtype),
                  native_api_exports_identical=True, forward_only_without_adjoint=True,
                  stale_diagnostics_cleared=True, input_and_nonfinite_guards=True,
                  native_single_and_multiband_parity=True, validation=validation,
                  elapsed_s=time.perf_counter() - started)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
