"""CPU checks of compact observations and sparse history boundaries."""

import argparse
import gc
import json
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from _scene import make_scene
from tama.backends.fdtdx import Band, FieldRegionTarget, PointTarget
from tama.backends.fdtdx.backend import MultiTDAObjective


def check_trace_lifetime(objective, rho):
    kernels = objective._get_kernels(2)
    forward, reverse = kernels["forward"], kernels["reverse"]
    traces = []

    def track_forward(*args):
        state, saved, observed = forward(*args)
        traces.extend(weakref.ref(trace) for trace in observed)
        return state, saved, observed

    def check_reverse(*args):
        gc.collect()
        assert all(
            trace() is None for trace in traces
        ), "forward block traces remain live"
        return reverse(*args)

    kernels.update(forward=track_forward, reverse=check_reverse)
    try:
        return objective.evaluate(rho, return_info=True)
    finally:
        kernels.update(forward=forward, reverse=reverse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("float64", "float32", "complex128", "complex64"),
        default="float64",
    )
    case = parser.parse_args().case
    single = case in ("float32", "complex64")
    complex_fields = case.startswith("complex")
    jax.config.update("jax_enable_x64", not single)
    scene = make_scene(
        dtype=jnp.float32 if single else jnp.float64,
        steps=258,
        periodic_z=True,
        backend="cpu",
        source_kind="dipole" if complex_fields else "uniform",
        bloch_vector=(0.0, 0.0, 1e6) if complex_fields else (0.0, 0.0, 0.0),
    )
    spec, design, rho = (scene[name] for name in ("spec", "design", "rho"))
    index = scene["target"]
    targets = [
        PointTarget(index, "Ez"),
        FieldRegionTarget(
            [index, (index[0], index[1] + 1, index[2])], "Hy", [0.3, 0.7]
        ),
    ]
    bands = [
        Band(0, kernel=[0.25, 0.5, 0.25], weight=1 / spec.dt),
        Band(1, kernel=[1.0, 0.3j], weight=0.2 / spec.dt),
    ]
    scalarize = lambda values: jnp.sum(values) + 0.01 * jnp.sum(values**2)
    compact = MultiTDAObjective(spec, design, targets, bands, scalarize, stride=2)
    full = MultiTDAObjective(
        spec,
        design,
        targets,
        [
            Band(b.target, b.kernel, b.weight, targets[b.target].functional)
            for b in bands
        ],
        scalarize,
        stride=2,
    )
    value, gradient, info = compact.evaluate(rho, return_info=True)
    expected_bytes = 258 * 3 * spec.arrays.fields.E.dtype.itemsize
    assert info["target_history_bytes"] == expected_bytes, info
    full_value, full_gradient, full_info = full.evaluate(rho, return_info=True)
    tolerance = 3e-5 if single else 2e-10
    np.testing.assert_allclose(value, full_value, rtol=tolerance, atol=0)
    np.testing.assert_allclose(
        gradient,
        full_gradient,
        rtol=tolerance,
        atol=tolerance * np.linalg.norm(full_gradient),
    )
    assert full_info["target_history_bytes"] == 6 * expected_bytes
    regrouped = MultiTDAObjective(
        spec, design, targets, bands, scalarize, stride=2, block_steps=32
    )
    regrouped_value, regrouped_gradient = regrouped.evaluate(rho)
    np.testing.assert_allclose(regrouped_value, value, rtol=tolerance, atol=0)
    np.testing.assert_allclose(
        regrouped_gradient,
        gradient,
        rtol=tolerance,
        atol=tolerance * np.linalg.norm(gradient),
    )

    # A custom band on a compact-capable target still receives all six fields.
    def callback(history, dt):
        assert history.shape == (260, 1, 6)
        return dt / 2 * jnp.sum(jnp.abs(history[..., 2] + 0.2 * history[..., 4]) ** 2)

    mixed_bands = [bands[0], Band(0, bands[0].kernel, 1 / spec.dt, callback)]
    mixed = MultiTDAObjective(spec, design, targets[:1], mixed_bands, stride=2)
    _, mixed_gradient, mixed_info = mixed.evaluate(rho, return_info=True)
    assert np.isfinite(np.asarray(mixed_gradient)).all()
    assert (
        mixed_info["target_history_bytes"]
        == 258 * 6 * spec.arrays.fields.E.dtype.itemsize
    )
    fields = spec.arrays.fields
    assert targets[0].sample(fields.E, fields.H).shape == (1, 6)

    repeated_value, repeated_gradient, _ = check_trace_lifetime(compact, rho)
    np.testing.assert_allclose(repeated_value, value, rtol=tolerance, atol=0)
    np.testing.assert_allclose(repeated_gradient, gradient, rtol=tolerance, atol=0)
    print(
        json.dumps(
            dict(
                passed=True,
                case=case,
                compact_target_history_bytes=info["target_history_bytes"],
                full_target_history_bytes=full_info["target_history_bytes"],
                short_final_chunk=True,
            )
        )
    )


if __name__ == "__main__":
    main()
