"""CPU checks of public API guards and actual custom-derivative rejection."""

from dataclasses import replace
from importlib import import_module
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
from tama.backends.fdtdx import Band, PointTarget
from tama.backends.fdtdx.backend import (
    AccuracyError,
    MultiTDAObjective,
    TDAObjective,
    verify_fdtdx_compatibility,
    verify_fdtdx_sources,
)
from tama.backends.fdtdx.targets import make_loss


def rejected(fn, exception, phrase):
    try:
        fn()
    except exception as error:
        assert phrase in str(error), str(error)
        return str(error)
    raise AssertionError(f"expected {exception.__name__} containing {phrase!r}")


def check_engine_compatibility():
    import fdtdx

    check = verify_fdtdx_compatibility
    assert verify_fdtdx_sources is check
    check.cache_clear()
    # Local builds need usable APIs, not identical version metadata or sources.
    with (
        patch("importlib.metadata.version", return_value="0.6.2+local"),
        patch.object(fdtdx, "__file__", None),
    ):
        assert check()
    for module_name, name in (
        ("fdtdx.fdtd.update", "update_H"),
        ("equinox.internal", "while_loop"),
    ):
        with patch.object(import_module(module_name), name, None):
            check.cache_clear()
            rejected(
                lambda: MultiTDAObjective(None, None, []),
                RuntimeError,
                f"{module_name}.{name}",
            )
    with patch(
        "tama.backends.fdtdx.backend.import_module",
        side_effect=ImportError("missing module"),
    ):
        check.cache_clear()
        rejected(check, RuntimeError, "requires module fdtdx.core.physics.curl")
    check.cache_clear()
    assert check()


def main():
    started = time.perf_counter()
    check_engine_compatibility()
    scene = make_scene(steps=128, periodic_z=True, backend="cpu")
    spec, design, rho = (scene[k] for k in ("spec", "design", "rho"))
    target = PointTarget(scene["target"])
    objective = TDAObjective(spec, design, target)

    class NonlinearTarget(PointTarget):
        def sample(self, E, H):
            return super().sample(E, H) ** 2

    rejected(
        lambda: TDAObjective(spec, design, NonlinearTarget(scene["target"])),
        TypeError,
        "Custom target sampling",
    )
    rejected(lambda: setattr(objective, "stride", 4), AttributeError, "immutable")
    rejected(
        lambda: setattr(objective, "cpu_offload", "on"), AttributeError, "immutable"
    )
    rejected(lambda: setattr(target, "component", "Hx"), AttributeError, "immutable")
    rejected(lambda: target.indices.__setitem__((0, 0), 0), ValueError, "read-only")
    rejected(
        lambda: setattr(design, "epsilon_high", 8.0), AttributeError, "cannot assign"
    )
    rejected(
        lambda: spec.objects.object_list.append(spec.objects.sources[0]),
        AttributeError,
        "append",
    )

    invalid_offload = ("on", 0, 1, np.bool_(False))
    for value in (*invalid_offload, None):
        rejected(
            lambda value=value: TDAObjective(spec, design, target, cpu_offload=value),
            TypeError,
            "Python bool",
        )
    for value in invalid_offload:
        rejected(
            lambda value=value: objective.evaluate(rho, cpu_offload=value),
            TypeError,
            "Python bool",
        )
    rejected(
        lambda: objective.evaluate(rho.reshape(-1)), ValueError, "rho must have shape"
    )
    rejected(
        lambda: objective.evaluate(jnp.ones_like(rho, dtype=jnp.int32)),
        ValueError,
        "finite real",
    )
    rejected(
        lambda: objective.evaluate(rho.at[0, 0, 0].set(jnp.nan)),
        ValueError,
        "finite real",
    )
    rejected(lambda: objective.evaluate(rho.at[0, 0, 0].set(1.1)), ValueError, "[0, 1]")
    outside = TDAObjective(spec, replace(design, transform=lambda x: x + 1), target)
    rejected(lambda: outside.evaluate(rho), ValueError, "transform must keep density")
    rejected(
        lambda: TDAObjective(spec, design, target, stride=3, block_steps=63),
        ValueError,
        "divisible by stride",
    )
    assert (
        not objective._kernels and not outside._kernels
    ), "invalid inputs reached FDTD kernel construction"

    history = (jnp.ones((4, 1, 6)),)
    for bad in (lambda h, dt: jnp.ones((2,)), lambda h, dt: jnp.asarray(1 + 1j)):
        loss = make_loss([target], [Band(0, fom_fn=bad)], spec.dt)
        rejected(lambda: loss(history), ValueError, "real floating-point scalar")
    for bad in (lambda v: v, lambda v: jnp.sum(v).astype(jnp.complex128)):
        loss = make_loss([target], [Band(0)], spec.dt, scalarization_fn=bad)
        rejected(lambda: loss(history), ValueError, "scalarization must return")

    # Both discrete adjoints agree on this deliberately wrong derivative, so AD
    # agreement alone must not certify it. Actual centered FDTD differences do.
    def wrong_derivative(history, dt):
        return jax.lax.stop_gradient(dt / 2 * jnp.sum(history[..., 2] ** 2))

    wrong = TDAObjective(spec, design, target, fom_fn=wrong_derivative)
    value = float(wrong.value(rho))
    assert (
        np.isfinite(value) and value > 0
    ), "the validation scene must have a nonzero target signal"
    fd_rejection = rejected(
        lambda: wrong.validate(rho, directions=1),
        AccuracyError,
        "Centered finite differences failed",
    )

    # Nonfinite callback derivatives must not escape as valid evaluations.
    nonfinite = MultiTDAObjective(
        spec,
        design,
        [target],
        [Band(0, fom_fn=lambda history, dt: jnp.sqrt(jnp.sum(history * 0)))],
    )
    rejected(
        lambda: nonfinite.evaluate(rho),
        FloatingPointError,
        "Nonfinite objective/gradient",
    )
    result = dict(
        passed=True,
        backend=jax.default_backend(),
        engine_api_compatibility=True,
        objective_configuration_immutable=True,
        exact_target_type_guard=True,
        input_guards_before_fdtd=True,
        callback_shape_and_dtype_guards=True,
        stopped_derivative_rejected_by_actual_fdtd_finite_differences=True,
        stopped_derivative_rejection=fd_rejection,
        tested_objective_value=value,
        nonfinite_callback_derivative_rejected=True,
        elapsed_s=time.perf_counter() - started,
    )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
