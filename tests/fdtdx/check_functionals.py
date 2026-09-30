"""CPU numerical checks for target, temporal-filter and fabrication pullbacks.

Run with JAX_PLATFORMS=cpu. No GPU or FDTDX execution is needed here.
"""

from pathlib import Path
import sys
import json

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
from tama.backends.fdtdx.targets import (PointTarget, FieldRegionTarget, FluxTarget,
    EigenmodeCoefficientTarget, Band, make_loss)
from tama.backends.fdtdx.filters import (bandpass_kernel, power_complementary_kernels,
    periodic_conic_filter, tanh_projection)


def check_directional(fn, x, direction, tolerance=2e-8):
    """Central differences detect incorrect component, sign and filter transpose."""
    step = 1e-5
    expected = (float(fn(x + step * direction)) - float(fn(x - step * direction))) / (2 * step)
    # JAX complex gradients are bilinear covectors, with no conjugation here.
    actual = float(jnp.real(jnp.sum(jax.grad(fn)(x) * direction)))
    error = abs(actual - expected) / max(1.0, abs(actual), abs(expected))
    assert error < tolerance, (actual, expected, error)
    return error


def must_reject(fn):
    try:
        fn()
    except (ValueError, TypeError):
        return
    raise AssertionError("Invalid configuration was accepted")


def main():
    class NonlinearTarget(PointTarget):
        def sample(self, E, H):
            return super().sample(E, H) ** 2

    rng = np.random.default_rng(107)
    indices = np.array([[2, 1, 1], [2, 2, 1]])
    point = PointTarget((2, 1, 1), "Ez")
    region = FieldRegionTarget(indices, "Hy", weights=[0.2, 0.8])
    flux = FluxTarget(indices, normal=(1, 0, 0), weights=[0.2, 0.8])
    reverse_flux = FluxTarget(indices, normal=(-1, 0, 0), weights=[0.2, 0.8])
    reference_E = np.array([[0., 2., 0.], [0., 3., 0.]])
    reference_H = np.array([[0., 0., 1.], [0., 0., 1.5]])
    mode = EigenmodeCoefficientTarget(indices, normal=(1, 0, 0), weights=[0.2, 0.8],
                                     reference_E=reference_E, reference_H=reference_H)
    fields = jnp.asarray(rng.normal(size=(6, 4, 4, 3)))
    np.testing.assert_array_equal(point.sample(fields[:3], fields[3:])[0], np.asarray(fields)[:, 2, 1, 1])
    for target in (point, region, flux, mode):
        target.validate_shape((4, 4, 3))
        must_reject(lambda: target.validate_shape((2, 4, 3)))
        # Reassigning configuration after JIT capture must not silently change
        # Python metadata while the compiled objective retains previous values.
        try:
            target.weights = np.ones(len(target.indices))
        except AttributeError:
            pass
        else:
            raise AssertionError("Target configuration can be reassigned after construction")

    dt = 0.2
    history = jnp.asarray(rng.normal(size=(9, 2, 6)))
    direction = jnp.asarray(rng.normal(size=history.shape))
    expected_region = 0.5 * dt * np.sum(np.asarray(history)[..., 4] ** 2 * [0.2, 0.8])
    np.testing.assert_allclose(region.functional(history, dt), expected_region, rtol=2e-15)
    expected_flux = dt * np.sum((np.asarray(history)[..., 1] * np.asarray(history)[..., 5]
                                - np.asarray(history)[..., 2] * np.asarray(history)[..., 4]) * [0.2, 0.8])
    np.testing.assert_allclose(flux.functional(history, dt), expected_flux, rtol=2e-15)
    np.testing.assert_allclose(reverse_flux.functional(history, dt), -expected_flux, rtol=2e-15)
    self_mode = jnp.asarray(np.concatenate((reference_E, reference_H), axis=-1)[None])
    np.testing.assert_allclose(mode.coefficient(self_mode), [1.0], atol=2e-15)
    backward_mode = self_mode.at[..., 3:].multiply(-1)
    np.testing.assert_allclose(mode.coefficient(backward_mode), [0.0], atol=2e-15)
    np.testing.assert_allclose(mode.functional(self_mode, dt), dt, atol=2e-15)
    complex_mode = EigenmodeCoefficientTarget(indices, normal=(1, 0, 0), weights=[0.2, 0.8],
        reference_E=reference_E * np.exp(0.4j), reference_H=reference_H * np.exp(0.4j))
    np.testing.assert_allclose(complex_mode.coefficient(self_mode), [np.exp(-0.4j)], atol=2e-15)
    errors = {name: check_directional(lambda x: target.functional(x, dt), history, direction)
              for name, target in (("region", region), ("flux", flux), ("fixed_mode", complex_mode))}

    # Literal full convolution [2, 3] * [1, 2] = [2, 7, 6]. The tail is required.
    short = jnp.zeros((2, 1, 6)).at[:, 0, 2].set(jnp.array([2., 3.]))
    loss = make_loss([point], [Band(0, kernel=[1., 2.], weight=3.)], dt=0.5)
    np.testing.assert_allclose(loss((short,)), 16.6875, atol=2e-15)
    identity = make_loss([point], [Band(0)], dt=0.5)
    np.testing.assert_allclose(identity((short,)), 3.25, atol=2e-15)
    bands = [Band(0, kernel=[0.4, 1.1, -0.2], weight=0.6),
             Band(0, kernel=[-0.3, 0.5], weight=1.4), Band(1, weight=0.7)]
    combined = make_loss([flux, complex_mode], bands, dt,
                         scalarization_fn=lambda values: jnp.sum(values) + 0.01 * jnp.prod(values))
    errors["multiband_filter_scalarization"] = check_directional(lambda x: combined((x, x)), history, direction)
    callback = make_loss([region], [Band(0, fom_fn=lambda h, step: step * jnp.sum(h[..., 4] ** 4))], dt)
    errors["custom_callback"] = check_directional(lambda x: callback((x,)), history, direction)
    complex_history = history + 1j * jnp.asarray(rng.normal(size=history.shape))
    complex_direction = direction + 1j * jnp.asarray(rng.normal(size=history.shape))
    complex_loss = make_loss([region, flux, complex_mode],
        [Band(0, kernel=[0.4 + 0.1j, -0.2j]), Band(1), Band(2)], dt)
    errors["complex_functionals_filter"] = check_directional(
        lambda x: complex_loss((x, x, x)), complex_history, complex_direction)

    kernel = bandpass_kernel(0.12, 0.24, 1.0, 61)
    response = np.sum(kernel * np.exp(-2j * np.pi * 0.18 * np.arange(61)))
    np.testing.assert_allclose(abs(response), 1.0, atol=2e-15)
    bank = power_complementary_kernels([(0.10, 0.25), (0.25, 0.40)], 1.0, 257, 0.05)
    frequencies = np.linspace(0.10, 0.40, 1001)
    responses = np.exp(-2j * np.pi * frequencies[:, None] * np.arange(257)) @ np.asarray(bank).T
    bank_error = float(np.max(np.abs(np.sum(np.abs(responses) ** 2, axis=1) - 1)))
    assert bank_error < 0.015, bank_error
    # A three-tap bank cannot resolve these narrow transitions; reject rather
    # than silently claiming power complementarity for its actual FIR response.
    must_reject(lambda: power_complementary_kernels([(0.10, 0.25), (0.25, 0.40)], 1.0, 3, 0.05))
    must_reject(lambda: power_complementary_kernels([(0.10, 0.25), (0.25, 0.40)], 1.0, 257, 0.05,
                                                   max_power_error=1e-5))

    density = jnp.asarray(rng.uniform(0.2, 0.8, size=(5, 4, 3)))
    gradient_direction = jnp.asarray(rng.normal(size=density.shape))
    filtered = periodic_conic_filter(density, radius=1.5, spacing=(1., 1., 2.), periodic_axes=(0, 1))
    # Radius 1.5: center weight 1, four axial neighbors 1/3, diagonals 1-sqrt(2)/1.5.
    expected = np.asarray(density).copy()
    normalization = 1.0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == dy == 0:
                continue
            weight = max(0., 1 - np.hypot(dx, dy) / 1.5)
            expected += weight * np.roll(np.roll(np.asarray(density), dx, 0), dy, 1)
            normalization += weight
    np.testing.assert_allclose(filtered, expected / normalization, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(periodic_conic_filter(jnp.ones_like(density), radius=2., spacing=1.), 1., atol=2e-15)
    np.testing.assert_array_equal(tanh_projection(density, beta=0.), density)
    np.testing.assert_allclose(tanh_projection(jnp.array([0., 1.]), beta=8.), [0., 1.], atol=2e-15)
    fabrication = lambda x: jnp.sum(jnp.sin(tanh_projection(periodic_conic_filter(
        x, radius=1.5, spacing=1., periodic_axes=(0, 1)), beta=4., eta=0.45)))
    errors["fabrication"] = check_directional(fabrication, density, gradient_direction)

    invalid = [lambda: PointTarget((0.5, 1, 1)), lambda: PointTarget((True, 1, 1)),
        lambda: PointTarget((-1, 1, 1)), lambda: PointTarget((1, 1, 1), "foo"),
        lambda: FieldRegionTarget(indices, "Ez", [0., 0.]),
        lambda: FluxTarget(indices, (0, 1, 0), [1., 1.]),
        lambda: FluxTarget(indices, (1, 1, 0), [1., 1.]),
        lambda: EigenmodeCoefficientTarget(indices, (1, 0, 0), [1., 1.], reference_E, -reference_H),
        lambda: Band(True), lambda: Band(0, kernel=[]), lambda: Band(0, weight=np.nan),
        lambda: make_loss([point], [Band(1)], dt), lambda: make_loss([point], [], dt),
        lambda: make_loss([NonlinearTarget((1, 1, 1))], [Band(0)], dt),
        lambda: bandpass_kernel(.1, .5, 1., 11), lambda: bandpass_kernel(.1, .2, 1., True),
        lambda: power_complementary_kernels([(.1, .2), (.3, .4)], 1., 257, .05),
        lambda: periodic_conic_filter(density, radius=0., spacing=1.),
        lambda: periodic_conic_filter(density, radius=1., spacing=1., periodic_axes=(0, 0)),
        lambda: tanh_projection(density, beta=-1.)]
    for fn in invalid:
        must_reject(fn)
    print(json.dumps({"passed": True, "backend": jax.default_backend(),
                      "directional_relative_errors": errors,
                      "power_complementary_max_ripple": bank_error,
                      "power_complementary_gate_rejections": 2,
                      "immutable_target_types": 4,
                      "invalid_configurations_rejected": len(invalid) + 2}, indent=2))


if __name__ == "__main__":
    main()
