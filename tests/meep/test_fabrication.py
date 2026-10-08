import numpy as np
import pytest

import tama as tm
from tama.fabrication import _periodic_conic_transfer


@pytest.mark.parametrize(
    "shape, spacing, periodic_axes",
    [
        ((7, 6), (0.17, 0.23), (0, 1)),
        ((6, 5, 3), (0.19, 0.21, 0.15), (0, 1)),
    ],
)
def test_periodic_conic_filter_vjp_satisfies_dot_product_identity(
    shape,
    spacing,
    periodic_axes,
):
    rng = np.random.default_rng(101)
    weights = rng.standard_normal(shape)
    cotangent = rng.standard_normal(shape)
    filtered = tm.periodic_conic_filter(
        weights,
        radius=0.48,
        spacing=spacing,
        periodic_axes=periodic_axes,
    )
    pulled_back = tm.periodic_conic_filter_vjp(
        cotangent,
        radius=0.48,
        spacing=spacing,
        periodic_axes=periodic_axes,
    )

    assert np.vdot(cotangent, filtered) == pytest.approx(
        np.vdot(pulled_back, weights),
        abs=2e-13,
    )


def test_periodic_filter_wraps_seams_without_coupling_unfiltered_axis():
    impulse = np.zeros((7, 7, 3))
    impulse[0, 3, 0] = 1.0
    filtered = tm.periodic_conic_filter(
        impulse,
        radius=1.5,
        spacing=(1.0, 1.0, 1.0),
        periodic_axes=(0, 1),
    )

    assert filtered[-1, 3, 0] > 0.0
    assert filtered[1, 3, 0] == pytest.approx(filtered[-1, 3, 0])
    assert np.sum(filtered) == pytest.approx(1.0)
    assert np.count_nonzero(filtered[:, :, 1:]) == 0

    shifted = np.roll(impulse, (2, -1), axis=(0, 1))
    shifted_filtered = tm.periodic_conic_filter(
        shifted,
        radius=1.5,
        spacing=(1.0, 1.0, 1.0),
        periodic_axes=(0, 1),
    )
    np.testing.assert_allclose(
        shifted_filtered,
        np.roll(filtered, (2, -1), axis=(0, 1)),
        atol=2e-14,
    )


def test_layerwise_3d_filter_matches_independent_2d_filters():
    rng = np.random.default_rng(912)
    weights = rng.standard_normal((8, 7, 4))
    spacing = (0.16, 0.19, 0.23)
    filtered = tm.periodic_conic_filter(
        weights,
        radius=0.44,
        spacing=spacing,
        periodic_axes=(0, 1),
    )
    expected = np.stack(
        [
            tm.periodic_conic_filter(
                weights[:, :, z_index],
                radius=0.44,
                spacing=spacing[:2],
                periodic_axes=(0, 1),
            )
            for z_index in range(weights.shape[2])
        ],
        axis=2,
    )
    transfer = _periodic_conic_transfer(
        weights.shape,
        0.44,
        spacing,
        (0, 1),
    )

    assert transfer.shape == (weights.shape[0], weights.shape[1], 1)
    np.testing.assert_allclose(filtered, expected, atol=2e-14)


def test_filtered_projection_vjp_matches_directional_finite_difference():
    rng = np.random.default_rng(20260816)
    weights = rng.uniform(0.15, 0.85, (8, 7, 2))
    direction = rng.standard_normal(weights.shape)
    direction /= np.linalg.norm(direction)
    target = rng.uniform(0.1, 0.9, weights.shape)
    settings = {
        "radius": 0.41,
        "spacing": (0.16, 0.19, 0.25),
        "periodic_axes": (0, 1),
    }

    filtered = tm.periodic_conic_filter(weights, **settings)
    projected = tm.tanh_projection(filtered, beta=5.0, eta=0.47)
    projected_cotangent = projected - target
    filtered_cotangent = tm.tanh_projection_vjp(
        filtered,
        projected_cotangent,
        beta=5.0,
        eta=0.47,
    )
    gradient = tm.periodic_conic_filter_vjp(
        filtered_cotangent,
        **settings,
    )

    def objective(candidate):
        candidate = tm.periodic_conic_filter(candidate, **settings)
        candidate = tm.tanh_projection(candidate, beta=5.0, eta=0.47)
        return 0.5 * np.sum((candidate - target) ** 2)

    step = 1e-6
    finite_difference = (
        objective(weights + step * direction) - objective(weights - step * direction)
    ) / (2.0 * step)

    assert float(np.sum(gradient * direction)) == pytest.approx(
        finite_difference,
        rel=2e-9,
        abs=2e-10,
    )


@pytest.mark.parametrize("beta", (0.0, np.nextafter(0.0, 1.0)))
def test_zero_or_numerically_tiny_beta_projection_and_vjp_are_identity(beta):
    weights = np.linspace(0.1, 0.9, 12).reshape(3, 4)
    cotangent = np.linspace(-1.0, 1.0, 12).reshape(3, 4)

    np.testing.assert_array_equal(
        tm.tanh_projection(weights, beta=beta),
        weights,
    )
    np.testing.assert_array_equal(
        tm.tanh_projection_vjp(weights, cotangent, beta=beta),
        cotangent,
    )


@pytest.mark.parametrize(
    "call, message",
    [
        (
            lambda: tm.periodic_conic_filter(np.zeros(4), radius=1.0, spacing=1.0),
            "2D or 3D",
        ),
        (
            lambda: tm.periodic_conic_filter(np.zeros((2, 2)), radius=0.0, spacing=1.0),
            "radius",
        ),
        (
            lambda: tm.periodic_conic_filter(
                np.zeros((2, 2)),
                radius=1.0,
                spacing=(1.0,),
            ),
            "one entry per array axis",
        ),
        (
            lambda: tm.tanh_projection(np.zeros((2, 2)), beta=-1.0),
            "beta",
        ),
        (
            lambda: tm.tanh_projection_vjp(
                np.zeros((2, 2)), np.zeros((3, 2)), beta=1.0
            ),
            "weights.shape",
        ),
    ],
)
def test_fabrication_filter_rejects_invalid_settings(call, message):
    with pytest.raises((TypeError, ValueError), match=message):
        call()
