import meep as mp
import numpy as np
import pytest

from tama.flux import (
    _validate_cylindrical_surface,
    flux_component_pairs,
    power_history,
    power_history_pullback,
)


@pytest.mark.parametrize(
    ("normal", "expected"),
    [
        (
            mp.Vector3(1.0),
            ((mp.Ey, mp.Hz, 1.0), (mp.Ez, mp.Hy, -1.0)),
        ),
        (
            mp.Vector3(-1.0),
            ((mp.Ey, mp.Hz, -1.0), (mp.Ez, mp.Hy, 1.0)),
        ),
        (
            mp.Vector3(0.0, 1.0),
            ((mp.Ez, mp.Hx, 1.0), (mp.Ex, mp.Hz, -1.0)),
        ),
        (
            mp.Vector3(0.0, -1.0),
            ((mp.Ez, mp.Hx, -1.0), (mp.Ex, mp.Hz, 1.0)),
        ),
        (
            mp.Vector3(0.0, 0.0, 1.0),
            ((mp.Ex, mp.Hy, 1.0), (mp.Ey, mp.Hx, -1.0)),
        ),
        (
            mp.Vector3(0.0, 0.0, -1.0),
            ((mp.Ex, mp.Hy, -1.0), (mp.Ey, mp.Hx, 1.0)),
        ),
    ],
)
def test_flux_component_pairs_follow_oriented_cross_product(
    normal,
    expected,
):
    assert flux_component_pairs(normal) == expected


@pytest.mark.parametrize(
    ("normal", "expected"),
    [
        (
            mp.Vector3(1.0),
            ((mp.Ep, mp.Hz, 1.0), (mp.Ez, mp.Hp, -1.0)),
        ),
        (
            mp.Vector3(-1.0),
            ((mp.Ep, mp.Hz, -1.0), (mp.Ez, mp.Hp, 1.0)),
        ),
        (
            mp.Vector3(0.0, 0.0, 1.0),
            ((mp.Er, mp.Hp, 1.0), (mp.Ep, mp.Hr, -1.0)),
        ),
        (
            mp.Vector3(0.0, 0.0, -1.0),
            ((mp.Er, mp.Hp, -1.0), (mp.Ep, mp.Hr, 1.0)),
        ),
    ],
)
def test_cylindrical_flux_component_pairs_follow_oriented_cross_product(
    normal,
    expected,
):
    assert flux_component_pairs(normal, cylindrical=True) == expected


def test_cylindrical_flux_rejects_azimuthal_normal():
    with pytest.raises(ValueError, match="radial or axial"):
        flux_component_pairs(
            mp.Vector3(0.0, 1.0),
            cylindrical=True,
        )


def test_cylindrical_surface_geometry_matches_physical_quadrature():
    _validate_cylindrical_surface(
        (mp.Vector3(0.0, 0.0, 0.4), mp.Vector3(0.5, 0.0, 0.4)),
        mp.Vector3(0.0, 0.0, 1.0),
        (0.0, np.pi),
    )
    _validate_cylindrical_surface(
        (mp.Vector3(0.7, 0.0, -0.2), mp.Vector3(0.7, 0.0, 0.3)),
        mp.Vector3(1.0),
        (0.4, 0.6),
    )

    with pytest.raises(ValueError, match="zero weight at r=0"):
        _validate_cylindrical_surface(
            (mp.Vector3(0.0, 0.0, 0.4), mp.Vector3(0.5, 0.0, 0.4)),
            mp.Vector3(0.0, 0.0, 1.0),
            (0.1, 0.9),
        )
    with pytest.raises(ValueError, match="positive radius"):
        _validate_cylindrical_surface(
            (mp.Vector3(0.0, 0.0, -0.2), mp.Vector3(0.0, 0.0, 0.3)),
            mp.Vector3(1.0),
            (0.4, 0.6),
        )
    with pytest.raises(ValueError, match="constant radius"):
        _validate_cylindrical_surface(
            (mp.Vector3(0.7, 0.0, -0.2), mp.Vector3(0.8, 0.0, 0.3)),
            mp.Vector3(1.0),
            (0.4, 0.6),
        )


def test_power_history_and_pullback_center_magnetic_samples_exactly():
    histories = {
        mp.Ez: np.asarray([2.0, 2.0, 2.0]),
        mp.Hy: np.asarray([1.0, 3.0, 5.0]),
    }
    covector = np.asarray([10.0, 20.0])

    np.testing.assert_allclose(
        power_history(histories, mp.Vector3(1.0)),
        [-4.0, -8.0],
    )
    pullback = power_history_pullback(
        histories,
        mp.Vector3(1.0),
        None,
        covector,
    )

    np.testing.assert_allclose(pullback[mp.Ez], [-20.0, -80.0, 0.0])
    np.testing.assert_allclose(pullback[mp.Hy], [-10.0, -30.0, -20.0])


def test_magnetic_time_centering_matches_meep_synchronization():
    mp.verbosity(0)
    simulation = mp.Simulation(
        cell_size=mp.Vector3(4.0, 4.0),
        resolution=10,
        sources=[
            mp.Source(
                mp.GaussianSource(frequency=1.0, fwidth=0.5),
                component=mp.Ez,
                center=mp.Vector3(-0.5, 0.0),
            )
        ],
    )
    position = mp.Vector3(0.0, 0.2)

    try:
        simulation.run(until=4.0)
        magnetic_before = simulation.get_field_point(mp.Hx, position)

        simulation.fields.synchronize_magnetic_fields()
        magnetic_synchronized = simulation.get_field_point(mp.Hx, position)
        simulation.fields.restore_magnetic_fields()

        magnetic_restored = simulation.get_field_point(mp.Hx, position)
        simulation.fields.step()
        magnetic_after = simulation.get_field_point(mp.Hx, position)
    finally:
        simulation.reset_meep()

    assert abs(magnetic_after - magnetic_before) > 1.0e-9
    np.testing.assert_array_equal(magnetic_restored, magnetic_before)
    np.testing.assert_allclose(
        magnetic_synchronized,
        0.5 * (magnetic_before + magnetic_after),
        rtol=1.0e-13,
        atol=1.0e-15,
    )


@pytest.mark.parametrize(
    "normal",
    [
        mp.Vector3(1.0),
        mp.Vector3(-1.0),
        mp.Vector3(0.0, 1.0),
        mp.Vector3(0.0, -1.0),
        mp.Vector3(0.0, 0.0, 1.0),
        mp.Vector3(0.0, 0.0, -1.0),
    ],
)
def test_power_history_pullback_matches_complex_directional_fd(normal):
    rng = np.random.default_rng(20260729)
    n_samples = 13
    n_points = 5
    weights = rng.uniform(0.1, 1.0, n_points)
    histories = {}
    for electric, magnetic, _ in flux_component_pairs(normal):
        for component in (electric, magnetic):
            histories[component] = (
                rng.standard_normal((n_samples, n_points))
                + 1j * rng.standard_normal((n_samples, n_points))
            )
    directions = {
        component: (
            rng.standard_normal(history.shape)
            + 1j * rng.standard_normal(history.shape)
        )
        for component, history in histories.items()
    }
    power_covector = rng.standard_normal(n_samples - 1)

    pullback = power_history_pullback(
        histories,
        normal,
        weights,
        power_covector,
    )
    predicted = np.real(
        sum(
            np.sum(pullback[component] * directions[component])
            for component in histories
        )
    )

    step = 1.0e-6
    plus = {
        component: history + step * directions[component]
        for component, history in histories.items()
    }
    minus = {
        component: history - step * directions[component]
        for component, history in histories.items()
    }
    finite_difference = (
        np.dot(power_covector, power_history(plus, normal, weights))
        - np.dot(power_covector, power_history(minus, normal, weights))
    ) / (2.0 * step)

    np.testing.assert_allclose(
        predicted,
        finite_difference,
        rtol=1.0e-8,
        atol=1.0e-9,
    )


@pytest.mark.parametrize(
    "normal",
    [mp.Vector3(1.0), mp.Vector3(0.0, 0.0, -1.0)],
)
def test_cylindrical_power_pullback_matches_complex_directional_fd(normal):
    rng = np.random.default_rng(20260805)
    n_samples = 9
    n_points = 4
    weights = rng.uniform(0.1, 1.0, n_points)
    histories = {}
    for electric, magnetic, _ in flux_component_pairs(
        normal,
        cylindrical=True,
    ):
        for component in (electric, magnetic):
            histories[component] = (
                rng.standard_normal((n_samples, n_points))
                + 1j * rng.standard_normal((n_samples, n_points))
            )
    directions = {
        component: (
            rng.standard_normal(history.shape)
            + 1j * rng.standard_normal(history.shape)
        )
        for component, history in histories.items()
    }
    power_covector = rng.standard_normal(n_samples - 1)

    pullback = power_history_pullback(
        histories,
        normal,
        weights,
        power_covector,
        cylindrical=True,
    )
    predicted = np.real(
        sum(
            np.sum(pullback[component] * directions[component])
            for component in histories
        )
    )
    step = 1.0e-6
    plus = {
        component: history + step * directions[component]
        for component, history in histories.items()
    }
    minus = {
        component: history - step * directions[component]
        for component, history in histories.items()
    }
    finite_difference = (
        np.dot(
            power_covector,
            power_history(
                plus,
                normal,
                weights,
                cylindrical=True,
            ),
        )
        - np.dot(
            power_covector,
            power_history(
                minus,
                normal,
                weights,
                cylindrical=True,
            ),
        )
    ) / (2.0 * step)

    np.testing.assert_allclose(
        predicted,
        finite_difference,
        rtol=1.0e-8,
        atol=1.0e-9,
    )


def test_power_history_pullback_satisfies_real_dot_product_identity():
    rng = np.random.default_rng(20260730)
    normal = mp.Vector3(0.0, 0.0, 1.0)
    n_samples = 9
    n_points = 4
    weights = rng.uniform(0.2, 1.2, n_points)
    histories = {}
    directions = {}
    for electric, magnetic, _ in flux_component_pairs(normal):
        for component in (electric, magnetic):
            histories[component] = (
                rng.standard_normal((n_samples, n_points))
                + 1j * rng.standard_normal((n_samples, n_points))
            )
            directions[component] = (
                rng.standard_normal((n_samples, n_points))
                + 1j * rng.standard_normal((n_samples, n_points))
            )
    power_covector = rng.standard_normal(n_samples - 1)

    tangent_power = np.zeros(n_samples - 1)
    for electric, magnetic, sign in flux_component_pairs(normal):
        electric_history = histories[electric][:-1]
        electric_direction = directions[electric][:-1]
        magnetic_history = 0.5 * (
            histories[magnetic][:-1] + histories[magnetic][1:]
        )
        magnetic_direction = 0.5 * (
            directions[magnetic][:-1] + directions[magnetic][1:]
        )
        tangent_power += sign * np.sum(
            weights
            * np.real(
                np.conjugate(electric_direction) * magnetic_history
                + np.conjugate(electric_history) * magnetic_direction
            ),
            axis=1,
        )

    pullback = power_history_pullback(
        histories,
        normal,
        weights,
        power_covector,
    )
    forward_pairing = np.dot(power_covector, tangent_power)
    transpose_pairing = np.real(
        sum(
            np.sum(pullback[component] * directions[component])
            for component in histories
        )
    )

    np.testing.assert_allclose(
        transpose_pairing,
        forward_pairing,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_power_history_accepts_one_complete_2d_component_pair():
    histories = {
        mp.Ez: np.asarray([1.0 + 2.0j, 3.0 - 1.0j, 2.0 + 0.5j]),
        mp.Hy: np.asarray([2.0 - 1.0j, 1.0 + 4.0j, -1.0 + 2.0j]),
    }

    result = power_history(
        histories,
        mp.Vector3(-1.0),
        spatial_weights=[2.0],
    )

    centered_h = 0.5 * (histories[mp.Hy][:-1] + histories[mp.Hy][1:])
    expected = 2.0 * np.real(
        np.conjugate(histories[mp.Ez][:-1]) * centered_h
    )
    np.testing.assert_allclose(result, expected)


def test_power_history_rejects_incomplete_pairs_and_inconsistent_shapes():
    with pytest.raises(ValueError, match="both components"):
        power_history({mp.Ex: np.ones((3, 2))}, mp.Vector3(0.0, 0.0, 1.0))

    with pytest.raises(ValueError, match="same shape"):
        power_history(
            {
                mp.Ex: np.ones((3, 2)),
                mp.Hy: np.ones((4, 2)),
            },
            mp.Vector3(0.0, 0.0, 1.0),
        )


def test_power_history_pullback_rejects_nonreal_or_wrong_shape_covector():
    histories = {
        mp.Ex: np.ones((4, 2)),
        mp.Hy: np.ones((4, 2)),
    }
    with pytest.raises(ValueError, match="power history shape"):
        power_history_pullback(
            histories,
            mp.Vector3(0.0, 0.0, 1.0),
            None,
            np.ones(4),
        )
    with pytest.raises(ValueError, match="must be real"):
        power_history_pullback(
            histories,
            mp.Vector3(0.0, 0.0, 1.0),
            None,
            np.asarray([1.0, 2.0j, 3.0]),
        )
