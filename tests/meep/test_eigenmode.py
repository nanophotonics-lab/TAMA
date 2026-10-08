import meep as mp
import numpy as np
import pytest

from tama.eigenmode import (
    coefficient_history,
    coefficient_history_pullback,
    mode_power,
    sample_reference_mode,
)
from tama.flux import flux_component_pairs
from tama.specs import EigenmodeCoefficientTarget


def _positive_power_mode(normal, rng, n_points):
    fields = {}
    for electric, magnetic, sign in flux_component_pairs(normal):
        electric_field = rng.standard_normal(n_points) + 1j * rng.standard_normal(
            n_points
        )
        fields[electric] = electric_field
        fields[magnetic] = sign * electric_field
    return fields


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
def test_self_mode_coefficient_squared_equals_modal_power(normal):
    rng = np.random.default_rng(20260729)
    n_samples = 7
    n_points = 5
    weights = rng.uniform(0.1, 1.0, n_points)
    mode_fields = _positive_power_mode(normal, rng, n_points)
    histories = {
        component: np.broadcast_to(
            mode_field,
            (n_samples, n_points),
        ).copy()
        for component, mode_field in mode_fields.items()
    }

    coefficient = coefficient_history(
        histories,
        mode_fields,
        normal,
        weights,
    )
    power = mode_power(mode_fields, normal, weights)

    np.testing.assert_allclose(
        np.abs(coefficient) ** 2,
        power,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


@pytest.mark.parametrize(
    "normal",
    [mp.Vector3(-1.0), mp.Vector3(0.0, 0.0, 1.0)],
)
def test_cylindrical_self_mode_coefficient_squared_equals_modal_power(
    normal,
):
    rng = np.random.default_rng(20260805)
    n_samples = 7
    n_points = 5
    weights = rng.uniform(0.1, 1.0, n_points)
    mode_fields = {}
    for electric, magnetic, sign in flux_component_pairs(
        normal,
        cylindrical=True,
    ):
        electric_field = rng.standard_normal(n_points) + 1j * rng.standard_normal(
            n_points
        )
        mode_fields[electric] = electric_field
        mode_fields[magnetic] = sign * electric_field
    histories = {
        component: np.broadcast_to(
            mode_field,
            (n_samples, n_points),
        ).copy()
        for component, mode_field in mode_fields.items()
    }

    coefficient = coefficient_history(
        histories,
        mode_fields,
        normal,
        weights,
        cylindrical=True,
    )
    power = mode_power(
        mode_fields,
        normal,
        weights,
        cylindrical=True,
    )

    np.testing.assert_allclose(
        np.abs(coefficient) ** 2,
        power,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


@pytest.mark.parametrize(
    "normal",
    [mp.Vector3(1.0), mp.Vector3(0.0, 0.0, -1.0)],
)
def test_cylindrical_coefficient_pullback_matches_directional_fd(normal):
    rng = np.random.default_rng(20260806)
    n_samples = 9
    n_points = 4
    weights = rng.uniform(0.1, 1.0, n_points)
    mode_fields = {}
    for electric, magnetic, sign in flux_component_pairs(
        normal,
        cylindrical=True,
    ):
        electric_field = rng.standard_normal(n_points) + 1j * rng.standard_normal(
            n_points
        )
        mode_fields[electric] = electric_field
        mode_fields[magnetic] = sign * electric_field
    histories = {
        component: (
            rng.standard_normal((n_samples, n_points))
            + 1j * rng.standard_normal((n_samples, n_points))
        )
        for component in mode_fields
    }
    directions = {
        component: (
            rng.standard_normal(history.shape) + 1j * rng.standard_normal(history.shape)
        )
        for component, history in histories.items()
    }
    coefficient_covector = rng.standard_normal(
        n_samples - 1
    ) + 1j * rng.standard_normal(n_samples - 1)

    pullback = coefficient_history_pullback(
        histories,
        mode_fields,
        normal,
        weights,
        coefficient_covector,
        cylindrical=True,
    )
    predicted = np.real(
        sum(
            np.sum(pullback[component] * directions[component])
            for component in histories
        )
    )

    def objective(values):
        return np.real(
            np.sum(
                coefficient_covector
                * coefficient_history(
                    values,
                    mode_fields,
                    normal,
                    weights,
                    cylindrical=True,
                )
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
    finite_difference = (objective(plus) - objective(minus)) / (2.0 * step)

    np.testing.assert_allclose(
        predicted,
        finite_difference,
        rtol=2.0e-8,
        atol=2.0e-9,
    )


def test_coefficient_history_centers_magnetic_samples():
    normal = mp.Vector3(0.0, 0.0, 1.0)
    weights = np.asarray([3.0])
    mode_fields = {
        mp.Ex: np.asarray([1.0 + 1.0j]),
        mp.Hy: np.asarray([2.0 - 0.5j]),
    }
    histories = {
        mp.Ex: np.asarray([2.0 + 0.5j, -1.0 + 3.0j, 4.0 - 2.0j]),
        mp.Hy: np.asarray([1.0 - 2.0j, 3.0 + 1.0j, -2.0 + 4.0j]),
    }

    result = coefficient_history(
        histories,
        mode_fields,
        normal,
        weights,
    )

    power = mode_power(mode_fields, normal, weights)
    centered_h = 0.5 * (histories[mp.Hy][:-1] + histories[mp.Hy][1:])
    expected = (
        0.5
        / np.sqrt(abs(power))
        * weights[0]
        * (
            histories[mp.Ex][:-1] * np.conjugate(mode_fields[mp.Hy][0])
            + np.conjugate(mode_fields[mp.Ex][0]) * centered_h
        )
    )
    np.testing.assert_allclose(result, expected)


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
def test_coefficient_pullback_matches_complex_directional_fd(normal):
    rng = np.random.default_rng(20260730)
    n_samples = 11
    n_points = 4
    weights = rng.uniform(0.2, 1.2, n_points)
    mode_fields = _positive_power_mode(normal, rng, n_points)
    histories = {
        component: (
            rng.standard_normal((n_samples, n_points))
            + 1j * rng.standard_normal((n_samples, n_points))
        )
        for component in mode_fields
    }
    directions = {
        component: (
            rng.standard_normal(history.shape) + 1j * rng.standard_normal(history.shape)
        )
        for component, history in histories.items()
    }
    coefficient_covector = rng.standard_normal(
        n_samples - 1
    ) + 1j * rng.standard_normal(n_samples - 1)

    pullback = coefficient_history_pullback(
        histories,
        mode_fields,
        normal,
        weights,
        coefficient_covector,
    )
    predicted = np.real(
        sum(
            np.sum(pullback[component] * directions[component])
            for component in histories
        )
    )

    def objective(values):
        return np.real(
            np.sum(
                coefficient_covector
                * coefficient_history(
                    values,
                    mode_fields,
                    normal,
                    weights,
                )
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
    finite_difference = (objective(plus) - objective(minus)) / (2.0 * step)

    np.testing.assert_allclose(
        predicted,
        finite_difference,
        rtol=2.0e-8,
        atol=2.0e-9,
    )


def test_coefficient_pullback_satisfies_real_dot_product_identity():
    rng = np.random.default_rng(20260731)
    normal = mp.Vector3(0.0, 1.0)
    n_samples = 8
    n_points = 3
    weights = rng.uniform(0.1, 1.0, n_points)
    mode_fields = _positive_power_mode(normal, rng, n_points)
    histories = {
        component: (
            rng.standard_normal((n_samples, n_points))
            + 1j * rng.standard_normal((n_samples, n_points))
        )
        for component in mode_fields
    }
    directions = {
        component: (
            rng.standard_normal(history.shape) + 1j * rng.standard_normal(history.shape)
        )
        for component, history in histories.items()
    }
    coefficient_covector = rng.standard_normal(
        n_samples - 1
    ) + 1j * rng.standard_normal(n_samples - 1)

    tangent = coefficient_history(
        directions,
        mode_fields,
        normal,
        weights,
    )
    pullback = coefficient_history_pullback(
        histories,
        mode_fields,
        normal,
        weights,
        coefficient_covector,
    )
    forward_pairing = np.real(np.sum(coefficient_covector * tangent))
    transpose_pairing = np.real(
        sum(
            np.sum(pullback[component] * directions[component])
            for component in histories
        )
    )

    np.testing.assert_allclose(
        transpose_pairing,
        forward_pairing,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_coefficient_history_preserves_one_point_history_shape():
    mode_fields = {
        mp.Ez: np.asarray([2.0 + 1.0j]),
        mp.Hy: np.asarray([-2.0 - 1.0j]),
    }
    histories = {
        component: np.repeat(mode_field, 5)
        for component, mode_field in mode_fields.items()
    }

    coefficient = coefficient_history(
        histories,
        mode_fields,
        mp.Vector3(1.0),
    )
    pullback = coefficient_history_pullback(
        histories,
        mode_fields,
        mp.Vector3(1.0),
        None,
        np.ones(4, dtype=np.complex128),
    )

    assert coefficient.shape == (4,)
    assert all(values.shape == (5,) for values in pullback.values())


def test_eigenmode_math_rejects_invalid_component_pairs_and_shapes():
    normal = mp.Vector3(0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="both components"):
        coefficient_history(
            {mp.Ex: np.ones((3, 2)), mp.Hy: np.ones((3, 2))},
            {mp.Ex: np.ones(2)},
            normal,
        )

    with pytest.raises(ValueError, match="same component pairs"):
        coefficient_history(
            {mp.Ex: np.ones((3, 2)), mp.Hy: np.ones((3, 2))},
            {mp.Ey: np.ones(2), mp.Hx: np.ones(2)},
            normal,
        )

    with pytest.raises(ValueError, match="same shape"):
        coefficient_history(
            {mp.Ex: np.ones((3, 2)), mp.Hy: np.ones((4, 2))},
            {mp.Ex: np.ones(2), mp.Hy: np.ones(2)},
            normal,
        )


def test_eigenmode_math_rejects_zero_power_and_wrong_covector_shape():
    normal = mp.Vector3(0.0, 0.0, 1.0)
    zero_power_mode = {
        mp.Ex: np.ones(2, dtype=np.complex128),
        mp.Hy: 1j * np.ones(2, dtype=np.complex128),
    }
    with pytest.raises(ValueError, match="nonzero signed power"):
        mode_power(zero_power_mode, normal)

    mode_fields = {
        mp.Ex: np.ones(2, dtype=np.complex128),
        mp.Hy: np.ones(2, dtype=np.complex128),
    }
    histories = {
        mp.Ex: np.ones((4, 2), dtype=np.complex128),
        mp.Hy: np.ones((4, 2), dtype=np.complex128),
    }
    with pytest.raises(ValueError, match="coefficient history shape"):
        coefficient_history_pullback(
            histories,
            mode_fields,
            normal,
            None,
            np.ones(4),
        )


def test_sample_reference_mode_selects_negative_power_direction():
    class FakeMode:
        def __init__(self, group_velocity, magnetic_sign, frequency):
            self.group_velocity = group_velocity
            self.k = mp.Vector3(0.75)
            self.freq = frequency
            self._fields = {
                mp.Ey: 1.0j,
                mp.Hz: magnetic_sign * 1.0j,
                mp.Ez: 0.0,
                mp.Hy: 0.0,
            }

        def amplitude(self, position, component):
            return self._fields[component]

    class FakeSimulation:
        dimensions = 2

        def __init__(self):
            self.fields = object()
            self.calls = []
            self.second_frequency = 0.75

        def get_eigenmode(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            if len(self.calls) == 1:
                return FakeMode(1.0, 1.0, 0.75)
            return FakeMode(-1.0, -1.0, self.second_frequency)

    simulation = FakeSimulation()
    target = type(
        "Target",
        (),
        {
            "positions": (mp.Vector3(0.2),),
            "normal": mp.Vector3(-1.0),
            "mode_region": mp.Volume(
                center=mp.Vector3(0.2),
                size=mp.Vector3(0.0, 1.0),
            ),
            "frequency": 0.75,
            "mode": 1,
            "spatial_weights": (1.0,),
            "eig_parity": mp.NO_PARITY,
            "eig_resolution": 0.0,
            "eig_tolerance": 1.0e-12,
        },
    )()

    fields = sample_reference_mode(simulation, target)

    assert len(simulation.calls) == 2
    assert simulation.calls[1][1]["match_frequency"] is False
    assert simulation.calls[1][0][4].x == -0.75
    assert mode_power(fields, target.normal, target.spatial_weights) > 0.0
    assert fields[mp.Ey][0] == pytest.approx(1.0)

    simulation = FakeSimulation()
    simulation.second_frequency = 0.8
    with pytest.raises(ValueError, match="reciprocal port medium"):
        sample_reference_mode(simulation, target)


def test_sample_reference_mode_supports_negative_normal_in_reciprocal_medium():
    mp.verbosity(0)
    simulation = mp.Simulation(
        cell_size=mp.Vector3(2.0, 2.0),
        resolution=10,
    )
    target = EigenmodeCoefficientTarget(
        positions=(
            mp.Vector3(0.0, -0.5),
            mp.Vector3(),
            mp.Vector3(0.0, 0.5),
        ),
        normal=mp.Vector3(-1.0),
        mode_region=mp.Volume(
            center=mp.Vector3(),
            size=mp.Vector3(0.0, 1.5),
        ),
        frequency=0.7,
        eig_parity=mp.ODD_Z,
    )

    try:
        mode_fields = sample_reference_mode(simulation, target)
    finally:
        simulation.reset_meep()

    assert (
        mode_power(
            mode_fields,
            target.normal,
            target.spatial_weights,
        )
        > 0.0
    )


def test_sample_reference_mode_uses_fixed_cylindrical_fields():
    positions = (mp.Vector3(0.4), mp.Vector3(0.8))
    target = EigenmodeCoefficientTarget(
        positions=positions,
        normal=mp.Vector3(0.0, 0.0, 1.0),
        spatial_weights=(0.3, 0.7),
        reference_mode_fields={
            mp.Er: (1.0 + 0.2j, 0.8 - 0.1j),
            mp.Hp: (0.7 + 0.1j, 1.1 - 0.2j),
            mp.Ep: (0.0, 0.0),
            mp.Hr: (0.0, 0.0),
        },
        reference_m=1,
    )
    simulation = type(
        "CylindricalSimulation",
        (),
        {"dimensions": mp.CYLINDRICAL, "is_cylindrical": True, "m": 1},
    )()

    fields = sample_reference_mode(simulation, target)

    assert set(fields) == {mp.Er, mp.Hp, mp.Ep, mp.Hr}
    assert (
        mode_power(
            fields,
            target.normal,
            target.spatial_weights,
            cylindrical=True,
        )
        > 0.0
    )


def test_sample_reference_mode_validates_cylindrical_reference_m():
    common = {
        "positions": (mp.Vector3(0.4), mp.Vector3(0.8)),
        "normal": mp.Vector3(0.0, 0.0, 1.0),
        "spatial_weights": (0.3, 0.7),
        "reference_mode_fields": {
            mp.Er: (1.0, 0.8),
            mp.Hp: (0.7, 1.1),
            mp.Ep: (0.0, 0.0),
            mp.Hr: (0.0, 0.0),
        },
    }
    simulation = type(
        "CylindricalSimulation",
        (),
        {"dimensions": mp.CYLINDRICAL, "is_cylindrical": True, "m": -1},
    )()

    with pytest.raises(ValueError, match="reference_m must match"):
        sample_reference_mode(
            simulation,
            EigenmodeCoefficientTarget(reference_m=1, **common),
        )

    with pytest.raises(ValueError, match="reference_m is required"):
        sample_reference_mode(
            simulation,
            EigenmodeCoefficientTarget(**common),
        )


def test_sample_reference_mode_requires_both_tangential_pairs():
    target = EigenmodeCoefficientTarget(
        positions=(mp.Vector3(0.4), mp.Vector3(0.8)),
        normal=mp.Vector3(0.0, 0.0, 1.0),
        spatial_weights=(0.3, 0.7),
        reference_mode_fields={
            mp.Er: (1.0, 0.8),
            mp.Hp: (0.7, 1.1),
        },
        reference_m=0,
    )
    simulation = type(
        "CylindricalSimulation",
        (),
        {"dimensions": mp.CYLINDRICAL, "is_cylindrical": True, "m": 0},
    )()

    with pytest.raises(ValueError, match="both complete tangential E/H pairs"):
        sample_reference_mode(simulation, target)


def test_cylindrical_reference_mode_requires_fixed_fields():
    target = EigenmodeCoefficientTarget(
        positions=(mp.Vector3(0.4), mp.Vector3(0.8)),
        normal=mp.Vector3(0.0, 0.0, 1.0),
        mode_region=mp.Volume(
            center=mp.Vector3(0.6),
            size=mp.Vector3(0.4),
        ),
        frequency=0.7,
    )
    simulation = type(
        "CylindricalSimulation",
        (),
        {"dimensions": mp.CYLINDRICAL, "is_cylindrical": True, "m": 0},
    )()

    with pytest.raises(ValueError, match="requires reference_mode_fields"):
        sample_reference_mode(simulation, target)


def test_sample_reference_mode_rejects_out_of_plane_2d_region():
    simulation = type(
        "Simulation",
        (),
        {"dimensions": 2, "fields": object()},
    )()
    target = type(
        "Target",
        (),
        {
            "normal": mp.Vector3(1.0),
            "mode_region": mp.Volume(size=mp.Vector3(0.0, 0.0, 1.0)),
        },
    )()

    with pytest.raises(ValueError, match="in-plane line"):
        sample_reference_mode(simulation, target)
