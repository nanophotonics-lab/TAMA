"""Cylindrical near-to-far values and discrete TAMA design derivatives.

The R/Z surface arrangement follows Meep tests/near2far.cpp::check_cyl.
These small finite-time scenes use tolerances from TAMA's Cartesian
near-to-far integration tests.
"""

import autograd.numpy as npa
import meep as mp
import numpy as np
import pytest

import tama as tm


def _make_problem(m, sampling_interval=1):
    air = mp.Medium(epsilon=1)
    dielectric = mp.Medium(epsilon=2.25)
    shape = (3, 3)
    size = mp.Vector3(0.4, 0, 0.4)
    center = mp.Vector3(0.65, 0, 0)
    grid = mp.MaterialGrid(mp.Vector3(3, 1, 3), air, dielectric)
    design = tm.DesignGrid(
        material_grid=grid,
        center=center,
        size=size,
        shape=shape,
        background=air,
        design_material=dielectric,
        coordinate_system="cylindrical",
    )
    target = tm.Near2FarTarget(
        near_regions=(
            mp.Near2FarRegion(
                center=mp.Vector3(1.2, 0, 0),
                size=mp.Vector3(0, 0, 1.6),
                direction=mp.R,
                weight=1,
            ),
            mp.Near2FarRegion(
                center=mp.Vector3(0.6, 0, 0.8),
                size=mp.Vector3(1.2, 0, 0),
                direction=mp.Z,
                weight=1,
            ),
            mp.Near2FarRegion(
                center=mp.Vector3(0.6, 0, -0.8),
                size=mp.Vector3(1.2, 0, 0),
                direction=mp.Z,
                weight=-1,
            ),
        ),
        frequencies=(0.7, 0.9),
        far_points=(mp.Vector3(0, 0, 3), mp.Vector3(2, 0, 3)),
    )
    pulse = mp.GaussianSource(0.8, fwidth=0.6, cutoff=3)
    simulation = tm.SimulationSpec(
        cell_size=mp.Vector3(1.8, 0, 2.8),
        dimensions=mp.CYLINDRICAL,
        m=m,
        resolution=12,
        courant=0.5,
        boundary_layers=[mp.PML(0.3, direction=mp.R), mp.PML(0.3, direction=mp.Z)],
        geometry=[mp.Block(center=center, size=size, material=grid)],
        sources=[
            mp.Source(pulse, component=mp.Er, center=mp.Vector3(0.6, 0, -0.55)),
            mp.Source(
                pulse,
                component=mp.Ep,
                center=mp.Vector3(0.65, 0, -0.5),
                amplitude=0.4 + 0.2j,
            ),
        ],
        eps_averaging=False,
    )
    objective = tm.TDAObjective(
        simulation=simulation,
        design=design,
        target=target,
        t_final=8,
        sampling_interval=sampling_interval,
        chunk_balancer=None,
    )
    weights = np.random.default_rng(821).uniform(0.25, 0.75, np.prod(shape))
    return objective, weights, target


@pytest.mark.mpi2
@pytest.mark.parametrize("m", [-1, 0, 1])
def test_cylindrical_farfield_matches_plain_meep_at_finite_time(m):
    mp.verbosity(0)
    objective, weights, target = _make_problem(m)
    value = objective.fom(weights)
    actual_time = objective.last_actual_time
    simulation = objective._simulation_spec.make()
    monitor = simulation.add_near2far(
        list(target.frequencies), *target.near_regions, decimation_factor=1
    )
    try:
        simulation.run(until=actual_time)
        expected = np.asarray(
            [simulation.get_farfield(monitor, point) for point in target.far_points]
        ).reshape(2, 2, 6)
        terminal_field = max(
            abs(simulation.get_field_point(component, region.center))
            for component in (mp.Er, mp.Ep, mp.Ez, mp.Hr, mp.Hp, mp.Hz)
            for region in target.near_regions
        )
    finally:
        simulation.reset_meep()
    assert terminal_field > 1e-6
    assert np.all(np.isfinite(expected))
    assert np.max(np.abs(expected)) > 1e-6
    np.testing.assert_allclose(
        objective.last_far_fields, expected, rtol=2e-12, atol=2e-13
    )
    np.testing.assert_allclose(
        value, 0.5 * np.sum(np.abs(expected[..., :3]) ** 2), rtol=2e-12
    )
    print("Cylindrical NFF forward:", m, value, terminal_field)


@pytest.mark.mpi2
@pytest.mark.parametrize("m", [-1, 0, 1])
@pytest.mark.parametrize("sampling_interval", [1, 2])
def test_cylindrical_farfield_gradient_matches_two_step_fd(m, sampling_interval):
    mp.verbosity(0)
    objective, weights, _ = _make_problem(m, sampling_interval)
    direction = np.random.default_rng(822).uniform(0.2, 1.0, weights.size)
    direction /= np.linalg.norm(direction)
    value, gradient = objective.fom_and_grad(weights)
    fields = objective.last_far_fields.copy()
    actual_time = objective.last_actual_time
    np.testing.assert_allclose(objective.fom(weights), value, rtol=2e-12)
    assert objective.last_actual_time == actual_time
    differences = [
        (
            objective.fom(weights + h * direction)
            - objective.fom(weights - h * direction)
        )
        / (2 * h)
        for h in (2e-4, 1e-4)
    ]
    assert value > 0
    assert fields.shape == (2, 2, 6)
    assert np.all(np.isfinite(gradient))
    assert abs(differences[-1]) > 1e-8
    np.testing.assert_allclose(differences[0], differences[1], rtol=1e-3)
    np.testing.assert_allclose(gradient @ direction, differences[-1], rtol=5e-3)
    if mp.count_processors() > 1:
        for rank_value, rank_gradient, rank_fields in mp.comm.allgather(
            (value, gradient, fields)
        ):
            np.testing.assert_allclose(rank_value, value, rtol=1e-12)
            np.testing.assert_allclose(rank_gradient, gradient, rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(rank_fields, fields, rtol=1e-12, atol=1e-14)
    print(
        "Cylindrical NFF FD:",
        m,
        sampling_interval,
        value,
        gradient @ direction,
        differences,
    )


@pytest.mark.parametrize(
    "invalid",
    ["near_negative_r", "far_negative_r", "far_y", "radial_pml", "axial_pml"],
)
def test_cylindrical_farfield_rejects_invalid_surfaces_and_points(invalid):
    objective, weights, target = _make_problem(0)
    if invalid == "near_negative_r":
        target.near_regions[1].center.x = 0.5
    elif invalid == "far_negative_r":
        target.far_points[0].x = -0.1
    elif invalid == "far_y":
        target.far_points[0].y = 0.1
    elif invalid == "radial_pml":
        target.near_regions[0].center.x = 1.6
    else:
        target.near_regions[1].center.z = 1.2
    message = "cylindrical far points" if invalid.startswith("far_") else "clearance"
    with pytest.raises(ValueError, match=message):
        objective.fom(weights)


@pytest.mark.mpi2
@pytest.mark.mpi4
@pytest.mark.mpi8
def test_cylindrical_farfield_complex_electric_magnetic_objective_gradient():
    mp.verbosity(0)
    objective, weights, _ = _make_problem(1)
    field_weights = np.asarray(
        [0.3 + 0.2j, -0.4 + 0.1j, 0.2 - 0.3j, 0.5 - 0.2j, -0.1 + 0.4j, 0.3j]
    )
    # Every E/H output contributes a nonzero covector when this sum is nonzero.
    objective.fom_fn = lambda fields, dt: npa.abs(npa.sum(fields * field_weights)) ** 2
    direction = np.random.default_rng(824).uniform(0.2, 1.0, weights.size)
    direction /= np.linalg.norm(direction)
    value, gradient = objective.fom_and_grad(weights)
    differences = [
        (
            objective.fom(weights + h * direction)
            - objective.fom(weights - h * direction)
        )
        / (2 * h)
        for h in (2e-4, 1e-4)
    ]
    assert value > 1e-8
    assert np.all(np.isfinite(gradient))
    assert abs(differences[-1]) > 1e-8
    np.testing.assert_allclose(differences[0], differences[1], rtol=1e-3)
    np.testing.assert_allclose(gradient @ direction, differences[-1], rtol=5e-3)
    print("Cylindrical NFF complex E/H FD:", value, gradient @ direction, differences)


def test_cylindrical_farfield_gradient_rejects_unsupported_angular_mode():
    with pytest.raises(ValueError, match="m=-1, 0, or"):
        objective, weights, _ = _make_problem(2)
        objective.fom_and_grad(weights)


@pytest.mark.mpi2
def test_cylindrical_mixed_temporal_farfield_matches_separate_objectives_and_fd():
    mp.verbosity(0)
    far_only, weights, far = _make_problem(1)
    temporal = tm.PointTarget(mp.Vector3(0.95, 0, 0.35), mp.Er)
    common = dict(
        design=far_only.design,
        simulation=far_only._simulation_spec,
        t_final=far_only.t_final,
        sampling_interval=1,
        chunk_balancer=None,
    )
    scalarize = (
        lambda values: 0.4 * values[0] + 0.6 * values[1] + 0.2 * values[0] * values[1]
    )
    temporal_only = tm.MultiTDAObjective(
        targets=[temporal],
        wavelength_bands=[(1.0, 1.6)],
        weights=[1.0],
        kernel_length=31,
        **common,
    )
    mixed = tm.MultiTDAObjective(
        targets=[temporal, far],
        wavelength_bands=[(1.0, 1.6), None],
        weights=[1.0, 1.0],
        kernel_length=31,
        scalarization_fn=scalarize,
        **common,
    )
    temporal_value, temporal_gradient = temporal_only.fom_and_grad(weights)
    far_value, far_gradient = far_only.fom_and_grad(weights)
    value, gradient = mixed.fom_and_grad(weights)
    expected_values = np.asarray([temporal_value, far_value])
    expected_gradient = (0.4 + 0.2 * far_value) * temporal_gradient + (
        0.6 + 0.2 * temporal_value
    ) * far_gradient
    np.testing.assert_allclose(
        mixed.last_band_objectives, expected_values, rtol=1e-11, atol=1e-13
    )
    np.testing.assert_allclose(value, scalarize(expected_values), rtol=1e-11)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=5e-3, atol=1e-8)
    direction = np.random.default_rng(823).uniform(0.2, 1.0, weights.size)
    direction /= np.linalg.norm(direction)
    differences = [
        (mixed.fom(weights + h * direction) - mixed.fom(weights - h * direction))
        / (2 * h)
        for h in (2e-4, 1e-4)
    ]
    assert abs(differences[-1]) > 1e-8
    np.testing.assert_allclose(differences[0], differences[1], rtol=1e-3)
    np.testing.assert_allclose(gradient @ direction, differences[-1], rtol=5e-3)
