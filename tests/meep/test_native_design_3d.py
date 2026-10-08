import meep as mp
import numpy as np
import pytest

import tama as tm
from tama.sampling_grid import NativeDesignField


def _make_3d_problem(
    component=mp.Ex,
    objective_kind="tda",
    finite_adjoint_source=False,
    regional_target=False,
    flux_target=False,
    eigenmode_target=False,
    target_position=None,
    chunk_layout=None,
    t_final=12.0,
    sampling_interval=1,
):
    resolution = 12
    air = mp.Medium(epsilon=1.0)
    dielectric = mp.Medium(epsilon=4.0)
    shape = (2, 2, 2)
    center = mp.Vector3(0.037, -0.041, 0.029)
    size = mp.Vector3(0.52, 0.46, 0.42)
    material_grid = mp.MaterialGrid(mp.Vector3(*shape), air, dielectric)
    design = tm.DesignGrid(
        material_grid=material_grid,
        center=center,
        size=size,
        shape=shape,
        background=air,
        design_material=dielectric,
    )
    source = [
        mp.Source(
            mp.GaussianSource(frequency=0.7, fwidth=0.5),
            component=component,
            center=mp.Vector3(-0.72, -0.18, 0.11),
        )
    ]
    geometry = [
        mp.Block(
            center=center,
            size=size,
            material=material_grid,
        )
    ]
    if eigenmode_target:
        geometry.append(
            mp.Block(
                center=mp.Vector3(0.71, 0.23, -0.13),
                size=mp.Vector3(0.5, 0.8, 0.6),
                material=mp.Medium(epsilon=6.0),
            )
        )
    simulation = tm.SimulationSpec(
        cell_size=mp.Vector3(2.5, 2.25, 2.0),
        boundary_layers=[mp.PML(0.375)],
        geometry=geometry,
        sources=source,
        resolution=resolution,
        eps_averaging=False,
        chunk_layout=chunk_layout,
    )
    adjoint_source_size = mp.Vector3()
    adjoint_source_amplitude = 1.0
    resolved_target_position = (
        mp.Vector3(0.71, 0.23, -0.13) if target_position is None else target_position
    )
    target_positions = (
        resolved_target_position + mp.Vector3(0.0, -0.11, 0.07),
        resolved_target_position + mp.Vector3(0.0, 0.09, -0.05),
    )
    target = (
        tm.EigenmodeCoefficientTarget(
            positions=target_positions,
            normal=mp.Vector3(1.0),
            mode_region=mp.Volume(
                center=resolved_target_position,
                size=mp.Vector3(0.0, 1.2, 1.0),
            ),
            frequency=0.7,
            spatial_weights=(0.8, 1.2),
        )
        if eigenmode_target
        else (
            tm.FluxTarget(
                positions=target_positions,
                normal=mp.Vector3(1.0),
                spatial_weights=(0.8, 1.2),
            )
            if flux_target
            else (
                tm.FieldRegionTarget(
                    positions=target_positions,
                    component=component,
                    sample_shape=(2,),
                    spatial_weights=(0.8, 1.2),
                    adjoint_source_amplitude=adjoint_source_amplitude,
                )
                if regional_target
                else tm.PointTarget(
                    position=resolved_target_position,
                    component=component,
                    adjoint_source_size=adjoint_source_size,
                    adjoint_source_amplitude=adjoint_source_amplitude,
                )
            )
        )
    )
    common = dict(
        design=design,
        simulation=simulation,
        t_final=t_final,
        dt=0.5 / resolution,
        sampling_interval=sampling_interval,
        chunk_balancer=(
            tm.AdaptiveAdjointChunkBalancer(source_boundary_mode="finite")
            if finite_adjoint_source
            else None
        ),
    )
    if objective_kind == "tda":
        objective = tm.TDAObjective(target=target, **common)
    else:
        objective = tm.MultiTDAObjective(
            targets=[target],
            wavelength_bands=[(1.1, 2.0)],
            weights=[1.0],
            kernel_length=31,
            pixel_chunk="auto",
            **common,
        )
    return objective, simulation


def _make_3d_gamma_periodic_multi_problem(
    *,
    reuse_simulation,
    full_cell=False,
):
    resolution = 10
    air = mp.Medium(epsilon=1.0)
    dielectric = mp.Medium(epsilon=4.0)
    cell_size = mp.Vector3(1.6, 1.4, 2.6)
    if full_cell:
        shape = (4, 4, 2)
        center = mp.Vector3()
        size = mp.Vector3(cell_size.x, cell_size.y, 0.40)
    else:
        shape = (2, 2, 2)
        center = mp.Vector3(0.03, -0.04, 0.02)
        size = mp.Vector3(0.50, 0.46, 0.40)
    material_grid = mp.MaterialGrid(mp.Vector3(*shape), air, dielectric)
    design = tm.DesignGrid(
        material_grid=material_grid,
        center=center,
        size=size,
        shape=shape,
        background=air,
        design_material=dielectric,
    )
    simulation = tm.SimulationSpec(
        cell_size=cell_size,
        boundary_layers=[mp.PML(0.3, direction=mp.Z)],
        geometry=[mp.Block(center=center, size=size, material=material_grid)],
        sources=[
            mp.Source(
                mp.GaussianSource(frequency=0.7, fwidth=0.5),
                component=mp.Ex,
                center=mp.Vector3(0.0, 0.0, -0.65),
                size=mp.Vector3(cell_size.x, cell_size.y, 0.0),
            )
        ],
        resolution=resolution,
        courant=0.125,
        dimensions=3,
        eps_averaging=False,
        k_point=mp.Vector3(),
    )
    objective = tm.MultiTDAObjective(
        design=design,
        simulation=simulation,
        targets=[
            tm.PointTarget(
                position=mp.Vector3(0.11, -0.07, 0.65),
                component=mp.Ex,
            )
        ],
        t_final=24.0,
        dt=0.125 / resolution,
        wavelength_bands=[(1.1, 2.0)],
        weights=[1.0],
        kernel_length=31,
        pixel_chunk="auto",
        reuse_simulation=reuse_simulation,
        chunk_balancer=None,
    )
    return objective


def _make_3d_gamma_periodic_rgb_flux_problem():
    resolution = 10
    cell_size = mp.Vector3(1.8, 1.4, 2.8)
    air = mp.Medium(epsilon=1.0)
    substrate = mp.Medium(epsilon=2.1)
    dielectric = mp.Medium(epsilon=6.0)
    shape = (4, 4, 2)
    design_center = mp.Vector3()
    design_size = mp.Vector3(cell_size.x, cell_size.y, 0.4)
    material_grid = mp.MaterialGrid(
        mp.Vector3(*shape),
        air,
        dielectric,
    )
    design = tm.DesignGrid(
        material_grid=material_grid,
        center=design_center,
        size=design_size,
        shape=shape,
        background=air,
        design_material=dielectric,
    )
    detector_z = -0.55
    detector_centers = (-0.55, 0.0, 0.55)
    detector_width = 0.32
    detector_height = 0.6
    detector_offsets = tuple(
        (offset_x, offset_y)
        for offset_x in (-detector_width / 4.0, detector_width / 4.0)
        for offset_y in (-detector_height / 4.0, detector_height / 4.0)
    )
    detector_weight = (detector_width / 2.0) * (detector_height / 2.0)
    targets = [
        tm.FluxTarget(
            positions=[
                mp.Vector3(center_x + offset_x, offset_y, detector_z)
                for offset_x, offset_y in detector_offsets
            ],
            normal=mp.Vector3(0.0, 0.0, -1.0),
            spatial_weights=[detector_weight] * len(detector_offsets),
        )
        for center_x in detector_centers
    ]
    simulation = tm.SimulationSpec(
        cell_size=cell_size,
        boundary_layers=[mp.PML(0.3, direction=mp.Z)],
        geometry=[
            mp.Block(
                center=mp.Vector3(0.0, 0.0, -0.8),
                size=mp.Vector3(cell_size.x, cell_size.y, 1.2),
                material=substrate,
            ),
            mp.Block(
                center=design_center,
                size=design_size,
                material=material_grid,
            ),
        ],
        sources=[
            mp.Source(
                mp.GaussianSource(frequency=0.7, fwidth=0.5),
                component=mp.Ex,
                center=mp.Vector3(0.0, 0.0, 0.75),
                size=mp.Vector3(cell_size.x, cell_size.y, 0.0),
            )
        ],
        resolution=resolution,
        courant=0.125,
        dimensions=3,
        eps_averaging=False,
        k_point=mp.Vector3(),
    )
    return tm.MultiTDAObjective(
        design=design,
        simulation=simulation,
        targets=targets,
        t_final=24.0,
        dt=0.125 / resolution,
        # Ordered short/mid/long bands form a scale-reduced RGB proxy.
        wavelength_bands=[
            (1.1, 1.35),
            (1.35, 1.65),
            (1.65, 2.0),
        ],
        weights=[1.0, 1.0, 1.0],
        kernel_length=401,
        pixel_chunk="auto",
        sampling_interval=1,
        chunk_balancer=None,
    )


def test_design_grid_supports_3d_material_grid():
    objective, _ = _make_3d_problem()
    design = objective.design

    weights = np.linspace(0.1, 0.8, 8)
    design.update_weights(weights)

    assert design.dimensions == 3
    assert design.spacing == pytest.approx((0.26, 0.23, 0.21))
    assert len(design.coords_z) == 2
    assert design.cell_volume == pytest.approx(0.012558)
    assert np.array_equal(design.material_grid.weights, weights)


def test_native_3d_plans_have_xyz_signatures():
    mp.verbosity(0)
    objective, simulation = _make_3d_problem()
    sim = simulation.make()
    sim.init_sim()
    plans = []
    try:
        for component in (mp.Ex, mp.Ey, mp.Ez):
            field = NativeDesignField(
                sim,
                component,
                objective.design,
            )
            plans.append(field)
            assert field.local_size > 0
            assert field.signature.shape == (field.local_size, 3)
    finally:
        for field in plans:
            field.release()
        sim.reset_meep()


def test_native_3d_real_accumulation_matches_complex_path():
    mp.verbosity(0)
    objective, simulation = _make_3d_problem()
    sim = simulation.make()
    sim.init_sim()
    real_field = None
    complex_field = None
    try:
        sim.run(until=4.0)
        real_field = NativeDesignField(sim, mp.Ex, objective.design)
        complex_field = NativeDesignField(sim, mp.Ex, objective.design)
        assert real_field.fields_are_real

        values = np.linspace(0.25, 1.25, real_field.local_size)
        real_accumulator = np.zeros(objective.design.shape, dtype=np.float64)
        complex_accumulator = np.zeros(
            objective.design.shape,
            dtype=np.complex128,
        )
        real_field.accumulate(values, real_accumulator)
        complex_field.accumulate(
            values.astype(np.complex128),
            complex_accumulator,
        )

        assert np.array_equal(real_accumulator, complex_accumulator.real)
        assert np.count_nonzero(complex_accumulator.imag) == 0
    finally:
        if real_field is not None:
            real_field.release()
        if complex_field is not None:
            complex_field.release()
        sim.reset_meep()


@pytest.mark.mpi2
@pytest.mark.parametrize("reuse_simulation", [False, True])
def test_native_3d_gamma_periodic_multi_gradient_matches_directional_fd(
    reuse_simulation,
):
    mp.verbosity(0)
    objective = _make_3d_gamma_periodic_multi_problem(
        reuse_simulation=reuse_simulation,
    )
    rng = np.random.default_rng(20260815)
    design = rng.uniform(0.2, 0.8, 8)
    direction = rng.uniform(0.2, 1.0, 8)
    direction /= np.linalg.norm(direction)

    _, gradient = objective.fom_and_grad(design)
    step = 2e-4
    finite_difference = (
        objective.fom(design + step * direction)
        - objective.fom(design - step * direction)
    ) / (2.0 * step)
    adjoint_derivative = float(gradient @ direction)
    relative_error = abs(adjoint_derivative - finite_difference) / abs(
        finite_difference
    )

    assert objective._simulation_spec.k_point is not False
    assert np.sign(adjoint_derivative) == np.sign(finite_difference)
    assert relative_error < 0.002


@pytest.mark.mpi2
def test_native_3d_gamma_periodic_full_cell_seams_match_directional_fd():
    mp.verbosity(0)
    objective = _make_3d_gamma_periodic_multi_problem(
        reuse_simulation=False,
        full_cell=True,
    )
    rng = np.random.default_rng(101)
    design = rng.uniform(0.2, 0.8, np.prod(objective.design.shape))

    _, gradient = objective.fom_and_grad(design)
    for axis in (0, 1):
        direction_grid = np.zeros(objective.design.shape)
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = 0
        upper[axis] = -1
        direction_grid[tuple(lower)] = 1.0
        direction_grid[tuple(upper)] = -1.0
        direction = direction_grid.ravel()
        direction /= np.linalg.norm(direction)

        step = 2e-4
        finite_difference = (
            objective.fom(design + step * direction)
            - objective.fom(design - step * direction)
        ) / (2.0 * step)
        adjoint_derivative = float(gradient @ direction)
        relative_error = abs(adjoint_derivative - finite_difference) / abs(
            finite_difference
        )

        assert np.sign(adjoint_derivative) == np.sign(finite_difference)
        assert relative_error < 0.002


@pytest.mark.mpi2
def test_native_3d_gamma_periodic_full_cell_rgb_flux_matches_directional_fd():
    mp.verbosity(0)
    objective = _make_3d_gamma_periodic_rgb_flux_problem()
    rng = np.random.default_rng(20260816)
    design = rng.uniform(0.2, 0.8, np.prod(objective.design.shape))
    direction = rng.uniform(-1.0, 1.0, design.size)
    direction /= np.linalg.norm(direction)

    value, gradient = objective.fom_and_grad(design)
    adjoint_derivative = float(gradient @ direction)
    finite_differences = []
    for step in (1e-3, 3e-4):
        finite_differences.append(
            (
                objective.fom(design + step * direction)
                - objective.fom(design - step * direction)
            )
            / (2.0 * step)
        )
    finite_difference = finite_differences[-1]
    relative_error = abs(adjoint_derivative - finite_difference) / max(
        abs(adjoint_derivative),
        abs(finite_difference),
    )
    plateau_error = abs(finite_differences[0] - finite_difference) / max(
        abs(finite_differences[0]),
        abs(finite_difference),
    )

    assert value > 0.0
    assert np.all(np.asarray(objective.last_band_objectives) > 0.0)
    np.testing.assert_array_equal(objective.last_band_coeffs, np.ones(3))
    assert objective.gradient_components == (mp.Ex, mp.Ey, mp.Ez)
    assert abs(finite_difference) > 1e-8
    assert np.sign(adjoint_derivative) == np.sign(finite_difference)
    assert plateau_error < 0.002
    assert relative_error < 0.005


@pytest.mark.parametrize(
    "component",
    [mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz],
)
def test_native_3d_target_uses_all_electric_gradient_components(component):
    objective, _ = _make_3d_problem(component)

    assert objective.gradient_components == (mp.Ex, mp.Ey, mp.Ez)


@pytest.mark.parametrize(
    ("objective_kind", "component", "relative_tolerance"),
    [
        ("tda", mp.Ex, 0.01),
        ("tda", mp.Hz, 0.01),
        ("multi", mp.Ex, 0.01),
        ("multi", mp.Hz, 0.02),
    ],
)
def test_native_3d_full_vector_gradient_matches_directional_finite_difference(
    objective_kind,
    component,
    relative_tolerance,
):
    mp.verbosity(0)
    objective, _ = _make_3d_problem(component, objective_kind)
    rng = np.random.default_rng(430 + int(component))
    design = rng.uniform(0.2, 0.8, 8)
    direction = rng.uniform(0.2, 1.0, 8)
    direction /= np.linalg.norm(direction)

    _, gradient = objective.fom_and_grad(design)
    step = 2e-4
    finite_difference = (
        objective.fom(design + step * direction)
        - objective.fom(design - step * direction)
    ) / (2.0 * step)
    adjoint_derivative = float(gradient @ direction)
    relative_error = abs(adjoint_derivative - finite_difference) / abs(
        finite_difference
    )

    assert objective.gradient_components == (mp.Ex, mp.Ey, mp.Ez)
    assert np.sign(adjoint_derivative) == np.sign(finite_difference)
    assert relative_error < relative_tolerance


@pytest.mark.parametrize(
    ("objective_kind", "component", "relative_tolerance"),
    [
        ("tda", mp.Ex, 0.01),
        ("multi", mp.Hz, 0.02),
    ],
)
def test_native_3d_nyquist_gradient_matches_directional_finite_difference(
    objective_kind,
    component,
    relative_tolerance,
):
    mp.verbosity(0)
    objective, _ = _make_3d_problem(
        component,
        objective_kind,
        sampling_interval=2,
    )
    rng = np.random.default_rng(20260805 + int(component))
    design = rng.uniform(0.2, 0.8, 8)
    direction = rng.uniform(0.2, 1.0, 8)
    direction /= np.linalg.norm(direction)

    _, gradient = objective.fom_and_grad(design)
    step = 2e-4
    finite_difference = (
        objective.fom(design + step * direction)
        - objective.fom(design - step * direction)
    ) / (2.0 * step)
    adjoint_derivative = float(gradient @ direction)
    relative_error = abs(adjoint_derivative - finite_difference) / abs(
        finite_difference
    )

    assert objective.last_sampling_interval == 2
    assert np.sign(adjoint_derivative) == np.sign(finite_difference)
    assert relative_error < relative_tolerance


@pytest.mark.parametrize("objective_kind", ["tda", "multi"])
def test_native_3d_field_region_matches_directional_fd(objective_kind):
    mp.verbosity(0)
    objective, _ = _make_3d_problem(
        mp.Ex,
        objective_kind,
        regional_target=True,
    )
    rng = np.random.default_rng(20260730)
    design = rng.uniform(0.2, 0.8, 8)
    direction = rng.uniform(0.2, 1.0, 8)
    direction /= np.linalg.norm(direction)

    _, gradient = objective.fom_and_grad(design)
    step = 2e-4
    finite_difference = (
        objective.fom(design + step * direction)
        - objective.fom(design - step * direction)
    ) / (2.0 * step)
    adjoint_derivative = float(gradient @ direction)
    relative_error = abs(adjoint_derivative - finite_difference) / abs(
        finite_difference
    )

    assert np.sign(adjoint_derivative) == np.sign(finite_difference)
    assert relative_error < 0.01


@pytest.mark.parametrize("objective_kind", ["tda", "multi"])
@pytest.mark.parametrize(
    ("target_kind", "seed"),
    [("flux", 20260731), ("eigenmode", 20260802)],
)
def test_native_3d_indexed_target_matches_directional_fd(
    objective_kind,
    target_kind,
    seed,
):
    mp.verbosity(0)
    objective, _ = _make_3d_problem(
        mp.Ey,
        objective_kind,
        flux_target=target_kind == "flux",
        eigenmode_target=target_kind == "eigenmode",
        t_final=20.0,
    )
    rng = np.random.default_rng(seed)
    design = rng.uniform(0.2, 0.8, 8)
    direction = rng.uniform(0.2, 1.0, 8)
    direction /= np.linalg.norm(direction)

    _, gradient = objective.fom_and_grad(design)
    step = 2e-4
    finite_difference = (
        objective.fom(design + step * direction)
        - objective.fom(design - step * direction)
    ) / (2.0 * step)
    adjoint_derivative = float(gradient @ direction)
    relative_error = abs(adjoint_derivative - finite_difference) / abs(
        finite_difference
    )

    assert objective.gradient_components == (mp.Ex, mp.Ey, mp.Ez)
    assert np.sign(adjoint_derivative) == np.sign(finite_difference)
    assert relative_error < 0.02


@pytest.mark.mpi2
@pytest.mark.parametrize("objective_kind", ["tda", "multi"])
@pytest.mark.parametrize("sampling_interval", [1, 2])
def test_native_3d_indexed_target_on_chunk_boundary_matches_directional_fd(
    objective_kind,
    sampling_interval,
):
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")
    mp.verbosity(0)
    objective, _ = _make_3d_problem(
        mp.Ex,
        objective_kind,
        target_position=mp.Vector3(0.0, 0.55, -0.13),
        chunk_layout=mp.BinaryPartition(data=[(mp.X, 0.0), 0, 1]),
        t_final=20.0,
        sampling_interval=sampling_interval,
    )
    rng = np.random.default_rng(1327)
    design = rng.uniform(0.2, 0.8, 8)
    direction = rng.uniform(0.2, 1.0, 8)
    direction /= np.linalg.norm(direction)

    _, gradient = objective.fom_and_grad(design)
    step = 2e-4
    finite_difference = (
        objective.fom(design + step * direction)
        - objective.fom(design - step * direction)
    ) / (2.0 * step)
    adjoint_derivative = float(gradient @ direction)
    relative_error = abs(adjoint_derivative - finite_difference) / abs(
        finite_difference
    )

    assert np.sign(adjoint_derivative) == np.sign(finite_difference)
    assert relative_error < 0.01


@pytest.mark.parametrize("objective_kind", ["tda", "multi"])
def test_native_3d_one_cell_finite_adjoint_source_matches_directional_fd(
    objective_kind,
):
    mp.verbosity(0)
    objective, _ = _make_3d_problem(
        mp.Ex,
        objective_kind,
        finite_adjoint_source=True,
    )
    rng = np.random.default_rng(915)
    design = rng.uniform(0.2, 0.8, 8)
    direction = rng.uniform(0.2, 1.0, 8)
    direction /= np.linalg.norm(direction)

    _, gradient = objective.fom_and_grad(design)
    step = 2e-4
    finite_difference = (
        objective.fom(design + step * direction)
        - objective.fom(design - step * direction)
    ) / (2.0 * step)
    adjoint_derivative = float(gradient @ direction)
    relative_error = abs(adjoint_derivative - finite_difference) / abs(
        finite_difference
    )

    assert np.sign(adjoint_derivative) == np.sign(finite_difference)
    assert relative_error < 0.02
