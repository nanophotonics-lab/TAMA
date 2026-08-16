from dataclasses import replace

import meep as mp
import numpy as np
import pytest

import tama as tm
import tama.multi_tda_objective as multi_tda_module
import tama.tda_objective as tda_module
from tama.native_design import _centered_derivative_into
from tama.nyquist import _WindowedSincReconstructor
from tama.sampling_grid import NativeDesignField


def _python_adjoint_loop_reference(
    sim,
    accumulator,
    field_histories,
    *,
    fine_step_count,
    actual_time,
    dt,
    sampling_interval,
    reconstruction_half_width,
    reconstruction_window,
    reconstruction_window_params,
):
    reconstructors = {}
    current = {}
    neighbor = {}
    derivative = {}
    if sampling_interval > 1:
        for component, history in field_histories.items():
            reconstructors[component] = _WindowedSincReconstructor(
                history,
                sampling_interval,
                half_width=reconstruction_half_width,
                window=reconstruction_window,
                window_params=reconstruction_window_params,
            )
            current[component] = np.empty(
                history.shape[1],
                dtype=reconstructors[component].dtype,
            )
            neighbor[component] = np.empty_like(current[component])
            derivative[component] = np.empty_like(current[component])

    state = {"step": 0}

    def accumulate(current_sim):
        fine_index = fine_step_count - state["step"]
        for component in accumulator.components:
            if sampling_interval > 1:
                forward_derivative = _centered_derivative_into(
                    reconstructors[component].sample_into,
                    fine_index,
                    fine_step_count,
                    dt,
                    current[component],
                    neighbor[component],
                    derivative[component],
                )
            else:
                history = field_histories[component]
                if fine_index == 0:
                    forward_derivative = (history[1] - history[0]) / dt
                elif fine_index == fine_step_count:
                    forward_derivative = (history[-1] - history[-2]) / dt
                else:
                    forward_derivative = (
                        history[fine_index + 1] - history[fine_index - 1]
                    ) / (2.0 * dt)
            accumulator.accumulate(
                current_sim,
                component,
                forward_derivative,
            )
        state["step"] += 1

    sim.run(accumulate, until=actual_time)
    assert state["step"] == fine_step_count + 1
    return 0.0


def _make_problem(
    objective_kind,
    component,
    sampling_interval=1,
    touch_pml=False,
    reuse_simulation=False,
    finite_adjoint_source=False,
    regional_target=False,
    target_position=None,
    chunk_layout=None,
    history_dtype=np.complex128,
    t_final=100.0,
    target_history_block_size=None,
):
    resolution = 20
    dt = 0.5 / resolution
    shape = (10, 3) if touch_pml else (5, 5)
    air = mp.Medium(epsilon=1.0)
    dielectric = mp.Medium(epsilon=4.0)
    center = mp.Vector3() if touch_pml else mp.Vector3(0.013, -0.017)
    size = mp.Vector3(4.0, 0.4) if touch_pml else mp.Vector3(0.237, 0.263)
    source_center = (
        mp.Vector3(0.0, -1.5)
        if touch_pml
        else mp.Vector3(-1.5, center.y)
    )
    source_size = mp.Vector3(4.0, 0.0) if touch_pml else mp.Vector3(0, 3.5)
    if target_position is None:
        target_position = (
            mp.Vector3(0.0, 1.5)
            if touch_pml
            else mp.Vector3(1.5, center.y)
        )
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
            mp.GaussianSource(frequency=0.7, fwidth=0.4),
            component=component,
            center=source_center,
            size=source_size,
        )
    ]
    simulation = tm.SimulationSpec(
        cell_size=mp.Vector3(6, 6),
        boundary_layers=[mp.PML(1.0)],
        geometry=[mp.Block(center=center, size=size, material=material_grid)],
        sources=source,
        resolution=resolution,
        eps_averaging=False,
        chunk_layout=chunk_layout,
    )
    adjoint_source_size = mp.Vector3()
    adjoint_source_amplitude = 1.0
    target = (
        tm.FieldRegionTarget(
            positions=(
                target_position + mp.Vector3(0.0, -0.13),
                target_position + mp.Vector3(0.0, 0.11),
            ),
            component=component,
            sample_shape=(2,),
            spatial_weights=(0.75, 1.25),
            adjoint_source_amplitude=adjoint_source_amplitude,
        )
        if regional_target
        else tm.PointTarget(
            position=target_position,
            component=component,
            adjoint_source_size=adjoint_source_size,
            adjoint_source_amplitude=adjoint_source_amplitude,
        )
    )
    common = dict(
        design=design,
        simulation=simulation,
        t_final=t_final,
        dt=dt,
        sampling_interval=sampling_interval,
        history_dtype=history_dtype,
        chunk_balancer=(
            tm.AdaptiveAdjointChunkBalancer(source_boundary_mode="finite")
            if finite_adjoint_source
            else None
        ),
    )
    if objective_kind == "tda":
        return tm.TDAObjective(
            target=target,
            reuse_simulation=reuse_simulation,
            **common,
        )
    return tm.MultiTDAObjective(
        targets=[target],
        wavelength_bands=[(1.1, 2.0)],
        weights=[1.0],
        kernel_length=101,
        pixel_chunk=5,
        reuse_simulation=True,
        target_history_block_size=target_history_block_size,
        **common,
    )


def _make_mixed_component_multi_problem(**kwargs):
    resolution = 20
    dt = 0.5 / resolution
    shape = (5, 5)
    air = mp.Medium(epsilon=1.0)
    dielectric = mp.Medium(epsilon=4.0)
    center = mp.Vector3(0.013, -0.017)
    size = mp.Vector3(0.237, 0.263)
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
        cell_size=mp.Vector3(6, 6),
        boundary_layers=[mp.PML(1.0)],
        geometry=[mp.Block(center=center, size=size, material=material_grid)],
        sources=[
            mp.Source(
                mp.GaussianSource(frequency=0.7, fwidth=0.4),
                component=mp.Hz,
                center=mp.Vector3(-1.5, center.y),
                size=mp.Vector3(0, 3.5),
            )
        ],
        resolution=resolution,
        eps_averaging=False,
    )
    return tm.MultiTDAObjective(
        design=design,
        simulation=simulation,
        targets=[
            tm.PointTarget(mp.Vector3(1.5, center.y + 0.35), mp.Ex),
            tm.PointTarget(mp.Vector3(1.5, center.y - 0.15), mp.Ey),
        ],
        t_final=100.0,
        dt=dt,
        wavelength_bands=[(1.1, 1.5), (1.5, 2.0)],
        weights=[1.0, 1.0],
        kernel_length=101,
        pixel_chunk=5,
        reuse_simulation=True,
        **kwargs,
    )


def _make_flux_problem(
    objective_kind,
    source_component=mp.Hz,
    *,
    reuse_simulation=False,
    eigenmode_target=False,
    sampling_interval=1,
    target_history_block_size=None,
    full_vector_sources=False,
):
    resolution = 20
    dt = 0.5 / resolution
    shape = (5, 5)
    air = mp.Medium(epsilon=1.0)
    dielectric = mp.Medium(epsilon=4.0)
    center = mp.Vector3(0.013, -0.017)
    size = mp.Vector3(0.237, 0.263)
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
        cell_size=mp.Vector3(6, 6),
        boundary_layers=[mp.PML(1.0)],
        geometry=[
            mp.Block(center=center, size=size, material=material_grid)
        ],
        sources=[
            mp.Source(
                mp.GaussianSource(frequency=0.7, fwidth=0.4),
                component=component,
                center=mp.Vector3(-1.5, center.y),
                size=mp.Vector3(0, 3.5),
            )
            for component in (
                (mp.Hz, mp.Ez)
                if full_vector_sources
                else (source_component,)
            )
        ],
        resolution=resolution,
        eps_averaging=False,
    )
    positions = (
        mp.Vector3(1.5, center.y - 0.1),
        mp.Vector3(1.5, center.y + 0.1),
    )
    target = (
        tm.EigenmodeCoefficientTarget(
            positions=positions,
            normal=mp.Vector3(1.0),
            mode_region=mp.Volume(
                center=mp.Vector3(1.5, center.y),
                size=mp.Vector3(0.0, 3.5),
            ),
            frequency=0.7,
            spatial_weights=(0.6, 1.4),
            eig_parity=(
                mp.EVEN_Z if source_component == mp.Hz else mp.ODD_Z
            ),
        )
        if eigenmode_target
        else tm.FluxTarget(
            positions,
            normal=mp.Vector3(1.0),
            spatial_weights=(0.6, 1.4),
        )
    )
    common = dict(
        design=design,
        simulation=simulation,
        target=target,
        t_final=100.0,
        dt=dt,
        reuse_simulation=reuse_simulation,
        sampling_interval=sampling_interval,
    )
    if objective_kind == "tda":
        return tm.TDAObjective(**common)
    return tm.MultiTDAObjective(
        targets=[common.pop("target")],
        wavelength_bands=[(1.1, 2.0)],
        weights=[1.0],
        kernel_length=101,
        pixel_chunk=5,
        target_history_block_size=target_history_block_size,
        **common,
    )


def test_native_2d_real_accumulation_matches_complex_path():
    mp.verbosity(0)
    objective = _make_problem("tda", mp.Ez)
    sim = objective._simulation_spec.make()
    sim.init_sim()
    real_field = None
    complex_field = None
    try:
        sim.run(until=4.0)
        real_field = NativeDesignField(sim, mp.Ez, objective.design)
        complex_field = NativeDesignField(sim, mp.Ez, objective.design)
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

        real_midpoint = np.zeros(objective.design.shape, dtype=np.float64)
        complex_midpoint = np.zeros(
            objective.design.shape,
            dtype=np.complex128,
        )
        assert not real_field.accumulate_midpoint(values, real_midpoint)
        assert not complex_field.accumulate_midpoint(
            values.astype(np.complex128),
            complex_midpoint,
        )
        sim.run(until=sim.round_time() + 0.1)
        assert real_field.accumulate_midpoint(values, real_midpoint)
        assert complex_field.accumulate_midpoint(
            values.astype(np.complex128),
            complex_midpoint,
        )
        assert np.array_equal(real_midpoint, complex_midpoint.real)
        assert np.count_nonzero(complex_midpoint.imag) == 0
    finally:
        if real_field is not None:
            real_field.release()
        if complex_field is not None:
            complex_field.release()
        sim.reset_meep()


def test_native_tda_reuse_matches_separate_simulations():
    mp.verbosity(0)
    design = np.linspace(0.2, 0.8, 25)
    separate_value, separate_gradient = _make_problem(
        "tda",
        mp.Ez,
    ).fom_and_grad(design)
    reused_value, reused_gradient = _make_problem(
        "tda",
        mp.Ez,
        reuse_simulation=True,
    ).fom_and_grad(design)

    assert np.allclose(reused_value, separate_value, rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(
        reused_gradient,
        separate_gradient,
        rtol=1.0e-11,
        atol=1.0e-12,
    )


@pytest.mark.parametrize(
    ("objective_kind", "component", "sampling_interval"),
    [
        ("tda", mp.Ez, 1),
        ("multi", mp.Hz, 2),
    ],
)
def test_full_native_adjoint_loop_matches_python_reference(
    monkeypatch,
    objective_kind,
    component,
    sampling_interval,
):
    mp.verbosity(0)
    design = np.linspace(0.2, 0.8, 25)
    native_value, native_gradient = _make_problem(
        objective_kind,
        component,
        sampling_interval=sampling_interval,
        t_final=10.0,
    ).fom_and_grad(design)

    objective_module = (
        tda_module if objective_kind == "tda" else multi_tda_module
    )
    monkeypatch.setattr(
        objective_module,
        "_run_native_adjoint_loop",
        _python_adjoint_loop_reference,
    )
    reference_value, reference_gradient = _make_problem(
        objective_kind,
        component,
        sampling_interval=sampling_interval,
        t_final=10.0,
    ).fom_and_grad(design)

    assert np.allclose(
        native_value,
        reference_value,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert np.allclose(
        native_gradient,
        reference_gradient,
        rtol=1.0e-11,
        atol=1.0e-12,
    )


def test_full_native_flux_loop_matches_python_reference(monkeypatch):
    mp.verbosity(0)
    design = np.linspace(0.2, 0.8, 25)
    native_value, native_gradient = _make_flux_problem(
        "tda",
        mp.Hz,
        sampling_interval=2,
    ).fom_and_grad(design)

    monkeypatch.setattr(
        tda_module,
        "_run_native_adjoint_loop",
        _python_adjoint_loop_reference,
    )
    reference_value, reference_gradient = _make_flux_problem(
        "tda",
        mp.Hz,
        sampling_interval=2,
    ).fom_and_grad(design)

    assert np.allclose(
        native_value,
        reference_value,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert np.allclose(
        native_gradient,
        reference_gradient,
        rtol=1.0e-11,
        atol=1.0e-12,
    )


@pytest.mark.parametrize(
    "history_dtype",
    [np.float16, np.float32, np.float64, np.longdouble],
)
def test_full_native_real_history_dtypes_match_python_reference(
    monkeypatch,
    history_dtype,
):
    mp.verbosity(0)
    design = np.linspace(0.2, 0.8, 25)
    native_value, native_gradient = _make_problem(
        "tda",
        mp.Ez,
        sampling_interval=2,
        history_dtype=history_dtype,
        t_final=10.0,
    ).fom_and_grad(design)

    monkeypatch.setattr(
        tda_module,
        "_run_native_adjoint_loop",
        _python_adjoint_loop_reference,
    )
    reference_value, reference_gradient = _make_problem(
        "tda",
        mp.Ez,
        sampling_interval=2,
        history_dtype=history_dtype,
        t_final=10.0,
    ).fom_and_grad(design)

    assert np.allclose(native_value, reference_value, rtol=0.0, atol=0.0)
    assert np.allclose(
        native_gradient,
        reference_gradient,
        rtol=2.0e-7,
        atol=1.0e-12,
    )


def test_full_native_loop_rejects_adjoint_timestep_mismatch():
    mp.verbosity(0)
    template = _make_problem("tda", mp.Ez, t_final=2.0)
    forward_spec = template._simulation_spec
    adjoint_spec = replace(forward_spec, courant=0.6)

    def make_simulation(sources=None):
        spec = forward_spec if sources is None else adjoint_spec
        return spec.make(sources)

    objective = tm.TDAObjective(
        design=template.design,
        sim_factory=make_simulation,
        target=tm.PointTarget(
            mp.Vector3(1.5, template.design.center.y),
            mp.Ez,
        ),
        t_final=2.0,
        dt=0.5 / forward_spec.resolution,
        resolution=forward_spec.resolution,
    )

    with pytest.raises(RuntimeError, match="forward time grid"):
        objective.fom_and_grad(np.linspace(0.2, 0.8, 25))


@pytest.mark.parametrize("objective_kind", ["tda", "multi"])
@pytest.mark.parametrize("source_component", [mp.Hz, mp.Ez])
def test_native_2d_flux_target_matches_directional_fd(
    objective_kind,
    source_component,
):
    mp.verbosity(0)
    objective = _make_flux_problem(objective_kind, source_component)
    rng = np.random.default_rng(20260730 + int(source_component))
    design = rng.uniform(0.15, 0.85, 25)
    direction = rng.uniform(0.2, 1.0, 25)
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
    assert relative_error < 0.01


@pytest.mark.parametrize("objective_kind", ["tda", "multi"])
def test_native_2d_flux_target_nyquist_path_matches_directional_fd(
    objective_kind,
):
    mp.verbosity(0)
    objective = _make_flux_problem(
        objective_kind,
        mp.Hz,
        sampling_interval=2,
    )
    rng = np.random.default_rng(20260803)
    design = rng.uniform(0.15, 0.85, 25)
    direction = rng.uniform(0.2, 1.0, 25)
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
    assert relative_error < 0.01


@pytest.mark.mpi2
@pytest.mark.parametrize("target_kind", ("regional", "flux"))
def test_distributed_target_history_matches_directional_fd(target_kind):
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")

    if target_kind == "regional":
        objective = _make_problem(
            "multi",
            mp.Ez,
            regional_target=True,
            target_history_block_size=1,
        )
    else:
        objective = _make_flux_problem(
            "multi",
            mp.Hz,
            target_history_block_size=1,
            full_vector_sources=True,
        )
    rng = np.random.default_rng(20260816)
    design = rng.uniform(0.15, 0.85, 25)
    direction = rng.uniform(0.2, 1.0, 25)
    direction /= np.linalg.norm(direction)

    _, gradient = objective.fom_and_grad(design)
    adjoint_derivative = float(gradient @ direction)
    finite_differences = []
    for step in (2.0e-4, 1.0e-4):
        finite_differences.append(
            (
                objective.fom(design + step * direction)
                - objective.fom(design - step * direction)
            )
            / (2.0 * step)
        )

    assert objective.last_target_history_mode == "distributed"
    assert all(abs(value) > 1.0e-3 for value in finite_differences)
    relative_errors = [
        abs(adjoint_derivative - value)
        / max(abs(adjoint_derivative), abs(value))
        for value in finite_differences
    ]
    plateau_error = abs(
        finite_differences[0] - finite_differences[1]
    ) / max(abs(value) for value in finite_differences)
    assert max(relative_errors) < 0.02
    assert plateau_error < 1.0e-4


@pytest.mark.parametrize("objective_kind", ["tda", "multi"])
@pytest.mark.parametrize("source_component", [mp.Hz, mp.Ez])
def test_native_2d_eigenmode_target_matches_directional_fd(
    objective_kind,
    source_component,
):
    mp.verbosity(0)
    objective = _make_flux_problem(
        objective_kind,
        source_component,
        eigenmode_target=True,
    )
    rng = np.random.default_rng(20260801 + int(source_component))
    design = rng.uniform(0.15, 0.85, 25)
    direction = rng.uniform(0.2, 1.0, 25)
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
    assert relative_error < 0.01


@pytest.mark.parametrize("objective_kind", ["tda", "multi"])
@pytest.mark.parametrize(
    ("component", "expected_gradient_components"),
    [
        (mp.Ez, (mp.Ez,)),
        (mp.Hx, (mp.Ez,)),
        (mp.Hz, (mp.Ex, mp.Ey)),
    ],
)
def test_native_2d_gradient_matches_directional_finite_difference(
    objective_kind,
    component,
    expected_gradient_components,
):
    mp.verbosity(0)
    objective = _make_problem(objective_kind, component)
    rng = np.random.default_rng(100 + int(component))
    design = rng.uniform(0.15, 0.85, 25)
    direction = rng.uniform(0.2, 1.0, 25)
    direction /= np.linalg.norm(direction)

    _, gradient = objective.fom_and_grad(design)
    step = 2e-4
    finite_difference = (
        objective.fom(design + step * direction)
        - objective.fom(design - step * direction)
    ) / (2.0 * step)
    adjoint_derivative = float(gradient @ direction)
    relative_error = abs(adjoint_derivative - finite_difference) / abs(finite_difference)

    assert objective.gradient_components == expected_gradient_components
    assert np.sign(adjoint_derivative) == np.sign(finite_difference)
    assert relative_error < 0.006


def test_native_multi_mixed_electric_targets_match_directional_fd():
    mp.verbosity(0)
    objective = _make_mixed_component_multi_problem()
    rng = np.random.default_rng(20260727)
    design = rng.uniform(0.15, 0.85, 25)
    direction = rng.uniform(0.2, 1.0, 25)
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

    assert objective.gradient_components == (mp.Ex, mp.Ey)
    assert np.sign(adjoint_derivative) == np.sign(finite_difference)
    assert relative_error < 0.006


def test_native_multi_power_complementary_bank_matches_directional_fd():
    mp.verbosity(0)
    objective = _make_mixed_component_multi_problem(
        filter_bank="power_complementary",
        transition_width=0.08,
    )
    rng = np.random.default_rng(20260812)
    design = rng.uniform(0.15, 0.85, 25)
    direction = rng.uniform(0.2, 1.0, 25)
    direction /= np.linalg.norm(direction)

    value, gradient = objective.fom_and_grad(design)
    step = 2e-4
    finite_difference = (
        objective.fom(design + step * direction)
        - objective.fom(design - step * direction)
    ) / (2.0 * step)
    adjoint_derivative = float(gradient @ direction)
    relative_error = abs(adjoint_derivative - finite_difference) / abs(
        finite_difference
    )

    assert np.isfinite(value)
    assert np.linalg.norm(gradient) > 0.0
    assert abs(finite_difference) > 1.0e-10
    assert np.sign(adjoint_derivative) == np.sign(finite_difference)
    assert relative_error < 0.006


@pytest.mark.parametrize("objective_kind", ["tda", "multi"])
def test_native_2d_field_region_matches_directional_fd(objective_kind):
    mp.verbosity(0)
    objective = _make_problem(
        objective_kind,
        mp.Ez,
        regional_target=True,
    )
    rng = np.random.default_rng(20260729)
    design = rng.uniform(0.15, 0.85, 25)
    direction = rng.uniform(0.2, 1.0, 25)
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
    assert relative_error < 0.006


@pytest.mark.parametrize("objective_kind", ["tda", "multi"])
def test_native_2d_one_cell_finite_adjoint_source_matches_directional_fd(
    objective_kind,
):
    mp.verbosity(0)
    objective = _make_problem(
        objective_kind,
        mp.Ez,
        finite_adjoint_source=True,
    )
    rng = np.random.default_rng(740)
    design = rng.uniform(0.15, 0.85, 25)
    direction = rng.uniform(0.2, 1.0, 25)
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


def test_native_tez_nyquist_gradient_matches_directional_finite_difference():
    mp.verbosity(0)
    objective = _make_problem("tda", mp.Hz, sampling_interval=2)
    rng = np.random.default_rng(309)
    design = rng.uniform(0.15, 0.85, 25)
    direction = rng.uniform(0.2, 1.0, 25)
    direction /= np.linalg.norm(direction)

    _, gradient = objective.fom_and_grad(design)
    step = 2e-4
    finite_difference = (
        objective.fom(design + step * direction)
        - objective.fom(design - step * direction)
    ) / (2.0 * step)
    adjoint_derivative = float(gradient @ direction)
    relative_error = abs(adjoint_derivative - finite_difference) / abs(finite_difference)

    assert objective.last_sampling_interval == 2
    assert relative_error < 0.006


def test_native_gradient_matches_fd_when_design_touches_pml_inner_edge():
    mp.verbosity(0)
    objective = _make_problem("tda", mp.Ez, touch_pml=True)
    rng = np.random.default_rng(902)
    variable_count = int(np.prod(objective.design.shape))
    design = rng.uniform(0.15, 0.85, variable_count)
    direction = rng.uniform(0.2, 1.0, variable_count)
    direction /= np.linalg.norm(direction)

    _, gradient = objective.fom_and_grad(design)
    step = 2e-4
    finite_difference = (
        objective.fom(design + step * direction)
        - objective.fom(design - step * direction)
    ) / (2.0 * step)
    adjoint_derivative = float(gradient @ direction)
    relative_error = abs(adjoint_derivative - finite_difference) / abs(finite_difference)

    assert relative_error < 0.006


@pytest.mark.mpi2
@pytest.mark.parametrize("objective_kind", ["tda", "multi"])
@pytest.mark.parametrize("sampling_interval", [1, 2])
def test_native_2d_indexed_target_on_chunk_boundary_matches_directional_fd(
    objective_kind,
    sampling_interval,
):
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")
    mp.verbosity(0)
    objective = _make_problem(
        objective_kind,
        mp.Ez,
        sampling_interval=sampling_interval,
        target_position=mp.Vector3(0.0, 0.8),
        chunk_layout=mp.BinaryPartition(data=[(mp.X, 0.0), 0, 1]),
    )
    rng = np.random.default_rng(1204)
    design = rng.uniform(0.15, 0.85, 25)
    direction = rng.uniform(0.2, 1.0, 25)
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
    assert relative_error < 0.006


@pytest.mark.mpi2
def test_full_native_loop_accepts_rank_without_owned_design_points():
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")
    mp.verbosity(0)
    objective = _make_problem(
        "tda",
        mp.Ez,
        sampling_interval=2,
        chunk_layout=mp.BinaryPartition(data=[(mp.X, 2.0), 0, 1]),
        t_final=10.0,
    )
    simulation = objective._simulation_spec.make()
    native_field = None
    try:
        simulation.init_sim()
        native_field = NativeDesignField(
            simulation,
            mp.Ez,
            objective.design,
        )
        zero_rank_count = int(
            mp.sum_to_all(int(native_field.local_size == 0))
        )
    finally:
        if native_field is not None:
            native_field.release()
        simulation.reset_meep()

    value, gradient = objective.fom_and_grad(
        np.linspace(0.2, 0.8, 25)
    )

    assert zero_rank_count >= 1
    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))
