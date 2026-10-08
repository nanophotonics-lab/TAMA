from contextlib import contextmanager
from pathlib import Path

import meep as mp
import numpy as np
import pytest
import scipy.interpolate as spi

import tama as tm
import tama.multi_tda_objective as multi_tda_module
from tama.sampling_grid import FastPointMonitor as _NativePointMonitor


class _LegacyPointMonitor:
    def __init__(self, sim, component, positions):
        self.sim = sim
        self.component = component
        self.positions = positions
        self._native_monitor = _NativePointMonitor(
            sim,
            component,
            positions,
        )

    def sample(self):
        return np.asarray(
            [
                self.sim.get_field_point(self.component, position)
                for position in self.positions
            ]
        )

    def sample_history_into(self, destination):
        values = self.sample()
        destination[:] = (
            values.real if np.issubdtype(destination.dtype, np.floating) else values
        )

    @staticmethod
    def reduce_history(history):
        return history

    def indexed_transpose_stencil(self):
        return self._native_monitor.indexed_transpose_stencil()


def _legacy_cubic_source(times, values):
    values = np.asarray(values)
    is_real = not np.iscomplexobj(values) or not np.any(values.imag != 0.0)
    interpolation = spi.interp1d(
        times,
        values.real if is_real else values,
        kind="cubic",
        fill_value=0.0 if is_real else 0j,
        bounds_error=False,
    )
    if is_real:
        return lambda time: float(interpolation(time))
    return lambda time: complex(interpolation(time))


def _legacy_cubic_sources(times, values):
    values = np.asarray(values)
    return tuple(
        _legacy_cubic_source(times, values[:, channel])
        for channel in range(values.shape[1])
    )


def _legacy_shifted_cubic_source(source, time_shift):
    return lambda time: source(time - time_shift)


def _legacy_forward_loop(
    sim,
    monitor_bindings,
    native_history,
    *,
    run_until,
    sampling_interval,
    fine_step_count,
):
    """Reference the former per-step Python forward-recording path."""
    state = {"sample_count": 0, "design_count": 0}
    monitor_times = []
    design_capacity = fine_step_count // sampling_interval + 1

    def record(active_simulation):
        sample_index = state["sample_count"]
        for monitor, history in monitor_bindings:
            monitor.sample_history_into(history[sample_index])
        monitor_times.append(active_simulation.round_time())
        if native_history is not None and sample_index % sampling_interval == 0:
            native_history.sample_forward(
                active_simulation,
                state["design_count"],
                design_capacity,
            )
            state["design_count"] += 1
        state["sample_count"] += 1

    sim.run(record, until=run_until)
    assert state["sample_count"] == fine_step_count + 1
    return (
        state["sample_count"],
        state["design_count"],
        np.asarray(monitor_times, dtype=float),
        0.0,
    )


@contextmanager
def _legacy_python_paths():
    original_monitor = multi_tda_module.FastPointMonitor
    original_source = multi_tda_module._tabulated_cubic_source
    original_sources = multi_tda_module._tabulated_cubic_sources
    original_shift = multi_tda_module._shift_tabulated_cubic_source
    original_forward_loop = multi_tda_module._run_native_forward_loop
    try:
        multi_tda_module.FastPointMonitor = _LegacyPointMonitor
        multi_tda_module._tabulated_cubic_source = _legacy_cubic_source
        multi_tda_module._tabulated_cubic_sources = _legacy_cubic_sources
        multi_tda_module._shift_tabulated_cubic_source = _legacy_shifted_cubic_source
        multi_tda_module._run_native_forward_loop = _legacy_forward_loop
        yield
    finally:
        multi_tda_module.FastPointMonitor = original_monitor
        multi_tda_module._tabulated_cubic_source = original_source
        multi_tda_module._tabulated_cubic_sources = original_sources
        multi_tda_module._shift_tabulated_cubic_source = original_shift
        multi_tda_module._run_native_forward_loop = original_forward_loop


def _make_objective(
    reuse_simulation=False,
    use_simulation_spec=False,
    max_frequency=None,
    t_final=1.0,
    reconstruction_window="kaiser",
    source_amplitude=1.0,
    adjoint_source_amplitude=1.0,
    force_complex_fields=False,
    mixed_targets=False,
    regional_targets=False,
    flux_targets=False,
    eigenmode_targets=False,
    flux_target_positions=None,
    flux_target_normals=None,
    regional_target_positions=None,
    band_indices=None,
    target_history_block_size=None,
    full_vector_sources=False,
):
    resolution = 8
    dt = 0.5 / resolution
    cell_size = mp.Vector3(6, 6)
    source_position = mp.Vector3(-1.1, -0.3)
    monitor_positions = [mp.Vector3(0.7, -0.4), mp.Vector3(0.7, 0.4)]
    layout = (
        mp.BinaryPartition(data=[(mp.X, 0.0), 0, 1])
        if mp.count_processors() == 2
        else None
    )
    forward_source_components = (
        (mp.Ez, mp.Hz) if full_vector_sources else (mp.Hz if mixed_targets else mp.Ez,)
    )
    forward_sources = [
        mp.Source(
            mp.GaussianSource(frequency=1.5, fwidth=1.0),
            component=source_component,
            center=source_position,
            amplitude=source_amplitude,
        )
        for source_component in forward_source_components
    ]
    air = mp.Medium(epsilon=1.0)
    dielectric = mp.Medium(epsilon=2.0)
    design_center = mp.Vector3()
    design_size = mp.Vector3(1.0, 1.0)
    material_grid = mp.MaterialGrid(
        mp.Vector3(2, 2),
        air,
        dielectric,
        do_averaging=False,
    )
    design = tm.DesignGrid(
        material_grid=material_grid,
        center=design_center,
        size=design_size,
        shape=(2, 2),
        background=air,
        design_material=dielectric,
    )
    geometry = [
        mp.Block(
            center=design_center,
            size=design_size,
            material=material_grid,
        )
    ]

    def sim_factory(sources=None):
        return mp.Simulation(
            cell_size=cell_size,
            resolution=resolution,
            chunk_layout=layout,
            geometry=geometry,
            sources=forward_sources if sources is None else sources,
            force_complex_fields=force_complex_fields,
            eps_averaging=False,
        )

    simulation_args = (
        {
            "simulation": tm.SimulationSpec(
                cell_size=cell_size,
                geometry=geometry,
                sources=forward_sources,
                resolution=resolution,
                chunk_layout=layout,
                eps_averaging=False,
            )
        }
        if (
            reuse_simulation
            or use_simulation_spec
            or regional_targets
            or flux_targets
            or eigenmode_targets
        )
        else {"sim_factory": sim_factory}
    )
    selected_band_indices = (
        tuple(range(2)) if band_indices is None else tuple(band_indices)
    )
    flux_target_positions = (
        (
            (mp.Vector3(0.0, -0.4),),
            (mp.Vector3(0.0, 0.4),),
        )
        if flux_target_positions is None
        else flux_target_positions
    )
    regional_spatial_weights = (
        ((0.75, 1.25), (0.5, 1.0, 1.5))
        if regional_target_positions is None
        else tuple(
            tuple(1.0 for _ in positions) for positions in regional_target_positions
        )
    )
    regional_target_positions = (
        (
            (mp.Vector3(0.0, -0.5), mp.Vector3(0.12, -0.35)),
            (
                mp.Vector3(0.0, 0.25),
                mp.Vector3(-0.1, 0.4),
                mp.Vector3(0.1, 0.5),
            ),
        )
        if regional_target_positions is None
        else regional_target_positions
    )
    target_args = (
        {
            "targets": [
                tm.EigenmodeCoefficientTarget(
                    positions=(
                        mp.Vector3(0.7, -0.4),
                        mp.Vector3(0.7, 0.4),
                    ),
                    normal=mp.Vector3(1.0),
                    mode_region=mp.Volume(
                        center=mp.Vector3(0.7),
                        size=mp.Vector3(0.0, 4.0),
                    ),
                    frequency=(1.65, 1.25)[band_index],
                    spatial_weights=(0.5, 0.5),
                    eig_parity=mp.ODD_Z,
                )
                for band_index in selected_band_indices
            ]
        }
        if eigenmode_targets
        else (
            {
                "targets": [
                    tm.FluxTarget(
                        flux_target_positions[band_index],
                        normal=(
                            mp.Vector3(1.0)
                            if flux_target_normals is None
                            else flux_target_normals[band_index]
                        ),
                    )
                    for band_index in selected_band_indices
                ]
            }
            if flux_targets
            else (
                {
                    "targets": [
                        tm.PointTarget(
                            monitor_positions[0],
                            mp.Ex,
                            adjoint_source_amplitude=1.0,
                        ),
                        tm.PointTarget(
                            monitor_positions[1],
                            mp.Ey,
                            adjoint_source_amplitude=1.0,
                        ),
                    ]
                }
                if mixed_targets
                else (
                    {
                        "targets": [
                            tm.FieldRegionTarget(
                                regional_target_positions[band_index],
                                mp.Ez,
                                sample_shape=(
                                    len(regional_target_positions[band_index]),
                                ),
                                spatial_weights=regional_spatial_weights[band_index],
                            )
                            for band_index in selected_band_indices
                        ]
                    }
                    if regional_targets
                    else {
                        "monitor_positions": monitor_positions,
                        "component": mp.Ez,
                        "adjoint_source_amplitude": adjoint_source_amplitude,
                    }
                )
            )
        )
    )
    return tm.MultiTDAObjective(
        design=design,
        t_final=t_final,
        wavelength_bands=[
            ((0.5, 0.7), (0.7, 0.9))[band_index] for band_index in selected_band_indices
        ],
        weights=[(1.0, 0.8)[band_index] for band_index in selected_band_indices],
        kernel_length=5,
        pixel_chunk=2,
        dt=dt,
        max_frequency=max_frequency,
        reconstruction_window=reconstruction_window,
        reuse_simulation=reuse_simulation,
        target_history_block_size=target_history_block_size,
        **simulation_args,
        **target_args,
    )


def _use_serial_target_transforms(objective):
    objective._distributed_filter_monitor_signals = objective.filter_monitor_signals

    def serial_transform(signals, transform, *, output_time_count=None):
        if output_time_count is None:
            output_time_count = signals.shape[0]
        result = np.empty(
            (output_time_count, signals.shape[1]),
            dtype=signals.dtype,
        )
        for target_index, target_slice in enumerate(objective._target_slices):
            transformed = np.asarray(transform(target_index))
            result[:, target_slice] = transformed.reshape(
                transformed.shape[0],
                -1,
            )
        return result

    objective._distributed_target_transform = serial_transform


@pytest.mark.parametrize("target_kind", ("regional", "flux"))
def test_blockwise_builtin_target_history_matches_serial_legacy(target_kind):
    if mp.count_processors() != 1:
        pytest.skip("serial parity check")
    target_args = (
        {"regional_targets": True}
        if target_kind == "regional"
        else {"flux_targets": True, "full_vector_sources": True}
    )
    x = np.full(4, 0.5)
    reference_value, reference_gradient = _make_objective(
        use_simulation_spec=True,
        **target_args,
    ).evaluate(x, need_gradient=True)
    blockwise = _make_objective(
        use_simulation_spec=True,
        target_history_block_size=1,
        **target_args,
    )
    value, gradient = blockwise.evaluate(x, need_gradient=True)

    assert blockwise.last_target_history_mode == "distributed"
    assert blockwise.last_target_history_layout["peak_work_columns"] <= 1
    assert np.allclose(value, reference_value, rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(
        gradient,
        reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_custom_regional_callback_keeps_legacy_history_path():
    if mp.count_processors() != 1:
        pytest.skip("serial fallback check")
    x = np.full(4, 0.5)
    regions = (
        (mp.Vector3(-0.4, -0.2), mp.Vector3(-0.4, 0.2)),
        (mp.Vector3(0.4, -0.2), mp.Vector3(0.4, 0.2)),
    )
    reference_value, reference_gradient = _make_objective(
        use_simulation_spec=True,
        regional_targets=True,
        regional_target_positions=regions,
    ).evaluate(x, need_gradient=True)
    objective = _make_objective(
        use_simulation_spec=True,
        regional_targets=True,
        regional_target_positions=regions,
        target_history_block_size=1,
    )
    objective.fom_fn = lambda history, dt: (0.5 * dt * np.sum(np.abs(history) ** 2))
    objective.adjoint_signal_fn = lambda history, dt: np.conjugate(history)
    value, gradient = objective.evaluate(x, need_gradient=True)

    assert objective.last_target_history_mode == "legacy"
    assert objective.last_target_history_layout is None
    assert np.allclose(value, reference_value, rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(
        gradient,
        reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_distributed_target_failure_cleans_native_history_memmaps():
    if mp.count_processors() != 1:
        pytest.skip("serial cleanup check")
    objective = _make_objective(
        use_simulation_spec=True,
        regional_targets=True,
        target_history_block_size=1,
    )
    history_paths = []
    make_history_memmap = objective._make_history_memmap

    def tracked_memmap(shape, *, dtype=None):
        history, path = make_history_memmap(shape, dtype=dtype)
        history_paths.append(path)
        return history, path

    objective._make_history_memmap = tracked_memmap
    transform = objective._distributed_builtin_band_objectives

    def fail_target_transform(layout):
        del layout
        raise RuntimeError("injected target failure")

    objective._distributed_builtin_band_objectives = fail_target_transform
    with pytest.raises(RuntimeError, match="injected target failure"):
        objective.evaluate(np.full(4, 0.5), need_gradient=True)

    assert history_paths
    assert all(not Path(path).exists() for path in history_paths)
    objective._distributed_builtin_band_objectives = transform
    assert np.isfinite(objective.fom(np.full(4, 0.5)))


@pytest.mark.mpi2
@pytest.mark.parametrize("eigenmode_targets", (False, True), ids=("band", "modal"))
@pytest.mark.parametrize("failure_stage", ("fom", "adjoint"))
def test_multi_tda_rank_local_callback_failure_is_synchronized(
    eigenmode_targets, failure_stage
):
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")

    objective = _make_objective(
        use_simulation_spec=True,
        eigenmode_targets=eigenmode_targets,
    )

    def fom_fn(history, dt):
        if failure_stage == "fom" and mp.my_rank() == 1:
            raise ValueError("intentional FoM callback failure")
        return float(0.5 * np.sum(np.abs(history) ** 2) * dt)

    def adjoint_signal_fn(history, dt):
        if failure_stage == "adjoint" and mp.my_rank() == 1:
            raise ValueError("intentional adjoint callback failure")
        return np.conjugate(history)

    objective.fom_fn = fom_fn
    objective.adjoint_signal_fn = adjoint_signal_fn
    expected_error = ValueError if mp.my_rank() == 1 else RuntimeError
    message = "intentional.*callback failure" if mp.my_rank() == 1 else "another.*rank"
    with pytest.raises(expected_error, match=message):
        objective.fom_and_grad(np.full(4, 0.5))


@pytest.mark.mpi2
@pytest.mark.parametrize("need_gradient", (False, True))
def test_multi_tda_rank_local_scalarization_failure_is_synchronized(need_gradient):
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")

    def scalarization(values):
        if mp.my_rank() == 1:
            raise ValueError("intentional scalarization callback failure")
        return float(np.sum(values)), np.ones_like(values)

    objective = _make_objective(use_simulation_spec=True)
    objective.scalarization_fn = scalarization
    expected_error = ValueError if mp.my_rank() == 1 else RuntimeError
    message = "intentional.*callback failure" if mp.my_rank() == 1 else "another.*rank"
    with pytest.raises(expected_error, match=message):
        objective.evaluate(np.full(4, 0.5), need_gradient=need_gradient)


@pytest.mark.mpi4
@pytest.mark.mpi8
def test_multi_tda_scalarization_failure_stays_within_active_process_group():
    if mp.comm.Get_size() not in (4, 8):
        pytest.skip("requires four or eight MPI ranks")

    group_index = mp.divide_parallel_processes(2)
    try:

        def scalarization(values):
            if group_index == 0 and mp.my_rank() == 1:
                raise ValueError("intentional subgroup callback failure")
            return float(np.sum(values)), np.ones_like(values)

        objective = _make_objective(use_simulation_spec=True)
        objective.scalarization_fn = scalarization
        if group_index == 0:
            expected_error = ValueError if mp.my_rank() == 1 else RuntimeError
            with pytest.raises(expected_error, match="callback|another.*rank"):
                objective.fom_and_grad(np.full(4, 0.5))
        else:
            value, gradient = objective.fom_and_grad(np.full(4, 0.5))
            assert np.isfinite(value)
            assert np.all(np.isfinite(gradient))
    finally:
        mp.end_divide_parallel()


@pytest.mark.mpi2
def test_native_multi_tda_per_band_callbacks_match_defaults():
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")

    reference_value, reference_gradient = _make_objective().evaluate(
        np.zeros(4),
        need_gradient=True,
    )

    objective = _make_objective()
    objective.fom_fn = (
        lambda history, dt: 0.5 * dt * np.sum(np.abs(history) ** 2),
        None,
    )
    objective.adjoint_signal_fn = (
        lambda history, dt: np.conjugate(history),
        None,
    )
    value, gradient = objective.evaluate(np.zeros(4), need_gradient=True)

    assert np.allclose(value, reference_value, rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(
        gradient,
        reference_gradient,
        rtol=1.0e-11,
        atol=1.0e-12,
    )


@pytest.mark.mpi2
def test_multi_tda_rejects_rank_inconsistent_per_band_callbacks():
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")

    callback = lambda history, dt: 0.5 * dt * np.sum(np.abs(history) ** 2)
    objective = _make_objective()
    objective.fom_fn = (callback, None) if mp.my_rank() == 0 else (None, callback)

    with pytest.raises(RuntimeError, match="fom_fn.*each band"):
        objective._validate_runtime_band_callbacks(validate_pairs=False)


@pytest.mark.mpi2
def test_native_multi_tda_paths_match_legacy_python_paths():
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")

    legacy_objective = _make_objective()
    _use_serial_target_transforms(legacy_objective)
    with _legacy_python_paths():
        legacy_value, legacy_gradient = legacy_objective.evaluate(
            np.zeros(4),
            need_gradient=True,
        )

    native_value, native_gradient = _make_objective().evaluate(
        np.zeros(4),
        need_gradient=True,
    )

    assert np.allclose(native_value, legacy_value, rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(
        native_gradient,
        legacy_gradient,
        rtol=1.0e-11,
        atol=1.0e-12,
    )

    regional_value, regional_gradient = _make_objective(
        regional_targets=True,
        use_simulation_spec=True,
    ).evaluate(
        np.zeros(4),
        need_gradient=True,
    )
    reused_value, reused_gradient = _make_objective(
        regional_targets=True,
        reuse_simulation=True,
    ).evaluate(
        np.zeros(4),
        need_gradient=True,
    )

    assert np.allclose(
        reused_value,
        regional_value,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert np.allclose(
        reused_gradient,
        regional_gradient,
        rtol=1.0e-11,
        atol=1.0e-12,
    )

    separate_value, separate_gradient = _make_objective(
        eigenmode_targets=True,
        use_simulation_spec=True,
    ).evaluate(
        np.zeros(4),
        need_gradient=True,
    )
    reused_value, reused_gradient = _make_objective(
        eigenmode_targets=True,
        reuse_simulation=True,
    ).evaluate(
        np.zeros(4),
        need_gradient=True,
    )

    assert np.allclose(
        reused_value,
        separate_value,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert np.allclose(
        reused_gradient,
        separate_gradient,
        rtol=1.0e-11,
        atol=1.0e-12,
    )


@pytest.mark.mpi2
def test_multiple_eigenmode_targets_stream_and_match_independent_bands(
    monkeypatch,
):
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")

    native_forward_loop = multi_tda_module._run_native_forward_loop
    monitor_widths = []

    def record_monitor_widths(sim, monitor_bindings, native_history, **kwargs):
        monitor_widths.append(
            tuple(history.shape[1] for _, history in monitor_bindings)
        )
        return native_forward_loop(
            sim,
            monitor_bindings,
            native_history,
            **kwargs,
        )

    def reject_raw_point_histories(*args, **kwargs):
        raise AssertionError("raw point-monitor histories were allocated")

    monkeypatch.setattr(
        multi_tda_module,
        "_run_native_forward_loop",
        record_monitor_widths,
    )
    monkeypatch.setattr(
        multi_tda_module,
        "FastPointMonitor",
        reject_raw_point_histories,
    )

    scalarization_weights = np.asarray((0.4, 1.7))
    combined_objective = _make_objective(
        eigenmode_targets=True,
        use_simulation_spec=True,
    )
    combined_objective.scalarization_fn = lambda values: (
        float(scalarization_weights @ values),
        scalarization_weights,
    )
    combined_value, combined_gradient = combined_objective.evaluate(
        np.zeros(4),
        need_gradient=True,
    )
    independent_results = [
        _make_objective(
            eigenmode_targets=True,
            use_simulation_spec=True,
            band_indices=(band_index,),
        ).evaluate(
            np.zeros(4),
            need_gradient=True,
        )
        for band_index in range(2)
    ]

    assert monitor_widths[0] == (2, 2)
    assert np.allclose(
        combined_value,
        sum(
            weight * value
            for weight, (value, _) in zip(
                scalarization_weights,
                independent_results,
            )
        ),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert np.allclose(
        combined_gradient,
        sum(
            weight * gradient
            for weight, (_, gradient) in zip(
                scalarization_weights,
                independent_results,
            )
        ),
        rtol=1.0e-11,
        atol=1.0e-12,
    )


@pytest.mark.mpi2
def test_native_mixed_target_groups_match_legacy_python_paths():
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")

    legacy_objective = _make_objective(mixed_targets=True)
    _use_serial_target_transforms(legacy_objective)
    with _legacy_python_paths():
        legacy_value, legacy_gradient = legacy_objective.evaluate(
            np.zeros(4),
            need_gradient=True,
        )

    native_value, native_gradient = _make_objective(
        mixed_targets=True,
    ).evaluate(
        np.zeros(4),
        need_gradient=True,
    )

    assert np.allclose(native_value, legacy_value, rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(
        native_gradient,
        legacy_gradient,
        rtol=1.0e-11,
        atol=1.0e-12,
    )


@pytest.mark.mpi2
def test_simulation_spec_reuse_matches_separate_simulations():
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")

    separate_value, separate_gradient = _make_objective(
        use_simulation_spec=True,
    ).evaluate(
        np.zeros(4),
        need_gradient=True,
    )
    reused_value, reused_gradient = _make_objective(
        reuse_simulation=True,
    ).evaluate(
        np.zeros(4),
        need_gradient=True,
    )

    assert np.allclose(reused_value, separate_value, rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(
        reused_gradient,
        separate_gradient,
        rtol=1.0e-11,
        atol=1.0e-12,
    )


@pytest.mark.mpi2
def test_flux_target_reuse_matches_separate_simulations():
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")

    separate_value, separate_gradient = _make_objective(
        flux_targets=True,
        use_simulation_spec=True,
    ).evaluate(
        np.zeros(4),
        need_gradient=True,
    )
    reused_value, reused_gradient = _make_objective(
        flux_targets=True,
        reuse_simulation=True,
    ).evaluate(
        np.zeros(4),
        need_gradient=True,
    )

    assert np.allclose(
        reused_value,
        separate_value,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert np.allclose(
        reused_gradient,
        separate_gradient,
        rtol=1.0e-11,
        atol=1.0e-12,
    )


@pytest.mark.mpi2
def test_duplicate_regional_channels_match_independent_band_objectives():
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")

    shared_region = (
        mp.Vector3(0.0, -0.4),
        mp.Vector3(0.0, 0.4),
    )
    regional_target_positions = (shared_region, shared_region)
    combined = _make_objective(
        regional_targets=True,
        use_simulation_spec=True,
        regional_target_positions=regional_target_positions,
    )
    combined_value, combined_gradient = combined.evaluate(
        np.zeros(4),
        need_gradient=True,
    )

    independent_values = []
    independent_gradients = []
    for band_index in range(2):
        value, gradient = _make_objective(
            regional_targets=True,
            use_simulation_spec=True,
            regional_target_positions=regional_target_positions,
            band_indices=(band_index,),
        ).evaluate(
            np.zeros(4),
            need_gradient=True,
        )
        independent_values.append(value)
        independent_gradients.append(gradient)

    assert len(combined._flat_target_positions) == 4
    assert len(combined._monitor_target_positions) == 2
    assert np.allclose(
        combined_value,
        np.sum(independent_values),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert np.allclose(
        combined_gradient,
        np.sum(independent_gradients, axis=0),
        rtol=1.0e-10,
        atol=1.0e-12,
    )


@pytest.mark.mpi2
def test_duplicate_flux_channels_match_independent_band_objectives():
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")

    shared_surface = (
        mp.Vector3(0.0, -0.4),
        mp.Vector3(0.0, 0.4),
    )
    flux_target_positions = (shared_surface, shared_surface)
    combined = _make_objective(
        flux_targets=True,
        use_simulation_spec=True,
        flux_target_positions=flux_target_positions,
    )
    combined_value, combined_gradient = combined.evaluate(
        np.zeros(4),
        need_gradient=True,
    )

    independent_values = []
    independent_gradients = []
    for band_index in range(2):
        value, gradient = _make_objective(
            flux_targets=True,
            use_simulation_spec=True,
            flux_target_positions=flux_target_positions,
            band_indices=(band_index,),
        ).evaluate(
            np.zeros(4),
            need_gradient=True,
        )
        independent_values.append(value)
        independent_gradients.append(gradient)

    assert len(combined._flat_target_positions) == 16
    assert len(combined._monitor_target_positions) == 8
    assert np.allclose(
        combined_value,
        np.sum(independent_values),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert np.allclose(
        combined_gradient,
        np.sum(independent_gradients, axis=0),
        rtol=1.0e-10,
        atol=1.0e-12,
    )


@pytest.mark.mpi2
@pytest.mark.mpi4
@pytest.mark.mpi8
@pytest.mark.parametrize("target_kind", ("regional", "flux"))
def test_distributed_builtin_history_matches_legacy_and_tracks_storage(
    target_kind,
):
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")

    if target_kind == "regional":
        shared_region = (
            mp.Vector3(-0.7, -0.4),
            mp.Vector3(0.07, 0.0),
            mp.Vector3(0.7, 0.4),
        )
        target_args = {
            "regional_targets": True,
            "regional_target_positions": (shared_region, shared_region),
        }
    else:
        shared_surface = (
            mp.Vector3(-0.7, 0.0),
            mp.Vector3(0.0, 0.0),
            mp.Vector3(0.7, 0.0),
        )
        target_args = {
            "flux_targets": True,
            "flux_target_positions": (shared_surface, shared_surface),
            "flux_target_normals": (mp.Vector3(0.0, 1.0),) * 2,
            "full_vector_sources": True,
        }

    x = np.full(4, 0.5)
    reference = _make_objective(
        use_simulation_spec=True,
        **target_args,
    )
    reference_value, reference_gradient = reference.evaluate(
        x,
        need_gradient=True,
    )
    distributed = _make_objective(
        use_simulation_spec=True,
        target_history_block_size=1,
        **target_args,
    )
    value, gradient = distributed.evaluate(x, need_gradient=True)

    assert distributed.last_target_history_mode == "distributed"
    layout = distributed.last_target_history_layout
    assert layout["peak_work_columns"] <= 1
    assert layout["replicated_channels"] > 0
    total_stored_channels = int(mp.sum_to_all(layout["stored_channels"]))
    assert (
        int(mp.sum_to_all(layout["local_unique_channels"]))
        + layout["replicated_channels"]
        == layout["total_channels"]
    )
    assert layout["stored_channels"] == (
        layout["local_unique_channels"] + layout["replicated_channels"]
    )
    assert layout["allocated_history_bytes"] == (
        layout["allocated_rows"]
        * layout["stored_channels"]
        * layout["history_itemsize"]
    )
    assert layout["legacy_history_bytes"] == (
        layout["allocated_rows"] * layout["total_channels"] * layout["history_itemsize"]
    )
    if mp.count_processors() == 2:
        assert total_stored_channels < (
            mp.count_processors() * layout["total_channels"]
        )
    if target_kind == "flux":
        assert layout["replicated_channels"] >= 4
    assert np.allclose(value, reference_value, rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(
        gradient,
        reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-12,
    )


@pytest.mark.mpi2
def test_distributed_flux_history_preserves_implicit_zero_components():
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")

    x = np.full(4, 0.5)
    reference_value, reference_gradient = _make_objective(
        flux_targets=True,
        use_simulation_spec=True,
    ).evaluate(x, need_gradient=True)
    objective = _make_objective(
        flux_targets=True,
        use_simulation_spec=True,
        target_history_block_size=1,
    )
    value, gradient = objective.evaluate(x, need_gradient=True)

    layout = objective.last_target_history_layout
    assert layout["implicit_zero_channels"] > 0
    assert layout["stored_channels"] < layout["total_channels"]
    assert np.allclose(value, reference_value, rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(
        gradient,
        reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-12,
    )


@pytest.mark.mpi2
@pytest.mark.parametrize(
    "flux_target_normals",
    (
        (mp.Vector3(1.0), mp.Vector3(0.0, 1.0)),
        (mp.Vector3(0.0, 1.0), mp.Vector3(1.0)),
    ),
)
def test_overlapping_flux_ownership_closure_is_target_order_independent(
    flux_target_normals,
):
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")

    shared_point = ((mp.Vector3(),),) * 2
    objective = _make_objective(
        flux_targets=True,
        flux_target_positions=shared_point,
        flux_target_normals=flux_target_normals,
        use_simulation_spec=True,
        target_history_block_size=1,
    )

    class Monitor:
        def configure_history_indices(self, indices):
            self.indices = np.asarray(indices)

    groups = []
    for component, target_indices in objective._monitor_groups:
        owner = 1 if component == mp.Ex else 0
        local_support = np.full(
            len(target_indices),
            mp.my_rank() == owner,
            dtype=np.intp,
        )
        groups.append(
            {
                "indices": np.asarray(target_indices, dtype=np.intp),
                "monitor": Monitor(),
                "indexed_stencil": (
                    np.concatenate(([0], np.cumsum(local_support))),
                    (),
                    np.empty(0),
                ),
            }
        )

    layout = objective._configure_distributed_monitor_histories(
        groups,
        row_count=3,
        dtype=np.float64,
    )

    assert layout["total_channels"] == 6
    assert layout["replicated_channels"] == 6
    assert layout["implicit_zero_channels"] == 0
    assert np.all(layout["owners"] == -1)


@pytest.mark.mpi2
def test_distributed_target_history_allows_an_empty_local_partition():
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")

    left_region = ((mp.Vector3(-0.7, -0.4),),) * 2
    x = np.full(4, 0.5)
    reference_value, reference_gradient = _make_objective(
        regional_targets=True,
        regional_target_positions=left_region,
        use_simulation_spec=True,
    ).evaluate(x, need_gradient=True)
    objective = _make_objective(
        regional_targets=True,
        regional_target_positions=left_region,
        use_simulation_spec=True,
        target_history_block_size=1,
    )
    value, gradient = objective.evaluate(x, need_gradient=True)

    layout = objective.last_target_history_layout
    assert int(mp.sum_to_all(layout["stored_channels"] == 0)) == 1
    assert int(mp.sum_to_all(layout["stored_channels"])) == 1
    assert np.allclose(value, reference_value, rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(
        gradient,
        reference_gradient,
        rtol=1.0e-10,
        atol=1.0e-12,
    )


@pytest.mark.mpi2
def test_distributed_source_construction_failure_is_synchronized():
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")

    objective = _make_objective(
        regional_targets=True,
        use_simulation_spec=True,
        target_history_block_size=1,
    )
    if mp.my_rank() == 1:

        def fail_source_construction(*args):
            del args
            raise ValueError("intentional source construction failure")

        objective._build_distributed_adjoint_sources = fail_source_construction

    with pytest.raises(
        (ValueError, RuntimeError),
        match="intentional|another MPI rank",
    ):
        objective.evaluate(np.full(4, 0.5), need_gradient=True)


@pytest.mark.mpi2
def test_distributed_history_configuration_failure_is_synchronized(
    monkeypatch,
):
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")

    if mp.my_rank() == 1:

        def fail_history_configuration(self, indices):
            del self, indices
            raise ValueError("intentional history configuration failure")

        monkeypatch.setattr(
            multi_tda_module.FastPointMonitor,
            "configure_history_indices",
            fail_history_configuration,
        )
    objective = _make_objective(
        regional_targets=True,
        use_simulation_spec=True,
        target_history_block_size=1,
    )

    with pytest.raises(
        (ValueError, RuntimeError),
        match="intentional|another MPI rank",
    ):
        objective.fom(np.full(4, 0.5))


@pytest.mark.mpi2
def test_distributed_reduction_workspace_failure_is_synchronized():
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")

    shared_region = (
        mp.Vector3(-0.7, -0.4),
        mp.Vector3(0.07, 0.0),
        mp.Vector3(0.7, 0.4),
    )
    objective = _make_objective(
        regional_targets=True,
        regional_target_positions=(shared_region, shared_region),
        use_simulation_spec=True,
        target_history_block_size=1,
    )
    if mp.my_rank() == 1:

        def fail_workspace_allocation(shape, dtype):
            del shape, dtype
            raise MemoryError("intentional reduction workspace failure")

        objective._make_distributed_reduction_scratch = fail_workspace_allocation

    with pytest.raises(
        (MemoryError, RuntimeError),
        match="intentional|another MPI rank",
    ):
        objective.fom(np.full(4, 0.5))


@pytest.mark.mpi2
@pytest.mark.parametrize("failure_stage", ("fom", "adjoint"))
def test_duplicate_flux_callback_failure_is_synchronized(failure_stage):
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")

    shared_surface = (
        mp.Vector3(0.0, -0.4),
        mp.Vector3(0.0, 0.4),
    )
    objective = _make_objective(
        flux_targets=True,
        use_simulation_spec=True,
        flux_target_positions=(shared_surface, shared_surface),
    )

    def fom_fn(power, dt):
        if failure_stage == "fom" and mp.my_rank() == 1:
            raise ValueError("intentional FoM callback failure")
        return np.sum(power) * dt

    def adjoint_signal_fn(power, dt):
        if failure_stage == "adjoint" and mp.my_rank() == 1:
            raise ValueError("intentional adjoint callback failure")
        return np.ones_like(power)

    objective.fom_fn = fom_fn
    objective.adjoint_signal_fn = adjoint_signal_fn
    with pytest.raises(
        (ValueError, RuntimeError),
        match="intentional|another MPI rank",
    ):
        objective.evaluate(np.zeros(4), need_gradient=True)


@pytest.mark.mpi2
def test_native_nyquist_path_matches_legacy_python_paths():
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")

    legacy_objective = _make_objective(max_frequency=3.0)
    _use_serial_target_transforms(legacy_objective)
    with _legacy_python_paths():
        legacy_value, legacy_gradient = legacy_objective.evaluate(
            np.zeros(4),
            need_gradient=True,
        )

    native_objective = _make_objective(max_frequency=3.0)
    native_value, native_gradient = native_objective.evaluate(
        np.zeros(4),
        need_gradient=True,
    )

    assert legacy_objective.last_sampling_interval == 2
    assert native_objective.last_sampling_interval == 2
    assert np.allclose(native_value, legacy_value, rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(
        native_gradient,
        legacy_gradient,
        rtol=1.0e-11,
        atol=1.0e-12,
    )


def test_multi_tda_nyquist_alignment_preserves_objective_value():
    design = np.zeros(4)
    full_value, _ = _make_objective().evaluate(
        design,
        need_gradient=False,
    )
    nyquist_objective = _make_objective(max_frequency=3.0)
    nyquist_value_with_gradient, _ = nyquist_objective.evaluate(
        design,
        need_gradient=True,
    )
    nyquist_value_only, _ = nyquist_objective.evaluate(
        design,
        need_gradient=False,
    )

    assert nyquist_objective.last_sampling_interval == 2
    assert np.allclose(
        nyquist_value_with_gradient,
        full_value,
        rtol=1.0e-12,
        atol=1.0e-15,
    )
    assert np.allclose(
        nyquist_value_only,
        full_value,
        rtol=1.0e-12,
        atol=1.0e-15,
    )


def test_multi_tda_nyquist_gradient_matches_full_rate_after_field_decay():
    design = np.zeros(4)
    _, full_gradient = _make_objective(t_final=8.0).evaluate(
        design,
        need_gradient=True,
    )
    _, sparse_gradient = _make_objective(
        max_frequency=3.0,
        t_final=8.0,
        reconstruction_window="hann",
    ).evaluate(
        design,
        need_gradient=True,
    )

    relative_error = np.linalg.norm(sparse_gradient - full_gradient) / np.linalg.norm(
        full_gradient
    )
    assert relative_error < 1.0e-3


def test_complex_source_phase_preserves_multi_tda_value_and_gradient():
    design = np.zeros(4)
    real_value, real_gradient = _make_objective(
        t_final=4.0,
        source_amplitude=1.0,
        force_complex_fields=True,
    ).evaluate(design, need_gradient=True)
    imaginary_value, imaginary_gradient = _make_objective(
        t_final=4.0,
        source_amplitude=1.0j,
        force_complex_fields=True,
    ).evaluate(design, need_gradient=True)

    assert np.linalg.norm(real_gradient) > 0.0
    assert np.allclose(imaginary_value, real_value, rtol=1.0e-12, atol=1.0e-15)
    assert np.allclose(
        imaginary_gradient,
        real_gradient,
        rtol=1.0e-11,
        atol=1.0e-12,
    )


def test_nonunit_adjoint_amplitude_is_rejected_before_meep_sources():
    amplitude = 0.6 + 0.8j
    objective = _make_objective(
        t_final=4.0,
        adjoint_source_amplitude=amplitude,
        force_complex_fields=True,
    )
    original_factory = objective.sim_factory
    adjoint_sources = []

    def recording_factory(sources=None):
        if sources is not None:
            adjoint_sources.extend(sources)
        return original_factory(sources)

    objective.sim_factory = recording_factory
    with pytest.raises(ValueError, match="requires adjoint_source_amplitude=1"):
        objective.evaluate(np.zeros(4), need_gradient=True)

    assert not adjoint_sources


if __name__ == "__main__":
    test_native_multi_tda_paths_match_legacy_python_paths()
    test_simulation_spec_reuse_matches_separate_simulations()
    test_native_nyquist_path_matches_legacy_python_paths()
    test_multi_tda_nyquist_alignment_preserves_objective_value()
    test_multi_tda_nyquist_gradient_matches_full_rate_after_field_decay()
    test_complex_source_phase_preserves_multi_tda_value_and_gradient()
    test_complex_adjoint_amplitude_reaches_forced_complex_meep_sources()
