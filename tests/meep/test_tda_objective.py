import numpy as np
import meep as mp
import pytest

import tama as tm
import tama.objectives as objectives_module
import tama.tda_objective as tda_module
from _objective_test_helpers import (
    _FakeFields,
    _fake_indexed_transpose_stencil,
    _run_fake_adjoint_loop,
)


class _FakeSimulation:
    def __init__(self, dt):
        self.fields = _FakeFields(dt)
        self.resolution = 10


class _OneSampleSimulation:
    def __init__(self):
        self.fields = _FakeFields(0.1)
        self.resolution = 10

    def run(self, callback, until):
        callback(self)

    def get_field_point(self, component, position):
        return 1.0

    def round_time(self):
        return 0.1

    def reset_meep(self):
        pass


class _FakeDesign:
    def __init__(
        self,
        shape=(4, 1),
        material_factor=2.0,
        is_cylindrical=False,
    ):
        self.shape = shape
        self.dimensions = len(shape)
        self.material_factor = material_factor
        self.is_cylindrical = is_cylindrical
        self.updates = []

    def update_weights(self, values):
        self.updates.append(np.array(values, copy=True))


class _MultiSampleSimulation:
    def __init__(self, dt, steps):
        self.fields = _FakeFields(dt)
        self.resolution = 10
        self.steps = steps
        self.step = 0

    def run(self, callback, until):
        for step in range(self.steps):
            self.step = step
            callback(self)

    def get_field_point(self, component, position):
        return complex(self.step + 1)

    def round_time(self):
        return self.steps * self.fields.dt

    def reset_meep(self):
        pass


class _FluxSampleSimulation(_MultiSampleSimulation):
    _component_scales = {
        mp.Ey: 2.0,
        mp.Hz: 3.0,
        mp.Ez: 5.0,
        mp.Hy: 7.0,
    }

    def get_field_point(self, component, position):
        return self._component_scales[component] * (self.step + 1)


class _CylindricalFluxSampleSimulation(_MultiSampleSimulation):
    _component_scales = {
        mp.Er: 2.0,
        mp.Hp: 3.0,
        mp.Ep: 5.0,
        mp.Hr: 7.0,
    }

    def get_field_point(self, component, position):
        return self._component_scales[component] * (self.step + 1)


class _ReusableSimulation(_MultiSampleSimulation):
    def __init__(self, dt, steps):
        super().__init__(dt, steps)
        self.events = []
        self.changed_sources = []

    def run(self, callback, until):
        self.events.append("run")
        super().run(callback, until)

    def restart_fields(self):
        self.events.append("restart_fields")
        self.step = 0

    def clear_dft_monitors(self):
        self.events.append("clear_dft_monitors")

    def change_sources(self, sources):
        self.events.append("change_sources")
        self.changed_sources.append(sources)

    def change_m(self, m):
        self.events.append(("change_m", m))
        self.m = m

    def reset_meep(self):
        self.events.append("reset_meep")


class _ReusableFluxSimulation(_ReusableSimulation):
    def get_field_point(self, component, position):
        return _FluxSampleSimulation._component_scales[component] * (
            self.step + 1
        )


class _TerminalSampleSimulation(_MultiSampleSimulation):
    def round_time(self):
        return (self.steps - 1) * self.fields.dt


class _FakeSimulationSpec:
    def __init__(self, factory, *, dimensions=2, m=0):
        self.factory = factory
        self.resolution = 10
        self.chunk_layout = "initial-layout"
        self.dimensions = dimensions
        self.m = m
        self.make_calls = []

    @property
    def is_cylindrical(self):
        return self.dimensions == mp.CYLINDRICAL

    def make(self, sources=None, *, m=None):
        selected_m = self.m if m is None else m
        self.make_calls.append(
            sources if m is None else (sources, selected_m)
        )
        simulation = self.factory(sources)
        simulation.dimensions = self.dimensions
        simulation.is_cylindrical = self.is_cylindrical
        simulation.m = selected_m
        return simulation


class _FakeAdaptiveBalancer:
    def __init__(self, source_boundary_decision=None, source_boundary_mode="auto"):
        self.prepared = []
        self.prepared_forward_sources = []
        self.prepared_sources = []
        self.prepared_indexed_masks = []
        self.captured = []
        self.rebalanced = []
        self.last_report = {"applied": True}
        self.source_boundary_decision = source_boundary_decision
        self.source_boundary_mode = source_boundary_mode

    def prepare_initial_layout(
        self,
        simulation_spec,
        *,
        forward_sources=(),
        adjoint_source_positions,
        adjoint_source_size=None,
        adjoint_source_amplitude=1.0,
        adjoint_source_sizes=None,
        adjoint_source_amplitudes=None,
        indexed_source_mask=None,
    ):
        self.prepared.append(simulation_spec)
        self.prepared_forward_sources.append(forward_sources)
        self.prepared_indexed_masks.append(indexed_source_mask)
        if adjoint_source_sizes is None:
            adjoint_source_sizes = (adjoint_source_size,) * len(
                adjoint_source_positions
            )
        if adjoint_source_amplitudes is None:
            adjoint_source_amplitudes = (adjoint_source_amplitude,) * len(
                adjoint_source_positions
            )
        self.prepared_sources.append(
            (
                tuple(adjoint_source_positions),
                (
                    adjoint_source_sizes
                    if len(adjoint_source_positions) > 1
                    else adjoint_source_sizes[0]
                ),
                (
                    adjoint_source_amplitudes
                    if len(adjoint_source_positions) > 1
                    else adjoint_source_amplitudes[0]
                ),
            )
        )
        if self.source_boundary_decision is not None:
            return self.source_boundary_decision
        return type(
            "SourceBoundaryDecision",
            (),
            {
                "source_sizes": tuple(adjoint_source_sizes),
                "source_amplitudes": tuple(adjoint_source_amplitudes),
            },
        )()

    def capture_timing(self, sim):
        timing = ("timing", len(self.captured))
        self.captured.append((sim, timing))
        return timing

    def rebalance(
        self,
        sim,
        *,
        local_extra_work_seconds,
        local_wall_seconds,
        previous_measurements,
    ):
        self.rebalanced.append(
            (sim, local_extra_work_seconds, local_wall_seconds, previous_measurements)
        )
        return "next-layout"


class _FakeNativeDesignHistorySet:
    instances = []

    def __init__(self, design, components, history_dtype, make_history_memmap):
        self.design = design
        self.components = tuple(components)
        self.history_dtype = history_dtype
        self.rows = {component: [] for component in self.components}
        self.sample_calls = 0
        self.release_calls = 0
        self.cleanup_calls = 0
        _FakeNativeDesignHistorySet.instances.append(self)

    def sample_forward(self, sim, row_index, row_capacity):
        self.sample_calls += 1
        width = int(np.prod(self.design.shape))
        for component in self.components:
            self.rows[component].append(
                np.full(width, row_index + 1, dtype=self.history_dtype)
            )

    def finish_forward(self, row_count):
        return {
            component: np.asarray(rows[:row_count], dtype=self.history_dtype)
            for component, rows in self.rows.items()
        }

    def width(self, component):
        return int(np.prod(self.design.shape))

    @property
    def signatures(self):
        return {
            component: np.asarray([component], dtype=np.int64)
            for component in self.components
        }

    def release_forward(self):
        self.release_calls += 1

    def cleanup_memmaps(self):
        self.cleanup_calls += 1


class _FakeNativeDesignAccumulator:
    instances = []

    def __init__(self, design, components, signatures, midpoint):
        self.design = design
        self.components = tuple(components)
        self.signatures = signatures
        self.midpoint = midpoint
        self.accumulated = []
        self.sample_steps = []
        self.release_calls = 0
        _FakeNativeDesignAccumulator.instances.append(self)

    def accumulate(self, sim, component, forward_derivative):
        self.sample_steps.append(sim.step)
        self.accumulated.append(np.array(forward_derivative, copy=True))

    def reduce(self):
        if not self.accumulated:
            return np.zeros(self.design.shape, dtype=np.complex128)
        return np.sum(self.accumulated, axis=0).reshape(self.design.shape)

    def release(self):
        self.release_calls += 1


def _install_fake_native_backend(monkeypatch):
    _FakeNativeDesignHistorySet.instances = []
    _FakeNativeDesignAccumulator.instances = []
    monkeypatch.setattr(
        tda_module,
        "_NativeDesignHistorySet",
        _FakeNativeDesignHistorySet,
    )
    monkeypatch.setattr(
        tda_module,
        "_NativeDesignAccumulator",
        _FakeNativeDesignAccumulator,
    )
    monkeypatch.setattr(
        tda_module,
        "_run_native_adjoint_loop",
        _run_fake_adjoint_loop,
    )
    monkeypatch.setattr(
        tda_module,
        "FastPointMonitor",
        _DeferredPointMonitor,
    )


class _DeferredPointMonitor:
    instances = []

    def __init__(self, sim, component, positions):
        self.sim = sim
        self.component = component
        self.positions = positions
        self.sample_calls = 0
        self.reduce_calls = 0
        self.reduced_shapes = []
        _DeferredPointMonitor.instances.append(self)

    def sample_history_into(self, destination):
        self.sample_calls += 1
        for index, position in enumerate(self.positions):
            destination[index] = (
                self.sim.get_field_point(self.component, position)
                + 0.25 * index
            )

    def reduce_history(self, history):
        self.reduce_calls += 1
        self.reduced_shapes.append(history.shape)
        return history

    def indexed_transpose_stencil(self):
        return _fake_indexed_transpose_stencil(self)


def _make_minimal_tda(**kwargs):
    design = _FakeDesign()
    params = dict(
        design=design,
        t_final=0.6,
        sim_factory=lambda sources=None: _ReusableSimulation(0.1, 6),
        monitor_position=mp.Vector3(),
        component=mp.Ez,
        dt=0.1,
        fom_fn=lambda history, dt: float(
            np.sum(np.abs(history) ** 2) * dt
        ),
        adjoint_signal_fn=lambda history, dt: np.ones_like(
            history,
            dtype=np.complex128,
        ),
    )
    params.update(kwargs)
    return tm.TDAObjective(**params)


@pytest.mark.parametrize(
    "t_final",
    (True, np.bool_(False), 0.0, -1.0, np.inf, -np.inf, np.nan),
)
def test_tda_requires_positive_finite_t_final(t_final):
    with pytest.raises(
        ValueError,
        match="t_final must be a positive finite number",
    ):
        _make_minimal_tda(t_final=t_final)


def test_tda_requires_fom_for_manual_adjoint_signal():
    with pytest.raises(ValueError, match="adjoint_signal_fn requires fom_fn"):
        _make_minimal_tda(
            fom_fn=None,
            adjoint_signal_fn=lambda history, dt: np.ones_like(history),
        )


def test_tda_rejects_manual_adjoint_signal_assigned_after_default_fom():
    obj = _make_minimal_tda(fom_fn=None, adjoint_signal_fn=None)
    obj.adjoint_signal_fn = lambda history, dt: np.ones_like(history)

    with pytest.raises(ValueError, match="adjoint_signal_fn requires fom_fn"):
        obj.evaluate(np.zeros(4), need_gradient=True)


def test_tda_rejects_fom_removed_after_manual_adjoint_signal():
    obj = _make_minimal_tda()
    obj.fom_fn = None

    with pytest.raises(ValueError, match="adjoint_signal_fn requires fom_fn"):
        obj.evaluate(np.zeros(4), need_gradient=True)


def test_tda_field_region_value_uses_spatial_shape_and_weights(monkeypatch):
    _install_fake_native_backend(monkeypatch)
    positions = (mp.Vector3(-0.2), mp.Vector3(0.2))
    target = tm.FieldRegionTarget(
        positions,
        mp.Ez,
        sample_shape=(1, 2),
        spatial_weights=(1.0, 3.0),
    )
    obj = tm.TDAObjective(
        update_design=lambda values: None,
        t_final=0.6,
        sim_factory=lambda sources=None: _MultiSampleSimulation(0.1, 6),
        target=target,
        dt=0.1,
    )

    value, gradient = obj.evaluate(np.zeros(1), need_gradient=False)

    history = np.column_stack(
        (
            np.arange(1.0, 7.0),
            np.arange(1.0, 7.0) + 0.25,
        )
    ).reshape(6, 1, 2)
    expected = (
        0.5
        * 0.1
        * np.sum(np.abs(history) ** 2 * np.asarray([1.0, 3.0]))
    )
    assert value == pytest.approx(expected)
    assert gradient is None


def test_tda_field_region_builds_point_specific_indexed_sources(monkeypatch):
    _install_fake_native_backend(monkeypatch)
    positions = (mp.Vector3(-0.2), mp.Vector3(0.2))
    target = tm.FieldRegionTarget(
        positions,
        mp.Ez,
        sample_shape=(2,),
        adjoint_source_amplitude=1.0,
    )
    simulation = _FakeSimulationSpec(
        lambda sources=None: _ReusableSimulation(0.1, 6)
    )
    balancer = _FakeAdaptiveBalancer()
    callback_shapes = []

    def fom_fn(history, dt):
        callback_shapes.append(history.shape)
        return float(np.sum(np.abs(history) ** 2) * dt)

    def adjoint_signal_fn(history, dt):
        callback_shapes.append(history.shape)
        return np.column_stack(
            (
                np.ones(history.shape[0]),
                2.0 * np.ones(history.shape[0]),
            )
        )

    obj = tm.TDAObjective(
        design=_FakeDesign(),
        simulation=simulation,
        target=target,
        t_final=0.6,
        dt=0.1,
        fom_fn=fom_fn,
        adjoint_signal_fn=adjoint_signal_fn,
        chunk_balancer=balancer,
    )

    obj.evaluate(np.zeros(4), need_gradient=True)

    assert callback_shapes == [(6, 2), (6, 2)]
    assert balancer.prepared_indexed_masks == [(True, True)]
    assert balancer.prepared_sources[0][0] == positions
    sources = simulation.make_calls[1]
    assert len(sources) == 2
    assert all(isinstance(source, mp.IndexedSource) for source in sources)
    assert all(np.array_equal(source.amp_arr, np.asarray([1.0])) for source in sources)


def test_tda_field_region_gradient_requires_simulation_spec():
    target = tm.FieldRegionTarget(
        (mp.Vector3(-0.2), mp.Vector3(0.2)),
        mp.Ez,
    )
    obj = tm.TDAObjective(
        design=_FakeDesign(),
        t_final=0.6,
        sim_factory=lambda sources=None: _ReusableSimulation(0.1, 6),
        target=target,
        dt=0.1,
    )

    with pytest.raises(ValueError, match="requires simulation=SimulationSpec"):
        obj.evaluate(np.zeros(4), need_gradient=True)


def test_tda_field_region_rejects_legacy_target_overrides():
    target = tm.FieldRegionTarget((mp.Vector3(),), mp.Ez)

    with pytest.raises(ValueError, match="legacy target arguments"):
        tm.TDAObjective(
            update_design=lambda values: None,
            t_final=0.6,
            sim_factory=lambda sources=None: _MultiSampleSimulation(0.1, 6),
            target=target,
            monitor_position=mp.Vector3(),
            dt=0.1,
        )


def test_tda_flux_target_value_exposes_total_power_history(monkeypatch):
    _DeferredPointMonitor.instances = []
    _install_fake_native_backend(monkeypatch)
    target = tm.FluxTarget(
        (mp.Vector3(),),
        normal=mp.Vector3(1.0),
    )
    callback_shapes = []

    def fom_fn(power, dt):
        callback_shapes.append(power.shape)
        return np.sum(power) * dt

    obj = tm.TDAObjective(
        update_design=lambda values: None,
        t_final=0.6,
        sim_factory=lambda sources=None: _FluxSampleSimulation(0.1, 6),
        target=target,
        dt=0.1,
        fom_fn=fom_fn,
    )

    value, gradient = obj.evaluate(np.zeros(1), need_gradient=False)

    step = np.arange(1.0, 6.0)
    expected_power = (2.0 * 3.0 - 5.0 * 7.0) * step * (step + 0.5)
    assert callback_shapes == [(5,)]
    assert value == pytest.approx(np.sum(expected_power) * 0.1)
    assert gradient is None
    assert sum(
        monitor.reduce_calls
        for monitor in _DeferredPointMonitor.instances
    ) == 1
    assert [
        shape
        for monitor in _DeferredPointMonitor.instances
        for shape in monitor.reduced_shapes
    ] == [(6, 4)]


def test_tda_flux_target_combines_electric_and_magnetic_adjoint_sources(
    monkeypatch,
):
    _install_fake_native_backend(monkeypatch)
    simulation = _FakeSimulationSpec(
        lambda sources=None: _FluxSampleSimulation(0.1, 6)
    )
    callback_shapes = []

    def fom_fn(power, dt):
        callback_shapes.append(power.shape)
        return np.sum(power) * dt

    def adjoint_signal_fn(power, dt):
        callback_shapes.append(power.shape)
        return np.ones_like(power)

    obj = tm.TDAObjective(
        design=_FakeDesign(),
        simulation=simulation,
        target=tm.FluxTarget(
            (mp.Vector3(),),
            normal=mp.Vector3(1.0),
        ),
        t_final=0.6,
        dt=0.1,
        fom_fn=fom_fn,
        adjoint_signal_fn=adjoint_signal_fn,
    )

    _, gradient = obj.evaluate(np.zeros(4), need_gradient=True)

    assert callback_shapes == [(5,), (5,)]
    assert obj.gradient_components == (mp.Ex, mp.Ey, mp.Ez)
    assert len(simulation.make_calls) == 2
    assert len(simulation.make_calls[1]) == 6
    assert all(
        isinstance(source, mp.IndexedSource)
        for sources in simulation.make_calls[1:]
        for source in sources
    )
    combined_sources = simulation.make_calls[1]
    assert np.allclose(
        [source.amp_arr[0] for source in combined_sources],
        [1.0, 0.5, 0.5, 1.0, 0.5, 0.5],
    )
    sample_times = (0.2, 0.3, 0.4)
    for base_index, delayed_index in ((1, 2), (4, 5)):
        base_source = combined_sources[base_index].src.src_func
        delayed_source = combined_sources[delayed_index].src.src_func
        assert np.allclose(
            [delayed_source(time + 0.1) for time in sample_times],
            [base_source(time) for time in sample_times],
        )
    assert [
        accumulator.midpoint
        for accumulator in _FakeNativeDesignAccumulator.instances
    ] == [False]
    assert gradient.shape == (4,)


def test_tda_flux_target_reuses_simulation_for_combined_adjoint(monkeypatch):
    _install_fake_native_backend(monkeypatch)
    simulation_instance = _ReusableFluxSimulation(0.1, 6)
    simulation = _FakeSimulationSpec(
        lambda sources=None: simulation_instance
    )
    obj = tm.TDAObjective(
        design=_FakeDesign(),
        simulation=simulation,
        target=tm.FluxTarget(
            (mp.Vector3(),),
            normal=mp.Vector3(1.0),
        ),
        t_final=0.6,
        dt=0.1,
        reuse_simulation=True,
    )

    _, gradient = obj.evaluate(np.zeros(4), need_gradient=True)

    assert simulation.make_calls == [None]
    assert simulation_instance.events == [
        "run",
        "restart_fields",
        "clear_dft_monitors",
        "change_sources",
        "run",
        "reset_meep",
    ]
    assert len(simulation_instance.changed_sources) == 1
    assert gradient.shape == (4,)


def test_tda_flux_target_balances_all_indexed_component_sources(
    monkeypatch,
):
    _install_fake_native_backend(monkeypatch)
    simulation = _FakeSimulationSpec(
        lambda sources=None: _FluxSampleSimulation(0.1, 6)
    )
    balancer = _FakeAdaptiveBalancer()
    obj = tm.TDAObjective(
        design=_FakeDesign(),
        simulation=simulation,
        target=tm.FluxTarget(
            (mp.Vector3(),),
            normal=mp.Vector3(1.0),
        ),
        t_final=0.6,
        dt=0.1,
        chunk_balancer=balancer,
    )

    obj.evaluate(np.zeros(4), need_gradient=True)

    assert balancer.prepared_indexed_masks == [(True, True, True, True)]
    assert len(balancer.prepared_sources[0][0]) == 4
    assert len(balancer.captured) == 1
    assert len(balancer.rebalanced) == 1
    assert len(balancer.rebalanced[0][3]) == 1
    assert simulation.chunk_layout == "next-layout"


def test_tda_flux_target_gradient_requires_simulation_spec():
    obj = tm.TDAObjective(
        design=_FakeDesign(),
        t_final=0.6,
        sim_factory=lambda sources=None: _FluxSampleSimulation(0.1, 6),
        target=tm.FluxTarget(
            (mp.Vector3(),),
            normal=mp.Vector3(1.0),
        ),
        dt=0.1,
    )

    with pytest.raises(ValueError, match="requires simulation=SimulationSpec"):
        obj.evaluate(np.zeros(4), need_gradient=True)


def test_tda_flux_target_rejects_out_of_plane_2d_normal():
    with pytest.raises(ValueError, match="x-y plane"):
        tm.TDAObjective(
            design=_FakeDesign(),
            simulation=_FakeSimulationSpec(
                lambda sources=None: _FluxSampleSimulation(0.1, 6)
            ),
            target=tm.FluxTarget(
                (mp.Vector3(),),
                normal=mp.Vector3(0.0, 0.0, 1.0),
            ),
            t_final=0.6,
            dt=0.1,
        )


def test_tda_flux_value_rejects_out_of_plane_2d_normal_at_runtime():
    simulation = _FluxSampleSimulation(0.1, 6)
    simulation.cell_size = mp.Vector3(2.0, 2.0)
    obj = tm.TDAObjective(
        update_design=lambda values: None,
        sim_factory=lambda sources=None: simulation,
        target=tm.FluxTarget(
            (mp.Vector3(),),
            normal=mp.Vector3(0.0, 0.0, 1.0),
        ),
        t_final=0.6,
        dt=0.1,
    )

    with pytest.raises(ValueError, match="x-y plane"):
        obj.evaluate(np.zeros(1), need_gradient=False)


def _make_cylindrical_flux_simulation():
    simulation = _CylindricalFluxSampleSimulation(0.1, 6)
    simulation.dimensions = mp.CYLINDRICAL
    simulation.is_cylindrical = True
    simulation.m = 0
    return simulation


def test_tda_cylindrical_flux_value_plain_factory_infers_coordinates(
    monkeypatch,
):
    _DeferredPointMonitor.instances = []
    _install_fake_native_backend(monkeypatch)
    simulations = []

    def make_simulation(sources=None):
        assert sources is None
        simulation = _make_cylindrical_flux_simulation()
        simulations.append(simulation)
        return simulation

    obj = tm.TDAObjective(
        update_design=lambda values: None,
        sim_factory=make_simulation,
        target=tm.FluxTarget(
            (mp.Vector3(0.5),),
            normal=mp.Vector3(0.0, 0.0, 1.0),
        ),
        t_final=0.6,
        dt=0.1,
    )

    value, gradient = obj.evaluate(np.zeros(1), need_gradient=False)

    step = np.arange(1.0, 6.0)
    expected_power = (2.0 * 3.0 - 5.0 * 7.0) * step * (step + 0.5)
    assert value == pytest.approx(np.sum(expected_power) * 0.1)
    assert gradient is None
    assert len(simulations) == 1
    assert {
        monitor.component for monitor in _DeferredPointMonitor.instances
    } == {mp.Er, mp.Hp, mp.Ep, mp.Hr}


def test_tda_cylindrical_eigenmode_value_plain_factory_infers_coordinates(
    monkeypatch,
):
    _DeferredPointMonitor.instances = []
    _install_fake_native_backend(monkeypatch)
    simulations = []

    def make_simulation(sources=None):
        assert sources is None
        simulation = _make_cylindrical_flux_simulation()
        simulations.append(simulation)
        return simulation

    target = tm.EigenmodeCoefficientTarget(
        positions=(mp.Vector3(0.5),),
        normal=mp.Vector3(0.0, 0.0, 1.0),
        reference_mode_fields={
            mp.Er: (1.0,),
            mp.Hp: (1.0,),
            mp.Ep: (0.0,),
            mp.Hr: (0.0,),
        },
        reference_m=0,
    )
    obj = tm.TDAObjective(
        update_design=lambda values: None,
        sim_factory=make_simulation,
        target=target,
        t_final=0.6,
        dt=0.1,
    )

    value, gradient = obj.evaluate(np.zeros(1), need_gradient=False)

    step = np.arange(1.0, 6.0)
    expected_coefficient = 0.5 * (2.0 * step + 3.0 * (step + 0.5))
    assert value == pytest.approx(
        np.sum(np.abs(expected_coefficient) ** 2) * 0.1
    )
    assert gradient is None
    assert len(simulations) == 1
    assert {
        monitor.component for monitor in _DeferredPointMonitor.instances
    } == {mp.Er, mp.Hp, mp.Ep, mp.Hr}


def test_tda_surface_value_plain_factory_locks_inferred_coordinates(
    monkeypatch,
):
    _DeferredPointMonitor.instances = []
    _install_fake_native_backend(monkeypatch)
    cartesian_simulation = _FluxSampleSimulation(0.1, 6)
    cylindrical_simulation = _ReusableSimulation(0.1, 6)
    cylindrical_simulation.dimensions = mp.CYLINDRICAL
    cylindrical_simulation.is_cylindrical = True
    simulations = iter((cartesian_simulation, cylindrical_simulation))
    factory_calls = []

    def make_simulation(sources=None):
        factory_calls.append(sources)
        return next(simulations)

    obj = tm.TDAObjective(
        update_design=lambda values: None,
        sim_factory=make_simulation,
        target=tm.FluxTarget(
            (mp.Vector3(0.5),),
            normal=mp.Vector3(1.0),
        ),
        t_final=0.6,
        dt=0.1,
    )

    _, gradient = obj.evaluate(np.zeros(1), need_gradient=False)
    monitor_count = len(_DeferredPointMonitor.instances)

    with pytest.raises(ValueError, match="coordinate system does not match"):
        obj.evaluate(np.zeros(1), need_gradient=False)

    assert gradient is None
    assert factory_calls == [None, None]
    assert len(_DeferredPointMonitor.instances) == monitor_count
    assert cylindrical_simulation.events == ["reset_meep"]


def test_tda_cylindrical_surface_value_plain_factory_validates_surface():
    simulation = _ReusableSimulation(0.1, 6)
    simulation.dimensions = mp.CYLINDRICAL
    simulation.is_cylindrical = True
    simulation.m = 0
    factory_calls = []

    def make_simulation(sources=None):
        factory_calls.append(sources)
        return simulation

    obj = tm.TDAObjective(
        update_design=lambda values: None,
        sim_factory=make_simulation,
        target=tm.FluxTarget(
            (mp.Vector3(),),
            normal=mp.Vector3(0.0, 0.0, 1.0),
        ),
        t_final=0.6,
        dt=0.1,
    )

    with pytest.raises(ValueError, match="zero weight at r=0"):
        obj.evaluate(np.zeros(1), need_gradient=False)

    assert factory_calls == [None]
    assert simulation.events == ["reset_meep"]


def test_tda_surface_value_rejects_runtime_coordinate_metadata_mismatch():
    simulation = _ReusableSimulation(0.1, 6)
    simulation.dimensions = mp.CYLINDRICAL
    simulation.is_cylindrical = True
    simulation.m = 0
    obj = tm.TDAObjective(
        design=_FakeDesign(is_cylindrical=False),
        sim_factory=lambda sources=None: simulation,
        target=tm.FluxTarget(
            (mp.Vector3(0.5),),
            normal=mp.Vector3(1.0),
        ),
        t_final=0.6,
        dt=0.1,
    )

    with pytest.raises(ValueError, match="coordinate system does not match"):
        obj.evaluate(np.zeros(4), need_gradient=False)

    assert simulation.events == ["reset_meep"]


def test_tda_flux_target_rejects_finite_source_fallback():
    with pytest.raises(ValueError, match="source_boundary_mode='finite'"):
        tm.TDAObjective(
            design=_FakeDesign(),
            simulation=_FakeSimulationSpec(
                lambda sources=None: _FluxSampleSimulation(0.1, 6)
            ),
            target=tm.FluxTarget(
                (mp.Vector3(),),
                normal=mp.Vector3(1.0),
            ),
            t_final=0.6,
            chunk_balancer=_FakeAdaptiveBalancer(
                source_boundary_mode="finite"
            ),
        )


@pytest.mark.parametrize(
    "adjoint_signal_fn",
    (
        lambda history, dt: history[:-1],
        lambda history, dt: history[:, np.newaxis],
    ),
)
def test_tda_rejects_manual_adjoint_signal_with_wrong_shape(
    adjoint_signal_fn,
):
    obj = _make_minimal_tda(adjoint_signal_fn=adjoint_signal_fn)

    with pytest.raises(ValueError, match="adjoint_signal_fn.*shape"):
        obj._fom_value_and_adjoint_signal(
            np.ones(4, dtype=np.complex128),
            0.1,
        )


def test_tda_objective_rejects_meep_symmetries_before_forward_run():
    class SymmetricSimulation:
        symmetries = [object()]

        def __init__(self):
            self.run_called = False
            self.reset_called = False

        def run(self, callback, until):
            self.run_called = True

        def reset_meep(self):
            self.reset_called = True

    simulation = SymmetricSimulation()
    objective = _make_minimal_tda(
        sim_factory=lambda sources=None: simulation,
    )

    with pytest.raises(ValueError, match="does not support Meep symmetries"):
        objective.evaluate(np.zeros(4), need_gradient=False)

    assert not simulation.run_called
    assert simulation.reset_called


def test_tda_objective_rejects_meep_symmetries_before_adjoint_run(monkeypatch):
    _install_fake_native_backend(monkeypatch)
    forward = _ReusableSimulation(0.1, 6)
    adjoint = _ReusableSimulation(0.1, 6)
    adjoint.symmetries = [object()]
    simulations = iter((forward, adjoint))
    objective = _make_minimal_tda(
        sim_factory=lambda sources=None: next(simulations),
    )

    with pytest.raises(ValueError, match="does not support Meep symmetries"):
        objective.evaluate(np.zeros(4), need_gradient=True)

    assert forward.events.count("run") == 1
    assert "run" not in adjoint.events
    assert "reset_meep" in adjoint.events


def test_tda_objective_constructor_is_keyword_only():
    with pytest.raises(TypeError, match="positional"):
        tm.TDAObjective(lambda _: None)


def test_tda_objective_simulation_reuse_is_opt_in(monkeypatch):
    _install_fake_native_backend(monkeypatch)
    simulation_instances = []

    def make_simulation(sources=None):
        simulation_instance = _ReusableSimulation(0.1, 6)
        simulation_instances.append(simulation_instance)
        return simulation_instance

    simulation = _FakeSimulationSpec(make_simulation)
    default_obj = _make_minimal_tda(
        sim_factory=None,
        simulation=simulation,
    )
    reused_obj = _make_minimal_tda(
        sim_factory=None,
        simulation=simulation,
        reuse_simulation=True,
    )

    assert default_obj.reuse_simulation is False
    assert default_obj._reuse_simulation_for_adjoint is False
    assert reused_obj.reuse_simulation is True
    assert reused_obj._reuse_simulation_for_adjoint is True

    default_obj.evaluate(np.zeros(4), need_gradient=True)

    assert len(simulation.make_calls) == 2
    assert simulation.make_calls[0] is None
    assert len(simulation.make_calls[1]) == 1
    assert isinstance(simulation.make_calls[1][0], mp.IndexedSource)
    assert simulation_instances[0].events == ["run", "reset_meep"]
    assert simulation_instances[1].events == ["run", "reset_meep"]

    try:
        _make_minimal_tda(reuse_simulation=True)
    except ValueError as exc:
        assert "SimulationSpec.make" in str(exc)
    else:
        raise AssertionError("expected SimulationSpec requirement")

    for invalid_reuse in (1, "false"):
        try:
            _make_minimal_tda(reuse_simulation=invalid_reuse)
        except ValueError as exc:
            assert "boolean" in str(exc)
        else:
            raise AssertionError("expected boolean requirement")


def test_tda_objective_reuses_simulation_spec_for_adjoint_run(monkeypatch):
    _install_fake_native_backend(monkeypatch)
    simulation_instance = _ReusableSimulation(0.1, 6)
    simulation = _FakeSimulationSpec(lambda sources=None: simulation_instance)
    obj = _make_minimal_tda(
        sim_factory=None,
        simulation=simulation,
        reuse_simulation=True,
    )

    _, gradient = obj.evaluate(np.zeros(4), need_gradient=True)

    assert simulation.make_calls == [None]
    assert simulation_instance.events == [
        "run",
        "restart_fields",
        "clear_dft_monitors",
        "change_sources",
        "run",
        "reset_meep",
    ]
    assert len(simulation_instance.changed_sources) == 1
    assert isinstance(
        simulation_instance.changed_sources[0][0],
        mp.IndexedSource,
    )
    assert np.allclose(gradient, 12.0)


def test_tda_objective_keeps_geometric_source_for_arbitrary_factory(monkeypatch):
    _install_fake_native_backend(monkeypatch)
    source_calls = []

    def make_simulation(sources=None):
        source_calls.append(sources)
        return _ReusableSimulation(0.1, 6)

    obj = _make_minimal_tda(sim_factory=make_simulation)
    obj.evaluate(np.zeros(4), need_gradient=True)

    assert source_calls[0] is None
    assert len(source_calls[1]) == 1
    assert isinstance(source_calls[1][0], mp.Source)
    assert not isinstance(source_calls[1][0], mp.IndexedSource)


def test_tda_cylindrical_separate_adjoint_uses_negative_forward_mode(monkeypatch):
    _install_fake_native_backend(monkeypatch)
    simulation_instances = []

    def make_simulation(sources=None):
        simulation = _ReusableSimulation(0.1, 6)
        simulation_instances.append(simulation)
        return simulation

    simulation = _FakeSimulationSpec(
        make_simulation,
        dimensions=mp.CYLINDRICAL,
        m=1,
    )
    obj = _make_minimal_tda(
        design=_FakeDesign(is_cylindrical=True),
        sim_factory=None,
        simulation=simulation,
        component=mp.Hp,
    )

    obj.evaluate(np.zeros(4), need_gradient=True)

    assert obj.gradient_components == (mp.Er, mp.Ep, mp.Ez)
    assert obj._native_adjoint_midpoint is True
    assert simulation.make_calls[0] is None
    assert simulation.make_calls[1][1] == -1
    assert simulation_instances[0].m == 1
    assert simulation_instances[1].m == -1
    assert _FakeNativeDesignAccumulator.instances[-1].midpoint is True


@pytest.mark.parametrize("forward_mode", [-1, 0, 1])
@pytest.mark.parametrize(
    ("component", "m0_components"),
    [
        (mp.Ep, (mp.Ep,)),
        (mp.Hr, (mp.Ep,)),
        (mp.Hz, (mp.Ep,)),
        (mp.Er, (mp.Er, mp.Ez)),
        (mp.Ez, (mp.Er, mp.Ez)),
        (mp.Hp, (mp.Er, mp.Ez)),
    ],
)
def test_tda_cylindrical_gradient_components_follow_mode_polarization(
    forward_mode,
    component,
    m0_components,
):
    simulation = _FakeSimulationSpec(
        lambda sources=None: _ReusableSimulation(0.1, 6),
        dimensions=mp.CYLINDRICAL,
        m=forward_mode,
    )

    obj = _make_minimal_tda(
        design=_FakeDesign(is_cylindrical=True),
        sim_factory=None,
        simulation=simulation,
        component=component,
    )

    expected = (
        m0_components
        if forward_mode == 0
        else (mp.Er, mp.Ep, mp.Ez)
    )
    assert obj.gradient_components == expected


@pytest.mark.parametrize("forward_mode", [-1, 0, 1])
def test_tda_cylindrical_reuse_flips_mode(
    monkeypatch,
    forward_mode,
):
    _install_fake_native_backend(monkeypatch)
    simulation_instance = _ReusableSimulation(0.1, 6)
    simulation = _FakeSimulationSpec(
        lambda sources=None: simulation_instance,
        dimensions=mp.CYLINDRICAL,
        m=forward_mode,
    )
    obj = _make_minimal_tda(
        design=_FakeDesign(is_cylindrical=True),
        sim_factory=None,
        simulation=simulation,
        monitor_position=mp.Vector3(0.5, 0.0, 0.0),
        component=mp.Hr,
        reuse_simulation=True,
    )

    obj.evaluate(np.zeros(4), need_gradient=True)

    assert simulation.make_calls == [None]
    assert simulation_instance.events == [
        "run",
        ("change_m", -forward_mode),
        "restart_fields",
        "clear_dft_monitors",
        "change_sources",
        "run",
        "reset_meep",
    ]
    assert simulation_instance.m == -forward_mode


def test_tda_rejects_custom_cylindrical_gradient_factory(monkeypatch):
    _install_fake_native_backend(monkeypatch)

    def make_simulation(sources=None):
        simulation = _ReusableSimulation(0.1, 6)
        simulation.dimensions = mp.CYLINDRICAL
        simulation.is_cylindrical = True
        simulation.m = 1
        return simulation

    obj = _make_minimal_tda(sim_factory=make_simulation)

    with pytest.raises(
        ValueError,
        match="requires simulation=SimulationSpec",
    ):
        obj.evaluate(np.zeros(4), need_gradient=True)


def test_tda_rejects_cylindrical_spec_with_cartesian_design():
    simulation = _FakeSimulationSpec(
        lambda sources=None: _ReusableSimulation(0.1, 6),
        dimensions=mp.CYLINDRICAL,
        m=0,
    )

    with pytest.raises(ValueError, match="coordinate system must match"):
        _make_minimal_tda(
            sim_factory=None,
            simulation=simulation,
        )


def test_tda_accepts_cylindrical_adaptive_chunk_balancing():
    simulation = _FakeSimulationSpec(
        lambda sources=None: _ReusableSimulation(0.1, 6),
        dimensions=mp.CYLINDRICAL,
        m=0,
    )
    balancer = _FakeAdaptiveBalancer()

    objective = _make_minimal_tda(
        design=_FakeDesign(is_cylindrical=True),
        sim_factory=None,
        simulation=simulation,
        chunk_balancer=balancer,
    )

    assert objective.chunk_balancer is balancer


def test_tda_auto_chunk_balancer_requires_direct_unpartitioned_simulation_spec(
    monkeypatch,
):
    created = []

    def make_balancer():
        balancer = _FakeAdaptiveBalancer()
        created.append(balancer)
        return balancer

    monkeypatch.setattr(
        tda_module,
        "AdaptiveAdjointChunkBalancer",
        make_balancer,
    )

    automatic_spec = tm.SimulationSpec(
        cell_size=mp.Vector3(1.0, 1.0),
        dimensions=2,
    )
    automatic = _make_minimal_tda(
        sim_factory=None,
        simulation=automatic_spec,
    )
    fixed = _make_minimal_tda(
        sim_factory=None,
        simulation=tm.SimulationSpec(
            cell_size=mp.Vector3(1.0, 1.0),
            dimensions=2,
            chunk_layout="fixed-layout",
        ),
    )
    opted_out = _make_minimal_tda(
        sim_factory=None,
        simulation=tm.SimulationSpec(
            cell_size=mp.Vector3(1.0, 1.0),
            dimensions=2,
        ),
        chunk_balancer=None,
    )
    fake_spec = _make_minimal_tda(
        sim_factory=None,
        simulation=_FakeSimulationSpec(
            lambda sources=None: _MultiSampleSimulation(0.1, 6)
        ),
    )

    assert automatic.chunk_balancer is created[0]
    assert fixed.chunk_balancer is None
    assert opted_out.chunk_balancer is None
    assert fake_spec.chunk_balancer is None
    assert len(created) == 1


def test_tda_rejects_unknown_chunk_balancer_mode():
    with pytest.raises(ValueError, match="chunk_balancer must be 'auto'"):
        _make_minimal_tda(chunk_balancer="automatic")


def test_tda_auto_balancer_resolves_forward_sources_once_per_evaluation(
    monkeypatch,
):
    _install_fake_native_backend(monkeypatch)
    balancer = _FakeAdaptiveBalancer()
    monkeypatch.setattr(
        tda_module,
        "AdaptiveAdjointChunkBalancer",
        lambda: balancer,
    )
    resolve_calls = []
    make_calls = []

    def resolve_sources():
        sources = (
            mp.Source(
                mp.GaussianSource(frequency=1.0, fwidth=0.5),
                component=mp.Ez,
                center=mp.Vector3(),
            ),
        )
        resolve_calls.append(sources)
        return sources

    simulation = tm.SimulationSpec(
        cell_size=mp.Vector3(1.0, 1.0),
        sources=resolve_sources,
        dimensions=2,
    )

    def make_simulation(sources=None, *, m=None):
        make_calls.append(sources)
        return _MultiSampleSimulation(0.1, 6)

    simulation.make = make_simulation
    objective = _make_minimal_tda(
        sim_factory=None,
        simulation=simulation,
    )

    objective.evaluate(np.zeros(4), need_gradient=False)
    assert balancer.captured == []
    assert balancer.rebalanced == []
    objective.evaluate(np.zeros(4), need_gradient=True)

    assert len(resolve_calls) == 2
    assert balancer.prepared_forward_sources == [
        resolve_calls[0],
        resolve_calls[1],
    ]
    assert resolve_calls[0] is not resolve_calls[1]
    assert balancer.prepared_indexed_masks == [(True,), (True,)]
    assert make_calls[0] is resolve_calls[0]
    assert make_calls[1] is resolve_calls[1]
    assert len(balancer.captured) == 1
    assert len(balancer.rebalanced) == 1


@pytest.mark.parametrize(
    ("target", "factory", "indexed_source_count"),
    (
        (
            tm.PointTarget(mp.Vector3(), mp.Ez),
            lambda sources=None: _MultiSampleSimulation(0.1, 6),
            1,
        ),
        (
            tm.FieldRegionTarget(
                (mp.Vector3(-0.2), mp.Vector3(0.2)),
                mp.Ez,
            ),
            lambda sources=None: _MultiSampleSimulation(0.1, 6),
            2,
        ),
        (
            tm.FluxTarget(
                (mp.Vector3(),),
                normal=mp.Vector3(1.0),
            ),
            lambda sources=None: _FluxSampleSimulation(0.1, 6),
            4,
        ),
    ),
)
def test_tda_value_then_gradient_keeps_balancer_source_signature(
    monkeypatch,
    target,
    factory,
    indexed_source_count,
):
    _install_fake_native_backend(monkeypatch)
    simulation = _FakeSimulationSpec(factory)
    balancer = _FakeAdaptiveBalancer()
    objective = tm.TDAObjective(
        design=_FakeDesign(),
        simulation=simulation,
        target=target,
        t_final=0.6,
        dt=0.1,
        chunk_balancer=balancer,
    )

    objective.evaluate(np.zeros(4), need_gradient=False)
    assert balancer.captured == []
    assert balancer.rebalanced == []
    objective.evaluate(np.zeros(4), need_gradient=True)

    expected_mask = (True,) * indexed_source_count
    assert balancer.prepared_indexed_masks == [expected_mask, expected_mask]
    assert balancer.prepared_sources[0] == balancer.prepared_sources[1]
    assert len(balancer.captured) == 1
    assert len(balancer.rebalanced) == 1


def test_tda_objective_value_only_reuse_resets_without_restart():
    simulation_instance = _ReusableSimulation(0.1, 6)
    simulation = _FakeSimulationSpec(lambda sources=None: simulation_instance)
    obj = _make_minimal_tda(
        design=None,
        update_design=lambda _: None,
        sim_factory=None,
        simulation=simulation,
        reuse_simulation=True,
    )

    original_monitor = tda_module.FastPointMonitor
    try:
        tda_module.FastPointMonitor = _DeferredPointMonitor
        _, gradient = obj.evaluate(np.zeros(4), need_gradient=False)
    finally:
        tda_module.FastPointMonitor = original_monitor

    assert gradient is None
    assert simulation.make_calls == [None]
    assert simulation_instance.events == ["run", "reset_meep"]


def test_tda_objective_cleans_reused_simulation_when_adjoint_run_raises(monkeypatch):
    _install_fake_native_backend(monkeypatch)

    class _FailingAdjointSimulation(_ReusableSimulation):
        def run(self, callback, until):
            self.events.append("run")
            if self.events.count("run") == 2:
                self.step = 0
                callback(self)
                raise RuntimeError("adjoint simulation failed")
            _MultiSampleSimulation.run(self, callback, until)

    simulation_instance = _FailingAdjointSimulation(0.1, 6)
    simulation = _FakeSimulationSpec(lambda sources=None: simulation_instance)
    obj = _make_minimal_tda(
        sim_factory=None,
        simulation=simulation,
        reuse_simulation=True,
    )

    try:
        obj.evaluate(np.zeros(4), need_gradient=True)
    except RuntimeError as exc:
        assert "adjoint simulation failed" in str(exc)
    else:
        raise AssertionError("expected adjoint simulation failure")

    assert simulation_instance.events == [
        "run",
        "restart_fields",
        "clear_dft_monitors",
        "change_sources",
        "run",
        "reset_meep",
    ]


def test_tda_objective_prefers_meep_fields_dt_over_resolution_fallback():
    obj = tm.TDAObjective(
        update_design=lambda _: None,
        t_final=1.0,
        sim_factory=lambda sources=None: _FakeSimulation(0.03125),
        monitor_position=mp.Vector3(),
        component=mp.Ez,
        resolution=10,
    )

    assert obj.time_step(_FakeSimulation(0.03125)) == 0.03125


def test_tda_objective_validates_explicit_dt_against_meep_fields_dt():
    obj = tm.TDAObjective(
        update_design=lambda _: None,
        t_final=1.0,
        sim_factory=lambda sources=None: _FakeSimulation(0.03125),
        monitor_position=mp.Vector3(),
        component=mp.Ez,
        dt=0.03125,
    )

    assert obj.time_step(_FakeSimulation(0.03125)) == 0.03125
    with pytest.raises(ValueError, match="does not match Meep fields.dt"):
        obj.time_step(_FakeSimulation(0.04))


def test_tda_objective_resolves_nyquist_interval_and_accumulates_full_rate(
    monkeypatch,
):
    _install_fake_native_backend(monkeypatch)
    obj = tm.TDAObjective(
        design=_FakeDesign(),
        t_final=0.8,
        sim_factory=lambda sources=None: _MultiSampleSimulation(0.1, 10),
        monitor_position=mp.Vector3(),
        component=mp.Ez,
        dt=0.1,
        max_frequency=1.5,
        reconstruction_window="hann",
        reconstruction_half_width=8,
        fom_fn=lambda monitor_history, dt: float(
            np.sum(np.abs(monitor_history) ** 2) * dt
        ),
        adjoint_signal_fn=lambda monitor_history, dt: np.ones_like(
            monitor_history,
            dtype=np.complex128,
        ),
    )

    obj.evaluate(np.zeros(4), need_gradient=True)

    assert obj.last_sampling_interval == 3
    assert obj.reconstruction_window == "hann"
    assert obj.reconstruction_half_width == 8
    assert _FakeNativeDesignHistorySet.instances[0].sample_calls == 4
    assert _FakeNativeDesignAccumulator.instances[0].sample_steps == list(range(10))


def test_tda_objective_stores_real_meep_histories_as_float64(monkeypatch):
    _install_fake_native_backend(monkeypatch)

    class _RealSampleSimulation(_MultiSampleSimulation):
        def __init__(self):
            super().__init__(0.1, 6)
            self.fields.is_real = True

        def get_field_point(self, component, position):
            return float(self.step + 1)

    obj = _make_minimal_tda(
        sim_factory=lambda sources=None: _RealSampleSimulation(),
        adjoint_signal_fn=lambda history, dt: np.ones_like(history),
    )

    obj.evaluate(np.zeros(4), need_gradient=True)

    assert _FakeNativeDesignHistorySet.instances[0].history_dtype == np.dtype(
        np.float64
    )


@pytest.mark.parametrize("regional_target", (False, True))
@pytest.mark.parametrize("cylindrical", (False, True))
def test_tda_gradient_rejects_nonunit_logical_amplitude_before_forward(
    monkeypatch,
    regional_target,
    cylindrical,
):
    _install_fake_native_backend(monkeypatch)
    simulations = []

    class _DeferredFieldRepresentationSimulation(_ReusableSimulation):
        def __init__(self, fields_are_real):
            super().__init__(0.1, 6)
            self._initialized_fields = self.fields
            self._initialized_fields.is_real = fields_are_real
            self.fields = None
            self.initialized = False

        def init_sim(self):
            self.fields = self._initialized_fields
            self.initialized = True

    def factory(sources=None):
        simulation = _DeferredFieldRepresentationSimulation(
            fields_are_real=sources is not None,
        )
        simulations.append(simulation)
        return simulation

    target = (
        tm.FieldRegionTarget(
            (mp.Vector3(0.2), mp.Vector3(0.4)),
            mp.Ez,
            adjoint_source_amplitude=1.0j,
        )
        if regional_target
        else tm.PointTarget(
            mp.Vector3(0.2),
            mp.Ez,
            adjoint_source_amplitude=1.0j,
        )
    )
    obj = tm.TDAObjective(
        design=_FakeDesign(is_cylindrical=cylindrical),
        simulation=_FakeSimulationSpec(
            factory,
            dimensions=mp.CYLINDRICAL if cylindrical else 2,
            m=0,
        ),
        target=target,
        t_final=0.6,
        dt=0.1,
        fom_fn=lambda history, dt: float(
            np.sum(np.abs(history) ** 2) * dt
        ),
        adjoint_signal_fn=lambda history, dt: np.ones_like(history),
    )

    with pytest.raises(
        ValueError,
        match="requires adjoint_source_amplitude=1",
    ):
        obj.evaluate(np.zeros(4), need_gradient=True)

    assert simulations == []


def test_tda_nonunit_adjoint_amplitude_allows_value_only_only(
    monkeypatch,
):
    _install_fake_native_backend(monkeypatch)

    class _FieldRepresentationSimulation(_ReusableSimulation):
        def __init__(self, fields_are_real):
            super().__init__(0.1, 6)
            self.fields.is_real = fields_are_real

        def get_field_point(self, component, position):
            value = float(self.step + 1)
            return value if self.fields.is_real else complex(value)

    real_value_only = _make_minimal_tda(
        sim_factory=lambda sources=None: _FieldRepresentationSimulation(True),
        adjoint_source_amplitude=1.0j,
    )
    value, gradient = real_value_only.evaluate(
        np.zeros(4),
        need_gradient=False,
    )
    assert np.isfinite(value)
    assert gradient is None

    simulations = []

    def real_forward_complex_adjoint_factory(sources=None):
        simulation = _FieldRepresentationSimulation(
            fields_are_real=sources is None,
        )
        simulations.append(simulation)
        return simulation

    complex_gradient = _make_minimal_tda(
        sim_factory=real_forward_complex_adjoint_factory,
        adjoint_source_amplitude=1.0j,
    )
    with pytest.raises(ValueError, match="requires adjoint_source_amplitude=1"):
        complex_gradient.evaluate(
            np.zeros(4),
            need_gradient=True,
        )
    assert simulations == []


def test_adjoint_amplitude_validation_is_collective(monkeypatch):
    simulation = _FakeSimulation(0.1)
    simulation.fields.is_real = False
    monkeypatch.setattr(objectives_module.mp, "count_processors", lambda: 2)
    monkeypatch.setattr(objectives_module.mp, "sum_to_all", lambda value: 1)

    with pytest.raises(
        ValueError,
        match="real Meep fields.*adjoint_source_amplitude.*real",
    ):
        objectives_module._validate_adjoint_source_amplitudes(
            simulation,
            (1.0j,),
        )


def test_logical_adjoint_amplitude_validation_is_collective(monkeypatch):
    monkeypatch.setattr(objectives_module.mp, "count_processors", lambda: 2)
    monkeypatch.setattr(objectives_module.mp, "sum_to_all", lambda value: 1)

    with pytest.raises(ValueError, match="requires adjoint_source_amplitude=1"):
        objectives_module._validate_logical_adjoint_source_amplitudes((1.0,))


@pytest.mark.parametrize("allocated", (False, True))
def test_indexed_monitor_support_allows_only_unallocated_zero_channels(allocated):
    fields = type(
        "Fields",
        (),
        {"have_component": lambda self, component: allocated},
    )()
    simulation = type("Simulation", (), {"fields": fields})()

    if allocated:
        with pytest.raises(ValueError, match="no interpolation support"):
            objectives_module._validate_indexed_monitor_support(
                simulation,
                (mp.Ez,),
                np.asarray([0, 0], dtype=np.intp),
            )
    else:
        objectives_module._validate_indexed_monitor_support(
            simulation,
            (mp.Ez,),
            np.asarray([0, 0], dtype=np.intp),
        )


def test_tda_rejects_simulation_mode_mutation_before_gradient_run():
    simulation = _FakeSimulationSpec(
        lambda sources=None: pytest.fail("simulation must not be constructed"),
        dimensions=mp.CYLINDRICAL,
        m=1,
    )
    obj = _make_minimal_tda(
        sim_factory=None,
        simulation=simulation,
        design=_FakeDesign(is_cylindrical=True),
        monitor_position=mp.Vector3(0.5),
        component=mp.Er,
    )
    simulation.m = 0

    with pytest.raises(ValueError, match="must not change"):
        obj.evaluate(np.zeros(4), need_gradient=True)


@pytest.mark.parametrize(
    "position",
    (mp.Vector3(-0.1, 0.0, 0.0), mp.Vector3(0.1, 0.2, 0.0)),
)
def test_tda_rejects_invalid_cylindrical_point_coordinates(position):
    simulation = _FakeSimulationSpec(
        lambda sources=None: _ReusableSimulation(0.1, 6),
        dimensions=mp.CYLINDRICAL,
        m=0,
    )
    obj = _make_minimal_tda(
        sim_factory=None,
        simulation=simulation,
        design=_FakeDesign(is_cylindrical=True),
        monitor_position=position,
        component=mp.Ez,
    )

    with pytest.raises(ValueError, match="cylindrical target positions"):
        obj.evaluate(np.zeros(4), need_gradient=False)


def test_tda_objective_keeps_terminal_sample_for_reconstruction(monkeypatch):
    _install_fake_native_backend(monkeypatch)
    obj = tm.TDAObjective(
        design=_FakeDesign(),
        t_final=0.6,
        sim_factory=lambda sources=None: _TerminalSampleSimulation(0.1, 7),
        monitor_position=mp.Vector3(),
        component=mp.Ez,
        dt=0.1,
        max_frequency=1.5,
        fom_fn=lambda monitor_history, dt: float(
            np.sum(np.abs(monitor_history) ** 2) * dt
        ),
        adjoint_signal_fn=lambda monitor_history, dt: np.ones_like(
            monitor_history,
            dtype=np.complex128,
        ),
    )

    obj.evaluate(np.zeros(4), need_gradient=True)

    assert obj.last_sampling_interval == 3
    assert _FakeNativeDesignHistorySet.instances[0].sample_calls == 3
    assert _FakeNativeDesignAccumulator.instances[0].sample_steps == list(range(7))


def test_tda_objective_validates_sampling_configuration():
    params = dict(
        update_design=lambda _: None,
        t_final=1.0,
        sim_factory=lambda sources=None: _FakeSimulation(0.1),
        monitor_position=mp.Vector3(),
        component=mp.Ez,
        dt=0.1,
    )
    for invalid_interval in (0, -1, 1.5, True):
        try:
            tm.TDAObjective(**params, sampling_interval=invalid_interval)
        except ValueError as exc:
            assert "sampling_interval" in str(exc)
        else:
            raise AssertionError("expected invalid sampling_interval error")

    try:
        tm.TDAObjective(
            **params,
            sampling_interval=2,
            max_frequency=1.0,
        )
    except ValueError as exc:
        assert "cannot be combined" in str(exc)
    else:
        raise AssertionError("expected conflicting sampling configuration error")

    configured = tm.TDAObjective(
        **params,
        reconstruction_window="hanning",
        reconstruction_half_width=12,
    )
    assert configured.reconstruction_window == "hann"
    assert configured.reconstruction_window_params == {}
    assert configured.reconstruction_half_width == 12

    try:
        tm.TDAObjective(**params, reconstruction_window="unknown")
    except ValueError as exc:
        assert "reconstruction_window" in str(exc)
    else:
        raise AssertionError("expected invalid reconstruction window error")


def test_tda_objective_reduces_point_monitor_history_once_after_forward():
    _DeferredPointMonitor.instances = []
    obj = tm.TDAObjective(
        update_design=lambda _: None,
        t_final=0.6,
        sim_factory=lambda sources=None: _MultiSampleSimulation(0.1, 6),
        monitor_position=mp.Vector3(),
        component=mp.Ez,
        dt=0.1,
    )

    original_monitor = tda_module.FastPointMonitor
    try:
        tda_module.FastPointMonitor = _DeferredPointMonitor
        value, gradient = obj.evaluate(np.array([0.0]), need_gradient=False)
    finally:
        tda_module.FastPointMonitor = original_monitor

    assert np.isfinite(value)
    assert gradient is None
    assert len(_DeferredPointMonitor.instances) == 1
    assert _DeferredPointMonitor.instances[0].sample_calls == 6
    assert _DeferredPointMonitor.instances[0].reduce_calls == 1
    assert _DeferredPointMonitor.instances[0].reduced_shapes == [(6, 1)]


def test_tda_objective_grows_preallocated_point_monitor_history(
    monkeypatch,
):
    _DeferredPointMonitor.instances = []
    monkeypatch.setattr(
        tda_module,
        "FastPointMonitor",
        _DeferredPointMonitor,
    )
    captured_histories = []
    obj = tm.TDAObjective(
        update_design=lambda _: None,
        t_final=0.6,
        sim_factory=lambda sources=None: _MultiSampleSimulation(0.1, 20),
        monitor_position=mp.Vector3(),
        component=mp.Ez,
        dt=0.1,
        fom_fn=lambda history, dt: (
            captured_histories.append(np.array(history, copy=True)) or 0.0
        ),
    )

    value, gradient = obj.evaluate(np.array([0.0]), need_gradient=False)

    assert value == 0.0
    assert gradient is None
    assert len(_DeferredPointMonitor.instances) == 1
    monitor = _DeferredPointMonitor.instances[0]
    assert monitor.sample_calls == 20
    assert monitor.reduce_calls == 1
    assert monitor.reduced_shapes == [(20, 1)]
    assert np.array_equal(captured_histories[0], np.arange(1.0, 8.0))


def test_tda_objective_requires_design_only_for_gradient(monkeypatch):
    monkeypatch.setattr(tda_module, "FastPointMonitor", _DeferredPointMonitor)
    updates = []
    obj = tm.TDAObjective(
        update_design=lambda values: updates.append(np.array(values, copy=True)),
        t_final=0.6,
        sim_factory=lambda sources=None: _MultiSampleSimulation(0.1, 6),
        monitor_position=mp.Vector3(),
        component=mp.Ez,
        dt=0.1,
    )
    design = np.array([0.25])

    value, gradient = obj.evaluate(design, need_gradient=False)

    assert np.isfinite(value)
    assert gradient is None
    assert len(updates) == 1
    with pytest.raises(ValueError, match="requires design=DesignGrid"):
        obj.evaluate(design, need_gradient=True)
    assert len(updates) == 1


def test_point_target_adjoint_source_uses_reversed_forward_sample_times():
    target = objectives_module._PointTarget(
        monitor_position=mp.Vector3(),
        component=mp.Ez,
    )
    seen = {}

    class _FakeInterp:
        def __call__(self, _):
            return 0.0

    def fake_tabulated_source(times, values):
        seen["times"] = np.array(times, copy=True)
        seen["values"] = np.array(values, copy=True)
        return _FakeInterp()

    original_tabulated_source = objectives_module._tabulated_cubic_source
    try:
        objectives_module._tabulated_cubic_source = fake_tabulated_source
        target.adjoint_sources(
            np.array([1.0, 2.0, 3.0]),
            np.array([0.1, 0.35, 0.6]),
            actual_time=0.75,
        )
    finally:
        objectives_module._tabulated_cubic_source = original_tabulated_source

    assert np.allclose(seen["times"], [0.15, 0.4, 0.65])
    assert np.allclose(seen["values"], [3.0, 2.0, 1.0])


def test_point_target_magnetic_adjoint_source_has_opposite_parity():
    captured = []

    class _FakeInterp:
        def __call__(self, _):
            return 0.0

    def fake_tabulated_source(_times, values):
        captured.append(np.array(values, copy=True))
        return _FakeInterp()

    original_tabulated_source = objectives_module._tabulated_cubic_source
    try:
        objectives_module._tabulated_cubic_source = fake_tabulated_source
        for component in (mp.Ez, mp.Er, mp.Ep, mp.Hz, mp.Hr, mp.Hp):
            target = objectives_module._PointTarget(
                monitor_position=mp.Vector3(),
                component=component,
            )
            target.adjoint_sources(
                np.array([1.0, 2.0, 3.0]),
                np.array([0.1, 0.35, 0.6]),
                actual_time=0.75,
            )
    finally:
        objectives_module._tabulated_cubic_source = original_tabulated_source

    for values in captured[:3]:
        assert np.allclose(values, [3.0, 2.0, 1.0])
    for values in captured[3:]:
        assert np.allclose(values, [-3.0, -2.0, -1.0])


def test_point_target_uses_cylindrical_indexed_transpose_stencil():
    target = objectives_module._PointTarget(
        monitor_position=mp.Vector3(0.63, 0.0, 0.17),
        component=mp.Er,
    )
    source_data = (object(), object())
    weights = np.array([0.25, -0.75])

    sources = target.adjoint_sources(
        np.array([1.0, 2.0, 3.0]),
        np.array([0.1, 0.35, 0.6]),
        actual_time=0.75,
        source_amplitude=2.0,
        indexed_stencil=(source_data, weights),
    )

    assert len(sources) == 2
    assert [source.srcdata for source in sources] == list(source_data)
    assert np.allclose(
        [source.amp_arr[0] for source in sources],
        [0.5, -1.5],
    )


def test_point_target_batches_indexed_sources_by_component_and_chunk(
    monkeypatch,
):
    class SourceData:
        def __init__(self, label, component, chunk_index):
            self.label = label
            self.near_fd_comp = component
            self.fc_idx = chunk_index

    source_data = (
        SourceData("a", mp.Ez, 0),
        SourceData("b", mp.Ez, 0),
        SourceData("c", mp.Ez, 1),
        SourceData("d", mp.Ey, 0),
        SourceData("e", mp.Ez, 0),
    )
    weights = np.asarray([0.25, -0.75, 0.5, 1.25, -0.125])
    merged_calls = []

    def merge_sourcedata(group):
        merged = SourceData(
            tuple(data.label for data in group),
            group[0].near_fd_comp,
            group[0].fc_idx,
        )
        merged_calls.append(merged)
        return merged

    monkeypatch.setattr(
        objectives_module,
        "merge_indexed_sourcedata",
        merge_sourcedata,
    )

    first = objectives_module._adjoint_point_sources(
        object(),
        mp.Ez,
        mp.Vector3(),
        mp.Vector3(),
        2.0,
        (source_data, weights),
    )
    second = objectives_module._adjoint_point_sources(
        object(),
        mp.Ez,
        mp.Vector3(),
        mp.Vector3(),
        2.0,
        (source_data, weights),
    )

    assert len(first) == len(second) == 3
    assert first[0].srcdata.label == ("a", "b", "e")
    assert first[1].srcdata is source_data[2]
    assert first[2].srcdata is source_data[3]
    assert np.array_equal(first[0].amp_arr, 2.0 * weights[[0, 1, 4]])
    assert np.array_equal(first[1].amp_arr, 2.0 * weights[[2]])
    assert np.array_equal(first[2].amp_arr, 2.0 * weights[[3]])
    assert second[0].srcdata is not first[0].srcdata
    assert len(merged_calls) == 2


@pytest.mark.parametrize("m", [-1, 0, 1])
@pytest.mark.parametrize(
    ("component", "m0_components"),
    [
        (mp.Ep, (mp.Ep,)),
        (mp.Hr, (mp.Ep,)),
        (mp.Hz, (mp.Ep,)),
        (mp.Er, (mp.Er, mp.Ez)),
        (mp.Ez, (mp.Er, mp.Ez)),
        (mp.Hp, (mp.Er, mp.Ez)),
    ],
)
def test_cylindrical_electric_components_follow_mode_polarization(
    m,
    component,
    m0_components,
):
    expected = m0_components if m == 0 else (mp.Er, mp.Ep, mp.Ez)
    assert objectives_module._electric_components(
        component,
        2,
        cylindrical=True,
        m=m,
    ) == expected


def test_tda_real_cubic_sources_share_one_batched_spline():
    times = np.linspace(0.07, 0.37, 9)
    values = np.column_stack(
        [
            np.sin(0.8 * times),
            np.cos(1.3 * times),
            times**2,
        ]
    )
    sources = objectives_module._tabulated_cubic_sources(times, values)
    reference = objectives_module.spi.CubicSpline(
        times,
        values,
        axis=0,
        extrapolate=False,
    )
    query = np.linspace(times[0], times[-1], 257)
    actual = np.column_stack(
        [[source(time) for time in query] for source in sources]
    )

    assert len(sources) == values.shape[1]
    assert all(isinstance(source(0.2), float) for source in sources)
    assert np.allclose(actual, reference(query), rtol=2.0e-14, atol=2.0e-14)
    assert all(source(times[0] - 0.01) == 0.0 for source in sources)
    assert all(source(times[-1] + 0.01) == 0.0 for source in sources)


@pytest.mark.parametrize("complex_values", [False, True])
def test_tda_cubic_sources_use_compact_shared_coefficient_bank(
    monkeypatch,
    complex_values,
):
    sample_count = 101
    channel_count = 7
    times = np.linspace(0.07, 0.37, sample_count)
    values = np.column_stack(
        [np.sin((channel + 1) * times) for channel in range(channel_count)]
    )
    factory_name = "make_native_tabulated_real_bspline_bank"
    expected_itemsize = np.dtype(np.float64).itemsize
    if complex_values:
        values = values.astype(np.complex128)
        values[:, 3] += 0.25j * np.cos(times)
        factory_name = "make_native_tabulated_bspline_bank"
        expected_itemsize = np.dtype(np.complex128).itemsize
    captured = {}

    def capture_bank(knots, coefficients):
        captured["knots"] = np.asarray(knots)
        captured["coefficients"] = np.asarray(coefficients)
        return tuple(object() for _ in range(channel_count))

    monkeypatch.setattr(objectives_module, factory_name, capture_bank)
    sources = objectives_module._tabulated_cubic_sources(times, values)

    assert len(sources) == channel_count
    assert captured["knots"].shape == (sample_count + 4,)
    assert captured["coefficients"].shape == (
        sample_count,
        channel_count,
    )
    assert captured["coefficients"].nbytes == (
        sample_count * channel_count * expected_itemsize
    )
    legacy_piecewise_bytes = (
        4 * (sample_count - 1) * channel_count * expected_itemsize
    )
    assert captured["coefficients"].nbytes < 0.26 * legacy_piecewise_bytes


def test_tda_cubic_sources_keep_complex_path_for_any_nonzero_phase():
    times = np.linspace(0.07, 0.37, 9)
    values = np.column_stack(
        [
            np.sin(times).astype(np.complex128),
            np.cos(times).astype(np.complex128),
        ]
    )
    values[4, 1] += 1.0e-300j
    sources = objectives_module._tabulated_cubic_sources(times, values)

    assert any(source(times[4]).imag != 0.0 for source in sources)


def test_tda_objective_rejects_gradient_with_fewer_than_two_field_samples(
    monkeypatch,
):
    _install_fake_native_backend(monkeypatch)
    obj = tm.TDAObjective(
        design=_FakeDesign(shape=(1, 1)),
        t_final=0.1,
        sim_factory=lambda sources=None: _OneSampleSimulation(),
        monitor_position=mp.Vector3(),
        component=mp.Ez,
        dt=0.1,
    )

    with pytest.raises(RuntimeError, match="at least two field samples"):
        obj.evaluate(np.array([0.0]), need_gradient=True)


def test_tda_objective_updates_simulation_spec_with_adaptive_layout(monkeypatch):
    _install_fake_native_backend(monkeypatch)
    simulation = _FakeSimulationSpec(
        lambda sources=None: _MultiSampleSimulation(0.1, 6)
    )
    balancer = _FakeAdaptiveBalancer()
    obj = tm.TDAObjective(
        design=_FakeDesign(),
        t_final=0.6,
        simulation=simulation,
        monitor_position=mp.Vector3(),
        component=mp.Ez,
        dt=0.1,
        fom_fn=lambda monitor_history, dt: float(np.sum(np.abs(monitor_history) ** 2) * dt),
        adjoint_signal_fn=lambda monitor_history, dt: np.ones_like(monitor_history, dtype=np.complex128),
        chunk_balancer=balancer,
    )

    obj.evaluate(np.zeros(4), need_gradient=True)

    assert simulation.chunk_layout == "next-layout"
    assert balancer.prepared == [simulation]
    assert len(balancer.captured) == 1
    assert len(balancer.rebalanced) == 1
    assert balancer.rebalanced[0][1] > 0.0
    assert balancer.rebalanced[0][2] > 0.0
    assert balancer.rebalanced[0][3] == (balancer.captured[0][1],)
    assert obj.last_chunk_balance == balancer.last_report


def test_tda_objective_uses_effective_adjoint_source_without_mutating_base(
    monkeypatch,
):
    _install_fake_native_backend(monkeypatch)
    adjoint_sources_seen = []

    def make_simulation(sources=None):
        if sources is not None:
            adjoint_sources_seen.append(sources)
        return _MultiSampleSimulation(0.1, 6)

    simulation = _FakeSimulationSpec(make_simulation)
    base_size = mp.Vector3(0.1, 0.0)
    effective_size = mp.Vector3(0.1, 0.2)
    decision = type(
        "SourceBoundaryDecision",
        (),
        {
            "source_sizes": (effective_size,),
            "source_amplitudes": (10.0,),
        },
    )()
    balancer = _FakeAdaptiveBalancer(decision)
    obj = tm.TDAObjective(
        design=_FakeDesign(),
        t_final=0.6,
        simulation=simulation,
        monitor_position=mp.Vector3(0.2, -0.1),
        component=mp.Ez,
        adjoint_source_size=base_size,
        adjoint_source_amplitude=1.0,
        dt=0.1,
        fom_fn=lambda monitor_history, dt: float(
            np.sum(np.abs(monitor_history) ** 2) * dt
        ),
        adjoint_signal_fn=lambda monitor_history, dt: np.ones_like(
            monitor_history,
            dtype=np.complex128,
        ),
        chunk_balancer=balancer,
    )

    obj.evaluate(np.zeros(4), need_gradient=True)
    obj.evaluate(np.zeros(4), need_gradient=True)

    assert obj.last_source_boundary_decision is decision
    assert len(adjoint_sources_seen) == 2
    for sources in adjoint_sources_seen:
        assert len(sources) == 1
        assert sources[0].size == effective_size
        assert sources[0].amplitude == 10.0
    assert obj.objective.adjoint_source_size == base_size
    assert obj.objective.adjoint_source_amplitude == 1.0
    assert len(balancer.prepared_sources) == 2
    for positions, source_size, source_amplitude in balancer.prepared_sources:
        assert positions == (obj.objective.monitor_position,)
        assert source_size == base_size
        assert source_amplitude == 1.0


def test_tda_objective_reuses_simulation_with_adaptive_layout(monkeypatch):
    _install_fake_native_backend(monkeypatch)
    simulation_instance = _ReusableSimulation(0.1, 6)
    simulation = _FakeSimulationSpec(lambda sources=None: simulation_instance)
    balancer = _FakeAdaptiveBalancer()
    obj = _make_minimal_tda(
        sim_factory=None,
        simulation=simulation,
        chunk_balancer=balancer,
        reuse_simulation=True,
    )

    obj.evaluate(np.zeros(4), need_gradient=True)

    assert simulation.chunk_layout == "next-layout"
    assert len(balancer.captured) == 1
    assert len(balancer.rebalanced) == 1
    assert balancer.captured[0][0] is balancer.rebalanced[0][0]
    assert balancer.rebalanced[0][3] == (balancer.captured[0][1],)
    assert simulation_instance.fields.reset_timer_calls == 1
    assert obj.last_chunk_balance == balancer.last_report
