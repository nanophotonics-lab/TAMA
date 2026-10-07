from contextlib import contextmanager

import numpy as np
import pytest

import tama as tm
import tama.sampling_grid as sampling_grid


class _FakeFields:
    this = 123
    is_real = False


class _FakeSimulation:
    fields = _FakeFields()
    dimensions = 2
    is_cylindrical = False
    symmetries = []

    def __init__(self):
        self.python_samples = 0

    def get_field_point(self, component, position):
        self.python_samples += 1
        return position.x + 1j * position.y


class _FakeNativeSampler:
    def __init__(self):
        self.local_indices = None
        self.support_reduced_indices = None
        self.accumulated_indices = None
        self.accumulated_values = None
        self.history_local_indices = None
        self.history_boundary_indices = None
        self.configured_indices = None
        self.monitor_x = None
        self.monitor_y = None
        self.monitor_local_calls = 0
        self.monitor_real_local_calls = 0
        self.monitor_complex_local_calls = 0
        self.monitor_allreduced_calls = 0
        self.monitor_history_indices = None
        self.reduction_calls = 0
        self.real_reduction_calls = 0
        self.complex_reduction_calls = 0
        self.difference_previous = None
        self.difference_current = 0.0

    @staticmethod
    def create_component_grid_plan(fields_addr, coords_x, coords_y, component):
        return object()

    def sample_component_grid_plan_allreduced(self, plan):
        return np.zeros((4, 1), dtype=np.complex128)

    def create_component_point_plan(
        self,
        fields_addr,
        coords_x,
        coords_y,
        component,
    ):
        self.monitor_x = np.asarray(coords_x)
        self.monitor_y = np.asarray(coords_y)
        return object()

    def sample_component_point_plan_allreduced(self, plan):
        self.monitor_allreduced_calls += 1
        return self.monitor_x + 1j * self.monitor_y

    def sample_component_point_plan_local_into(self, plan, destination):
        self.monitor_local_calls += 1
        self.monitor_complex_local_calls += 1
        indices = (
            np.arange(self.monitor_x.size)
            if self.monitor_history_indices is None
            else self.monitor_history_indices
        )
        destination[:] = self.monitor_x[indices] + 1j * self.monitor_y[indices]

    def sample_component_point_plan_local_real_into(self, plan, destination):
        self.monitor_local_calls += 1
        self.monitor_real_local_calls += 1
        indices = (
            np.arange(self.monitor_x.size)
            if self.monitor_history_indices is None
            else self.monitor_history_indices
        )
        destination[:] = self.monitor_x[indices]

    def configure_component_point_plan_history(self, plan, indices):
        self.monitor_history_indices = np.asarray(indices, dtype=np.int64)

    @staticmethod
    def create_eigenmode_overlap_plan(plans, weights, output_channels):
        del plans, weights, output_channels
        return object()

    @staticmethod
    def sample_eigenmode_overlap_plan_local_into(plan, destination):
        del plan
        destination[:] = 0.0

    def reduce_complex_grid_sum(self, local_grid):
        self.reduction_calls += 1
        self.complex_reduction_calls += 1
        return np.asarray(local_grid)

    def reduce_complex_grid_sum_inplace(self, local_grid):
        self.reduction_calls += 1
        self.complex_reduction_calls += 1
        return local_grid

    def reduce_real_grid_sum_inplace(self, local_grid):
        self.reduction_calls += 1
        self.real_reduction_calls += 1
        return local_grid

    @staticmethod
    def component_grid_plan_local_complete_mask(plan):
        return np.array([[True], [False], [False], [False]])

    @staticmethod
    def component_grid_plan_local_boundary_mask(plan):
        return np.array([[False], [False], [True], [True]])

    def configure_component_grid_plan_history(
        self,
        plan,
        local_indices,
        boundary_indices,
    ):
        self.history_local_indices = np.asarray(local_indices)
        self.history_boundary_indices = np.asarray(boundary_indices)

    def sample_component_grid_plan_history_into(self, plan, destination):
        destination[:] = np.concatenate(
            [
                10.0 + self.history_local_indices,
                20.0 + self.history_boundary_indices,
            ]
        )

    def sample_component_grid_plan_points_local(self, plan, indices):
        self.local_indices = np.asarray(indices)
        return np.asarray(10.0 + self.local_indices, dtype=np.complex128)

    def sample_component_grid_plan_points_support_reduced(self, plan, indices):
        self.support_reduced_indices = np.asarray(indices)
        return np.asarray(
            20.0 + self.support_reduced_indices,
            dtype=np.complex128,
        )

    def accumulate_component_product_plan_local_inplace(
        self,
        plan,
        multiplier,
        accumulator,
    ):
        accumulator[:] += 2.0 * np.asarray(multiplier)

    def accumulate_component_product_plan_points_local_inplace(
        self,
        plan,
        indices,
        values,
        accumulator,
    ):
        self.accumulated_indices = np.asarray(indices)
        self.accumulated_values = np.asarray(values)
        accumulator.reshape(-1)[self.accumulated_indices] += (
            2.0 * self.accumulated_values
        )

    def configure_component_grid_plan_accumulation(self, plan, indices):
        self.configured_indices = np.asarray(indices)
        self.difference_previous = None

    def accumulate_component_product_plan_configured_local_inplace(
        self,
        plan,
        values,
        accumulator,
    ):
        self.accumulated_indices = np.array(self.configured_indices, copy=True)
        self.accumulated_values = np.asarray(values)
        accumulator.reshape(-1)[self.accumulated_indices] += (
            2.0 * self.accumulated_values
        )

    def accumulate_component_difference_product_plan_configured_local_inplace(
        self,
        plan,
        values,
        dt,
        accumulator,
    ):
        self.difference_current += 2.0
        current = np.full(len(self.configured_indices), self.difference_current)
        if self.difference_previous is None:
            self.difference_previous = current
            return False
        accumulator.reshape(-1)[self.configured_indices] += (
            np.asarray(values) * (current - self.difference_previous) / dt
        )
        self.difference_previous = current
        return True


class _FailingPointSampler(_FakeNativeSampler):
    def sample_component_point_plan_allreduced(self, plan):
        raise RuntimeError("injected point-monitor failure")


class _FailingHistorySampler(_FakeNativeSampler):
    def sample_component_grid_plan_history_into(self, plan, destination):
        raise RuntimeError("injected history failure")


class _FailingReductionSampler(_FakeNativeSampler):
    def reduce_complex_grid_sum(self, local_grid):
        raise RuntimeError("injected reduction failure")


@contextmanager
def _native_sampler(fake_native):
    original_native = sampling_grid.native_sampler
    try:
        sampling_grid.native_sampler = fake_native
        yield
    finally:
        sampling_grid.native_sampler = original_native


def _exact_design_validation_case(*, center=None, material_factor=None, k_point=None):
    air = sampling_grid.mp.Medium(epsilon=1.0)
    dielectric = sampling_grid.mp.Medium(epsilon=4.0)
    center = center or sampling_grid.mp.Vector3()
    material_grid = sampling_grid.mp.MaterialGrid(
        sampling_grid.mp.Vector3(2, 2),
        air,
        dielectric,
    )
    design = tm.DesignGrid(
        material_grid=material_grid,
        center=center,
        size=sampling_grid.mp.Vector3(0.5, 0.5),
        shape=(2, 2),
        background=air,
        design_material=dielectric,
        material_factor=material_factor,
    )
    sim = sampling_grid.mp.Simulation(
        cell_size=sampling_grid.mp.Vector3(6, 6),
        boundary_layers=[sampling_grid.mp.PML(1.0)],
        geometry=[
            sampling_grid.mp.Block(
                center=center,
                size=design.size,
                material=material_grid,
            )
        ],
        resolution=10,
        eps_averaging=False,
        k_point=k_point or False,
    )
    return sim, design


def test_native_sampler_is_required_and_current():
    assert sampling_grid.native_sampler_available()
    assert sampling_grid.native_sampler.API_VERSION == 13
    assert all(
        callable(getattr(sampling_grid.native_sampler, name, None))
        for name in sampling_grid._REQUIRED_NATIVE_OPERATIONS
    )


@pytest.fixture(scope="module")
def material_component_simulation():
    simulation = sampling_grid.mp.Simulation(
        cell_size=sampling_grid.mp.Vector3(1.0, 1.0),
        default_material=sampling_grid.mp.Medium(epsilon=4.0, mu=2.0),
        resolution=10,
    )
    simulation.init_sim()
    try:
        yield simulation
    finally:
        simulation.reset_meep()


@pytest.mark.parametrize(
    ("component", "expected"),
    (
        pytest.param(sampling_grid.mp.Dielectric, 4.0, id="dielectric"),
        pytest.param(sampling_grid.mp.Permeability, 2.0, id="permeability"),
    ),
)
@pytest.mark.parametrize("monitor_kind", ("point", "grid"))
def test_native_monitors_reject_material_components(
    material_component_simulation,
    component,
    expected,
    monitor_kind,
):
    position = sampling_grid.mp.Vector3()
    assert material_component_simulation.get_field_point(
        component,
        position,
    ) == pytest.approx(expected)

    with pytest.raises(ValueError, match="E/H/D/B Meep field component"):
        if monitor_kind == "point":
            monitor = sampling_grid.FastPointMonitor(
                material_component_simulation,
                component,
                [position],
            )
        else:
            monitor = sampling_grid.FastFieldGrid(
                material_component_simulation,
                component,
                [position.x],
                [position.y],
            )
        monitor.sample()


@pytest.mark.parametrize("component", (sampling_grid.mp.Dz, sampling_grid.mp.Bz))
@pytest.mark.parametrize("monitor_kind", ("point", "grid"))
def test_native_monitors_accept_dynamic_d_and_b_components(
    material_component_simulation,
    component,
    monitor_kind,
):
    position = sampling_grid.mp.Vector3()
    if monitor_kind == "point":
        monitor = sampling_grid.FastPointMonitor(
            material_component_simulation,
            component,
            [position],
        )
    else:
        monitor = sampling_grid.FastFieldGrid(
            material_component_simulation,
            component,
            [position.x],
            [position.y],
        )

    monitor.sample()


@pytest.mark.parametrize(
    ("dimensions", "is_cylindrical"),
    ((3, False), (2, True)),
)
def test_fast_field_grid_rejects_non_cartesian_2d_simulations(
    dimensions,
    is_cylindrical,
):
    simulation = _FakeSimulation()
    simulation.dimensions = dimensions
    simulation.is_cylindrical = is_cylindrical

    with _native_sampler(_FakeNativeSampler()):
        with pytest.raises(
            ValueError,
            match="requires a 2D Cartesian Meep simulation",
        ):
            sampling_grid.FastFieldGrid(
                simulation,
                component=0,
                coords_x=[0.0],
                coords_y=[0.0],
            )


def test_history_storage_dtype_tracks_initialized_meep_fields():
    real_sim = type(
        "RealSimulation",
        (),
        {"fields": type("RealFields", (), {"is_real": True})()},
    )()
    complex_sim = type(
        "ComplexSimulation",
        (),
        {"fields": type("ComplexFields", (), {"is_real": False})()},
    )()

    assert sampling_grid.history_storage_dtype(
        real_sim,
        np.complex128,
    ) == np.dtype(np.float64)
    assert sampling_grid.history_storage_dtype(
        real_sim,
        np.complex64,
    ) == np.dtype(np.float32)
    assert sampling_grid.history_storage_dtype(
        complex_sim,
        np.complex128,
    ) == np.dtype(np.complex128)
    with pytest.raises(ValueError, match="complex Meep fields"):
        sampling_grid.history_storage_dtype(complex_sim, np.float64)


def test_native_sampler_capability_mismatch_is_rank_synchronized(monkeypatch):
    reduced = []

    def sum_to_all(value):
        reduced.append(value)
        return 1

    monkeypatch.setattr(sampling_grid, "_native_sampler_compatible", False)
    monkeypatch.setattr(
        sampling_grid,
        "_native_sampler_state_synchronized",
        False,
    )
    monkeypatch.setattr(
        sampling_grid,
        "_missing_native_operations",
        ["missing_operation"],
    )
    monkeypatch.setattr(sampling_grid.mp, "count_processors", lambda: 2)
    monkeypatch.setattr(sampling_grid.mp, "sum_to_all", sum_to_all)

    with pytest.raises(RuntimeError, match="one or more active Meep ranks"):
        sampling_grid._require_native_sampler()

    assert reduced == [1]


def test_native_design_rejects_nonzero_bloch_wavevector():
    sim, design = _exact_design_validation_case(
        k_point=sampling_grid.mp.Vector3(0.1),
    )
    with pytest.raises(ValueError, match="k_point=0"):
        sampling_grid.NativeDesignField(sim, sampling_grid.mp.Ez, design)


def test_native_design_rejects_design_overlapping_pml():
    sim, design = _exact_design_validation_case(
        center=sampling_grid.mp.Vector3(2.1, 0.0),
    )
    with pytest.raises(ValueError, match="overlapping PML or absorber"):
        sampling_grid.NativeDesignField(sim, sampling_grid.mp.Ez, design)


def test_native_design_rejects_inconsistent_material_factor():
    sim, design = _exact_design_validation_case(material_factor=2.0)
    with pytest.raises(ValueError, match="MaterialGrid media contrast"):
        sampling_grid.NativeDesignField(sim, sampling_grid.mp.Ez, design)


def test_fast_point_monitor_uses_one_native_paired_point_plan():
    fake_native = _FakeNativeSampler()
    simulation = _FakeSimulation()
    with _native_sampler(fake_native):
        monitor = sampling_grid.FastPointMonitor(
            simulation,
            component=0,
            positions=[
                sampling_grid.mp.Vector3(1.0, 2.0),
                sampling_grid.mp.Vector3(3.0, 4.0),
            ],
        )
        values = monitor.sample()

    assert np.array_equal(fake_native.monitor_x, [1.0, 3.0])
    assert np.array_equal(fake_native.monitor_y, [2.0, 4.0])
    assert np.array_equal(values, [1.0 + 2.0j, 3.0 + 4.0j])
    assert fake_native.monitor_allreduced_calls == 1
    assert simulation.python_samples == 0


def test_cached_native_monitor_plans_reject_replaced_fields_proxy():
    fake_native = _FakeNativeSampler()
    simulations = [_FakeSimulation() for _ in range(3)]
    with _native_sampler(fake_native):
        point = sampling_grid.FastPointMonitor(
            simulations[0],
            component=0,
            positions=[sampling_grid.mp.Vector3()],
        )
        field = sampling_grid.FastFieldGrid(
            simulations[1],
            component=0,
            coords_x=[0.0],
            coords_y=[0.0],
        )
        overlap = sampling_grid.FastEigenmodeOverlapMonitor(
            simulations[2],
            positions=[sampling_grid.mp.Vector3()],
            electric_weights={0: np.ones(1)},
            magnetic_weights={1: np.ones(1)},
        )
        point.ensure_plan()
        field.ensure_plan()
        overlap.ensure_plan()

        for simulation in simulations:
            simulation.fields = _FakeFields()

        with pytest.raises(RuntimeError, match="plan lifetime validation failed"):
            point.sample()
        with pytest.raises(RuntimeError, match="plan lifetime validation failed"):
            field.sample()
        with pytest.raises(RuntimeError, match="plan lifetime validation failed"):
            overlap.sample_history_into(np.empty(2, dtype=np.complex128))


def test_native_design_plan_survives_restart_but_rejects_reinitialized_fields():
    sim, design = _exact_design_validation_case()
    sim.init_sim()
    field = None
    try:
        field = sampling_grid.NativeDesignField(
            sim,
            sampling_grid.mp.Ez,
            design,
        )
        destination = np.empty(field.local_size, dtype=np.float64)
        sim.restart_fields()
        field.sample_into(destination)

        sim.reset_meep()
        sim.init_sim()
        with pytest.raises(RuntimeError, match="plan lifetime validation failed"):
            field.sample_into(destination)
    finally:
        if field is not None:
            field.release()
        sim.reset_meep()


def test_fast_point_monitor_defers_history_reduction_until_finalize():
    fake_native = _FakeNativeSampler()
    with _native_sampler(fake_native):
        monitor = sampling_grid.FastPointMonitor(
            _FakeSimulation(),
            component=0,
            positions=[
                sampling_grid.mp.Vector3(1.0, 2.0),
                sampling_grid.mp.Vector3(3.0, 4.0),
            ],
        )
        local_history = np.empty((3, 2), dtype=np.complex128)
        for row in local_history:
            monitor.sample_history_into(row)
        history = monitor.reduce_history(local_history)

    assert fake_native.monitor_local_calls == 3
    assert fake_native.monitor_real_local_calls == 0
    assert fake_native.monitor_complex_local_calls == 3
    assert fake_native.monitor_allreduced_calls == 0
    assert fake_native.reduction_calls == 1
    assert fake_native.real_reduction_calls == 0
    assert fake_native.complex_reduction_calls == 1
    assert np.array_equal(history, local_history)


def test_fast_point_monitor_records_real_history_without_complex_staging():
    fake_native = _FakeNativeSampler()
    simulation = _FakeSimulation()
    simulation.fields = type(
        "RealFields",
        (),
        {"this": 123, "is_real": True},
    )()
    with _native_sampler(fake_native):
        monitor = sampling_grid.FastPointMonitor(
            simulation,
            component=0,
            positions=[
                sampling_grid.mp.Vector3(1.0, 2.0),
                sampling_grid.mp.Vector3(3.0, 4.0),
            ],
        )
        local_history = np.empty((3, 2), dtype=np.float64)
        for row in local_history:
            monitor.sample_history_into(row)
        history = monitor.reduce_history(local_history)

    assert history is local_history
    assert history.dtype == np.float64
    assert np.array_equal(history, [[1.0, 3.0]] * 3)
    assert fake_native.monitor_local_calls == 3
    assert fake_native.monitor_real_local_calls == 3
    assert fake_native.monitor_complex_local_calls == 0
    assert fake_native.reduction_calls == 1
    assert fake_native.real_reduction_calls == 1
    assert fake_native.complex_reduction_calls == 0


def test_fast_point_monitor_records_only_configured_history_indices():
    fake_native = _FakeNativeSampler()
    with _native_sampler(fake_native):
        monitor = sampling_grid.FastPointMonitor(
            _FakeSimulation(),
            component=0,
            positions=[
                sampling_grid.mp.Vector3(1.0, 2.0),
                sampling_grid.mp.Vector3(3.0, 4.0),
                sampling_grid.mp.Vector3(5.0, 6.0),
            ],
        )
        configured = monitor.configure_history_indices([2, 0])
        destination = np.empty(2, dtype=np.complex128)
        monitor.sample_history_into(destination)

        assert np.array_equal(configured, [2, 0])
        assert np.array_equal(destination, [5.0 + 6.0j, 1.0 + 2.0j])
        with pytest.raises(RuntimeError, match="already configured"):
            monitor.configure_history_indices([1])


def test_fast_point_monitor_native_failure_raises_in_serial():
    simulation = _FakeSimulation()
    with _native_sampler(_FailingPointSampler()):
        monitor = sampling_grid.FastPointMonitor(
            simulation,
            component=0,
            positions=[sampling_grid.mp.Vector3(1.0, 2.0)],
        )
        with pytest.raises(RuntimeError, match="packed monitor sampling failed"):
            monitor.sample()
    assert simulation.python_samples == 0


def test_fast_field_grid_splits_native_history_ownership():
    fake_native = _FakeNativeSampler()
    with _native_sampler(fake_native):
        grid = sampling_grid.FastFieldGrid(
            _FakeSimulation(),
            component=0,
            coords_x=[0.0, 1.0, 2.0, 3.0],
            coords_y=[0.0],
        )
        local_indices, boundary_indices = grid.history_sampling_indices()
        values = grid.sample_history_values(local_indices, boundary_indices)

    assert np.array_equal(local_indices, [0])
    assert np.array_equal(boundary_indices, [2, 3])
    assert np.array_equal(fake_native.local_indices, [0])
    assert np.array_equal(fake_native.support_reduced_indices, [2, 3])
    assert np.allclose(values, [10.0, 22.0, 23.0])


def test_fast_field_grid_writes_configured_history_directly():
    fake_native = _FakeNativeSampler()
    with _native_sampler(fake_native):
        grid = sampling_grid.FastFieldGrid(
            _FakeSimulation(),
            component=0,
            coords_x=[0.0, 1.0, 2.0, 3.0],
            coords_y=[0.0],
        )
        local_indices, boundary_indices = grid.history_sampling_indices()
        destination = np.empty(
            local_indices.size + boundary_indices.size,
            dtype=np.complex128,
        )
        grid.sample_history_values_into(
            local_indices,
            boundary_indices,
            destination,
        )

    assert np.array_equal(fake_native.history_local_indices, [0])
    assert np.array_equal(fake_native.history_boundary_indices, [2, 3])
    assert np.allclose(destination, [10.0, 22.0, 23.0])


def test_fast_field_grid_does_not_reuse_history_plan_for_different_indices():
    fake_native = _FakeNativeSampler()
    with _native_sampler(fake_native):
        grid = sampling_grid.FastFieldGrid(
            _FakeSimulation(),
            component=0,
            coords_x=[0.0, 1.0, 2.0, 3.0],
            coords_y=[0.0],
        )
        grid.history_sampling_indices()
        destination = np.empty(3, dtype=np.complex128)
        grid.sample_history_values_into(
            np.array([1], dtype=np.int64),
            np.array([0, 3], dtype=np.int64),
            destination,
        )

    assert np.array_equal(fake_native.local_indices, [1])
    assert np.array_equal(fake_native.support_reduced_indices, [0, 3])
    assert np.allclose(destination, [11.0, 20.0, 23.0])


def test_fast_field_grid_native_history_failure_raises_in_serial():
    with _native_sampler(_FailingHistorySampler()):
        grid = sampling_grid.FastFieldGrid(
            _FakeSimulation(),
            component=0,
            coords_x=[0.0, 1.0, 2.0, 3.0],
            coords_y=[0.0],
        )
        local_indices, boundary_indices = grid.history_sampling_indices()
        destination = np.empty(
            local_indices.size + boundary_indices.size,
            dtype=np.complex128,
        )
        with pytest.raises(RuntimeError, match="direct history sampling failed"):
            grid.sample_history_values_into(
                local_indices,
                boundary_indices,
                destination,
            )


def test_fast_gradient_grid_accumulates_indexed_values_natively():
    fake_native = _FakeNativeSampler()
    with _native_sampler(fake_native):
        grid = sampling_grid.FastGradientGrid(
            _FakeSimulation(),
            component=0,
            coords_x=[0.0, 1.0, 2.0, 3.0],
            coords_y=[0.0],
        )
        grid.accumulate_indexed(
            np.array([1, 3], dtype=np.int64),
            np.array([5.0, 7.0], dtype=np.complex128),
        )
        result = grid.finalize()

    assert np.array_equal(fake_native.accumulated_indices, [1, 3])
    assert np.allclose(fake_native.accumulated_values, [5.0, 7.0])
    assert np.allclose(result.reshape(-1), [0.0, 10.0, 0.0, 14.0])


def test_fast_gradient_grid_accumulates_native_time_difference():
    fake_native = _FakeNativeSampler()
    with _native_sampler(fake_native):
        grid = sampling_grid.FastGradientGrid(
            _FakeSimulation(),
            component=0,
            coords_x=[0.0, 1.0, 2.0, 3.0],
            coords_y=[0.0],
        )
        grid.configure_fixed_indices(np.array([1, 3], dtype=np.int64))
        initialized = grid.accumulate_fixed_time_difference(
            np.array([99.0, 99.0], dtype=np.complex128),
            0.5,
        )
        accumulated = grid.accumulate_fixed_time_difference(
            np.array([5.0, 7.0], dtype=np.complex128),
            0.5,
        )
        result = grid.finalize()

    assert initialized is False
    assert accumulated is True
    assert np.allclose(result.reshape(-1), [0.0, 20.0, 0.0, 28.0])


def test_native_reduction_failure_raises_in_serial():
    local = np.ones((2, 2), dtype=np.complex128)
    with _native_sampler(_FailingReductionSampler()):
        with pytest.raises(RuntimeError, match="gradient reduction failed"):
            sampling_grid.FastFieldGrid.reduce(local)


@pytest.mark.parametrize("dtype, reducer", (
    (np.complex64, "reduce_inplace"),
    (np.complex128, "reduce_inplace"),
    (np.float32, "reduce_real_inplace"),
    (np.float64, "reduce_real_inplace"),
))
def test_native_inplace_reduction_reuses_owned_buffer(dtype, reducer):
    local = np.arange(6, dtype=dtype).reshape(2, 3)
    with _native_sampler(_FakeNativeSampler()):
        reduced = getattr(sampling_grid.FastFieldGrid, reducer)(local)

    assert reduced is local
    assert reduced.dtype == dtype


@pytest.mark.parametrize("dtype, reducer", (
    (np.complex64, "reduce_complex_grid_sum_inplace"),
    (np.complex128, "reduce_complex_grid_sum_inplace"),
    (np.float32, "reduce_real_grid_sum_inplace"),
    (np.float64, "reduce_real_grid_sum_inplace"),
))
def test_compiled_native_inplace_reduction_preserves_serial_buffer(dtype, reducer):
    local = np.arange(6, dtype=dtype).reshape(2, 3)
    expected = local.copy()
    pointer = local.__array_interface__["data"][0]

    reduced = getattr(sampling_grid.native_sampler, reducer)(local)

    assert reduced is local
    assert reduced.dtype == dtype
    assert local.__array_interface__["data"][0] == pointer
    assert np.array_equal(local, expected)


def test_compiled_native_reducers_preserve_out_of_place_contract():
    local = np.arange(6, dtype=np.complex128).reshape(2, 3)
    expected = local.copy()

    reduced = sampling_grid.native_sampler.reduce_complex_grid_sum(local)

    assert reduced is not local
    assert np.array_equal(local, expected)
    assert np.array_equal(reduced, expected)


def test_compiled_native_inplace_reduction_accepts_empty_buffer():
    local = np.empty((0, 3), dtype=np.complex128)

    reduced = sampling_grid.native_sampler.reduce_complex_grid_sum_inplace(local)

    assert reduced is local
    assert reduced.shape == (0, 3)


def test_compiled_native_inplace_reduction_rejects_unsupported_layouts():
    noncontiguous = np.ones((2, 4), dtype=np.complex128)[:, ::2]
    readonly = np.ones((2, 2), dtype=np.complex128)
    readonly.flags.writeable = False
    swapped = np.ones((2, 2), dtype=np.dtype(">c16"))

    with pytest.raises(ValueError, match="writable C-contiguous native-endian"):
        sampling_grid.native_sampler.reduce_complex_grid_sum_inplace(noncontiguous)
    with pytest.raises(ValueError, match="writable C-contiguous native-endian"):
        sampling_grid.native_sampler.reduce_complex_grid_sum_inplace(readonly)
    with pytest.raises(ValueError, match="writable C-contiguous native-endian"):
        sampling_grid.native_sampler.reduce_complex_grid_sum_inplace(swapped)


def test_native_reduction_failure_aborts_mpi_job(monkeypatch):
    abort_messages = []

    class AbortCalled(RuntimeError):
        pass

    def abort(message):
        abort_messages.append(message)
        raise AbortCalled(message)

    monkeypatch.setattr(sampling_grid.mp, "count_processors", lambda: 2)
    monkeypatch.setattr(sampling_grid.mp, "my_rank", lambda: 1)
    monkeypatch.setattr(sampling_grid.mp, "abort", abort)
    with _native_sampler(_FailingReductionSampler()):
        with pytest.raises(AbortCalled):
            sampling_grid.FastFieldGrid.reduce(
                np.ones((2, 2), dtype=np.complex128)
            )

    assert abort_messages == ["TAMA native sampler failure"]


@pytest.mark.parametrize(
    "factory",
    [
        lambda sim: sampling_grid.FastPointMonitor(
            sim,
            component=0,
            positions=[sampling_grid.mp.Vector3()],
        ),
        lambda sim: sampling_grid.FastFieldGrid(
            sim,
            component=0,
            coords_x=[0.0],
            coords_y=[0.0],
        ),
    ],
)
def test_native_sampling_rejects_meep_symmetries(factory):
    simulation = _FakeSimulation()
    simulation.symmetries = [object()]
    with pytest.raises(ValueError, match="does not support Meep symmetries"):
        factory(simulation)
