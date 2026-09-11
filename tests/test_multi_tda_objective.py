from contextlib import contextmanager
import inspect
import os
import tempfile

import autograd.numpy as npa
import meep as mp
import numpy as np
import pytest
import scipy.interpolate as spi
from scipy.signal import convolve, freqz

import tama as tm
import tama.eigenmode as eigenmode_module
import tama.flux as flux_module
import tama.multi_tda_objective as multi_tda_module
from tama.objectives import _tabulated_cubic_source
from _objective_test_helpers import (
    _FakeFields,
    _fake_indexed_transpose_stencil,
    _run_fake_adjoint_loop,
)


def _make_minimal_multi_tda(**kwargs):
    params = dict(
        update_design=lambda _: None,
        sim_factory=lambda sources=None: None,
        t_final=1.0,
        monitor_positions=[mp.Vector3()],
        component=mp.Ez,
        wavelength_bands=[(0.4, 0.5)],
        weights=[1.0],
        kernel_length=9,
        pixel_chunk=1,
        adjoint_source_size=mp.Vector3(),
        adjoint_source_amplitude=1.0,
        dt=0.05,
    )
    params.update(kwargs)
    return tm.MultiTDAObjective(**params)


def test_multi_tda_auto_balancer_is_default_for_direct_unset_simulation_spec():
    spec = tm.SimulationSpec(
        cell_size=mp.Vector3(1.0, 1.0),
        sources=[],
    )
    objective = _make_minimal_multi_tda(
        simulation=spec,
        sim_factory=None,
    )

    assert isinstance(objective.chunk_balancer, tm.AdaptiveAdjointChunkBalancer)


def test_multi_tda_none_disables_automatic_balancing():
    spec = tm.SimulationSpec(
        cell_size=mp.Vector3(1.0, 1.0),
        sources=[],
    )
    objective = _make_minimal_multi_tda(
        simulation=spec,
        sim_factory=None,
        chunk_balancer=None,
    )

    assert objective.chunk_balancer is None


def test_multi_tda_auto_balancer_resolves_fresh_sources_once_per_evaluation():
    resolved = []

    def make_sources():
        sources = (
            mp.Source(
                mp.GaussianSource(frequency=1.0, fwidth=0.2),
                mp.Ez,
                center=mp.Vector3(),
            ),
        )
        resolved.append(sources)
        return sources

    spec = tm.SimulationSpec(
        cell_size=mp.Vector3(1.0, 1.0),
        sources=make_sources,
    )
    objective = _make_minimal_multi_tda(
        simulation=spec,
        sim_factory=None,
    )
    made = []
    spec.make = lambda sources=None: made.append(sources) or object()

    first = objective._forward_sources_for_evaluation()
    objective._make_forward_simulation(first)
    second = objective._forward_sources_for_evaluation()
    objective._make_forward_simulation(second)

    assert len(resolved) == 2
    assert first is resolved[0]
    assert second is resolved[1]
    assert first is not second
    assert made == [first, second]


@pytest.mark.parametrize(
    "t_final",
    (True, np.bool_(False), 0.0, -1.0, np.inf, -np.inf, np.nan),
)
def test_multi_tda_requires_positive_finite_t_final(t_final):
    with pytest.raises(
        ValueError,
        match="t_final must be a positive finite number",
    ):
        _make_minimal_multi_tda(t_final=t_final)


@pytest.mark.parametrize(
    "weights",
    (1.0, np.ones((1, 1)), [np.nan], [np.inf], [-np.inf]),
)
def test_multi_tda_requires_one_dimensional_finite_weights(weights):
    with pytest.raises(ValueError, match="weights must"):
        _make_minimal_multi_tda(weights=weights)


def test_multi_tda_allows_negative_finite_weights():
    obj = _make_minimal_multi_tda(weights=[-1.0])

    assert np.array_equal(obj.weights, np.asarray([-1.0]))


def _make_joint_filter_bank(**kwargs):
    params = dict(
        monitor_positions=[
            mp.Vector3(0.0),
            mp.Vector3(0.1),
            mp.Vector3(0.2),
            mp.Vector3(0.3),
        ],
        wavelength_bands=[
            (0.4, 0.5),
            (0.5, 0.6),
            (0.6, 0.7),
            (0.7, 0.8),
        ],
        weights=np.ones(4),
        kernel_length=2001,
        dt=0.0125,
        filter_bank="power_complementary",
        transition_width=0.08,
    )
    params.update(kwargs)
    return _make_minimal_multi_tda(**params)


def test_multi_tda_explicit_independent_filter_bank_preserves_default_kernels():
    default = _make_minimal_multi_tda()
    explicit = _make_minimal_multi_tda(filter_bank="independent")

    assert len(default.kernels) == len(explicit.kernels)
    for default_kernel, explicit_kernel in zip(
        default.kernels,
        explicit.kernels,
    ):
        np.testing.assert_array_equal(default_kernel, explicit_kernel)


def test_multi_tda_rejects_unknown_filter_bank():
    with pytest.raises(ValueError, match="filter_bank"):
        _make_minimal_multi_tda(filter_bank="equiripple")


def test_power_complementary_filter_bank_requires_multiple_bands():
    with pytest.raises(ValueError, match="at least two"):
        _make_minimal_multi_tda(filter_bank="power_complementary")


@pytest.mark.parametrize(
    "wavelength_bands",
    (
        [(0.4, 0.5), (0.55, 0.65)],
        [(0.4, 0.55), (0.5, 0.6)],
        [(0.5, 0.6), (0.4, 0.5)],
    ),
)
def test_power_complementary_filter_bank_requires_ordered_contiguous_bands(
    wavelength_bands,
):
    with pytest.raises(ValueError, match="contiguous"):
        _make_minimal_multi_tda(
            monitor_positions=[mp.Vector3(), mp.Vector3(0.1)],
            wavelength_bands=wavelength_bands,
            weights=[1.0, 1.0],
            kernel_length=101,
            filter_bank="power_complementary",
            transition_width=0.2,
        )


def test_power_complementary_filter_bank_requires_unit_filter_weights():
    with pytest.raises(ValueError, match="unit"):
        _make_joint_filter_bank(weights=[1.0, 1.0, 0.9, 1.0])


@pytest.mark.parametrize(
    "transition_width",
    (True, 0.0, -0.1, np.nan, np.inf, -np.inf),
)
def test_power_complementary_filter_bank_requires_positive_transition_width(
    transition_width,
):
    with pytest.raises(
        ValueError,
        match="transition_width.*positive finite",
    ):
        _make_joint_filter_bank(transition_width=transition_width)


def test_power_complementary_filter_bank_rejects_overlapping_transitions():
    with pytest.raises(ValueError, match="smaller than every band width"):
        _make_joint_filter_bank(transition_width=0.2)


def test_power_complementary_filter_bank_initially_requires_odd_kernel_length():
    with pytest.raises(ValueError, match="odd"):
        _make_joint_filter_bank(kernel_length=2000)


@pytest.mark.parametrize("kernel_length", (True, 1.5, 0, -1))
def test_multi_tda_requires_positive_integer_kernel_length(kernel_length):
    with pytest.raises(
        ValueError,
        match="kernel_length must be a positive integer",
    ):
        _make_minimal_multi_tda(kernel_length=kernel_length)


@pytest.mark.parametrize(
    "block_size",
    (True, np.bool_(False), 0, -1, 1.5),
)
def test_multi_tda_requires_positive_integer_target_history_block_size(
    block_size,
):
    with pytest.raises(ValueError, match="target_history_block_size"):
        _make_minimal_multi_tda(target_history_block_size=block_size)


def test_mixed_builtin_targets_keep_legacy_target_history_path():
    objective = _make_minimal_multi_tda(
        simulation=tm.SimulationSpec(
            cell_size=mp.Vector3(2.0, 2.0),
            sources=[],
            resolution=10,
        ),
        sim_factory=None,
        monitor_positions=None,
        component=None,
        adjoint_source_size=None,
        adjoint_source_amplitude=None,
        targets=(
            tm.FieldRegionTarget((mp.Vector3(),), mp.Ez),
            tm.FluxTarget((mp.Vector3(0.5),), mp.Vector3(1.0)),
        ),
        wavelength_bands=((0.4, 0.5), (0.5, 0.6)),
        weights=(1.0, 1.0),
        target_history_block_size=1,
        chunk_balancer=None,
    )

    assert not objective._uses_distributed_builtin_target_history()


@pytest.mark.parametrize(
    "wavelength_bands",
    (
        [],
        [(0.4, 0.4)],
        [(0.5, 0.4)],
        [(0.0, 0.5)],
        [(0.4, 0.0)],
        [(-0.4, 0.5)],
        [(0.4, -0.5)],
        [(np.nan, 0.5)],
        [(0.4, np.inf)],
        [(-np.inf, 0.5)],
    ),
)
def test_multi_tda_requires_nonempty_positive_finite_ordered_wavelength_bands(
    wavelength_bands,
):
    kwargs = {}
    if not wavelength_bands:
        kwargs = {"monitor_positions": [], "weights": []}

    with pytest.raises(ValueError, match=r"wavelength.?band"):
        _make_minimal_multi_tda(
            wavelength_bands=wavelength_bands,
            **kwargs,
        )


@pytest.mark.parametrize("name", ("dt", "resolution"))
@pytest.mark.parametrize("value", (0.0, -1.0, np.nan, np.inf, -np.inf))
def test_multi_tda_requires_positive_finite_time_grid_inputs(name, value):
    kwargs = {name: value}
    if name == "resolution":
        kwargs["dt"] = None

    with pytest.raises(
        ValueError,
        match=rf"{name} must be a positive finite number",
    ):
        _make_minimal_multi_tda(**kwargs)


@pytest.mark.parametrize("beta", (np.nan, np.inf, -np.inf))
def test_multi_tda_requires_finite_kernel_kaiser_beta(beta):
    with pytest.raises(ValueError, match=r"(?i)beta.*finite"):
        _make_minimal_multi_tda(
            kernel_window="kaiser",
            kernel_window_params={"beta": beta},
        )


def test_distributed_target_transform_preserves_serial_exception():
    obj = _make_minimal_multi_tda(
        wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
        monitor_positions=[mp.Vector3(), mp.Vector3(0.1)],
        weights=[1.0, 1.0],
    )
    assert not obj._uses_deduplicated_monitors
    signals = np.zeros((3, 2))

    def transform(target_index):
        if target_index == 0:
            raise ValueError("intentional target transform failure")
        return obj._raw_target_history(signals, target_index)

    with pytest.raises(ValueError, match="intentional target transform failure"):
        obj._distributed_target_transform(signals, transform)


def test_multi_tda_requires_fom_for_manual_adjoint_signal():
    with pytest.raises(ValueError, match="adjoint_signal_fn requires fom_fn"):
        _make_minimal_multi_tda(
            adjoint_signal_fn=lambda history, dt: np.ones_like(history),
        )


def test_multi_tda_rejects_manual_adjoint_signal_assigned_after_default_fom():
    obj = _make_minimal_multi_tda(design=_make_design())
    obj.adjoint_signal_fn = lambda history, dt: np.ones_like(history)

    with pytest.raises(ValueError, match="adjoint_signal_fn requires fom_fn"):
        obj.evaluate(np.zeros(1), need_gradient=True)


def test_multi_tda_validates_per_band_callback_sequences():
    callback = lambda history, dt: np.sum(np.abs(history) ** 2) * dt

    with pytest.raises(ValueError, match="fom_fn.*wavelength bands"):
        _make_minimal_multi_tda(
            wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
            monitor_positions=[mp.Vector3(), mp.Vector3(0.1)],
            weights=[1.0, 1.0],
            fom_fn=(callback,),
        )

    with pytest.raises(TypeError, match="fom_fn.*callable"):
        _make_minimal_multi_tda(fom_fn=(object(),))

    with pytest.raises(ValueError, match="band 1.*requires fom_fn"):
        _make_minimal_multi_tda(
            wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
            monitor_positions=[mp.Vector3(), mp.Vector3(0.1)],
            weights=[1.0, 1.0],
            fom_fn=(callback, None),
            adjoint_signal_fn=(None, lambda history, dt: history),
        )

    obj = _make_minimal_multi_tda(
        wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
        monitor_positions=[mp.Vector3(), mp.Vector3(0.1)],
        weights=[1.0, 1.0],
    )
    obj.fom_fn = (callback, None)
    history = np.ones((4, 1), dtype=np.complex128)
    custom_value, _ = obj._band_fom_value_and_adjoint_signal(
        history,
        0,
        need_gradient=False,
    )
    default_value, _ = obj._band_fom_value_and_adjoint_signal(
        history,
        1,
        need_gradient=False,
    )
    assert custom_value == pytest.approx(4 * obj.dt)
    assert default_value == pytest.approx(2 * obj.dt)

    obj.fom_fn = (callback,)
    with pytest.raises(ValueError, match="fom_fn.*wavelength bands"):
        obj.evaluate(np.zeros(1), need_gradient=False)

    obj = _make_minimal_multi_tda(
        design=_make_design(),
        wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
        monitor_positions=[mp.Vector3(), mp.Vector3(0.1)],
        weights=[1.0, 1.0],
    )
    obj.fom_fn = (callback, None)
    obj.adjoint_signal_fn = (None, lambda history, dt: history)
    with pytest.raises(ValueError, match="band 1.*requires fom_fn"):
        obj.evaluate(np.zeros(1), need_gradient=True)


def test_multi_tda_rejects_rank_inconsistent_custom_callback_state(
    monkeypatch,
):
    reductions = iter((0, 1))
    monkeypatch.setattr(multi_tda_module.mp, "count_processors", lambda: 2)
    monkeypatch.setattr(
        multi_tda_module.mp,
        "sum_to_all",
        lambda value: next(reductions),
    )
    obj = _make_minimal_multi_tda()

    with pytest.raises(RuntimeError, match="configured consistently"):
        obj._validate_runtime_band_callbacks(validate_pairs=False)


def test_multi_tda_rejects_rank_inconsistent_adjoint_callback_state(
    monkeypatch,
):
    callback = lambda history, dt: np.sum(np.abs(history) ** 2) * dt
    reductions = iter((0, 2, 1))
    monkeypatch.setattr(multi_tda_module.mp, "count_processors", lambda: 2)
    monkeypatch.setattr(
        multi_tda_module.mp,
        "sum_to_all",
        lambda value: next(reductions),
    )
    obj = _make_minimal_multi_tda(fom_fn=callback)

    with pytest.raises(
        RuntimeError,
        match="adjoint_signal_fn.*each band",
    ):
        obj._validate_runtime_band_callbacks(validate_pairs=False)


def test_multi_tda_objective_rejects_meep_symmetries_before_forward_run():
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
    objective = _make_minimal_multi_tda(
        sim_factory=lambda sources=None: simulation,
    )

    with pytest.raises(ValueError, match="does not support Meep symmetries"):
        objective.evaluate(np.zeros(1), need_gradient=False)

    assert not simulation.run_called
    assert simulation.reset_called


def test_multi_tda_objective_rejects_meep_symmetries_before_adjoint_run():
    forward = _FakeSimulation(0.05, 8)
    adjoint = _FakeSimulation(0.05, 8)
    adjoint.symmetries = [object()]
    simulations = iter((forward, adjoint))
    objective = _make_native_multi_tda(
        sim_factory=lambda sources=None: next(simulations),
        dt=0.05,
    )

    with _fake_native_path():
        with pytest.raises(ValueError, match="does not support Meep symmetries"):
            objective.evaluate(np.zeros(1), need_gradient=True)

    assert forward.events.count("run") == 1
    assert "run" not in adjoint.events
    assert "reset_meep" in adjoint.events


def _make_design(shape=(1, 1)):
    air = mp.Medium(epsilon=1.0)
    dielectric = mp.Medium(epsilon=4.0)
    return tm.DesignGrid(
        material_grid=mp.MaterialGrid(mp.Vector3(*shape), air, dielectric),
        center=mp.Vector3(),
        size=mp.Vector3(*shape),
        shape=shape,
        background=air,
        design_material=dielectric,
    )


def _make_cylindrical_design(shape=(1, 1)):
    air = mp.Medium(epsilon=1.0)
    dielectric = mp.Medium(epsilon=4.0)
    nr, nz = shape
    return tm.DesignGrid(
        material_grid=mp.MaterialGrid(
            mp.Vector3(nr, 1, nz),
            air,
            dielectric,
        ),
        center=mp.Vector3(0.5 * nr, 0, 0),
        size=mp.Vector3(nr, 0, nz),
        shape=shape,
        background=air,
        design_material=dielectric,
        coordinate_system="cylindrical",
    )


def _make_native_multi_tda(*, shape=(1, 1), **kwargs):
    design = kwargs.pop("design", None)
    if design is None:
        design = _make_design(shape)
    return _make_minimal_multi_tda(design=design, **kwargs)


def test_distributed_band_transform_requires_native_reduction(
    monkeypatch,
):
    monkeypatch.setattr(multi_tda_module.mp, "count_processors", lambda: 2)
    monkeypatch.setattr(multi_tda_module.mp, "my_rank", lambda: 0)
    monkeypatch.setattr(
        multi_tda_module.FastFieldGrid,
        "reduce",
        lambda status: status,
    )
    monkeypatch.setattr(
        multi_tda_module.FastFieldGrid,
        "reduce_real_inplace",
        lambda local: local
        + np.array([[0, 2, 0], [0, 2, 0]], dtype=np.float64),
    )
    signals = np.zeros((2, 3), dtype=np.complex128)
    transformed_bands = []

    def transform(band_index):
        transformed_bands.append(band_index)
        return np.full(2, band_index + 1, dtype=np.complex128)

    result = tm.MultiTDAObjective._distributed_band_transform(
        signals,
        transform,
    )

    assert transformed_bands == [0, 2]
    assert np.array_equal(
        result,
        np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float64),
    )


def test_distributed_band_transform_preserves_serial_exception():
    signals = np.zeros((2, 1))

    def transform(band_index):
        raise ValueError("intentional band transform failure")

    with pytest.raises(ValueError, match="intentional band transform failure"):
        tm.MultiTDAObjective._distributed_band_transform(
            signals,
            transform,
        )


def test_multi_tda_objective_accepts_numpy_kernel_window_options():
    rectangular = _make_minimal_multi_tda(kernel_window="rectangular")
    hamming = _make_minimal_multi_tda(kernel_window="hamming")
    kaiser_4 = _make_minimal_multi_tda(
        kernel_window="kaiser",
        kernel_window_params={"beta": 4.0},
    )
    kaiser_8 = _make_minimal_multi_tda(
        kernel_window="kaiser",
        kernel_window_params={"beta": 8.0},
    )

    assert not np.allclose(rectangular.kernels[0], hamming.kernels[0])
    assert not np.allclose(kaiser_4.kernels[0], kaiser_8.kernels[0])


def _filter_bank_power_response(obj, frequencies):
    sample_frequency = 1.0 / obj.dt
    return np.asarray(
        [
            np.abs(
                freqz(
                    kernel * obj.dt,
                    worN=frequencies,
                    fs=sample_frequency,
                )[1]
            )
            ** 2
            for kernel in obj.kernels
        ]
    )


def test_power_complementary_panel_b_response_covers_requested_union():
    obj = _make_joint_filter_bank()
    frequencies = np.linspace(1.0 / 0.8, 1.0 / 0.4, 4097)
    power = _filter_bank_power_response(obj, frequencies)
    total_power = np.sum(power, axis=0)

    assert np.min(total_power) >= 0.96
    assert np.max(total_power) <= 1.04

    for edge_index, wavelength_edge in enumerate((0.5, 0.6, 0.7)):
        edge_power = _filter_bank_power_response(
            obj,
            np.asarray([1.0 / wavelength_edge]),
        )[:, 0]
        adjacent = edge_power[edge_index : edge_index + 2]
        assert np.all(adjacent >= 0.4)
        assert np.all(adjacent <= 0.6)
        assert np.sum(edge_power) == pytest.approx(1.0, abs=0.05)


def test_power_complementary_filter_bank_resolves_automatic_transition_width():
    obj = _make_joint_filter_bank(transition_width=None)
    edge_power = _filter_bank_power_response(
        obj,
        np.asarray([1.0 / 0.6]),
    )[:, 0]

    assert all(np.all(np.isfinite(kernel)) for kernel in obj.kernels)
    assert obj.resolved_transition_width == pytest.approx(
        2.0 / (obj.kernel_length * obj.dt)
    )
    assert np.sum(edge_power) == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize("complex_signals", (False, True))
def test_power_complementary_filter_transpose_satisfies_bilinear_identity(
    complex_signals,
):
    rng = np.random.default_rng(20260812 + int(complex_signals))
    obj = _make_joint_filter_bank()
    signals = rng.standard_normal((257, 4))
    filtered_covectors = rng.standard_normal((257 + obj.kernel_length - 1, 4))
    if complex_signals:
        signals = signals + 1j * rng.standard_normal(signals.shape)
        filtered_covectors = (
            filtered_covectors
            + 1j * rng.standard_normal(filtered_covectors.shape)
        )

    filtered = obj.filter_monitor_signals(signals)
    pulled_back = obj._filter_transpose_adjoint_signals(
        filtered_covectors,
        np.ones(4),
    )

    assert np.real(np.sum(filtered_covectors * filtered)) == pytest.approx(
        np.real(np.sum(pulled_back * signals)),
        rel=2.0e-12,
        abs=2.0e-12,
    )


@pytest.mark.parametrize("complex_signals", (False, True))
def test_power_complementary_smooth_min_matches_directional_finite_difference(
    complex_signals,
):
    rng = np.random.default_rng(20260814 + int(complex_signals))
    beta = 0.7

    def smooth_min(values):
        return -npa.log(npa.sum(npa.exp(-beta * values))) / beta

    obj = _make_joint_filter_bank(scalarization_fn=smooth_min)
    signals = rng.standard_normal((257, 4))
    direction = rng.standard_normal(signals.shape)
    if complex_signals:
        signals = signals + 1j * rng.standard_normal(signals.shape)
        direction = direction + 1j * rng.standard_normal(direction.shape)

    filtered = obj.filter_monitor_signals(signals)
    band_objectives, filtered_adjoint_signals = (
        obj._band_fom_values_and_adjoint_signals(
            filtered,
            need_gradient=True,
        )
    )
    _, band_coeffs, _ = obj._evaluate_scalarization(
        band_objectives,
        need_gradient=True,
    )
    monitor_adjoint_signals = obj._filter_transpose_adjoint_signals(
        filtered_adjoint_signals,
        band_coeffs,
    )
    adjoint_derivative = obj.dt * np.real(
        np.sum(monitor_adjoint_signals * direction)
    )

    def objective(values):
        filtered_values = obj.filter_monitor_signals(values)
        values_by_band = obj._band_fom_values_and_adjoint_signals(
            filtered_values,
            need_gradient=False,
        )[0]
        return float(smooth_min(values_by_band))

    step = 1.0e-6
    finite_difference = (
        objective(signals + step * direction)
        - objective(signals - step * direction)
    ) / (2.0 * step)

    assert adjoint_derivative == pytest.approx(
        finite_difference,
        rel=2.0e-8,
        abs=2.0e-9,
    )


def test_multi_tda_objective_preserves_complex_adjoint_source_amplitude():
    amplitude = 0.6 - 0.8j

    obj = _make_minimal_multi_tda(
        adjoint_source_amplitude=amplitude,
    )

    assert obj.adjoint_source_amplitude == amplitude


def test_multi_tda_objective_exports_and_filters_signals():
    obj = tm.MultiTDAObjective(
        update_design=lambda _: None,
        sim_factory=lambda sources=None: None,
        t_final=1.0,
        monitor_positions=[mp.Vector3()],
        component=mp.Ez,
        wavelength_bands=[(0.4, 0.5)],
        weights=[1.0],
        kernel_length=9,
        pixel_chunk=1,
        adjoint_source_size=mp.Vector3(),
        adjoint_source_amplitude=1.0,
        dt=0.05,
    )

    signal = np.zeros((32, 1), dtype=np.complex128)
    signal[16, 0] = 1.0
    filtered = obj.filter_monitor_signals(signal)

    assert filtered.shape == (signal.shape[0] + obj.kernel_length - 1, 1)
    assert np.iscomplexobj(filtered)
    assert obj.last_total_fom is None


@pytest.mark.parametrize("kernel_length", (4, 5, 2001))
@pytest.mark.parametrize("channel_count", (1, 3))
def test_temporal_fft_convolution_matches_direct_linear_convolution(
    kernel_length,
    channel_count,
):
    rng = np.random.default_rng(20260731 + kernel_length + channel_count)
    sample_count = 10001 if kernel_length == 2001 else 23
    signal = rng.standard_normal((sample_count, channel_count))
    signal = signal + 1j * rng.standard_normal(signal.shape)
    kernel = rng.standard_normal(kernel_length)
    dt = 0.0125

    expected = convolve(
        signal,
        kernel[:, None],
        mode="full",
        method="direct",
    ) * dt
    actual = multi_tda_module.temporal_convolve_signal(signal, kernel, dt)

    assert np.allclose(actual, expected, rtol=2.0e-12, atol=2.0e-12)

    cotangent = rng.standard_normal(expected.shape) + 1j * rng.standard_normal(expected.shape)
    expected_transpose = convolve(
        cotangent,
        kernel[::-1, None].conj(),
        mode="valid",
        method="direct",
    ) * dt
    actual_transpose = multi_tda_module.temporal_convolve_signal_transpose(
        cotangent,
        kernel,
        dt,
    )

    assert np.allclose(
        actual_transpose,
        expected_transpose,
        rtol=2.0e-12,
        atol=2.0e-12,
    )


@pytest.mark.parametrize(
    ("signal", "kernel"),
    (
        (np.array([1]), np.array([-1, -1, 4])),
        (
            np.array([True, False, True]),
            np.array([1.0, -1.0, 1.0]),
        ),
    ),
)
def test_temporal_convolution_promotes_nonfloating_signal(signal, kernel):
    expected = np.convolve(signal.astype(float), kernel, "full")
    actual = multi_tda_module.temporal_convolve_signal(signal, kernel, 1.0)

    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)

    cotangent = np.arange(expected.size).astype(signal.dtype)
    expected_transpose = np.convolve(cotangent.astype(float), kernel[::-1], "valid")
    actual_transpose = multi_tda_module.temporal_convolve_signal_transpose(
        cotangent,
        kernel,
        1.0,
    )

    assert actual_transpose.dtype == np.float64
    np.testing.assert_allclose(actual_transpose, expected_transpose, rtol=2e-12, atol=2e-12)


def test_complex_band_adjoint_signal_matches_directional_finite_difference():
    rng = np.random.default_rng(20260723)
    signal = rng.standard_normal(31) + 1j * rng.standard_normal(31)
    direction = rng.standard_normal(31) + 1j * rng.standard_normal(31)
    kernel = rng.standard_normal(8)
    dt = 0.075

    def objective(values):
        filtered = multi_tda_module.temporal_convolve_signal(values, kernel, dt)
        return 0.5 * np.sum(np.abs(filtered) ** 2) * dt

    filtered = multi_tda_module.temporal_convolve_signal(signal, kernel, dt)
    adjoint_signal = np.conjugate(
        multi_tda_module.temporal_convolve_signal_transpose(
            filtered,
            kernel,
            dt,
        )
    )
    adjoint_derivative = dt * np.real(np.sum(adjoint_signal * direction))
    step = 1.0e-6
    finite_difference = (
        objective(signal + step * direction)
        - objective(signal - step * direction)
    ) / (2.0 * step)

    assert np.allclose(
        adjoint_derivative,
        finite_difference,
        rtol=1.0e-8,
        atol=1.0e-10,
    )


@pytest.mark.parametrize("kernel_length", (4, 5))
def test_custom_band_fom_and_scalarization_match_directional_finite_difference(
    kernel_length,
):
    rng = np.random.default_rng(20260727)
    dt = 0.075
    signals = (
        rng.standard_normal((29, 2))
        + 1j * rng.standard_normal((29, 2))
    )
    direction = (
        rng.standard_normal(signals.shape)
        + 1j * rng.standard_normal(signals.shape)
    )
    kernels = [
        np.array([0.2, -0.4, 0.7, 0.1, 0.3])[:kernel_length],
        np.array([-0.3, 0.5, 0.8, -0.2, 0.4])[:kernel_length],
    ]

    def fom_fn(history, sample_dt):
        return sample_dt * (
            0.25 * npa.sum(npa.abs(history) ** 4)
            + 0.1 * npa.sum(npa.real(history) ** 2)
        )

    def scalarization_fn(band_objectives):
        return band_objectives[0] + 0.7 * band_objectives[1] ** 2

    obj = _make_minimal_multi_tda(
        monitor_positions=[mp.Vector3(), mp.Vector3(0.1)],
        wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
        weights=[1.0, 1.0],
        fom_fn=fom_fn,
        scalarization_fn=scalarization_fn,
        dt=dt,
        kernel_length=kernel_length,
    )
    obj.weighted_kernels = kernels

    def filtered(values):
        return np.column_stack([
            multi_tda_module.temporal_convolve_signal(
                values[:, band_index],
                kernels[band_index],
                dt,
            )
            for band_index in range(values.shape[1])
        ])

    filtered_signals = filtered(signals)
    band_objectives, filtered_adjoint_signals = (
        obj._band_fom_values_and_adjoint_signals(
            filtered_signals,
            need_gradient=True,
        )
    )
    _, band_coeffs, _ = obj._evaluate_scalarization(
        band_objectives,
        need_gradient=True,
    )
    monitor_adjoint_signals = obj._filter_transpose_adjoint_signals(
        filtered_adjoint_signals,
        band_coeffs,
    )
    adjoint_derivative = dt * np.real(
        np.sum(monitor_adjoint_signals * direction)
    )

    def objective(values):
        values_by_band = np.array([
            float(fom_fn(filtered(values)[:, band_index], dt))
            for band_index in range(values.shape[1])
        ])
        return scalarization_fn(values_by_band)

    step = 1.0e-6
    finite_difference = (
        objective(signals + step * direction)
        - objective(signals - step * direction)
    ) / (2.0 * step)

    assert np.allclose(
        adjoint_derivative,
        finite_difference,
        rtol=1.0e-8,
        atol=1.0e-9,
    )


@pytest.mark.parametrize("kernel_length", (4, 5))
def test_mixed_custom_and_default_band_foms_match_directional_finite_difference(
    kernel_length,
):
    rng = np.random.default_rng(20260801)
    dt = 0.075
    signals = (
        rng.standard_normal((31, 2))
        + 1j * rng.standard_normal((31, 2))
    )
    direction = (
        rng.standard_normal(signals.shape)
        + 1j * rng.standard_normal(signals.shape)
    )
    kernels = (
        np.array([0.2, -0.4, 0.7, 0.1, 0.3])[:kernel_length],
        np.array([-0.3, 0.5, 0.8, -0.2, 0.4])[:kernel_length],
    )

    def first_fom(history, sample_dt):
        return sample_dt * npa.sum(npa.abs(history) ** 4)

    def first_adjoint(history, sample_dt):
        return 4.0 * np.abs(history) ** 2 * np.conjugate(history)

    fom_fns = (first_fom, None)
    obj = _make_minimal_multi_tda(
        monitor_positions=[mp.Vector3(), mp.Vector3(0.1)],
        wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
        weights=[1.0, 1.0],
        fom_fn=fom_fns,
        adjoint_signal_fn=(first_adjoint, None),
        scalarization_fn=lambda values: values[0] + 0.4 * values[1],
        dt=dt,
        kernel_length=kernel_length,
    )
    obj.weighted_kernels = list(kernels)

    def filtered(values):
        return np.column_stack(
            [
                multi_tda_module.temporal_convolve_signal(
                    values[:, band_index],
                    kernels[band_index],
                    dt,
                )
                for band_index in range(values.shape[1])
            ]
        )

    filtered_signals = filtered(signals)
    band_objectives, filtered_adjoint_signals = (
        obj._band_fom_values_and_adjoint_signals(
            filtered_signals,
            need_gradient=True,
        )
    )
    _, band_coeffs, _ = obj._evaluate_scalarization(
        band_objectives,
        need_gradient=True,
    )
    monitor_adjoint_signals = obj._filter_transpose_adjoint_signals(
        filtered_adjoint_signals,
        band_coeffs,
    )
    adjoint_derivative = dt * np.real(
        np.sum(monitor_adjoint_signals * direction)
    )

    def objective(values):
        filtered_values = filtered(values)
        objectives = np.asarray([
            first_fom(filtered_values[:, 0], dt),
            0.5 * dt * np.sum(np.abs(filtered_values[:, 1]) ** 2),
        ])
        return objectives[0] + 0.4 * objectives[1]

    step = 1.0e-6
    finite_difference = (
        objective(signals + step * direction)
        - objective(signals - step * direction)
    ) / (2.0 * step)

    assert np.allclose(
        adjoint_derivative,
        finite_difference,
        rtol=1.0e-8,
        atol=1.0e-9,
    )


def test_multi_tda_flux_filter_pullback_matches_directional_finite_difference():
    rng = np.random.default_rng(20260730)
    dt = 0.075
    target = tm.FluxTarget(
        (mp.Vector3(-0.1), mp.Vector3(0.1)),
        normal=mp.Vector3(-1.0),
        spatial_weights=(0.7, 1.3),
    )
    obj = tm.MultiTDAObjective(
        update_design=lambda _: None,
        sim_factory=lambda sources=None: None,
        targets=[target],
        t_final=1.0,
        wavelength_bands=[(0.4, 0.5)],
        weights=[1.0],
        kernel_length=5,
        pixel_chunk=1,
        dt=dt,
    )
    kernel = np.array([0.2, -0.4, 0.7, 0.1])
    obj.kernel_length = kernel.size
    obj.kernels = [kernel]
    obj.weighted_kernels = [kernel]
    signals = (
        rng.standard_normal((23, len(obj._flat_target_positions)))
        + 1j
        * rng.standard_normal((23, len(obj._flat_target_positions)))
    )
    direction = (
        rng.standard_normal(signals.shape)
        + 1j * rng.standard_normal(signals.shape)
    )

    filtered = obj.filter_monitor_signals(signals)
    band_objectives, filtered_covectors = (
        obj._band_fom_values_and_adjoint_signals(
            filtered,
            need_gradient=True,
        )
    )
    monitor_covectors = obj._filter_transpose_adjoint_signals(
        filtered_covectors,
        np.ones(1),
    )
    adjoint_derivative = dt * np.real(
        np.sum(monitor_covectors * direction)
    )

    def objective(values):
        return float(
            obj._band_fom_values_and_adjoint_signals(
                obj.filter_monitor_signals(values),
                need_gradient=False,
            )[0][0]
        )

    step = 1.0e-6
    finite_difference = (
        objective(signals + step * direction)
        - objective(signals - step * direction)
    ) / (2.0 * step)

    assert np.isfinite(band_objectives[0])
    assert np.allclose(
        adjoint_derivative,
        finite_difference,
        rtol=1.0e-8,
        atol=1.0e-9,
    )


def test_scalar_scalarization_derives_band_coefficients():
    beta = 2.4
    band_objectives = np.array([0.17, 0.63, -0.21])

    def scalarization_fn(values):
        return npa.log(npa.sum(npa.exp(beta * values))) / beta

    obj = _make_minimal_multi_tda(scalarization_fn=scalarization_fn)
    total_fom, band_coeffs, info = obj._evaluate_scalarization(
        band_objectives,
        need_gradient=True,
    )

    exponentials = np.exp(beta * band_objectives)
    expected_coeffs = exponentials / np.sum(exponentials)
    assert np.allclose(
        total_fom,
        np.log(np.sum(exponentials)) / beta,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert np.allclose(
        band_coeffs,
        expected_coeffs,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert info is None


def test_scalar_scalarization_gradient_reports_autograd_requirement():
    def scalarization_fn(values):
        return float(np.sum(np.exp(values)))

    obj = _make_minimal_multi_tda(scalarization_fn=scalarization_fn)
    with pytest.raises(TypeError, match="autograd.numpy"):
        obj._evaluate_scalarization(
            np.array([0.2, 0.4]),
            need_gradient=True,
        )


def test_scalarization_rejects_non_scalar_automatic_result():
    obj = _make_minimal_multi_tda(
        scalarization_fn=lambda values: values,
    )
    with pytest.raises(ValueError, match="must return a scalar"):
        obj._evaluate_scalarization(
            np.array([0.2, 0.4]),
            need_gradient=False,
        )


@pytest.mark.parametrize(
    "source_factory",
    [_tabulated_cubic_source, multi_tda_module._tabulated_cubic_source],
    ids=["shared", "multi"],
)
def test_native_tabulated_cubic_matches_legacy_interp1d(source_factory):
    times = np.linspace(0.07, 0.37, 9)
    values = np.exp(1j * (0.4 * times + 0.2 * times**2))
    native_source = source_factory(times, values)
    reference = spi.interp1d(
        times,
        values,
        kind="cubic",
        fill_value=0j,
        bounds_error=False,
    )
    query = np.linspace(times[0], times[-1], 257)
    native_values = np.asarray([native_source(time) for time in query])

    assert np.allclose(native_values, reference(query), rtol=2.0e-14, atol=2.0e-14)
    assert native_source(times[0] - 0.01) == 0j
    assert native_source(times[-1] + 0.01) == 0j

    native_custom_source = mp.CustomSource(src_func=native_source)
    legacy_custom_source = mp.CustomSource(
        src_func=lambda time: complex(reference(time))
    )
    native_currents = np.asarray([
        native_custom_source.swigobj.current(time, 0.01) for time in query
    ])
    legacy_currents = np.asarray([
        legacy_custom_source.swigobj.current(time, 0.01) for time in query
    ])
    assert np.allclose(native_currents, legacy_currents, rtol=2.0e-14, atol=2.0e-14)


def test_multi_tda_objective_has_no_gradient_mode_parameter():
    assert "gradient_mode" not in inspect.signature(tm.MultiTDAObjective).parameters


def test_multi_tda_objective_validates_sampling_interval():
    assert _make_minimal_multi_tda().sampling_interval == 1
    assert _make_minimal_multi_tda(sampling_interval=2).sampling_interval == 2

    for invalid_interval in (0, -1, 1.5, True):
        try:
            _make_minimal_multi_tda(sampling_interval=invalid_interval)
        except ValueError as exc:
            assert "sampling_interval" in str(exc)
        else:
            raise AssertionError("expected ValueError for invalid sampling_interval")

    try:
        _make_minimal_multi_tda(
            sampling_interval=2,
            max_frequency=1.0,
        )
    except ValueError as exc:
        assert "cannot be combined" in str(exc)
    else:
        raise AssertionError("expected conflicting sampling configuration error")

    configured = _make_minimal_multi_tda(
        reconstruction_window="kaiser",
        reconstruction_window_params={"beta": 4.0},
        reconstruction_half_width=12,
    )
    assert configured.reconstruction_window == "kaiser"
    assert configured.reconstruction_window_params == {"beta": 4.0}
    assert configured.reconstruction_half_width == 12

    try:
        _make_minimal_multi_tda(reconstruction_half_width=0)
    except ValueError as exc:
        assert "reconstruction_half_width" in str(exc)
    else:
        raise AssertionError("expected invalid reconstruction half-width error")


@pytest.mark.parametrize(
    ("wavelength_bands", "max_frequency"),
    (
        ([(0.4, 0.5)], 2.49),
        ([(0.5, 0.6), (0.25, 0.3)], 3.99),
    ),
)
def test_multi_tda_max_frequency_must_cover_band_upper_edges(
    wavelength_bands,
    max_frequency,
):
    monitor_positions = [
        mp.Vector3(0.1 * index)
        for index in range(len(wavelength_bands))
    ]

    with pytest.raises(ValueError, match="highest wavelength-band frequency"):
        _make_minimal_multi_tda(
            monitor_positions=monitor_positions,
            wavelength_bands=wavelength_bands,
            weights=np.ones(len(wavelength_bands)),
            max_frequency=max_frequency,
        )


def test_multi_tda_max_frequency_allows_exact_band_upper_edge():
    obj = _make_minimal_multi_tda(max_frequency=2.5)

    assert obj.max_frequency == 2.5


def test_multi_tda_max_frequency_preserves_iterable_bands():
    wavelength_bands = (band for band in [(0.4, 0.5)])

    obj = _make_minimal_multi_tda(
        wavelength_bands=wavelength_bands,
        max_frequency=2.5,
    )

    assert obj.wavelength_bands == [(0.4, 0.5)]


@pytest.mark.parametrize(
    "kwargs",
    (
        {"f_low": 0.0},
        {"f_low": 2.0, "f_high": 2.0},
        {"f_high": 5.0},
        {"f_high": 5.1},
        {"dt": np.nan},
        {"length": True},
        {"length": 0},
    ),
)
def test_bandpass_kernel_rejects_invalid_or_aliased_time_grid(kwargs):
    parameters = dict(f_low=1.0, f_high=2.0, dt=0.1, length=9)
    parameters.update(kwargs)

    with pytest.raises(ValueError):
        multi_tda_module.bandpass_kernel(**parameters)


def test_multi_tda_time_grid_reconfiguration_is_transactional():
    obj = _make_minimal_multi_tda(dt=0.05)
    original = (
        obj.dt,
        obj.filter_time,
        obj.run_time,
        tuple(np.array(kernel, copy=True) for kernel in obj.kernels),
        tuple(np.array(kernel, copy=True) for kernel in obj.weighted_kernels),
    )

    with pytest.raises(ValueError, match="Nyquist"):
        obj._configure_time_grid(0.25)

    assert (obj.dt, obj.filter_time, obj.run_time) == original[:3]
    for actual, expected in zip(obj.kernels, original[3]):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(obj.weighted_kernels, original[4]):
        np.testing.assert_array_equal(actual, expected)


def test_multi_tda_validates_band_nyquist_against_runtime_time_step():
    simulation = _FakeSimulation(0.25, 8)
    obj = _make_minimal_multi_tda(
        sim_factory=lambda sources=None: simulation,
        dt=None,
        resolution=10,
    )

    with pytest.raises(ValueError, match="Nyquist"):
        obj.evaluate(np.array([0.0]), need_gradient=False)

    assert "run" not in simulation.events


def test_multi_tda_objective_requires_design_only_for_gradients(monkeypatch):
    monkeypatch.setattr(
        multi_tda_module,
        "FastPointMonitor",
        _NativePathPointMonitor,
    )
    parameters = inspect.signature(tm.MultiTDAObjective).parameters
    assert "coords_x" not in parameters
    assert "coords_y" not in parameters
    assert "cell_area" not in parameters
    assert "material_factor" not in parameters

    update_calls = []
    obj = _make_minimal_multi_tda(
        update_design=lambda x: update_calls.append(np.array(x, copy=True)),
        sim_factory=lambda sources=None: _FakeSimulation(0.05, 48),
    )
    value, gradient = obj.evaluate(np.array([0.25]), need_gradient=False)
    assert np.isfinite(value)
    assert gradient is None
    assert len(update_calls) == 1

    with np.testing.assert_raises_regex(ValueError, "requires design=DesignGrid"):
        obj.evaluate(np.array([0.25]), need_gradient=True)
    assert len(update_calls) == 1


class _FakeSimulation:
    def __init__(self, dt, samples, actual_time=None):
        self.dt = dt
        self.fields = _FakeFields(dt)
        self.samples = samples
        self.actual_time = actual_time
        self.index = 0
        self.events = []
        self.changed_sources = []

    def run(self, callback, until):
        self.events.append("run")
        for index in range(self.samples):
            self.index = index
            callback(self)

    def get_field_point(self, component, position):
        return np.exp(1j * 0.1 * self.index)

    def round_time(self):
        if self.actual_time is not None:
            return self.actual_time
        return self.dt * (self.samples - 1)

    def restart_fields(self):
        self.events.append("restart_fields")
        self.index = 0

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


class _FluxFakeSimulation(_FakeSimulation):
    _component_scales = {
        mp.Ey: 2.0,
        mp.Hz: 3.0,
        mp.Ez: 5.0,
        mp.Hy: 7.0,
    }

    def get_field_point(self, component, position):
        return self._component_scales[component] * np.exp(
            1j * 0.1 * self.index
        )


class _CylindricalFluxFakeSimulation(_FakeSimulation):
    _component_scales = {
        mp.Er: 2.0,
        mp.Hp: 3.0,
        mp.Ep: 5.0,
        mp.Hr: 7.0,
        mp.Hz: 11.0,
        mp.Ez: 13.0,
    }

    def __init__(self, dt, samples, actual_time=None):
        super().__init__(dt, samples, actual_time=actual_time)
        self.dimensions = mp.CYLINDRICAL
        self.is_cylindrical = True
        self.m = 0

    def get_field_point(self, component, position):
        return self._component_scales[component] * np.exp(
            1j * 0.1 * self.index
        )


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

    def resolve_sources(self, sources=None):
        return () if sources is None else sources


def test_multi_tda_objective_simulation_reuse_is_opt_in():
    simulation_instances = []

    def make_simulation(sources=None):
        simulation_instance = _FakeSimulation(0.05, 8)
        simulation_instances.append(simulation_instance)
        return simulation_instance

    simulation = _FakeSimulationSpec(make_simulation)
    default_obj = _make_native_multi_tda(
        sim_factory=None,
        simulation=simulation,
    )
    reused_obj = _make_native_multi_tda(
        sim_factory=None,
        simulation=simulation,
        reuse_simulation=True,
    )

    assert default_obj.reuse_simulation is False
    assert default_obj._reuse_simulation_for_adjoint is False
    assert reused_obj.reuse_simulation is True
    assert reused_obj._reuse_simulation_for_adjoint is True

    with _fake_native_path():
        default_obj.evaluate(np.array([0.0]), need_gradient=True)

    assert len(simulation.make_calls) == 2
    assert simulation.make_calls[0] is None
    assert len(simulation.make_calls[1]) == 1
    assert simulation_instances[0].events == ["run", "reset_meep"]
    assert simulation_instances[1].events == ["run", "reset_meep"]

    try:
        _make_minimal_multi_tda(reuse_simulation=True)
    except ValueError as exc:
        assert "SimulationSpec.make" in str(exc)
    else:
        raise AssertionError("expected SimulationSpec requirement")

    for invalid_reuse in (1, "false"):
        try:
            _make_minimal_multi_tda(reuse_simulation=invalid_reuse)
        except ValueError as exc:
            assert "boolean" in str(exc)
        else:
            raise AssertionError("expected boolean requirement")


class _FakeAdaptiveBalancer:
    def __init__(self, source_boundary_decision=None, source_boundary_mode="auto"):
        self.prepared = []
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
        adjoint_source_positions,
        adjoint_source_size,
        adjoint_source_amplitude,
        indexed_source_mask=None,
        forward_sources=(),
    ):
        self.prepared.append(simulation_spec)
        self.prepared_indexed_masks.append(indexed_source_mask)
        self.prepared_sources.append(
            (
                tuple(adjoint_source_positions),
                adjoint_source_size,
                adjoint_source_amplitude,
            )
        )
        if self.source_boundary_decision is not None:
            return self.source_boundary_decision
        return type(
            "SourceBoundaryDecision",
            (),
            {
                "source_sizes": (adjoint_source_size,)
                * len(adjoint_source_positions),
                "source_amplitudes": (adjoint_source_amplitude,)
                * len(adjoint_source_positions),
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


class _OffsetSampleSimulation(_FakeSimulation):
    def __init__(self, dt, samples, sample_start, actual_time):
        super().__init__(dt, samples, actual_time=actual_time)
        self.sample_start = sample_start

    def meep_time(self):
        return self.sample_start + self.index * self.dt


class _FakeNativeHistorySet:
    instances = []

    def __init__(self, design, components, history_dtype, make_history_memmap):
        self.design = design
        self.components = tuple(components)
        self.history_dtype = history_dtype
        self.make_history_memmap = make_history_memmap
        self.states = {}
        self.sample_calls = 0
        self.cleaned_histories = {}
        self.__class__.instances.append(self)

    def sample_forward(self, sim, row_index, row_capacity):
        self.sample_calls += 1
        width = int(np.prod(self.design.shape))
        sample_values = getattr(sim, "forward_history_values", None)
        sample_value = (
            sim.index + 1.0
            if sample_values is None
            else sample_values[sim.index]
        )
        for component in self.components:
            if component not in self.states:
                history, path = self.make_history_memmap(
                    (row_capacity, max(width, 1))
                )
                self.states[component] = {
                    "array": history,
                    "path": path,
                    "width": width,
                    "signature": np.array([component, width], dtype=np.int64),
                }
            self.states[component]["array"][row_index, :width] = sample_value

    def finish_forward(self, row_count):
        histories = {}
        for component, state in self.states.items():
            state["array"].flush()
            histories[component] = state["array"][:row_count, :state["width"]]
        return histories

    def width(self, component):
        return self.states[component]["width"]

    @property
    def signatures(self):
        return {
            component: state["signature"]
            for component, state in self.states.items()
        }

    def release_forward(self):
        pass

    def cleanup_memmaps(self):
        for component, state in self.states.items():
            history = state["array"]
            if history is not None:
                self.cleaned_histories[component] = np.array(
                    history[:self.sample_calls, :state["width"]],
                    copy=True,
                )
                history.flush()
                state["array"] = None
            path = state["path"]
            if path is not None and os.path.exists(path):
                os.unlink(path)
                state["path"] = None


class _FakeNativeAccumulator:
    instances = []

    def __init__(self, design, components, signatures, midpoint):
        self.design = design
        self.components = tuple(components)
        self.signatures = signatures
        self.midpoint = midpoint
        self.local = np.zeros(design.shape, dtype=np.complex128)
        self.accumulated = []
        self.sample_steps = []
        self.__class__.instances.append(self)

    def accumulate(self, sim, component, forward_derivative):
        values = np.asarray(forward_derivative).reshape(self.design.shape)
        self.local += values
        self.accumulated.append((component, np.array(values, copy=True)))
        self.sample_steps.append(sim.index)

    def reduce(self):
        return self.local.copy()

    def release(self):
        pass


class _NativePathPointMonitor:
    def __init__(self, sim, component, positions):
        self.sim = sim
        self.component = component
        self.positions = positions

    def sample_history_into(self, destination):
        destination[:] = [
            self.sim.get_field_point(self.component, position)
            for position in self.positions
        ]

    @staticmethod
    def reduce_history(history):
        return history

    def indexed_transpose_stencil(self):
        return _fake_indexed_transpose_stencil(self)


@contextmanager
def _fake_native_path():
    original_history = multi_tda_module._NativeDesignHistorySet
    original_accumulator = multi_tda_module._NativeDesignAccumulator
    original_point_monitor = multi_tda_module.FastPointMonitor
    original_adjoint_loop = multi_tda_module._run_native_adjoint_loop
    _FakeNativeHistorySet.instances = []
    _FakeNativeAccumulator.instances = []
    try:
        multi_tda_module._NativeDesignHistorySet = _FakeNativeHistorySet
        multi_tda_module._NativeDesignAccumulator = _FakeNativeAccumulator
        multi_tda_module.FastPointMonitor = _NativePathPointMonitor
        multi_tda_module._run_native_adjoint_loop = _run_fake_adjoint_loop
        yield
    finally:
        multi_tda_module._NativeDesignHistorySet = original_history
        multi_tda_module._NativeDesignAccumulator = original_accumulator
        multi_tda_module.FastPointMonitor = original_point_monitor
        multi_tda_module._run_native_adjoint_loop = original_adjoint_loop


class _RaisingSimulation:
    def __init__(self):
        self.index = 0

    def run(self, callback, until):
        callback(self)
        raise RuntimeError("simulation failed")

    def get_field_point(self, component, position):
        return 1.0

    def reset_meep(self):
        pass


def test_multi_tda_value_only_scalarization_skips_autograd(monkeypatch):
    monkeypatch.setattr(
        multi_tda_module,
        "FastPointMonitor",
        _NativePathPointMonitor,
    )
    dt = 0.05

    def scalarization_fn(band_objectives):
        return float(2.0 * np.sum(band_objectives))

    obj = tm.MultiTDAObjective(
        update_design=lambda _: None,
        sim_factory=lambda sources=None: _FakeSimulation(dt, 48),
        t_final=1.0,
        monitor_positions=[mp.Vector3()],
        component=mp.Ez,
        wavelength_bands=[(0.4, 0.5)],
        weights=[1.0],
        kernel_length=9,
        pixel_chunk=1,
        adjoint_source_size=mp.Vector3(),
        adjoint_source_amplitude=1.0,
        dt=dt,
        scalarization_fn=scalarization_fn,
    )

    value, gradient = obj.evaluate(np.array([0.0]), need_gradient=False)

    assert gradient is None
    assert np.isfinite(value)
    assert obj.last_band_objectives.shape == (1,)
    assert obj.last_band_losses.shape == (1,)
    assert obj.last_band_coeffs is None
    assert obj.last_total_fom == 2.0 * obj.last_band_objectives[0]
    assert obj.last_total_fom is not None


def test_multi_tda_value_only_uses_custom_band_fom_without_adjoint_callback(
    monkeypatch,
):
    monkeypatch.setattr(
        multi_tda_module,
        "FastPointMonitor",
        _NativePathPointMonitor,
    )
    dt = 0.05
    filtered = np.array(
        [
            [1.0 + 0.5j, 2.0 - 0.25j],
            [0.5 - 0.2j, -1.0 + 0.75j],
        ],
        dtype=np.complex128,
    )
    adjoint_calls = []

    def fom_fn(history, sample_dt):
        return 0.25 * np.sum(np.abs(history) ** 4) * sample_dt

    def adjoint_signal_fn(history, sample_dt):
        adjoint_calls.append(history)
        return np.zeros_like(history)

    obj = tm.MultiTDAObjective(
        update_design=lambda _: None,
        sim_factory=lambda sources=None: _FakeSimulation(dt, 48),
        t_final=1.0,
        monitor_positions=[mp.Vector3(), mp.Vector3(0.1)],
        component=mp.Ez,
        wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
        weights=[1.0, 1.0],
        kernel_length=9,
        pixel_chunk=1,
        adjoint_source_size=mp.Vector3(),
        adjoint_source_amplitude=1.0,
        dt=dt,
        fom_fn=fom_fn,
        adjoint_signal_fn=adjoint_signal_fn,
    )
    obj._distributed_filter_monitor_signals = lambda _: filtered

    value, gradient = obj.evaluate(np.array([0.0]), need_gradient=False)

    expected = np.array([
        fom_fn(filtered[:, band_index], dt)
        for band_index in range(filtered.shape[1])
    ])
    assert gradient is None
    assert np.allclose(obj.last_band_objectives, expected)
    assert value == np.sum(expected)
    assert adjoint_calls == []

    obj.adjoint_signal_fn = lambda history, sample_dt: history[:-1]
    with pytest.raises(ValueError, match="adjoint_signal_fn"):
        obj._band_fom_values_and_adjoint_signals(
            filtered,
            need_gradient=True,
        )


def test_multi_tda_objective_reduces_point_monitor_history_once_after_forward():
    class _DeferredPointMonitor:
        instances = []

        def __init__(self, sim, component, positions):
            self.sim = sim
            self.component = component
            self.positions = positions
            self.sample_calls = 0
            self.reduce_calls = 0
            self.reduced_shape = None
            self.__class__.instances.append(self)

        def sample_history_into(self, destination):
            self.sample_calls += 1
            destination[:] = [
                self.sim.get_field_point(self.component, position)
                for position in self.positions
            ]

        def reduce_history(self, history):
            self.reduce_calls += 1
            self.reduced_shape = history.shape
            return history

        def indexed_transpose_stencil(self):
            return _fake_indexed_transpose_stencil(self)

    dt = 0.05
    obj = _make_minimal_multi_tda(
        sim_factory=lambda sources=None: _FakeSimulation(dt, 8),
        t_final=0.2,
        kernel_length=3,
        dt=dt,
    )

    original_point_monitor = multi_tda_module.FastPointMonitor
    try:
        multi_tda_module.FastPointMonitor = _DeferredPointMonitor
        value, gradient = obj.evaluate(np.array([0.0]), need_gradient=False)
    finally:
        multi_tda_module.FastPointMonitor = original_point_monitor

    monitor = _DeferredPointMonitor.instances[-1]
    assert np.isfinite(value)
    assert gradient is None
    assert monitor.sample_calls == 8
    assert monitor.reduce_calls == 1
    assert monitor.reduced_shape == (8, 1)


def test_multi_tda_objective_reuses_simulation_spec_for_adjoint_run():
    dt = 0.05
    simulation_instance = _FakeSimulation(dt, 8)
    simulation = _FakeSimulationSpec(lambda sources=None: simulation_instance)
    obj = tm.MultiTDAObjective(
        update_design=lambda _: None,
        design=_make_design(),
        simulation=simulation,
        t_final=0.2,
        monitor_positions=[mp.Vector3()],
        component=mp.Ez,
        wavelength_bands=[(0.4, 0.5)],
        weights=[1.0],
        kernel_length=3,
        pixel_chunk=1,
        dt=dt,
        reuse_simulation=True,
    )

    with _fake_native_path():
        obj.evaluate(np.array([0.0]), need_gradient=True)

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


def test_multi_tda_uses_indexed_source_only_for_zero_size_targets():
    dt = 0.05
    simulation_instance = _FakeSimulation(dt, 8)
    simulation = _FakeSimulationSpec(lambda sources=None: simulation_instance)
    obj = tm.MultiTDAObjective(
        update_design=lambda _: None,
        design=_make_design(),
        simulation=simulation,
        targets=[
            tm.PointTarget(mp.Vector3(-0.1), mp.Ez),
            tm.PointTarget(
                mp.Vector3(0.1),
                mp.Ez,
                adjoint_source_size=mp.Vector3(0.1),
            ),
        ],
        t_final=0.2,
        wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
        weights=[1.0, 1.0],
        kernel_length=3,
        pixel_chunk=1,
        dt=dt,
        reuse_simulation=True,
    )

    with _fake_native_path():
        obj.evaluate(np.array([0.0]), need_gradient=True)

    sources = simulation_instance.changed_sources[0]
    assert len(sources) == 2
    assert isinstance(sources[0], mp.IndexedSource)
    assert isinstance(sources[1], mp.Source)
    assert not isinstance(sources[1], mp.IndexedSource)


def test_multi_tda_cylindrical_separate_adjoint_uses_negative_forward_mode():
    dt = 0.05
    simulation_instances = []

    def make_simulation(sources=None):
        simulation = _FakeSimulation(dt, 8)
        simulation_instances.append(simulation)
        return simulation

    simulation = _FakeSimulationSpec(
        make_simulation,
        dimensions=mp.CYLINDRICAL,
        m=-1,
    )
    obj = _make_native_multi_tda(
        design=_make_cylindrical_design(),
        sim_factory=None,
        simulation=simulation,
        component=mp.Hp,
        t_final=0.2,
        kernel_length=3,
        dt=dt,
    )

    with _fake_native_path():
        obj.evaluate(np.array([0.0]), need_gradient=True)

    assert obj.gradient_components == (mp.Er, mp.Ep, mp.Ez)
    assert simulation.make_calls[0] is None
    assert simulation.make_calls[1][1] == 1
    assert simulation_instances[0].m == -1
    assert simulation_instances[1].m == 1
    assert _FakeNativeAccumulator.instances[-1].midpoint is True


def test_multi_tda_cylindrical_builds_indexed_source_for_each_target():
    dt = 0.05
    amplitude = 1.0
    simulation = _FakeSimulationSpec(
        lambda sources=None: _FakeSimulation(dt, 8),
        dimensions=mp.CYLINDRICAL,
        m=1,
    )
    obj = _make_native_multi_tda(
        design=_make_cylindrical_design(),
        sim_factory=None,
        simulation=simulation,
        monitor_positions=[mp.Vector3(0.7), mp.Vector3(0.9)],
        component=mp.Er,
        wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
        weights=[1.0, 1.0],
        t_final=0.2,
        kernel_length=3,
        adjoint_source_amplitude=amplitude,
        dt=dt,
    )

    with _fake_native_path():
        obj.evaluate(np.array([0.0]), need_gradient=True)

    adjoint_sources, adjoint_mode = simulation.make_calls[1]
    assert adjoint_mode == -1
    assert len(adjoint_sources) == 2
    assert all(
        isinstance(source, mp.IndexedSource)
        for source in adjoint_sources
    )
    assert all(
        np.array_equal(source.amp_arr, np.array([amplitude]))
        for source in adjoint_sources
    )
    assert (
        adjoint_sources[0].srcdata
        is not adjoint_sources[1].srcdata
    )


def test_multi_tda_cylindrical_maps_grouped_stencils_to_target_order(
    monkeypatch,
):
    class _TaggedPointMonitor:
        def __init__(self, sim, component, positions):
            self.component = component
            self.positions = tuple(positions)

        def sample_history_into(self, destination):
            destination[:] = 1.0

        @staticmethod
        def reduce_history(history):
            return history

        def indexed_transpose_stencil(self):
            return (
                np.arange(len(self.positions) + 1, dtype=np.intp),
                tuple(
                    (self.component, local_index)
                    for local_index in range(len(self.positions))
                ),
                np.arange(1, len(self.positions) + 1, dtype=float),
            )

    targets = [
        tm.PointTarget(mp.Vector3(0.7), mp.Er, adjoint_source_amplitude=1.0),
        tm.PointTarget(mp.Vector3(0.8), mp.Ep, adjoint_source_amplitude=1.0),
        tm.PointTarget(mp.Vector3(0.9), mp.Er, adjoint_source_amplitude=1.0),
    ]
    simulation = _FakeSimulationSpec(
        lambda sources=None: _FakeSimulation(0.05, 8),
        dimensions=mp.CYLINDRICAL,
        m=1,
    )
    obj = tm.MultiTDAObjective(
        design=_make_cylindrical_design(),
        simulation=simulation,
        targets=targets,
        t_final=0.2,
        wavelength_bands=[(0.4, 0.5), (0.5, 0.6), (0.6, 0.7)],
        weights=[1.0, 1.0, 1.0],
        kernel_length=3,
        pixel_chunk=1,
        dt=0.05,
    )
    source_calls = []

    def capture_source(
        time_source,
        component,
        center,
        size,
        amplitude,
        indexed_stencil,
    ):
        source_calls.append((component, amplitude, indexed_stencil))
        return [object()]

    monkeypatch.setattr(
        multi_tda_module,
        "_adjoint_point_sources",
        capture_source,
    )
    monkeypatch.setattr(
        multi_tda_module,
        "_NativeDesignHistorySet",
        _FakeNativeHistorySet,
    )
    monkeypatch.setattr(
        multi_tda_module,
        "_NativeDesignAccumulator",
        _FakeNativeAccumulator,
    )
    monkeypatch.setattr(
        multi_tda_module,
        "_run_native_adjoint_loop",
        _run_fake_adjoint_loop,
    )
    monkeypatch.setattr(
        multi_tda_module,
        "FastPointMonitor",
        _TaggedPointMonitor,
    )

    obj.evaluate(np.array([0.0]), need_gradient=True)

    assert [
        (component, amplitude, stencil[0], stencil[1].tolist())
        for component, amplitude, stencil in source_calls
    ] == [
        (mp.Er, 1.0, ((mp.Er, 0),), [1.0]),
        (mp.Ep, 1.0, ((mp.Ep, 0),), [1.0]),
        (mp.Er, 1.0, ((mp.Er, 1),), [2.0]),
    ]


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
def test_multi_tda_cylindrical_gradient_components_follow_mode_polarization(
    forward_mode,
    component,
    m0_components,
):
    simulation = _FakeSimulationSpec(
        lambda sources=None: _FakeSimulation(0.05, 8),
        dimensions=mp.CYLINDRICAL,
        m=forward_mode,
    )

    obj = _make_native_multi_tda(
        design=_make_cylindrical_design(),
        sim_factory=None,
        simulation=simulation,
        component=component,
        t_final=0.2,
        kernel_length=3,
        dt=0.05,
    )

    expected = (
        m0_components
        if forward_mode == 0
        else (mp.Er, mp.Ep, mp.Ez)
    )
    assert obj.gradient_components == expected


def test_multi_tda_cylindrical_reuse_flips_mode():
    dt = 0.05
    simulation_instance = _FakeSimulation(dt, 8)
    simulation = _FakeSimulationSpec(
        lambda sources=None: simulation_instance,
        dimensions=mp.CYLINDRICAL,
        m=1,
    )
    obj = _make_native_multi_tda(
        design=_make_cylindrical_design(),
        sim_factory=None,
        simulation=simulation,
        component=mp.Hr,
        t_final=0.2,
        kernel_length=3,
        dt=dt,
        reuse_simulation=True,
    )

    with _fake_native_path():
        obj.evaluate(np.array([0.0]), need_gradient=True)

    assert simulation.make_calls == [None]
    assert simulation_instance.events == [
        "run",
        ("change_m", -1),
        "restart_fields",
        "clear_dft_monitors",
        "change_sources",
        "run",
        "reset_meep",
    ]
    assert simulation_instance.m == -1


def test_multi_tda_rejects_custom_cylindrical_gradient_factory():
    dt = 0.05

    def make_simulation(sources=None):
        simulation = _FakeSimulation(dt, 8)
        simulation.dimensions = mp.CYLINDRICAL
        simulation.is_cylindrical = True
        simulation.m = 1
        return simulation

    obj = _make_native_multi_tda(
        sim_factory=make_simulation,
        t_final=0.2,
        kernel_length=3,
        dt=dt,
    )

    with np.testing.assert_raises_regex(
        ValueError,
        "requires simulation=SimulationSpec",
    ):
        obj.evaluate(np.array([0.0]), need_gradient=True)


def test_multi_tda_rejects_cylindrical_spec_with_cartesian_design():
    simulation = _FakeSimulationSpec(
        lambda sources=None: _FakeSimulation(0.05, 8),
        dimensions=mp.CYLINDRICAL,
        m=0,
    )

    with np.testing.assert_raises_regex(
        ValueError,
        "coordinate system must match",
    ):
        _make_native_multi_tda(
            sim_factory=None,
            simulation=simulation,
        )


def test_multi_tda_accepts_cylindrical_adaptive_chunk_balancing():
    simulation = _FakeSimulationSpec(
        lambda sources=None: _FakeSimulation(0.05, 8),
        dimensions=mp.CYLINDRICAL,
        m=0,
    )
    balancer = _FakeAdaptiveBalancer()

    objective = _make_native_multi_tda(
        design=_make_cylindrical_design(),
        sim_factory=None,
        simulation=simulation,
        chunk_balancer=balancer,
    )

    assert objective.chunk_balancer is balancer


def test_multi_tda_objective_updates_time_grid_from_meep_fields_dt(monkeypatch):
    monkeypatch.setattr(
        multi_tda_module,
        "FastPointMonitor",
        _NativePathPointMonitor,
    )
    dt = 0.03125
    obj = tm.MultiTDAObjective(
        update_design=lambda _: None,
        sim_factory=lambda sources=None: _FakeSimulation(dt, 16),
        t_final=0.25,
        monitor_positions=[mp.Vector3()],
        component=mp.Ez,
        wavelength_bands=[(0.4, 0.5)],
        weights=[1.0],
        kernel_length=5,
        pixel_chunk=1,
        adjoint_source_size=mp.Vector3(),
        adjoint_source_amplitude=1.0,
    )

    obj.evaluate(np.array([0.0]), need_gradient=False)

    assert obj.dt == dt
    assert obj.filter_time == 5 * dt
    assert obj.run_time == obj.t_final


def test_multi_tda_objective_falls_back_to_simulation_resolution():
    obj = _make_minimal_multi_tda(dt=None)
    sim = type("ResolutionOnlySimulation", (), {"resolution": 8})()

    assert obj.time_step(sim) == 0.0625


def test_multi_tda_objective_uses_courant_for_resolution_fallback():
    obj = _make_minimal_multi_tda(dt=None)
    sim = type(
        "ResolutionOnlySimulation",
        (),
        {"resolution": 8, "Courant": 0.25},
    )()

    assert obj.time_step(sim) == 0.03125


def test_multi_tda_objective_validates_explicit_dt_against_meep_fields_dt():
    obj = _make_minimal_multi_tda(dt=0.03125)

    assert obj.time_step(_FakeSimulation(0.03125, 1)) == 0.03125
    with pytest.raises(ValueError, match="does not match Meep fields.dt"):
        obj.time_step(_FakeSimulation(0.04, 1))


def test_multi_tda_objective_initializes_time_grid_from_simulation_courant():
    simulation = tm.SimulationSpec(
        cell_size=mp.Vector3(1.0, 1.0),
        resolution=10,
        courant=0.3,
    )
    obj = _make_minimal_multi_tda(
        dt=None,
        resolution=None,
        simulation=simulation,
    )

    assert obj.dt == pytest.approx(0.03)


def test_filter_monitor_signals_requires_resolved_time_grid():
    obj = _make_minimal_multi_tda(dt=None)

    with np.testing.assert_raises_regex(
        RuntimeError,
        "requires dt, resolution, or a prior evaluation",
    ):
        obj.filter_monitor_signals(np.zeros((4, 1), dtype=np.complex128))


@pytest.mark.parametrize("shape", ((4,), (2, 4, 1)))
def test_filter_monitor_signals_requires_two_dimensional_input(shape):
    obj = _make_minimal_multi_tda()

    with pytest.raises(ValueError, match=r"signals must have shape"):
        obj.filter_monitor_signals(np.zeros(shape, dtype=np.complex128))


@pytest.mark.parametrize("channel_count", (0, 2))
def test_filter_monitor_signals_requires_expected_channel_width(channel_count):
    obj = _make_minimal_multi_tda()

    with pytest.raises(ValueError, match=r"signals.*channel"):
        obj.filter_monitor_signals(
            np.zeros((4, channel_count), dtype=np.complex128)
        )


def test_multi_tda_objective_adjoint_source_uses_reversed_forward_sample_times():
    dt = 0.05
    actual_time = 0.37
    sample_start = 0.07
    seen = {}
    class _FakeInterp:
        def __call__(self, _):
            return 0.0

    def fake_tabulated_sources(times, values):
        seen.setdefault("times", []).append(np.array(times, copy=True))
        seen.setdefault("values", []).append(np.array(values, copy=True))
        return tuple(_FakeInterp() for _ in range(np.asarray(values).shape[1]))

    obj = tm.MultiTDAObjective(
        update_design=lambda _: None,
        design=_make_design(),
        sim_factory=lambda sources=None: _OffsetSampleSimulation(dt, 4, sample_start, actual_time),
        t_final=sample_start + 3 * dt,
        monitor_positions=[mp.Vector3()],
        component=mp.Ez,
        wavelength_bands=[(0.4, 0.5)],
        weights=[1.0],
        kernel_length=3,
        pixel_chunk=1,
        adjoint_source_size=mp.Vector3(),
        adjoint_source_amplitude=1.0,
        dt=dt,
    )

    original_tabulated_sources = multi_tda_module._tabulated_cubic_sources
    try:
        multi_tda_module._tabulated_cubic_sources = fake_tabulated_sources
        with _fake_native_path():
            obj.evaluate(np.array([0.0]), need_gradient=True)
    finally:
        multi_tda_module._tabulated_cubic_sources = original_tabulated_sources

    assert len(seen["times"]) == 1
    expected_forward_times = sample_start + dt * np.arange(4)
    assert np.allclose(seen["times"][0], actual_time - expected_forward_times[::-1])


def test_multi_tda_objective_stores_real_meep_histories_as_float64():
    class _RealSampleSimulation(_FakeSimulation):
        def __init__(self):
            super().__init__(0.05, 8)
            self.fields.is_real = True

        def get_field_point(self, component, position):
            return float(self.index + 1)

    obj = _make_native_multi_tda(
        sim_factory=lambda sources=None: _RealSampleSimulation(),
        dt=0.05,
    )

    with _fake_native_path():
        obj.evaluate(np.array([0.0]), need_gradient=True)

    assert _FakeNativeHistorySet.instances[0].history_dtype == np.dtype(
        np.float64
    )


@pytest.mark.parametrize("regional_target", (False, True))
def test_multi_tda_gradient_rejects_nonunit_logical_amplitude_before_forward(
    regional_target,
):
    simulations = []

    class _FieldRepresentationSimulation(_FakeSimulation):
        def __init__(self, fields_are_real):
            super().__init__(0.05, 8)
            self.fields.is_real = fields_are_real

        def get_field_point(self, component, position):
            value = float(self.index + 1)
            return value if self.fields.is_real else complex(value)

    def factory(sources=None):
        simulation = _FieldRepresentationSimulation(
            fields_are_real=sources is not None,
        )
        simulations.append(simulation)
        return simulation

    target = (
        tm.FieldRegionTarget(
            (mp.Vector3(-0.2), mp.Vector3(0.2)),
            mp.Ez,
            adjoint_source_amplitude=1.0j,
        )
        if regional_target
        else tm.PointTarget(
            mp.Vector3(),
            mp.Ez,
            adjoint_source_amplitude=1.0j,
        )
    )
    simulation = _FakeSimulationSpec(factory)
    obj = _make_native_multi_tda(
        sim_factory=None,
        simulation=simulation,
        targets=[target],
        monitor_positions=None,
        component=None,
        adjoint_source_size=None,
        adjoint_source_amplitude=None,
        dt=0.05,
    )

    with _fake_native_path():
        with pytest.raises(
            ValueError,
            match="requires adjoint_source_amplitude=1",
        ):
            obj.evaluate(np.array([0.0]), need_gradient=True)

    assert simulations == []


def test_multi_tda_nonunit_adjoint_amplitude_allows_value_only_only():
    class _FieldRepresentationSimulation(_FakeSimulation):
        def __init__(self, fields_are_real):
            super().__init__(0.05, 8)
            self.fields.is_real = fields_are_real

        def get_field_point(self, component, position):
            value = float(self.index + 1)
            return value if self.fields.is_real else complex(value)

    real_value_only = _make_native_multi_tda(
        sim_factory=lambda sources=None: _FieldRepresentationSimulation(True),
        adjoint_source_amplitude=1.0j,
        dt=0.05,
    )
    with _fake_native_path():
        value, gradient = real_value_only.evaluate(
            np.array([0.0]),
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

    complex_gradient = _make_native_multi_tda(
        sim_factory=real_forward_complex_adjoint_factory,
        adjoint_source_amplitude=1.0j,
        dt=0.05,
    )
    with _fake_native_path():
        with pytest.raises(
            ValueError,
            match="requires adjoint_source_amplitude=1",
        ):
            complex_gradient.evaluate(
                np.array([0.0]),
                need_gradient=True,
            )
    assert simulations == []


def test_multi_tda_rejects_simulation_mode_mutation_before_gradient_run():
    simulation = _FakeSimulationSpec(
        lambda sources=None: pytest.fail("simulation must not be constructed"),
        dimensions=mp.CYLINDRICAL,
        m=1,
    )
    obj = _make_native_multi_tda(
        design=_make_cylindrical_design(),
        sim_factory=None,
        simulation=simulation,
        monitor_positions=[mp.Vector3(0.5)],
        component=mp.Er,
        wavelength_bands=[(0.4, 0.5)],
        weights=[1.0],
        dt=0.05,
    )
    simulation.m = 0

    with pytest.raises(ValueError, match="must not change"):
        obj.evaluate(np.array([0.0]), need_gradient=True)


@pytest.mark.parametrize(
    "position",
    (mp.Vector3(-0.1, 0.0, 0.0), mp.Vector3(0.1, 0.2, 0.0)),
)
def test_multi_tda_rejects_invalid_cylindrical_point_coordinates(position):
    simulation = _FakeSimulationSpec(
        lambda sources=None: _FakeSimulation(0.05, 8),
        dimensions=mp.CYLINDRICAL,
        m=0,
    )
    obj = _make_native_multi_tda(
        design=_make_cylindrical_design(),
        sim_factory=None,
        simulation=simulation,
        monitor_positions=[position],
        component=mp.Ez,
        wavelength_bands=[(0.4, 0.5)],
        weights=[1.0],
        dt=0.05,
    )

    with pytest.raises(ValueError, match="cylindrical target positions"):
        obj.evaluate(np.array([0.0]), need_gradient=False)


def test_multi_tda_objective_cleans_history_memmaps_when_forward_run_raises():
    obj = _make_native_multi_tda(
        sim_factory=lambda sources=None: _RaisingSimulation(),
    )
    history_paths = []

    with tempfile.TemporaryDirectory() as tmpdir:
        def make_history_memmap(shape, *, dtype=None):
            fd, path = tempfile.mkstemp(
                prefix="tama_test_history_",
                suffix=".dat",
                dir=tmpdir,
            )
            os.close(fd)
            history_paths.append(path)
            return np.memmap(
                path,
                dtype=obj.history_dtype if dtype is None else dtype,
                mode="w+",
                shape=shape,
            ), path

        obj._make_history_memmap = make_history_memmap
        try:
            with _fake_native_path():
                obj.evaluate(np.array([0.0]), need_gradient=True)
        except RuntimeError as exc:
            assert "simulation failed" in str(exc)
        else:
            raise AssertionError("expected simulation failure")

        assert history_paths
        assert not any(os.path.exists(path) for path in history_paths)


def test_multi_tda_objective_cleans_reused_simulation_when_adjoint_run_raises():
    class _FailingAdjointSimulation(_FakeSimulation):
        def run(self, callback, until):
            if self.events.count("run") == 1:
                self.events.append("run")
                self.index = 0
                callback(self)
                raise RuntimeError("adjoint simulation failed")
            super().run(callback, until)

    dt = 0.05
    simulation_instance = _FailingAdjointSimulation(dt, 8)
    simulation = _FakeSimulationSpec(lambda sources=None: simulation_instance)
    obj = tm.MultiTDAObjective(
        update_design=lambda _: None,
        design=_make_design(),
        simulation=simulation,
        t_final=0.2,
        monitor_positions=[mp.Vector3()],
        component=mp.Ez,
        wavelength_bands=[(0.4, 0.5)],
        weights=[1.0],
        kernel_length=3,
        pixel_chunk=1,
        dt=dt,
        reuse_simulation=True,
    )
    history_paths = []

    with tempfile.TemporaryDirectory() as tmpdir:
        def make_history_memmap(shape, *, dtype=None):
            fd, path = tempfile.mkstemp(
                prefix="tama_test_history_",
                suffix=".dat",
                dir=tmpdir,
            )
            os.close(fd)
            history_paths.append(path)
            return np.memmap(
                path,
                dtype=obj.history_dtype if dtype is None else dtype,
                mode="w+",
                shape=shape,
            ), path

        obj._make_history_memmap = make_history_memmap
        try:
            with _fake_native_path():
                obj.evaluate(np.array([0.0]), need_gradient=True)
        except RuntimeError as exc:
            assert "adjoint simulation failed" in str(exc)
        else:
            raise AssertionError("expected adjoint simulation failure")

        assert history_paths
        assert not any(os.path.exists(path) for path in history_paths)

    assert simulation_instance.events == [
        "run",
        "restart_fields",
        "clear_dft_monitors",
        "change_sources",
        "run",
        "reset_meep",
    ]


def test_multi_tda_objective_sampling_interval_subsamples_design_grid_only():
    dt = 0.05

    def evaluate(
        *,
        sampling_interval=1,
        max_frequency=None,
        samples=9,
        reconstruction_window="kaiser",
        reconstruction_half_width=64,
    ):
        obj = _make_native_multi_tda(
            sim_factory=lambda sources=None: _FakeSimulation(dt, samples),
            dt=dt,
            sampling_interval=sampling_interval,
            max_frequency=max_frequency,
            reconstruction_window=reconstruction_window,
            reconstruction_half_width=reconstruction_half_width,
        )
        assert obj.reconstruction_window == reconstruction_window
        assert obj.reconstruction_half_width == reconstruction_half_width
        monitor_sample_counts = []
        original_filter = obj.filter_monitor_signals

        def capture_monitor_samples(signals):
            monitor_sample_counts.append(signals.shape[0])
            return original_filter(signals)

        obj._distributed_filter_monitor_signals = capture_monitor_samples
        value, gradient = obj.evaluate(np.array([0.0]), need_gradient=True)
        history = _FakeNativeHistorySet.instances[-1]
        accumulator = _FakeNativeAccumulator.instances[-1]
        return (
            value,
            gradient,
            history.sample_calls,
            len(accumulator.accumulated),
            monitor_sample_counts[0],
            accumulator.sample_steps,
            obj.last_sampling_interval,
        )

    with _fake_native_path():
        full_rate = evaluate()
        half_rate = evaluate(
            sampling_interval=2,
            reconstruction_window="blackman",
            reconstruction_half_width=8,
        )
        nyquist_rate = evaluate(max_frequency=4.0)
        aligned_terminal = evaluate(sampling_interval=2, samples=7)

    assert full_rate[2:] == (9, 9, 9, list(range(9)), 1)
    assert half_rate[2:] == (5, 9, 9, list(range(9)), 2)
    assert nyquist_rate[2:] == half_rate[2:]
    assert aligned_terminal[2:] == (4, 7, 7, list(range(7)), 2)
    assert np.allclose(half_rate[0], full_rate[0])
    assert np.allclose(nyquist_rate[0], full_rate[0])


def test_multi_tda_objective_uses_automatic_scalarization_coefficients():
    dt = 0.05
    seen = {"adjoint_values": []}

    class _FakeInterp:
        def __call__(self, _):
            return 0.0

    def fake_tabulated_sources(times, values):
        seen["adjoint_values"].append(np.array(values, copy=True))
        return tuple(_FakeInterp() for _ in range(np.asarray(values).shape[1]))

    def scalarization_fn(band_objectives):
        return (
            2.0 * band_objectives[0]
            + 1.5 * band_objectives[1] ** 2
        )

    obj = tm.MultiTDAObjective(
        update_design=lambda _: None,
        design=_make_design(),
        sim_factory=lambda sources=None: _FakeSimulation(dt, 8),
        t_final=7 * dt,
        monitor_positions=[mp.Vector3(), mp.Vector3(0.1)],
        component=mp.Ez,
        wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
        weights=[1.0, 1.0],
        kernel_length=3,
        pixel_chunk=1,
        adjoint_source_size=mp.Vector3(),
        adjoint_source_amplitude=1.0,
        dt=dt,
        scalarization_fn=scalarization_fn,
    )
    obj.weighted_kernels = [
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
    ]
    filtered = np.column_stack([
        np.linspace(1.0, 2.0, 10) + 1j * np.linspace(0.2, 0.9, 10),
        np.linspace(3.0, 4.0, 10) - 1j * np.linspace(0.4, 1.1, 10),
    ]).astype(np.complex128)
    obj._distributed_filter_monitor_signals = lambda _: filtered

    original_tabulated_sources = multi_tda_module._tabulated_cubic_sources
    try:
        multi_tda_module._tabulated_cubic_sources = fake_tabulated_sources
        with _fake_native_path():
            obj.evaluate(np.array([0.0]), need_gradient=True)
    finally:
        multi_tda_module._tabulated_cubic_sources = original_tabulated_sources

    assert len(seen["adjoint_values"]) == 1
    expected_coefficients = (
        2.0,
        3.0 * obj.last_band_objectives[1],
    )
    assert np.allclose(obj.last_band_coeffs, expected_coefficients)
    for band_index, coefficient in enumerate(expected_coefficients):
        expected = np.conjugate(
            multi_tda_module.temporal_convolve_signal_transpose(
                filtered[:, band_index] * coefficient,
                obj.weighted_kernels[band_index],
                dt,
            )
        )[::-1]
        assert np.allclose(seen["adjoint_values"][0][:, band_index], expected)


def test_multi_tda_objective_uses_custom_per_band_adjoint_signals():
    dt = 0.05
    seen = {"adjoint_values": []}

    class _FakeInterp:
        def __call__(self, _):
            return 0.0

    def fake_tabulated_sources(times, values):
        seen["adjoint_values"].append(np.array(values, copy=True))
        return tuple(_FakeInterp() for _ in range(np.asarray(values).shape[1]))

    def fom_fn(history, sample_dt):
        return 0.25 * np.sum(np.abs(history) ** 4) * sample_dt

    def adjoint_signal_fn(history, sample_dt):
        return np.abs(history) ** 2 * np.conjugate(history)

    def scalarization_fn(band_objectives):
        return float(np.sum(band_objectives)), np.array([2.0, 3.0])

    obj = tm.MultiTDAObjective(
        update_design=lambda _: None,
        design=_make_design(),
        sim_factory=lambda sources=None: _FakeSimulation(dt, 8),
        t_final=7 * dt,
        monitor_positions=[mp.Vector3(), mp.Vector3(0.1)],
        component=mp.Ez,
        wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
        weights=[1.0, 1.0],
        kernel_length=3,
        pixel_chunk=1,
        adjoint_source_size=mp.Vector3(),
        adjoint_source_amplitude=1.0,
        dt=dt,
        fom_fn=fom_fn,
        adjoint_signal_fn=adjoint_signal_fn,
        scalarization_fn=scalarization_fn,
    )
    obj.weighted_kernels = [
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
    ]
    filtered = np.column_stack([
        np.linspace(1.0, 2.0, 10) + 1j * np.linspace(0.2, 0.9, 10),
        np.linspace(3.0, 4.0, 10) - 1j * np.linspace(0.4, 1.1, 10),
    ]).astype(np.complex128)
    obj._distributed_filter_monitor_signals = lambda _: filtered

    original_tabulated_sources = multi_tda_module._tabulated_cubic_sources
    try:
        multi_tda_module._tabulated_cubic_sources = fake_tabulated_sources
        with _fake_native_path():
            obj.evaluate(np.array([0.0]), need_gradient=True)
    finally:
        multi_tda_module._tabulated_cubic_sources = original_tabulated_sources

    expected_objectives = np.array([
        fom_fn(filtered[:, band_index], dt)
        for band_index in range(filtered.shape[1])
    ])
    assert np.allclose(obj.last_band_objectives, expected_objectives)
    assert len(seen["adjoint_values"]) == 1
    for band_index, coefficient in enumerate((2.0, 3.0)):
        filtered_signal = adjoint_signal_fn(
            filtered[:, band_index],
            dt,
        )
        expected = np.conjugate(
            multi_tda_module.temporal_convolve_signal_transpose(
                np.conjugate(filtered_signal * coefficient),
                obj.weighted_kernels[band_index],
                dt,
            )
        )[::-1]
        assert np.allclose(seen["adjoint_values"][0][:, band_index], expected)


def test_multi_tda_objective_dense_derivative_preserves_forward_history():
    dt = 0.05
    samples = 8
    steps = np.arange(1, samples + 1, dtype=float)
    history_values = (steps**2 + 1j * steps**3).astype(np.complex128)

    def make_simulation(sources=None):
        simulation = _FakeSimulation(dt, samples)
        simulation.forward_history_values = history_values
        return simulation

    obj = _make_native_multi_tda(
        sim_factory=make_simulation,
        dt=dt,
    )

    with _fake_native_path():
        obj.evaluate(np.array([0.0]), need_gradient=True)

    history = _FakeNativeHistorySet.instances[-1]
    accumulator = _FakeNativeAccumulator.instances[-1]
    expected_history = history_values[:, None]
    expected_derivative = np.empty_like(history_values)
    expected_derivative[0] = (history_values[1] - history_values[0]) / dt
    expected_derivative[-1] = (history_values[-1] - history_values[-2]) / dt
    expected_derivative[1:-1] = (
        history_values[2:] - history_values[:-2]
    ) / (2.0 * dt)
    expected_derivative = expected_derivative.reshape(
        samples,
        *obj.design.shape,
    )[::-1]
    accumulated = np.stack([values for _, values in accumulator.accumulated])

    assert np.array_equal(history.cleaned_histories[mp.Ez], expected_history)
    assert np.array_equal(accumulated, expected_derivative)


def test_multi_tda_objective_rejects_dense_history_with_one_sample():
    obj = _make_native_multi_tda(
        sim_factory=lambda sources=None: _FakeSimulation(0.05, 1),
        dt=0.05,
    )

    with _fake_native_path():
        with pytest.raises(RuntimeError, match="at least two field samples"):
            obj.evaluate(np.array([0.0]), need_gradient=True)


def test_multi_tda_objective_auto_pixel_chunk():
    assert tm.auto_pixel_chunk(
        20,
        nproc=1,
        target_chunks_per_rank=10,
        min_pixel_chunk=3,
        max_pixel_chunk=7,
    ) == 3

    obj = tm.MultiTDAObjective(
        update_design=lambda _: None,
        design=_make_design((10, 2)),
        sim_factory=lambda sources=None: None,
        t_final=1.0,
        monitor_positions=[mp.Vector3()],
        component=mp.Ez,
        wavelength_bands=[(0.4, 0.5)],
        weights=[1.0],
        kernel_length=9,
        pixel_chunk="auto",
        target_chunks_per_rank=10,
        min_pixel_chunk=3,
        max_pixel_chunk=7,
        adjoint_source_size=mp.Vector3(),
        adjoint_source_amplitude=1.0,
        dt=0.05,
    )

    assert obj.pixel_chunk == 3


def test_multi_tda_objective_accepts_bundled_design_simulation_and_targets():
    air = mp.Medium(epsilon=1.0)
    dielectric = mp.Medium(epsilon=4.0)
    design = tm.DesignGrid(
        material_grid=mp.MaterialGrid(mp.Vector3(2, 2), air, dielectric),
        center=mp.Vector3(),
        size=mp.Vector3(2.0, 2.0),
        shape=(2, 2),
        background=air,
        design_material=dielectric,
    )
    simulation = tm.SimulationSpec(
        cell_size=mp.Vector3(1.0, 1.0),
        boundary_layers=[],
        geometry=[],
        sources=[],
        resolution=10,
    )
    amplitude = 0.6 - 0.8j
    targets = [
        tm.PointTarget(
            position=mp.Vector3(0.0, 0.1),
            component=mp.Ez,
            adjoint_source_amplitude=amplitude,
        ),
        tm.PointTarget(
            position=mp.Vector3(0.0, 0.2),
            component=mp.Ez,
            adjoint_source_amplitude=amplitude,
        ),
    ]

    obj = tm.MultiTDAObjective(
        design=design,
        simulation=simulation,
        targets=targets,
        t_final=1.0,
        wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
        weights=[1.0, 1.0],
        kernel_length=9,
        pixel_chunk=1,
        dt=0.05,
    )

    assert obj.design is design
    assert obj.monitor_positions == [target.position for target in targets]
    assert obj.component == mp.Ez
    assert obj.gradient_components == (mp.Ez,)
    assert obj.adjoint_source_amplitude == amplitude


def test_multi_tda_objective_accepts_heterogeneous_target_settings():
    targets = [
        tm.PointTarget(
            position=mp.Vector3(-0.2, 0.1),
            component=mp.Ex,
            adjoint_source_size=mp.Vector3(0.1, 0.0),
            adjoint_source_amplitude=0.5 + 0.25j,
        ),
        tm.PointTarget(
            position=mp.Vector3(0.3, -0.1),
            component=mp.Ey,
            adjoint_source_size=mp.Vector3(0.0, 0.2),
            adjoint_source_amplitude=1.5 - 0.75j,
        ),
    ]

    obj = tm.MultiTDAObjective(
        update_design=lambda _: None,
        design=_make_design(),
        sim_factory=lambda sources=None: None,
        targets=targets,
        t_final=1.0,
        wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
        weights=[1.0, 1.0],
        kernel_length=9,
        pixel_chunk=1,
        dt=0.05,
    )

    assert obj.monitor_positions == [target.position for target in targets]
    assert obj.target_components == (mp.Ex, mp.Ey)
    assert obj.adjoint_source_sizes == (
        mp.Vector3(0.1, 0.0),
        mp.Vector3(0.0, 0.2),
    )
    assert obj.adjoint_source_amplitudes == (
        0.5 + 0.25j,
        1.5 - 0.75j,
    )
    assert obj.component is None
    assert obj.adjoint_source_size is None
    assert obj.adjoint_source_amplitude is None
    assert obj.gradient_components == (mp.Ex, mp.Ey)


def test_multi_tda_field_regions_preserve_each_band_history_shape_and_sources():
    dt = 0.05
    targets = [
        tm.PointTarget(
            mp.Vector3(-0.3, 0.1),
            mp.Ez,
            adjoint_source_amplitude=1.0,
        ),
        tm.FieldRegionTarget(
            (
                mp.Vector3(-0.1, 0.2),
                mp.Vector3(0.1, 0.2),
            ),
            mp.Ez,
            sample_shape=(1, 2),
            spatial_weights=(1.0, 2.0),
            adjoint_source_amplitude=1.0,
        ),
        tm.FieldRegionTarget(
            (
                mp.Vector3(-0.2, -0.2),
                mp.Vector3(0.0, -0.2),
                mp.Vector3(0.2, -0.2),
            ),
            mp.Ez,
            sample_shape=(3,),
            spatial_weights=(1.0, 1.5, 2.0),
            adjoint_source_amplitude=1.0,
        ),
    ]
    simulation = _FakeSimulationSpec(
        lambda sources=None: _FakeSimulation(dt, 8)
    )
    fom_shapes = []
    adjoint_shapes = []

    def fom_fn(history, sample_dt):
        fom_shapes.append(history.shape)
        return float(np.sum(np.abs(history) ** 2) * sample_dt)

    def adjoint_signal_fn(history, sample_dt):
        adjoint_shapes.append(history.shape)
        return np.ones_like(history, dtype=np.complex128)

    obj = tm.MultiTDAObjective(
        update_design=lambda _: None,
        design=_make_design(),
        simulation=simulation,
        targets=targets,
        t_final=7 * dt,
        wavelength_bands=[
            (0.4, 0.5),
            (0.5, 0.6),
            (0.6, 0.7),
        ],
        weights=[1.0, 1.0, 1.0],
        kernel_length=3,
        pixel_chunk=1,
        dt=dt,
        fom_fn=fom_fn,
        adjoint_signal_fn=adjoint_signal_fn,
    )

    with _fake_native_path():
        value, gradient = obj.evaluate(np.array([0.0]), need_gradient=True)

    expected_shapes = [(10,), (10, 1, 2), (10, 3)]
    assert fom_shapes == expected_shapes
    assert adjoint_shapes == expected_shapes
    assert np.isfinite(value)
    assert gradient.shape == (1,)

    sources = simulation.make_calls[1]
    assert len(sources) == 6
    assert all(isinstance(source, mp.IndexedSource) for source in sources)
    assert [
        complex(np.asarray(source.amp_arr).reshape(-1)[0])
        for source in sources
    ] == [1.0] * 6


def test_multi_tda_field_regions_deduplicate_matching_monitor_channels():
    positions = (
        mp.Vector3(-0.1),
        mp.Vector3(),
        mp.Vector3(0.1),
    )
    targets = (
        tm.FieldRegionTarget(
            positions[:2],
            mp.Ez,
            sample_shape=(1, 2),
            spatial_weights=(0.4, 0.6),
            adjoint_source_amplitude=1.0 + 0.5j,
        ),
        tm.FieldRegionTarget(
            positions[1:],
            mp.Ez,
            sample_shape=(2,),
            spatial_weights=(0.3, 0.7),
            adjoint_source_amplitude=1.0 + 0.5j,
        ),
    )
    obj = _make_minimal_multi_tda(
        monitor_positions=None,
        component=None,
        adjoint_source_size=None,
        adjoint_source_amplitude=None,
        targets=targets,
        wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
        weights=[1.0, 1.0],
    )

    assert len(obj._flat_target_positions) == 4
    assert len(obj._monitor_target_positions) == 3
    assert obj._uses_deduplicated_monitors
    assert np.array_equal(obj._target_monitor_indices[0], (0, 1))
    assert np.array_equal(obj._target_monitor_indices[1], (1, 2))


def test_multi_tda_field_regions_preserve_distinct_source_operators():
    position = mp.Vector3(0.1, -0.2)
    common = dict(
        positions=(position,),
        component=mp.Ez,
        adjoint_source_amplitude=1.0,
    )
    changed_amplitude = (
        tm.FieldRegionTarget(**common),
        tm.FieldRegionTarget(
            **{**common, "adjoint_source_amplitude": 2.0}
        ),
    )
    changed_component = (
        tm.FieldRegionTarget(**common),
        tm.FieldRegionTarget(
            (position,),
            mp.Ey,
            adjoint_source_amplitude=1.0,
        ),
    )

    for targets in (changed_amplitude, changed_component):
        obj = _make_minimal_multi_tda(
            monitor_positions=None,
            component=None,
            adjoint_source_size=None,
            adjoint_source_amplitude=None,
            targets=targets,
            wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
            weights=[1.0, 1.0],
        )

        assert len(obj._flat_target_positions) == 2
        assert len(obj._monitor_target_positions) == 2
        assert not obj._uses_deduplicated_monitors


def test_multi_tda_field_regions_sample_only_unique_monitor_channels(
    monkeypatch,
):
    class RecordingPointMonitor(_NativePathPointMonitor):
        instances = []

        def __init__(self, sim, component, positions):
            super().__init__(sim, component, positions)
            self.__class__.instances.append(self)

    monkeypatch.setattr(
        multi_tda_module,
        "FastPointMonitor",
        RecordingPointMonitor,
    )
    positions = (
        mp.Vector3(-0.1),
        mp.Vector3(),
        mp.Vector3(0.1),
    )
    targets = (
        tm.FieldRegionTarget(positions[:2], mp.Ez),
        tm.FieldRegionTarget(positions[1:], mp.Ez),
    )
    obj = _make_minimal_multi_tda(
        sim_factory=lambda sources=None: _FakeSimulation(0.05, 8),
        monitor_positions=None,
        component=None,
        adjoint_source_size=None,
        adjoint_source_amplitude=None,
        targets=targets,
        wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
        weights=[1.0, 1.0],
    )
    history_shapes = []
    evaluate_bands = obj._distributed_deduplicated_band_objectives

    def record_history_shape(history):
        history_shapes.append(history.shape)
        return evaluate_bands(history)

    monkeypatch.setattr(
        obj,
        "_distributed_deduplicated_band_objectives",
        record_history_shape,
    )

    value, gradient = obj.evaluate(np.array([0.0]), need_gradient=False)

    assert np.isfinite(value)
    assert gradient is None
    assert len(RecordingPointMonitor.instances) == 1
    assert len(RecordingPointMonitor.instances[0].positions) == 3
    assert history_shapes == [(8, 3)]


def test_multi_tda_deduplicated_regional_pullback_matches_logical_channels():
    rng = np.random.default_rng(20260803)
    positions = (
        mp.Vector3(-0.1),
        mp.Vector3(),
        mp.Vector3(0.1),
    )
    targets = (
        tm.FieldRegionTarget(
            positions[:2],
            mp.Ez,
            sample_shape=(1, 2),
            spatial_weights=(0.4, 0.6),
        ),
        tm.FieldRegionTarget(
            positions[1:],
            mp.Ez,
            sample_shape=(2,),
            spatial_weights=(0.3, 0.7),
        ),
    )

    def second_fom(history, dt):
        return dt * npa.sum(npa.abs(history) ** 4)

    obj = _make_minimal_multi_tda(
        monitor_positions=None,
        component=None,
        adjoint_source_size=None,
        adjoint_source_amplitude=None,
        targets=targets,
        wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
        weights=[1.0, 0.8],
        fom_fn=(None, second_fom),
        scalarization_fn=lambda values: values[0] + 0.3 * values[1] ** 2,
    )
    unique_history = (
        rng.standard_normal((23, len(obj._monitor_target_positions)))
        + 1j
        * rng.standard_normal((23, len(obj._monitor_target_positions)))
    )
    logical_history = unique_history[:, obj._flat_to_monitor_indices]
    filtered = obj.filter_monitor_signals(logical_history)
    expected_objectives, filtered_adjoint = (
        obj._band_fom_values_and_adjoint_signals(
            filtered,
            need_gradient=True,
        )
    )
    expected_total, band_coeffs, _ = obj._evaluate_scalarization(
        expected_objectives,
        need_gradient=True,
    )
    logical_adjoint = obj._filter_transpose_adjoint_signals(
        filtered_adjoint,
        band_coeffs,
    )
    expected_unique_adjoint = np.zeros_like(unique_history)
    for logical_index, monitor_index in enumerate(
        obj._flat_to_monitor_indices
    ):
        expected_unique_adjoint[:, monitor_index] += logical_adjoint[
            :,
            logical_index,
        ]

    actual_objectives = obj._distributed_deduplicated_band_objectives(
        unique_history
    )
    actual_total, actual_coeffs, _ = obj._evaluate_scalarization(
        actual_objectives,
        need_gradient=True,
    )
    actual_unique_adjoint = (
        obj._distributed_deduplicated_adjoint_signals(
            unique_history,
            actual_coeffs,
        )
    )

    assert np.allclose(actual_objectives, expected_objectives)
    assert np.allclose(actual_total, expected_total)
    assert np.allclose(actual_coeffs, band_coeffs)
    assert np.allclose(actual_unique_adjoint, expected_unique_adjoint)


def test_multi_tda_field_region_gradient_requires_simulation_spec():
    target = tm.FieldRegionTarget(
        (mp.Vector3(-0.1), mp.Vector3(0.1)),
        mp.Ez,
    )
    obj = tm.MultiTDAObjective(
        update_design=lambda _: None,
        design=_make_design(),
        sim_factory=lambda sources=None: _FakeSimulation(0.05, 8),
        targets=[target],
        t_final=0.2,
        wavelength_bands=[(0.4, 0.5)],
        weights=[1.0],
        kernel_length=3,
        pixel_chunk=1,
        dt=0.05,
    )

    with pytest.raises(ValueError, match="requires simulation=SimulationSpec"):
        obj.evaluate(np.array([0.0]), need_gradient=True)


def test_multi_tda_flux_target_filters_fields_before_power(monkeypatch):
    monkeypatch.setattr(
        multi_tda_module,
        "FastPointMonitor",
        _NativePathPointMonitor,
    )
    dt = 0.05
    target = tm.FluxTarget(
        (mp.Vector3(),),
        normal=mp.Vector3(1.0),
        spatial_weights=(1.5,),
    )
    callback_shapes = []

    def fom_fn(power, sample_dt):
        callback_shapes.append(power.shape)
        return np.sum(power) * sample_dt

    obj = tm.MultiTDAObjective(
        update_design=lambda _: None,
        sim_factory=lambda sources=None: _FluxFakeSimulation(dt, 8),
        targets=[target],
        t_final=7 * dt,
        wavelength_bands=[(0.4, 0.5)],
        weights=[1.25],
        kernel_length=3,
        pixel_chunk=1,
        dt=dt,
        fom_fn=fom_fn,
    )

    value, gradient = obj.evaluate(np.array([0.0]), need_gradient=False)

    phase = np.exp(1j * 0.1 * np.arange(8))
    raw = np.column_stack(
        [
            _FluxFakeSimulation._component_scales[component] * phase
            for component in obj._flat_target_components
        ]
    )
    filtered = obj.filter_monitor_signals(raw)
    expected_power = flux_module.power_history(
        obj._target_component_histories(filtered, 0),
        target.normal,
        target.spatial_weights,
    )
    assert callback_shapes == [(9,)]
    assert value == pytest.approx(np.sum(expected_power) * dt)
    assert gradient is None


def test_multi_tda_flux_targets_deduplicate_exact_monitor_channels():
    positions = (mp.Vector3(-0.1), mp.Vector3(0.1))
    target = tm.FluxTarget(
        positions,
        normal=mp.Vector3(1.0),
        spatial_weights=(0.4, 0.6),
    )
    obj = tm.MultiTDAObjective(
        update_design=lambda _: None,
        sim_factory=lambda sources=None: _FluxFakeSimulation(0.05, 8),
        targets=[target, target],
        t_final=0.2,
        wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
        weights=[1.0, 1.0],
        kernel_length=3,
        pixel_chunk=1,
        dt=0.05,
    )

    assert len(obj._flat_target_positions) == 16
    assert len(obj._monitor_target_positions) == 8
    assert np.array_equal(
        obj._target_monitor_indices[0],
        obj._target_monitor_indices[1],
    )


def test_multi_tda_flux_targets_sample_only_unique_overlapping_channels(
    monkeypatch,
):
    class RecordingPointMonitor(_NativePathPointMonitor):
        instances = []

        def __init__(self, sim, component, positions):
            super().__init__(sim, component, positions)
            self.__class__.instances.append(self)

    monkeypatch.setattr(
        multi_tda_module,
        "FastPointMonitor",
        RecordingPointMonitor,
    )
    positions = (
        mp.Vector3(-0.1),
        mp.Vector3(),
        mp.Vector3(0.1),
    )
    targets = [
        tm.FluxTarget(
            positions[:2],
            normal=mp.Vector3(1.0),
        ),
        tm.FluxTarget(
            positions[1:],
            normal=mp.Vector3(1.0),
        ),
    ]
    obj = tm.MultiTDAObjective(
        update_design=lambda _: None,
        sim_factory=lambda sources=None: _FluxFakeSimulation(0.05, 8),
        targets=targets,
        t_final=0.2,
        wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
        weights=[1.0, 1.0],
        kernel_length=3,
        pixel_chunk=1,
        dt=0.05,
    )
    history_shapes = []
    evaluate_bands = obj._distributed_deduplicated_band_objectives

    def record_history_shape(history):
        history_shapes.append(history.shape)
        return evaluate_bands(history)

    monkeypatch.setattr(
        obj,
        "_distributed_deduplicated_band_objectives",
        record_history_shape,
    )

    obj.evaluate(np.array([0.0]), need_gradient=False)

    assert len(obj._flat_target_positions) == 16
    assert len(obj._monitor_target_positions) == 12
    assert len(RecordingPointMonitor.instances) == 4
    assert {
        len(monitor.positions)
        for monitor in RecordingPointMonitor.instances
    } == {3}
    assert len(history_shapes) == 1
    assert history_shapes[0][1] == 12


def test_multi_tda_deduplicated_flux_pullback_matches_logical_channels():
    rng = np.random.default_rng(17)
    positions = (mp.Vector3(-0.1), mp.Vector3(0.1))
    target = tm.FluxTarget(
        positions,
        normal=mp.Vector3(1.0),
        spatial_weights=(0.4, 0.6),
    )
    obj = tm.MultiTDAObjective(
        update_design=lambda _: None,
        sim_factory=lambda sources=None: _FluxFakeSimulation(0.05, 8),
        targets=[target, target],
        t_final=0.2,
        wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
        weights=[1.0, 0.8],
        kernel_length=3,
        pixel_chunk=1,
        dt=0.05,
    )
    unique_history = (
        rng.standard_normal((23, len(obj._monitor_target_positions)))
        + 1j
        * rng.standard_normal((23, len(obj._monitor_target_positions)))
    )
    logical_history = unique_history[:, obj._flat_to_monitor_indices]
    filtered = obj.filter_monitor_signals(logical_history)
    expected_objectives, filtered_adjoint = (
        obj._band_fom_values_and_adjoint_signals(
            filtered,
            need_gradient=True,
        )
    )
    band_coeffs = np.array([0.7, -0.3])
    logical_adjoint = obj._filter_transpose_adjoint_signals(
        filtered_adjoint,
        band_coeffs,
    )
    expected_unique_adjoint = np.zeros_like(unique_history)
    for logical_index, monitor_index in enumerate(
        obj._flat_to_monitor_indices
    ):
        expected_unique_adjoint[:, monitor_index] += logical_adjoint[
            :,
            logical_index,
        ]

    actual_objectives = obj._distributed_deduplicated_band_objectives(
        unique_history
    )
    actual_unique_adjoint = (
        obj._distributed_deduplicated_adjoint_signals(
            unique_history,
            band_coeffs,
        )
    )

    assert np.allclose(actual_objectives, expected_objectives)
    assert np.allclose(actual_unique_adjoint, expected_unique_adjoint)


def test_multi_tda_deduplicated_second_pass_skips_custom_fom():
    calls = []
    target = tm.FluxTarget(
        (mp.Vector3(),),
        normal=mp.Vector3(1.0),
    )

    def fom_fn(power, dt):
        calls.append("fom")
        return np.sum(power) * dt

    def adjoint_signal_fn(power, dt):
        calls.append("adjoint")
        return np.ones_like(power)

    obj = tm.MultiTDAObjective(
        update_design=lambda _: None,
        sim_factory=lambda sources=None: _FluxFakeSimulation(0.05, 8),
        targets=[target, target],
        t_final=0.2,
        wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
        weights=[1.0, 0.8],
        kernel_length=3,
        pixel_chunk=1,
        dt=0.05,
        fom_fn=fom_fn,
        adjoint_signal_fn=adjoint_signal_fn,
    )
    history = np.ones(
        (8, len(obj._monitor_target_positions)),
        dtype=np.complex128,
    )

    obj._distributed_deduplicated_band_objectives(history)
    obj._distributed_deduplicated_adjoint_signals(
        history,
        np.ones(2),
    )

    assert calls == ["fom", "fom", "adjoint", "adjoint"]


def test_multi_tda_overlapping_flux_targets_sum_shared_adjoint_sources():
    class KeyedPointMonitor(_NativePathPointMonitor):
        def indexed_transpose_stencil(self):
            return (
                np.arange(len(self.positions) + 1, dtype=np.intp),
                tuple(
                    (
                        int(self.component),
                        float(position.x),
                        float(position.y),
                        float(position.z),
                    )
                    for position in self.positions
                ),
                np.ones(len(self.positions), dtype=float),
            )

    dt = 0.05
    positions = (
        mp.Vector3(-0.1),
        mp.Vector3(),
        mp.Vector3(0.1),
    )
    targets = (
        tm.FluxTarget(
            positions[:2],
            normal=mp.Vector3(1.0),
            spatial_weights=(0.4, 0.6),
        ),
        tm.FluxTarget(
            positions[1:],
            normal=mp.Vector3(1.0),
            spatial_weights=(0.3, 0.7),
        ),
    )
    bands = ((0.4, 0.5), (0.5, 0.6))
    weights = (1.0, 0.8)

    def make_objective(simulation, selected_targets, selected_bands, selected_weights):
        return tm.MultiTDAObjective(
            design=_make_design(),
            simulation=simulation,
            targets=selected_targets,
            t_final=0.2,
            wavelength_bands=selected_bands,
            weights=selected_weights,
            kernel_length=3,
            pixel_chunk=1,
            dt=dt,
        )

    combined_simulation = _FakeSimulationSpec(
        lambda sources=None: _FluxFakeSimulation(dt, 8)
    )
    separate_simulations = [
        _FakeSimulationSpec(
            lambda sources=None: _FluxFakeSimulation(dt, 8)
        )
        for _ in targets
    ]
    combined = make_objective(
        combined_simulation,
        targets,
        bands,
        weights,
    )
    separate = [
        make_objective(
            simulation,
            (target,),
            (band,),
            (weight,),
        )
        for simulation, target, band, weight in zip(
            separate_simulations,
            targets,
            bands,
            weights,
        )
    ]

    with _fake_native_path():
        multi_tda_module.FastPointMonitor = KeyedPointMonitor
        combined.evaluate(np.array([0.0]), need_gradient=True)
        for objective in separate:
            objective.evaluate(np.array([0.0]), need_gradient=True)

    assert [
        len(sources)
        for sources in combined_simulation.make_calls[1:]
    ] == [18]
    times = np.array([0.15, 0.2, 0.25])

    def source_signals(simulation):
        signals = {}
        for sources in simulation.make_calls[1:]:
            for source in sources:
                key = tuple(source.srcdata)
                values = source.amp_arr[0] * np.asarray(
                    [source.src.src_func(time) for time in times]
                )
                signals[key] = signals.get(key, 0.0) + values
        return signals

    actual = source_signals(combined_simulation)
    expected = {}
    for simulation in separate_simulations:
        for key, values in source_signals(simulation).items():
            expected[key] = expected.get(key, 0.0) + values

    assert actual.keys() == expected.keys()
    for key in actual:
        assert np.allclose(
            actual[key],
            expected[key],
            rtol=1.0e-12,
            atol=1.0e-12,
        )


def test_multi_tda_flux_target_combines_electric_and_magnetic_adjoint_sources():
    dt = 0.05
    simulation = _FakeSimulationSpec(
        lambda sources=None: _FluxFakeSimulation(dt, 8)
    )
    callback_shapes = []

    def fom_fn(power, sample_dt):
        callback_shapes.append(power.shape)
        return np.sum(power) * sample_dt

    def adjoint_signal_fn(power, sample_dt):
        callback_shapes.append(power.shape)
        return np.ones_like(power)

    obj = tm.MultiTDAObjective(
        design=_make_design(),
        simulation=simulation,
        targets=[
            tm.FluxTarget(
                (mp.Vector3(),),
                normal=mp.Vector3(1.0),
            )
        ],
        t_final=7 * dt,
        wavelength_bands=[(0.4, 0.5)],
        weights=[1.0],
        kernel_length=3,
        pixel_chunk=1,
        dt=dt,
        fom_fn=fom_fn,
        adjoint_signal_fn=adjoint_signal_fn,
    )

    with _fake_native_path():
        _, gradient = obj.evaluate(np.array([0.0]), need_gradient=True)

    assert callback_shapes == [(9,), (9,)]
    assert obj.gradient_components == (mp.Ex, mp.Ey, mp.Ez)
    assert len(simulation.make_calls) == 2
    assert [len(sources) for sources in simulation.make_calls[1:]] == [6]
    assert all(
        isinstance(source, mp.IndexedSource)
        for sources in simulation.make_calls[1:]
        for source in sources
    )
    assert [
        accumulator.midpoint
        for accumulator in _FakeNativeAccumulator.instances
    ] == [False]
    assert gradient.shape == (1,)


def test_multi_tda_flux_target_reuses_simulation_for_combined_adjoint():
    dt = 0.05
    simulation_instance = _FluxFakeSimulation(dt, 8)
    simulation = _FakeSimulationSpec(
        lambda sources=None: simulation_instance
    )
    obj = tm.MultiTDAObjective(
        design=_make_design(),
        simulation=simulation,
        targets=[
            tm.FluxTarget(
                (mp.Vector3(),),
                normal=mp.Vector3(1.0),
            )
        ],
        t_final=0.2,
        wavelength_bands=[(0.4, 0.5)],
        weights=[1.0],
        kernel_length=3,
        pixel_chunk=1,
        dt=dt,
        reuse_simulation=True,
    )

    with _fake_native_path():
        obj.evaluate(np.array([0.0]), need_gradient=True)

    assert simulation.make_calls == [None]
    assert simulation_instance.events == [
        "run",
        "restart_fields",
        "clear_dft_monitors",
        "change_sources",
        "run",
        "reset_meep",
    ]


def test_multi_tda_flux_reuse_cleans_failed_adjoint_setup():
    class _FailingRestartSimulation(_FluxFakeSimulation):
        def restart_fields(self):
            self.events.append("restart_fields")
            raise RuntimeError("restart failed")

    dt = 0.05
    simulation_instance = _FailingRestartSimulation(dt, 8)
    simulation = _FakeSimulationSpec(
        lambda sources=None: simulation_instance
    )
    obj = tm.MultiTDAObjective(
        design=_make_design(),
        simulation=simulation,
        targets=[
            tm.FluxTarget(
                (mp.Vector3(),),
                normal=mp.Vector3(1.0),
            )
        ],
        t_final=0.2,
        wavelength_bands=[(0.4, 0.5)],
        weights=[1.0],
        kernel_length=3,
        pixel_chunk=1,
        dt=dt,
        reuse_simulation=True,
    )

    with _fake_native_path():
        with pytest.raises(RuntimeError, match="restart failed"):
            obj.evaluate(np.array([0.0]), need_gradient=True)

    assert simulation_instance.events == [
        "run",
        "restart_fields",
        "reset_meep",
    ]


def test_multi_tda_flux_target_rejects_unsupported_gradient_paths():
    target = tm.FluxTarget(
        (mp.Vector3(),),
        normal=mp.Vector3(1.0),
    )
    arbitrary_factory = tm.MultiTDAObjective(
        design=_make_design(),
        sim_factory=lambda sources=None: _FluxFakeSimulation(0.05, 8),
        targets=[target],
        t_final=0.2,
        wavelength_bands=[(0.4, 0.5)],
        weights=[1.0],
        kernel_length=3,
        pixel_chunk=1,
        dt=0.05,
    )
    with pytest.raises(
        ValueError,
        match="FluxTarget gradient evaluation requires",
    ):
        arbitrary_factory.evaluate(np.array([0.0]), need_gradient=True)

    simulation = _FakeSimulationSpec(
        lambda sources=None: _FluxFakeSimulation(0.05, 8)
    )
    with pytest.raises(ValueError, match="source_boundary_mode='finite'"):
        tm.MultiTDAObjective(
            design=_make_design(),
            simulation=simulation,
            targets=[target],
            t_final=0.2,
            wavelength_bands=[(0.4, 0.5)],
            weights=[1.0],
            kernel_length=3,
            pixel_chunk=1,
            dt=0.05,
            chunk_balancer=_FakeAdaptiveBalancer(
                source_boundary_mode="finite"
            ),
        )
    with pytest.raises(ValueError, match="x-y plane"):
        tm.MultiTDAObjective(
            design=_make_design(),
            simulation=simulation,
            targets=[
                tm.FluxTarget(
                    (mp.Vector3(),),
                    normal=mp.Vector3(0.0, 0.0, 1.0),
                )
            ],
            t_final=0.2,
            wavelength_bands=[(0.4, 0.5)],
            weights=[1.0],
            kernel_length=3,
            pixel_chunk=1,
            dt=0.05,
        )


def test_multi_tda_flux_value_rejects_out_of_plane_2d_normal_at_runtime():
    dt = 0.05
    simulation = _FluxFakeSimulation(dt, 8)
    simulation.cell_size = mp.Vector3(2.0, 2.0)
    obj = tm.MultiTDAObjective(
        update_design=lambda values: None,
        sim_factory=lambda sources=None: simulation,
        targets=[
            tm.FluxTarget(
                (mp.Vector3(),),
                normal=mp.Vector3(0.0, 0.0, 1.0),
            )
        ],
        t_final=0.2,
        wavelength_bands=[(0.4, 0.5)],
        weights=[1.0],
        kernel_length=3,
        pixel_chunk=1,
        dt=dt,
    )

    with pytest.raises(ValueError, match="x-y plane"):
        obj.evaluate(np.array([0.0]), need_gradient=False)


def test_multi_tda_cylindrical_flux_value_plain_factory_uses_runtime_channels(
    monkeypatch,
):
    monkeypatch.setattr(
        multi_tda_module,
        "FastPointMonitor",
        _NativePathPointMonitor,
    )
    dt = 0.05
    simulation = _CylindricalFluxFakeSimulation(dt, 8)
    target = tm.FluxTarget(
        (mp.Vector3(0.5),),
        normal=mp.Vector3(0.0, 0.0, 1.0),
        spatial_weights=(1.25,),
    )
    obj = tm.MultiTDAObjective(
        update_design=lambda values: None,
        sim_factory=lambda sources=None: simulation,
        targets=[target],
        t_final=7 * dt,
        wavelength_bands=[(0.4, 0.5)],
        weights=[1.0],
        kernel_length=3,
        pixel_chunk=1,
        dt=dt,
    )

    value, gradient = obj.evaluate(
        np.array([0.0]),
        need_gradient=False,
    )

    assert obj._flat_target_components == (
        mp.Er,
        mp.Hp,
        mp.Ep,
        mp.Hr,
    )
    phase = np.exp(1j * 0.1 * np.arange(8))
    raw = np.column_stack(
        [
            simulation._component_scales[component] * phase
            for component in obj._flat_target_components
        ]
    )
    filtered = obj.filter_monitor_signals(raw)
    expected_power = flux_module.power_history(
        obj._target_component_histories(filtered, 0),
        target.normal,
        target.spatial_weights,
        cylindrical=True,
    )
    assert value == pytest.approx(np.sum(expected_power) * dt)
    assert gradient is None


def test_multi_tda_cylindrical_eigenmode_value_plain_factory_uses_runtime_channels(
    monkeypatch,
):
    monkeypatch.setattr(
        multi_tda_module,
        "FastPointMonitor",
        _NativePathPointMonitor,
    )
    dt = 0.05
    simulation = _CylindricalFluxFakeSimulation(dt, 8)
    target = tm.EigenmodeCoefficientTarget(
        positions=(mp.Vector3(0.5),),
        normal=mp.Vector3(1.0),
        spatial_weights=(1.5,),
        reference_mode_fields={
            mp.Ep: (1.0,),
            mp.Hz: (1.0,),
            mp.Ez: (0.0,),
            mp.Hp: (0.0,),
        },
        reference_m=0,
    )
    obj = tm.MultiTDAObjective(
        update_design=lambda values: None,
        sim_factory=lambda sources=None: simulation,
        targets=[target],
        t_final=7 * dt,
        wavelength_bands=[(0.4, 0.5)],
        weights=[1.0],
        kernel_length=3,
        pixel_chunk=1,
        dt=dt,
    )

    value, gradient = obj.evaluate(
        np.array([0.0]),
        need_gradient=False,
    )

    assert obj._flat_target_components == (
        mp.Ep,
        mp.Hz,
        mp.Ez,
        mp.Hp,
    )
    phase = np.exp(1j * 0.1 * np.arange(8))
    raw = np.column_stack(
        [
            simulation._component_scales[component] * phase
            for component in obj._flat_target_components
        ]
    )
    filtered = obj.filter_monitor_signals(raw)
    expected_coefficients = eigenmode_module.coefficient_history(
        obj._target_component_histories(filtered, 0),
        obj._reference_mode_fields[0],
        target.normal,
        target.spatial_weights,
        cylindrical=True,
    )
    assert value == pytest.approx(
        np.sum(np.abs(expected_coefficients) ** 2) * dt
    )
    assert gradient is None


def test_multi_tda_plain_factory_rejects_surface_coordinate_change(
    monkeypatch,
):
    monkeypatch.setattr(
        multi_tda_module,
        "FastPointMonitor",
        _NativePathPointMonitor,
    )
    dt = 0.05
    cylindrical_simulation = _CylindricalFluxFakeSimulation(dt, 8)
    cartesian_simulation = _FluxFakeSimulation(dt, 8)
    simulations = iter(
        (cylindrical_simulation, cartesian_simulation)
    )
    obj = tm.MultiTDAObjective(
        update_design=lambda values: None,
        sim_factory=lambda sources=None: next(simulations),
        targets=[
            tm.FluxTarget(
                (mp.Vector3(0.5),),
                normal=mp.Vector3(1.0),
            )
        ],
        t_final=0.2,
        wavelength_bands=[(0.4, 0.5)],
        weights=[1.0],
        kernel_length=3,
        pixel_chunk=1,
        dt=dt,
    )

    value, gradient = obj.evaluate(
        np.array([0.0]),
        need_gradient=False,
    )
    with pytest.raises(
        ValueError,
        match="target coordinate system does not match",
    ):
        obj.evaluate(np.array([0.0]), need_gradient=False)

    assert np.isfinite(value)
    assert gradient is None
    assert cylindrical_simulation.events == ["run", "reset_meep"]
    assert cartesian_simulation.events == ["reset_meep"]


def test_multi_tda_explicit_cartesian_metadata_rejects_cylindrical_factory():
    dt = 0.05
    simulation = _CylindricalFluxFakeSimulation(dt, 8)
    simulation_metadata = _FakeSimulationSpec(
        lambda sources=None: _FluxFakeSimulation(dt, 8),
        dimensions=2,
    )
    obj = tm.MultiTDAObjective(
        update_design=lambda values: None,
        sim_factory=lambda sources=None: simulation,
        simulation=simulation_metadata,
        targets=[
            tm.FluxTarget(
                (mp.Vector3(0.5),),
                normal=mp.Vector3(1.0),
            )
        ],
        t_final=0.2,
        wavelength_bands=[(0.4, 0.5)],
        weights=[1.0],
        kernel_length=3,
        pixel_chunk=1,
        dt=dt,
    )

    with pytest.raises(
        ValueError,
        match="target coordinate system does not match",
    ):
        obj.evaluate(np.array([0.0]), need_gradient=False)

    assert simulation.events == ["reset_meep"]
    assert simulation_metadata.make_calls == []


@pytest.mark.parametrize(
    "legacy_override",
    [
        {"monitor_positions": [mp.Vector3(-0.2), mp.Vector3(0.2)]},
        {"component": mp.Ey},
        {"adjoint_source_size": mp.Vector3(0.1)},
        {"adjoint_source_amplitude": 2.0},
    ],
)
def test_multi_tda_field_region_rejects_legacy_target_overrides(
    legacy_override,
):
    targets = [
        tm.PointTarget(mp.Vector3(-0.2), mp.Ez),
        tm.FieldRegionTarget(
            (mp.Vector3(0.1), mp.Vector3(0.2)),
            mp.Ez,
        ),
    ]

    with pytest.raises(ValueError, match="legacy target arguments"):
        tm.MultiTDAObjective(
            update_design=lambda _: None,
            sim_factory=lambda sources=None: _FakeSimulation(0.05, 8),
            targets=targets,
            t_final=0.2,
            wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
            weights=[1.0, 1.0],
            kernel_length=3,
            pixel_chunk=1,
            dt=0.05,
            **legacy_override,
        )


@pytest.mark.parametrize(
    "monitor_positions",
    [
        [mp.Vector3()],
        [mp.Vector3(), mp.Vector3(), mp.Vector3()],
    ],
)
def test_multi_tda_point_targets_validate_monitor_position_override_count(
    monitor_positions,
):
    targets = [
        tm.PointTarget(mp.Vector3(-0.2), mp.Ez),
        tm.PointTarget(mp.Vector3(0.2), mp.Ez),
    ]

    with pytest.raises(
        ValueError,
        match="monitor_positions must match the number of targets",
    ):
        tm.MultiTDAObjective(
            update_design=lambda _: None,
            sim_factory=lambda sources=None: _FakeSimulation(0.05, 8),
            targets=targets,
            monitor_positions=monitor_positions,
            t_final=0.2,
            wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
            weights=[1.0, 1.0],
            kernel_length=3,
            pixel_chunk=1,
            dt=0.05,
        )


def test_multi_tda_objective_requires_one_target_per_band():
    with pytest.raises(ValueError, match="targets must match"):
        tm.MultiTDAObjective(
            update_design=lambda _: None,
            sim_factory=lambda sources=None: None,
            targets=[tm.PointTarget(mp.Vector3(), mp.Ez)],
            monitor_positions=[mp.Vector3(-0.1), mp.Vector3(0.1)],
            t_final=1.0,
            wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
            weights=[1.0, 1.0],
            kernel_length=9,
            pixel_chunk=1,
            dt=0.05,
        )


def test_multi_tda_explicit_source_settings_override_target_settings():
    source_size = mp.Vector3(0.2, 0.3)
    amplitude = 2.0 - 1.0j
    obj = tm.MultiTDAObjective(
        update_design=lambda _: None,
        sim_factory=lambda sources=None: None,
        targets=[
            tm.PointTarget(
                mp.Vector3(-0.1),
                mp.Ex,
                mp.Vector3(0.1, 0.0),
                0.5,
            ),
            tm.PointTarget(
                mp.Vector3(0.1),
                mp.Ey,
                mp.Vector3(0.0, 0.1),
                1.5,
            ),
        ],
        component=mp.Ez,
        adjoint_source_size=source_size,
        adjoint_source_amplitude=amplitude,
        t_final=1.0,
        wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
        weights=[1.0, 1.0],
        kernel_length=9,
        pixel_chunk=1,
        dt=0.05,
    )

    assert obj.target_components == (mp.Ez, mp.Ez)
    assert obj.adjoint_source_sizes == (source_size, source_size)
    assert obj.adjoint_source_amplitudes == (amplitude, amplitude)


def test_multi_tda_objective_groups_forward_monitors_by_component(monkeypatch):
    class _GroupedPointMonitor:
        instances = []

        def __init__(self, sim, component, positions):
            self.component = component
            self.positions = tuple(positions)
            self.sample_calls = 0
            self.reduce_calls = 0
            self.reduced_shape = None
            self.__class__.instances.append(self)

        def sample_history_into(self, destination):
            self.sample_calls += 1
            destination[:] = (
                10 * self.component + np.arange(len(self.positions))
            )

        def reduce_history(self, history):
            self.reduce_calls += 1
            self.reduced_shape = history.shape
            return history

        def indexed_transpose_stencil(self):
            return _fake_indexed_transpose_stencil(self)

    targets = [
        tm.PointTarget(mp.Vector3(-0.2), mp.Ex),
        tm.PointTarget(mp.Vector3(0.0), mp.Ey),
        tm.PointTarget(mp.Vector3(0.2), mp.Ex),
    ]
    obj = tm.MultiTDAObjective(
        update_design=lambda _: None,
        sim_factory=lambda sources=None: _FakeSimulation(0.05, 8),
        targets=targets,
        t_final=7 * 0.05,
        wavelength_bands=[(0.4, 0.5), (0.5, 0.6), (0.6, 0.7)],
        weights=[1.0, 1.0, 1.0],
        kernel_length=3,
        pixel_chunk=1,
        dt=0.05,
    )
    monkeypatch.setattr(
        multi_tda_module,
        "FastPointMonitor",
        _GroupedPointMonitor,
    )
    monitor_histories = []

    def capture_monitor_history(history):
        monitor_histories.append(np.array(history, copy=True))
        return np.zeros(
            (history.shape[0] + obj.kernel_length - 1, history.shape[1]),
            dtype=history.dtype,
        )

    obj._distributed_filter_monitor_signals = capture_monitor_history

    value, gradient = obj.evaluate(np.array([0.0]), need_gradient=False)

    assert np.isfinite(value)
    assert gradient is None
    assert [
        (monitor.component, monitor.positions)
        for monitor in _GroupedPointMonitor.instances
    ] == [
        (mp.Ex, (targets[0].position, targets[2].position)),
        (mp.Ey, (targets[1].position,)),
    ]
    assert all(monitor.sample_calls == 8 for monitor in _GroupedPointMonitor.instances)
    assert [
        monitor.reduce_calls for monitor in _GroupedPointMonitor.instances
    ] == [1, 0]
    assert _GroupedPointMonitor.instances[0].reduced_shape == (8, 3)
    assert np.array_equal(
        monitor_histories[0],
        np.tile(np.array([0.0, 10.0, 1.0]), (8, 1)),
    )


def test_multi_tda_objective_uses_each_target_for_adjoint_source(monkeypatch):
    class _HeterogeneousBalancer(_FakeAdaptiveBalancer):
        def prepare_initial_layout(
            self,
            simulation_spec,
            *,
            adjoint_source_positions,
            adjoint_source_sizes,
            adjoint_source_amplitudes,
            indexed_source_mask=None,
            forward_sources=(),
        ):
            self.prepared.append(simulation_spec)
            self.prepared_indexed_masks.append(indexed_source_mask)
            self.prepared_sources.append(
                (
                    tuple(adjoint_source_positions),
                    tuple(adjoint_source_sizes),
                    tuple(adjoint_source_amplitudes),
                )
            )
            return type(
                "SourceBoundaryDecision",
                (),
                {
                    "source_sizes": tuple(adjoint_source_sizes),
                    "source_amplitudes": tuple(adjoint_source_amplitudes),
                },
            )()

    targets = [
        tm.PointTarget(
            mp.Vector3(-0.2, 0.1),
            mp.Ex,
            mp.Vector3(),
            1.0,
        ),
        tm.PointTarget(
            mp.Vector3(0.3, -0.1),
            mp.Ey,
            mp.Vector3(0.0, 0.2),
            1.0,
        ),
    ]
    simulation_instance = _FakeSimulation(0.05, 8)
    simulation = _FakeSimulationSpec(lambda sources=None: simulation_instance)
    balancer = _HeterogeneousBalancer()
    obj = tm.MultiTDAObjective(
        update_design=lambda _: None,
        design=_make_design(),
        simulation=simulation,
        targets=targets,
        t_final=0.2,
        wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
        weights=[1.0, 1.0],
        kernel_length=3,
        pixel_chunk=1,
        dt=0.05,
        chunk_balancer=balancer,
        reuse_simulation=True,
    )
    source_calls = []

    def capture_source(
        time_source,
        component,
        center,
        size,
        amplitude,
        indexed_stencil,
    ):
        source_calls.append(
            (component, center, size, amplitude, indexed_stencil)
        )
        return [object()]

    monkeypatch.setattr(
        multi_tda_module,
        "_adjoint_point_sources",
        capture_source,
    )

    with _fake_native_path():
        obj.evaluate(np.array([0.0]), need_gradient=True)

    assert [
        (component, center, size, amplitude)
        for component, center, size, amplitude, _ in source_calls
    ] == [
        (
            target.component,
            target.position,
            target.adjoint_source_size,
            target.adjoint_source_amplitude,
        )
        for target in targets
    ]
    assert balancer.prepared_sources == [
        (
            tuple(target.position for target in targets),
            tuple(target.adjoint_source_size for target in targets),
            tuple(target.adjoint_source_amplitude for target in targets),
        )
    ]
    assert balancer.prepared_indexed_masks == [(True, False)]


def test_multi_tda_objective_rejects_mixed_electric_magnetic_gradient():
    obj = tm.MultiTDAObjective(
        update_design=lambda _: None,
        design=_make_design(),
        sim_factory=lambda sources=None: None,
        targets=[
            tm.PointTarget(mp.Vector3(-0.1), mp.Ex),
            tm.PointTarget(mp.Vector3(0.1), mp.Hz),
        ],
        t_final=0.2,
        wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
        weights=[1.0, 1.0],
        kernel_length=3,
        pixel_chunk=1,
        dt=0.05,
    )

    with pytest.raises(
        ValueError,
        match=(
            "electric and magnetic components across PointTarget and "
            "FieldRegionTarget"
        ),
    ):
        obj.evaluate(np.array([0.0]), need_gradient=True)


def test_multi_tda_flux_does_not_enable_mixed_non_flux_gradient():
    dt = 0.05
    obj = tm.MultiTDAObjective(
        design=_make_design(),
        simulation=_FakeSimulationSpec(
            lambda sources=None: _FluxFakeSimulation(dt, 8)
        ),
        targets=[
            tm.FluxTarget(
                (mp.Vector3(),),
                normal=mp.Vector3(1.0),
            ),
            tm.PointTarget(mp.Vector3(-0.1), mp.Ex),
            tm.PointTarget(mp.Vector3(0.1), mp.Hz),
        ],
        t_final=0.2,
        wavelength_bands=[
            (0.4, 0.5),
            (0.5, 0.6),
            (0.6, 0.7),
        ],
        weights=[1.0, 1.0, 1.0],
        kernel_length=3,
        pixel_chunk=1,
        dt=dt,
    )

    with pytest.raises(ValueError, match="electric and magnetic"):
        obj.evaluate(np.array([0.0]), need_gradient=True)


def test_multi_tda_objective_updates_simulation_spec_with_adaptive_layout():
    dt = 0.05
    simulation_instance = _FakeSimulation(dt, 8)
    simulation = _FakeSimulationSpec(
        lambda sources=None: simulation_instance
    )
    balancer = _FakeAdaptiveBalancer()
    obj = tm.MultiTDAObjective(
        update_design=lambda _: None,
        design=_make_design(),
        simulation=simulation,
        t_final=0.2,
        monitor_positions=[mp.Vector3()],
        component=mp.Ez,
        wavelength_bands=[(0.4, 0.5)],
        weights=[1.0],
        kernel_length=3,
        pixel_chunk=1,
        dt=dt,
        chunk_balancer=balancer,
        reuse_simulation=True,
    )

    with _fake_native_path():
        obj.evaluate(np.array([0.0]), need_gradient=True)

    assert simulation.chunk_layout == "next-layout"
    assert balancer.prepared == [simulation]
    assert len(balancer.captured) == 1
    assert len(balancer.rebalanced) == 1
    assert balancer.captured[0][0] is balancer.rebalanced[0][0]
    assert balancer.rebalanced[0][1] > 0.0
    assert balancer.rebalanced[0][2] > 0.0
    assert balancer.rebalanced[0][3] == (balancer.captured[0][1],)
    assert simulation_instance.fields.reset_timer_calls == 1
    assert obj.last_chunk_balance == balancer.last_report


def test_multi_tda_flux_target_updates_adaptive_layout_after_combined_adjoint():
    class _FluxBalancer(_FakeAdaptiveBalancer):
        def prepare_initial_layout(
            self,
            simulation_spec,
            *,
            adjoint_source_positions,
            adjoint_source_sizes,
            adjoint_source_amplitudes,
            indexed_source_mask=None,
            forward_sources=(),
        ):
            self.prepared.append(simulation_spec)
            self.prepared_indexed_masks.append(indexed_source_mask)
            return type(
                "SourceBoundaryDecision",
                (),
                {
                    "source_sizes": tuple(adjoint_source_sizes),
                    "source_amplitudes": tuple(
                        adjoint_source_amplitudes
                    ),
                },
            )()

    dt = 0.05
    simulation_instance = _FluxFakeSimulation(dt, 8)
    simulation = _FakeSimulationSpec(
        lambda sources=None: simulation_instance
    )
    balancer = _FluxBalancer()
    obj = tm.MultiTDAObjective(
        design=_make_design(),
        simulation=simulation,
        targets=[
            tm.FluxTarget(
                (mp.Vector3(),),
                normal=mp.Vector3(1.0),
            )
        ],
        t_final=0.2,
        wavelength_bands=[(0.4, 0.5)],
        weights=[1.0],
        kernel_length=3,
        pixel_chunk=1,
        dt=dt,
        chunk_balancer=balancer,
        reuse_simulation=True,
    )

    with _fake_native_path():
        obj.evaluate(np.array([0.0]), need_gradient=True)

    assert simulation.chunk_layout == "next-layout"
    assert balancer.prepared == [simulation]
    assert balancer.prepared_indexed_masks == [(True, True, True, True)]
    assert len(balancer.captured) == 1
    assert len(balancer.rebalanced) == 1
    assert balancer.rebalanced[0][3] == (balancer.captured[0][1],)
    assert simulation_instance.fields.reset_timer_calls == 1
    assert obj.last_chunk_balance == balancer.last_report


def test_multi_tda_objective_uses_per_target_effective_adjoint_sources():
    dt = 0.05
    simulation_instance = _FakeSimulation(dt, 8)
    simulation = _FakeSimulationSpec(
        lambda sources=None: simulation_instance
    )
    positions = [mp.Vector3(-0.1, 0.0), mp.Vector3(0.2, 0.1)]
    effective_sizes = (
        mp.Vector3(0.2, 0.0),
        mp.Vector3(0.0, 0.3),
    )
    base_amplitude = 1.0
    effective_amplitudes = (5.0 + 1.0j, 7.0 - 2.0j)
    decision = type(
        "SourceBoundaryDecision",
        (),
        {
            "source_sizes": effective_sizes,
            "source_amplitudes": effective_amplitudes,
        },
    )()
    balancer = _FakeAdaptiveBalancer(decision, source_boundary_mode="finite")
    obj = tm.MultiTDAObjective(
        update_design=lambda _: None,
        design=_make_design(),
        simulation=simulation,
        t_final=0.2,
        monitor_positions=positions,
        component=mp.Ez,
        wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
        weights=[1.0, 1.0],
        kernel_length=3,
        pixel_chunk=1,
        adjoint_source_size=mp.Vector3(),
        adjoint_source_amplitude=base_amplitude,
        dt=dt,
        chunk_balancer=balancer,
        reuse_simulation=True,
    )

    with _fake_native_path():
        obj.evaluate(np.array([0.0]), need_gradient=True)

    assert obj.last_source_boundary_decision is decision
    assert len(simulation_instance.changed_sources) == 1
    sources = simulation_instance.changed_sources[0]
    assert len(sources) == 2
    for source, position, source_size, source_amplitude in zip(
        sources,
        positions,
        effective_sizes,
        effective_amplitudes,
    ):
        assert source.center == position
        assert source.size == source_size
        assert source.amplitude == source_amplitude
    assert obj.adjoint_source_size == mp.Vector3()
    assert obj.adjoint_source_amplitude == base_amplitude
    assert balancer.prepared_sources == [
        (tuple(positions), mp.Vector3(), base_amplitude)
    ]


if __name__ == "__main__":
    test_multi_tda_objective_accepts_numpy_kernel_window_options()
    test_multi_tda_objective_exports_and_filters_signals()
    test_temporal_convolve_signal_transpose_satisfies_dot_product_identity()
    test_complex_band_adjoint_signal_matches_directional_finite_difference()
    test_native_tabulated_cubic_matches_legacy_interp1d()
    test_multi_tda_objective_has_no_gradient_mode_parameter()
    test_multi_tda_objective_validates_sampling_interval()
    test_multi_tda_objective_requires_design_only_for_gradients()
    test_multi_tda_value_only_scalarization_skips_autograd()
    test_multi_tda_objective_reduces_point_monitor_history_once_after_forward()
    test_multi_tda_objective_simulation_reuse_is_opt_in()
    test_multi_tda_objective_reuses_simulation_spec_for_adjoint_run()
    test_multi_tda_objective_updates_time_grid_from_meep_fields_dt()
    test_multi_tda_objective_adjoint_source_uses_reversed_forward_sample_times()
    test_multi_tda_objective_cleans_history_memmaps_when_forward_run_raises()
    test_multi_tda_objective_cleans_reused_simulation_when_adjoint_run_raises()
    test_multi_tda_objective_sampling_interval_subsamples_design_grid_only()
    test_multi_tda_objective_uses_automatic_scalarization_coefficients()
    test_multi_tda_objective_dense_derivative_preserves_forward_history()
    test_multi_tda_objective_rejects_dense_history_with_one_sample()
    test_multi_tda_objective_auto_pixel_chunk()
    test_multi_tda_objective_accepts_bundled_design_simulation_and_targets()
    test_multi_tda_objective_updates_simulation_spec_with_adaptive_layout()
