"""Full observation support and exact pullback, using synthetic histories only."""

import autograd.numpy as npa
import meep as mp
import numpy as np
import pytest
from scipy.signal import fftconvolve

import tama as tm
from tama.multi_tda_objective import (
    temporal_convolve_signal,
    temporal_convolve_signal_transpose,
)


@pytest.fixture(autouse=True)
def forbid_fdtd(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("These regression tests must not initialize or run FDTD")

    monkeypatch.setattr(mp.Simulation, "init_sim", forbidden)
    monkeypatch.setattr(mp.Simulation, "run", forbidden)


def make_objective(kind="region", length=5, *, duplicate=False, count=2, dt=0.075, **kwargs):
    targets = []
    for band in range(count):
        x = 0.0 if duplicate else 0.3 * band
        positions = (mp.Vector3(x, -0.1), mp.Vector3(x, 0.1))
        if kind == "point":
            target = tm.PointTarget(positions[0], mp.Ez)
        elif kind == "region":
            target = tm.FieldRegionTarget(
                positions, mp.Ez, sample_shape=(1, 2), spatial_weights=(0.4, 0.6)
            )
        elif kind == "flux":
            target = tm.FluxTarget(
                positions, normal=mp.Vector3(-1.0), spatial_weights=(0.4, 0.6)
            )
        else:
            target = tm.EigenmodeCoefficientTarget(
                positions, normal=mp.Vector3(1.0), spatial_weights=(0.4, 0.6),
                reference_mode_fields={
                    mp.Ey: (1.0 + 0.3j, 0.7 - 0.2j),
                    mp.Hz: (1.0 + 0.3j, 0.7 - 0.2j),
                    mp.Ez: (0.2j, 0.1), mp.Hy: (-0.2j, -0.1),
                },
            )
        targets.append(target)
    obj = tm.MultiTDAObjective(
        update_design=lambda _: None, sim_factory=lambda sources=None: None,
        targets=targets, t_final=0.9, dt=dt,
        wavelength_bands=[(0.4, 0.5), (0.5, 0.6)][:count], weights=[1.0, 0.8][:count],
        kernel_length=length, chunk_balancer=None, **kwargs,
    )
    # Non-symmetric, complex kernels test full support and conjugation.
    rng = np.random.default_rng(40 + length)
    obj.kernels = [rng.normal(size=length) + 0.2j * rng.normal(size=length)
                   for _ in targets]
    obj.weighted_kernels = [w * k for w, k in zip(obj.weights, obj.kernels)]
    if kind == "eigenmode":
        obj._reference_mode_fields = tuple(t.reference_mode_fields for t in targets)
    return obj


def history(obj, *, unique=False):
    width = len(obj._monitor_target_positions if unique else obj._flat_target_positions)
    rng = np.random.default_rng(20260911)
    return rng.normal(size=(9, width)) + 1j * rng.normal(size=(9, width))


def direct_filtered(obj, raw):
    bands = []
    for channels, kernel in zip(obj._target_slices, obj.weighted_kernels):
        full = obj.dt * fftconvolve(raw[:, channels], kernel[:, None], mode="full", axes=0)
        bands.append(full)
    return np.concatenate(bands, axis=1)


def direct_values(obj, filtered):
    values = []
    for band, channels in enumerate(obj._target_slices):
        raw = filtered[:, channels]
        weights = np.asarray(obj.target_spatial_weights[band]).reshape(-1)
        if not (obj._flux_target_mask[band] or obj._eigenmode_target_mask[band]):
            values.append(0.5 * obj.dt * np.sum(np.abs(raw) ** 2 * weights))
            continue
        components = {c: filtered[:, s] for c, s in obj._target_component_slices[band].items()}
        pairs = ((mp.Ey, mp.Hz, 1.0), (mp.Ez, mp.Hy, -1.0))
        if obj._flux_target_mask[band]:
            power = np.zeros(len(raw) - 1)
            for electric, magnetic, sign in pairs:
                h_mid = 0.5 * (components[magnetic][:-1] + components[magnetic][1:])
                power += -sign * np.sum(weights * np.real(np.conj(components[electric][:-1]) * h_mid), axis=1)
            values.append(obj.dt * np.sum(power))
        else:
            mode = obj._reference_mode_fields[band]
            mode_power = sum(sign * np.sum(weights * np.real(np.conj(mode[e]) * mode[h]))
                             for e, h, sign in pairs)
            coefficient = np.zeros(len(raw) - 1, dtype=complex)
            for electric, magnetic, sign in pairs:
                h_mid = 0.5 * (components[magnetic][:-1] + components[magnetic][1:])
                coefficient += 0.5 * sign / np.sqrt(abs(mode_power)) * np.sum(
                    weights * (components[electric][:-1] * np.conj(mode[magnetic])
                               + np.conj(mode[electric]) * h_mid), axis=1)
            values.append(obj.dt * np.sum(np.abs(coefficient) ** 2))
    return np.asarray(values)


def evaluate_history(obj, raw):
    filtered = obj._distributed_filter_monitor_signals(raw)
    bands, filtered_q = obj._band_fom_values_and_adjoint_signals(filtered, need_gradient=True)
    value, coefficients, _ = obj._evaluate_scalarization(bands, need_gradient=True)
    q = obj._filter_transpose_adjoint_signals(filtered_q, coefficients)
    return bands, value, q


def assert_directional_fd(value_fn, raw, q, dt):
    rng = np.random.default_rng(17)
    direction = rng.normal(size=raw.shape) + 1j * rng.normal(size=raw.shape)
    step = 1.0e-6
    fd = (value_fn(raw + step * direction) - value_fn(raw - step * direction)) / (2 * step)
    predicted = dt * np.real(np.sum(q * direction))
    np.testing.assert_allclose(predicted, fd, rtol=2e-7, atol=2e-9)


@pytest.mark.parametrize("length", [1, 4, 5, 13])
def test_physical_duration_is_independent_of_filter_length(length):
    obj = make_objective(length=length)
    assert obj.run_time == obj.t_final == 0.9
    obj._configure_time_grid(0.05)
    assert obj.run_time == obj.t_final


def test_late_time_grid_resolution_keeps_physical_duration():
    obj = make_objective(dt=None)
    assert obj.dt is None
    obj._configure_time_grid(0.075)
    assert obj.run_time == obj.t_final == 0.9


@pytest.mark.parametrize("dtype", [np.bool_, np.int64, np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("complex_kernel", [False, True])
def test_public_full_filter_preserves_primitive_dtype_and_values(dtype, complex_kernel):
    obj = make_objective("point")
    if not complex_kernel:
        obj.kernels = [kernel.real for kernel in obj.kernels]
        obj.weighted_kernels = [weight * kernel for weight, kernel in zip(obj.weights, obj.kernels)]
    values = np.arange(18).reshape(9, 2)
    raw = (values % 3 == 0).astype(dtype) if dtype == np.bool_ else values.astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        raw += 1j * raw[::-1]
    expected = np.concatenate([
        temporal_convolve_signal(raw[:, channels], kernel, obj.dt)
        for channels, kernel in zip(obj._target_slices, obj.weighted_kernels)
    ], axis=1)
    actual = obj.filter_monitor_signals(raw)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("kind", ["point", "region", "flux", "eigenmode"])
@pytest.mark.parametrize("length", [1, 4, 5, 13])
def test_ordinary_values_and_complex_scalarized_pullback(kind, length):
    obj = make_objective(kind, length,
                         scalarization_fn=lambda v: v[0] + 0.4 * v[1] ** 2)
    raw = history(obj)
    expected = direct_filtered(obj, raw)
    filtered = obj._distributed_filter_monitor_signals(raw)
    assert filtered.shape == (len(raw) + length - 1, raw.shape[1])
    np.testing.assert_allclose(filtered, expected, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(obj.filter_monitor_signals(raw), expected, rtol=2e-13, atol=2e-13)
    bands, _, q = evaluate_history(obj, raw)
    np.testing.assert_allclose(bands, direct_values(obj, expected), rtol=2e-13, atol=2e-13)
    assert q.shape == raw.shape
    assert_directional_fd(lambda x: evaluate_history(obj, x)[1], raw, q, obj.dt)


@pytest.mark.parametrize("kind", ["region", "flux", "eigenmode"])
def test_custom_callback_receives_full_support_and_preserves_sample_shape(kind):
    observed = []

    def fom(values, dt):
        observed.append(values.shape)
        return dt * npa.sum(npa.abs(values) ** 4)

    obj = make_objective(kind, fom_fn=(fom, None))
    raw = history(obj)
    _, _, q = evaluate_history(obj, raw)
    expected = (13, 1, 2) if kind == "region" else (12,)
    assert observed and all(shape == expected for shape in observed)
    assert_directional_fd(lambda x: evaluate_history(obj, x)[1], raw, q, obj.dt)


@pytest.mark.parametrize("kind", ["region", "flux"])
def test_shared_monitor_channels_sum_full_pullbacks(kind):
    obj = make_objective(kind, duplicate=True)
    assert obj._uses_deduplicated_monitors
    unique = history(obj, unique=True)
    logical = unique[:, obj._flat_to_monitor_indices]
    bands, _, logical_q = evaluate_history(obj, logical)
    _, coefficients, _ = obj._evaluate_scalarization(bands, need_gradient=True)
    actual = obj._distributed_deduplicated_band_objectives(unique)
    np.testing.assert_allclose(actual, direct_values(obj, direct_filtered(obj, logical)), rtol=2e-13, atol=2e-13)
    q = obj._distributed_deduplicated_adjoint_signals(unique, coefficients)
    expected_q = np.zeros_like(unique)
    for column, monitor in enumerate(obj._flat_to_monitor_indices):
        expected_q[:, monitor] += logical_q[:, column]
    np.testing.assert_allclose(q, expected_q, rtol=2e-13, atol=2e-13)

    def value(x):
        values = obj._distributed_deduplicated_band_objectives(x)
        return obj._evaluate_scalarization(values, need_gradient=False)[0]

    assert_directional_fd(value, unique, q, obj.dt)


@pytest.mark.parametrize("length", [1, 4, 5, 13])
@pytest.mark.parametrize("count", [1, 2])
def test_streaming_eigenmode_full_support_and_overlap_pullback(length, count):
    # Native monitors supply one contracted electric/magnetic pair per band.
    obj = make_objective("eigenmode", length, count=count)
    raw = history(obj)[:, :2 * count]
    filtered = np.stack([
        obj.dt * fftconvolve(raw[:, 2 * band:2 * band + 2], kernel[:, None], mode="full", axes=0)
        for band, kernel in enumerate(obj.weighted_kernels)
    ], axis=1)
    modal = filtered[:-1, :, 0] + 0.5 * (filtered[:-1, :, 1] + filtered[1:, :, 1])
    values, filtered_q = obj._eigenmode_fom_values_and_overlap_covectors(raw, need_gradient=True)
    assert filtered_q.shape == filtered.shape
    np.testing.assert_allclose(values, obj.dt * np.sum(np.abs(modal) ** 2, axis=0), rtol=2e-13, atol=2e-13)
    coefficients = np.asarray([1.0, -0.4][:count])
    q = obj._filter_transpose_eigenmode_overlap_covectors(filtered_q, coefficients)
    expected_q = np.stack([
        np.conj(temporal_convolve_signal_transpose(
            np.conj(filtered_q[:, band, :] * coefficients[band]), kernel, obj.dt))
        for band, kernel in enumerate(obj.weighted_kernels)
    ], axis=1)
    assert q.shape == (len(raw), count, 2)
    np.testing.assert_allclose(q, expected_q, rtol=2e-13, atol=2e-13)
    assert_directional_fd(
        lambda x: np.dot(coefficients, obj._eigenmode_fom_values_and_overlap_covectors(x, need_gradient=False)[0]),
        raw, q.reshape(raw.shape), obj.dt)


def distributed_layout(obj, raw, *, replicated, implicit_zero):
    owners = np.full(raw.shape[1], -1 if replicated else 0, dtype=int)
    if implicit_zero:
        zero = [i for i, component in enumerate(obj._monitor_target_components) if component == mp.Hy]
        owners[zero] = -2
        raw[:, zero] = 0
    groups, lookup, stored = [], {}, []
    for _, indices in obj._monitor_groups:
        indices = np.asarray([i for i in indices if owners[i] == -1 or owners[i] == mp.my_rank()], dtype=int)
        group = len(groups)
        groups.append({"history": raw[:, indices].copy()})
        lookup.update({int(index): (group, column) for column, index in enumerate(indices)})
        stored.append(indices)
    return dict(groups=groups, lookup=lookup, owners=owners, peak_work_columns=0), stored


@pytest.mark.parametrize("kind", ["region", "flux"])
@pytest.mark.parametrize("replicated", [False, True])
def test_distributed_builtin_blocks_and_empty_owned_partitions(kind, replicated):
    obj = make_objective(kind, duplicate=True, target_history_block_size=1)
    unique = history(obj, unique=True)
    layout, stored = distributed_layout(obj, unique, replicated=replicated, implicit_zero=(kind == "flux"))
    obj.last_target_history_layout = {}
    logical = unique[:, obj._flat_to_monitor_indices]
    bands, _, logical_q = evaluate_history(obj, logical)
    _, coefficients, _ = obj._evaluate_scalarization(bands, need_gradient=True)
    expected = direct_values(obj, direct_filtered(obj, logical))
    np.testing.assert_allclose(obj._distributed_builtin_band_objectives(layout), expected, rtol=2e-13, atol=2e-13)
    expected_q = np.zeros_like(unique)
    for column, monitor in enumerate(obj._flat_to_monitor_indices):
        expected_q[:, monitor] += logical_q[:, column]
    actual_q = obj._distributed_builtin_adjoint_histories(layout, coefficients)
    for values, indices in zip(actual_q, stored):
        assert values.shape == (len(unique), len(indices))
        np.testing.assert_allclose(values, expected_q[:, indices], rtol=2e-13, atol=2e-13)
    assert layout["peak_work_columns"] <= 1
