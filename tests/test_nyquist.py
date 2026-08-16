import meep as mp
import numpy as np

import tama as tm
import tama.nyquist as nyquist_module
from tama.nyquist import (
    _aligned_run_until,
    _monitor_history_stop,
    _WindowedSincReconstructor,
)


def test_nyquist_sampling_interval_matches_paper_formula():
    dt = 0.5 / 40
    max_frequency = 1.0 / 0.38

    assert tm.nyquist_sampling_interval(max_frequency, dt) == 15
    assert tm.nyquist_sampling_interval(39.0, dt) == 1

    for aliased in (40.0, 40.01, 100.0):
        with np.testing.assert_raises_regex(
            ValueError,
            "full-rate temporal Nyquist",
        ):
            tm.nyquist_sampling_interval(aliased, dt)

    for invalid in (0.0, -1.0, np.inf, True, "2.0"):
        try:
            tm.nyquist_sampling_interval(invalid, dt)
        except ValueError as exc:
            assert "max_frequency" in str(exc)
        else:
            raise AssertionError("expected invalid max_frequency error")


def test_nyquist_run_limit_aligns_completed_steps():
    requested_time = 10.01
    dt = 0.01
    sampling_interval = 8

    run_until = _aligned_run_until(
        requested_time,
        dt=dt,
        sampling_interval=sampling_interval,
    )
    completed_steps = int(np.ceil(run_until / dt))

    assert completed_steps == 1008
    assert completed_steps % sampling_interval == 0
    assert completed_steps * dt - requested_time < sampling_interval * dt
    assert _aligned_run_until(
        requested_time,
        dt=dt,
        sampling_interval=1,
    ) == requested_time

    extended_times = np.arange(completed_steps + 1) * dt
    assert _monitor_history_stop(extended_times, requested_time) == 1002


def test_nyquist_run_limit_aligns_actual_meep_callbacks():
    dt = 0.5 / 8
    sampling_interval = 3
    run_until = _aligned_run_until(
        0.26,
        dt=dt,
        sampling_interval=sampling_interval,
    )
    sample_times = []
    sim = mp.Simulation(cell_size=mp.Vector3(2, 2), resolution=8)
    sim.run(lambda active: sample_times.append(active.meep_time()), until=run_until)
    sim.reset_meep()

    assert len(sample_times) - 1 == 6
    assert (len(sample_times) - 1) % sampling_interval == 0
    assert np.isclose(sample_times[-1], 6 * dt)


def test_windowed_sinc_k1_is_exact_identity():
    rng = np.random.default_rng(20260714)
    history = rng.standard_normal((9, 3)) + 1j * rng.standard_normal((9, 3))
    reconstructor = _WindowedSincReconstructor(history, 1)

    for fine_index in range(history.shape[0]):
        assert np.array_equal(
            reconstructor.sample_at(fine_index),
            history[fine_index],
        )


def test_windowed_sinc_preserves_real_history_storage():
    rng = np.random.default_rng(20260803)
    history = rng.standard_normal((9, 3)).astype(np.float64)
    reconstructor = _WindowedSincReconstructor(history, 1)

    assert reconstructor.history.dtype == np.float64
    assert reconstructor._cached_block.dtype == np.float64
    assert reconstructor.sample_at(0).dtype == np.float64
    for fine_index in range(history.shape[0]):
        assert np.array_equal(
            reconstructor.sample_at(fine_index),
            history[fine_index],
        )

    single_precision = _WindowedSincReconstructor(
        history.astype(np.float32),
        1,
    )
    assert single_precision.history.dtype == np.float32
    assert single_precision._cached_block.dtype == np.float64
    assert single_precision.sample_at(0).dtype == np.float64


def test_windowed_sinc_default_matches_explicit_kaiser_six():
    history = np.ones((20, 2), dtype=np.complex128)
    default = _WindowedSincReconstructor(history, 5)
    explicit = _WindowedSincReconstructor(
        history,
        5,
        window="kaiser",
        window_params={"beta": 6.0},
    )

    assert np.array_equal(default._weight_matrix, explicit._weight_matrix)
    offsets = np.arange(-default.half_width + 1, default.half_width + 1)
    legacy_weights = np.zeros_like(default._weight_matrix)
    legacy_weights[0, np.flatnonzero(offsets == 0)[0]] = 1.0
    for phase in range(1, default.sampling_interval):
        distance = phase / default.sampling_interval - offsets
        normalized_distance = distance / default.half_width
        window = np.i0(
            6.0 * np.sqrt(np.maximum(0.0, 1.0 - normalized_distance**2))
        ) / np.i0(6.0)
        weights = np.sinc(distance) * window
        weights[np.abs(distance) >= default.half_width] = 0.0
        legacy_weights[phase] = weights / np.sum(weights)
    assert np.array_equal(default._weight_matrix, legacy_weights)


def test_windowed_sinc_supports_configurable_windows_and_aliases():
    rng = np.random.default_rng(20260716)
    history = rng.standard_normal((30, 2)) + 1j * rng.standard_normal((30, 2))
    configurations = [
        (None, None, "rectangular"),
        ("boxcar", None, "rectangular"),
        ("none", None, "rectangular"),
        ("hamming", None, "hamming"),
        ("hann", None, "hann"),
        ("hanning", None, "hann"),
        ("blackman", None, "blackman"),
        ("bartlett", None, "bartlett"),
        ("kaiser", {"beta": 4.0}, "kaiser"),
    ]

    for window, params, expected_name in configurations:
        reconstructor = _WindowedSincReconstructor(
            history,
            4,
            half_width=8,
            window=window,
            window_params=params,
        )
        assert reconstructor.window == expected_name
        assert np.all(np.isfinite(reconstructor._weight_matrix))
        assert np.allclose(
            np.sum(reconstructor._weight_matrix, axis=1),
            1.0,
            rtol=0.0,
            atol=1.0e-14,
        )
        for coarse_index in range(history.shape[0]):
            assert np.array_equal(
                reconstructor.sample_at(4 * coarse_index),
                history[coarse_index],
            )
        assert np.all(np.isfinite(reconstructor.sample_at(41)))
        minimum_support = _WindowedSincReconstructor(
            history,
            4,
            half_width=1,
            window=window,
            window_params=params,
        )
        assert np.all(np.isfinite(minimum_support._weight_matrix))
        assert np.allclose(
            np.sum(minimum_support._weight_matrix, axis=1),
            1.0,
            rtol=0.0,
            atol=1.0e-14,
        )


def test_windowed_sinc_kaiser_zero_matches_rectangular():
    history = np.ones((20, 1), dtype=np.complex128)
    rectangular = _WindowedSincReconstructor(
        history,
        5,
        window="rectangular",
    )
    kaiser_zero = _WindowedSincReconstructor(
        history,
        5,
        window="kaiser",
        window_params={"beta": 0.0},
    )

    assert np.array_equal(
        rectangular._weight_matrix,
        kaiser_zero._weight_matrix,
    )
    kaiser_four = _WindowedSincReconstructor(
        history,
        5,
        window="kaiser",
        window_params={"beta": 4.0},
    )
    kaiser_eight = _WindowedSincReconstructor(
        history,
        5,
        window="kaiser",
        window_params={"beta": 8.0},
    )
    assert not np.allclose(
        kaiser_four._weight_matrix,
        kaiser_eight._weight_matrix,
    )


def test_windowed_sinc_does_not_raise_existing_blas_thread_limit(monkeypatch):
    setter_calls = []
    monkeypatch.setattr(nyquist_module, "_OPENBLAS_GET_THREADS", lambda: 1)
    monkeypatch.setattr(
        nyquist_module,
        "_OPENBLAS_SET_LOCAL_THREADS",
        setter_calls.append,
    )
    monkeypatch.setattr(nyquist_module, "_RECONSTRUCTION_BLAS_THREADS", 8)
    history = np.ones((5, 2), dtype=np.complex128)

    _WindowedSincReconstructor(history, 2).sample_at(1)

    assert setter_calls == []


def test_reconstruction_blas_thread_limit_is_restored(monkeypatch):
    setter_calls = []
    monkeypatch.setattr(nyquist_module, "_OPENBLAS_GET_THREADS", lambda: 16)
    monkeypatch.setattr(
        nyquist_module,
        "_OPENBLAS_SET_LOCAL_THREADS",
        setter_calls.append,
    )
    monkeypatch.setattr(nyquist_module, "_RECONSTRUCTION_BLAS_THREADS", 4)

    with nyquist_module._limited_reconstruction_blas_threads():
        pass

    assert setter_calls == [4, 16]


def test_windowed_sinc_preserves_every_sparse_node_and_terminal_row():
    rng = np.random.default_rng(20260714)
    sampling_interval = 8
    history = rng.standard_normal((21, 2)) + 1j * rng.standard_normal((21, 2))
    reconstructor = _WindowedSincReconstructor(history, sampling_interval)

    for coarse_index in range(history.shape[0]):
        assert np.array_equal(
            reconstructor.sample_at(coarse_index * sampling_interval),
            history[coarse_index],
        )


def test_windowed_sinc_reconstructs_bandlimited_complex_field_at_all_phases():
    sampling_interval = 8
    fine_step_count = sampling_interval * 400
    fine_index = np.arange(fine_step_count + 1)
    envelope = np.sin(np.pi * fine_index / fine_step_count) ** 8
    forward = envelope * (
        np.exp(1j * np.pi * 0.2 * fine_index / sampling_interval)
        + 0.3 * np.exp(-1j * np.pi * 0.6 * fine_index / sampling_interval)
    )
    reconstructor = _WindowedSincReconstructor(
        forward[::sampling_interval, None].copy(),
        sampling_interval,
    )
    reconstructed = np.asarray([
        reconstructor.sample_at(index)[0]
        for index in range(fine_step_count + 1)
    ])

    relative_error = np.linalg.norm(reconstructed - forward) / np.linalg.norm(forward)
    assert relative_error < 1.0e-4
    for phase in range(sampling_interval):
        phase_error = np.linalg.norm(
            reconstructed[phase::sampling_interval]
            - forward[phase::sampling_interval]
        ) / np.linalg.norm(forward[phase::sampling_interval])
        assert phase_error < 2.0e-4


def test_windowed_sinc_full_rate_contraction_allows_high_frequency_adjoint():
    sampling_interval = 8
    fine_step_count = sampling_interval * 400
    fine_index = np.arange(fine_step_count + 1)
    envelope = np.sin(np.pi * fine_index / fine_step_count) ** 8
    forward = envelope * np.exp(
        1j * np.pi * 0.55 * fine_index / sampling_interval
    )
    adjoint_callback = (
        np.exp(1j * 0.23 * fine_index)
        + 0.25 * (-1.0) ** fine_index
    )
    reconstructor = _WindowedSincReconstructor(
        forward[::sampling_interval, None].copy(),
        sampling_interval,
    )
    reconstructed_reverse = np.asarray([
        reconstructor.sample_at(fine_step_count - step)[0]
        for step in range(fine_step_count + 1)
    ])
    exact_reverse = forward[::-1]
    adjoint_difference = np.diff(adjoint_callback)
    exact = np.sum(
        0.5 * (exact_reverse[:-1] + exact_reverse[1:])
        * adjoint_difference
    )
    reconstructed = np.sum(
        0.5 * (reconstructed_reverse[:-1] + reconstructed_reverse[1:])
        * adjoint_difference
    )

    assert np.allclose(reconstructed, exact, rtol=5.0e-4, atol=5.0e-6)


def test_trapezoidal_adjoint_difference_preserves_k1_centered_gradient():
    rng = np.random.default_rng(20260716)
    dt = 0.01
    forward = rng.standard_normal(257) + 1j * rng.standard_normal(257)
    adjoint_callback = rng.standard_normal(257) + 1j * rng.standard_normal(257)
    forward[:2] = 0.0
    forward[-2:] = 0.0
    forward_derivative = np.empty_like(forward)
    forward_derivative[0] = (forward[1] - forward[0]) / dt
    forward_derivative[-1] = (forward[-1] - forward[-2]) / dt
    forward_derivative[1:-1] = (
        forward[2:] - forward[:-2]
    ) / (2.0 * dt)
    centered_gradient = dt * np.sum(
        adjoint_callback * forward_derivative[::-1]
    )

    reverse_forward = forward[::-1]
    difference_gradient = np.sum(
        0.5 * (reverse_forward[:-1] + reverse_forward[1:])
        * np.diff(adjoint_callback)
    )

    assert np.allclose(
        difference_gradient,
        centered_gradient,
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_windowed_sinc_zero_extension_never_wraps_record_edges():
    sampling_interval = 8
    history = np.zeros((150, 1), dtype=np.complex128)
    history[-1, 0] = 1.0
    reconstructor = _WindowedSincReconstructor(history, sampling_interval)

    for fine_index in range(2 * sampling_interval):
        assert reconstructor.sample_at(fine_index)[0] == 0.0


def test_windowed_sinc_validates_history_and_indices():
    history = np.zeros((4, 2), dtype=np.complex128)
    reconstructor = _WindowedSincReconstructor(history, 3)

    for invalid_index in (-1, reconstructor.last_fine_index + 1):
        try:
            reconstructor.sample_at(invalid_index)
        except IndexError:
            pass
        else:
            raise AssertionError("expected out-of-range fine index error")

    for invalid_interval in (0, 1.5, True):
        try:
            _WindowedSincReconstructor(history, invalid_interval)
        except ValueError as exc:
            assert "sampling_interval" in str(exc)
        else:
            raise AssertionError("expected invalid sampling interval error")

    invalid_settings = [
        ({"window": "unknown"}, "reconstruction_window"),
        ({"window": 3}, "reconstruction_window"),
        ({"window": "hann", "window_params": {"beta": 2.0}}, "does not accept"),
        ({"window": "kaiser", "window_params": {"unknown": 2.0}}, "only accepts"),
        ({"window": "kaiser", "window_params": {"beta": -1.0}}, "beta"),
        ({"window": "kaiser", "window_params": {"beta": np.inf}}, "beta"),
        ({"window": "kaiser", "window_params": {"beta": 1000.0}}, "too large"),
        ({"window": "kaiser", "window_params": []}, "mapping"),
        ({"half_width": 0}, "reconstruction_half_width"),
        ({"half_width": 1.5}, "reconstruction_half_width"),
    ]
    for kwargs, message in invalid_settings:
        try:
            _WindowedSincReconstructor(history, 3, **kwargs)
        except (TypeError, ValueError) as exc:
            assert message in str(exc)
        else:
            raise AssertionError("expected invalid reconstruction setting error")
