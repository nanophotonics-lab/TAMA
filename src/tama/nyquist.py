"""Nyquist-rate helpers for sparse time-domain field histories."""

import ctypes
import ctypes.util
import os
from collections.abc import Mapping
from contextlib import contextmanager
from numbers import Real

import numpy as np

_DEFAULT_SINC_HALF_WIDTH = 64
_DEFAULT_KAISER_BETA = 6.0
_RECONSTRUCTION_WINDOWS = {
    "rectangular",
    "hamming",
    "hann",
    "blackman",
    "bartlett",
    "kaiser",
}


def _openblas_thread_controls():
    """Return OpenBLAS's thread-local setter and current-thread getter."""
    try:
        library_path = ctypes.util.find_library("blas")
        if not library_path:
            return None, None
        library = ctypes.CDLL(library_path)
        setter = library.openblas_set_num_threads_local
        setter.argtypes = [ctypes.c_int]
        setter.restype = ctypes.c_int
        getter = library.openblas_get_num_threads
        getter.argtypes = []
        getter.restype = ctypes.c_int
        return setter, getter
    except (AttributeError, OSError):
        return None, None


def _reconstruction_blas_threads() -> int:
    try:
        available_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        available_cpus = os.cpu_count() or 1
    local_processes = 1
    for variable in (
        "MPI_LOCALNRANKS",
        "OMPI_COMM_WORLD_LOCAL_SIZE",
        "MPI_LOCAL_SIZE",
    ):
        try:
            local_processes = max(local_processes, int(os.environ.get(variable, 1)))
        except ValueError:
            pass
    return max(1, min(8, available_cpus // local_processes))


_OPENBLAS_SET_LOCAL_THREADS, _OPENBLAS_GET_THREADS = _openblas_thread_controls()
_RECONSTRUCTION_BLAS_THREADS = _reconstruction_blas_threads()


@contextmanager
def _limited_reconstruction_blas_threads():
    """Temporarily cap this process's reconstruction BLAS threads."""
    previous_threads = None
    if _OPENBLAS_SET_LOCAL_THREADS is not None and _OPENBLAS_GET_THREADS is not None:
        current_threads = _OPENBLAS_GET_THREADS()
        if current_threads > _RECONSTRUCTION_BLAS_THREADS:
            previous_threads = current_threads
            _OPENBLAS_SET_LOCAL_THREADS(_RECONSTRUCTION_BLAS_THREADS)
    try:
        yield
    finally:
        if previous_threads is not None:
            _OPENBLAS_SET_LOCAL_THREADS(previous_threads)


def _positive_finite_float(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _validate_reconstruction_settings(
    window,
    window_params,
    half_width,
):
    """Return canonical validated finite-sinc reconstruction settings."""
    if window is None:
        window = "rectangular"
    if not isinstance(window, str):
        raise TypeError("reconstruction_window must be a string or None")
    window = window.lower()
    window = {
        "boxcar": "rectangular",
        "none": "rectangular",
        "hanning": "hann",
    }.get(window, window)
    if window not in _RECONSTRUCTION_WINDOWS:
        raise ValueError(
            "reconstruction_window must be one of: rectangular, hamming, "
            "hann, blackman, bartlett, kaiser"
        )

    if window_params is None:
        window_params = {}
    elif not isinstance(window_params, Mapping):
        raise TypeError("reconstruction_window_params must be a mapping or None")
    else:
        window_params = dict(window_params)

    if window == "kaiser":
        unknown = set(window_params) - {"beta"}
        if unknown:
            raise ValueError("kaiser reconstruction window only accepts beta")
        beta = window_params.get("beta", _DEFAULT_KAISER_BETA)
        if (
            isinstance(beta, (bool, np.bool_))
            or not isinstance(beta, Real)
            or not np.isfinite(beta)
            or beta < 0.0
        ):
            raise ValueError(
                "reconstruction Kaiser beta must be nonnegative and finite"
            )
        beta = float(beta)
        with np.errstate(over="ignore", invalid="ignore"):
            beta_normalization = np.i0(beta)
        if not np.isfinite(beta_normalization):
            raise ValueError("reconstruction Kaiser beta is too large")
        window_params = {"beta": beta}
    elif window_params:
        raise ValueError(f"{window} reconstruction window does not accept parameters")

    if (
        isinstance(half_width, (bool, np.bool_))
        or not isinstance(half_width, (int, np.integer))
        or half_width < 1
    ):
        raise ValueError("reconstruction_half_width must be a positive integer")
    return window, window_params, int(half_width)


def _reconstruction_window_values(
    window: str,
    normalized_distance: np.ndarray,
    window_params,
) -> np.ndarray:
    """Evaluate a supported reconstruction window on `[-1, 1]`."""
    x = np.asarray(normalized_distance, dtype=float)
    if window == "rectangular":
        return np.ones_like(x)
    if window == "hamming":
        return 0.54 + 0.46 * np.cos(np.pi * x)
    if window == "hann":
        return 0.5 * (1.0 + np.cos(np.pi * x))
    if window == "blackman":
        return 0.42 + 0.5 * np.cos(np.pi * x) + 0.08 * np.cos(2.0 * np.pi * x)
    if window == "bartlett":
        return 1.0 - np.abs(x)
    beta = window_params["beta"]
    return np.i0(beta * np.sqrt(np.maximum(0.0, 1.0 - x**2))) / np.i0(beta)


def nyquist_sampling_interval(max_frequency: float, dt: float) -> int:
    """Return the largest FDTD-step interval satisfying the Nyquist limit.

    This implements `floor(1 / (2 * max_frequency * dt))`. The caller is
    responsible for choosing a maximum frequency that covers the source,
    objective, adjoint response, generated frequencies, and a safety margin
    that keeps relevant energy strictly below the resulting sparse Nyquist
    frequency. If even full-rate FDTD sampling cannot satisfy that strict
    bound, the configuration is rejected instead of silently returning one.

    Args:
        max_frequency: Highest relevant temporal frequency in Meep units.
        dt: FDTD time step in Meep units.

    Returns:
        Positive integer number of FDTD steps between stored field samples.

    Raises:
        ValueError: If either argument is non-finite or not positive, or if
            `max_frequency` is not below the full-rate temporal Nyquist limit.
    """
    max_frequency = _positive_finite_float(max_frequency, "max_frequency")
    dt = _positive_finite_float(dt, "dt")
    if max_frequency >= 0.5 / dt:
        raise ValueError(
            "max_frequency must be below the full-rate temporal Nyquist "
            "frequency 1 / (2 * dt)"
        )
    return int(np.floor(1.0 / (2.0 * max_frequency * dt)))


def _aligned_run_until(
    requested_time: float,
    *,
    dt: float,
    sampling_interval: int,
) -> float:
    """Return a run limit whose completed step count is divisible by K."""
    if sampling_interval <= 1:
        return float(requested_time)
    step_ratio = float(requested_time) / float(dt)
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(step_ratio))
    requested_steps = max(1, int(np.ceil(step_ratio - tolerance)))
    aligned_steps = (
        (requested_steps + sampling_interval - 1) // sampling_interval
    ) * sampling_interval
    aligned_time = aligned_steps * float(dt)
    return aligned_time - 0.5 * float(dt)


def _monitor_history_stop(sample_times: np.ndarray, requested_time: float) -> int:
    """Return the slice stop matching Meep's first step at or past `until`."""
    stop = int(np.searchsorted(sample_times, requested_time, side="left")) + 1
    return min(stop, len(sample_times))


def _windowed_sinc_stencils(
    sampling_interval: int,
    half_width: int,
    window: str,
    window_params,
):
    """Return the sparse-row offset and phase reconstruction weights."""
    if sampling_interval == 1:
        return 0, np.ones((1, 1), dtype=float)

    offsets = np.arange(
        -half_width + 1,
        half_width + 1,
        dtype=np.int64,
    )
    weight_matrix = np.zeros(
        (sampling_interval, offsets.size),
        dtype=float,
    )
    weight_matrix[0, np.flatnonzero(offsets == 0)[0]] = 1.0
    for phase in range(1, sampling_interval):
        fractional_index = phase / sampling_interval
        distance = fractional_index - offsets
        normalized_distance = distance / half_width
        phase_window = _reconstruction_window_values(
            window,
            normalized_distance,
            window_params,
        )
        weights = np.sinc(distance) * phase_window
        weights[np.abs(distance) >= half_width] = 0.0
        weights /= np.sum(weights)
        weight_matrix[phase] = weights
    return int(offsets[0]), weight_matrix


class _WindowedSincReconstructor:
    """Evaluate a fine time row from a Nyquist-sampled field history.

    The paper specifies reconstruction from the sparse representation but does
    not prescribe a numerical interpolation method. This implementation uses
    a configurable finite-windowed cardinal-sinc kernel. Values outside the
    recorded time interval are treated as zero; the record is never wrapped
    periodically. The forward field should therefore be negligible at both
    temporal endpoints.
    """

    def __init__(
        self,
        history: np.ndarray,
        sampling_interval: int,
        *,
        half_width: int = _DEFAULT_SINC_HALF_WIDTH,
        window="kaiser",
        window_params=None,
    ) -> None:
        history_array = np.asarray(history)
        if history_array.ndim != 2:
            raise ValueError("history must be a 2D (time, point) array")
        if history_array.shape[0] < 2:
            raise ValueError("history must contain at least two time samples")
        if not (
            np.issubdtype(history_array.dtype, np.floating)
            or np.issubdtype(history_array.dtype, np.complexfloating)
        ):
            raise ValueError("history must have a real or complex floating dtype")
        if (
            isinstance(sampling_interval, (bool, np.bool_))
            or not isinstance(sampling_interval, (int, np.integer))
            or sampling_interval < 1
        ):
            raise ValueError("sampling_interval must be a positive integer")
        window, window_params, half_width = _validate_reconstruction_settings(
            window,
            window_params,
            half_width,
        )

        self.history = history_array
        self.sampling_interval = int(sampling_interval)
        self.half_width = half_width
        self.window = window
        self.window_params = window_params
        self.last_fine_index = (history_array.shape[0] - 1) * self.sampling_interval
        self._first_offset, self._weight_matrix = _windowed_sinc_stencils(
            self.sampling_interval,
            self.half_width,
            self.window,
            self.window_params,
        )
        self._cached_coarse_index = None
        self._cached_block = np.empty(
            (self.sampling_interval, history_array.shape[1]),
            dtype=np.result_type(history_array.dtype, np.float64),
        )
        self.dtype = self._cached_block.dtype

    def _fill_cached_block(self, coarse_index: int) -> None:
        first_row = coarse_index + self._first_offset
        history_start = max(0, first_row)
        history_stop = min(
            self.history.shape[0],
            first_row + self._weight_matrix.shape[1],
        )
        self._cached_block.fill(0.0)
        if history_start < history_stop:
            weight_start = history_start - first_row
            weight_stop = weight_start + history_stop - history_start
            with _limited_reconstruction_blas_threads():
                np.matmul(
                    self._weight_matrix[:, weight_start:weight_stop],
                    self.history[history_start:history_stop],
                    out=self._cached_block,
                )
        self._cached_coarse_index = coarse_index

    def _validate_fine_index(self, fine_index: int) -> int:
        if isinstance(fine_index, (bool, np.bool_)) or not isinstance(
            fine_index, (int, np.integer)
        ):
            raise ValueError("fine_index must be an integer")
        fine_index = int(fine_index)
        if fine_index < 0 or fine_index > self.last_fine_index:
            raise IndexError("fine_index is outside the recorded history")
        return fine_index

    def sample_into(
        self,
        fine_index: int,
        destination: np.ndarray,
    ) -> np.ndarray:
        """Write one reconstructed fine-grid row into `destination`."""
        fine_index = self._validate_fine_index(fine_index)
        destination_array = np.asarray(destination)
        if (
            destination_array.ndim != 1
            or destination_array.shape[0] != self.history.shape[1]
            or destination_array.dtype != self.dtype
        ):
            raise ValueError(
                "destination must match the reconstructed history dtype and width"
            )

        coarse_index, phase = divmod(fine_index, self.sampling_interval)
        if phase == 0:
            destination_array[:] = self.history[coarse_index]
            return destination
        if self._cached_coarse_index != coarse_index:
            self._fill_cached_block(coarse_index)
        destination_array[:] = self._cached_block[phase]
        return destination

    def sample_at(self, fine_index: int) -> np.ndarray:
        """Return one reconstructed fine-grid row."""
        destination = np.empty(self.history.shape[1], dtype=self.dtype)
        self.sample_into(fine_index, destination)
        return destination
