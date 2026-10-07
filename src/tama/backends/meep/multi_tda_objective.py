import gc
import tempfile
import time
from typing import Callable, Optional, Sequence, Tuple, Union

from autograd import grad, value_and_grad
import meep as mp
import numpy as np
from scipy.fft import fft, ifft, irfft, next_fast_len, rfft
from scipy.signal import firwin2

from .objectives import (
    _adjoint_indexed_profile_sources,
    _adjoint_point_sources,
    _electric_components,
    _is_cartesian_2d_simulation,
    _is_cylindrical_simulation,
    _is_magnetic_component,
    _is_zero_source_size,
    _sample_time_from_simulation,
    _simulation_time_step,
    _shift_tabulated_cubic_source,
    _tabulated_cubic_source as _shared_tabulated_cubic_source,
    _tabulated_cubic_sources as _shared_tabulated_cubic_sources,
    _validate_adjoint_source_amplitudes,
    _validate_cylindrical_axis_target,
    _validate_cylindrical_target_position,
    _validate_indexed_monitor_support,
    _validate_logical_adjoint_source_amplitudes,
)
from .sampling_grid import (
    FastEigenmodeOverlapMonitor,
    FastFieldGrid,
    FastPointMonitor,
    _require_no_meep_symmetries,
    history_storage_dtype,
    _prepare_native_material,
)
from .design_collection import (
    _DesignCollection,
    _resolve_designs,
    _collection_components as _native_design_components,
    _collection_gradient_scale as _native_design_gradient_scale,
    _collection_uses_material_jacobian as _uses_material_jacobian,
)
from .native_design import (
    _NativeDesignAccumulator,
    _NativeDesignHistorySet,
    _run_native_adjoint_loop,
    _run_native_forward_loop,
    _native_forward_step_count,
)
from .specs import (
    DesignGrid,
    EigenmodeCoefficientTarget,
    FieldRegionTarget,
    FluxTarget,
    Near2FarTarget,
    PointTarget,
    SimulationSpec,
)
from .flux import (
    _validate_cylindrical_surface,
    flux_component_pairs,
    power_history,
    power_history_pullback,
)
from .eigenmode import (
    _coefficient_history_from_overlaps,
    _coefficient_overlap_pullback,
    _coefficient_projection_weights,
    coefficient_history,
    coefficient_history_pullback,
    sample_reference_mode,
)
from .adaptive_chunking import AdaptiveAdjointChunkBalancer
from ...nyquist import (
    _aligned_run_until,
    _monitor_history_stop,
    _positive_finite_float,
    _validate_reconstruction_settings,
    nyquist_sampling_interval,
)


_NEAR2FAR_SAMPLING_ATTRIBUTES = (
    "t_final", "dt", "resolution", "_fallback_courant", "sampling_interval",
    "max_frequency", "history_dtype", "reconstruction_window",
    "reconstruction_window_params", "reconstruction_half_width",
)


def _tabulated_cubic_source(times: np.ndarray, values: np.ndarray):
    """Return one native complex cubic callable."""
    return _shared_tabulated_cubic_source(times, values)


def _tabulated_cubic_sources(times: np.ndarray, values: np.ndarray):
    """Build compact shared cubic B-spline callables."""
    return _shared_tabulated_cubic_sources(times, values)


def _normalize_band_callbacks(callbacks, band_count: int, name: str):
    """Broadcast one callback or validate one optional callback per band."""
    if callbacks is None:
        return (None,) * band_count
    if callable(callbacks):
        return (callbacks,) * band_count
    if isinstance(callbacks, (str, bytes)) or not isinstance(
        callbacks,
        Sequence,
    ):
        raise TypeError(
            f"{name} must be callable or a sequence of optional callables"
        )
    normalized = tuple(callbacks)
    if len(normalized) != band_count:
        raise ValueError(
            f"{name} sequence must match the number of wavelength bands"
        )
    if any(
        callback is not None and not callable(callback)
        for callback in normalized
    ):
        raise TypeError(f"{name} sequence entries must be callable or None")
    return normalized


def _validate_band_callback_pairs(fom_fns, adjoint_signal_fns) -> None:
    for band_index, (fom_fn, adjoint_signal_fn) in enumerate(
        zip(fom_fns, adjoint_signal_fns)
    ):
        if adjoint_signal_fn is not None and fom_fn is None:
            raise ValueError(
                f"band {band_index} adjoint_signal_fn requires fom_fn "
                "so the objective value and derivative remain consistent"
            )


def _validate_wavelength_bands(wavelength_bands) -> tuple[tuple[float, float], ...]:
    bands = tuple(wavelength_bands)
    if not bands:
        raise ValueError("wavelength_bands must not be empty")
    validated = []
    for band_index, band in enumerate(bands):
        try:
            lambda_min, lambda_max = band
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "each wavelength band must contain exactly two values"
            ) from exc
        lambda_min = _positive_finite_float(
            lambda_min,
            f"wavelength_bands[{band_index}][0]",
        )
        lambda_max = _positive_finite_float(
            lambda_max,
            f"wavelength_bands[{band_index}][1]",
        )
        if lambda_min >= lambda_max:
            raise ValueError(
                "each wavelength band must satisfy lambda_min < lambda_max"
            )
        validated.append((lambda_min, lambda_max))
    return tuple(validated)


def _numpy_window(window, length: int, window_params=None) -> np.ndarray:
    """Return a supported NumPy window by public TAMA window name."""
    if window_params is None:
        window_params = {}
    if window is None:
        window = "rectangular"
    if not isinstance(window, str):
        raise TypeError("window must be a string or None")

    name = window.lower()
    if name in ("rectangular", "boxcar", "none"):
        if window_params:
            raise ValueError("rectangular window does not accept parameters")
        return np.ones(length)
    if name == "hamming":
        if window_params:
            raise ValueError("hamming window does not accept parameters")
        return np.hamming(length)
    if name in ("hann", "hanning"):
        if window_params:
            raise ValueError("hann window does not accept parameters")
        return np.hanning(length)
    if name == "blackman":
        if window_params:
            raise ValueError("blackman window does not accept parameters")
        return np.blackman(length)
    if name == "bartlett":
        if window_params:
            raise ValueError("bartlett window does not accept parameters")
        return np.bartlett(length)
    if name == "kaiser":
        unknown = set(window_params) - {"beta"}
        if unknown:
            raise ValueError("kaiser window only accepts beta")
        if "beta" not in window_params:
            raise ValueError("kaiser window requires window_params={'beta': ...}")
        beta = float(window_params["beta"])
        if not np.isfinite(beta):
            raise ValueError("kaiser beta must be finite")
        return np.kaiser(length, beta)
    raise ValueError(
        "window must be one of: rectangular, hamming, hann, blackman, bartlett, kaiser"
    )


def bandpass_kernel(
    f_low: float,
    f_high: float,
    dt: float,
    length: int,
    *,
    window: Optional[str] = "hamming",
    window_params=None,
) -> np.ndarray:
    """Return a windowed ideal bandpass kernel normalized at band center.

    Args:
        f_low: Lower band edge in Meep frequency units.
        f_high: Upper band edge in Meep frequency units.
        dt: Time step of the sampled signal.
        length: Number of samples in the finite impulse response kernel.
        window: NumPy window used to taper the finite impulse response.
            Supported values are `rectangular`/`None`, `hamming`, `hann`,
            `blackman`, `bartlett`, and `kaiser`; `none`, `boxcar`, and
            `hanning` are accepted aliases.
        window_params: Optional window parameters. Currently only
            `window="kaiser"` uses this, with `{"beta": value}`.

    Returns:
        Real-valued 1D kernel. The response magnitude is normalized to one at
        the band-center frequency when the center response is nonzero.
    """
    f_low = _positive_finite_float(f_low, "f_low")
    f_high = _positive_finite_float(f_high, "f_high")
    dt = _positive_finite_float(dt, "dt")
    if f_low >= f_high:
        raise ValueError("bandpass frequencies must satisfy f_low < f_high")
    if f_high >= 0.5 / dt:
        raise ValueError(
            "f_high must be below the temporal Nyquist frequency 0.5 / dt"
        )
    if (
        isinstance(length, (bool, np.bool_))
        or not isinstance(length, (int, np.integer))
        or length < 1
    ):
        raise ValueError("length must be a positive integer")
    length = int(length)

    n = np.arange(length)
    tau = (n - (length - 1) / 2) * dt
    kernel = (
        2 * f_high * np.sinc(2 * f_high * tau)
        - 2 * f_low * np.sinc(2 * f_low * tau)
    )
    kernel *= _numpy_window(window, kernel.size, window_params)

    f_center = 0.5 * (f_low + f_high)
    response = np.sum(kernel * np.exp(-2j * np.pi * f_center * tau)) * dt
    gain = np.abs(response)
    if gain > 0:
        kernel /= gain
    return kernel


def _validate_power_complementary_bands(
    wavelength_bands: Sequence[tuple[float, float]],
) -> None:
    if len(wavelength_bands) < 2:
        raise ValueError(
            "power_complementary filter_bank requires at least two bands"
        )
    for left, right in zip(wavelength_bands[:-1], wavelength_bands[1:]):
        if not np.isclose(left[1], right[0], rtol=1.0e-12, atol=0.0):
            raise ValueError(
                "power_complementary wavelength bands must be ordered "
                "and contiguous"
            )


def _power_complementary_kernels(
    wavelength_bands: Sequence[tuple[float, float]],
    dt: float,
    length: int,
    transition_width: float,
) -> list[np.ndarray]:
    """Return an approximately power-complementary linear-phase FIR bank."""
    sample_frequency = 1.0 / dt
    nyquist_frequency = 0.5 * sample_frequency
    half_transition = 0.5 * transition_width
    frequency_bands = tuple(
        (1.0 / lambda_max, 1.0 / lambda_min)
        for lambda_min, lambda_max in wavelength_bands
    )
    outer_low = frequency_bands[-1][0]
    outer_high = frequency_bands[0][1]
    if outer_low - transition_width <= 0.0:
        raise ValueError(
            "transition_width extends below zero frequency"
        )
    if outer_high + transition_width >= nyquist_frequency:
        raise ValueError(
            "transition_width extends to or above the temporal Nyquist "
            "frequency"
        )
    minimum_bandwidth = min(
        f_high - f_low for f_low, f_high in frequency_bands
    )
    if transition_width >= minimum_bandwidth:
        raise ValueError(
            "transition_width must be smaller than every band width"
        )

    design_point_count = 1 << int(
        np.ceil(np.log2(max(512, 2 * length)))
    )
    design_frequencies = np.linspace(
        0.0,
        nyquist_frequency,
        design_point_count + 1,
    )
    target_power = np.zeros(
        (len(frequency_bands), design_frequencies.size),
        dtype=float,
    )
    for band_index, (f_low, f_high) in enumerate(frequency_bands):
        inside = (
            (design_frequencies >= f_low)
            & (design_frequencies <= f_high)
        )
        target_power[band_index, inside] = 1.0

    for band_index in range(len(frequency_bands) - 1):
        boundary = frequency_bands[band_index][0]
        crossover = (
            (design_frequencies >= boundary - half_transition)
            & (design_frequencies <= boundary + half_transition)
        )
        coordinate = (
            design_frequencies[crossover]
            - (boundary - half_transition)
        ) / transition_width
        target_power[band_index, crossover] = np.sin(
            0.5 * np.pi * coordinate
        ) ** 2
        target_power[band_index + 1, crossover] = np.cos(
            0.5 * np.pi * coordinate
        ) ** 2

    lower_taper = (
        (design_frequencies >= outer_low - transition_width)
        & (design_frequencies < outer_low)
    )
    lower_coordinate = (
        design_frequencies[lower_taper]
        - (outer_low - transition_width)
    ) / transition_width
    target_power[-1, lower_taper] = np.sin(
        0.5 * np.pi * lower_coordinate
    ) ** 2

    upper_taper = (
        (design_frequencies > outer_high)
        & (design_frequencies <= outer_high + transition_width)
    )
    upper_coordinate = (
        design_frequencies[upper_taper] - outer_high
    ) / transition_width
    target_power[0, upper_taper] = np.cos(
        0.5 * np.pi * upper_coordinate
    ) ** 2

    return [
        firwin2(
            length,
            design_frequencies,
            np.sqrt(power),
            nfreqs=design_point_count + 1,
            window=None,
            fs=sample_frequency,
        )
        / dt
        for power in target_power
    ]


def temporal_convolve_signal(
    signal: np.ndarray,
    kernel: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Return the full linear convolution of a time signal and temporal kernel.

    Args:
        signal: Time-domain signal. Convolution is applied along axis 0.
        kernel: 1D temporal convolution kernel.
        dt: Time step used to scale the discrete convolution as a Riemann sum.

    Returns:
        All N + K - 1 output samples, preserving the remaining signal axes.
        The time axis must be nonempty. Integer and boolean signals are promoted
        to floating point.
    """
    signal = np.asarray(signal)
    kernel = np.asarray(kernel)
    if signal.ndim == 0 or signal.shape[0] == 0:
        raise ValueError("full convolution requires a nonempty time axis")
    return _temporal_fft_convolve(
        signal,
        kernel,
        output_start=0,
        output_length=signal.shape[0] + kernel.size - 1,
    ) * dt


def _temporal_fft_convolve(
    signal: np.ndarray,
    kernel: np.ndarray,
    *,
    output_start: int,
    output_length: int,
) -> np.ndarray:
    """Apply bounded-workspace linear convolution and retain a selected interval."""
    signal = np.asarray(signal)
    kernel = np.asarray(kernel)
    if kernel.ndim != 1:
        raise ValueError("kernel must be one-dimensional")
    if kernel.size == 0:
        raise ValueError("kernel must not be empty")
    if signal.ndim == 0:
        raise ValueError("signal must have a time axis")

    sample_count = signal.shape[0]
    output_dtype = (
        signal.dtype if np.issubdtype(signal.dtype, np.inexact)
        else np.result_type(signal.dtype, kernel.dtype, np.float64)
    )
    if np.iscomplexobj(kernel) and not np.iscomplexobj(signal):
        output_dtype = np.result_type(output_dtype, np.complex64)
    output = np.empty((output_length,) + signal.shape[1:], dtype=output_dtype)
    if sample_count == 0 or signal.size == 0:
        return output

    full_length = sample_count + kernel.size - 1
    fft_length = next_fast_len(full_length)
    flat_signal = signal.reshape(sample_count, -1)
    flat_output = output.reshape(output_length, -1)

    real_itemsize = np.empty((), dtype=output.dtype).real.dtype.itemsize
    complex_dtype = np.complex64 if real_itemsize <= 4 else np.complex128
    use_real_fft = not np.iscomplexobj(signal) and not np.iscomplexobj(kernel)
    if use_real_fft:
        real_dtype = np.float32 if real_itemsize <= 4 else np.float64
        kernel_spectrum = rfft(
            kernel.astype(real_dtype, copy=False),
            n=fft_length,
        )
    else:
        kernel_spectrum = fft(
            kernel.astype(complex_dtype, copy=False),
            n=fft_length,
        )

    # Bound the padded transform workspace while retaining batched FFTs. For
    # the real path, one half-spectrum plus one real inverse-transform output
    # has the same combined byte count as one full complex spectrum.
    workspace_bytes = 16 * 1024**2
    columns_per_block = max(
        1,
        workspace_bytes // (fft_length * np.dtype(complex_dtype).itemsize),
    )
    stop = output_start + output_length
    for first_column in range(0, flat_signal.shape[1], columns_per_block):
        last_column = min(
            first_column + columns_per_block,
            flat_signal.shape[1],
        )
        transform = rfft if use_real_fft else fft
        spectrum = transform(
            flat_signal[:, first_column:last_column],
            n=fft_length,
            axis=0,
        )
        spectrum *= kernel_spectrum[:, np.newaxis]
        if use_real_fft:
            filtered = irfft(
                spectrum,
                n=fft_length,
                axis=0,
                overwrite_x=True,
            )[output_start:stop]
        else:
            filtered = ifft(
                spectrum,
                axis=0,
                overwrite_x=True,
            )[output_start:stop]
        if not np.issubdtype(output.dtype, np.complexfloating):
            filtered = filtered.real
        flat_output[:, first_column:last_column] = filtered
    return output


def temporal_convolve_signal_transpose(
    signal: np.ndarray,
    kernel: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Apply the exact Hermitian adjoint of `temporal_convolve_signal()`.

    Apply valid convolution with the conjugate-reversed kernel, returning
    N = signal.shape[0] - kernel.size + 1 samples; N must be positive. For real
    kernels this is also the ordinary transpose. Integer and boolean signals
    are promoted to floating point.
    """
    signal = np.asarray(signal)
    kernel = np.asarray(kernel)
    if kernel.ndim != 1:
        raise ValueError("kernel must be one-dimensional")
    if (
        signal.ndim == 0 or signal.shape[0] == 0 or signal.shape[0] < kernel.size
    ):
        raise ValueError("full transpose requires at least kernel.size time samples")
    return _temporal_fft_convolve(
        signal,
        np.conjugate(kernel[::-1]),
        output_start=kernel.size - 1,
        output_length=signal.shape[0] - kernel.size + 1,
    ) * dt


def auto_pixel_chunk(
    n_pixels: int,
    *,
    nproc: Optional[int] = None,
    target_chunks_per_rank: int = 8,
    min_pixel_chunk: int = 16,
    max_pixel_chunk: int = 128,
) -> int:
    """Choose the legacy design-pixel block size retained for compatibility.

    Args:
        n_pixels: Total number of design pixels.
        nproc: MPI rank count. Defaults to `mp.count_processors()`.
        target_chunks_per_rank: Target number of chunks per rank used by the
            sizing heuristic.
        min_pixel_chunk: Lower bound for the returned chunk size.
        max_pixel_chunk: Upper bound for the returned chunk size.

    Returns:
        Positive pixel chunk size clipped to the requested bounds.
    """
    if n_pixels <= 0:
        raise ValueError("n_pixels must be positive")
    if nproc is None:
        nproc = mp.count_processors()
    if nproc <= 0:
        raise ValueError("nproc must be positive")
    if target_chunks_per_rank <= 0:
        raise ValueError("target_chunks_per_rank must be positive")
    if min_pixel_chunk <= 0 or max_pixel_chunk <= 0:
        raise ValueError("pixel chunk limits must be positive")
    if min_pixel_chunk > max_pixel_chunk:
        raise ValueError("min_pixel_chunk must be <= max_pixel_chunk")

    chunk = int(np.ceil(n_pixels / (nproc * target_chunks_per_rank)))
    return int(np.clip(chunk, min_pixel_chunk, max_pixel_chunk))


class MultiTDAObjective:
    """Multi-band time-domain adjoint objective using temporal convolution.

    The class evaluates several wavelength bands from one broadband time-domain
    run by filtering monitor histories with bandpass kernels. Each filtered
    band history may use a custom FoM and adjoint signal. The resulting band
    objectives are combined by a user-provided scalarization function.

    Sign convention: `evaluate` / `fom_and_grad` return
    `(total_fom, d total_fom / d rho)` with the maximization sign, matching
    `TDAObjective` and Meep adjoint's `OptimizationProblem`. Negate both in
    the optimizer callback when driving a minimizer such as nlopt or
    `scipy.optimize.minimize`.
    """

    def __init__(
        self,
        *,
        update_design: Optional[Callable[[np.ndarray], None]] = None,
        sim_factory: Optional[Callable[..., mp.Simulation]] = None,
        t_final: Optional[float] = None,
        monitor_positions: Optional[Sequence[mp.Vector3]] = None,
        component: Optional[int] = None,
        wavelength_bands: Optional[Sequence[Optional[tuple[float, float]]]] = None,
        weights: Optional[Sequence[float]] = None,
        kernel_length: Optional[int] = None,
        kernel_window: Optional[str] = "hamming",
        kernel_window_params=None,
        filter_bank: str = "independent",
        transition_width: Optional[float] = None,
        pixel_chunk="auto",
        target_chunks_per_rank: int = 8,
        min_pixel_chunk: int = 16,
        max_pixel_chunk: int = 128,
        adjoint_source_size: Optional[mp.Vector3] = None,
        adjoint_source_amplitude: Optional[complex] = None,
        dt: Optional[float] = None,
        resolution: Optional[float] = None,
        fom_fn: Optional[
            Union[
                Callable[[np.ndarray, float], float],
                Sequence[
                    Optional[Callable[[np.ndarray, float], float]]
                ],
            ]
        ] = None,
        adjoint_signal_fn: Optional[
            Union[
                Callable[[np.ndarray, float], np.ndarray],
                Sequence[
                    Optional[
                        Callable[[np.ndarray, float], np.ndarray]
                    ]
                ],
            ]
        ] = None,
        scalarization_fn: Optional[Callable[[np.ndarray], object]] = None,
        history_dtype=np.complex128,
        target_history_block_size: Optional[int] = None,
        design: Optional[DesignGrid] = None,
        designs: Optional[Sequence[DesignGrid]] = None,
        simulation: Optional[SimulationSpec] = None,
        targets: Optional[
            Sequence[
                Union[
                    PointTarget,
                    FieldRegionTarget,
                    FluxTarget,
                    EigenmodeCoefficientTarget,
                    Near2FarTarget,
                ]
            ]
        ] = None,
        chunk_balancer: Union[
            str,
            AdaptiveAdjointChunkBalancer,
            None,
        ] = "auto",
        sampling_interval: int = 1,
        max_frequency: Optional[float] = None,
        reuse_simulation: bool = False,
        reconstruction_window: Optional[str] = "kaiser",
        reconstruction_window_params=None,
        reconstruction_half_width: int = 64,
    ) -> None:
        """Create a multi-band temporal-convolution adjoint problem.

        Args:
            update_design: Function that writes the flat design vector `x` into
                the Meep design object, usually a `MaterialGrid`.
            sim_factory: Function returning a Meep `Simulation`. It is called
                with no arguments for the forward run and with a source list for
                the adjoint run. Arbitrary factories use ordinary Cartesian
                `mp.Source` injection and do not support cylindrical gradient
                evaluation.
            t_final: Positive finite physical forward recording duration.
                No filter tail is added to the FDTD run. The duration must
                independently capture the physical response; sparse-history
                alignment may extend it to the next stored time step.
            monitor_positions: Point-monitor positions. The current API expects
                one monitor position per wavelength band. For cylindrical
                gradients, targets may lie on or near `r=0`; on-axis
                components must satisfy modal regularity. This may override
                `PointTarget` positions, but cannot be combined with a
                regional, flux, or eigenmode target.
            component: Meep field component to monitor, inject, and sample, e.g.
                `mp.Ez` or `mp.Er`. Cylindrical gradients accept
                `Er/Ep/Ez/Hr/Hp/Hz`. This may override `PointTarget`
                components, but cannot be combined with a regional, flux, or
                eigenmode target.
            wavelength_bands: Wavelength intervals `(lambda_min, lambda_max)`.
                Each interval defines one bandpass temporal-convolution kernel.
                Mixed target lists require one entry per target, with `None`
                for each `Near2FarTarget`.
            weights: One-dimensional finite per-band amplitude weights applied
                to monitor filtering and the matching adjoint filters. Its
                length must match `wavelength_bands`. The
                `power_complementary` filter bank requires unit weights;
                express unequal objective priorities in `scalarization_fn`.
                Near-to-far entries require weight `1.0` and have no FIR kernel.
            kernel_length: Positive integer number of time samples in each
                bandpass kernel.
            kernel_window: NumPy window used to taper each temporal-convolution
                bandpass kernel. Supported values are `rectangular`/`None`,
                `hamming`, `hann`, `blackman`, `bartlett`, and `kaiser`;
                `none`, `boxcar`, and `hanning` are accepted aliases. This
                setting applies only to `filter_bank="independent"`.
            kernel_window_params: Optional parameters for `kernel_window`.
                Currently only `kaiser` uses this, with
                `{"beta": value}`. This setting applies only to
                `filter_bank="independent"`.
            filter_bank: Temporal-filter construction. `"independent"`
                retains the per-band windowed-sinc filters.
                `"power_complementary"` constructs coordinated complementary
                target responses and fits each FIR on a shared frequency grid
                for ordered, contiguous bands.
            transition_width: Total frequency width of each crossover in a
                power-complementary bank. The outer tapers use the same width
                outside the requested band union. `None` selects
                `2 / (kernel_length * dt)`. It is not used by independent
                filters.
            pixel_chunk: Legacy compatibility option retained for existing
                callers. Dense forward-field derivatives are computed one
                time row at a time and no longer use pixel blocks.
            target_chunks_per_rank: Legacy compatibility option used only to
                resolve `pixel_chunk="auto"`.
            min_pixel_chunk: Legacy lower bound for automatic pixel chunk size.
            max_pixel_chunk: Legacy upper bound for automatic pixel chunk size.
            adjoint_source_size: Meep source size for each adjoint monitor
                source. With direct `SimulationSpec.make`, each default
                zero-size Cartesian target uses its exact indexed monitor
                transpose. Explicit nonzero Cartesian sizes use ordinary
                `mp.Source` injection. Cylindrical gradients require zero
                size. This may override `PointTarget` sizes, but cannot be
                combined with a regional, flux, or eigenmode target.
            adjoint_source_amplitude: Logical source amplitude for each
                adjoint monitor source. Gradient evaluation requires `1.0`;
                finite-source density normalization is applied internally.
                Other finite values are accepted only for value-only
                evaluation. This may override
                `PointTarget` amplitudes, but cannot be combined with a
                regional, flux, or eigenmode target.
            dt: Optional legacy expected time step. The created simulation's
                `fields.dt` is authoritative; when supplied, `dt` must
                match it. It is used only as a fallback for simulation doubles
                that do not expose `fields.dt`.
            resolution: Meep resolution used with the simulation's Courant
                factor to infer a fallback time step when `dt` is omitted and
                the created simulation does not expose `fields.dt`.
            fom_fn: Optional scalar objective applied independently to each
                filtered band history as `fom_fn(filtered_history, dt)`.
                Supply one callable for all bands or one optional callable per
                band. A per-band `None` selects that target's default FoM.
                Each callable must use autograd-compatible operations when the
                matching `adjoint_signal_fn` is omitted.
            adjoint_signal_fn: Optional manual continuous-time bilinear
                covector provider applied independently to each band. For
                filtered history `u` and returned `q`, the convention is
                `delta FoM = dt * Re(sum(q * delta u))`. Supply one callable
                for all bands or one optional callable per band. Every supplied
                callback requires a matching `fom_fn` and its output must match
                the filtered history shape.
            scalarization_fn: Function that maps band FoMs to a scalar FoM.
                An autograd-compatible function may return the scalar directly;
                its band coefficients are differentiated automatically.
                Existing `(total_fom, band_coeffs)` and
                `(total_fom, band_coeffs, info)` returns provide manual
                derivatives and optional diagnostics.
            history_dtype: Real or complex floating dtype requested for
                temporary field histories. Its real counterpart is used
                automatically when Meep initializes real forward fields;
                complex forward fields require a complex dtype.
            target_history_block_size: Optional positive spatial-column block
                size for the MPI-distributed built-in target-history path.
                This opt-in path applies only to direct Cartesian
                `SimulationSpec` runs whose targets are either all
                `FieldRegionTarget` or all `FluxTarget` and whose band
                callbacks are all `None`. Uniquely supported monitor channels
                are stored only on their supporting rank; chunk-boundary and
                mixed-owner flux channels are replicated. Custom callbacks and
                all other target combinations retain the legacy full-history
                path.
            design: Optional `DesignGrid` bundle. It is required for gradient
                evaluation and supplies the native design-grid sampling, update,
                and material factor.
            simulation: Optional `SimulationSpec` bundle. When supplied, it
                fills `sim_factory` and `resolution` unless those are
                explicitly supplied. Its direct `make` path enables exact
                indexed injection independently for zero-size Cartesian
                targets. It is required for cylindrical gradients and supplies
                the one forward angular mode `m` shared by all wavelength
                bands.
            targets: Optional sequence containing one `PointTarget`,
                `FieldRegionTarget`, `FluxTarget`, or
                `EigenmodeCoefficientTarget` per wavelength band.
                `Near2FarTarget` entries may be interleaved. Their callbacks
                receive `(points, frequencies, 6)` complex far fields. Callback
                lists and scalarization inputs follow the full target order.
                Regional target histories use time as the first axis and the
                configured `sample_shape` as the remaining axes. Flux
                targets expose a real signed-power history with one fewer time
                sample because magnetic fields are centered onto the
                electric-field time grid. Flux gradients combine electric and
                causally time-centered magnetic sources in one adjoint run.
                Eigenmode targets expose a complex time-domain coefficient
                obtained from a fixed, power-normalized reference mode and use
                the same combined-adjoint structure.
            chunk_balancer: `"auto"` enables source-safe Meep-native adaptive
                balancing for a direct `SimulationSpec` whose `chunk_layout`
                is unset. `None` opts out, and an
                `AdaptiveAdjointChunkBalancer` supplies custom settings.
                Near-to-far targets disable automatic balancing and reject
                explicit adaptive balancing.
                Ordinary forward sources are protected automatically.
                Workload calibration uses three to eight gradient evaluations
                and freezes the best measured layout after the critical-rank
                work converges.
                Exact indexed targets do not constrain chunk boundaries.
                `source_boundary_mode="finite"` selects ordinary finite
                Cartesian sources for eligible `PointTarget` entries and is
                incompatible with exact regional, flux, and eigenmode targets.
            sampling_interval: Dimensionless number of Meep time steps between
                stored design-grid forward-field samples. Point monitors,
                temporal filters, and adjoint gradient accumulation remain
                full rate.
                Values above one may extend the forward run by less than one
                sparse interval to align its closing sample.
            max_frequency: Optional maximum relevant frequency `f_max` used
                to choose `sampling_interval` from the Nyquist criterion.
                It must cover the highest wavelength-band frequency. Also
                include the source, objective, adjoint response, and any safety
                margin. This does not verify the remaining band-limit or
                temporal-boundary assumptions. Cannot be combined with
                `sampling_interval > 1`.
            reuse_simulation: Reuse the initialized forward `Simulation` for
                the adjoint run. This requires `simulation=SimulationSpec(...)`
                and should only be enabled when the simulation state can be
                safely restarted, such as for time-invariant media. Cylindrical
                reuse changes the mode from `m` to `-m` before the adjoint
                run.
            reconstruction_window: Window applied to the finite sinc used to
                reconstruct sparse forward fields. Supported values are
                `rectangular`/`None`, `hamming`, `hann`, `blackman`,
                `bartlett`, and `kaiser`; `none`, `boxcar`, and `hanning` are
                accepted aliases.
            reconstruction_window_params: Optional reconstruction-window
                parameters. Only `kaiser` accepts `{"beta": value}`;
                its default beta is 6.
            reconstruction_half_width: Number of sparse samples on each side
                of the finite sinc support. Default is 64.
        """
        self._near2far_objective = None
        self._near2far_targets = ()
        if targets is not None:
            targets = tuple(targets)
        if targets and all(isinstance(target, Near2FarTarget) for target in targets):
            if any(value is not None for value in
                   (wavelength_bands, weights, kernel_length, transition_width,
                    kernel_window_params, target_history_block_size)) or (
                        filter_bank != "independent" or kernel_window != "hamming"
                    ):
                raise ValueError(
                    "Near2FarTarget uses discrete frequencies, not PC-FIR "
                    "wavelength_bands, weights, kernel settings, or temporal history settings"
                )
            from .tda_objective import TDAObjective

            fns = _normalize_band_callbacks(fom_fn, len(targets), "fom_fn")
            signals = _normalize_band_callbacks(adjoint_signal_fn, len(targets), "adjoint_signal_fn")
            _validate_band_callback_pairs(fns, signals)
            core = TDAObjective(
                update_design=update_design, sim_factory=sim_factory, t_final=t_final,
                monitor_position=monitor_positions, component=component,
                adjoint_source_size=adjoint_source_size,
                adjoint_source_amplitude=adjoint_source_amplitude,
                design=design, designs=designs, simulation=simulation, target=targets[0],
                dt=dt, resolution=resolution, history_dtype=history_dtype,
                chunk_balancer=chunk_balancer, sampling_interval=sampling_interval,
                max_frequency=max_frequency, reuse_simulation=reuse_simulation,
                reconstruction_window=reconstruction_window,
                reconstruction_window_params=reconstruction_window_params,
                reconstruction_half_width=reconstruction_half_width,
            )
            self._near2far_objective = core
            self.targets = targets
            self.design, self.designs = core.design, core.designs
            self.gradient_components = core.gradient_components
            self.chunk_balancer = core.chunk_balancer
            self.reuse_simulation = core.reuse_simulation
            self._simulation_spec = simulation
            self.update_design = core.update_design
            self._explicit_dt = None if dt is None else float(dt)
            for name in _NEAR2FAR_SAMPLING_ATTRIBUTES:
                setattr(self, name, getattr(core, name))
            self.fom_fn, self.adjoint_signal_fn = fom_fn, adjoint_signal_fn
            self.scalarization_fn = scalarization_fn or self._weighted_sum_scalarization
            self.last_far_fields = None
            self.last_band_objectives = self.last_band_losses = self.last_band_coeffs = None
            self.last_scalarization_info = self.last_smooth_min = self.last_total_fom = None
            self.last_sampling_interval = None
            self.last_actual_time = None
            return

        if targets and any(isinstance(target, Near2FarTarget) for target in targets):
            self.targets = targets
            self._near2far_indices = tuple(i for i, target in enumerate(targets)
                                          if isinstance(target, Near2FarTarget))
            self._temporal_indices = tuple(i for i, target in enumerate(targets)
                                          if not isinstance(target, Near2FarTarget))
            self._near2far_targets = tuple(targets[i] for i in self._near2far_indices)
            if (not isinstance(simulation, SimulationSpec)
                    or (sim_factory is not None
                        and getattr(sim_factory, "__self__", None) is not simulation)):
                raise ValueError("Near2FarTarget requires direct simulation=SimulationSpec(...)")
            if chunk_balancer == "auto":
                chunk_balancer = None
            if chunk_balancer is not None:
                raise ValueError("Near2FarTarget does not support adaptive chunk balancing")
            if (wavelength_bands is None or len(wavelength_bands) != len(targets)
                    or any(wavelength_bands[i] is not None for i in self._near2far_indices)):
                raise ValueError("mixed wavelength_bands must match targets with None for Near2FarTarget")
            if weights is None or len(weights) != len(targets):
                raise ValueError("mixed weights must match targets")
            if any(weights[i] != 1 for i in self._near2far_indices):
                raise ValueError("Near2FarTarget requires unit filter weights; use scalarization_fn")
            if any(value is not None for value in (monitor_positions, component,
                                                  adjoint_source_size, adjoint_source_amplitude)):
                raise ValueError("mixed near-to-far targets cannot use legacy target arguments")
            wavelength_bands = [wavelength_bands[i] for i in self._temporal_indices]
            weights = [weights[i] for i in self._temporal_indices]
            targets = tuple(targets[i] for i in self._temporal_indices)
            self.last_far_fields = None
            self.last_actual_time = None

        uses_simulation_spec = simulation is not None and (
            sim_factory is None or getattr(sim_factory, "__self__", None) is simulation
        )
        uses_concrete_simulation_spec = uses_simulation_spec and isinstance(
            simulation,
            SimulationSpec,
        )
        resolved_designs = _resolve_designs(design, designs)
        design = (
            _DesignCollection(resolved_designs) if len(resolved_designs) > 1
            else next(iter(resolved_designs), None)
        )
        if design is not None:
            update_design = update_design if update_design is not None else design.update_weights
        if simulation is not None:
            sim_factory = sim_factory if sim_factory is not None else simulation.make
            resolution = resolution if resolution is not None else simulation.resolution
        if dt is not None:
            dt = _positive_finite_float(dt, "dt")
        if resolution is not None:
            resolution = _positive_finite_float(resolution, "resolution")
        is_cylindrical = (
            _is_cylindrical_simulation(simulation)
            if simulation is not None
            else bool(getattr(design, "is_cylindrical", False))
        )
        has_coordinate_system_metadata = (
            simulation is not None or design is not None
        )
        if wavelength_bands is not None:
            wavelength_bands = _validate_wavelength_bands(wavelength_bands)
        has_regional_targets = False
        has_flux_targets = False
        has_eigenmode_targets = False
        target_positions = None
        target_sample_shapes = None
        target_spatial_weights = None
        target_is_regional = None
        target_is_flux = None
        target_is_eigenmode = None
        if targets is not None:
            targets = tuple(targets)
            if not targets:
                raise ValueError("targets must not be empty")
            if (
                wavelength_bands is not None
                and len(targets) != len(wavelength_bands)
            ):
                raise ValueError(
                    "targets must match the number of wavelength bands"
                )
            if any(
                not isinstance(
                    target,
                    (
                        PointTarget,
                        FieldRegionTarget,
                        FluxTarget,
                        EigenmodeCoefficientTarget,
                    ),
                )
                for target in targets
            ):
                raise TypeError(
                    "targets must contain PointTarget, FieldRegionTarget, "
                    "FluxTarget, or EigenmodeCoefficientTarget"
                )
            has_regional_targets = any(
                isinstance(target, FieldRegionTarget)
                for target in targets
            )
            has_flux_targets = any(
                isinstance(target, FluxTarget)
                for target in targets
            )
            has_eigenmode_targets = any(
                isinstance(target, EigenmodeCoefficientTarget)
                for target in targets
            )
            has_exact_surface_targets = (
                has_flux_targets or has_eigenmode_targets
            )
            if has_regional_targets or has_exact_surface_targets:
                conflicting = [
                    name
                    for name, value in (
                        ("monitor_positions", monitor_positions),
                        ("component", component),
                        ("adjoint_source_size", adjoint_source_size),
                        (
                            "adjoint_source_amplitude",
                            adjoint_source_amplitude,
                        ),
                    )
                    if value is not None
                ]
                if conflicting:
                    raise ValueError(
                        "exact regional, flux, and eigenmode targets cannot "
                        "be combined "
                        "with legacy "
                        "target arguments: " + ", ".join(conflicting)
                    )
            point_positions = (
                tuple(monitor_positions)
                if monitor_positions is not None
                else tuple(
                    target.position
                    for target in targets
                )
            ) if not (
                has_regional_targets or has_exact_surface_targets
            ) else None
            if (
                point_positions is not None
                and len(point_positions) != len(targets)
            ):
                raise ValueError(
                    "monitor_positions must match the number of targets"
                )
            target_positions = tuple(
                (
                    target.positions
                    if isinstance(
                        target,
                        (
                            FieldRegionTarget,
                            FluxTarget,
                            EigenmodeCoefficientTarget,
                        ),
                    )
                    else (
                        point_positions[target_index]
                        if point_positions is not None
                        else target.position,
                    )
                )
                for target_index, target in enumerate(targets)
            )
            target_sample_shapes = tuple(
                (
                    target.sample_shape
                    if isinstance(target, FieldRegionTarget)
                    else ()
                )
                for target in targets
            )
            target_is_regional = tuple(
                isinstance(target, FieldRegionTarget)
                for target in targets
            )
            target_is_flux = tuple(
                isinstance(target, FluxTarget)
                for target in targets
            )
            target_is_eigenmode = tuple(
                isinstance(target, EigenmodeCoefficientTarget)
                for target in targets
            )
            target_normals = tuple(
                (
                    target.normal
                    if isinstance(
                        target,
                        (FluxTarget, EigenmodeCoefficientTarget),
                    )
                    else None
                )
                for target in targets
            )
            target_spatial_weights = tuple(
                np.asarray(
                    (
                        target.spatial_weights
                        if isinstance(
                            target,
                            (
                                FieldRegionTarget,
                                FluxTarget,
                                EigenmodeCoefficientTarget,
                            ),
                        )
                        else (1.0,)
                    ),
                    dtype=float,
                )
                for target in targets
            )
            if is_cylindrical:
                for target in targets:
                    if isinstance(
                        target,
                        (FluxTarget, EigenmodeCoefficientTarget),
                    ):
                        _validate_cylindrical_surface(
                            target.positions,
                            target.normal,
                            target.spatial_weights,
                        )
            monitor_positions = [
                positions[0] if len(positions) == 1 else positions
                for positions in target_positions
            ]
            target_components = tuple(
                (
                    None
                    if isinstance(
                        target,
                        (FluxTarget, EigenmodeCoefficientTarget),
                    )
                    else (
                        component
                        if component is not None
                        else target.component
                    )
                )
                for target in targets
            )
            adjoint_source_sizes = tuple(
                adjoint_source_size
                if adjoint_source_size is not None
                else (
                    (
                        target.adjoint_source_size
                        if target.adjoint_source_size is not None
                        else mp.Vector3()
                    )
                    if isinstance(target, PointTarget)
                    else mp.Vector3()
                )
                for target in targets
            )
            adjoint_source_amplitudes = tuple(
                complex(
                    adjoint_source_amplitude
                    if adjoint_source_amplitude is not None
                    else (
                        target.adjoint_source_amplitude
                        if not isinstance(
                            target,
                            (FluxTarget, EigenmodeCoefficientTarget),
                        )
                        else 1.0
                    )
                )
                for target in targets
            )
        else:
            target_components = None
            adjoint_source_sizes = None
            adjoint_source_amplitudes = None

        missing = [
            name for name, value in (
                ("update_design", update_design),
                ("sim_factory", sim_factory),
                ("t_final", t_final),
                ("monitor_positions", monitor_positions),
                (
                    "component",
                    component if target_components is None else target_components,
                ),
                ("wavelength_bands", wavelength_bands),
                ("weights", weights),
                ("kernel_length", kernel_length),
            )
            if value is None
        ]
        if missing:
            raise ValueError("MultiTDAObjective missing required inputs: " + ", ".join(missing))
        t_final = _positive_finite_float(t_final, "t_final")
        if (
            isinstance(kernel_length, (bool, np.bool_))
            or not isinstance(kernel_length, (int, np.integer))
            or kernel_length < 1
        ):
            raise ValueError("kernel_length must be a positive integer")
        kernel_length = int(kernel_length)
        if not isinstance(filter_bank, str) or filter_bank not in (
            "independent",
            "power_complementary",
        ):
            raise ValueError(
                "filter_bank must be 'independent' or 'power_complementary'"
            )
        if transition_width is not None:
            if isinstance(transition_width, (bool, np.bool_)):
                raise ValueError(
                    "transition_width must be a positive finite number"
                )
            transition_width = _positive_finite_float(
                transition_width,
                "transition_width",
            )
        if filter_bank == "independent" and transition_width is not None:
            raise ValueError(
                "transition_width requires filter_bank='power_complementary'"
            )
        if filter_bank == "power_complementary":
            _validate_power_complementary_bands(wavelength_bands)
            if kernel_length % 2 == 0:
                raise ValueError(
                    "power_complementary filter_bank requires an odd "
                    "kernel_length"
                )
            if kernel_window not in (None, "hamming") or kernel_window_params:
                raise ValueError(
                    "kernel_window and kernel_window_params do not apply to "
                    "the power_complementary filter_bank"
                )
        wavelength_bands = list(wavelength_bands)
        normalized_fom_fns = _normalize_band_callbacks(
            fom_fn,
            len(self.targets) if self._near2far_targets else len(wavelength_bands),
            "fom_fn",
        )
        normalized_adjoint_signal_fns = _normalize_band_callbacks(
            adjoint_signal_fn,
            len(self.targets) if self._near2far_targets else len(wavelength_bands),
            "adjoint_signal_fn",
        )
        _validate_band_callback_pairs(
            normalized_fom_fns,
            normalized_adjoint_signal_fns,
        )
        if (
            isinstance(sampling_interval, (bool, np.bool_))
            or not isinstance(sampling_interval, (int, np.integer))
            or sampling_interval < 1
        ):
            raise ValueError("sampling_interval must be a positive integer")
        sampling_interval = int(sampling_interval)
        if target_history_block_size is not None:
            if (
                isinstance(target_history_block_size, (bool, np.bool_))
                or not isinstance(
                    target_history_block_size,
                    (int, np.integer),
                )
                or target_history_block_size < 1
            ):
                raise ValueError(
                    "target_history_block_size must be a positive integer or None"
                )
            target_history_block_size = int(target_history_block_size)
        if max_frequency is not None:
            max_frequency = _positive_finite_float(
                max_frequency,
                "max_frequency",
            )
            if sampling_interval != 1:
                raise ValueError(
                    "max_frequency cannot be combined with sampling_interval > 1"
                )
            highest_band_frequency = max(
                (
                    1.0 / float(lambda_min)
                    for lambda_min, _ in wavelength_bands
                ),
                default=None,
            )
            if (
                highest_band_frequency is not None
                and max_frequency < highest_band_frequency
            ):
                raise ValueError(
                    "max_frequency must be at least "
                    f"{highest_band_frequency:g} to cover the highest "
                    "wavelength-band frequency"
                )
            if (self._near2far_targets and max_frequency < max(
                    f for target in self._near2far_targets for f in target.frequencies)):
                raise ValueError("max_frequency must cover every near-to-far target frequency")
        (
            reconstruction_window,
            reconstruction_window_params,
            reconstruction_half_width,
        ) = _validate_reconstruction_settings(
            reconstruction_window,
            reconstruction_window_params,
            reconstruction_half_width,
        )
        if not isinstance(reuse_simulation, (bool, np.bool_)):
            raise ValueError("reuse_simulation must be a boolean")
        reuse_simulation = bool(reuse_simulation)
        if reuse_simulation and not uses_simulation_spec:
            raise ValueError("reuse_simulation=True requires SimulationSpec.make")
        if isinstance(chunk_balancer, str):
            if chunk_balancer != "auto":
                raise ValueError(
                    "chunk_balancer must be 'auto', None, or an adaptive balancer"
                )
            chunk_balancer = (
                AdaptiveAdjointChunkBalancer()
                if (
                    uses_concrete_simulation_spec
                    and simulation.chunk_layout is None
                    and not simulation.symmetries
                )
                else None
            )
        if chunk_balancer is not None and getattr(simulation, "symmetries", ()):
            raise ValueError("Mirror simulations do not support adaptive chunk balancing")
        if chunk_balancer is not None and not uses_simulation_spec:
            raise ValueError("adaptive chunk balancing requires SimulationSpec.make")
        if (
            (has_flux_targets or has_eigenmode_targets)
            and chunk_balancer is not None
            and getattr(chunk_balancer, "source_boundary_mode", "auto")
            == "finite"
        ):
            raise ValueError(
                "FluxTarget and EigenmodeCoefficientTarget require exact "
                "indexed adjoint sources and cannot "
                "use source_boundary_mode='finite'"
            )

        self.update_design = update_design
        self.sim_factory = sim_factory
        self.t_final = t_final
        self._explicit_dt = None if dt is None else float(dt)
        self.resolution = resolution
        self._fallback_courant = (
            float(getattr(simulation, "courant", 0.5))
            if simulation is not None
            else 0.5
        )
        self.sampling_interval = sampling_interval
        self.max_frequency = max_frequency
        self.last_sampling_interval = None
        self.reconstruction_window = reconstruction_window
        self.reconstruction_window_params = reconstruction_window_params
        self.reconstruction_half_width = reconstruction_half_width
        self.kernel_length = kernel_length
        self.filter_bank = filter_bank
        self.transition_width = transition_width
        self.wavelength_bands = wavelength_bands
        self.monitor_positions = list(monitor_positions)
        if targets is not None and len(targets) != len(self.wavelength_bands):
            raise ValueError("targets must match the number of wavelength bands")
        if target_components is None:
            target_components = (component,) * len(self.monitor_positions)
            source_size = (
                adjoint_source_size
                if adjoint_source_size is not None
                else mp.Vector3()
            )
            adjoint_source_sizes = (source_size,) * len(self.monitor_positions)
            source_amplitude = (
                1.0
                if adjoint_source_amplitude is None
                else adjoint_source_amplitude
            )
            adjoint_source_amplitudes = (
                complex(source_amplitude),
            ) * len(self.monitor_positions)
            target_positions = tuple(
                (position,) for position in self.monitor_positions
            )
            target_sample_shapes = tuple(
                () for _ in self.monitor_positions
            )
            target_spatial_weights = tuple(
                np.ones(1, dtype=float)
                for _ in self.monitor_positions
            )
            target_is_regional = tuple(
                False for _ in self.monitor_positions
            )
            target_is_flux = tuple(
                False for _ in self.monitor_positions
            )
            target_is_eigenmode = tuple(
                False for _ in self.monitor_positions
            )
            target_normals = tuple(
                None for _ in self.monitor_positions
            )
        self.target_components = tuple(target_components)
        self.adjoint_source_sizes = tuple(adjoint_source_sizes)
        self.adjoint_source_amplitudes = tuple(adjoint_source_amplitudes)
        self._has_regional_targets = has_regional_targets
        self._has_flux_targets = has_flux_targets
        self._has_eigenmode_targets = has_eigenmode_targets
        self._regional_target_mask = tuple(target_is_regional)
        self._flux_target_mask = tuple(target_is_flux)
        self._eigenmode_target_mask = tuple(target_is_eigenmode)
        self._mixed_surface_target_mask = tuple(
            flux or eigenmode
            for flux, eigenmode in zip(
                self._flux_target_mask,
                self._eigenmode_target_mask,
            )
        )
        self._eigenmode_targets = tuple(
            (
                target
                if isinstance(target, EigenmodeCoefficientTarget)
                else None
            )
            for target in (
                targets
                if targets is not None
                else (None,) * len(self.monitor_positions)
            )
        )
        self._reference_mode_fields = tuple(
            None for _ in self.monitor_positions
        )
        self.target_normals = tuple(target_normals)
        self.target_positions = tuple(
            tuple(positions) for positions in target_positions
        )
        self.target_sample_shapes = tuple(target_sample_shapes)
        self.target_spatial_weights = tuple(
            np.asarray(spatial_weights, dtype=float).reshape(weight_shape)
            for spatial_weights, weight_shape in zip(
                target_spatial_weights,
                (
                    (
                        (len(positions),)
                        if is_flux or is_eigenmode
                        else (sample_shape if sample_shape else (1,))
                    )
                    for (
                        positions,
                        sample_shape,
                        is_flux,
                        is_eigenmode,
                    ) in zip(
                        self.target_positions,
                        self.target_sample_shapes,
                        self._flux_target_mask,
                        self._eigenmode_target_mask,
                    )
                ),
            )
        )
        self.design = design
        self.designs = resolved_designs
        self._simulation_spec = simulation
        self._configured_simulation_m = (
            int(simulation.m) if uses_simulation_spec else None
        )
        self._is_cylindrical = is_cylindrical
        self._has_coordinate_system_metadata = (
            has_coordinate_system_metadata
        )
        self._inferred_is_cylindrical = None
        self._configure_target_channel_layout()
        self.component = (
            self.target_components[0]
            if len(set(self.target_components)) == 1
            else None
        )
        source_size_keys = {
            (source_size.x, source_size.y, source_size.z)
            for source_size in self.adjoint_source_sizes
        }
        self.adjoint_source_size = (
            self.adjoint_source_sizes[0]
            if len(source_size_keys) == 1
            else None
        )
        self.adjoint_source_amplitude = (
            self.adjoint_source_amplitudes[0]
            if len(set(self.adjoint_source_amplitudes)) == 1
            else None
        )
        self.weights = np.asarray(weights, dtype=float)
        if self.weights.ndim != 1:
            raise ValueError("weights must be a one-dimensional sequence")
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("weights must contain only finite values")
        if filter_bank == "power_complementary" and not np.array_equal(
            self.weights,
            np.ones_like(self.weights),
        ):
            raise ValueError(
                "power_complementary filter_bank requires unit filter "
                "weights; apply objective weights in scalarization_fn"
            )
        self.kernel_window = kernel_window
        self.kernel_window_params = dict(kernel_window_params or {})
        design_variable_count = int(np.prod(design.shape)) if design is not None else 1
        self.pixel_chunk = self._resolve_pixel_chunk(
            pixel_chunk,
            design_variable_count=design_variable_count,
            target_chunks_per_rank=target_chunks_per_rank,
            min_pixel_chunk=min_pixel_chunk,
            max_pixel_chunk=max_pixel_chunk,
        )
        self.fom_fn = fom_fn
        self.adjoint_signal_fn = adjoint_signal_fn
        self.scalarization_fn = scalarization_fn if scalarization_fn is not None else self._weighted_sum_scalarization
        self.history_dtype = history_dtype
        self.target_history_block_size = target_history_block_size
        self.chunk_balancer = chunk_balancer
        self._uses_simulation_spec = uses_simulation_spec
        self._uses_concrete_simulation_spec = uses_concrete_simulation_spec
        if (
            (self._has_flux_targets or self._has_eigenmode_targets)
            and not self._is_cylindrical
            and design is not None
            and design.dimensions == 2
            and any(
                abs(float(normal.z)) > 0.0
                for normal, is_surface in zip(
                    self.target_normals,
                    self._mixed_surface_target_mask,
                )
                if is_surface
            )
        ):
            raise ValueError(
                "2D flux and eigenmode target normals must lie in the x-y plane"
            )
        if simulation is not None and design is not None and (
            bool(getattr(design, "is_cylindrical", False))
            != self._is_cylindrical
        ):
            raise ValueError(
                "DesignGrid coordinate system must match SimulationSpec dimensions"
            )
        self.reuse_simulation = reuse_simulation
        self._reuse_simulation_for_adjoint = (
            (self.reuse_simulation or bool(self._near2far_targets)) and uses_simulation_spec
        )
        self.last_chunk_balance = None
        self.last_source_boundary_decision = None

        if len(self.monitor_positions) != len(self.wavelength_bands):
            raise ValueError("monitor_positions must match the number of wavelength bands")
        if self.weights.size != len(self.wavelength_bands):
            raise ValueError("weights must match the number of wavelength bands")

        self.dt = None
        self.filter_time = None
        self.run_time = None
        self.resolved_transition_width = None
        self.kernels = []
        self.weighted_kernels = []
        if dt is not None or resolution is not None:
            self._configure_time_grid(
                self._resolve_dt(
                    dt,
                    resolution,
                    courant=self._fallback_courant,
                )
            )
        self.last_band_objectives = None
        self.last_band_losses = None
        self.last_band_coeffs = None
        self.last_scalarization_info = None
        self.last_smooth_min = None
        self.last_total_fom = None
        self.last_target_history_mode = None
        self.last_target_history_layout = None
        non_flux_target_staggering = {
            _is_magnetic_component(target_component)
            for target_component, is_mixed_surface in zip(
                self.target_components,
                self._mixed_surface_target_mask,
            )
            if not is_mixed_surface
        }
        self._mixed_non_flux_target_time_staggering = (
            len(non_flux_target_staggering) > 1
        )

    def _configure_target_channel_layout(self) -> None:
        """Build raw target channels and their deduplicated monitors."""
        target_slices = []
        flat_target_positions = []
        flat_target_components = []
        flat_adjoint_source_sizes = []
        flat_adjoint_source_amplitudes = []
        sample_target_indices = []
        target_component_slices = []
        sample_offset = 0
        for target_index, (
            positions,
            target_component,
            source_size,
            source_amplitude,
            target_normal,
            is_flux_target,
            is_eigenmode_target,
        ) in enumerate(
            zip(
                self.target_positions,
                self.target_components,
                self.adjoint_source_sizes,
                self.adjoint_source_amplitudes,
                self.target_normals,
                self._flux_target_mask,
                self._eigenmode_target_mask,
            )
        ):
            component_slices = {}
            target_start = sample_offset
            raw_components = (
                tuple(
                    component
                    for pair in flux_component_pairs(
                        target_normal,
                        cylindrical=self._is_cylindrical,
                    )
                    for component in pair[:2]
                )
                if is_flux_target or is_eigenmode_target
                else (target_component,)
            )
            for raw_component in raw_components:
                component_start = sample_offset
                flat_target_positions.extend(positions)
                flat_target_components.extend(
                    [raw_component] * len(positions)
                )
                flat_adjoint_source_sizes.extend(
                    [source_size] * len(positions)
                )
                flat_adjoint_source_amplitudes.extend(
                    [source_amplitude] * len(positions)
                )
                sample_target_indices.extend(
                    [target_index] * len(positions)
                )
                sample_offset += len(positions)
                component_slices[raw_component] = slice(
                    component_start,
                    sample_offset,
                )
            target_slices.append(slice(target_start, sample_offset))
            target_component_slices.append(component_slices)
        self._target_slices = tuple(target_slices)
        self._target_component_slices = tuple(target_component_slices)
        self._flat_target_positions = tuple(flat_target_positions)
        self._flat_target_components = tuple(flat_target_components)
        self._flat_adjoint_source_sizes = tuple(
            flat_adjoint_source_sizes
        )
        self._flat_adjoint_source_amplitudes = tuple(
            flat_adjoint_source_amplitudes
        )
        self._sample_target_indices = tuple(sample_target_indices)

        flat_to_monitor_indices = np.empty(
            len(self._flat_target_positions),
            dtype=np.intp,
        )
        monitor_key_to_index = {}
        monitor_target_positions = []
        monitor_target_components = []
        monitor_adjoint_source_sizes = []
        monitor_adjoint_source_amplitudes = []
        monitor_sample_target_indices = []
        for sample_index, (
            position,
            target_component,
            source_size,
            source_amplitude,
            target_index,
        ) in enumerate(
            zip(
                self._flat_target_positions,
                self._flat_target_components,
                self._flat_adjoint_source_sizes,
                self._flat_adjoint_source_amplitudes,
                self._sample_target_indices,
            )
        ):
            if self._flux_target_mask[target_index]:
                key = (
                    int(target_component),
                    float(position.x),
                    float(position.y),
                    float(position.z),
                )
            elif self._regional_target_mask[target_index]:
                key = (
                    "regional",
                    int(target_component),
                    float(position.x),
                    float(position.y),
                    float(position.z),
                    float(source_size.x),
                    float(source_size.y),
                    float(source_size.z),
                    complex(source_amplitude),
                )
            else:
                key = ("logical", sample_index)
            monitor_index = monitor_key_to_index.get(key)
            if monitor_index is None:
                monitor_index = len(monitor_target_positions)
                monitor_key_to_index[key] = monitor_index
                monitor_target_positions.append(position)
                monitor_target_components.append(target_component)
                monitor_adjoint_source_sizes.append(source_size)
                monitor_adjoint_source_amplitudes.append(source_amplitude)
                monitor_sample_target_indices.append(target_index)
            flat_to_monitor_indices[sample_index] = monitor_index
        self._flat_to_monitor_indices = flat_to_monitor_indices
        self._target_monitor_indices = tuple(
            flat_to_monitor_indices[target_slice]
            for target_slice in self._target_slices
        )
        self._monitor_target_positions = tuple(monitor_target_positions)
        self._monitor_target_components = tuple(monitor_target_components)
        self._monitor_adjoint_source_sizes = tuple(
            monitor_adjoint_source_sizes
        )
        self._monitor_adjoint_source_amplitudes = tuple(
            monitor_adjoint_source_amplitudes
        )
        self._monitor_sample_target_indices = tuple(
            monitor_sample_target_indices
        )
        self._uses_deduplicated_monitors = (
            len(self._monitor_target_positions)
            < len(self._flat_target_positions)
        )

        if self.design is not None:
            gradient_components = []
            for target_component in self._flat_target_components:
                for gradient_component in _electric_components(
                    target_component,
                    self.design.dimensions,
                    cylindrical=self._is_cylindrical,
                    m=getattr(self._simulation_spec, "m", 0),
                ):
                    if gradient_component not in gradient_components:
                        gradient_components.append(gradient_component)
            self.gradient_components = tuple(gradient_components)
        else:
            self.gradient_components = ()

        if (
            self.design is not None
            and self._simulation_spec is not None
            and (self._near2far_targets
                 or _uses_material_jacobian(self._simulation_spec, self.design))
        ):
            self.gradient_components = (mp.Ex, mp.Ey, mp.Ez)

        monitor_groups = {}
        for monitor_index, target_component in enumerate(
            self._monitor_target_components
        ):
            monitor_groups.setdefault(target_component, []).append(
                monitor_index
            )
        self._monitor_groups = tuple(
            (target_component, tuple(target_indices))
            for target_component, target_indices in monitor_groups.items()
        )

    def _resolve_runtime_surface_coordinate_system(
        self,
        is_cylindrical: bool,
    ) -> None:
        """Infer a plain factory's surface channel layout from its simulation."""
        if not (self._has_flux_targets or self._has_eigenmode_targets):
            return
        if self._has_coordinate_system_metadata:
            if is_cylindrical != self._is_cylindrical:
                raise ValueError(
                    "target coordinate system does not match the forward "
                    "simulation"
                )
            return
        if (
            self._inferred_is_cylindrical is not None
            and is_cylindrical != self._inferred_is_cylindrical
        ):
            raise ValueError(
                "target coordinate system does not match the forward "
                "simulation"
            )
        if is_cylindrical:
            for normal, positions, spatial_weights, is_surface in zip(
                self.target_normals,
                self.target_positions,
                self.target_spatial_weights,
                self._mixed_surface_target_mask,
            ):
                if is_surface:
                    _validate_cylindrical_surface(
                        positions,
                        normal,
                        spatial_weights,
                    )
        if is_cylindrical != self._is_cylindrical:
            self._is_cylindrical = is_cylindrical
            self._configure_target_channel_layout()
        self._inferred_is_cylindrical = is_cylindrical

    def _configure_time_grid(self, dt: float) -> None:
        configured_dt = _positive_finite_float(dt, "dt")
        if self.filter_bank == "power_complementary":
            transition_width = self.transition_width
            if transition_width is None:
                transition_width = 2.0 / (
                    self.kernel_length * configured_dt
                )
            kernels = _power_complementary_kernels(
                self.wavelength_bands,
                configured_dt,
                self.kernel_length,
                transition_width,
            )
            self.resolved_transition_width = transition_width
        else:
            kernels = []
            for lam_min, lam_max in self.wavelength_bands:
                f_low = 1 / lam_max
                f_high = 1 / lam_min
                kernels.append(
                    bandpass_kernel(
                        f_low,
                        f_high,
                        configured_dt,
                        self.kernel_length,
                        window=self.kernel_window,
                        window_params=self.kernel_window_params,
                    )
                )
            self.resolved_transition_width = None
        weighted_kernels = [
            weight * kernel for weight, kernel in zip(self.weights, kernels)
        ]
        self.dt = configured_dt
        self.filter_time = self.kernel_length * configured_dt
        self.run_time = self.t_final
        self.kernels = kernels
        self.weighted_kernels = weighted_kernels

    def _resolve_sampling_interval(self, dt: float) -> int:
        if self.max_frequency is None:
            return self.sampling_interval
        return nyquist_sampling_interval(self.max_frequency, dt)

    @staticmethod
    def _resolve_dt(
        dt: Optional[float],
        resolution: Optional[float],
        *,
        courant: float = 0.5,
    ) -> float:
        if dt is not None:
            return _positive_finite_float(dt, "dt")
        if resolution is not None:
            resolution = _positive_finite_float(resolution, "resolution")
            courant = _positive_finite_float(courant, "courant")
            return courant / resolution
        raise ValueError("MultiTDAObjective requires dt or resolution")

    def _sync_near2far_sampling(self):
        core = self._near2far_objective
        for name in _NEAR2FAR_SAMPLING_ATTRIBUTES:
            setattr(core, name, getattr(self, name))
        return core

    def time_step(self, sim: mp.Simulation) -> float:
        """Return the time step used by TAMA sampling.

        Args:
            sim: Forward Meep simulation created by `sim_factory`.

        Returns:
            The initialized Meep time step when available, otherwise explicit
            `dt` when supplied, otherwise `Courant / resolution`.
        """
        if self._near2far_objective is not None:
            return self._sync_near2far_sampling().time_step(sim)
        sim_dt = _simulation_time_step(sim, self._explicit_dt)
        if sim_dt is not None:
            return sim_dt
        resolution = (
            self.resolution
            if self.resolution is not None
            else getattr(sim, "resolution", None)
        )
        if resolution is None:
            raise ValueError(
                "MultiTDAObjective requires dt, resolution, or a Simulation "
                "with a resolution attribute"
            )
        courant = float(getattr(sim, "Courant", self._fallback_courant))
        return self._resolve_dt(None, resolution, courant=courant)

    def _resolve_pixel_chunk(
        self,
        pixel_chunk,
        *,
        design_variable_count: int,
        target_chunks_per_rank: int,
        min_pixel_chunk: int,
        max_pixel_chunk: int,
    ) -> int:
        if isinstance(pixel_chunk, str):
            if pixel_chunk != "auto":
                raise ValueError('pixel_chunk must be a positive integer or "auto"')
            return auto_pixel_chunk(
                design_variable_count,
                target_chunks_per_rank=target_chunks_per_rank,
                min_pixel_chunk=min_pixel_chunk,
                max_pixel_chunk=max_pixel_chunk,
            )
        pixel_chunk = int(pixel_chunk)
        if pixel_chunk <= 0:
            raise ValueError("pixel_chunk must be positive")
        return pixel_chunk

    @staticmethod
    def _weighted_sum_scalarization(band_objectives: np.ndarray):
        return float(np.sum(band_objectives)), np.ones_like(band_objectives)

    @staticmethod
    def _parse_scalarization_result(result, band_objectives: np.ndarray):
        if not isinstance(result, tuple) or len(result) not in (2, 3):
            raise ValueError("scalarization_fn must return (total_fom, band_coeffs) or (total_fom, band_coeffs, info)")
        total_fom, band_coeffs = result[:2]
        info = result[2] if len(result) == 3 else None
        band_coeffs = np.asarray(band_coeffs, dtype=float)
        if band_coeffs.shape != band_objectives.shape:
            raise ValueError("scalarization_fn band_coeffs must match band_objectives shape")
        return float(total_fom), band_coeffs, info

    def _evaluate_scalarization(
        self,
        band_objectives: np.ndarray,
        *,
        need_gradient: bool,
    ):
        result = self.scalarization_fn(band_objectives)
        if isinstance(result, tuple):
            return self._parse_scalarization_result(result, band_objectives)

        result_array = np.asarray(result)
        if result_array.ndim != 0:
            raise ValueError(
                "scalarization_fn must return a scalar or "
                "(total_fom, band_coeffs[, info])"
            )
        if not need_gradient:
            return float(result_array), None, None

        try:
            differentiated_total, differentiated_coeffs = value_and_grad(
                self.scalarization_fn
            )(band_objectives)
            band_coeffs = np.asarray(
                differentiated_coeffs,
                dtype=float,
            )
        except Exception as exc:
            raise TypeError(
                "scalarization_fn must use autograd.numpy operations when "
                "returning a scalar for gradient evaluation; otherwise return "
                "(total_fom, band_coeffs) with explicit derivatives"
            ) from exc
        if band_coeffs.shape != band_objectives.shape:
            raise ValueError(
                "scalarization_fn derivative must match band_objectives shape"
            )
        return float(differentiated_total), band_coeffs, None

    def _raw_target_history(
        self,
        flat_histories: np.ndarray,
        target_index: int,
    ) -> np.ndarray:
        """Return one target's flat raw monitor channels."""
        return np.asarray(flat_histories)[:, self._target_slices[target_index]]

    def _target_component_histories(
        self,
        flat_histories: np.ndarray,
        target_index: int,
    ):
        """Return one target's raw histories keyed by Meep component."""
        return self._target_component_histories_from_raw(
            self._raw_target_history(flat_histories, target_index),
            target_index,
        )

    def _target_component_histories_from_raw(
        self,
        raw_history: np.ndarray,
        target_index: int,
    ):
        """Return one target-local raw history keyed by Meep component."""
        values = np.asarray(raw_history)
        target_start = self._target_slices[target_index].start
        return {
            component: values[
                :,
                slice(
                    component_slice.start - target_start,
                    component_slice.stop - target_start,
                ),
            ]
            for component, component_slice in (
                self._target_component_slices[target_index].items()
            )
        }

    def _target_history_from_raw(
        self,
        raw_history: np.ndarray,
        target_index: int,
    ) -> np.ndarray:
        """Return one target's public history from target-local channels."""
        values = np.asarray(raw_history)
        component_histories = self._target_component_histories_from_raw(
            values,
            target_index,
        )
        if self._eigenmode_target_mask[target_index]:
            mode_fields = self._reference_mode_fields[target_index]
            if mode_fields is None:
                raise RuntimeError(
                    "EigenmodeCoefficientTarget reference mode is not initialized"
                )
            return coefficient_history(
                component_histories,
                mode_fields,
                self.target_normals[target_index],
                self.target_spatial_weights[target_index],
                cylindrical=self._is_cylindrical,
            )
        if self._flux_target_mask[target_index]:
            return power_history(
                component_histories,
                self.target_normals[target_index],
                self.target_spatial_weights[target_index],
                cylindrical=self._is_cylindrical,
            )

        sample_shape = self.target_sample_shapes[target_index]
        if not sample_shape:
            return values[:, 0]
        return values.reshape((values.shape[0],) + sample_shape)

    def _target_history(
        self,
        flat_histories: np.ndarray,
        target_index: int,
    ) -> np.ndarray:
        """Return one target's public field, flux, or modal history."""
        return self._target_history_from_raw(
            self._raw_target_history(flat_histories, target_index),
            target_index,
        )

    def _distributed_target_transform(
        self,
        signals: np.ndarray,
        transform,
        *,
        output_time_count: Optional[int] = None,
    ) -> np.ndarray:
        """Evaluate one potentially spatial transform per wavelength band."""
        signals = np.asarray(signals)
        expected_width = len(self._flat_target_positions)
        if signals.ndim != 2 or signals.shape[1] != expected_width:
            raise ValueError(
                "regional monitor histories must have shape "
                "(n_time, total_target_points)"
            )
        nproc = mp.count_processors()
        if output_time_count is None:
            output_time_count = signals.shape[0]
        requires_complex = bool(
            np.iscomplexobj(signals)
            and np.any(signals.imag != 0.0)
        )
        local = np.zeros(
            (output_time_count, signals.shape[1]),
            dtype=np.complex128 if requires_complex else np.float64,
        )
        transform_error = None
        try:
            for target_index in range(
                mp.my_rank() if nproc > 1 else 0,
                len(self._target_slices),
                nproc if nproc > 1 else 1,
            ):
                transformed = np.asarray(transform(target_index))
                expected_shape = (
                    output_time_count,
                    self._target_slices[target_index].stop
                    - self._target_slices[target_index].start,
                )
                if transformed.shape != expected_shape:
                    raise ValueError(
                        "target transform must match the requested target shape"
                    )
                transformed_is_complex = bool(
                    np.iscomplexobj(transformed)
                    and np.any(transformed.imag != 0.0)
                )
                if transformed_is_complex and not requires_complex:
                    local = local.astype(np.complex128)
                    requires_complex = True
                transformed = transformed.reshape(transformed.shape[0], -1)
                local[:, self._target_slices[target_index]] = (
                    transformed
                    if requires_complex or not np.iscomplexobj(transformed)
                    else transformed.real
                )
        except Exception as exc:
            transform_error = exc
        if nproc > 1:
            status = FastFieldGrid.reduce(
                np.asarray(
                    [[transform_error is not None, requires_complex]],
                    dtype=np.complex128,
                )
            )[0]
            transform_failed = bool(status[0].real)
            requires_complex = bool(status[1].real)
            if transform_failed:
                if transform_error is not None:
                    raise transform_error
                raise RuntimeError(
                    "target transform failed on another active Meep rank"
                )
            if requires_complex and not np.iscomplexobj(local):
                local = local.astype(np.complex128)
            local = (
                FastFieldGrid.reduce_inplace(local)
                if requires_complex
                else FastFieldGrid.reduce_real_inplace(local)
            )
        elif transform_error is not None:
            raise transform_error
        return local

    def _band_fom_value_and_adjoint_signal(
        self,
        filtered_raw_history: np.ndarray,
        band_index: int,
        *,
        need_gradient: bool,
        evaluate_fom: bool = True,
    ):
        """Evaluate one filtered-band FoM and target-local covector."""
        fom_fns, adjoint_signal_fns = self._resolved_band_callbacks(
            validate_pairs=need_gradient,
        )
        band_fom_fn = fom_fns[band_index]
        band_adjoint_signal_fn = adjoint_signal_fns[band_index]
        band_history = self._target_history_from_raw(
            filtered_raw_history,
            band_index,
        )
        band_objective = None
        if evaluate_fom:
            if band_fom_fn is None:
                band_objective = (
                    np.sum(band_history) * self.dt
                    if self._flux_target_mask[band_index]
                    else (
                        np.sum(np.abs(band_history) ** 2) * self.dt
                        if self._eigenmode_target_mask[band_index]
                        else (
                            0.5
                            * np.sum(
                                np.abs(band_history) ** 2
                                * self.target_spatial_weights[band_index]
                            )
                            * self.dt
                        )
                    )
                )
            else:
                band_objective = float(band_fom_fn(band_history, self.dt))

        if not need_gradient:
            return float(band_objective), None

        if band_adjoint_signal_fn is not None:
            band_signal = np.asarray(
                band_adjoint_signal_fn(band_history, self.dt)
            )
            signal_name = "adjoint_signal_fn"
            signal_scale = 1.0
        elif band_fom_fn is not None:
            band_signal = np.asarray(
                grad(band_fom_fn, 0)(band_history, self.dt)
            )
            signal_name = "fom_fn derivative"
            signal_scale = 1.0 / self.dt
        elif self._flux_target_mask[band_index]:
            band_signal = np.ones_like(band_history, dtype=float)
            signal_name = "default flux derivative"
            signal_scale = 1.0
        elif self._eigenmode_target_mask[band_index]:
            band_signal = 2.0 * np.conjugate(band_history)
            signal_name = "default eigenmode derivative"
            signal_scale = 1.0
        else:
            band_signal = (
                np.conjugate(band_history)
                * self.target_spatial_weights[band_index]
            )
            signal_name = "default field derivative"
            signal_scale = 1.0

        if band_signal.shape != band_history.shape:
            raise ValueError(
                f"{signal_name} must match each filtered band history shape"
            )
        band_signal = signal_scale * band_signal
        if self._flux_target_mask[band_index]:
            component_covectors = power_history_pullback(
                self._target_component_histories_from_raw(
                    filtered_raw_history,
                    band_index,
                ),
                self.target_normals[band_index],
                self.target_spatial_weights[band_index],
                band_signal,
                cylindrical=self._is_cylindrical,
            )
        elif self._eigenmode_target_mask[band_index]:
            component_covectors = coefficient_history_pullback(
                self._target_component_histories_from_raw(
                    filtered_raw_history,
                    band_index,
                ),
                self._reference_mode_fields[band_index],
                self.target_normals[band_index],
                self.target_spatial_weights[band_index],
                band_signal,
                cylindrical=self._is_cylindrical,
            )
        else:
            component_covectors = None

        if component_covectors is not None:
            adjoint_dtype = np.result_type(*(
                np.asarray(covector).dtype
                for covector in component_covectors.values()
            ))
        else:
            adjoint_dtype = np.asarray(band_signal).dtype
        filtered_adjoint = np.zeros_like(
            filtered_raw_history,
            dtype=adjoint_dtype,
        )
        if component_covectors is not None:
            target_start = self._target_slices[band_index].start
            for component, component_covector in component_covectors.items():
                component_slice = self._target_component_slices[
                    band_index
                ][component]
                filtered_adjoint[
                    :,
                    slice(
                        component_slice.start - target_start,
                        component_slice.stop - target_start,
                    ),
                ] = component_covector
        else:
            filtered_adjoint[:] = band_signal.reshape(
                band_signal.shape[0],
                -1,
            )
        return (
            None if band_objective is None else float(band_objective)
        ), filtered_adjoint

    def _eigenmode_fom_values_and_overlap_covectors(
        self,
        overlap_history: np.ndarray,
        *,
        need_gradient: bool,
    ):
        """Evaluate modal bands from their contracted E/H histories."""
        overlap_history = np.asarray(overlap_history)
        band_count = len(self.wavelength_bands)
        if (
            overlap_history.ndim != 2
            or overlap_history.shape[1] != 2 * band_count
        ):
            raise ValueError(
                "eigenmode overlap history must have shape "
                "(n_time, 2 * n_bands)"
            )
        overlaps_by_band = overlap_history.reshape(
            overlap_history.shape[0],
            band_count,
            2,
        )
        fom_fns, adjoint_signal_fns = self._resolved_band_callbacks(
            validate_pairs=need_gradient,
        )
        band_objectives = []
        overlap_covectors = []
        for band_index in range(band_count):
            filtered_overlaps = temporal_convolve_signal(
                overlaps_by_band[:, band_index, :],
                self.weighted_kernels[band_index],
                self.dt,
            )
            band_history = _coefficient_history_from_overlaps(
                filtered_overlaps
            )
            band_fom_fn = fom_fns[band_index]
            band_adjoint_signal_fn = adjoint_signal_fns[band_index]
            band_objective = (
                np.sum(np.abs(band_history) ** 2) * self.dt
                if band_fom_fn is None
                else float(band_fom_fn(band_history, self.dt))
            )
            band_objectives.append(float(band_objective))
            if not need_gradient:
                continue

            if band_adjoint_signal_fn is not None:
                band_signal = np.asarray(
                    band_adjoint_signal_fn(band_history, self.dt)
                )
                signal_name = "adjoint_signal_fn"
                signal_scale = 1.0
            elif band_fom_fn is not None:
                band_signal = np.asarray(
                    grad(band_fom_fn, 0)(band_history, self.dt)
                )
                signal_name = "fom_fn derivative"
                signal_scale = 1.0 / self.dt
            else:
                band_signal = 2.0 * np.conjugate(band_history)
                signal_name = "default eigenmode derivative"
                signal_scale = 1.0
            if band_signal.shape != band_history.shape:
                raise ValueError(
                    f"{signal_name} must match each filtered band history shape"
                )
            overlap_covectors.append(
                _coefficient_overlap_pullback(
                    signal_scale * band_signal,
                    filtered_overlaps.shape[0],
                )
            )
        return np.asarray(band_objectives), (
            np.stack(overlap_covectors, axis=1)
            if need_gradient
            else None
        )

    def _filter_transpose_eigenmode_overlap_covectors(
        self,
        filtered_overlap_covectors: np.ndarray,
        band_coeffs: np.ndarray,
    ) -> np.ndarray:
        """Pull modal covectors back through each band's temporal filter."""
        values = np.asarray(filtered_overlap_covectors)
        band_count = len(self.wavelength_bands)
        if values.ndim != 3 or values.shape[1:] != (band_count, 2):
            raise ValueError(
                "filtered eigenmode overlap covectors must have shape "
                "(n_time, n_bands, 2)"
            )
        return np.stack(
            [
                np.conjugate(
                    temporal_convolve_signal_transpose(
                        np.conjugate(
                            values[:, band_index, :]
                            * band_coeffs[band_index]
                        ),
                        self.weighted_kernels[band_index],
                        self.dt,
                    )
                )
                for band_index in range(band_count)
            ],
            axis=1,
        )

    def _eigenmode_adjoint_sources(
        self,
        overlap_monitors,
        indexed_stencils_by_band,
        overlap_covectors: np.ndarray,
        monitor_times: np.ndarray,
        actual_time: float,
        dt: float,
        effective_source_amplitudes,
    ) -> list:
        """Expand modal covectors into batched exact indexed sources."""
        values = np.asarray(overlap_covectors)
        times = np.asarray(monitor_times, dtype=float)
        band_count = len(self.wavelength_bands)
        if values.shape != (times.size, band_count, 2):
            raise ValueError(
                "eigenmode overlap covectors must have shape "
                "(n_time, n_bands, 2)"
            )
        if (
            len(overlap_monitors) != band_count
            or len(indexed_stencils_by_band) != band_count
        ):
            raise ValueError(
                "eigenmode overlap monitors must match the band count"
            )

        t_array = float(actual_time) - times[::-1]
        source_amplitudes = np.asarray(
            effective_source_amplitudes,
            dtype=np.complex128,
        )
        sources = []
        for band_index, (overlap_monitor, indexed_stencils) in enumerate(
            zip(overlap_monitors, indexed_stencils_by_band)
        ):
            adjoint_values = values[:, band_index, :][::-1].copy()
            adjoint_values[:, 1] *= -1.0
            base_functions = _tabulated_cubic_sources(
                t_array,
                adjoint_values,
            )
            shifted_magnetic_function = (
                _shift_tabulated_cubic_source(base_functions[1], dt)
                if t_array.size >= 4
                else _tabulated_cubic_sources(
                    t_array + dt,
                    adjoint_values[:, 1:2],
                )[0]
            )
            time_sources = (
                (mp.CustomSource(src_func=base_functions[0]),),
                (
                    mp.CustomSource(src_func=base_functions[1]),
                    mp.CustomSource(src_func=shifted_magnetic_function),
                ),
            )

            for component, projection_weights in (
                overlap_monitor.component_weights.items()
            ):
                component_slice = self._target_component_slices[
                    band_index
                ][component]
                flat_indices = np.arange(
                    component_slice.start,
                    component_slice.stop,
                    dtype=np.intp,
                )
                monitor_indices = self._flat_to_monitor_indices[flat_indices]
                point_amplitudes = (
                    projection_weights * source_amplitudes[monitor_indices]
                )
                channel = 1 if _is_magnetic_component(component) else 0
                if channel == 1:
                    point_amplitudes = 0.5 * point_amplitudes
                if not np.any(point_amplitudes != 0.0):
                    continue
                for time_source in time_sources[channel]:
                    sources.extend(
                        _adjoint_indexed_profile_sources(
                            time_source,
                            point_amplitudes,
                            indexed_stencils[component],
                        )
                    )
        return sources

    def _resolved_target_callbacks(self, *, validate_pairs: bool):
        band_count = len(self.targets) if self._near2far_targets else len(self.wavelength_bands)
        fom_fns = _normalize_band_callbacks(
            self.fom_fn,
            band_count,
            "fom_fn",
        )
        adjoint_signal_fns = _normalize_band_callbacks(
            self.adjoint_signal_fn,
            band_count,
            "adjoint_signal_fn",
        )
        if validate_pairs:
            _validate_band_callback_pairs(fom_fns, adjoint_signal_fns)
        return fom_fns, adjoint_signal_fns

    def _resolved_band_callbacks(self, *, validate_pairs: bool):
        fns, signals = self._resolved_target_callbacks(validate_pairs=validate_pairs)
        if self._near2far_targets:
            return (tuple(fns[i] for i in self._temporal_indices),
                    tuple(signals[i] for i in self._temporal_indices))
        return fns, signals

    def _validate_runtime_band_callbacks(self, *, validate_pairs: bool):
        callback_error = None
        fom_fns = None
        adjoint_signal_fns = None
        try:
            fom_fns, adjoint_signal_fns = self._resolved_target_callbacks(
                validate_pairs=validate_pairs,
            )
        except Exception as exc:
            callback_error = exc

        nproc = mp.count_processors()
        if nproc > 1:
            failed_ranks = int(
                mp.sum_to_all(int(callback_error is not None))
            )
            if failed_ranks:
                if callback_error is not None:
                    raise callback_error
                raise RuntimeError(
                    "fom_fn or adjoint_signal_fn validation failed on another "
                    "active Meep rank"
                )
            for name, callbacks in (
                ("fom_fn", fom_fns),
                ("adjoint_signal_fn", adjoint_signal_fns),
            ):
                for callback in callbacks:
                    configured_ranks = int(
                        mp.sum_to_all(int(callback is not None))
                    )
                    if configured_ranks not in (0, nproc):
                        raise RuntimeError(
                            f"{name} must be configured consistently for "
                            "each band on every active Meep rank"
                        )
        elif callback_error is not None:
            raise callback_error
        return fom_fns, adjoint_signal_fns

    def _uses_distributed_builtin_target_history(self) -> bool:
        """Return whether this evaluation is eligible for the opt-in path."""
        if (
            self.target_history_block_size is None
            or not self._uses_concrete_simulation_spec
            or bool(getattr(self._simulation_spec, "symmetries", ()))
            or self._is_cylindrical
            or not (
                all(self._regional_target_mask)
                or all(self._flux_target_mask)
            )
        ):
            return False
        fom_fns, adjoint_signal_fns = self._resolved_band_callbacks(
            validate_pairs=False,
        )
        return all(callback is None for callback in (
            *fom_fns,
            *adjoint_signal_fns,
        ))

    def _close_distributed_flux_ownership(self, owners) -> None:
        """Replicate connected mixed-owner flux point channels to closure."""
        flux_targets = []
        for target_index, is_flux in enumerate(self._flux_target_mask):
            if not is_flux:
                continue
            component_indices = []
            for component_slice in self._target_component_slices[
                target_index
            ].values():
                flat_indices = np.arange(
                    component_slice.start,
                    component_slice.stop,
                    dtype=np.intp,
                )
                component_indices.append(
                    self._flat_to_monitor_indices[flat_indices]
                )
            flux_targets.append(np.stack(component_indices, axis=1))

        changed = True
        while changed:
            changed = False
            for target_points in flux_targets:
                for point_channels in target_points:
                    active_channels = point_channels[
                        owners[point_channels] != -2
                    ]
                    if active_channels.size == 0:
                        continue
                    point_owners = owners[active_channels]
                    common_owner = (
                        point_owners[0] >= 0
                        and np.all(point_owners == point_owners[0])
                    )
                    if not common_owner and np.any(point_owners != -1):
                        owners[active_channels] = -1
                        changed = True

    def _configure_distributed_monitor_histories(
        self,
        groups,
        *,
        row_count: int,
        dtype,
    ):
        """Select unique-owner and replicated monitor channels per rank."""
        channel_count = len(self._monitor_target_positions)
        rank = mp.my_rank()
        ownership_reduction = None
        support_error = None
        try:
            local_support = np.zeros(channel_count, dtype=float)
            for group in groups:
                offsets = np.asarray(
                    group["indexed_stencil"][0],
                    dtype=np.intp,
                )
                supported = np.diff(offsets) != 0
                if supported.shape != group["indices"].shape:
                    raise RuntimeError(
                        "point-monitor support metadata does not match its group"
                    )
                local_support[group["indices"]] = supported
            ownership_reduction = np.vstack(
                (local_support, local_support * float(rank + 1))
            )
        except Exception as exc:
            support_error = exc
        self._synchronize_distributed_target_error(
            support_error,
            "support configuration",
        )
        if mp.count_processors() > 1:
            ownership_reduction = FastFieldGrid.reduce_real_inplace(
                ownership_reduction
            )
        owners = None
        lookup = {}
        local_unique_channels = 0
        replicated_channels = 0
        implicit_zero_channels = 0
        configuration_error = None
        try:
            support_counts = np.rint(ownership_reduction[0]).astype(
                np.int64
            )
            # Nonnegative entries are unique-owner ranks, -1 is replicated,
            # and -2 is a globally unallocated component represented as zero.
            owners = np.full(channel_count, -2, dtype=np.int64)
            unique = support_counts == 1
            owners[unique] = (
                np.rint(ownership_reduction[1, unique]).astype(np.int64) - 1
            )
            owners[support_counts > 1] = -1
            # A flux point is local only when all globally allocated E/H
            # channels have the same unique owner. Mixed-owner channels are
            # replicated transitively across overlapping flux targets.
            self._close_distributed_flux_ownership(owners)
            replicated_channels = int(np.count_nonzero(owners == -1))
            implicit_zero_channels = int(np.count_nonzero(owners == -2))
            for group_index, group in enumerate(groups):
                group_global_indices = np.asarray(
                    group["indices"],
                    dtype=np.intp,
                )
                unique_indices = np.flatnonzero(
                    owners[group_global_indices] == rank
                ).astype(np.int64)
                replicated_indices = np.flatnonzero(
                    owners[group_global_indices] == -1
                ).astype(np.int64)
                history_indices = np.concatenate(
                    (unique_indices, replicated_indices)
                )
                group["monitor"].configure_history_indices(history_indices)
                group["history_indices"] = history_indices
                group["unique_count"] = int(unique_indices.size)
                group["replicated_count"] = int(replicated_indices.size)
                group["history_global_indices"] = group_global_indices[
                    history_indices
                ]
                group["history"] = np.empty(
                    (row_count, history_indices.size),
                    dtype=dtype,
                )
                local_unique_channels += int(unique_indices.size)
                for column, global_index in enumerate(
                    group["history_global_indices"]
                ):
                    lookup[int(global_index)] = (group_index, column)
        except Exception as exc:
            configuration_error = exc
        self._synchronize_distributed_target_error(
            configuration_error,
            "history configuration",
        )

        layout = {
            "groups": groups,
            "lookup": lookup,
            "owners": owners,
            "local_unique_channels": local_unique_channels,
            "replicated_channels": replicated_channels,
            "implicit_zero_channels": implicit_zero_channels,
            "stored_channels": local_unique_channels + replicated_channels,
            "total_channels": channel_count,
            "block_size": self.target_history_block_size,
            "peak_work_columns": 0,
            "allocated_rows": row_count,
            "history_itemsize": np.dtype(dtype).itemsize,
        }
        layout["allocated_history_bytes"] = (
            row_count
            * layout["stored_channels"]
            * layout["history_itemsize"]
        )
        layout["legacy_history_bytes"] = (
            row_count * channel_count * layout["history_itemsize"]
        )
        self.last_target_history_layout = {
            key: value
            for key, value in layout.items()
            if key not in ("groups", "lookup", "owners")
        }
        return layout

    @staticmethod
    def _make_distributed_reduction_scratch(shape, dtype):
        return np.empty(shape, dtype=dtype)

    def _reduce_replicated_monitor_histories(self, layout, row_count: int) -> None:
        """All-reduce only replicated monitor columns in bounded blocks."""
        block_size = int(self.target_history_block_size)
        max_work_bytes = 16 * 1024**2
        for group in layout["groups"]:
            group["history"] = group["history"][:row_count]
            unique_count = group["unique_count"]
            replicated_count = group["replicated_count"]
            for first in range(0, replicated_count, block_size):
                width = min(block_size, replicated_count - first)
                rows_per_block = max(
                    1,
                    max_work_bytes // (
                        width * group["history"].dtype.itemsize
                    ),
                )
                scratch = None
                scratch_error = None
                try:
                    scratch = self._make_distributed_reduction_scratch(
                        (min(row_count, rows_per_block), width),
                        group["history"].dtype,
                    )
                except Exception as exc:
                    scratch_error = exc
                self._synchronize_distributed_target_error(
                    scratch_error,
                    "replicated-history reduction workspace",
                )
                column_start = unique_count + first
                column_stop = column_start + width
                for row_start in range(0, row_count, rows_per_block):
                    row_stop = min(row_count, row_start + rows_per_block)
                    work = scratch[:row_stop - row_start]
                    work[:] = group["history"][
                        row_start:row_stop,
                        column_start:column_stop,
                    ]
                    if mp.count_processors() > 1:
                        work = (
                            FastFieldGrid.reduce_inplace(work)
                            if np.issubdtype(
                                work.dtype,
                                np.complexfloating,
                            )
                            else FastFieldGrid.reduce_real_inplace(work)
                        )
                    group["history"][
                        row_start:row_stop,
                        column_start:column_stop,
                    ] = work

    @staticmethod
    def _distributed_history_columns(layout, global_indices) -> np.ndarray:
        """Return stored histories for one component's logical channels."""
        global_indices = np.asarray(global_indices, dtype=np.intp)
        entries = [layout["lookup"].get(int(index)) for index in global_indices]
        missing = np.asarray([entry is None for entry in entries])
        if np.any(
            missing
            & (layout["owners"][global_indices] != -2)
        ):
            raise RuntimeError(
                "distributed target point is missing a required component history"
            )
        present_entries = [entry for entry in entries if entry is not None]
        group_indices = {entry[0] for entry in present_entries}
        if len(group_indices) != 1:
            if not group_indices:
                return np.zeros(
                    (
                        layout["groups"][0]["history"].shape[0],
                        global_indices.size,
                    ),
                    dtype=layout["groups"][0]["history"].dtype,
                )
            raise RuntimeError(
                "one field component spans multiple distributed monitor groups"
            )
        group_index = present_entries[0][0]
        history = layout["groups"][group_index]["history"]
        present_columns = np.asarray(
            [entry[1] for entry in present_entries],
            dtype=np.intp,
        )
        if not np.any(missing):
            return history[:, present_columns]
        result = np.zeros(
            (history.shape[0], global_indices.size),
            dtype=history.dtype,
        )
        result[:, ~missing] = history[:, present_columns]
        return result

    def _distributed_filtered_target_blocks(self, layout, *, objective_pass):
        """Yield bounded filtered spatial blocks for every local target."""
        for target_index, component_slices in enumerate(
            self._target_component_slices
        ):
            component_indices = {}
            for component, component_slice in component_slices.items():
                flat_indices = np.arange(
                    component_slice.start,
                    component_slice.stop,
                    dtype=np.intp,
                )
                component_indices[component] = self._flat_to_monitor_indices[
                    flat_indices
                ]
            point_channels = np.stack(
                tuple(component_indices.values()),
                axis=1,
            )
            stored_counts = np.asarray(
                [
                    sum(
                        int(channel) in layout["lookup"]
                        for channel in channels
                    )
                    for channels in point_channels
                ]
            )
            active_counts = np.sum(
                layout["owners"][point_channels] != -2,
                axis=1,
            )
            if np.any(
                (stored_counts != 0)
                & (stored_counts != active_counts)
            ):
                raise RuntimeError(
                    "distributed flux point components do not share an owner"
                )
            stored = (active_counts > 0) & (stored_counts == active_counts)
            if objective_pass:
                replicated = np.any(
                    layout["owners"][point_channels] == -1,
                    axis=1,
                )
                assigned = (
                    np.arange(point_channels.shape[0])
                    % max(1, mp.count_processors())
                ) == mp.my_rank()
                stored &= ~replicated | assigned
            positions = np.flatnonzero(stored)
            weights = self.target_spatial_weights[target_index].reshape(-1)
            for first in range(
                0,
                positions.size,
                self.target_history_block_size,
            ):
                block = positions[
                    first:first + self.target_history_block_size
                ]
                layout["peak_work_columns"] = max(
                    layout["peak_work_columns"],
                    int(block.size),
                )
                filtered = {
                    component: temporal_convolve_signal(
                        self._distributed_history_columns(
                            layout,
                            indices[block],
                        ),
                        self.weighted_kernels[target_index],
                        self.dt,
                    )
                    for component, indices in component_indices.items()
                }
                yield (
                    target_index,
                    block,
                    weights[block],
                    component_indices,
                    filtered,
                )

    @staticmethod
    def _synchronize_distributed_target_error(error, operation: str) -> None:
        if mp.count_processors() <= 1:
            if error is not None:
                raise error
            return
        status = FastFieldGrid.reduce_real_inplace(
            np.asarray([[error is not None]], dtype=float)
        )
        if bool(status[0, 0]):
            if error is not None:
                raise error
            raise RuntimeError(
                f"distributed target {operation} failed on another MPI rank"
            )

    def _distributed_builtin_band_objectives(self, layout) -> np.ndarray:
        """Evaluate built-in regional/flux FoMs from rank-local columns."""
        local_objectives = np.zeros(len(self._target_slices), dtype=float)
        transform_error = None
        try:
            for (
                target_index,
                _,
                weights,
                _,
                filtered,
            ) in self._distributed_filtered_target_blocks(
                layout,
                objective_pass=True,
            ):
                if self._flux_target_mask[target_index]:
                    local_objectives[target_index] += float(
                        np.sum(
                            power_history(
                                filtered,
                                self.target_normals[target_index],
                                weights,
                                cylindrical=False,
                            )
                        ).real
                        * self.dt
                    )
                else:
                    field_history = next(iter(filtered.values()))
                    local_objectives[target_index] += float(
                        0.5
                        * np.sum(
                            np.abs(field_history) ** 2
                            * weights[np.newaxis, :]
                        )
                        * self.dt
                    )
        except Exception as exc:
            transform_error = exc

        self._synchronize_distributed_target_error(
            transform_error,
            "objective",
        )
        if mp.count_processors() > 1:
            local_objectives = FastFieldGrid.reduce_real_inplace(
                local_objectives[np.newaxis, :]
            )[0]

        self.last_target_history_layout["peak_work_columns"] = layout[
            "peak_work_columns"
        ]
        return local_objectives

    @staticmethod
    def _add_distributed_adjoint_columns(
        layout,
        adjoint_histories,
        global_indices,
        values,
    ) -> None:
        """Accumulate one block, including repeated deduplicated channels."""
        entries = [layout["lookup"].get(int(index)) for index in global_indices]
        global_indices = np.asarray(global_indices, dtype=np.intp)
        missing = np.asarray([entry is None for entry in entries])
        if np.any(
            missing
            & (layout["owners"][global_indices] != -2)
        ):
            raise RuntimeError(
                "distributed adjoint block is missing a monitor channel"
            )
        present_entries = [entry for entry in entries if entry is not None]
        if not present_entries:
            return
        group_indices = {entry[0] for entry in present_entries}
        if len(group_indices) != 1:
            raise RuntimeError(
                "one adjoint component spans multiple monitor groups"
            )
        group_index = present_entries[0][0]
        columns = np.asarray(
            [entry[1] for entry in present_entries],
            dtype=np.intp,
        )
        values = np.asarray(values)[:, ~missing]
        for column in np.unique(columns):
            adjoint_histories[group_index][:, column] += np.sum(
                values[:, columns == column],
                axis=1,
            )

    def _distributed_builtin_adjoint_histories(
        self,
        layout,
        band_coeffs: np.ndarray,
    ):
        """Blockwise pull built-in covectors back to compact raw histories."""
        adjoint_histories = None
        transform_error = None
        try:
            adjoint_histories = [
                np.zeros_like(group["history"])
                for group in layout["groups"]
            ]
            for (
                target_index,
                block,
                weights,
                component_indices,
                filtered,
            ) in self._distributed_filtered_target_blocks(
                layout,
                objective_pass=False,
            ):
                if self._flux_target_mask[target_index]:
                    filtered_covectors = power_history_pullback(
                        filtered,
                        self.target_normals[target_index],
                        weights,
                        np.ones(
                            next(iter(filtered.values())).shape[0] - 1,
                            dtype=float,
                        ),
                        cylindrical=False,
                    )
                else:
                    component = next(iter(filtered))
                    filtered_covectors = {
                        component: (
                            np.conjugate(filtered[component])
                            * weights[np.newaxis, :]
                        )
                    }

                for component, filtered_covector in filtered_covectors.items():
                    raw_covector = np.conjugate(
                        temporal_convolve_signal_transpose(
                            np.conjugate(
                                filtered_covector
                                * band_coeffs[target_index]
                            ),
                            self.weighted_kernels[target_index],
                            self.dt,
                        )
                    )
                    self._add_distributed_adjoint_columns(
                        layout,
                        adjoint_histories,
                        component_indices[component][block],
                        raw_covector,
                    )
        except Exception as exc:
            transform_error = exc

        self._synchronize_distributed_target_error(
            transform_error,
            "pullback",
        )

        self.last_target_history_layout["peak_work_columns"] = layout[
            "peak_work_columns"
        ]
        return adjoint_histories

    def _distributed_adjoint_sources(
        self,
        layout,
        adjoint_histories,
        indexed_stencils,
        monitor_times: np.ndarray,
        actual_time: float,
        dt: float,
        source_sizes,
        source_amplitudes,
    ):
        """Synchronize compact adjoint-source construction failures."""
        result = None
        source_error = None
        try:
            result = self._build_distributed_adjoint_sources(
                layout,
                adjoint_histories,
                indexed_stencils,
                monitor_times,
                actual_time,
                dt,
                source_sizes,
                source_amplitudes,
            )
        except Exception as exc:
            source_error = exc
        self._synchronize_distributed_target_error(
            source_error,
            "source construction",
        )
        return result

    def _build_distributed_adjoint_sources(
        self,
        layout,
        adjoint_histories,
        indexed_stencils,
        monitor_times: np.ndarray,
        actual_time: float,
        dt: float,
        source_sizes,
        source_amplitudes,
    ):
        """Create sources only for compact histories with local stencils."""
        active = []
        for group_index, group in enumerate(layout["groups"]):
            for column, global_index in enumerate(
                group["history_global_indices"]
            ):
                stencil = indexed_stencils[int(global_index)]
                if stencil is not None and len(stencil[0]) != 0:
                    active.append((int(global_index), group_index, column))

        source_staggering = {
            _is_magnetic_component(component)
            for component in self._monitor_target_components
        }
        combine_staggered_sources = len(source_staggering) > 1 or bool(self._near2far_targets)
        t_array = float(actual_time) - np.asarray(monitor_times)[::-1]
        base_functions = {}
        shifted_functions = {}
        for first in range(
            0,
            len(active),
            self.target_history_block_size,
        ):
            active_block = active[
                first:first + self.target_history_block_size
            ]
            values = np.column_stack(
                [
                    adjoint_histories[group_index][:, column]
                    for _, group_index, column in active_block
                ]
            )
            parity = np.asarray(
                [
                    -1.0
                    if _is_magnetic_component(
                        self._monitor_target_components[global_index]
                    )
                    else 1.0
                    for global_index, _, _ in active_block
                ]
            )
            values = parity[np.newaxis, :] * values[::-1]
            block_functions = _tabulated_cubic_sources(t_array, values)
            base_functions.update(
                (
                    active_entry[0],
                    source_function,
                )
                for active_entry, source_function in zip(
                    active_block,
                    block_functions,
                )
            )
            magnetic_columns = [
                column
                for column, (global_index, _, _) in enumerate(active_block)
                if (
                    combine_staggered_sources
                    and _is_magnetic_component(
                        self._monitor_target_components[global_index]
                    )
                )
            ]
            if t_array.size >= 4:
                shifted_functions.update(
                    (
                        active_block[column][0],
                        _shift_tabulated_cubic_source(
                            block_functions[column],
                            dt,
                        ),
                    )
                    for column in magnetic_columns
                )
            elif magnetic_columns:
                shifted_functions.update(zip(
                    (active_block[column][0] for column in magnetic_columns),
                    _tabulated_cubic_sources(
                        t_array + dt,
                        values[:, magnetic_columns],
                    ),
                ))

        sources = []
        for global_index, _, _ in active:
            component = self._monitor_target_components[global_index]
            point_functions = (base_functions[global_index],)
            source_amplitude = source_amplitudes[global_index]
            if combine_staggered_sources and _is_magnetic_component(component):
                point_functions += (shifted_functions[global_index],)
                source_amplitude *= 0.5
            for source_function in point_functions:
                sources.extend(
                    _adjoint_point_sources(
                        mp.CustomSource(src_func=source_function),
                        component,
                        self._monitor_target_positions[global_index],
                        source_sizes[global_index],
                        source_amplitude,
                        indexed_stencils[global_index],
                    )
                )
        return sources, source_staggering == {True} and not self._near2far_targets

    def _band_fom_values_and_adjoint_signals(
        self,
        filtered_monitors: np.ndarray,
        *,
        need_gradient: bool,
    ):
        """Evaluate each filtered-band FoM and its continuous-time covector."""
        band_objectives = []
        band_adjoint_signals = [] if need_gradient else None
        for band_index in range(len(self._target_slices)):
            filtered_raw_history = self._raw_target_history(
                filtered_monitors,
                band_index,
            )
            band_objective, filtered_adjoint = (
                self._band_fom_value_and_adjoint_signal(
                    filtered_raw_history,
                    band_index,
                    need_gradient=need_gradient,
                )
            )
            band_objectives.append(band_objective)
            if need_gradient:
                band_adjoint_signals.append(filtered_adjoint)
        adjoint_signals = None
        if need_gradient:
            adjoint_dtype = np.result_type(*(
                signal.dtype for signal in band_adjoint_signals
            ))
            adjoint_signals = np.zeros_like(
                filtered_monitors,
                dtype=adjoint_dtype,
            )
            for band_index, filtered_adjoint in enumerate(
                band_adjoint_signals
            ):
                adjoint_signals[:, self._target_slices[band_index]] = (
                    filtered_adjoint
                )
        return np.asarray(band_objectives), adjoint_signals

    def _filter_transpose_adjoint_signals(
        self,
        filtered_adjoint_signals: np.ndarray,
        band_coeffs: np.ndarray,
    ) -> np.ndarray:
        """Pull filtered-band covectors back to the point-monitor histories."""

        def pullback(band_index):
            band_signal = self._raw_target_history(
                filtered_adjoint_signals,
                band_index,
            )
            # Convert the filter's Hermitian transpose to the transpose for
            # the bilinear complex-covector convention used by Meep sources.
            return np.conjugate(
                temporal_convolve_signal_transpose(
                    np.conjugate(
                        band_signal * band_coeffs[band_index]
                    ),
                    self.weighted_kernels[band_index],
                    self.dt,
                )
            )

        return self._distributed_target_transform(
            filtered_adjoint_signals,
            pullback,
            output_time_count=(
                filtered_adjoint_signals.shape[0]
                - self.kernel_length + 1
            ),
        )

    def _distributed_deduplicated_band_objectives(
        self,
        monitor_history: np.ndarray,
    ) -> np.ndarray:
        """Evaluate band FoMs without expanding shared monitor channels."""
        monitor_history = np.asarray(monitor_history)
        expected_width = len(self._monitor_target_positions)
        if (
            monitor_history.ndim != 2
            or monitor_history.shape[1] != expected_width
        ):
            raise ValueError(
                "deduplicated monitor history must have shape "
                "(n_time, n_monitor_channels)"
            )

        band_objectives = np.zeros(
            len(self._target_slices),
            dtype=np.complex128,
        )
        nproc = mp.count_processors()
        fom_fns, _ = self._resolved_band_callbacks(validate_pairs=False)
        distribute_bands = nproc > 1 and all(
            band_fom_fn is None for band_fom_fn in fom_fns
        )
        transform_error = None
        raw_history = None
        filtered_raw_history = None
        try:
            for band_index in range(
                mp.my_rank() if distribute_bands else 0,
                len(self._target_slices),
                nproc if distribute_bands else 1,
            ):
                raw_history = monitor_history[
                    :,
                    self._target_monitor_indices[band_index],
                ]
                filtered_raw_history = temporal_convolve_signal(
                    raw_history,
                    self.weighted_kernels[band_index],
                    self.dt,
                )
                band_objectives[band_index], _ = (
                    self._band_fom_value_and_adjoint_signal(
                        filtered_raw_history,
                        band_index,
                        need_gradient=False,
                    )
                )
        except Exception as exc:
            transform_error = exc
        raw_history = None
        filtered_raw_history = None
        if nproc > 1:
            if distribute_bands:
                reduction = np.zeros(
                    (1, band_objectives.size + 1),
                    dtype=np.complex128,
                )
                reduction[0, :-1] = band_objectives
                reduction[0, -1] = transform_error is not None
                reduction = FastFieldGrid.reduce(reduction)[0]
                band_objectives = reduction[:-1]
                transform_failed = bool(reduction[-1].real)
            else:
                transform_failed = bool(
                    FastFieldGrid.reduce(
                        np.asarray(
                            [[transform_error is not None]],
                            dtype=np.complex128,
                        )
                    )[0, 0].real
                )
        else:
            transform_failed = transform_error is not None
        if transform_failed:
            if transform_error is not None:
                raise transform_error
            raise RuntimeError(
                "band objective evaluation failed on another MPI rank"
            )
        return np.asarray(band_objectives.real, dtype=float)

    def _distributed_deduplicated_adjoint_signals(
        self,
        monitor_history: np.ndarray,
        band_coeffs: np.ndarray,
    ) -> np.ndarray:
        """Pull band covectors back and sum them on shared channels."""
        monitor_history = np.asarray(monitor_history)
        expected_width = len(self._monitor_target_positions)
        if (
            monitor_history.ndim != 2
            or monitor_history.shape[1] != expected_width
        ):
            raise ValueError(
                "deduplicated monitor history must have shape "
                "(n_time, n_monitor_channels)"
            )

        local = np.zeros_like(monitor_history)
        nproc = mp.count_processors()
        fom_fns, _ = self._resolved_band_callbacks(validate_pairs=False)
        distribute_bands = nproc > 1 and all(
            band_fom_fn is None for band_fom_fn in fom_fns
        )
        transform_error = None
        raw_history = None
        filtered_raw_history = None
        filtered_adjoint = None
        raw_adjoint = None
        try:
            for band_index in range(
                mp.my_rank() if distribute_bands else 0,
                len(self._target_slices),
                nproc if distribute_bands else 1,
            ):
                monitor_indices = self._target_monitor_indices[band_index]
                raw_history = monitor_history[:, monitor_indices]
                filtered_raw_history = temporal_convolve_signal(
                    raw_history,
                    self.weighted_kernels[band_index],
                    self.dt,
                )
                _, filtered_adjoint = (
                    self._band_fom_value_and_adjoint_signal(
                        filtered_raw_history,
                        band_index,
                        need_gradient=True,
                        evaluate_fom=False,
                    )
                )
                raw_adjoint = np.conjugate(
                    temporal_convolve_signal_transpose(
                        np.conjugate(
                            filtered_adjoint * band_coeffs[band_index]
                        ),
                        self.weighted_kernels[band_index],
                        self.dt,
                    )
                )
                result_dtype = np.result_type(
                    local.dtype,
                    raw_adjoint.dtype,
                )
                if result_dtype != local.dtype:
                    local = local.astype(result_dtype)
                for monitor_index in np.unique(monitor_indices):
                    local[:, monitor_index] += np.sum(
                        raw_adjoint[:, monitor_indices == monitor_index],
                        axis=1,
                )
        except Exception as exc:
            transform_error = exc
        raw_history = None
        filtered_raw_history = None
        filtered_adjoint = None
        raw_adjoint = None
        if nproc > 1:
            status = FastFieldGrid.reduce(
                np.asarray(
                    [[transform_error is not None, np.iscomplexobj(local)]],
                    dtype=np.complex128,
                )
            )[0]
            transform_failed = bool(status[0].real)
            requires_complex = bool(status[1].real)
        else:
            transform_failed = transform_error is not None
            requires_complex = np.iscomplexobj(local)
        if transform_failed:
            if transform_error is not None:
                raise transform_error
            raise RuntimeError(
                "band adjoint evaluation failed on another MPI rank"
            )
        if requires_complex and not np.iscomplexobj(local):
            local = local.astype(np.complex128)
        if distribute_bands:
            local = (
                FastFieldGrid.reduce_inplace(local)
                if np.iscomplexobj(local)
                else FastFieldGrid.reduce_real_inplace(local)
            )
        return local

    def _make_history_memmap(self, shape, *, dtype=None):
        tmp = tempfile.NamedTemporaryFile(prefix="tama_history_", suffix=".dat", delete=False)
        path = tmp.name
        tmp.close()
        return np.memmap(
            path,
            dtype=self.history_dtype if dtype is None else dtype,
            mode="w+",
            shape=shape,
        ), path

    def _history_values_to_grid(self, values, history_indices, grid_shape):
        grid = np.zeros(grid_shape, dtype=self.history_dtype)
        grid_flat = grid.reshape(-1)
        grid_flat[history_indices] = values
        return grid

    @staticmethod
    def _distributed_band_transform(signals: np.ndarray, transform) -> np.ndarray:
        """Evaluate independent signal columns across MPI ranks."""
        n_time, n_bands = signals.shape
        nproc = mp.count_processors()
        if nproc <= 1:
            return np.column_stack([
                transform(band_index) for band_index in range(n_bands)
            ])

        requires_complex = bool(
            np.iscomplexobj(signals)
            and np.any(signals.imag != 0.0)
        )
        local = np.zeros(
            (n_time, n_bands),
            dtype=np.complex128 if requires_complex else np.float64,
        )
        transform_error = None
        try:
            for band_index in range(mp.my_rank(), n_bands, nproc):
                transformed = np.asarray(transform(band_index))
                transformed_is_complex = bool(
                    np.iscomplexobj(transformed)
                    and np.any(transformed.imag != 0.0)
                )
                if transformed_is_complex and not requires_complex:
                    local = local.astype(np.complex128)
                    requires_complex = True
                local[:, band_index] = (
                    transformed
                    if requires_complex or not np.iscomplexobj(transformed)
                    else transformed.real
                )
        except Exception as exc:
            transform_error = exc
        status = FastFieldGrid.reduce(
            np.asarray(
                [[transform_error is not None, requires_complex]],
                dtype=np.complex128,
            )
        )[0]
        transform_failed = bool(status[0].real)
        requires_complex = bool(status[1].real)
        if transform_failed:
            if transform_error is not None:
                raise transform_error
            raise RuntimeError(
                "band transform failed on another active Meep rank"
            )
        if requires_complex and not np.iscomplexobj(local):
            local = local.astype(np.complex128)
        return (
            FastFieldGrid.reduce_inplace(local)
            if requires_complex
            else FastFieldGrid.reduce_real_inplace(local)
        )

    def __call__(
        self,
        x: np.ndarray,
        need_gradient: bool = True,
    ) -> Tuple[float, Optional[np.ndarray]]:
        """Evaluate the multi-band objective.

        Args:
            x: Flat design vector passed to `update_design`.
            need_gradient: If `True`, run the adjoint simulation and return a
                gradient. If `False`, run only the forward simulation.

        Returns:
            `(value, gradient)` from `evaluate`. `gradient` is `None`
            when `need_gradient=False`.
        """
        return self.evaluate(x, need_gradient=need_gradient)

    def fom(self, x: np.ndarray) -> float:
        """Evaluate only the scalar value returned by `evaluate`.

        Args:
            x: Flat design vector.

        Returns:
            Scalar total FoM from the value-only path.
        """
        value, _ = self.evaluate(x, need_gradient=False)
        return value

    def fom_and_grad(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Evaluate the scalar value and flat design gradient.

        Args:
            x: Flat design vector.

        Returns:
            `(value, gradient)` using the maximization convention
            `value=total_fom` and `gradient=d(value)/d(x)`.
        """
        return self.evaluate(x, need_gradient=True)

    def _forward_sources_for_evaluation(self):
        if (
            self.chunk_balancer is None
            or not self._uses_concrete_simulation_spec
        ):
            return ()
        return tuple(self._simulation_spec.resolve_sources())

    def _make_forward_simulation(self, forward_sources=()):
        if self.chunk_balancer is None or not self._uses_concrete_simulation_spec:
            sim = self.sim_factory()
        else:
            sim = self._simulation_spec.make(forward_sources)
        try:
            _prepare_native_material(sim, self.design)
        except BaseException:
            sim.reset_meep()
            raise
        return sim

    def evaluate(
        self,
        x: np.ndarray,
        need_gradient: bool = True,
    ) -> Tuple[float, Optional[np.ndarray]]:
        """Run forward/adjoint simulations for one design vector.

        Args:
            x: Flat design vector passed to `update_design`.
            need_gradient: Whether to run the adjoint simulation and compute a
                design gradient.

        Returns:
            `(total_fom, gradient)`. `gradient` is a flat real array with
            the same number of entries as `x` when `need_gradient=True`;
            otherwise it is `None`.
        """
        if self.chunk_balancer is not None and getattr(self._simulation_spec, "symmetries", ()):
            raise ValueError("Mirror simulations do not support adaptive chunk balancing")
        if self._near2far_targets and self.chunk_balancer is not None:
            raise ValueError("Near2FarTarget does not support adaptive chunk balancing")
        if self._near2far_objective is not None:
            from .near2far import evaluate_near2far

            if need_gradient and self.design is None:
                raise ValueError("gradient evaluation requires a DesignGrid")
            callbacks = _normalize_band_callbacks(self.fom_fn, len(self.targets), "fom_fn")
            signals = _normalize_band_callbacks(
                self.adjoint_signal_fn, len(self.targets), "adjoint_signal_fn"
            )
            if need_gradient:
                _validate_band_callback_pairs(callbacks, signals)
            core = self._sync_near2far_sampling()
            core.update_design = self.update_design
            result = evaluate_near2far(
                core, x, need_gradient, targets=self.targets, fom_fns=callbacks,
                adjoint_signal_fns=signals, scalarize=self._evaluate_scalarization,
            )
            for name in ("last_far_fields", "last_band_objectives", "last_band_losses",
                         "last_band_coeffs", "last_scalarization_info", "last_smooth_min",
                         "last_total_fom", "last_sampling_interval", "last_actual_time"):
                setattr(self, name, getattr(core, name))
            return result

        if need_gradient and self.design is None:
            raise ValueError(
                "MultiTDAObjective gradient evaluation requires design=DesignGrid(...)"
            )
        if (
            need_gradient
            and self._uses_simulation_spec
            and self._simulation_spec.m != self._configured_simulation_m
        ):
            raise ValueError(
                "SimulationSpec.m must not change after "
                "MultiTDAObjective construction"
            )
        if need_gradient:
            _validate_logical_adjoint_source_amplitudes(
                amplitude
                for amplitude, is_mixed_surface in zip(
                    self.adjoint_source_amplitudes,
                    self._mixed_surface_target_mask,
                )
                if not is_mixed_surface
            )
        if (
            need_gradient
            and self._mixed_non_flux_target_time_staggering
            and not self._near2far_targets
        ):
            raise ValueError(
                "MultiTDAObjective gradients cannot mix electric and magnetic "
                "components across PointTarget and FieldRegionTarget entries "
                "in one adjoint run"
            )
        self._validate_runtime_band_callbacks(validate_pairs=need_gradient)
        if (
            need_gradient
            and (
                self._has_regional_targets
                or self._has_flux_targets
                or self._has_eigenmode_targets
            )
            and not self._uses_simulation_spec
        ):
            target_name = (
                "EigenmodeCoefficientTarget"
                if (
                    self._has_eigenmode_targets
                    and not self._has_flux_targets
                    and not self._has_regional_targets
                )
                else "FluxTarget"
                if (
                    self._has_flux_targets
                    and not self._has_regional_targets
                )
                else "FieldRegionTarget"
            )
            raise ValueError(
                f"{target_name} gradient evaluation requires "
                "simulation=SimulationSpec(...)"
            )
        forward_sources = self._forward_sources_for_evaluation()
        self.update_design(x)
        effective_adjoint_source_sizes = self._monitor_adjoint_source_sizes
        effective_adjoint_source_amplitudes = (
            self._monitor_adjoint_source_amplitudes
        )
        indexed_source_mask = tuple(
            bool(
                self._uses_simulation_spec
                and (
                    self._regional_target_mask[target_index]
                    or self._flux_target_mask[target_index]
                    or self._eigenmode_target_mask[target_index]
                    or self._is_cylindrical
                    or (
                        _is_zero_source_size(source_size)
                        and getattr(
                            self.chunk_balancer,
                            "source_boundary_mode",
                            "auto",
                        )
                        != "finite"
                    )
                )
            )
            for source_size, target_index in zip(
                self._monitor_adjoint_source_sizes,
                self._monitor_sample_target_indices,
            )
        )
        if self.chunk_balancer is not None:
            source_settings = (
                {
                    "adjoint_source_size": self.adjoint_source_size,
                    "adjoint_source_amplitude": self.adjoint_source_amplitude,
                }
                if (
                    len(self._monitor_target_positions)
                    == len(self.target_components)
                    and self.adjoint_source_size is not None
                    and self.adjoint_source_amplitude is not None
                )
                else {
                    "adjoint_source_sizes": (
                        self._monitor_adjoint_source_sizes
                    ),
                    "adjoint_source_amplitudes": (
                        self._monitor_adjoint_source_amplitudes
                    ),
                }
            )
            source_boundary_decision = self.chunk_balancer.prepare_initial_layout(
                self._simulation_spec,
                adjoint_source_positions=self._monitor_target_positions,
                indexed_source_mask=indexed_source_mask,
                forward_sources=forward_sources,
                **source_settings,
            )
            self.last_source_boundary_decision = source_boundary_decision
            effective_adjoint_source_sizes = source_boundary_decision.source_sizes
            effective_adjoint_source_amplitudes = (
                source_boundary_decision.source_amplitudes
            )
        if any(
            indexed
            and not _is_zero_source_size(source_size)
            for indexed, source_size in zip(
                indexed_source_mask,
                effective_adjoint_source_sizes,
            )
        ):
            raise RuntimeError(
                "indexed adjoint target acquired a nonzero effective source size"
            )

        native_history = None
        native_accumulator = None
        sim_fwd = None
        sim_adj = None
        fwd_monitor = None
        eigenmode_overlap_monitors = None
        adjoint_indexed_stencils = None
        distributed_history_layout = None
        distributed_history_active = False
        self.last_target_history_mode = "legacy"
        self.last_target_history_layout = None
        streaming_eigenmode_overlap = bool(
            self._uses_simulation_spec
            and self._eigenmode_target_mask
            and all(self._eigenmode_target_mask)
        )
        try:
            balance_enabled = bool(
                need_gradient
                and self.chunk_balancer is not None
                and getattr(
                    self.chunk_balancer,
                    "needs_rebalance",
                    not getattr(self.chunk_balancer, "is_frozen", False),
                )
            )
            balance_wall_start = time.perf_counter() if balance_enabled else None
            sim_fwd = self._make_forward_simulation(forward_sources)
            if self._near2far_targets:
                from .near2far import (add_near2far_monitors, farfield_values,
                                      farfield_adjoint_sources)

                near_monitors = add_near2far_monitors(
                    sim_fwd, self._near2far_targets, self.max_frequency,
                    chunk_layout=self._simulation_spec.chunk_layout)
                fns, signals = self._resolved_target_callbacks(validate_pairs=need_gradient)
                near_fns = tuple(fns[i] for i in self._near2far_indices)
                near_signals = tuple(signals[i] for i in self._near2far_indices)
            _require_no_meep_symmetries(sim_fwd)
            if (need_gradient and getattr(sim_fwd, "symmetries", ())
                    and (not self._uses_simulation_spec or not all(indexed_source_mask))):
                raise ValueError(
                    "Mirror gradients require direct SimulationSpec and exact indexed adjoint sources"
                )
            is_cylindrical = _is_cylindrical_simulation(sim_fwd)
            forward_mode = getattr(sim_fwd, "m", 0)
            forward_mode = 0 if forward_mode is None else forward_mode
            if is_cylindrical:
                for target_index, target_positions in enumerate(
                    self.target_positions
                ):
                    if self._mixed_surface_target_mask[target_index]:
                        continue
                    for target_position in target_positions:
                        _validate_cylindrical_target_position(target_position)
            if need_gradient and is_cylindrical and not self._uses_simulation_spec:
                raise ValueError(
                    "cylindrical gradient evaluation requires "
                    "simulation=SimulationSpec(...)"
                )
            self._resolve_runtime_surface_coordinate_system(
                is_cylindrical
            )
            if (
                (self._has_flux_targets or self._has_eigenmode_targets)
                and _is_cartesian_2d_simulation(sim_fwd)
                and any(
                    abs(float(normal.z)) > 0.0
                    for normal, is_surface in zip(
                        self.target_normals,
                        self._mixed_surface_target_mask,
                    )
                    if is_surface
                )
            ):
                raise ValueError(
                    "2D flux and eigenmode target normals must lie in the "
                    "x-y plane"
                )
            if need_gradient and is_cylindrical:
                if forward_mode not in (-1, 0, 1):
                    raise ValueError(
                        "cylindrical gradient evaluation currently supports "
                        "m=-1, 0, or +1"
                    )
                for target_component, monitor_position, target_index in zip(
                    self._monitor_target_components,
                    self._monitor_target_positions,
                    self._monitor_sample_target_indices,
                ):
                    if self._mixed_surface_target_mask[target_index]:
                        continue
                    _validate_cylindrical_axis_target(
                        target_component,
                        forward_mode,
                        monitor_position,
                    )
            dt = self.time_step(sim_fwd)
            run_history_dtype = history_storage_dtype(
                sim_fwd,
                self.history_dtype,
            )
            if need_gradient:
                gradient_components = _native_design_components(
                    sim_fwd, self.design, self.gradient_components
                )
                gradient_scale = _native_design_gradient_scale(sim_fwd, self.design)
                native_history = _NativeDesignHistorySet(
                    self.design,
                    gradient_components,
                    run_history_dtype,
                    lambda shape: self._make_history_memmap(
                        shape,
                        dtype=run_history_dtype,
                    ),
                )
            if self.dt != dt:
                self._configure_time_grid(dt)
            if (
                self._has_eigenmode_targets
                and any(
                    fields is None
                    for target, fields in zip(
                        self._eigenmode_targets,
                        self._reference_mode_fields,
                    )
                    if target is not None
                )
            ):
                reference_mode_fields = list(self._reference_mode_fields)
                for target_index, target in enumerate(
                    self._eigenmode_targets
                ):
                    if (
                        target is not None
                        and reference_mode_fields[target_index] is None
                    ):
                        reference_mode_fields[target_index] = (
                            sample_reference_mode(sim_fwd, target)
                        )
                self._reference_mode_fields = tuple(reference_mode_fields)
            eigenmode_projection_weights = None
            if streaming_eigenmode_overlap:
                eigenmode_projection_weights = tuple(
                    _coefficient_projection_weights(
                        self._reference_mode_fields[target_index],
                        self.target_normals[target_index],
                        self.target_spatial_weights[target_index],
                        cylindrical=self._is_cylindrical,
                    )
                    for target_index in range(len(self.wavelength_bands))
                )
            sampling_interval = self._resolve_sampling_interval(dt)
            self.last_sampling_interval = sampling_interval
            dt_eff = dt * sampling_interval
            run_until = _aligned_run_until(
                self.run_time,
                dt=dt,
                sampling_interval=(sampling_interval if need_gradient or self._near2far_targets else 1),
            )
            monitor_times = []
            forward_sampling_interval = (
                sampling_interval if need_gradient else 1
            )
            use_native_forward = hasattr(
                getattr(sim_fwd, "fields", None),
                "this",
            )
            distributed_history_active = bool(
                use_native_forward
                and self._uses_distributed_builtin_target_history()
            )
            if distributed_history_active:
                self.last_target_history_mode = "distributed"
            if use_native_forward:
                native_fine_step_count = _native_forward_step_count(
                    sim_fwd,
                    run_until,
                    forward_sampling_interval,
                )
                n_expected = (
                    native_fine_step_count // forward_sampling_interval + 1
                )
                n_monitor_expected = native_fine_step_count + 1
            else:
                n_expected = int(np.ceil(run_until / dt_eff)) + 8
                n_monitor_expected = int(np.ceil(run_until / dt)) + 8
            monitor_history = (
                None
                if distributed_history_active
                else np.empty(
                    (
                        n_monitor_expected,
                        (
                            2 * len(self.wavelength_bands)
                            if streaming_eigenmode_overlap
                            else len(self._monitor_target_positions)
                        ),
                    ),
                    dtype=(
                        np.complex128
                        if streaming_eigenmode_overlap
                        else run_history_dtype
                    ),
                )
            )
            fwd_monitor = {
                "groups": None,
                "overlaps": None,
                "overlap_stencils": None,
            }
            fwd_count = {"value": 0}
            sample_count = {"value": 0}
            balance_timing = {"extra_seconds": 0.0}

            def prepare_fwd_monitors(sim, native_group_histories=False):
                nonlocal distributed_history_layout
                if (
                    streaming_eigenmode_overlap
                    and fwd_monitor["overlaps"] is None
                ):
                    overlap_monitors = []
                    overlap_stencils_by_band = []
                    for target_index, (
                        electric_weights,
                        magnetic_weights,
                    ) in enumerate(eigenmode_projection_weights):
                        overlap_monitor = FastEigenmodeOverlapMonitor(
                            sim,
                            self.target_positions[target_index],
                            electric_weights,
                            magnetic_weights,
                        )
                        overlap_stencils = (
                            overlap_monitor.indexed_transpose_stencils()
                        )
                        for component, stencil in overlap_stencils.items():
                            _validate_indexed_monitor_support(
                                sim,
                                (component,) * len(
                                    self.target_positions[target_index]
                                ),
                                stencil[0],
                            )
                        overlap_monitors.append(overlap_monitor)
                        overlap_stencils_by_band.append(overlap_stencils)
                    fwd_monitor["overlaps"] = tuple(overlap_monitors)
                    fwd_monitor["overlap_stencils"] = tuple(
                        overlap_stencils_by_band
                    )
                elif (
                    not streaming_eigenmode_overlap
                    and fwd_monitor["groups"] is None
                ):
                    fwd_monitor["groups"] = []
                    for target_component, target_indices in self._monitor_groups:
                        indices = np.asarray(target_indices, dtype=np.intp)
                        monitor = FastPointMonitor(
                            sim,
                            target_component,
                            [
                                self._monitor_target_positions[index]
                                for index in target_indices
                            ],
                        )
                        indexed_stencil = monitor.indexed_transpose_stencil()
                        _validate_indexed_monitor_support(
                            sim,
                            (target_component,) * len(target_indices),
                            indexed_stencil[0],
                        )
                        group = {
                            "indices": indices,
                            "monitor": monitor,
                            "indexed_stencil": indexed_stencil,
                        }
                        if (
                            native_group_histories
                            and not distributed_history_active
                        ):
                            consecutive = np.array_equal(
                                indices,
                                np.arange(
                                    int(indices[0]),
                                    int(indices[0]) + len(indices),
                                    dtype=np.intp,
                                ),
                            )
                            if consecutive:
                                group["history"] = monitor_history[
                                    :, int(indices[0]) : int(indices[-1]) + 1
                                ]
                            else:
                                group["history"] = np.empty(
                                    (n_monitor_expected, len(target_indices)),
                                    dtype=run_history_dtype,
                                )
                        elif not distributed_history_active:
                            group["values"] = np.empty(
                                len(target_indices),
                                dtype=run_history_dtype,
                            )
                        fwd_monitor["groups"].append(group)
                    if distributed_history_active:
                        distributed_history_layout = (
                            self._configure_distributed_monitor_histories(
                                fwd_monitor["groups"],
                                row_count=n_monitor_expected,
                                dtype=run_history_dtype,
                            )
                        )

            if use_native_forward:
                native_setup_start = (
                    time.perf_counter() if balance_enabled else None
                )
                prepare_fwd_monitors(
                    sim_fwd,
                    native_group_histories=(
                        not streaming_eigenmode_overlap
                    ),
                )
                if streaming_eigenmode_overlap:
                    monitor_bindings = tuple(
                        (
                            overlap_monitor,
                            monitor_history[
                                :, 2 * band_index : 2 * band_index + 2
                            ],
                        )
                        for band_index, overlap_monitor in enumerate(
                            fwd_monitor["overlaps"]
                        )
                    )
                else:
                    monitor_bindings = tuple(
                        (group["monitor"], group["history"])
                        for group in fwd_monitor["groups"]
                    )
                native_setup_seconds = (
                    time.perf_counter() - native_setup_start
                    if balance_enabled
                    else 0.0
                )
                (
                    sample_count["value"],
                    fwd_count["value"],
                    monitor_times,
                    native_forward_seconds,
                ) = _run_native_forward_loop(
                    sim_fwd,
                    monitor_bindings,
                    native_history if need_gradient else None,
                    run_until=run_until,
                    sampling_interval=forward_sampling_interval,
                    fine_step_count=native_fine_step_count,
                )
                del monitor_bindings
                if (
                    not streaming_eigenmode_overlap
                    and not distributed_history_active
                ):
                    for group in fwd_monitor["groups"]:
                        if not np.shares_memory(
                            monitor_history,
                            group["history"],
                        ):
                            monitor_history[:, group["indices"]] = (
                                group["history"]
                            )
                        del group["history"]
                    del group
                if balance_enabled:
                    balance_timing["extra_seconds"] += (
                        native_setup_seconds + native_forward_seconds
                    )

            def record_fwd_impl(sim):
                nonlocal monitor_history
                sample_index = sample_count["value"]
                if sample_index >= monitor_history.shape[0]:
                    grown_history = np.empty(
                        (
                            2 * monitor_history.shape[0],
                            monitor_history.shape[1],
                        ),
                        dtype=monitor_history.dtype,
                    )
                    grown_history[:sample_index] = monitor_history
                    monitor_history = grown_history
                prepare_fwd_monitors(sim)
                monitor_row = monitor_history[sample_index]
                if streaming_eigenmode_overlap:
                    for band_index, overlap_monitor in enumerate(
                        fwd_monitor["overlaps"]
                    ):
                        overlap_monitor.sample_history_into(
                            monitor_row[
                                2 * band_index : 2 * band_index + 2
                            ]
                        )
                else:
                    for group in fwd_monitor["groups"]:
                        group["monitor"].sample_history_into(group["values"])
                        monitor_row[group["indices"]] = group["values"]
                monitor_times.append(_sample_time_from_simulation(sim, sample_index, self.dt))
                if need_gradient and sample_count["value"] % sampling_interval == 0:
                    native_history.sample_forward(
                        sim,
                        fwd_count["value"],
                        n_expected,
                    )
                    fwd_count["value"] += 1
                sample_count["value"] += 1

            if not use_native_forward:
                record_fwd = record_fwd_impl
                if balance_enabled:
                    def record_fwd(sim):
                        start = time.perf_counter()
                        try:
                            record_fwd_impl(sim)
                        finally:
                            balance_timing["extra_seconds"] += (
                                time.perf_counter() - start
                            )

                sim_fwd.run(record_fwd, until=run_until)
            actual_time = sim_fwd.round_time()
            if not distributed_history_active:
                monitor_history = monitor_history[:sample_count["value"]]
            if distributed_history_active:
                pass
            elif streaming_eigenmode_overlap:
                eigenmode_overlap_monitors = fwd_monitor["overlaps"]
                monitor_history = (
                    eigenmode_overlap_monitors[0].reduce_history(
                        monitor_history
                    )
                )
            else:
                # Every component group uses the same active Meep communicator.
                # Reduce the packed history once instead of reducing per group.
                monitor_history = fwd_monitor["groups"][0]["monitor"].reduce_history(
                    monitor_history
                )
                monitor_history = np.asarray(
                    monitor_history,
                    dtype=run_history_dtype,
                )
            full_monitor_times = np.asarray(monitor_times, dtype=float)
            if need_gradient:
                fwd_histories = native_history.finish_forward(fwd_count["value"])
                fine_step_count = sample_count["value"] - 1
                expected_history_rows = fine_step_count // sampling_interval + 1
                if (
                    fine_step_count % sampling_interval != 0
                    or any(
                        history.shape[0] != expected_history_rows
                        for history in fwd_histories.values()
                    )
                ):
                    raise RuntimeError(
                        "forward callback count is not aligned with the Nyquist "
                        "sampling interval"
                    )
            monitor_stop = _monitor_history_stop(
                full_monitor_times,
                self.run_time,
            )
            if distributed_history_active:
                self._reduce_replicated_monitor_histories(
                    distributed_history_layout,
                    monitor_stop,
                )
            else:
                monitor_history = monitor_history[:monitor_stop]
            monitor_times = full_monitor_times[:monitor_stop]
            if streaming_eigenmode_overlap and need_gradient:
                adjoint_indexed_stencils = fwd_monitor["overlap_stencils"]
            elif need_gradient and any(indexed_source_mask):
                adjoint_indexed_stencils = [
                    None
                ] * len(self._monitor_target_positions)
                for group in fwd_monitor["groups"]:
                    if not any(
                        indexed_source_mask[target_index]
                        for target_index in group["indices"]
                    ):
                        continue
                    offsets, source_data, source_weights = group[
                        "indexed_stencil"
                    ]
                    if offsets.shape != (len(group["indices"]) + 1,):
                        raise RuntimeError(
                            "MultiTDAObjective indexed monitor "
                            "stencil count does not match its targets"
                        )
                    for group_index, target_index in enumerate(group["indices"]):
                        if not indexed_source_mask[target_index]:
                            continue
                        adjoint_indexed_stencils[target_index] = (
                            source_data[
                                int(offsets[group_index]):
                                int(offsets[group_index + 1])
                            ],
                            source_weights[
                                int(offsets[group_index]):
                                int(offsets[group_index + 1])
                            ],
                        )
            fwd_timing = self.chunk_balancer.capture_timing(sim_fwd) if balance_enabled else None
            if self._near2far_targets:
                near_fields, near_values = farfield_values(
                    sim_fwd, near_monitors, self._near2far_targets, near_fns, dt)
                self.last_far_fields = tuple(f.copy() for f in near_fields)
                self.last_actual_time = actual_time
            fwd_monitor["groups"] = None
            fwd_monitor["overlaps"] = None
            fwd_monitor["overlap_stencils"] = None
            if native_history is not None:
                native_history.release_forward()
            if not (need_gradient and self._reuse_simulation_for_adjoint):
                sim_fwd.reset_meep()
                sim_fwd = None
            gc.collect()

            filtered_monitors = None
            filtered_adjoint_signals = None
            if distributed_history_active:
                band_objectives = self._distributed_builtin_band_objectives(
                    distributed_history_layout
                )
            elif streaming_eigenmode_overlap:
                band_objectives, filtered_adjoint_signals = (
                    self._eigenmode_fom_values_and_overlap_covectors(
                        monitor_history,
                        need_gradient=need_gradient,
                    )
                )
            elif self._uses_deduplicated_monitors:
                band_objectives = (
                    self._distributed_deduplicated_band_objectives(
                        monitor_history
                    )
                )
            else:
                filtered_monitors = (
                    self._distributed_filter_monitor_signals(
                        monitor_history
                    )
                )
                band_objectives, filtered_adjoint_signals = (
                    self._band_fom_values_and_adjoint_signals(
                        filtered_monitors,
                        need_gradient=need_gradient,
                    )
                )
            if self._near2far_targets:
                combined_values = np.empty(len(self.targets))
                combined_values[list(self._temporal_indices)] = band_objectives
                combined_values[list(self._near2far_indices)] = near_values
                band_objectives = combined_values
            band_losses = -band_objectives
            scalarization_error = None
            try:
                total_fom, band_coeffs, scalarization_info = self._evaluate_scalarization(
                    band_objectives,
                    need_gradient=need_gradient,
                )
                if self._near2far_targets and (not np.isfinite(total_fom)
                        or (need_gradient and not np.all(np.isfinite(band_coeffs)))):
                    raise ValueError("mixed scalarization value and coefficients must be finite")
            except Exception as exc:
                scalarization_error = exc
            if self._near2far_targets:
                self._synchronize_distributed_target_error(scalarization_error, "scalarization")
            elif scalarization_error is not None:
                raise scalarization_error
            self.last_band_objectives = band_objectives
            self.last_band_losses = band_losses
            self.last_band_coeffs = band_coeffs
            self.last_scalarization_info = scalarization_info
            self.last_smooth_min = (
                scalarization_info.get("smooth_min")
                if isinstance(scalarization_info, dict)
                else None
            )
            self.last_total_fom = total_fom
            if self._near2far_targets:
                if need_gradient:
                    near_sources = farfield_adjoint_sources(
                        sim_fwd, near_monitors, self._near2far_targets, near_fields,
                        near_fns, near_signals, np.asarray(band_coeffs)[list(self._near2far_indices)],
                        actual_time, dt)
                    band_coeffs = np.asarray(band_coeffs)[list(self._temporal_indices)]
            if not need_gradient:
                return total_fom, None
            histories_too_short = any(
                history.shape[0] < 2 for history in fwd_histories.values()
            )
            if histories_too_short:
                raise RuntimeError(
                    "MultiTDAObjective gradient requires at least two field samples; "
                    "increase t_final or decrease sampling_interval"
                )
            if monitor_times.shape[0] < 4:
                raise RuntimeError(
                    "MultiTDAObjective gradient requires at least four monitor samples "
                    "for cubic adjoint-source interpolation; increase t_final"
                )

            preprocess_start = time.perf_counter() if balance_enabled else None
            distributed_adjoint_histories = None
            if distributed_history_active:
                distributed_adjoint_histories = (
                    self._distributed_builtin_adjoint_histories(
                        distributed_history_layout,
                        band_coeffs,
                    )
                )
                adj_signals = None
            elif streaming_eigenmode_overlap:
                adj_signals = (
                    self._filter_transpose_eigenmode_overlap_covectors(
                        filtered_adjoint_signals,
                        band_coeffs,
                    )
                )
            elif self._uses_deduplicated_monitors:
                adj_signals = (
                    self._distributed_deduplicated_adjoint_signals(
                        monitor_history,
                        band_coeffs,
                    )
                )
            else:
                adj_signals = self._filter_transpose_adjoint_signals(
                    filtered_adjoint_signals,
                    band_coeffs,
                )
            del filtered_adjoint_signals
            if not distributed_history_active:
                del monitor_history
            del filtered_monitors
            if balance_enabled:
                balance_timing["extra_seconds"] += time.perf_counter() - preprocess_start
            if distributed_history_active:
                for group in distributed_history_layout["groups"]:
                    group["history"] = None
                adjoint_sources, adjoint_midpoint = (
                    self._distributed_adjoint_sources(
                        distributed_history_layout,
                        distributed_adjoint_histories,
                        adjoint_indexed_stencils,
                        monitor_times,
                        actual_time,
                        dt,
                        effective_adjoint_source_sizes,
                        effective_adjoint_source_amplitudes,
                    )
                )
                distributed_adjoint_histories = None
            elif streaming_eigenmode_overlap:
                adjoint_sources = self._eigenmode_adjoint_sources(
                    eigenmode_overlap_monitors,
                    adjoint_indexed_stencils,
                    adj_signals,
                    monitor_times,
                    actual_time,
                    dt,
                    effective_adjoint_source_amplitudes,
                )
                adjoint_midpoint = False
                del adj_signals
            else:
                source_parity = np.asarray(
                    [
                        -1.0 if _is_magnetic_component(component) else 1.0
                        for component in self._monitor_target_components
                    ]
                )
                adj_signals = source_parity[np.newaxis, :] * adj_signals[::-1]
                t_array = float(actual_time) - monitor_times[::-1]

                source_staggering = {
                    _is_magnetic_component(component)
                    for component in self._monitor_target_components
                }
                combine_staggered_sources = len(source_staggering) > 1 or bool(self._near2far_targets)
                active_source_indices = [
                    sample_index
                    for sample_index in range(len(self._monitor_target_positions))
                    if (
                        adjoint_indexed_stencils is None
                        or adjoint_indexed_stencils[sample_index] is None
                        or len(adjoint_indexed_stencils[sample_index][0]) != 0
                    )
                ]
                base_source_functions = dict(zip(
                    active_source_indices,
                    _tabulated_cubic_sources(
                        t_array,
                        adj_signals[:, active_source_indices],
                    ),
                ))
                shifted_magnetic_indices = [
                    sample_index
                    for sample_index in active_source_indices
                    if (
                        combine_staggered_sources
                        and _is_magnetic_component(
                            self._monitor_target_components[sample_index]
                        )
                    )
                ]
                if t_array.size >= 4:
                    shifted_source_functions = {
                        sample_index: _shift_tabulated_cubic_source(
                            base_source_functions[sample_index],
                            dt,
                        )
                        for sample_index in shifted_magnetic_indices
                    }
                else:
                    shifted_source_functions = dict(zip(
                        shifted_magnetic_indices,
                        _tabulated_cubic_sources(
                            t_array + dt,
                            adj_signals[:, shifted_magnetic_indices],
                        ),
                    ))
                adjoint_sources = []
                for sample_index, (
                        target_component,
                        monitor_position,
                        source_size,
                        source_amplitude,
                    ) in enumerate(zip(
                        self._monitor_target_components,
                        self._monitor_target_positions,
                        effective_adjoint_source_sizes,
                        effective_adjoint_source_amplitudes,
                    )):
                    indexed_stencil = (
                        None
                        if adjoint_indexed_stencils is None
                        else adjoint_indexed_stencils[sample_index]
                    )
                    if (
                        indexed_stencil is not None
                        and len(indexed_stencil[0]) == 0
                    ):
                        continue
                    source_functions = (base_source_functions[sample_index],)
                    if (
                        combine_staggered_sources
                        and _is_magnetic_component(target_component)
                    ):
                        # Causal two-tap filtering makes ordinary accumulation
                        # equal the previous midpoint-adjoint accumulation.
                        source_functions += (
                            shifted_source_functions[sample_index],
                        )
                        source_amplitude *= 0.5
                    for adj_source_func in source_functions:
                        adjoint_sources.extend(
                            _adjoint_point_sources(
                                mp.CustomSource(src_func=adj_source_func),
                                target_component,
                                monitor_position,
                                source_size,
                                source_amplitude,
                                indexed_stencil,
                            )
                        )
                adjoint_midpoint = source_staggering == {True} and not self._near2far_targets
                del adj_signals, t_array

            if self._near2far_targets:
                adjoint_sources.extend(near_sources)
            if self._reuse_simulation_for_adjoint:
                if is_cylindrical:
                    sim_fwd.change_m(-forward_mode)
                sim_adj = sim_fwd
                sim_fwd = None
                sim_adj.restart_fields()
                sim_adj.clear_dft_monitors()
                if balance_enabled:
                    # Meep timers are cumulative across field restarts.
                    sim_adj.fields.reset_timers()
                sim_adj.change_sources(adjoint_sources)
            elif is_cylindrical:
                sim_adj = self._simulation_spec.make(
                    adjoint_sources,
                    m=-forward_mode,
                )
            else:
                sim_adj = self.sim_factory(adjoint_sources)
            _require_no_meep_symmetries(sim_adj)
            _prepare_native_material(sim_adj, self.design)
            _validate_adjoint_source_amplitudes(
                sim_adj,
                effective_adjoint_source_amplitudes,
            )
            native_accumulator = _NativeDesignAccumulator(
                self.design,
                gradient_components,
                native_history.signatures,
                adjoint_midpoint,
            )
            balance_timing["extra_seconds"] += _run_native_adjoint_loop(
                sim_adj,
                native_accumulator,
                fwd_histories,
                fine_step_count=fine_step_count,
                actual_time=actual_time,
                dt=dt,
                sampling_interval=sampling_interval,
                reconstruction_half_width=self.reconstruction_half_width,
                reconstruction_window=self.reconstruction_window,
                reconstruction_window_params=(
                    self.reconstruction_window_params
                ),
            )
            grad_grid = native_accumulator.reduce()
            native_accumulator.release()
            native_accumulator = None

            previous_timings = [fwd_timing] if balance_enabled else []
            if balance_enabled:
                self._simulation_spec.chunk_layout = self.chunk_balancer.rebalance(
                    sim_adj,
                    local_extra_work_seconds=balance_timing["extra_seconds"],
                    local_wall_seconds=time.perf_counter() - balance_wall_start,
                    previous_measurements=tuple(previous_timings),
                )
                self.last_chunk_balance = self.chunk_balancer.last_report
            sim_adj.reset_meep()
            sim_adj = None
            del fwd_histories
            gc.collect()

            gradient = (
                grad_grid * dt * gradient_scale
            ).real.flatten()
            del grad_grid
            gc.collect()
            return total_fom, gradient
        finally:
            if native_accumulator is not None:
                native_accumulator.release()
            if native_history is not None:
                native_history.release_forward()
            if fwd_monitor is not None:
                fwd_monitor["groups"] = None
                fwd_monitor["overlaps"] = None
            eigenmode_overlap_monitors = None
            gc.collect()
            try:
                try:
                    if sim_adj is not None:
                        sim_adj.reset_meep()
                finally:
                    if sim_fwd is not None:
                        sim_fwd.reset_meep()
            finally:
                if native_history is not None:
                    native_history.cleanup_memmaps()
                gc.collect()

    def filter_monitor_signals(self, signals: np.ndarray) -> np.ndarray:
        """Apply one temporal bandpass filter to each monitor signal.

        Args:
            signals: Flat real or complex monitor history with shape
                `(n_time, total_raw_target_channels)`. The contiguous slice for
                target `i` is filtered by kernel `i` and weight `i`.

        Returns:
            Flat filtered history with all `n_time + kernel_length - 1` rows.
        """
        if self.dt is None:
            raise RuntimeError(
                "filter_monitor_signals requires dt, resolution, or a prior evaluation"
            )
        signals = np.asarray(signals)
        expected_width = len(self._flat_target_positions)
        if signals.ndim != 2 or signals.shape[1] != expected_width:
            raise ValueError(
                "signals must have shape "
                "(n_time, total_raw_target_channels)"
            )
        output_time_count = signals.shape[0] + self.kernel_length - 1
        output_dtype = signals.dtype
        if not np.issubdtype(output_dtype, np.inexact):
            output_dtype = np.result_type(output_dtype, np.float64)
        if any(np.iscomplexobj(kernel) for kernel in self.kernels):
            output_dtype = np.result_type(output_dtype, np.complex64)
        filtered = np.empty(
            (output_time_count, signals.shape[1]), dtype=output_dtype,
        )
        for band_index in range(len(self._target_slices)):
            band_history = self._raw_target_history(signals, band_index)
            filtered[:, self._target_slices[band_index]] = (
                temporal_convolve_signal(
                    band_history,
                    self.weights[band_index] * self.kernels[band_index],
                    self.dt,
                )
            )
        return filtered

    def _distributed_filter_monitor_signals(self, signals: np.ndarray) -> np.ndarray:
        return self._distributed_target_transform(
            signals,
            lambda band_index: temporal_convolve_signal(
                self._raw_target_history(signals, band_index),
                self.weights[band_index] * self.kernels[band_index],
                self.dt,
            ),
            output_time_count=signals.shape[0] + self.kernel_length - 1,
        )
