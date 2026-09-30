"""FIR kernels and differentiable density filters without Meep imports.

Windowed-sinc and complementary FIR conventions follow TAMA's published
filter definitions. The complementary bank is approximate after finite FIR
truncation and its sampled power ripple is checked before it is returned.
"""

from numbers import Integral

import jax.numpy as jnp
import numpy as np

from .targets import _real_scalar


def _tap_count(taps):
    if isinstance(taps, (bool, np.bool_)) or not isinstance(taps, Integral) or taps < 1:
        raise ValueError("taps must be a positive integer")
    return int(taps)


def bandpass_kernel(low_hz, high_hz, dt, taps):
    """Hamming-windowed ideal bandpass, normalized to unit center gain.

    Returned coefficients require dt-scaled convolution, as performed by Band.
    Frequencies are Hz and dt is seconds. All frequencies must be below the
    temporal Nyquist frequency; FIR accuracy still depends on filter length.
    """
    low_hz = _real_scalar(low_hz, "low_hz", positive=True)
    high_hz = _real_scalar(high_hz, "high_hz", positive=True)
    dt = _real_scalar(dt, "dt", positive=True)
    taps = _tap_count(taps)
    if not low_hz < high_hz < 0.5 / dt:
        raise ValueError("frequencies must satisfy 0 < low_hz < high_hz < 0.5/dt")
    tau = (np.arange(taps) - (taps - 1) / 2) * dt
    kernel = (2 * high_hz * np.sinc(2 * high_hz * tau)
              - 2 * low_hz * np.sinc(2 * low_hz * tau)) * np.hamming(taps)
    gain = abs(dt * np.sum(kernel * np.exp(-2j * np.pi * (low_hz + high_hz) / 2 * tau)))
    if not np.isfinite(gain) or gain <= np.finfo(float).eps:
        raise ValueError("FIR center gain is too small to normalize")
    return kernel / gain


def power_complementary_kernels(bands_hz, dt, taps, transition_width_hz, *, max_power_error=0.01):
    """Approximately power-complementary real FIRs for contiguous Hz bands.

    Bands must be ordered by increasing frequency. Cosine/sine crossovers
    partition ideal power exactly, but finite firwin2 kernels have ripple.
    Construction rejects a bank whose sampled abs(sum(abs(H_b)**2)-1) exceeds
    max_power_error on the complete requested band interval, including internal
    crossovers and band edges. Outer stop-band transitions are excluded. The
    FFT check uses at least 16384 points and 64 points per tap, plus exact band
    edges and crossover endpoints. This is a dense numerical check, not a
    continuous-frequency proof. Objective priorities belong in Band.weight.
    """
    from scipy.signal import firwin2

    dt = _real_scalar(dt, "dt", positive=True)
    taps = _tap_count(taps)
    width = _real_scalar(transition_width_hz, "transition_width_hz", positive=True)
    tolerance = _real_scalar(max_power_error, "max_power_error", positive=True)
    if tolerance >= 1:
        raise ValueError("max_power_error must be less than one")
    bands = np.asarray(bands_hz)
    if bands.dtype.kind not in "iuf" or bands.ndim != 2 or bands.shape[1] != 2 or len(bands) < 2 or not np.all(np.isfinite(bands)):
        raise ValueError("bands_hz must contain at least two finite (low, high) pairs")
    if taps % 2 != 1 or np.any(bands[:, 1] - bands[:, 0] <= width):
        raise ValueError("use odd taps and transition width smaller than every band")
    if not np.allclose(bands[:-1, 1], bands[1:, 0], rtol=1e-12, atol=0):
        raise ValueError("frequency bands must be ordered and contiguous")
    if bands[0, 0] - width <= 0 or bands[-1, 1] + width >= 0.5 / dt:
        raise ValueError("outer transitions must stay between zero and Nyquist")
    count = 1 << int(np.ceil(np.log2(max(512, 2 * taps))))
    frequency = np.linspace(0, 0.5 / dt, count + 1)
    power = ((frequency[None, :] >= bands[:, :1]) & (frequency[None, :] <= bands[:, 1:])).astype(float)
    for index, boundary in enumerate(bands[:-1, 1]):
        selected = (frequency >= boundary - width / 2) & (frequency <= boundary + width / 2)
        position = (frequency[selected] - boundary + width / 2) / width
        power[index, selected] = np.cos(np.pi / 2 * position) ** 2
        power[index + 1, selected] = np.sin(np.pi / 2 * position) ** 2
    selected = (frequency >= bands[0, 0] - width) & (frequency < bands[0, 0])
    power[0, selected] = np.sin(np.pi / 2 * (frequency[selected] - bands[0, 0] + width) / width) ** 2
    selected = (frequency > bands[-1, 1]) & (frequency <= bands[-1, 1] + width)
    power[-1, selected] = np.cos(np.pi / 2 * (frequency[selected] - bands[-1, 1]) / width) ** 2
    kernels = tuple(firwin2(taps, frequency, np.sqrt(values), nfreqs=count + 1,
                           window=None, fs=1 / dt) / dt for values in power)
    fft_length = 1 << int(np.ceil(np.log2(max(16384, 64 * taps))))
    sampled_frequency = np.fft.rfftfreq(fft_length, d=dt)
    selected = (sampled_frequency >= bands[0, 0]) & (sampled_frequency <= bands[-1, 1])
    responses = np.fft.rfft(np.asarray(kernels) * dt, n=fft_length, axis=1)[:, selected]
    errors = np.abs(np.sum(np.abs(responses) ** 2, axis=0) - 1)
    critical = np.unique(np.concatenate((bands.ravel(), bands[:-1, 1] - width / 2,
                                        bands[:-1, 1] + width / 2)))
    exact_response = (np.asarray(kernels) * dt) @ np.exp(-2j * np.pi * dt * np.arange(taps)[:, None] * critical)
    critical_error = np.abs(np.sum(np.abs(exact_response) ** 2, axis=0) - 1)
    measured = float(max(errors.max(initial=0.), critical_error.max(initial=0.)))
    if not np.isfinite(measured) or measured > tolerance:
        raise ValueError(f"power-complementary FIR sampled power ripple {measured:.8g} exceeds "
                         f"max_power_error={tolerance:.8g}; increase taps or transition width")
    return kernels


def periodic_conic_filter(weights, *, radius, spacing, periodic_axes=None):
    """Normalized conic density filter with explicit circular boundary axes.

    Radius and spacing use the same length unit. Unselected axes are uncoupled.
    JAX differentiates the exact discrete FFT operator; no hand-coded transpose
    or straight-through estimator is used. This is smoothing, not a guarantee
    of minimum feature size.
    """
    weights = jnp.asarray(weights)
    if weights.ndim not in (2, 3) or not all(weights.shape) or not jnp.issubdtype(weights.dtype, jnp.floating):
        raise ValueError("weights must be a nonempty real floating-point 2D or 3D array")
    radius = _real_scalar(radius, "radius", positive=True)
    spacing = (spacing,) * weights.ndim if np.isscalar(spacing) else tuple(spacing)
    if len(spacing) != weights.ndim:
        raise ValueError("spacing must have one entry per array axis")
    spacing = tuple(_real_scalar(step, "spacing", positive=True) for step in spacing)
    axes = tuple(range(weights.ndim)) if periodic_axes is None else tuple(periodic_axes)
    if not axes or any(isinstance(a, (bool, np.bool_)) or not isinstance(a, Integral)
                       or not 0 <= a < weights.ndim for a in axes) or len(set(axes)) != len(axes):
        raise ValueError("periodic_axes must be unique valid integer axes")
    shape = tuple(n if axis in axes else 1 for axis, n in enumerate(weights.shape))
    distance_squared = np.zeros(shape)
    for axis in axes:
        index = np.arange(weights.shape[axis])
        reshape = [1] * weights.ndim
        reshape[axis] = weights.shape[axis]
        distance_squared += (np.minimum(index, weights.shape[axis] - index).reshape(reshape) * spacing[axis]) ** 2
    kernel = np.maximum(0., 1 - np.sqrt(distance_squared) / radius)
    kernel /= kernel.sum()
    transfer = jnp.fft.fftn(jnp.asarray(kernel, dtype=weights.dtype), axes=axes)
    return jnp.fft.ifftn(jnp.fft.fftn(weights, axes=axes) * transfer, axes=axes).real


def tanh_projection(weights, *, beta, eta=0.5):
    """Smooth density projection with exact JAX derivatives, beta=0 identity."""
    weights = jnp.asarray(weights)
    if not jnp.issubdtype(weights.dtype, jnp.floating):
        raise TypeError("weights must contain real floating-point values")
    beta = _real_scalar(beta, "beta")
    eta = _real_scalar(eta, "eta")
    if beta < 0 or not 0 <= eta <= 1:
        raise ValueError("beta must be nonnegative and eta must lie in [0, 1]")
    if beta <= np.sqrt(np.finfo(float).eps):
        return weights
    return (jnp.tanh(beta * eta) + jnp.tanh(beta * (weights - eta))) / (
        jnp.tanh(beta * eta) + jnp.tanh(beta * (1 - eta)))
