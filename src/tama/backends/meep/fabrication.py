"""Fabrication filters for periodic MaterialGrid design variables."""

import numpy as np


_IDENTITY_PROJECTION_BETA = np.sqrt(np.finfo(float).eps)


def _real_array(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim not in (2, 3):
        raise ValueError(f"{name} must be a 2D or 3D array")
    if np.iscomplexobj(array):
        raise TypeError(f"{name} must be real")
    try:
        array = np.asarray(array, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _positive_float(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(np.asarray(value)):
        raise TypeError(f"{name} must be a real scalar")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real scalar") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _filter_settings(
    shape: tuple[int, ...],
    radius,
    spacing,
    periodic_axes,
) -> tuple[float, tuple[float, ...], tuple[int, ...]]:
    radius = _positive_float(radius, "radius")
    if np.isscalar(spacing):
        spacing = (_positive_float(spacing, "spacing"),) * len(shape)
    else:
        try:
            spacing = tuple(spacing)
        except TypeError as exc:
            raise TypeError("spacing must be a real scalar or sequence")
        if len(spacing) != len(shape):
            raise ValueError("spacing must have one entry per array axis")
        spacing = tuple(
            _positive_float(value, f"spacing[{axis}]")
            for axis, value in enumerate(spacing)
        )

    if periodic_axes is None:
        periodic_axes = tuple(range(len(shape)))
    else:
        try:
            periodic_axes = tuple(periodic_axes)
        except TypeError as exc:
            raise TypeError("periodic_axes must be a sequence of axes") from exc
        if not periodic_axes:
            raise ValueError("periodic_axes must not be empty")
        if any(
            isinstance(axis, (bool, np.bool_))
            or not isinstance(axis, (int, np.integer))
            for axis in periodic_axes
        ):
            raise TypeError("periodic_axes entries must be integers")
        periodic_axes = tuple(int(axis) for axis in periodic_axes)
        if any(axis < 0 or axis >= len(shape) for axis in periodic_axes):
            raise ValueError("periodic_axes entries must index an array axis")
        if len(set(periodic_axes)) != len(periodic_axes):
            raise ValueError("periodic_axes entries must be unique")
        periodic_axes = tuple(sorted(periodic_axes))
    return radius, spacing, periodic_axes


def _periodic_conic_transfer(
    shape: tuple[int, ...],
    radius: float,
    spacing: tuple[float, ...],
    periodic_axes: tuple[int, ...],
) -> np.ndarray:
    kernel_shape = tuple(
        shape[axis] if axis in periodic_axes else 1
        for axis in range(len(shape))
    )
    squared_distance = np.zeros(kernel_shape, dtype=float)
    for axis in periodic_axes:
        indices = np.arange(shape[axis], dtype=float)
        wrapped_distance = np.minimum(indices, shape[axis] - indices)
        reshape = [1] * len(shape)
        reshape[axis] = shape[axis]
        squared_distance += (
            wrapped_distance.reshape(reshape) * spacing[axis]
        ) ** 2
    kernel = np.maximum(0.0, 1.0 - np.sqrt(squared_distance) / radius)
    kernel /= np.sum(kernel)
    return np.fft.fftn(kernel, axes=periodic_axes)


def _apply_periodic_conic_filter(
    values,
    *,
    radius,
    spacing,
    periodic_axes,
    transpose: bool,
) -> np.ndarray:
    values = _real_array(values, "cotangent" if transpose else "weights")
    radius, spacing, periodic_axes = _filter_settings(
        values.shape,
        radius,
        spacing,
        periodic_axes,
    )
    transfer = _periodic_conic_transfer(
        values.shape,
        radius,
        spacing,
        periodic_axes,
    )
    if transpose:
        transfer = np.conjugate(transfer)
    transformed = np.fft.fftn(values, axes=periodic_axes)
    return np.fft.ifftn(
        transformed * transfer,
        axes=periodic_axes,
    ).real


def periodic_conic_filter(
    weights,
    *,
    radius,
    spacing,
    periodic_axes=None,
) -> np.ndarray:
    """Apply a normalized conic filter with circular periodic boundaries.

    The radius and spacing use the same physical length unit. Only axes in
    ``periodic_axes`` are filtered; the other axes are left uncoupled. Thus a
    3D full-cell design can use ``periodic_axes=(0, 1)`` to filter each z layer
    independently while wrapping across the x and y unit-cell seams.

    Args:
        weights: Real 2D or 3D design-variable array.
        radius: Positive physical support radius of the conic kernel.
        spacing: Positive scalar or one positive value per array axis.
        periodic_axes: Filtered circular-convolution axes. The default filters
            and wraps every axis.

    Returns:
        Filtered float array with the same shape as ``weights``.
    """
    return _apply_periodic_conic_filter(
        weights,
        radius=radius,
        spacing=spacing,
        periodic_axes=periodic_axes,
        transpose=False,
    )


def periodic_conic_filter_vjp(
    cotangent,
    *,
    radius,
    spacing,
    periodic_axes=None,
) -> np.ndarray:
    """Apply the exact discrete transpose of :func:`periodic_conic_filter`.

    The normalized conic kernel is real and centrosymmetric, so its transpose
    has the same numerical action as the forward filter. This function applies
    the conjugated Fourier transfer explicitly instead of relying on that
    symmetry.
    """
    return _apply_periodic_conic_filter(
        cotangent,
        radius=radius,
        spacing=spacing,
        periodic_axes=periodic_axes,
        transpose=True,
    )


def _projection_settings(beta, eta) -> tuple[float, float]:
    if isinstance(beta, (bool, np.bool_)) or np.iscomplexobj(np.asarray(beta)):
        raise TypeError("beta must be a real scalar")
    if isinstance(eta, (bool, np.bool_)) or np.iscomplexobj(np.asarray(eta)):
        raise TypeError("eta must be a real scalar")
    try:
        beta = float(beta)
        eta = float(eta)
    except (TypeError, ValueError) as exc:
        raise TypeError("beta and eta must be real scalars") from exc
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError("beta must be finite and nonnegative")
    if not np.isfinite(eta) or not 0.0 <= eta <= 1.0:
        raise ValueError("eta must be finite and lie in [0, 1]")
    return beta, eta


def tanh_projection(weights, *, beta, eta=0.5) -> np.ndarray:
    """Apply the standard density-based hyperbolic-tangent projection."""
    weights = _real_array(weights, "weights")
    beta, eta = _projection_settings(beta, eta)
    if beta <= _IDENTITY_PROJECTION_BETA:
        return weights.copy()
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return (
        np.tanh(beta * eta) + np.tanh(beta * (weights - eta))
    ) / denominator


def tanh_projection_vjp(
    weights,
    cotangent,
    *,
    beta,
    eta=0.5,
) -> np.ndarray:
    """Apply the projection VJP at ``weights`` to ``cotangent``."""
    weights = _real_array(weights, "weights")
    cotangent = _real_array(cotangent, "cotangent")
    if cotangent.shape != weights.shape:
        raise ValueError("cotangent must match weights.shape")
    beta, eta = _projection_settings(beta, eta)
    if beta <= _IDENTITY_PROJECTION_BETA:
        return cotangent.copy()
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    shifted_tanh = np.tanh(beta * (weights - eta))
    derivative = beta * (1.0 - shifted_tanh**2) / denominator
    return cotangent * derivative
