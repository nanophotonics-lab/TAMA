"""Full FIR primitives, independent of FDTD and objective scalarization."""

import inspect

import numpy as np
import pytest
from scipy.signal import convolve

import tama.multi_tda_objective as module


@pytest.mark.parametrize("shape", [(1,), (3, 2), (17, 2, 3)])
@pytest.mark.parametrize("length", [1, 2, 5, 20])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_full_matches_direct_convolution_and_hermitian_adjoint(shape, length, dtype):
    rng = np.random.default_rng(481)
    x = rng.standard_normal(shape)
    kernel = rng.standard_normal(length)
    qshape = (shape[0] + length - 1,) + shape[1:]
    q = rng.standard_normal(qshape)
    if np.issubdtype(dtype, np.complexfloating):
        x = x + 1j * rng.standard_normal(shape)
        kernel = kernel + 1j * rng.standard_normal(length)
        q = q + 1j * rng.standard_normal(qshape)
    x, kernel, q = (array.astype(dtype) for array in (x, kernel, q))
    dt = 0.125
    h = kernel.reshape((length,) + (1,) * (x.ndim - 1))
    expected = convolve(x, h, mode="full", method="direct") * dt
    expected_adjoint = convolve(q, h[::-1].conj(), mode="valid", method="direct") * dt
    actual = module.temporal_convolve_signal(x, kernel, dt)
    adjoint = module.temporal_convolve_signal_transpose(q, kernel, dt)
    tol = 2e-5 if np.dtype(dtype).itemsize <= 4 or dtype is np.complex64 else 2e-12
    assert actual.shape == qshape and adjoint.shape == shape
    assert actual.dtype == x.dtype and adjoint.dtype == q.dtype
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)
    np.testing.assert_allclose(adjoint, expected_adjoint, rtol=tol, atol=tol)
    np.testing.assert_allclose(np.vdot(actual, q), np.vdot(x, adjoint), rtol=tol, atol=tol)


def test_helpers_have_no_output_mode_parameter():
    for operation in (module.temporal_convolve_signal, module.temporal_convolve_signal_transpose):
        assert tuple(inspect.signature(operation).parameters) == ("signal", "kernel", "dt")


@pytest.mark.parametrize("dtype", [np.bool_, np.int64])
def test_full_nonfloating_input_is_promoted_without_integer_roundoff(dtype):
    x = np.array([1, 0, 1], dtype=dtype)
    kernel = np.array([0.2, -0.3])
    actual = module.temporal_convolve_signal(x, kernel, 1)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, np.convolve(x, kernel), atol=1e-15)
    q = np.array([1, 0, 1, 1], dtype=dtype)
    adjoint = module.temporal_convolve_signal_transpose(q, kernel, 1)
    assert adjoint.dtype == np.float64
    np.testing.assert_allclose(adjoint, np.convolve(q, kernel[::-1], "valid"), atol=1e-15)


def test_real_signal_complex_kernel():
    x, h = np.arange(5.0), np.array([1 + 2j, 3 - 4j])
    actual = module.temporal_convolve_signal(x, h, 0.1)
    np.testing.assert_allclose(actual, np.convolve(x, h) * 0.1, atol=1e-14)
    q = np.arange(6.0)
    adjoint = module.temporal_convolve_signal_transpose(q, h, 0.1)
    np.testing.assert_allclose(adjoint, np.convolve(q, h[::-1].conj(), "valid") * 0.1)


@pytest.mark.parametrize("shape", [(3, 0), (3, 2, 0)])
def test_zero_width_channels(shape):
    x = np.empty(shape)
    y = module.temporal_convolve_signal(x, np.ones(4), 1)
    assert y.shape == (6,) + shape[1:]
    assert module.temporal_convolve_signal_transpose(y, np.ones(4), 1).shape == shape


def test_empty_time_axis_and_invalid_inputs():
    for operation in (module.temporal_convolve_signal, module.temporal_convolve_signal_transpose):
        with pytest.raises(ValueError, match="full"):
            operation(np.empty((0, 2)), np.ones(3), 1)
        with pytest.raises(ValueError, match="full"):
            operation(np.array(1.0), np.ones(3), 1)
        with pytest.raises(ValueError, match="one-dimensional"):
            operation(np.ones(4), np.ones((1, 2)), 1)
        with pytest.raises(ValueError, match="empty"):
            operation(np.ones(4), np.array([]), 1)
    with pytest.raises(ValueError, match="at least"):
        module.temporal_convolve_signal_transpose(np.ones(2), np.ones(3), 1)


def test_full_fft_workspace_is_split_into_channel_blocks(monkeypatch):
    rng = np.random.default_rng(482)
    x, h = rng.standard_normal((16384, 80)), np.array([0.3, -0.2, 0.7, 0.1])
    calls = []
    original = module.rfft

    def tracked(values, *args, **kwargs):
        if values.ndim == 2:
            calls.append(values.shape[1])
        return original(values, *args, **kwargs)

    monkeypatch.setattr(module, "rfft", tracked)
    actual = module.temporal_convolve_signal(x, h, 1)
    assert len(calls) > 1 and sum(calls) == x.shape[1]
    expected = convolve(x, h[:, None], mode="full", method="direct")
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
