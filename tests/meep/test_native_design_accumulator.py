import numpy as np
import pytest

import tama.native_design as native_design


class _Design:
    shape = (2, 3)


class _Simulation:
    def __init__(self, fields_are_real):
        self.fields_are_real = fields_are_real


class _NativeField:
    def __init__(
        self,
        sim,
        component,
        design,
        expected_signature=None,
    ):
        del component, design, expected_signature
        self.fields_are_real = sim.fields_are_real

    def accumulate(self, forward_values, accumulator):
        accumulator += np.sum(forward_values)

    def accumulate_midpoint(self, forward_values, accumulator):
        accumulator += np.sum(forward_values)
        return True

    def release(self):
        pass


def _accumulator(midpoint=False):
    return native_design._NativeDesignAccumulator(
        _Design(),
        (1,),
        {1: np.array([[0, 0]], dtype=np.int64)},
        midpoint,
    )


def test_native_accumulator_uses_real_storage_and_reduction(monkeypatch):
    monkeypatch.setattr(native_design, "NativeDesignField", _NativeField)
    reduced_dtypes = []

    def reduce_real(array):
        reduced_dtypes.append(array.dtype)
        return array

    monkeypatch.setattr(
        native_design.FastFieldGrid,
        "reduce_real_inplace",
        reduce_real,
    )
    accumulator = _accumulator()
    accumulator.accumulate(
        _Simulation(True),
        1,
        np.arange(4, dtype=np.float64),
    )

    assert accumulator.local.dtype == np.float64
    assert accumulator.reduce().dtype == np.float64
    assert reduced_dtypes == [np.dtype(np.float64)]


def test_native_accumulator_retains_complex_fallback(monkeypatch):
    monkeypatch.setattr(native_design, "NativeDesignField", _NativeField)
    reduced_dtypes = []

    def reduce_complex(array):
        reduced_dtypes.append(array.dtype)
        return array

    monkeypatch.setattr(
        native_design.FastFieldGrid,
        "reduce_inplace",
        reduce_complex,
    )
    accumulator = _accumulator(midpoint=True)
    accumulator.accumulate(
        _Simulation(False),
        1,
        np.arange(4, dtype=np.complex128) + 1j,
    )

    assert accumulator.local.dtype == np.complex128
    assert accumulator.reduce().dtype == np.complex128
    assert reduced_dtypes == [np.dtype(np.complex128)]


def test_native_accumulator_rejects_rank_inconsistent_dtype(monkeypatch):
    monkeypatch.setattr(native_design, "NativeDesignField", _NativeField)
    monkeypatch.setattr(native_design.mp, "count_processors", lambda: 2)
    monkeypatch.setattr(native_design.mp, "sum_to_all", lambda value: 1)
    accumulator = _accumulator()

    with pytest.raises(RuntimeError, match="differs across active MPI ranks"):
        accumulator.accumulate(
            _Simulation(True),
            1,
            np.arange(4, dtype=np.float64),
        )
