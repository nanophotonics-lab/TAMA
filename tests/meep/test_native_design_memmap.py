import numpy as np
import pytest

import tama.native_design as native_design


class _FakeMapping:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def madvise(self, advice):
        self.calls.append(advice)
        if self.error is not None:
            raise self.error


class _FakeHistory:
    def __init__(self, mapping):
        self._mmap = mapping
        self.flush_count = 0

    def flush(self):
        self.flush_count += 1


def test_discard_memmap_pages_flushes_before_madvise(monkeypatch):
    monkeypatch.setattr(
        native_design.mmap,
        "MADV_DONTNEED",
        123,
        raising=False,
    )
    mapping = _FakeMapping()
    history = _FakeHistory(mapping)

    released = native_design._discard_memmap_resident_pages(
        history,
        "/unused/history.dat",
    )

    assert released is True
    assert history.flush_count == 1
    assert mapping.calls == [123]


def test_discard_memmap_pages_falls_back_to_posix_fadvise(monkeypatch):
    monkeypatch.setattr(
        native_design.mmap,
        "MADV_DONTNEED",
        123,
        raising=False,
    )
    monkeypatch.setattr(
        native_design.os,
        "POSIX_FADV_DONTNEED",
        456,
        raising=False,
    )
    mapping = _FakeMapping(OSError("unsupported advice"))
    history = _FakeHistory(mapping)
    calls = []
    monkeypatch.setattr(native_design.os, "open", lambda path, flags: 17)
    monkeypatch.setattr(
        native_design.os,
        "close",
        lambda descriptor: calls.append(("close", descriptor)),
    )
    monkeypatch.setattr(
        native_design.os,
        "posix_fadvise",
        lambda descriptor, offset, length, advice: calls.append(
            ("fadvise", descriptor, offset, length, advice)
        ),
    )

    released = native_design._discard_memmap_resident_pages(
        history,
        "/history.dat",
    )

    assert released is True
    assert history.flush_count == 1
    assert mapping.calls == [123]
    assert calls == [
        ("fadvise", 17, 0, 0, 456),
        ("close", 17),
    ]


def test_discard_memmap_pages_is_noop_when_platform_has_no_advice(monkeypatch):
    monkeypatch.delattr(native_design.mmap, "MADV_DONTNEED", raising=False)
    monkeypatch.delattr(native_design.os, "posix_fadvise", raising=False)
    history = _FakeHistory(mapping=None)

    released = native_design._discard_memmap_resident_pages(
        history,
        "/history.dat",
    )

    assert released is False
    assert history.flush_count == 1


def test_finish_forward_preserves_memmap_values_after_discard(tmp_path):
    path = tmp_path / "history.dat"
    history = np.memmap(path, mode="w+", dtype=np.float64, shape=(8, 5))
    expected = np.arange(history.size, dtype=np.float64).reshape(history.shape)
    history[:] = expected
    owner = native_design._NativeDesignHistorySet(
        design=None,
        components=(1,),
        history_dtype=np.float64,
        make_history_memmap=lambda shape: (history, str(path)),
    )
    owner.states[1] = {
        "array": history,
        "path": str(path),
        "width": 3,
        "signature": np.empty((0, 2), dtype=np.int64),
    }

    result = owner.finish_forward(6)[1]

    np.testing.assert_array_equal(result, expected[:6, :3])
    np.testing.assert_array_equal(
        np.memmap(path, mode="r", dtype=np.float64, shape=(8, 5)),
        expected,
    )
    del result
    owner.cleanup_memmaps()
    assert not path.exists()


def test_material_plan_transfer_checks_fields_and_signature(monkeypatch):
    from types import SimpleNamespace

    from tama.backends.meep.sampling_grid import NativeDesignField

    class Simulation:
        pass

    class Fields:
        pass

    sim = Simulation()
    sim.fields = Fields()
    field = object.__new__(NativeDesignField)
    field._bind_native_plan_fields(sim)
    field.material_jacobian = True
    field.signature = np.array([[1, 2, 0, -1]])
    field.plan = object()
    owner = native_design._NativeDesignHistorySet(None, (1,), np.float64, None)
    owner.forward_fields[1] = field
    owner.states[1] = {"signature": field.signature.copy()}
    monkeypatch.setattr(native_design, "_native_design_call", lambda _, fn: fn())

    with pytest.raises(RuntimeError, match="different Meep fields"):
        owner.take_material_plans(SimpleNamespace(fields=Fields()))
    owner.states[1]["signature"][0, 0] = 8
    with pytest.raises(RuntimeError, match="signature"):
        owner.take_material_plans(sim)
    owner.states[1]["signature"] = field.signature.copy()
    assert owner.take_material_plans(sim) == {1: field}
    assert owner.forward_fields == {}
