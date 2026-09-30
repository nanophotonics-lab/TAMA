import mmap
import os
import time
from typing import Callable, Dict, Optional, Sequence

import meep as mp
import numpy as np

from . import native_sampler
from .sampling_grid import (
    FastFieldGrid,
    NativeDesignField,
    _native_design_call,
)
from ...nyquist import (
    _limited_reconstruction_blas_threads,
    _validate_reconstruction_settings,
    _windowed_sinc_stencils,
)


def _discard_memmap_resident_pages(history, path: Optional[str]) -> bool:
    """Best-effort release of clean file-backed history pages.

    The history bytes remain on disk and are faulted back in on the next read.
    Unsupported platforms simply retain the resident pages.
    """
    history.flush()
    mapping = getattr(history, "_mmap", None)
    advice = getattr(mmap, "MADV_DONTNEED", None)
    if mapping is not None and advice is not None and hasattr(mapping, "madvise"):
        try:
            mapping.madvise(advice)
            return True
        except (OSError, TypeError, ValueError):
            pass

    posix_fadvise = getattr(os, "posix_fadvise", None)
    file_advice = getattr(os, "POSIX_FADV_DONTNEED", None)
    if path is None or posix_fadvise is None or file_advice is None:
        return False
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            posix_fadvise(descriptor, 0, 0, file_advice)
        finally:
            os.close(descriptor)
    except OSError:
        return False
    return True


class _NativeDesignHistorySet:
    """Own rank-local native forward plans and their on-disk histories."""

    def __init__(
        self,
        design,
        components: Sequence[int],
        history_dtype,
        make_history_memmap: Callable,
    ) -> None:
        self.design = design
        self.components = tuple(components)
        self.history_dtype = history_dtype
        self.make_history_memmap = make_history_memmap
        self.states: Dict[int, dict] = {}
        self.forward_fields: Dict[int, NativeDesignField] = {}

    def prepare_forward(
        self,
        sim: mp.Simulation,
        row_capacity: int,
    ) -> None:
        """Create all forward plans and fixed-capacity history arrays."""
        for component in self.components:
            if component not in self.forward_fields:
                self.forward_fields[component] = NativeDesignField(
                    sim,
                    component,
                    self.design,
                )
            if component not in self.states:
                native_field = self.forward_fields[component]
                history, path = self.make_history_memmap(
                    (row_capacity, max(native_field.local_size, 1))
                )
                self.states[component] = {
                    "array": history,
                    "path": path,
                    "width": native_field.local_size,
                    "signature": native_field.signature.copy(),
                }

    def sample_forward(
        self,
        sim: mp.Simulation,
        row_index: int,
        row_capacity: int,
    ) -> None:
        self.prepare_forward(sim, row_capacity)
        for component in self.components:
            state = self.states[component]
            if row_index >= state["array"].shape[0]:
                raise RuntimeError("forward field history buffer is too small")
            self.forward_fields[component].sample_into(
                state["array"][row_index, :state["width"]]
            )

    def native_bindings(self) -> tuple[tuple, tuple[np.ndarray, ...]]:
        """Return prepared native plans and their writable history arrays."""
        if set(self.states) != set(self.components):
            raise RuntimeError("forward native Yee field history is not prepared")
        return (
            tuple(
                self.forward_fields[component].ensure_plan()
                for component in self.components
            ),
            tuple(
                self.states[component]["array"][
                    :, :self.states[component]["width"]
                ]
                for component in self.components
            ),
        )

    def finish_forward(self, row_count: int) -> Dict[int, np.ndarray]:
        if set(self.states) != set(self.components):
            raise RuntimeError("forward native Yee field history was not recorded")
        histories = {}
        for component in self.components:
            state = self.states[component]
            histories[component] = state["array"][:row_count, :state["width"]]
            _discard_memmap_resident_pages(state["array"], state["path"])
        return histories

    def width(self, component: int) -> int:
        return int(self.states[component]["width"])

    @property
    def signatures(self) -> Dict[int, np.ndarray]:
        return {
            component: self.states[component]["signature"]
            for component in self.components
        }

    def release_forward(self) -> None:
        for native_field in self.forward_fields.values():
            native_field.release()
        self.forward_fields.clear()

    def cleanup_memmaps(self) -> None:
        for state in self.states.values():
            history = state.get("array")
            if history is not None:
                history.flush()
                state["array"] = None
            path = state.get("path")
            if path is not None and os.path.exists(path):
                os.unlink(path)
                state["path"] = None


def _centered_derivative_into(
    sample_into: Callable[[int, np.ndarray], None],
    index: int,
    last_index: int,
    dt: float,
    current: np.ndarray,
    neighbor: np.ndarray,
    destination: np.ndarray,
) -> np.ndarray:
    """Sample a history and write its endpoint/centered time derivative."""
    if index < 0 or index > last_index:
        raise IndexError("forward derivative index is outside the time grid")
    if index == 0:
        sample_into(0, current)
        sample_into(1, neighbor)
        np.subtract(neighbor, current, out=destination)
        np.divide(destination, dt, out=destination)
    elif index == last_index:
        sample_into(index, current)
        sample_into(index - 1, neighbor)
        np.subtract(current, neighbor, out=destination)
        np.divide(destination, dt, out=destination)
    else:
        sample_into(index + 1, current)
        sample_into(index - 1, neighbor)
        np.subtract(current, neighbor, out=destination)
        np.divide(destination, 2.0 * dt, out=destination)
    return destination


class _NativeDesignAccumulator:
    """Own signature-checked native adjoint plans and their local gradient."""

    def __init__(
        self,
        design,
        components: Sequence[int],
        signatures: Dict[int, np.ndarray],
        midpoint: bool,
    ) -> None:
        self.design = design
        self.components = tuple(components)
        self.signatures = signatures
        self.midpoint = bool(midpoint)
        self.fields: Dict[int, NativeDesignField] = {}
        self.local: Optional[np.ndarray] = None
        self._use_real: Optional[bool] = None

    def accumulate(
        self,
        sim: mp.Simulation,
        component: int,
        forward_derivative: np.ndarray,
    ) -> None:
        if component not in self.fields:
            self.fields[component] = NativeDesignField(
                sim,
                component,
                self.design,
                expected_signature=self.signatures[component],
            )
        native_field = self.fields[component]
        derivative = np.asarray(forward_derivative)
        local_use_real = native_field.fields_are_real and np.issubdtype(
            derivative.dtype,
            np.floating,
        )
        if self._use_real is None:
            real_ranks = int(local_use_real)
            nproc = mp.count_processors()
            if nproc > 1:
                real_ranks = int(mp.sum_to_all(real_ranks))
            if real_ranks not in (0, nproc):
                raise RuntimeError(
                    "native gradient accumulation dtype differs across active MPI ranks"
                )
            self._use_real = real_ranks == nproc
        elif local_use_real != self._use_real:
            raise RuntimeError(
                "native gradient accumulation dtype changed during the adjoint run"
            )
        if self.local is None:
            self.local = np.zeros(
                self.design.shape,
                dtype=np.float64 if self._use_real else np.complex128,
            )
        if self.midpoint:
            native_field.accumulate_midpoint(derivative, self.local)
        else:
            native_field.accumulate(derivative, self.local)

    def run_segment(
        self,
        sim: mp.Simulation,
        field_histories: Dict[int, np.ndarray],
        reconstruction_weights: np.ndarray,
        reconstruction_first_offset: int,
        sampling_interval: int,
        fine_step_count: int,
        start_fine_index: int,
        sample_count: int,
        dt: float,
    ) -> float:
        """Run native adjoint accumulation and stepping for one time segment."""
        setup_start = time.perf_counter()
        local_use_real = True
        ordered_histories = []
        ordered_plans = []
        for component in self.components:
            if component not in self.fields:
                self.fields[component] = NativeDesignField(
                    sim,
                    component,
                    self.design,
                    expected_signature=self.signatures[component],
                )
            native_field = self.fields[component]
            history = np.asarray(field_histories[component])
            local_use_real = local_use_real and (
                native_field.fields_are_real
                and np.issubdtype(history.dtype, np.floating)
            )
            ordered_histories.append(history)
            ordered_plans.append(native_field.ensure_plan())

        if self._use_real is None:
            real_ranks = int(local_use_real)
            nproc = mp.count_processors()
            if nproc > 1:
                real_ranks = int(mp.sum_to_all(real_ranks))
            if real_ranks not in (0, nproc):
                raise RuntimeError(
                    "native gradient accumulation dtype differs across active MPI ranks"
                )
            self._use_real = real_ranks == nproc
        elif local_use_real != self._use_real:
            raise RuntimeError(
                "native gradient accumulation dtype changed during the adjoint run"
            )
        if self.local is None:
            self.local = np.zeros(
                self.design.shape,
                dtype=np.float64 if self._use_real else np.complex128,
            )

        setup_seconds = time.perf_counter() - setup_start
        return setup_seconds + float(
            _native_design_call(
                "native adjoint segment",
                native_sampler.run_native_design_adjoint_segment,
                tuple(ordered_plans),
                tuple(ordered_histories),
                np.ascontiguousarray(
                    reconstruction_weights,
                    dtype=np.float64,
                ),
                self.local.reshape(-1),
                int(sampling_interval),
                int(reconstruction_first_offset),
                int(fine_step_count),
                int(start_fine_index),
                int(sample_count),
                float(dt),
                bool(self.midpoint),
            )
        )

    def reduce(self) -> np.ndarray:
        if self.local is None:
            self.local = np.zeros(self.design.shape, dtype=np.complex128)
        local_matrix = self.local.reshape(self.local.shape[0], -1)
        if self.local.dtype == np.float64:
            reduced = FastFieldGrid.reduce_real_inplace(local_matrix)
        else:
            reduced = FastFieldGrid.reduce_inplace(local_matrix)
        return reduced.reshape(self.local.shape)

    def release(self) -> None:
        for native_field in self.fields.values():
            native_field.release()
        self.fields.clear()


def _native_forward_step_count(
    sim: mp.Simulation,
    run_until: float,
    sampling_interval: int,
) -> int:
    """Return Meep's exact forward step count and validate history alignment."""
    fields = getattr(sim, "fields", None)
    if fields is None or not hasattr(fields, "this"):
        raise RuntimeError("native forward sampling requires initialized Meep fields")
    fine_step_count = int(
        _native_design_call(
            "native forward step-count query",
            native_sampler.native_forward_step_count,
            int(fields.this),
            float(run_until),
        )
    )
    if fine_step_count < 1 or fine_step_count % sampling_interval != 0:
        raise RuntimeError(
            "native forward step count is not aligned with the sampling interval"
        )
    return fine_step_count


def _run_native_forward_loop(
    sim: mp.Simulation,
    monitor_bindings,
    native_history: Optional[_NativeDesignHistorySet],
    *,
    run_until: float,
    sampling_interval: int,
    fine_step_count: int,
) -> tuple[int, int, np.ndarray, float]:
    """Run forward sampling and stepping without a per-step Python callback."""
    setup_start = time.perf_counter()
    fields = getattr(sim, "fields", None)
    if fields is None or not hasattr(fields, "this"):
        raise RuntimeError("native forward sampling requires initialized Meep fields")
    if fine_step_count < 1 or fine_step_count % sampling_interval != 0:
        raise RuntimeError(
            "native forward step count is not aligned with the sampling interval"
        )
    monitor_times = np.empty(fine_step_count + 1, dtype=np.float64)
    monitor_plans = tuple(
        monitor.ensure_plan() for monitor, _ in monitor_bindings
    )
    monitor_histories = tuple(
        np.asarray(history) for _, history in monitor_bindings
    )
    if native_history is None:
        design_plans = ()
        design_histories = ()
    else:
        native_history.prepare_forward(
            sim,
            fine_step_count // sampling_interval + 1,
        )
        design_plans, design_histories = native_history.native_bindings()

    state = {
        "samples": 0,
        "extra_seconds": time.perf_counter() - setup_start,
    }

    def native_segment(_sim):
        if state["samples"] == 0:
            start_fine_index = 0
            sample_count = fine_step_count
        elif state["samples"] == fine_step_count:
            start_fine_index = fine_step_count
            sample_count = 1
        else:
            raise RuntimeError(
                "native forward callback count exceeds the forward time grid"
            )
        state["extra_seconds"] += float(
            _native_design_call(
                "native forward segment",
                native_sampler.run_native_forward_segment,
                monitor_plans,
                monitor_histories,
                design_plans,
                design_histories,
                int(sampling_interval),
                fine_step_count,
                start_fine_index,
                sample_count,
                monitor_times,
            )
        )
        state["samples"] += sample_count

    sim.run(native_segment, until=run_until)
    if state["samples"] != fine_step_count + 1:
        raise RuntimeError(
            "native forward callback count does not match the forward time grid"
        )
    return (
        fine_step_count + 1,
        fine_step_count // sampling_interval + 1,
        monitor_times,
        float(state["extra_seconds"]),
    )


def _run_native_adjoint_loop(
    sim: mp.Simulation,
    accumulator: _NativeDesignAccumulator,
    field_histories: Dict[int, np.ndarray],
    *,
    fine_step_count: int,
    actual_time: float,
    dt: float,
    sampling_interval: int,
    reconstruction_half_width: int,
    reconstruction_window,
    reconstruction_window_params,
) -> float:
    """Run the adjoint loop with reconstruction and VJP work in C++."""
    if sampling_interval > 1:
        (
            reconstruction_window,
            reconstruction_window_params,
            reconstruction_half_width,
        ) = _validate_reconstruction_settings(
            reconstruction_window,
            reconstruction_window_params,
            reconstruction_half_width,
        )
        (
            reconstruction_first_offset,
            reconstruction_weights,
        ) = _windowed_sinc_stencils(
            sampling_interval,
            reconstruction_half_width,
            reconstruction_window,
            reconstruction_window_params,
        )
    else:
        reconstruction_weights = np.ones((1, 1), dtype=np.float64)
        reconstruction_first_offset = 0

    state = {"samples": 0, "extra_seconds": 0.0}

    def native_segment(_sim):
        if state["samples"] == 0:
            start_fine_index = fine_step_count
            sample_count = fine_step_count
        elif state["samples"] == fine_step_count:
            start_fine_index = 0
            sample_count = 1
        else:
            raise RuntimeError(
                "native adjoint callback count exceeds the forward time grid"
            )
        state["extra_seconds"] += accumulator.run_segment(
            _sim,
            field_histories,
            reconstruction_weights,
            reconstruction_first_offset,
            sampling_interval,
            fine_step_count,
            start_fine_index,
            sample_count,
            dt,
        )
        state["samples"] += sample_count

    with _limited_reconstruction_blas_threads():
        sim.run(native_segment, until=actual_time)
    if state["samples"] != fine_step_count + 1:
        raise RuntimeError(
            "native adjoint callback count does not match the forward time grid"
        )
    return float(state["extra_seconds"])
