"""Shared fakes for TDA and MultiTDA objective tests."""

import numpy as np

from tama.native_design import _forward_derivative_into
from tama.nyquist import _WindowedSincReconstructor


class _FakeFields:
    def __init__(self, dt):
        self.dt = dt
        self.reset_timer_calls = 0

    def reset_timers(self):
        self.reset_timer_calls += 1


def _run_fake_adjoint_loop(
    sim,
    accumulator,
    field_histories,
    *,
    fine_step_count,
    actual_time,
    dt,
    sampling_interval,
    reconstruction_half_width,
    reconstruction_window,
    reconstruction_window_params,
):
    reconstructors = {}
    buffers = {}
    if sampling_interval > 1:
        for component, history in field_histories.items():
            reconstructor = _WindowedSincReconstructor(
                history,
                sampling_interval,
                half_width=reconstruction_half_width,
                window=reconstruction_window,
                window_params=reconstruction_window_params,
            )
            reconstructors[component] = reconstructor
            buffers[component] = tuple(
                np.empty(reconstructor.history.shape[1], dtype=reconstructor.dtype)
                for _ in range(3)
            )

    status = {"step": 0}

    def accumulate(active_sim):
        fine_index = fine_step_count - status["step"]
        if fine_index < 0:
            raise RuntimeError("adjoint callback count exceeds the forward time grid")
        for component, history in field_histories.items():
            if sampling_interval > 1:
                current, neighbor, derivative = buffers[component]
                forward_derivative = _forward_derivative_into(
                    reconstructors[component].sample_into,
                    fine_index,
                    fine_step_count,
                    dt,
                    current,
                    neighbor,
                    derivative,
                )
            elif fine_index == fine_step_count:
                forward_derivative = np.zeros_like(history[-1])
            else:
                forward_derivative = (
                    history[fine_index + 1] - history[fine_index]
                ) / dt
            accumulator.accumulate(
                active_sim,
                component,
                forward_derivative,
            )
        status["step"] += 1

    sim.run(accumulate, until=actual_time)
    if status["step"] != fine_step_count + 1:
        raise RuntimeError(
            "adjoint callback count does not match the forward time grid"
        )
    return 0.0


def _fake_indexed_transpose_stencil(monitor):
    return (
        np.arange(len(monitor.positions) + 1, dtype=np.intp),
        tuple(object() for _ in monitor.positions),
        np.ones(len(monitor.positions), dtype=float),
    )
