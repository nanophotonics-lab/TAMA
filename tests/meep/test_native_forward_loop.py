import meep as mp
import numpy as np
import pytest

import tama as tm
from tama import native_sampler
from tama.sampling_grid import FastPointMonitor, NativeDesignField


_RUN_UNTIL = 0.35
_DT = 0.05
_MONITOR_POSITIONS = (
    mp.Vector3(-0.8, 0.1),
    mp.Vector3(-0.65, 0.1),
)


def _make_simulation(force_complex_fields):
    air = mp.Medium(epsilon=1.0)
    dielectric = mp.Medium(epsilon=2.0)
    material_grid = mp.MaterialGrid(
        mp.Vector3(2, 2),
        air,
        dielectric,
        do_averaging=False,
    )
    design_center = mp.Vector3()
    design_size = mp.Vector3(0.8, 0.8)
    design = tm.DesignGrid(
        material_grid=material_grid,
        center=design_center,
        size=design_size,
        shape=(2, 2),
        background=air,
        design_material=dielectric,
    )
    simulation = mp.Simulation(
        cell_size=mp.Vector3(3.0, 3.0),
        resolution=10,
        geometry=[
            mp.Block(
                center=design_center,
                size=design_size,
                material=material_grid,
            )
        ],
        sources=[
            mp.Source(
                mp.ContinuousSource(frequency=0.8),
                component=mp.Ez,
                center=_MONITOR_POSITIONS[0],
                amplitude=1.0 + 0.5j if force_complex_fields else 1.0,
            )
        ],
        force_complex_fields=force_complex_fields,
        eps_averaging=False,
    )
    return simulation, design


def _make_custom_source_simulation(source_callback):
    simulation, design = _make_simulation(False)
    simulation.sources = [
        mp.Source(
            mp.CustomSource(src_func=source_callback),
            component=mp.Ez,
            center=_MONITOR_POSITIONS[0],
        )
    ]
    return simulation, design


def _python_forward_reference(force_complex_fields, sampling_interval):
    simulation, design = _make_simulation(force_complex_fields)
    dtype = np.complex128 if force_complex_fields else np.float64
    state = {"monitor": None, "design": None}
    monitor_rows = []
    design_rows = []
    monitor_times = []
    signature = None
    actual_time = None

    def record(active_simulation):
        if state["monitor"] is None:
            state["monitor"] = FastPointMonitor(
                active_simulation,
                mp.Ez,
                _MONITOR_POSITIONS,
            )
            state["design"] = NativeDesignField(
                active_simulation,
                mp.Ez,
                design,
            )
        sample_index = len(monitor_times)
        monitor_row = np.empty(len(_MONITOR_POSITIONS), dtype=dtype)
        state["monitor"].sample_history_into(monitor_row)
        monitor_rows.append(monitor_row)
        if sample_index % sampling_interval == 0:
            design_row = np.empty(state["design"].local_size, dtype=dtype)
            state["design"].sample_into(design_row)
            design_rows.append(design_row)
        monitor_times.append(active_simulation.meep_time())

    try:
        simulation.run(record, until=_RUN_UNTIL)
        actual_time = simulation.round_time()
        signature = state["design"].signature.copy()
    finally:
        if state["design"] is not None:
            state["design"].release()
        if state["monitor"] is not None:
            state["monitor"].plan = None
        simulation.reset_meep()

    return {
        "monitor_history": np.asarray(monitor_rows, dtype=dtype),
        "design_history": np.asarray(design_rows, dtype=dtype),
        "monitor_times": np.asarray(monitor_times, dtype=float),
        "fine_step_count": len(monitor_times) - 1,
        "actual_time": actual_time,
        "design_signature": signature,
        "python_callback_count": len(monitor_times),
    }


def _native_forward(force_complex_fields, sampling_interval):
    simulation, design = _make_simulation(force_complex_fields)
    dtype = np.complex128 if force_complex_fields else np.float64
    monitor = None
    design_field = None
    result = None
    simulation.init_sim()
    try:
        assert simulation.fields.dt == pytest.approx(_DT)
        monitor = FastPointMonitor(simulation, mp.Ez, _MONITOR_POSITIONS)
        design_field = NativeDesignField(simulation, mp.Ez, design)
        fine_step_count = int(
            native_sampler.native_forward_step_count(
                int(simulation.fields.this),
                _RUN_UNTIL,
            )
        )
        monitor_history = np.empty(
            (fine_step_count + 1, len(_MONITOR_POSITIONS)),
            dtype=dtype,
        )
        design_history = np.empty(
            (fine_step_count // sampling_interval + 1, design_field.local_size),
            dtype=dtype,
        )
        monitor_times = np.empty(fine_step_count + 1, dtype=np.float64)
        state = {"callback_count": 0, "extra_seconds": 0.0}

        def native_segment(active_simulation):
            callback_index = state["callback_count"]
            if callback_index == 0:
                start_index = 0
                sample_count = fine_step_count
            elif callback_index == 1:
                start_index = fine_step_count
                sample_count = 1
            else:
                raise AssertionError("native forward trampoline ran too many times")
            state["extra_seconds"] += float(
                native_sampler.run_native_forward_segment(
                    (monitor.ensure_plan(),),
                    (monitor_history,),
                    (design_field.ensure_plan(),),
                    (design_history,),
                    sampling_interval,
                    fine_step_count,
                    start_index,
                    sample_count,
                    monitor_times,
                )
            )
            state["callback_count"] += 1

        simulation.run(native_segment, until=_RUN_UNTIL)
        result = {
            "monitor_history": monitor_history.copy(),
            "design_history": design_history.copy(),
            "monitor_times": monitor_times.copy(),
            "fine_step_count": fine_step_count,
            "actual_time": simulation.round_time(),
            "design_signature": design_field.signature.copy(),
            "python_callback_count": state["callback_count"],
            "extra_seconds": state["extra_seconds"],
        }
    finally:
        if design_field is not None:
            design_field.release()
        if monitor is not None:
            monitor.plan = None
        simulation.reset_meep()
    return result


@pytest.mark.parametrize(
    ("force_complex_fields", "sampling_interval"),
    ((False, 1), (True, 2)),
    ids=("real-k1", "complex-k2"),
)
def test_native_forward_matches_python_callback_and_preserves_rounded_endpoint(
    force_complex_fields,
    sampling_interval,
):
    if mp.count_processors() != 1:
        pytest.skip("serial native-forward reference test")
    mp.verbosity(0)

    reference = _python_forward_reference(
        force_complex_fields,
        sampling_interval,
    )
    native = _native_forward(force_complex_fields, sampling_interval)

    # Meep rounds this nominal seven-step limit to eight completed steps.
    assert reference["fine_step_count"] == native["fine_step_count"] == 8
    assert reference["python_callback_count"] == 9
    assert native["python_callback_count"] == 2
    assert reference["actual_time"] == pytest.approx(0.4, abs=1.0e-7)
    assert native["actual_time"] == reference["actual_time"]
    assert np.allclose(
        native["monitor_times"],
        _DT * np.arange(9),
        rtol=0.0,
        atol=1.0e-15,
    )
    assert np.array_equal(native["monitor_times"], reference["monitor_times"])
    assert np.array_equal(native["monitor_history"], reference["monitor_history"])
    assert np.array_equal(native["design_history"], reference["design_history"])
    assert np.array_equal(native["design_signature"], reference["design_signature"])
    assert native["design_history"].shape[0] == 8 // sampling_interval + 1
    assert np.any(native["monitor_history"] != 0.0)
    assert np.any(native["design_history"] != 0.0)
    if force_complex_fields:
        assert np.any(native["monitor_history"].imag != 0.0)
        assert np.any(native["design_history"].imag != 0.0)
    assert np.isfinite(native["extra_seconds"])
    assert native["extra_seconds"] >= 0.0


def test_native_forward_preserves_python_custom_source_callbacks():
    if mp.count_processors() != 1:
        pytest.skip("serial custom-source native-forward test")
    calls = []
    simulation, design = _make_custom_source_simulation(
        lambda time: calls.append(float(time)) or np.cos(float(time))
    )
    monitor = None
    design_field = None
    simulation.init_sim()
    try:
        monitor = FastPointMonitor(simulation, mp.Ez, _MONITOR_POSITIONS)
        design_field = NativeDesignField(simulation, mp.Ez, design)
        fine_step_count = int(
            native_sampler.native_forward_step_count(
                int(simulation.fields.this),
                _RUN_UNTIL,
            )
        )
        monitor_history = np.empty(
            (fine_step_count + 1, len(_MONITOR_POSITIONS)),
            dtype=np.float64,
        )
        design_history = np.empty(
            (fine_step_count + 1, design_field.local_size),
            dtype=np.float64,
        )
        monitor_times = np.empty(fine_step_count + 1, dtype=np.float64)
        state = {"samples": 0}

        def native_segment(_simulation):
            start = state["samples"]
            count = fine_step_count if start == 0 else 1
            native_sampler.run_native_forward_segment(
                (monitor.ensure_plan(),),
                (monitor_history,),
                (design_field.ensure_plan(),),
                (design_history,),
                1,
                fine_step_count,
                start,
                count,
                monitor_times,
            )
            state["samples"] += count

        simulation.run(native_segment, until=_RUN_UNTIL)
        assert state["samples"] == fine_step_count + 1
        assert calls
        assert np.all(np.isfinite(monitor_history))
        assert np.any(monitor_history != 0.0)
    finally:
        if design_field is not None:
            design_field.release()
        if monitor is not None:
            monitor.plan = None
        simulation.reset_meep()
