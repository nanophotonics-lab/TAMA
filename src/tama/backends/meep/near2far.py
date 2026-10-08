"""Meep near-to-far pullback with TAMA's finite-time source convention.

Mirror runs require Meep's default chunk layout: explicit partitions can
duplicate near-surface quadrature in Meep 1.34.
"""

import gc

from autograd import grad
import autograd.numpy as npa
import meep as mp
from meep.simulation import py_v3_to_vec
import numpy as np

from .native_design import (
    _NativeDesignAccumulator,
    _NativeDesignHistorySet,
    _native_forward_step_count,
    _run_native_forward_loop,
    _run_native_adjoint_loop,
)
from .sampling_grid import (
    _require_no_meep_symmetries,
    history_storage_dtype,
)
from .design_collection import (
    _collection_components as _native_design_components,
    _collection_gradient_scale as _native_design_gradient_scale,
)
from . import native_sampler
from ...nyquist import _aligned_run_until


def farfield_intensity(fields, dt):
    """Sum electric-field intensity over the requested points and frequencies."""
    return 0.5 * npa.sum(npa.abs(fields[..., :3]) ** 2)


def _validate_simulation(sim, chunk_layout):
    if sim._infer_dimensions(sim.k_point) not in (2, 3, mp.CYLINDRICAL):
        raise ValueError(
            "Near2FarTarget requires Cartesian 2D/3D or cylindrical coordinates"
        )
    _require_no_meep_symmetries(sim)
    if sim.symmetries and chunk_layout is not None:
        raise ValueError(
            "Near2FarTarget with Mirrors does not support explicit chunk_layout; "
            "Meep can duplicate near-surface quadrature for this combination"
        )
    if sim.k_point and any(complex(x) != 0 for x in sim.k_point):
        raise ValueError("Near2FarTarget does not support nonzero Bloch wavevectors")
    if not sim.sources:
        raise ValueError("Near2FarTarget requires at least one forward source")


def _validate_surfaces(sim, targets):
    dimensions = sim._infer_dimensions(sim.k_point)
    cylindrical = dimensions == mp.CYLINDRICAL or sim.is_cylindrical
    axes = [0, 2] if cylindrical else list(range(dimensions))
    directions = [mp.R, mp.Z] if cylindrical else axes
    center = np.asarray(tuple(sim.geometry_center or mp.Vector3()))
    half_size = np.asarray(tuple(sim.cell_size)) / 2
    low, high = center - half_size, center + half_size
    if cylindrical:
        low[0], high[0] = center[0], center[0] + sim.cell_size.x
    includes_axis = cylindrical and low[0] == 0
    for layer in sim.boundary_layers:
        for axis, direction in zip(axes, directions):
            if layer.direction not in (mp.ALL, direction):
                continue
            if layer.side in (mp.ALL, mp.Low) and not (includes_axis and axis == 0):
                low[axis] += layer.thickness
            if layer.side in (mp.ALL, mp.High):
                high[axis] -= layer.thickness
    clearance = 0.5 / sim.resolution
    low += clearance
    high -= clearance
    if includes_axis:
        # A cap may reach the axis, where there is neither a boundary nor PML.
        low[0] = 0
    for target in targets:
        if cylindrical and any(
            point.x < 0 or point.y != 0 for point in target.far_points
        ):
            raise ValueError("cylindrical far points must use (r, 0, z) with r >= 0")
        if (
            not cylindrical
            and dimensions == 2
            and any(point.z != 0 for point in target.far_points)
        ):
            raise ValueError("2D far points must lie in the x-y plane")
        for region in target.near_regions:
            c, s = np.asarray(tuple(region.center)), np.asarray(tuple(region.size))
            if (
                not np.all(np.isfinite(c))
                or not np.all(np.isfinite(s))
                or np.any(s < 0)
                or np.count_nonzero(s[axes] == 0) != 1
                or (cylindrical and (c[1] != 0 or s[1] != 0))
                or (not cylindrical and dimensions == 2 and (c[2] != 0 or s[2] != 0))
                or not np.isfinite(region.weight)
            ):
                raise ValueError(
                    "near regions must be finite axis-aligned surfaces with one zero extent"
                )
            normal = directions[int(np.flatnonzero(s[axes] == 0)[0])]
            if region.direction not in (mp.AUTOMATIC, normal):
                raise ValueError(
                    "near-region direction must match its zero-extent axis"
                )
            if np.any(c[axes] - s[axes] / 2 < low[axes]) or np.any(
                c[axes] + s[axes] / 2 > high[axes]
            ):
                raise ValueError(
                    "near regions require half a grid cell of clearance from PML and cell boundaries"
                )
            for point in target.far_points:
                if np.all(np.abs(np.asarray(tuple(point)) - c) <= s / 2 + 1e-12):
                    raise ValueError("far points must not lie on a near surface")


def farfield_sources(sim, monitor, target, covector, actual_time):
    """Transpose the spatial Green transform and finite-time Fourier sum.

    Meep's near_sourcedata includes surface quadrature and flips electric
    components. TAMA's time reversal flips magnetic components instead, hence
    the common minus sign. Indexed amplitudes are current densities, so divide
    by the Yee-cell volume. The native cylindrical transpose also corrects
    Meep's radial normalization. No frequency-domain i*omega scale is used.
    """
    frequencies = np.asarray(target.frequencies)
    points = np.asarray([tuple(p) for p in target.far_points]).reshape(-1)
    point0 = py_v3_to_vec(sim.dimensions, target.far_points[0], sim.is_cylindrical)
    raw_data = monitor.swigobj.near_sourcedata(
        point0,
        points,
        len(target.far_points),
        np.ascontiguousarray(covector, dtype=np.complex128),
        1e-3,
    )
    # SWIG vector iteration returns temporary owning wrappers; retain them while
    # the native pullback reads their addresses.
    data = tuple(raw_data)
    sources = []
    scale = -(1 if sim.is_cylindrical else sim.resolution**sim.dimensions) / np.sqrt(
        2 * np.pi
    )
    source_end_time = actual_time + 0.5 * sim.fields.dt
    if sim.symmetries or sim.is_cylindrical:
        groups = native_sampler.fold_near2far_sources(
            int(sim.fields.this),
            int(monitor.swigobj.this),
            tuple(int(datum.this) for datum in data),
        )
        indexed_data = []
        for component, chunk, indices, amplitudes in groups:
            singletons = []
            for index in indices:
                datum = mp.sourcedata()
                native_sampler.populate_sourcedata(
                    int(datum.this), component, chunk, int(index)
                )
                singletons.append(datum)
            datum = mp.sourcedata()
            native_sampler.merge_sourcedata(
                int(datum.this), tuple(int(d.this) for d in singletons)
            )
            indexed_data.append((datum, np.asarray(amplitudes)))
    else:
        indexed_data = [
            (datum, np.asarray(datum.amp_arr).reshape(-1, frequencies.size))
            for datum in data
        ]
    for datum, amplitudes in indexed_data:
        if not np.all(np.isfinite(amplitudes)):
            raise ValueError("near-to-far source amplitudes must be finite")
        for index, frequency in enumerate(frequencies):
            if not np.any(amplitudes[:, index]):
                continue

            def waveform(t, frequency=frequency):
                # DFT recording has no sample beyond its final forward step.
                # In particular, do not inject an extra magnetic current at 0.
                if t <= 0 or t > actual_time:
                    return 0j
                # DFT fields include H's half-step phase. Both current types
                # therefore share the discrete adjoint's half-step delay.
                return np.exp(2j * np.pi * frequency * (source_end_time - t))

            sources.append(
                mp.IndexedSource(
                    mp.CustomSource(
                        src_func=waveform, start_time=0, end_time=actual_time
                    ),
                    datum,
                    np.ascontiguousarray(scale * amplitudes[:, index]),
                )
            )
    return sources


def add_near2far_monitors(sim, targets, max_frequency, *, chunk_layout):
    # Tensor material setup may already have populated Meep's automatic layout.
    _validate_simulation(sim, chunk_layout)
    if max_frequency is not None and max_frequency < max(
        f for target in targets for f in target.frequencies
    ):
        raise ValueError("max_frequency must cover every near-to-far target frequency")
    _validate_surfaces(sim, targets)
    return [
        sim.add_near2far(
            np.asarray(target.frequencies),
            *target.near_regions,
            decimation_factor=1,
            nperiods=1,
        )
        for target in targets
    ]


def _synchronize_callback_error(error, operation="near-to-far callback"):
    failed = (
        mp.sum_to_all(int(error is not None))
        if mp.count_processors() > 1
        else error is not None
    )
    if failed:
        if error is not None:
            raise error
        raise RuntimeError(f"{operation} failed on another MPI rank")


def farfield_values(sim, monitors, targets, fom_fns, dt):
    fields = [
        np.asarray(
            [sim.get_farfield(monitor, point) for point in target.far_points]
        ).reshape(len(target.far_points), len(target.frequencies), 6)
        for target, monitor in zip(targets, monitors)
    ]
    error = None
    try:
        if any(not np.all(np.isfinite(f)) for f in fields):
            raise ValueError(
                "near-to-far fields must be finite; check far points and near surfaces"
            )
        values = np.asarray(
            [float((fn or farfield_intensity)(f, dt)) for fn, f in zip(fom_fns, fields)]
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("near-to-far objective values must be finite")
    except Exception as exc:
        error = exc
    _synchronize_callback_error(error)
    return fields, values


def farfield_adjoint_sources(
    sim,
    monitors,
    targets,
    fields,
    fom_fns,
    adjoint_signal_fns,
    coefficients,
    actual_time,
    dt,
):
    covectors = []
    error = None
    try:
        for f, fn, adjoint_fn in zip(fields, fom_fns, adjoint_signal_fns):
            covector = (
                grad(fn or farfield_intensity, 0)(f, dt)
                if adjoint_fn is None
                else np.asarray(adjoint_fn(f, dt)) * dt
            )
            if covector.shape != f.shape:
                raise ValueError(
                    "near-to-far adjoint_signal_fn must match far_fields shape"
                )
            if not np.all(np.isfinite(covector)):
                raise ValueError("near-to-far adjoint covectors must be finite")
            covectors.append(covector)
    except Exception as exc:
        error = exc
    # All local callbacks finish before any rank enters near_sourcedata collectives.
    _synchronize_callback_error(error)
    sources = []
    for target, monitor, covector, coefficient in zip(
        targets, monitors, covectors, coefficients
    ):
        sources.extend(
            farfield_sources(sim, monitor, target, coefficient * covector, actual_time)
        )
    return sources


def evaluate_near2far(
    objective,
    x,
    need_gradient,
    forward_sources=(),
    *,
    targets=None,
    fom_fns=None,
    adjoint_signal_fns=None,
    scalarize=None,
):
    """Run the shared native design-history loop with a spectral target."""
    objective.update_design(x)
    sim = None
    history = accumulator = None
    try:
        sim = objective._make_forward_simulation(forward_sources)
        if (
            need_gradient
            and (sim.is_cylindrical or sim.dimensions == mp.CYLINDRICAL)
            and sim.m not in (-1, 0, 1)
        ):
            raise ValueError(
                "cylindrical gradient evaluation currently supports m=-1, 0, or +1"
            )
        multiple = targets is not None
        targets = tuple(targets) if multiple else (objective.objective,)
        fom_fns = tuple(fom_fns) if multiple else (objective.fom_fn,)
        adjoint_signal_fns = (
            tuple(adjoint_signal_fns) if multiple else (objective.adjoint_signal_fn,)
        )
        monitors = add_near2far_monitors(
            sim,
            targets,
            objective.max_frequency,
            chunk_layout=objective._simulation_spec.chunk_layout,
        )
        sim.init_sim()
        dt = objective.time_step(sim)
        interval = objective._resolve_sampling_interval(dt)
        objective.last_sampling_interval = interval
        run_until = _aligned_run_until(
            objective.t_final, dt=dt, sampling_interval=interval
        )
        if need_gradient:
            components = _native_design_components(
                sim, objective.design, objective.gradient_components
            )
            scale = _native_design_gradient_scale(sim, objective.design)
            dtype = history_storage_dtype(sim, objective.history_dtype)
            history = _NativeDesignHistorySet(
                objective.design,
                components,
                dtype,
                lambda shape: objective._make_history_memmap(shape, dtype=dtype),
            )
            fine_steps = _native_forward_step_count(sim, run_until, interval)
            _, count, _, _ = _run_native_forward_loop(
                sim,
                (),
                history,
                run_until=run_until,
                sampling_interval=interval,
                fine_step_count=fine_steps,
            )
            fields = history.finish_forward(count)
            history.release_forward()
        else:
            sim.run(until=run_until)
        actual_time = sim.round_time()
        objective.last_actual_time = actual_time
        far_fields, values = farfield_values(sim, monitors, targets, fom_fns, dt)
        objective.last_far_fields = (
            tuple(f.copy() for f in far_fields) if multiple else far_fields[0].copy()
        )
        error = None
        try:
            if multiple:
                value, coefficients, info = scalarize(
                    values, need_gradient=need_gradient
                )
            else:
                value, coefficients = float(values[0]), (1.0,)
            if not np.isfinite(value) or (
                need_gradient and not np.all(np.isfinite(coefficients))
            ):
                raise ValueError(
                    "near-to-far scalarization value and coefficients must be finite"
                )
        except Exception as exc:
            error = exc
        _synchronize_callback_error(error)
        if multiple:
            objective.last_band_objectives = values
            objective.last_band_losses = -values
            objective.last_band_coeffs = coefficients
            objective.last_scalarization_info = info
            objective.last_total_fom = value
            objective.last_smooth_min = (
                info.get("smooth_min") if isinstance(info, dict) else None
            )
        if not need_gradient:
            return value, None
        sources = farfield_adjoint_sources(
            sim,
            monitors,
            targets,
            far_fields,
            fom_fns,
            adjoint_signal_fns,
            coefficients,
            actual_time,
            dt,
        )
        # Keep the exact forward chunk/index ownership used by near_sourcedata.
        if sim.is_cylindrical and sim.m != 0:
            sim.change_m(-sim.m)
        sim.restart_fields()
        sim.clear_dft_monitors()
        sim.change_sources(sources)
        accumulator = _NativeDesignAccumulator(
            objective.design, components, history.signatures, False
        )
        _run_native_adjoint_loop(
            sim,
            accumulator,
            fields,
            fine_step_count=fine_steps,
            actual_time=actual_time,
            dt=dt,
            sampling_interval=interval,
            reconstruction_half_width=objective.reconstruction_half_width,
            reconstruction_window=objective.reconstruction_window,
            reconstruction_window_params=objective.reconstruction_window_params,
        )
        gradient = accumulator.reduce().real.ravel() * dt * scale
        return value, gradient
    finally:
        if accumulator is not None:
            accumulator.release()
        if history is not None:
            history.release_forward()
        gc.collect()
        if sim is not None:
            sim.reset_meep()
        if history is not None:
            history.cleanup_memmaps()
