"""A native FDTDX scene shared by the examples and numerical checks."""

import jax
import jax.numpy as jnp
import numpy as np

import tama

tm = tama.get_backend("fdtdx")


def make_scene(
    dtype=jnp.float64,
    steps=256,
    periodic_z=False,
    backend="gpu",
    source_kind="uniform",
    boundary_kind="pml",
    bloch_vector=(0.0, 0.0, 0.0),
):
    """Return a native 3-D or thin periodic scene, design and initial density.

    The six-component native detector covers a downstream observation plane and
    is used as an independent reference against the extension's observations.
    """
    import fdtdx

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if type(periodic_z) is not bool:
        raise TypeError("periodic_z must be a bool")
    if boundary_kind not in ("pml", "pec", "pmc"):
        raise ValueError("boundary_kind must be pml, pec or pmc")
    bloch_vector = tuple(bloch_vector)
    if len(bloch_vector) != 3 or not np.isfinite(bloch_vector).all():
        raise ValueError("bloch_vector must contain three finite values in rad/m")
    dtype = jnp.dtype(dtype)
    if dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise ValueError("dtype must be float32 or float64")
    if dtype == jnp.dtype(jnp.float64) and not jax.config.x64_enabled:
        raise ValueError("enable jax_enable_x64 before making a float64 scene")
    dx, c0 = 100e-9, 299792458.0
    dt = 0.5 * dx / c0
    # Native 0.6.2 plane-source interpolation squeezes a one-cell transverse
    # dimension to rank one. Two periodic cells preserve its required rank.
    shape = (40, 32, 2 if periodic_z else 12)
    design_shape = (16, 16, 2 if periodic_z else 8)
    design_start = (12, 8, 0 if periodic_z else 2)
    parameter_shape = (4, 4, 1 if periodic_z else 2)
    source_start = (9, 6, 0 if periodic_z else 2)
    source_shape = (1, 20, 2 if periodic_z else 8)
    detector_start = (31, 14, 0 if periodic_z else 4)
    detector_shape = (1, 4, 2 if periodic_z else 4)
    target = (31, 16, 1 if periodic_z else 6)
    config = fdtdx.SimulationConfig(
        time=steps * dt,
        resolution=dx,
        backend=backend,
        dtype=dtype,
        use_complex_fields=True if any(bloch_vector) else None,
        courant_factor=float(np.sqrt(3) * 0.5),
        gradient_config=fdtdx.GradientConfig(method="checkpointed", num_checkpoints=8),
    )
    air = fdtdx.Material(permittivity=1.0)
    volume = fdtdx.SimulationVolume(
        name="volume",
        partial_grid_shape=shape,
        partial_real_position=tuple(n * dx / 2 for n in shape),
        material=air,
    )
    device = fdtdx.Device(
        name="design",
        partial_grid_shape=design_shape,
        partial_voxel_grid_shape=tuple(
            n // p for n, p in zip(design_shape, parameter_shape)
        ),
        materials={"air": air, "dielectric": fdtdx.Material(permittivity=4.0)},
        param_transforms=[],
    )
    n, f0 = np.arange(121), c0 / 1e-6
    waveform = np.exp(-(((n - 50) / 16) ** 2)) * np.cos(2 * np.pi * f0 * (n - 50) * dt)
    waveform[[0, -1]] = 0
    source_kwargs = dict(
        name="source",
        partial_grid_shape=source_shape,
        wave_character=fdtdx.WaveCharacter(frequency=f0),
        temporal_profile=fdtdx.CustomTimeSignalProfile(
            signal=jnp.asarray(waveform, dtype=dtype), time_step_duration=dt
        ),
    )
    if source_kind == "dipole":
        source_kwargs["partial_grid_shape"] = (1, 1, 1)
        source_start = (9, 16, 0 if periodic_z else 6)
        source = fdtdx.PointDipoleSource(**source_kwargs, polarization=2)
    elif source_kind in ("uniform", "gaussian"):
        source_kwargs.update(
            direction="+",
            fixed_E_polarization_vector=(0.0, 0.0, 1.0),
            normalize_by_energy=False,
        )
        source = (
            fdtdx.UniformPlaneSource(**source_kwargs)
            if source_kind == "uniform"
            else fdtdx.GaussianPlaneSource(**source_kwargs, radius=10 * dx)
        )
    else:
        raise ValueError("source_kind must be uniform, gaussian or dipole")
    field_dtype = (
        (jnp.complex128 if dtype == jnp.dtype(jnp.float64) else jnp.complex64)
        if any(bloch_vector)
        else dtype
    )
    detector = fdtdx.FieldDetector(
        name="reference",
        partial_grid_shape=detector_shape,
        components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        dtype=field_dtype,
        exact_interpolation=True,
    )
    boundary_kwargs = {"bloch_vector": bloch_vector}
    for i, (axis, thickness) in enumerate(zip("xyz", (5, 5, 1))):
        for side in ("min", "max"):
            boundary_kwargs[f"boundary_type_{side}{axis}"] = (
                "bloch"
                if bloch_vector[i]
                else "periodic" if periodic_z and axis == "z" else boundary_kind
            )
            boundary_kwargs[f"thickness_grid_{side}{axis}"] = thickness
    boundaries, constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig(**boundary_kwargs), volume
    )
    constraints = list(constraints)
    for obj, start in (
        (device, design_start),
        (source, source_start),
        (detector, detector_start),
    ):
        constraints.append(
            obj.set_grid_coordinates(
                axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=start
            )
        )
    key = jax.random.PRNGKey(17)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=[volume, *boundaries.values(), device, source, detector],
        config=config,
        constraints=constraints,
        key=key,
    )
    arrays, objects, _ = fdtdx.apply_params(
        arrays=arrays, objects=objects, params=params, key=key
    )
    spec = tm.SimulationSpec(arrays, objects, config, key)
    design = tm.DesignGrid.from_device(objects["design"]).validate(spec)
    rho = jnp.asarray(
        np.random.default_rng(41).uniform(0.25, 0.75, parameter_shape), dtype=dtype
    )
    return dict(
        spec=spec,
        design=design,
        rho=rho,
        target=target,
        detector_name="reference",
        detector_start=detector_start,
        detector_shape=detector_shape,
        frequency=f0,
    )
