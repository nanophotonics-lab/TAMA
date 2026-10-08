"""Discrete-frequency far fields and their TAMA time-domain adjoint."""

import autograd.numpy as npa
from autograd import grad
import meep as mp
import numpy as np
import pytest

import tama as tm


def make_problem(
    *,
    component=mp.Ez,
    dimensions=2,
    averaging=False,
    courant=0.25,
    reuse=False,
    complex_fields=False,
    fom_fn=None,
    anisotropic=False,
    sampling_interval=1,
):
    resolution = 12 if dimensions == 2 else 8
    size = mp.Vector3(0.4, 0.4, 0.4 if dimensions == 3 else 0)
    shape = (3, 3, 3) if dimensions == 3 else (3, 3)
    air = mp.Medium(epsilon=1)
    dielectric = (
        mp.Medium(
            epsilon_diag=mp.Vector3(2.6, 3, 3.4),
            epsilon_offdiag=mp.Vector3(0.15, 0.2, 0.1),
        )
        if anisotropic
        else mp.Medium(epsilon=3)
    )
    grid = mp.MaterialGrid(
        mp.Vector3(*shape),
        air,
        dielectric,
        beta=4 if averaging else 0,
        do_averaging=averaging,
    )
    design = tm.DesignGrid(
        material_grid=grid,
        center=mp.Vector3(),
        size=size,
        shape=shape,
        background=air,
        design_material=dielectric,
    )
    regions = []
    for axis in range(dimensions):
        for sign in (-1, 1):
            center = [0.0, 0.0, 0.0]
            center[axis] = sign * 0.8
            extent = [1.6, 1.6, 1.6 if dimensions == 3 else 0.0]
            extent[axis] = 0.0
            regions.append(
                mp.Near2FarRegion(
                    center=mp.Vector3(*center), size=mp.Vector3(*extent), weight=sign
                )
            )
    target = tm.Near2FarTarget(
        near_regions=regions,
        frequencies=(0.65, 0.75),
        far_points=(
            mp.Vector3(3, 0.4, 0.2 if dimensions == 3 else 0),
            mp.Vector3(2, -2, 0.3 if dimensions == 3 else 0),
        ),
    )
    spec = tm.SimulationSpec(
        cell_size=mp.Vector3(3, 3, 3 if dimensions == 3 else 0),
        dimensions=dimensions,
        resolution=resolution,
        courant=courant,
        boundary_layers=[mp.PML(0.4)],
        geometry=[mp.Block(size=size, material=grid)],
        sources=[
            mp.Source(
                mp.GaussianSource(0.7, fwidth=0.4),
                component=component,
                center=mp.Vector3(-0.55, 0.09, 0.07 if dimensions == 3 else 0),
            )
        ],
        eps_averaging=averaging,
    )
    if complex_fields:
        original_make = spec.make

        def make(sources=None, **kwargs):
            sim = original_make(sources, **kwargs)
            sim.force_complex_fields = True
            return sim

        spec.make = make
    obj = tm.TDAObjective(
        simulation=spec,
        design=design,
        target=target,
        t_final=24,
        chunk_balancer=None,
        reuse_simulation=reuse,
        fom_fn=fom_fn,
        sampling_interval=sampling_interval,
    )
    x = np.random.default_rng(715).uniform(0.25, 0.75, np.prod(shape))
    return obj, x, target


@pytest.mark.parametrize("component", [mp.Ez, mp.Hz])
@pytest.mark.parametrize("courant", [0.25, 0.5])
@pytest.mark.parametrize("t_final", [12, 24])
@pytest.mark.mpi2
def test_farfield_gradient_matches_two_step_directional_fd(component, courant, t_final):
    mp.verbosity(0)
    obj, x, _ = make_problem(component=component, courant=courant)
    # The shorter run retains a large final field and exposes endpoint errors.
    obj.t_final = t_final
    direction = np.random.default_rng(716).uniform(0.2, 1.0, x.size)
    direction /= np.linalg.norm(direction)
    value, gradient = obj.fom_and_grad(x)
    fd = [
        (obj.fom(x + h * direction) - obj.fom(x - h * direction)) / (2 * h)
        for h in (2e-4, 1e-4)
    ]
    adjoint = float(gradient @ direction)
    assert value > 0
    assert obj.last_far_fields.shape == (2, 2, 6)
    assert abs(fd[0]) > 1e-8
    assert abs(fd[0] - fd[1]) / abs(fd[1]) < 1e-3
    assert abs(adjoint - fd[1]) / abs(fd[1]) < 5e-3, (adjoint, fd)
    print("Near2far FD:", component, courant, adjoint, fd)


@pytest.mark.parametrize("dimensions,component", [(2, mp.Hz), (3, mp.Ex)])
@pytest.mark.parametrize("sampling_interval", [1, 3])
@pytest.mark.mpi2
def test_farfield_averaged_gradient(dimensions, component, sampling_interval):
    mp.verbosity(0)
    obj, x, _ = make_problem(
        dimensions=dimensions,
        component=component,
        averaging=True,
        anisotropic=True,
        sampling_interval=sampling_interval,
    )
    direction = np.random.default_rng(718).uniform(0.2, 1.0, x.size)
    direction /= np.linalg.norm(direction)
    _, gradient = obj.fom_and_grad(x)
    differences = [
        (obj.fom(x + h * direction) - obj.fom(x - h * direction)) / (2 * h)
        for h in (2e-4, 1e-4)
    ]
    np.testing.assert_allclose(differences[0], differences[1], rtol=1e-3)
    np.testing.assert_allclose(gradient @ direction, differences, rtol=5e-3)
    print(
        "Near2far tensor averaged FD:",
        dimensions,
        sampling_interval,
        gradient @ direction,
        differences,
    )


@pytest.mark.parametrize("complex_fields", [False, True])
def test_farfield_custom_complex_objective_and_reuse(complex_fields):
    mp.verbosity(0)
    obj, x, _ = make_problem(
        complex_fields=complex_fields,
        reuse=True,
        fom_fn=lambda fields, dt: npa.real(
            fields[0, 0, 2] * (0.7 + 0.3j) + fields[1, 1, 3] * (0.2 - 0.6j)
        ),
    )
    direction = np.random.default_rng(717).uniform(0.2, 1.0, x.size)
    direction /= np.linalg.norm(direction)
    value, gradient = obj.fom_and_grad(x)
    h = 1e-4
    fd = (obj.fom(x + h * direction) - obj.fom(x - h * direction)) / (2 * h)
    assert np.isfinite(value)
    assert abs(gradient @ direction - fd) / abs(fd) < 5e-3


def test_farfield_manual_covector_keeps_dt_contract():
    mp.verbosity(0)
    obj, x, _ = make_problem(
        complex_fields=True,
        fom_fn=lambda fields, dt: npa.real(fields[0, 0, 2] * (0.7 + 0.3j)),
    )
    automatic = obj.fom_and_grad(x)
    obj.adjoint_signal_fn = lambda fields, dt: grad(obj.fom_fn, 0)(fields, dt) / dt
    manual = obj.fom_and_grad(x)
    np.testing.assert_allclose(manual[0], automatic[0], rtol=1e-13)
    np.testing.assert_allclose(manual[1], automatic[1], rtol=1e-12, atol=1e-14)


def test_sparse_farfield_value_and_gradient_use_same_end_time():
    mp.verbosity(0)
    obj, x, _ = make_problem(component=mp.Hz, sampling_interval=3)
    obj.t_final = 23.98
    value, gradient = obj.fom_and_grad(x)
    actual_time = obj.last_actual_time
    np.testing.assert_allclose(obj.fom(x), value, rtol=1e-13)
    assert obj.last_actual_time == actual_time
    assert obj.t_final <= actual_time < obj.t_final + 3 * 0.25 / 12
    direction = np.ones_like(x) / np.sqrt(x.size)
    h = 1e-4
    fd = (obj.fom(x + h * direction) - obj.fom(x - h * direction)) / (2 * h)
    np.testing.assert_allclose(gradient @ direction, fd, rtol=5e-3)


@pytest.mark.parametrize(
    "invalid", ["pml", "volume", "far_point", "far_z", "frequency", "no_source"]
)
def test_farfield_rejects_invalid_surface_configuration(invalid):
    obj, x, target = make_problem()
    if invalid == "pml":
        target.near_regions[0].center = mp.Vector3(-1.2, 0)
    elif invalid == "volume":
        target.near_regions[0].size = mp.Vector3(0.1, 1.6)
    elif invalid == "far_point":
        target.far_points[0].x, target.far_points[0].y = -0.8, 0
    elif invalid == "far_z":
        target.far_points[0].z = 1
    elif invalid == "no_source":
        obj._simulation_spec.sources = []
    else:
        obj.max_frequency = 0.7
    with pytest.raises(ValueError):
        obj.fom(x)


def test_farfield_rejects_nonfinite_objective_or_covector():
    obj, x, _ = make_problem(fom_fn=lambda fields, dt: np.nan)
    with pytest.raises(ValueError, match="objective values must be finite"):
        obj.fom(x)
    obj.fom_fn = lambda fields, dt: npa.sum(npa.abs(fields) ** 2)
    obj.adjoint_signal_fn = lambda fields, dt: np.full_like(fields, np.nan)
    with pytest.raises(ValueError, match="covectors must be finite"):
        obj.fom_and_grad(x)


@pytest.mark.parametrize("need_gradient", [False, True])
def test_farfield_rejects_explicit_mirror_chunk_layout(need_gradient):
    obj, x, _ = make_problem()
    x[:] = 0.4
    obj._simulation_spec.symmetries = [mp.Mirror(mp.Y)]
    obj._simulation_spec.chunk_layout = mp.BinaryPartition(data=[(mp.X, 0.0), 0, 0])
    with pytest.raises(
        ValueError, match="Mirrors does not support explicit chunk_layout"
    ):
        (obj.fom_and_grad if need_gradient else obj.fom)(x)


@pytest.mark.parametrize(
    "component,mirrors",
    [
        (mp.Hz, (mp.Mirror(mp.Y, phase=-1),)),
        (mp.Ez, (mp.Mirror(mp.X), mp.Mirror(mp.Y))),
    ],
)
@pytest.mark.mpi2
def test_farfield_mirror_parity_and_multiple_planes(component, mirrors):
    mp.verbosity(0)
    full, x, _ = make_problem(component=component)
    reduced, _, _ = make_problem(component=component)
    for obj in (full, reduced):
        obj._simulation_spec.sources[0].center = mp.Vector3()
    reduced._simulation_spec.symmetries = mirrors
    weights = x.reshape(3, 3)
    direction = np.random.default_rng(720).uniform(0.2, 1.0, (3, 3))
    for mirror in mirrors:
        weights = 0.5 * (weights + np.flip(weights, int(mirror.direction)))
        direction = 0.5 * (direction + np.flip(direction, int(mirror.direction)))
    x, direction = weights.ravel(), direction.ravel()
    direction /= np.linalg.norm(direction)
    baseline = full.fom_and_grad(x)
    value, gradient = reduced.fom_and_grad(x)
    np.testing.assert_allclose(value, baseline[0], rtol=1e-11)
    np.testing.assert_allclose(
        gradient @ direction, baseline[1] @ direction, rtol=1e-10
    )
    h = 1e-4
    fd = (reduced.fom(x + h * direction) - reduced.fom(x - h * direction)) / (2 * h)
    np.testing.assert_allclose(gradient @ direction, fd, rtol=5e-3)


@pytest.mark.mpi4
def test_farfield_mirror_sources_stay_within_active_process_groups():
    if mp.comm.Get_size() != 4:
        pytest.skip("requires exactly four MPI ranks")
    mp.verbosity(0)
    group = mp.divide_parallel_processes(2)
    try:
        full, x, target = make_problem(component=mp.Hz)
        reduced, _, _ = make_problem(component=mp.Hz)
        for obj in (full, reduced):
            obj._simulation_spec.sources[0].center = mp.Vector3()
            obj.objective = tm.Near2FarTarget(
                target.near_regions,
                target.frequencies,
                [mp.Vector3(3 + group, 0.4 + 0.2 * group), mp.Vector3(2, -2 - group)],
            )
        reduced._simulation_spec.symmetries = [mp.Mirror(mp.Y, phase=-1)]
        weights = x.reshape(3, 3) + 0.08 * group
        x = (0.5 * (weights + np.flip(weights, 1))).ravel()
        direction = np.random.default_rng(721 + group).uniform(0.2, 1.0, (3, 3))
        direction = (0.5 * (direction + np.flip(direction, 1))).ravel()
        direction /= np.linalg.norm(direction)
        baseline = full.fom_and_grad(x)
        value, gradient = reduced.fom_and_grad(x)
        np.testing.assert_allclose(value, baseline[0], rtol=1e-11)
        adjoint = float(gradient @ direction)
        np.testing.assert_allclose(adjoint, baseline[1] @ direction, rtol=1e-10)
        h = 1e-4
        fd = (reduced.fom(x + h * direction) - reduced.fom(x - h * direction)) / (2 * h)
        np.testing.assert_allclose(adjoint, fd, rtol=5e-3)
        result = (group, value, adjoint, fd)
    finally:
        mp.end_divide_parallel()
    results = mp.comm.allgather(result)
    if mp.am_master():
        for result in sorted(set(results)):
            print("Near2far active-group FD:", *result)


@pytest.mark.parametrize("dimensions", [2, 3])
def test_farfield_matches_plain_meep(dimensions):
    mp.verbosity(0)
    obj, x, target = make_problem(dimensions=dimensions)
    value = obj.fom(x)
    sim = obj._simulation_spec.make()
    monitor = sim.add_near2far(
        list(target.frequencies), *target.near_regions, decimation_factor=1
    )
    sim.run(until=obj.t_final)
    expected = np.asarray(
        [sim.get_farfield(monitor, p) for p in target.far_points]
    ).reshape(2, 2, 6)
    sim.reset_meep()
    np.testing.assert_allclose(obj.last_far_fields, expected, rtol=2e-12, atol=2e-13)
    assert value == pytest.approx(0.5 * np.sum(abs(expected[..., :3]) ** 2), rel=2e-12)
