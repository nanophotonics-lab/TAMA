"""Independent design regions share one forward/adjoint solve and flat gradients."""

import meep as mp
import numpy as np
import pytest

import tama as tm


def _problem(
    kind="single", *, averaging=False, sampling_interval=1, tensor=False, split=False
):
    designs = []
    geometry = []
    for center, shape, epsilon in [
        (mp.Vector3(-0.36, -0.07), (3, 4), 3.0),
        (mp.Vector3(0.34, 0.05), (5, 3), 4.2),
    ]:
        low, high = mp.Medium(epsilon=1), mp.Medium(epsilon=epsilon)
        if tensor and designs:
            high = mp.Medium(
                epsilon_diag=mp.Vector3(3.0, 4.2, 5.0),
                epsilon_offdiag=mp.Vector3(0.2, -0.3, 0.4),
            )
        grid = mp.MaterialGrid(
            mp.Vector3(*shape),
            low,
            high,
            do_averaging=averaging,
            beta=8 if averaging else 0,
        )
        design = tm.DesignGrid(
            grid,
            center,
            mp.Vector3(0.38, 0.64),
            shape,
            background=low,
            design_material=high,
        )
        designs.append(design)
        geometry.append(mp.Block(center=center, size=design.size, material=grid))
    spec = tm.SimulationSpec(
        cell_size=mp.Vector3(2.6, 2.4),
        resolution=14,
        courant=0.25,
        boundary_layers=[mp.PML(0.35)],
        geometry=geometry,
        dimensions=2,
        eps_averaging=averaging,
        chunk_layout=mp.BinaryPartition(data=[(mp.X, 0.0), 0, 1]) if split else None,
        sources=[
            mp.Source(
                mp.GaussianSource(0.7, fwidth=0.5),
                component=mp.Ez,
                center=mp.Vector3(-0.84, -0.17),
            )
        ],
    )
    target = tm.PointTarget(mp.Vector3(0.82, 0.23), component=mp.Ez)
    common = dict(
        designs=designs,
        simulation=spec,
        t_final=24,
        dt=0.25 / 14,
        sampling_interval=sampling_interval,
        chunk_balancer=None,
    )
    if kind == "multi":
        objective = tm.MultiTDAObjective(
            targets=[target],
            wavelength_bands=[(1.1, 2.0)],
            weights=[1.0],
            kernel_length=31,
            **common,
        )
    else:
        objective = tm.TDAObjective(target=target, **common)
    return objective, designs


@pytest.mark.parametrize("kind", ["single", "multi"])
@pytest.mark.parametrize("averaging", [False, True])
def test_separated_designs_gradient_matches_each_region_finite_difference(
    kind, averaging
):
    mp.verbosity(0)
    objective, designs = _problem(kind, averaging=averaging)
    _check_region_derivatives(objective, designs)


def _check_region_derivatives(objective, designs):
    rng = np.random.default_rng(7107)
    x = rng.uniform(0.25, 0.75, sum(int(np.prod(region.shape)) for region in designs))
    value, gradient = objective.fom_and_grad(x)
    assert np.isfinite(value)
    assert gradient.shape == x.shape
    assert np.all(np.isfinite(gradient))
    for index, design in enumerate(designs):
        direction = np.zeros_like(x)
        offset = sum(int(np.prod(region.shape)) for region in designs[:index])
        size = int(np.prod(design.shape))
        direction[offset : offset + size] = rng.uniform(-1, 1, size)
        direction /= np.linalg.norm(direction)
        adjoint = float(gradient @ direction)
        finite_differences = [
            (objective.fom(x + h * direction) - objective.fom(x - h * direction))
            / (2 * h)
            for h in (2e-4, 1e-4)
        ]
        print(type(objective).__name__, index, adjoint, finite_differences)
        assert abs(finite_differences[-1]) > 1e-9
        np.testing.assert_allclose(
            finite_differences[0], finite_differences[1], rtol=0.001, atol=1e-10
        )
        np.testing.assert_allclose(
            adjoint, finite_differences[-1], rtol=0.005, atol=1e-9
        )


@pytest.mark.parametrize("kind", ["single", "multi"])
def test_multiple_designs_reject_wrong_flat_vector_length(kind):
    objective, _ = _problem(kind)
    with pytest.raises(ValueError, match="27"):
        objective.fom(np.full(26, 0.5))


@pytest.mark.parametrize("kind", ["single", "multi"])
@pytest.mark.parametrize(
    "invalid", ["both", "empty", "duplicate", "overlap", "dimensions"]
)
def test_invalid_independent_designs_are_rejected(kind, invalid):
    _, designs = _problem(kind)
    constructor = tm.TDAObjective if kind == "single" else tm.MultiTDAObjective
    arguments = dict(designs=designs)
    if invalid == "both":
        arguments["design"] = designs[0]
        message = "either design or designs"
    elif invalid == "empty":
        arguments["designs"] = []
        message = "at least one"
    elif invalid == "duplicate":
        arguments["designs"] = [designs[0], designs[0]]
        message = "distinct MaterialGrid"
    elif invalid == "overlap":
        designs[1].center = designs[0].center
        message = "must not overlap"
    else:
        low, high = mp.Medium(epsilon=1), mp.Medium(epsilon=4.2)
        designs[1] = tm.DesignGrid(
            mp.MaterialGrid(mp.Vector3(5, 3, 1), low, high),
            designs[1].center,
            mp.Vector3(0.38, 0.64, 0.2),
            (5, 3, 1),
            background=low,
            design_material=high,
        )
        message = "same dimensions"
    with pytest.raises(ValueError, match=message):
        constructor(**arguments)


@pytest.mark.parametrize("kind", ["single", "multi"])
def test_two_regions_share_exactly_one_forward_and_adjoint_run(kind, monkeypatch):
    mp.verbosity(0)
    objective, _ = _problem(kind)
    runs = []
    original_run = mp.Simulation.run

    def record_run(sim, *args, **kwargs):
        runs.append(sim)
        return original_run(sim, *args, **kwargs)

    monkeypatch.setattr(mp.Simulation, "run", record_run)
    value, gradient = objective.fom_and_grad(np.linspace(0.25, 0.75, 27))
    assert np.isfinite(value) and np.all(np.isfinite(gradient))
    assert len(runs) == 2


@pytest.mark.parametrize("kind", ["single", "multi"])
def test_multiple_designs_sparse_history_matches_dense(kind):
    mp.verbosity(0)
    dense, _ = _problem(kind)
    sparse, _ = _problem(kind, sampling_interval=2)
    x = np.random.default_rng(7107).uniform(0.25, 0.75, 27)
    dense_value, dense_gradient = dense.fom_and_grad(x)
    sparse_value, sparse_gradient = sparse.fom_and_grad(x)
    np.testing.assert_allclose(sparse_value, dense_value, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(sparse_gradient, dense_gradient, rtol=0.005, atol=1e-8)


@pytest.mark.parametrize("kind", ["single", "multi"])
def test_independent_scalar_and_tensor_regions_match_finite_difference(kind):
    mp.verbosity(0)
    objective, designs = _problem(kind, averaging=True, tensor=True)
    _check_region_derivatives(objective, designs)


@pytest.mark.mpi2
@pytest.mark.parametrize("kind", ["single", "multi"])
def test_two_rank_independent_regions_match_finite_difference(kind):
    if mp.count_processors() != 2:
        pytest.skip("requires exactly two MPI ranks")
    mp.verbosity(0)
    objective, designs = _problem(kind, averaging=True, tensor=True, split=True)
    _check_region_derivatives(objective, designs)


def _multi_farfield_problem(**kwargs):
    point_objective, designs = _problem()
    region = mp.Near2FarRegion(
        center=mp.Vector3(0.72, 0), size=mp.Vector3(0, 1.4), direction=mp.X
    )
    targets = [
        tm.Near2FarTarget((region,), (0.65, 0.75), (mp.Vector3(3, 0.4),)),
        tm.Near2FarTarget((region,), (0.6, 0.7, 0.8), (mp.Vector3(2, -2),)),
    ]
    weights = np.array([0.3, 0.7])
    options = dict(
        designs=designs,
        simulation=point_objective._simulation_spec,
        targets=targets,
        t_final=24,
        chunk_balancer=None,
        scalarization_fn=lambda values: (float(weights @ values), weights),
    )
    options.update(kwargs)
    return tm.MultiTDAObjective(**options), weights


def test_multi_farfield_targets_and_designs_share_one_adjoint(monkeypatch):
    mp.verbosity(0)
    objective, weights = _multi_farfield_problem()
    runs = []
    original_run = mp.Simulation.run

    def record_run(sim, *args, **kwargs):
        runs.append(sim)
        return original_run(sim, *args, **kwargs)

    monkeypatch.setattr(mp.Simulation, "run", record_run)
    x = np.random.default_rng(7107).uniform(0.25, 0.75, 27)
    value, gradient = objective.fom_and_grad(x)
    assert len(runs) == 2
    assert tuple(fields.shape for fields in objective.last_far_fields) == (
        (1, 2, 6),
        (1, 3, 6),
    )
    assert value == pytest.approx(weights @ objective.last_band_objectives, rel=1e-13)
    direction = np.random.default_rng(7108).normal(size=x.size)
    direction /= np.linalg.norm(direction)
    differences = [
        (objective.fom(x + h * direction) - objective.fom(x - h * direction)) / (2 * h)
        for h in (2e-4, 1e-4)
    ]
    np.testing.assert_allclose(differences[0], differences[1], rtol=0.001, atol=1e-10)
    np.testing.assert_allclose(
        gradient @ direction, differences[-1], rtol=0.005, atol=1e-9
    )


@pytest.mark.parametrize("invalid", ["bands", "kernel", "window", "history"])
def test_multi_farfield_rejects_fir_settings(invalid):
    target = tm.Near2FarTarget(
        (mp.Near2FarRegion(center=mp.Vector3(0.72, 0), size=mp.Vector3(0, 1.4)),),
        (0.7,),
        (mp.Vector3(3, 0.4),),
    )
    options = dict(targets=[target])
    if invalid == "bands":
        options["wavelength_bands"] = [(1.1, 2)]
        message = "PC-FIR"
    else:
        if invalid == "kernel":
            options["kernel_length"] = 31
        elif invalid == "window":
            options["kernel_window"] = "hann"
        else:
            options["target_history_block_size"] = 16
        message = "PC-FIR"
    with pytest.raises(ValueError, match=message):
        tm.MultiTDAObjective(**options)


def test_multi_farfield_autograd_scalarization_and_distinct_callbacks():
    import autograd.numpy as npa

    mp.verbosity(0)
    objective, _ = _multi_farfield_problem(
        fom_fn=[
            lambda f, dt: npa.real(f[0, 0, 2] * (0.7 + 0.3j)),
            lambda f, dt: npa.sum(npa.abs(f[..., :3]) ** 2),
        ],
        scalarization_fn=lambda values: values[0] ** 2 + 0.7 * values[1],
    )
    x = np.random.default_rng(7107).uniform(0.25, 0.75, 27)
    _, gradient = objective.fom_and_grad(x)
    direction = np.random.default_rng(7108).normal(size=x.size)
    direction /= np.linalg.norm(direction)
    h = 1e-4
    difference = (
        objective.fom(x + h * direction) - objective.fom(x - h * direction)
    ) / (2 * h)
    np.testing.assert_allclose(gradient @ direction, difference, rtol=0.005, atol=1e-9)


@pytest.mark.parametrize("kind", ["single", "multi"])
def test_independent_designs_reject_rotated_block(kind):
    mp.verbosity(0)
    objective, _ = _problem(kind)
    block = objective._simulation_spec.geometry[1]
    block.e1, block.e2 = mp.Vector3(0.8, 0.6), mp.Vector3(-0.6, 0.8)
    with pytest.raises(ValueError, match="axis-aligned"):
        objective.fom_and_grad(np.full(27, 0.5))


def test_multi_farfield_manual_callbacks_match_automatic_covectors():
    from autograd import grad
    import autograd.numpy as npa

    mp.verbosity(0)
    callbacks = [
        lambda f, dt: npa.real(f[0, 0, 2] * (0.7 + 0.3j)),
        lambda f, dt: npa.sum(npa.abs(f[..., :3]) ** 2),
    ]
    objective, _ = _multi_farfield_problem(fom_fn=callbacks)
    x = np.random.default_rng(7107).uniform(0.25, 0.75, 27)
    automatic = objective.fom_and_grad(x)
    objective.adjoint_signal_fn = [
        lambda f, dt, callback=callback: grad(callback, 0)(f, dt) / dt
        for callback in callbacks
    ]
    manual = objective.fom_and_grad(x)
    np.testing.assert_allclose(manual[0], automatic[0], rtol=1e-13)
    np.testing.assert_allclose(manual[1], automatic[1], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("kind", ["single", "multi"])
@pytest.mark.parametrize("coordinate_system", ["cartesian", "cylindrical"])
def test_independent_regions_in_3d_and_cylindrical_match_fd(kind, coordinate_system):
    mp.verbosity(0)
    cylindrical = coordinate_system == "cylindrical"
    designs, geometry = [], []
    centers = (
        (mp.Vector3(0.4, 0, -0.05), mp.Vector3(1.13, 0, 0.07))
        if cylindrical
        else (mp.Vector3(-0.35, -0.04, 0.03), mp.Vector3(0.35, 0.03, -0.04))
    )
    shape = (3, 4) if cylindrical else (3, 3, 2)
    size = mp.Vector3(0.5, 0, 0.6) if cylindrical else mp.Vector3(0.38, 0.58, 0.54)
    for center, epsilon in zip(centers, (2.25, 3.0)):
        low, high = mp.Medium(epsilon=1), mp.Medium(epsilon=epsilon)
        grid_shape = (
            mp.Vector3(shape[0], 1, shape[1]) if cylindrical else mp.Vector3(*shape)
        )
        grid = mp.MaterialGrid(grid_shape, low, high)
        design = tm.DesignGrid(
            grid,
            center,
            size,
            shape,
            background=low,
            design_material=high,
            coordinate_system=coordinate_system,
        )
        designs.append(design)
        geometry.append(mp.Block(center=center, size=size, material=grid))
    resolution, courant = (10, 0.2) if cylindrical else (8, 0.15)
    component = mp.Er if cylindrical else mp.Ez
    source_position = (
        mp.Vector3(0.65, 0, -0.8) if cylindrical else mp.Vector3(-0.84, -0.17, 0.09)
    )
    target_position = (
        mp.Vector3(1.1, 0, 0.8) if cylindrical else mp.Vector3(0.82, 0.23, -0.11)
    )
    spec = tm.SimulationSpec(
        cell_size=mp.Vector3(2, 0, 3) if cylindrical else mp.Vector3(2.6, 2.4, 2),
        resolution=resolution,
        courant=courant,
        boundary_layers=(
            [mp.PML(0.4, direction=mp.R), mp.PML(0.4, direction=mp.Z)]
            if cylindrical
            else [mp.PML(0.35)]
        ),
        geometry=geometry,
        dimensions=mp.CYLINDRICAL if cylindrical else 3,
        m=1 if cylindrical else 0,
        eps_averaging=False,
        sources=[
            mp.Source(
                mp.GaussianSource(0.8, fwidth=0.6, cutoff=3),
                component=component,
                center=source_position,
            )
        ],
    )
    target = tm.PointTarget(target_position, component=component)
    common = dict(
        designs=designs,
        simulation=spec,
        t_final=14,
        dt=courant / resolution,
        sampling_interval=1,
        chunk_balancer=None,
    )
    objective = (
        tm.MultiTDAObjective(
            targets=[target],
            wavelength_bands=[(0.9, 2)],
            weights=[1],
            kernel_length=31,
            **common,
        )
        if kind == "multi"
        else tm.TDAObjective(target=target, **common)
    )
    _check_region_derivatives(objective, designs)


def test_multi_farfield_checks_max_frequency_of_every_target():
    objective, _ = _multi_farfield_problem(max_frequency=0.77)
    with pytest.raises(ValueError, match="max_frequency"):
        objective.fom(np.linspace(0.25, 0.75, 27))


def test_multi_farfield_rejects_nonfinite_scalarization_coefficients():
    objective, _ = _multi_farfield_problem(
        scalarization_fn=lambda values: (
            float(np.sum(values)),
            np.array([1.0, np.nan]),
        ),
    )
    with pytest.raises(ValueError, match="finite"):
        objective.fom_and_grad(np.linspace(0.25, 0.75, 27))


def test_mirror_regions_keep_replicated_target_history_for_source_routing():
    point_objective, designs = _problem()
    spec = point_objective._simulation_spec
    objective = tm.MultiTDAObjective(
        designs=designs,
        simulation=spec,
        targets=[tm.FieldRegionTarget((mp.Vector3(0.82, 0.23),), mp.Ez)],
        wavelength_bands=[(1.1, 2)],
        weights=[1],
        kernel_length=31,
        t_final=24,
        target_history_block_size=1,
        chunk_balancer=None,
    )
    assert objective._uses_distributed_builtin_target_history()
    spec.symmetries = (mp.Mirror(mp.Y),)
    assert not objective._uses_distributed_builtin_target_history()


def test_multi_farfield_public_time_step_and_sampling_metadata():
    from types import SimpleNamespace

    objective, _ = _multi_farfield_problem(
        dt=0.01,
        resolution=19,
        max_frequency=1.0,
        reconstruction_window="hann",
        reconstruction_half_width=24,
    )
    sim = SimpleNamespace(resolution=17, Courant=0.4)
    assert objective.time_step(sim) == pytest.approx(0.01)
    assert objective._explicit_dt == 0.01
    assert objective.resolution == 19
    assert objective.max_frequency == 1.0
    assert objective.reconstruction_window == "hann"
    assert objective.reconstruction_half_width == 24
    assert objective.chunk_balancer is None
    assert objective.reuse_simulation is False
    objective.dt = None
    objective.resolution = 20
    assert objective.time_step(sim) == pytest.approx(0.02)
    assert not hasattr(objective, "kernels")


def test_multi_farfield_runtime_sampling_changes_reach_core():
    objective, _ = _multi_farfield_problem()
    objective.max_frequency = 0.77
    objective.resolution = 19
    objective.reconstruction_window = "hann"
    objective.reconstruction_window_params = {}
    objective.reconstruction_half_width = 24
    with pytest.raises(ValueError, match="max_frequency"):
        objective.fom(np.linspace(0.25, 0.75, 27))
    core = objective._near2far_objective
    for name in (
        "max_frequency",
        "resolution",
        "reconstruction_window",
        "reconstruction_window_params",
        "reconstruction_half_width",
    ):
        assert getattr(core, name) == getattr(objective, name)
