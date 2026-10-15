"""Cylindrical tensor and averaged material derivatives at finite time."""

from dataclasses import replace

import meep as mp
import numpy as np
import pytest

import tama as tm
from test_adaptive_chunking_mpi import _make_cylindrical_objective
from test_cylindrical_near2far import _make_problem as _make_farfield_problem
from test_multiple_designs import _check_region_derivatives
from test_native_material_tensor import _design_and_direction, _rotated_medium


def _tensor_media():
    return (
        _rotated_medium((1.3, 1.7, 2.1), (0.2, -0.3, 0.4)),
        _rotated_medium((3.0, 4.2, 5.1), (-0.4, 0.5, -0.2)),
    )


def _make_material_problem(
    material_kind,
    target_kind,
    m,
    *,
    sampling_interval=1,
    reuse_simulation=False,
    axis_design=False,
):
    base, weights, far_target = _make_farfield_problem(m, sampling_interval)
    medium1 = mp.Medium(epsilon=1.0)
    medium2 = mp.Medium(epsilon=2.25)
    if material_kind == "diagonal":
        # A tensor on the axis must be invariant under rotations about z.
        medium1 = mp.Medium(epsilon_diag=mp.Vector3(1.2, 1.2, 1.6))
        medium2 = mp.Medium(epsilon_diag=mp.Vector3(2.8, 2.8, 3.8))
    elif material_kind in ("offdiagonal", "tensor_smoothing"):
        # In cylindrical coordinates these are r-phi, r-z, and phi-z entries.
        medium1, medium2 = _tensor_media()
    do_averaging = material_kind in ("smoothing", "tensor_smoothing")
    grid = mp.MaterialGrid(
        mp.Vector3(3, 1, 3),
        medium1,
        medium2,
        do_averaging=do_averaging,
        beta=8 if do_averaging else 0,
        eta=0.42,
    )
    center = mp.Vector3(0.2, 0, 0) if axis_design else base.design.center
    design = replace(
        base.design,
        material_grid=grid,
        center=center,
        background=medium1,
        design_material=medium2,
        material_factor=None,
    )
    simulation = replace(
        base._simulation_spec,
        geometry=[mp.Block(center=center, size=design.size, material=grid)],
        eps_averaging=material_kind
        in ("eps_averaging", "smoothing", "tensor_smoothing"),
    )
    temporal_target = tm.PointTarget(
        mp.Vector3(0.95, 0, 0.35), mp.Ez if axis_design else mp.Er
    )
    common = dict(
        design=design,
        simulation=simulation,
        t_final=base.t_final,
        sampling_interval=sampling_interval,
        chunk_balancer=None,
        reuse_simulation=reuse_simulation,
    )
    if target_kind == "near2far":
        objective = tm.TDAObjective(target=far_target, **common)
    elif target_kind == "temporal":
        objective = tm.TDAObjective(target=temporal_target, **common)
    else:
        objective = tm.MultiTDAObjective(
            targets=[temporal_target, far_target],
            wavelength_bands=[(1.0, 1.6), None],
            weights=[1.0, 1.0],
            kernel_length=31,
            scalarization_fn=_scalarize,
            **common,
        )
    return objective, weights, common, temporal_target, far_target


def _scalarize(values):
    return 0.4 * values[0] + 0.6 * values[1] + 0.2 * values[0] * values[1]


@pytest.mark.parametrize(
    (
        "material_kind",
        "target_kind",
        "m",
        "sampling_interval",
        "reuse_simulation",
        "axis_design",
    ),
    [
        pytest.param(
            "diagonal", "temporal", 0, 1, False, True, id="diagonal-axis-m0-ez"
        ),
        pytest.param(
            "offdiagonal",
            "temporal",
            1,
            1,
            True,
            False,
            marks=pytest.mark.mpi2,
            id="offdiagonal-temporal-m1-reuse",
        ),
        pytest.param(
            "offdiagonal",
            "near2far",
            -1,
            1,
            False,
            False,
            marks=[pytest.mark.mpi2, pytest.mark.mpi4],
            id="offdiagonal-near2far-mminus1",
        ),
        pytest.param(
            "eps_averaging", "temporal", 0, 1, False, False, id="eps-averaging-m0"
        ),
        pytest.param(
            "smoothing", "temporal", -1, 1, True, False, id="smoothing-temporal-reuse"
        ),
        pytest.param(
            "smoothing",
            "near2far",
            1,
            2,
            False,
            False,
            marks=pytest.mark.mpi2,
            id="smoothing-near2far-m1-sparse",
        ),
        pytest.param(
            "smoothing", "mixed", 0, 1, False, False, id="smoothing-mixed-m0"
        ),
        pytest.param(
            "tensor_smoothing",
            "mixed",
            -1,
            2,
            True,
            False,
            marks=[pytest.mark.mpi2, pytest.mark.mpi4],
            id="tensor-smoothing-mixed-mminus1-sparse",
        ),
    ],
)
def test_cylindrical_material_gradient_matches_two_step_fd(
    material_kind,
    target_kind,
    m,
    sampling_interval,
    reuse_simulation,
    axis_design,
):
    mp.verbosity(0)
    objective, weights, common, temporal_target, far_target = _make_material_problem(
        material_kind,
        target_kind,
        m,
        sampling_interval=sampling_interval,
        reuse_simulation=reuse_simulation,
        axis_design=axis_design,
    )
    direction = np.random.default_rng(822).uniform(0.2, 1.0, weights.size)
    if axis_design:
        # Isolate the innermost material row, including the m=0 Ez axis volume.
        direction[3:] = 0
    direction /= np.linalg.norm(direction)
    value, gradient = objective.fom_and_grad(weights)
    directional_adjoint = float(np.asarray(gradient).ravel() @ direction)
    directional_fds = [
        (
            objective.fom(weights + h * direction)
            - objective.fom(weights - h * direction)
        )
        / (2 * h)
        for h in (2e-4, 1e-4)
    ]
    print(
        "Cylindrical material FD:",
        material_kind,
        target_kind,
        m,
        sampling_interval,
        value,
        directional_adjoint,
        directional_fds,
    )
    assert np.isfinite(value) and np.all(np.isfinite(gradient))
    assert abs(directional_fds[-1]) > 1e-9
    np.testing.assert_allclose(
        directional_fds[0], directional_fds[1], rtol=1e-3, atol=0
    )
    np.testing.assert_allclose(
        directional_adjoint,
        directional_fds[-1],
        rtol=5e-5 if sampling_interval == 1 else 5e-3,
        atol=0,
    )
    if target_kind == "mixed":
        temporal = tm.MultiTDAObjective(
            targets=[temporal_target],
            wavelength_bands=[(1.0, 1.6)],
            weights=[1.0],
            kernel_length=31,
            **common,
        )
        far = tm.TDAObjective(target=far_target, **common)
        temporal_value, temporal_gradient = temporal.fom_and_grad(weights)
        far_value, far_gradient = far.fom_and_grad(weights)
        expected_gradient = (0.4 + 0.2 * far_value) * temporal_gradient + (
            0.6 + 0.2 * temporal_value
        ) * far_gradient
        np.testing.assert_allclose(
            value, _scalarize([temporal_value, far_value]), rtol=1e-11, atol=1e-13
        )
        np.testing.assert_allclose(gradient, expected_gradient, rtol=5e-3, atol=1e-8)
    if mp.count_processors() > 1:
        for rank_value, rank_gradient in mp.comm.allgather((value, gradient)):
            np.testing.assert_allclose(rank_value, value, rtol=1e-12, atol=0)
            np.testing.assert_allclose(rank_gradient, gradient, rtol=1e-12, atol=1e-14)


def _inverse_cylindrical_tensor(
    medium1,
    medium2,
    *,
    native=True,
    do_averaging=True,
    eps_averaging=True,
    weights=None,
):
    if weights is None:
        weights = np.linspace(0.1, 0.9, 9)
    grid = mp.MaterialGrid(
        mp.Vector3(3, 1, 3),
        medium1,
        medium2,
        weights=np.asarray(weights).reshape(3, 1, 3),
        do_averaging=do_averaging,
        beta=16,
        eta=0.5,
    )
    design = tm.DesignGrid(
        material_grid=grid,
        center=mp.Vector3(0.8, 0, 0.027),
        size=mp.Vector3(1.2, 0, 1.2),
        shape=(3, 3),
        background=medium1,
        design_material=medium2,
        coordinate_system="cylindrical",
    )
    simulation = mp.Simulation(
        cell_size=mp.Vector3(1.8, 0, 2),
        dimensions=mp.CYLINDRICAL,
        resolution=20,
        force_all_components=True,
        geometry=[mp.Block(center=design.center, size=design.size, material=grid)],
        eps_averaging=eps_averaging,
        subpixel_tol=1e-10,
    )
    try:
        if native:
            from tama.backends.meep.sampling_grid import _prepare_native_material

            _prepare_native_material(simulation, design)
        simulation.init_sim()
        return np.asarray(
            [
                [
                    [
                        simulation.fields.get_chi1inv(
                            component, direction, mp.veccyl(r, z)
                        )
                        for direction in (mp.R, mp.P, mp.Z)
                    ]
                    for component in (mp.Er, mp.Ep, mp.Ez)
                ]
                for r in np.linspace(0.6, 1.0, 5)
                for z in np.linspace(-0.2, 0.2, 5)
            ]
        )
    finally:
        simulation.reset_meep()


def test_cylindrical_averaging_changes_forward_tensor():
    mp.verbosity(0)
    media = _tensor_media()
    averaged = _inverse_cylindrical_tensor(*media)
    unaveraged = _inverse_cylindrical_tensor(*media, do_averaging=False)
    difference = np.max(np.abs(averaged - unaveraged))
    print("Cylindrical averaging inverse-tensor difference:", difference)
    assert difference > 1e-5


@pytest.mark.parametrize("constant", ["endpoints", "weights"])
def test_cylindrical_averaging_preserves_homogeneous_tensor(constant):
    mp.verbosity(0)
    first, second = _tensor_media()
    if constant == "endpoints":
        second = first
    weights = np.full(9, 0.37) if constant == "weights" else None
    averaged = _inverse_cylindrical_tensor(first, second, weights=weights)
    unaveraged = _inverse_cylindrical_tensor(
        first, second, weights=weights, do_averaging=False
    )
    np.testing.assert_allclose(averaged, unaveraged, rtol=1e-12, atol=1e-12)


def test_cylindrical_global_eps_averaging_disables_grid_averaging():
    mp.verbosity(0)
    media = _tensor_media()
    globally_disabled = _inverse_cylindrical_tensor(*media, eps_averaging=False)
    both_disabled = _inverse_cylindrical_tensor(
        *media, do_averaging=False, eps_averaging=False
    )
    np.testing.assert_allclose(globally_disabled, both_disabled, rtol=1e-12, atol=1e-12)


def test_cylindrical_tensor_averaging_has_meep_isotropic_limit():
    mp.verbosity(0)
    stock = _inverse_cylindrical_tensor(
        mp.Medium(epsilon=1), mp.Medium(epsilon=4), native=False
    )
    near_isotropic = _inverse_cylindrical_tensor(
        mp.Medium(epsilon=1),
        mp.Medium(epsilon_diag=mp.Vector3(4, 4, 4 + 1e-8)),
    )
    difference = np.max(np.abs(near_isotropic - stock))
    print("Cylindrical near-isotropic inverse-tensor difference:", difference)
    np.testing.assert_allclose(near_isotropic, stock, rtol=1e-7, atol=1e-8)


@pytest.mark.mpi2
@pytest.mark.mpi4
def test_cylindrical_tensor_independent_regions_match_each_region_fd():
    mp.verbosity(0)
    base, _, _, target, _ = _make_material_problem("offdiagonal", "temporal", -1)
    designs, geometry = [], []
    for index, (radius, shape) in enumerate(((0.4, (2, 3)), (1.1, (3, 3)))):
        low, high = (
            _tensor_media()
            if index
            else (mp.Medium(epsilon=1), mp.Medium(epsilon=2.25))
        )
        grid = mp.MaterialGrid(
            mp.Vector3(shape[0], 1, shape[1]),
            low,
            high,
            do_averaging=bool(index),
            beta=8 if index else 0,
        )
        design = replace(
            base.design,
            material_grid=grid,
            center=mp.Vector3(radius, 0, 0),
            shape=shape,
            background=low,
            design_material=high,
            material_factor=None,
        )
        designs.append(design)
        geometry.append(mp.Block(center=design.center, size=design.size, material=grid))
    objective = tm.MultiTDAObjective(
        designs=designs,
        simulation=replace(
            base._simulation_spec, geometry=geometry, eps_averaging=True
        ),
        targets=[target],
        wavelength_bands=[(1.0, 1.6)],
        weights=[1.0],
        kernel_length=31,
        t_final=base.t_final,
        sampling_interval=1,
        chunk_balancer=None,
    )
    _check_region_derivatives(objective, designs)


@pytest.mark.mpi2
@pytest.mark.mpi4
def test_cylindrical_tensor_adaptive_chunks_match_fixed_and_fd():
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")
    mp.verbosity(0)
    base, shape = _make_cylindrical_objective("tda")
    first, second = _tensor_media()
    grid = mp.MaterialGrid(mp.Vector3(shape[0], 1, shape[1]), first, second)
    design = replace(
        base.design,
        material_grid=grid,
        background=first,
        design_material=second,
        material_factor=None,
    )
    common = dict(
        design=design,
        simulation=replace(
            base._simulation_spec,
            geometry=[mp.Block(center=design.center, size=design.size, material=grid)],
        ),
        target=tm.PointTarget(mp.Vector3(0.8, 0, 0.65), mp.Er),
        t_final=base.t_final,
        dt=base.dt,
        sampling_interval=1,
    )
    adaptive = tm.TDAObjective(**common)
    fixed = tm.TDAObjective(
        chunk_balancer=None,
        **dict(common, simulation=replace(common["simulation"], chunk_layout=None)),
    )
    weights = np.linspace(0.3, 0.7, int(np.prod(shape)))
    value, gradient = adaptive.fom_and_grad(weights)
    assert adaptive.last_source_boundary_decision.method == "layout"
    assert adaptive.last_chunk_balance is not None
    assert set(adaptive.last_chunk_balance.topology) & {"R", "Z"}
    fixed_value, fixed_gradient = fixed.fom_and_grad(weights)
    np.testing.assert_allclose(value, fixed_value, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(gradient, fixed_gradient, rtol=1e-9, atol=1e-12)
    direction = np.array([0.2, 0.5, 0.7, 0.3])
    direction /= np.linalg.norm(direction)
    directional_adjoint = float(gradient @ direction)
    directional_fds = [
        (adaptive.fom(weights + h * direction) - adaptive.fom(weights - h * direction))
        / (2 * h)
        for h in (2e-4, 1e-4)
    ]
    print("Cylindrical tensor adaptive FD:", directional_adjoint, directional_fds)
    assert abs(directional_fds[-1]) > 1e-9
    np.testing.assert_allclose(
        directional_fds[0], directional_fds[1], rtol=1e-3, atol=0
    )
    np.testing.assert_allclose(
        directional_adjoint, directional_fds[-1], rtol=5e-5, atol=0
    )


def _fixed_tensor_background(kind, *, overlap_pml=False, dispersive=False):
    base, _, common, target, _ = _make_material_problem("scalar", "temporal", 1)
    first, second = _tensor_media()
    if dispersive:
        second.E_susceptibilities = [
            mp.LorentzianSusceptibility(frequency=0.9, gamma=0.05, sigma=0.1)
        ]
    material = (
        mp.MaterialGrid(
            mp.Vector3(2, 1, 2), first, second, weights=np.array([0.3, 0.7, 0.6, 0.4])
        )
        if kind == "material_grid"
        else second
    )
    fixed_block = mp.Block(
        center=mp.Vector3(1.45 if overlap_pml else 1.0, 0, -0.1),
        size=mp.Vector3(0.4 if overlap_pml else 0.18, 0, 0.5),
        material=material,
    )
    common["simulation"] = replace(
        base._simulation_spec,
        geometry=[*base._simulation_spec.geometry, fixed_block],
    )
    return tm.TDAObjective(target=target, **common)


@pytest.mark.parametrize("kind", ["medium", "material_grid"])
def test_cylindrical_fixed_tensor_background_matches_two_step_fd(kind):
    from tama.backends.meep.sampling_grid import _uses_material_jacobian

    mp.verbosity(0)
    objective = _fixed_tensor_background(kind)
    assert not objective._simulation_spec.eps_averaging
    assert _uses_material_jacobian(objective._simulation_spec, objective.design)
    weights, direction = _design_and_direction(objective)
    value, gradient = objective.fom_and_grad(weights)
    directional_adjoint = float(np.asarray(gradient).ravel() @ direction)
    directional_fds = [
        (
            objective.fom(weights + h * direction)
            - objective.fom(weights - h * direction)
        )
        / (2 * h)
        for h in (2e-4, 1e-4)
    ]
    print("Cylindrical fixed tensor FD:", kind, directional_adjoint, directional_fds)
    assert np.isfinite(value) and np.all(np.isfinite(gradient))
    assert abs(directional_fds[-1]) > 1e-9
    np.testing.assert_allclose(
        directional_fds[0], directional_fds[1], rtol=1e-3, atol=0
    )
    np.testing.assert_allclose(
        directional_adjoint, directional_fds[-1], rtol=5e-5, atol=0
    )


@pytest.mark.parametrize(
    "invalid, exception, message",
    [
        (
            "design_dispersion",
            ValueError,
            "native material gradients require nondispersive endpoints",
        ),
        (
            "fixed_tensor_pml",
            RuntimeError,
            "cylindrical off-diagonal adjoint correction does not support "
            "material overlapping PML",
        ),
        (
            "fixed_tensor_dispersion",
            RuntimeError,
            "cylindrical off-diagonal adjoint correction requires displacement "
            "fields without polarization subtraction",
        ),
    ],
)
def test_cylindrical_material_gradient_rejects_unsupported_media(
    invalid, exception, message
):
    if mp.count_processors() > 1 and exception is RuntimeError:
        pytest.skip("native failures abort MPI instead of raising to Python")
    mp.verbosity(0)
    if invalid == "design_dispersion":
        objective, _, _, _, _ = _make_material_problem("offdiagonal", "temporal", 1)
        objective.design.material_grid.medium2.E_susceptibilities = [
            mp.LorentzianSusceptibility(frequency=0.9, gamma=0.05, sigma=0.1)
        ]
    else:
        objective = _fixed_tensor_background(
            "medium",
            overlap_pml=invalid == "fixed_tensor_pml",
            dispersive=invalid == "fixed_tensor_dispersion",
        )
    with pytest.raises(exception, match=message):
        objective.fom_and_grad(np.linspace(0.25, 0.75, 9))
