"""Directional checks for the actual averaged/tensor material operator."""

import meep as mp
import numpy as np
import pytest

import tama as tm


def _rotated_medium(eigenvalues, angles):
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    rx = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
    ry = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
    rz = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))
    rotation = rz @ ry @ rx
    epsilon = rotation @ np.diag(eigenvalues) @ rotation.T
    return mp.Medium(
        epsilon_diag=mp.Vector3(*np.diag(epsilon)),
        epsilon_offdiag=mp.Vector3(epsilon[0, 1], epsilon[0, 2], epsilon[1, 2]),
    )


def _make_problem(case, *, sampling_interval=1, split=False, reuse_simulation=False):
    three_d = case in (
        "tensor_3d",
        "tensor_3d_singleton",
        "smooth_3d",
        "smooth_3d_singleton",
    )
    resolution = 10 if three_d else 14
    # Near-cancelling singleton direction: 1.04% error at 0.25, 0.224% at 0.125.
    courant = 0.125 if case == "smooth_3d_singleton" else 0.25
    shape = (
        (3, 3, 1) if case.endswith("3d_singleton") else (3, 3, 3) if three_d else (4, 5)
    )
    center = mp.Vector3(0.027, -0.031, 0.019 if three_d else 0)
    size = mp.Vector3(0.64, 0.58, 0.48 if three_d else 0)
    medium1, medium2 = mp.Medium(epsilon=1.0), mp.Medium(epsilon=4.0)
    component = mp.Hz if case in ("tensor_hz", "smooth_hz") else mp.Ez
    do_averaging = case in (
        "smooth_ez",
        "smooth_hz",
        "tensor_3d",
        "tensor_3d_singleton",
        "tensor_xz_2d",
        "smooth_3d",
        "smooth_3d_singleton",
    )
    beta = 32 if case == "smooth_ez" else 8 if do_averaging else 0
    if case == "tensor_hz":
        medium1 = _rotated_medium((1.4, 2.0, 1.6), (0, 0, 0.4))
        medium2 = _rotated_medium((3.0, 5.0, 3.5), (0, 0, -0.6))
    elif case in ("tensor_3d", "tensor_3d_singleton", "tensor_xz_2d"):
        medium1 = _rotated_medium((1.3, 1.7, 2.1), (0.2, -0.3, 0.4))
        medium2 = _rotated_medium((3.0, 4.2, 5.1), (-0.4, 0.5, -0.2))
    material_grid = mp.MaterialGrid(
        mp.Vector3(*shape),
        medium1,
        medium2,
        do_averaging=do_averaging,
        beta=beta,
        eta=0.42 if case in ("smooth_ez", "smooth_3d", "smooth_3d_singleton") else 0.5,
    )
    design = tm.DesignGrid(
        material_grid=material_grid,
        center=center,
        size=size,
        shape=shape,
        background=medium1,
        design_material=medium2,
    )
    geometry = [mp.Block(center=center, size=size, material=material_grid)]
    if case in ("fixed_geometry", "fixed_tensor"):
        geometry.append(
            mp.Cylinder(
                radius=0.19,
                center=mp.Vector3(0.58, -0.36),
                material=(
                    _rotated_medium((2.0, 3.0, 4.0), (0.3, -0.4, 0.2))
                    if case == "fixed_tensor"
                    else mp.Medium(epsilon=3.0)
                ),
            )
        )
    simulation = tm.SimulationSpec(
        cell_size=mp.Vector3(2.6, 2.4, 2.0 if three_d else 0),
        boundary_layers=[mp.PML(0.35)],
        geometry=geometry,
        sources=[
            mp.Source(
                mp.GaussianSource(frequency=0.7, fwidth=0.5),
                component=component,
                center=mp.Vector3(-0.84, -0.17, 0.09 if three_d else 0),
            )
        ],
        resolution=resolution,
        courant=courant,
        dimensions=3 if three_d else 2,
        eps_averaging=case not in ("tensor_hz", "fixed_tensor"),
        chunk_layout=mp.BinaryPartition(data=[(mp.X, 0.0), 0, 1]) if split else None,
    )
    target = tm.PointTarget(
        position=mp.Vector3(0.82, 0.23, -0.11 if three_d else 0),
        component=component,
    )
    common = dict(
        design=design,
        simulation=simulation,
        t_final=24.0,
        dt=courant / resolution,
        sampling_interval=sampling_interval,
        chunk_balancer=None,
        reuse_simulation=reuse_simulation,
    )
    if case == "tensor_hz":
        return tm.MultiTDAObjective(
            targets=[target],
            wavelength_bands=[(1.1, 2.0)],
            weights=[1.0],
            kernel_length=31,
            pixel_chunk="auto",
            **common,
        )
    return tm.TDAObjective(target=target, **common)


def _design_and_direction(objective):
    rng = np.random.default_rng(7301)
    count = int(np.prod(objective.design.shape))
    weights = rng.uniform(0.2, 0.8, count)
    direction = rng.uniform(-1, 1, count)
    return weights, direction / np.linalg.norm(direction)


def _check_directional_derivative(objective):
    weights, direction = _design_and_direction(objective)
    value, gradient = objective.fom_and_grad(weights)
    adjoint = float(np.asarray(gradient).ravel() @ direction)
    differences = [
        (
            objective.fom(weights + h * direction)
            - objective.fom(weights - h * direction)
        )
        / (2 * h)
        for h in (2e-4, 1e-4)
    ]
    print(f"directional derivative: adjoint={adjoint:.12g}, central_fd={differences}")
    assert np.isfinite(value) and np.all(np.isfinite(gradient))
    assert abs(differences[-1]) > 1e-9
    np.testing.assert_allclose(differences[0], differences[1], rtol=0.001, atol=1e-10)
    assert np.sign(adjoint) == np.sign(differences[-1])
    np.testing.assert_allclose(adjoint, differences[-1], rtol=0.005, atol=1e-10)


@pytest.mark.parametrize(
    "case",
    [
        "smooth_ez",
        "smooth_hz",
        "tensor_hz",
        "tensor_3d",
        "tensor_3d_singleton",
        "tensor_xz_2d",
        "fixed_geometry",
        "fixed_tensor",
        "smooth_3d",
        "smooth_3d_singleton",
    ],
)
def test_material_tensor_gradient_matches_two_step_directional_fd(case):
    mp.verbosity(0)
    objective = _make_problem(case)
    if case in ("tensor_3d", "tensor_3d_singleton", "tensor_xz_2d", "fixed_tensor"):
        assert objective.gradient_components == (mp.Ex, mp.Ey, mp.Ez)
    _check_directional_derivative(objective)


@pytest.mark.parametrize("case", ["smooth_hz", "tensor_xz_2d"])
@pytest.mark.parametrize("beta", [128, 256])
def test_high_beta_averaged_gradient_matches_stable_directional_fd(case, beta):
    mp.verbosity(0)
    objective = _make_problem(case)
    objective.design.material_grid.beta = beta
    _check_directional_derivative(objective)


def test_averaged_tensor_sparse_history_matches_dense():
    mp.verbosity(0)
    dense = _make_problem("smooth_hz")
    sparse = _make_problem("smooth_hz", sampling_interval=2)
    weights, _ = _design_and_direction(dense)
    dense_value, dense_gradient = dense.fom_and_grad(weights)
    sparse_value, sparse_gradient = sparse.fom_and_grad(weights)
    print(
        "sparse/dense relative gradient error:",
        np.linalg.norm(sparse_gradient - dense_gradient)
        / np.linalg.norm(dense_gradient),
    )
    assert sparse.last_sampling_interval == 2
    np.testing.assert_allclose(sparse_value, dense_value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sparse_gradient, dense_gradient, rtol=0.005, atol=1e-8)


def _inverse_tensor_field(
    medium1,
    medium2,
    *,
    do_averaging=True,
    eps_averaging=True,
    mode="tama",
    weights=None,
    beta=16,
    eta=0.5,
):
    if weights is None:
        weights = np.linspace(0.1, 0.9, 9).reshape(3, 3)
    grid = mp.MaterialGrid(
        mp.Vector3(3, 3),
        medium1,
        medium2,
        weights=weights,
        do_averaging=do_averaging,
        beta=beta,
        eta=eta,
    )
    design = tm.DesignGrid(
        material_grid=grid,
        center=mp.Vector3(0.027, -0.031),
        size=mp.Vector3(1.2, 1.2),
        shape=(3, 3),
        background=medium1,
        design_material=medium2,
    )
    sim = mp.Simulation(
        cell_size=mp.Vector3(2, 2),
        resolution=20,
        force_all_components=True,
        geometry=[mp.Block(center=design.center, size=design.size, material=grid)],
        eps_averaging=eps_averaging,
    )
    try:
        if mode == "tama":
            from tama.backends.meep.sampling_grid import _prepare_native_material

            _prepare_native_material(sim, design)
        elif mode == "tensor":
            from tama_meep_native import native_sampler

            sim._init_structure(sim.k_point)
            native_sampler.configure_native_material_operator(
                int(sim.structure.this),
                int(sim.geps.this),
                eps_averaging,
                sim.subpixel_tol,
                sim.subpixel_maxeval,
            )
        sim.init_sim()
        return np.asarray(
            [
                [
                    [
                        sim.fields.get_chi1inv(c, d, mp.vec(x, y))
                        for d in (mp.X, mp.Y, mp.Z)
                    ]
                    for c in (mp.Ex, mp.Ey, mp.Ez)
                ]
                for x in np.linspace(-0.2, 0.2, 5)
                for y in np.linspace(-0.2, 0.2, 5)
            ]
        )
    finally:
        sim.reset_meep()


def test_anisotropic_averaging_changes_forward_constitutive_operator():
    mp.verbosity(0)
    medium1 = _rotated_medium((1.3, 1.7, 2.1), (0.2, -0.3, 0.4))
    medium2 = _rotated_medium((3.0, 4.2, 5.1), (-0.4, 0.5, -0.2))
    averaged = _inverse_tensor_field(medium1, medium2)
    unaveraged = _inverse_tensor_field(medium1, medium2, do_averaging=False)
    difference = np.max(np.abs(averaged - unaveraged))
    print("anisotropic averaging maximum inverse-tensor difference:", difference)
    assert difference > 1e-5


def test_anisotropic_averaging_preserves_homogeneous_material():
    mp.verbosity(0)
    medium = _rotated_medium((1.3, 1.7, 2.1), (0.2, -0.3, 0.4))
    averaged = _inverse_tensor_field(medium, medium)
    unaveraged = _inverse_tensor_field(medium, medium, do_averaging=False)
    np.testing.assert_allclose(averaged, unaveraged, rtol=1e-12, atol=1e-12)


def test_global_eps_averaging_disables_material_grid_tensor_averaging():
    mp.verbosity(0)
    medium1 = _rotated_medium((1.3, 1.7, 2.1), (0.2, -0.3, 0.4))
    medium2 = _rotated_medium((3.0, 4.2, 5.1), (-0.4, 0.5, -0.2))
    globally_disabled = _inverse_tensor_field(medium1, medium2, eps_averaging=False)
    both_disabled = _inverse_tensor_field(
        medium1,
        medium2,
        do_averaging=False,
        eps_averaging=False,
    )
    np.testing.assert_allclose(globally_disabled, both_disabled, rtol=1e-12, atol=1e-12)


def test_anisotropic_averaging_preserves_constant_weights():
    mp.verbosity(0)
    medium1 = _rotated_medium((1.3, 1.7, 2.1), (0.2, -0.3, 0.4))
    medium2 = _rotated_medium((3.0, 4.2, 5.1), (-0.4, 0.5, -0.2))
    weights = np.full((3, 3), 0.37)
    averaged = _inverse_tensor_field(medium1, medium2, weights=weights)
    unaveraged = _inverse_tensor_field(
        medium1, medium2, do_averaging=False, weights=weights
    )
    np.testing.assert_allclose(averaged, unaveraged, rtol=1e-12, atol=1e-12)


def test_anisotropic_projection_at_noncentral_eta_is_continuous_and_normalized():
    mp.verbosity(0)
    medium1 = _rotated_medium((1.3, 1.7, 2.1), (0.2, -0.3, 0.4))
    medium2 = _rotated_medium((3.0, 4.2, 5.1), (-0.4, 0.5, -0.2))
    tensors = []
    for medium in (medium1, medium2):
        d, o = medium.epsilon_diag, medium.epsilon_offdiag
        tensors.append(np.array(((d.x, o.x, o.y), (o.x, d.y, o.z), (o.y, o.z, d.z))))
    for value in (0.3 - 1e-7, 0.3, 0.3 + 1e-7):
        projected = (np.tanh(0.6) + np.tanh(2 * (value - 0.3))) / (
            np.tanh(0.6) + np.tanh(1.4)
        )
        expected = np.linalg.inv((1 - projected) * tensors[0] + projected * tensors[1])
        actual = _inverse_tensor_field(
            medium1,
            medium2,
            weights=np.full((3, 3), value),
            beta=2,
            eta=0.3,
        )
        np.testing.assert_allclose(
            actual, np.broadcast_to(expected, actual.shape), rtol=1e-11
        )


def test_tensor_averaging_reduces_to_meep_isotropic_operator():
    mp.verbosity(0)
    medium1, medium2 = mp.Medium(epsilon=1.3), mp.Medium(epsilon=4.2)
    tensor = _inverse_tensor_field(medium1, medium2, mode="tensor")
    stock = _inverse_tensor_field(medium1, medium2, mode="stock")
    np.testing.assert_allclose(tensor, stock, rtol=1e-11, atol=1e-12)


def test_uniform_mixed_averaged_grid_allows_forward_but_rejects_gradient():
    mp.verbosity(0)
    objective = _make_problem("tensor_xz_2d")
    weights = np.full(int(np.prod(objective.design.shape)), 0.37)
    assert np.isfinite(objective.fom(weights))
    with pytest.raises(RuntimeError, match="undefined interface normal"):
        objective.fom_and_grad(weights)


def test_hard_projection_away_from_threshold_has_zero_gradient():
    mp.verbosity(0)
    objective = _make_problem("fixed_geometry")
    objective.design.material_grid.beta = np.inf
    value, gradient = objective.fom_and_grad(
        np.full(int(np.prod(objective.design.shape)), 0.2)
    )
    assert np.isfinite(value)
    np.testing.assert_array_equal(gradient, np.zeros_like(gradient))


@pytest.mark.parametrize(
    "beta, eta, message",
    [
        (np.inf, 0.5, "beta=inf projection threshold"),
        (2.0, 0.3, "asymmetric projection threshold"),
    ],
)
def test_projection_threshold_rejects_undefined_gradient(beta, eta, message):
    mp.verbosity(0)
    objective = _make_problem("fixed_geometry")
    objective.design.material_grid.beta = beta
    objective.design.material_grid.eta = eta
    with pytest.raises(RuntimeError, match=message):
        objective.fom_and_grad(np.full(int(np.prod(objective.design.shape)), eta))


@pytest.mark.parametrize("case", ["tensor_xz_2d", "smooth_3d"])
def test_reused_tensor_simulation_matches_fresh_after_design_update(case):
    mp.verbosity(0)
    reused = _make_problem(case, reuse_simulation=True)
    weights, direction = _design_and_direction(reused)
    reused.fom_and_grad(weights)
    updated = weights + 0.01 * direction
    reused_value, reused_gradient = reused.fom_and_grad(updated)
    fresh = _make_problem(case)
    fresh_value, fresh_gradient = fresh.fom_and_grad(updated)
    np.testing.assert_allclose(reused_value, fresh_value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(reused_gradient, fresh_gradient, rtol=1e-11, atol=1e-12)


def test_tensor_gradient_requires_half_cell_clearance_from_pml():
    mp.verbosity(0)
    objective = _make_problem("smooth_ez")
    objective._simulation_spec.boundary_layers = [mp.PML(0.85)]
    weights, _ = _design_and_direction(objective)
    with pytest.raises(ValueError, match="half a grid cell of clearance"):
        objective.fom_and_grad(weights)


@pytest.mark.parametrize(
    "eps_averaging, do_averaging, requested, expected",
    [
        (True, True, 1e-4, 1e-8),
        (True, True, 1e-10, 1e-10),
        (True, False, 1e-4, 1e-4),
        (False, True, 1e-4, 1e-4),
    ],
)
def test_averaging_quadrature_tolerance_cap_preserves_stricter_values(
    eps_averaging,
    do_averaging,
    requested,
    expected,
):
    from tama.backends.meep.sampling_grid import _prepare_native_material

    objective = _make_problem("smooth_hz")
    objective.design.material_grid.do_averaging = do_averaging
    sim = objective._simulation_spec.make()
    sim.eps_averaging = eps_averaging
    sim.subpixel_tol = requested
    try:
        _prepare_native_material(sim, objective.design)
        assert sim.subpixel_tol == expected
    finally:
        sim.reset_meep()


@pytest.mark.parametrize("initialize_fields", [False, True])
def test_loose_averaging_tolerance_requires_fresh_simulation(initialize_fields):
    from tama.backends.meep.sampling_grid import _prepare_native_material

    mp.verbosity(0)
    objective = _make_problem("smooth_hz")
    sim = objective._simulation_spec.make()
    try:
        if initialize_fields:
            sim.init_sim()
        else:
            sim._init_structure(sim.k_point)
        with pytest.raises(ValueError, match="uninitialized|fresh"):
            _prepare_native_material(sim, objective.design)
        assert sim.subpixel_tol == 1e-4
    finally:
        sim.reset_meep()


@pytest.mark.mpi2
@pytest.mark.parametrize("case", ["smooth_hz", "smooth_3d"])
def test_averaged_tensor_design_across_mpi_boundary_matches_fd(case):
    if mp.count_processors() != 2:
        pytest.skip("requires exactly two MPI ranks")
    mp.verbosity(0)
    _check_directional_derivative(_make_problem(case, sampling_interval=2, split=True))
