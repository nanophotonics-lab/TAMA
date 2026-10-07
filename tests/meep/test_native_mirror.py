"""Actual reduced-domain Mirror gradients against full-domain simulations."""

import meep as mp
import numpy as np
import pytest

import tama as tm


def _problem(*, mirror, tensor=False, phase=1):
    shape = (4, 4 if tensor else 5)
    low = mp.Medium(epsilon=1.0)
    high = (
        mp.Medium(epsilon_diag=mp.Vector3(3.0, 4.0, 5.0),
                  epsilon_offdiag=mp.Vector3(0, 0.35, 0))
        if tensor else mp.Medium(epsilon=4.0)
    )
    grid = mp.MaterialGrid(mp.Vector3(*shape), low, high,
                          do_averaging=tensor, beta=8 if tensor else 0)
    design = tm.DesignGrid(material_grid=grid, center=mp.Vector3(),
                           size=mp.Vector3(0.7, 0.7), shape=shape,
                           background=low, design_material=high)
    sources = [mp.Source(mp.GaussianSource(0.7, fwidth=0.5),
                         component=mp.Ez, center=mp.Vector3(-0.8, 0))]
    if phase == -1:
        sources = [mp.Source(mp.GaussianSource(0.7, fwidth=0.5),
                             component=mp.Ez, center=mp.Vector3(-0.8, y),
                             amplitude=sign)
                   for y, sign in ((0.25, 1), (-0.25, -1))]
    simulation = tm.SimulationSpec(
        cell_size=mp.Vector3(2.6, 2.4), resolution=14, courant=0.2,
        boundary_layers=[mp.PML(0.35)],
        geometry=[mp.Block(center=design.center, size=design.size, material=grid)],
        sources=sources, eps_averaging=tensor,
        symmetries=[mp.Mirror(mp.Y, phase=phase)] if mirror else [],
    )
    return tm.TDAObjective(
        simulation=simulation, design=design, t_final=24,
        dt=0.2 / 14, chunk_balancer=None,
        target=tm.PointTarget(position=mp.Vector3(0.8, -0.23), component=mp.Ez),
    )


@pytest.mark.parametrize("tensor,phase", [(False, 1), (False, -1), (True, 1)])
def test_mirror_matches_full_domain_value_and_symmetric_directional_derivative(tensor, phase):
    mp.verbosity(0)
    rng = np.random.default_rng(8193)
    weights = rng.uniform(0.2, 0.8, (4, 4 if tensor else 5))
    weights = 0.5 * (weights + weights[:, ::-1])
    direction = rng.normal(size=weights.shape)
    direction = 0.5 * (direction + direction[:, ::-1])
    direction /= np.linalg.norm(direction)
    weights, direction = weights.ravel(), direction.ravel()
    full = _problem(mirror=False, tensor=tensor, phase=phase)
    reduced = _problem(mirror=True, tensor=tensor, phase=phase)
    if tensor:
        # The unconstrained full-grid Jacobian is undefined at a zero normal;
        # the Mirror-preserving directional derivative is still well-defined.
        full_value, full_gradient = full.fom(weights), None
    else:
        full_value, full_gradient = full.fom_and_grad(weights)
    reduced_value, reduced_gradient = reduced.fom_and_grad(weights)
    np.testing.assert_allclose(reduced_value, full_value, rtol=1e-9, atol=1e-12)
    if full_gradient is not None:
        np.testing.assert_allclose(reduced_gradient @ direction,
                                   full_gradient @ direction, rtol=0.002, atol=1e-9)
    for h in (2e-4, 1e-4):
        finite_difference = (reduced.fom(weights + h * direction)
                             - reduced.fom(weights - h * direction)) / (2 * h)
        full_fd = (full.fom(weights + h * direction)
                   - full.fom(weights - h * direction)) / (2 * h)
        print("Mirror derivative:", tensor, phase, h, reduced_gradient @ direction, full_fd, finite_difference)
        np.testing.assert_allclose(finite_difference, full_fd, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(reduced_gradient @ direction, finite_difference,
                                   rtol=0.005, atol=1e-9)


def test_mirror_rejects_density_that_breaks_design_symmetry():
    objective = _problem(mirror=True)
    weights = np.full(20, 0.4)
    weights[0] += 0.1
    with pytest.raises(ValueError, match="(?i)symmetr"):
        objective.fom_and_grad(weights)


def test_averaged_mirror_rejects_odd_center_interpolation_knot():
    objective = _problem(mirror=True)
    objective.design.material_grid.do_averaging = True
    objective.design.material_grid.beta = 8
    objective._simulation_spec.eps_averaging = True
    with pytest.raises(ValueError, match="even grid size"):
        objective.fom_and_grad(np.full(20, 0.4))


def _parity_problem(case, mirror):
    three_d = case == "tensor3d"
    shape = (4, 4, 4) if three_d else (4, 4)
    component = mp.Hz if case == "magnetic" else mp.Ez
    symmetries = ([mp.Mirror(mp.Z, phase=-1)] if three_d else
                  [mp.Mirror(mp.Y, phase=-1)] if case == "magnetic" else
                  [mp.Mirror(mp.X), mp.Mirror(mp.Y)])
    low = mp.Medium(epsilon=1)
    high = (mp.Medium(epsilon_diag=mp.Vector3(3, 4, 5),
                      epsilon_offdiag=mp.Vector3(0.3, 0, 0))
            if three_d else mp.Medium(epsilon=4))
    grid = mp.MaterialGrid(mp.Vector3(*shape), low, high,
                          do_averaging=three_d, beta=8 if three_d else 0)
    size = mp.Vector3(0.6, 0.6, 0.6 if three_d else 0)
    design = tm.DesignGrid(material_grid=grid, center=mp.Vector3(), size=size,
                           shape=shape, background=low, design_material=high)
    simulation = tm.SimulationSpec(
        cell_size=mp.Vector3(2.25, 2.25, 2.25 if three_d else 0),
        resolution=8, courant=0.15, dimensions=3 if three_d else 2,
        boundary_layers=[mp.PML(0.375)], eps_averaging=three_d,
        geometry=[mp.Block(size=size, material=grid)],
        sources=[mp.Source(mp.GaussianSource(0.8, fwidth=0.6), component=component,
                           center=mp.Vector3(-0.6 if three_d else 0, 0, 0))],
        symmetries=symmetries if mirror else [],
    )
    return tm.TDAObjective(
        simulation=simulation, design=design, t_final=14, dt=0.15 / 8,
        chunk_balancer=None,
        target=tm.PointTarget(position=mp.Vector3(0.6, -0.23, -0.13 if three_d else 0),
                              component=component),
    ), [int(s.direction) for s in symmetries]


@pytest.mark.parametrize("case", ["magnetic", "two_planes", "tensor3d"])
def test_mirror_component_parity_multiple_planes_and_three_dimensions(case):
    mp.verbosity(0)
    full, axes = _parity_problem(case, False)
    reduced, _ = _parity_problem(case, True)
    rng = np.random.default_rng(341)
    weights = rng.uniform(0.2, 0.8, full.design.shape)
    direction = rng.normal(size=weights.shape)
    for axis in axes:
        weights = (weights + np.flip(weights, axis)) / 2
        direction = (direction + np.flip(direction, axis)) / 2
    direction /= np.linalg.norm(direction)
    weights, direction = weights.ravel(), direction.ravel()
    if case == "tensor3d":
        value_full, gradient_full = full.fom(weights), None
    else:
        value_full, gradient_full = full.fom_and_grad(weights)
    value_reduced, gradient_reduced = reduced.fom_and_grad(weights)
    np.testing.assert_allclose(value_reduced, value_full, rtol=1e-9, atol=1e-12)
    if gradient_full is not None:
        np.testing.assert_allclose(gradient_reduced @ direction, gradient_full @ direction,
                                   rtol=0.002, atol=1e-9)
    for h in (2e-4, 1e-4):
        fd = (reduced.fom(weights + h * direction) - reduced.fom(weights - h * direction)) / (2 * h)
        full_fd = (full.fom(weights + h * direction) - full.fom(weights - h * direction)) / (2 * h)
        print(case, "Mirror derivative:", h, gradient_reduced @ direction, full_fd, fd)
        np.testing.assert_allclose(fd, full_fd, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(gradient_reduced @ direction, fd, rtol=0.005, atol=1e-9)


@pytest.mark.mpi2
@pytest.mark.parametrize("case", ["magnetic", "two_planes", "tensor3d"])
def test_mirror_mpi2_chunk_boundary(case):
    if mp.count_processors() != 2:
        pytest.skip("requires exactly two MPI ranks")
    test_mirror_component_parity_multiple_planes_and_three_dimensions(case)


@pytest.mark.parametrize("direction", [mp.Z, mp.NO_DIRECTION])
def test_mirror_rejects_inactive_or_invalid_direction(direction):
    objective = _problem(mirror=False)
    objective._simulation_spec.symmetries = [mp.Mirror(direction)]
    with pytest.raises(ValueError, match="active Cartesian"):
        objective.fom(np.full(20, 0.4))


@pytest.mark.mpi2
@pytest.mark.parametrize("mismatch", ["position", "component"])
def test_mirror_mpi2_rejects_different_monitor_identity(mismatch):
    if mp.count_processors() != 2:
        pytest.skip("requires exactly two MPI ranks")
    from tama.backends.meep import native_sampler

    objective = _problem(mirror=True)
    objective.design.update_weights(np.full(20, 0.4))
    sim = objective._make_forward_simulation()
    sim.force_all_components = True
    try:
        sim.init_sim()
        x = 0.6 + (0.01 * mp.my_rank() if mismatch == "position" else 0)
        component = mp.Hz if mismatch == "component" and mp.my_rank() else mp.Ez
        plan = native_sampler.create_component_point_plan(int(sim.fields.this), [x], [0.2], component)
        with pytest.raises(RuntimeError, match="same monitor component and coordinates"):
            native_sampler.component_point_plan_indexed_stencil(plan)
    finally:
        sim.reset_meep()


def _adaptive_problem(objective_type, *, mirror, **kwargs):
    base = _problem(mirror=mirror)
    common = dict(simulation=base._simulation_spec, design=base.design,
                  t_final=24, dt=0.2 / 14, **kwargs)
    target = tm.PointTarget(position=mp.Vector3(0.8, -0.23), component=mp.Ez)
    if objective_type == "multi":
        return tm.MultiTDAObjective(targets=[target], wavelength_bands=[(1.1, 2.0)],
                                    weights=[1.0], kernel_length=31, **common)
    return tm.TDAObjective(target=target, **common)


@pytest.mark.parametrize("objective_type", ["single", "multi"])
def test_mirror_disables_default_adaptive_balancer(objective_type):
    assert _adaptive_problem(objective_type, mirror=True).chunk_balancer is None


@pytest.mark.parametrize("objective_type", ["single", "multi"])
def test_mirror_rejects_explicit_adaptive_balancer(objective_type):
    with pytest.raises(ValueError, match="Mirror.*adaptive chunk balancing"):
        _adaptive_problem(objective_type, mirror=True,
                          chunk_balancer=tm.AdaptiveAdjointChunkBalancer())


@pytest.mark.parametrize("objective_type", ["single", "multi"])
def test_mirror_rejects_adaptive_balancing_after_spec_mutation(objective_type):
    objective = _adaptive_problem(objective_type, mirror=False)
    objective._simulation_spec.symmetries = (mp.Mirror(mp.Y),)
    objective.update_design = lambda _: pytest.fail("must reject before design update")
    with pytest.raises(ValueError, match="Mirror.*adaptive chunk balancing"):
        objective.fom(np.full(20, 0.4))
