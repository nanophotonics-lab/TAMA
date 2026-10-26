"""Isotropic 2D MaterialGrids retain their uncoupled polarization."""

import meep as mp
import numpy as np
import pytest

import tama as tm
from tama.backends.meep import design_collection, sampling_grid


def _scene(component=mp.Ez, beta=1, coordinates="2d", eps_only=False, both_on=False):
    cylindrical = coordinates == "cylindrical"
    three_d = coordinates == "3d"
    shape = (4, 5, 3) if three_d else (4, 5)
    center = mp.Vector3(0.5, 0, 0) if cylindrical else mp.Vector3(0.027, -0.031)
    size = (
        mp.Vector3(0.64, 0, 0.58)
        if cylindrical
        else mp.Vector3(0.64, 0.58, 0.48 if three_d else 0)
    )
    low, high = mp.Medium(epsilon=1), mp.Medium(epsilon=4)
    grid = mp.MaterialGrid(
        mp.Vector3(4, 1, 5) if cylindrical else mp.Vector3(*shape),
        low,
        high,
        do_averaging=not eps_only,
        beta=beta,
    )
    design = tm.DesignGrid(
        grid,
        center,
        size,
        shape,
        background=low,
        design_material=high,
        coordinate_system="cylindrical" if cylindrical else "cartesian",
    )
    spec = tm.SimulationSpec(
        cell_size=(
            mp.Vector3(2.6, 0, 2.4)
            if cylindrical
            else mp.Vector3(2.6, 2.4, 2 if three_d else 0)
        ),
        dimensions=mp.CYLINDRICAL if cylindrical else 3 if three_d else 2,
        boundary_layers=[mp.PML(0.35)],
        resolution=14,
        courant=0.25,
        eps_averaging=eps_only or both_on,
        geometry=[mp.Block(center=center, size=size, material=grid)],
        sources=[
            mp.Source(
                mp.GaussianSource(0.7, fwidth=0.5),
                component,
                mp.Vector3(-0.84, -0.17),
            )
        ],
    )
    return design, spec


def _objective(kind, component, beta, eps_only=False, both_on=False):
    design, spec = _scene(component, beta, eps_only=eps_only, both_on=both_on)
    if mp.count_processors() == 2:
        spec.chunk_layout = mp.BinaryPartition(data=[(mp.X, 0.0), 0, 1])
    target = tm.PointTarget(mp.Vector3(0.82, 0.23), component=component)
    common = dict(
        design=design,
        simulation=spec,
        t_final=24,
        dt=0.25 / 14,
        sampling_interval=1,
        chunk_balancer=None,
    )
    if kind == "multi":
        return tm.MultiTDAObjective(
            targets=[target],
            wavelength_bands=[(1.1, 2.0)],
            weights=[1],
            kernel_length=31,
            **common,
        )
    return tm.TDAObjective(target=target, **common)


@pytest.mark.parametrize("beta", [1, 64])
@pytest.mark.parametrize("components", [(mp.Ez,), (mp.Ex, mp.Ey)])
@pytest.mark.parametrize(
    "eps_only,both_on",
    [(False, False), (True, False), (False, True)],
    ids=["do-only", "eps-only", "both-on"],
)
def test_projection_keeps_material_jacobian_without_forcing_all_components(
    beta, components, eps_only, both_on
):
    component = mp.Ez if components == (mp.Ez,) else mp.Hz
    design, spec = _scene(component, beta, eps_only=eps_only, both_on=both_on)
    sim = spec.make()
    try:
        assert sampling_grid._uses_material_jacobian(sim, design)
        assert (
            sampling_grid._native_design_components(sim, design, components)
            == components
        )
        assert not sampling_grid._requires_all_material_components(sim, design)
        sampling_grid._prepare_native_material(sim, design)
        assert not sim.force_all_components
        assert sim.structure is None and sim.fields is None
        assert sim.subpixel_tol == (1e-8 if both_on else 1e-4)
    finally:
        sim.reset_meep()


@pytest.mark.parametrize(
    "eps_only,both_on",
    [(False, False), (True, False), (False, True)],
    ids=["do-only", "eps-only", "both-on"],
)
@pytest.mark.parametrize(
    "case",
    [
        "3d",
        "cylindrical",
        "endpoint",
        "fixed_medium",
        "fixed_grid",
        "default",
        "extra",
        "mu_offdiag",
        "mu_diagonal",
        "electric_dispersion",
        "magnetic_dispersion",
        "callback",
        "kz",
    ],
)
def test_coupled_or_unproven_materials_keep_all_components(case, eps_only, both_on):
    design, spec = _scene(
        coordinates=case if case in ("3d", "cylindrical") else "2d",
        eps_only=eps_only,
        both_on=both_on,
    )
    tensor = mp.Medium(
        epsilon_diag=mp.Vector3(2, 3, 4), epsilon_offdiag=mp.Vector3(0, 0.2, 0)
    )
    if case == "endpoint":
        design.material_grid.medium2 = tensor
    elif case == "kz":
        spec.k_point = mp.Vector3(0, 0, 0.2)
    elif case not in ("3d", "cylindrical", "default", "extra"):
        material = tensor
        if case == "fixed_grid":
            material = mp.MaterialGrid(mp.Vector3(2, 2), mp.Medium(), tensor)
        elif case == "mu_offdiag":
            material = mp.Medium(mu_offdiag=mp.Vector3(0, 0.1, 0))
        elif case == "mu_diagonal":
            material = mp.Medium(mu_diag=mp.Vector3(1, 2, 1))
        elif case in ("electric_dispersion", "magnetic_dispersion"):
            material = mp.Medium()
            attribute = (
                "E_susceptibilities"
                if case == "electric_dispersion"
                else "H_susceptibilities"
            )
            setattr(
                material,
                attribute,
                [mp.LorentzianSusceptibility(frequency=0.9, gamma=0.05, sigma=0.1)],
            )
        elif case == "callback":
            material = lambda point: mp.Medium(epsilon=3)
        spec.geometry = [
            *spec.geometry,
            mp.Block(
                center=mp.Vector3(0.7),
                size=mp.Vector3(0.1, 0.2),
                material=material,
            ),
        ]
    sim = spec.make()
    if case == "default":
        sim.default_material = tensor
    elif case == "extra":
        sim.extra_materials = [tensor]
    try:
        expected = (
            (mp.Er, mp.Ep, mp.Ez)
            if case == "cylindrical"
            else (mp.Ex, mp.Ey, mp.Ez)
        )
        assert sampling_grid._requires_all_material_components(sim, design)
        assert (
            sampling_grid._native_design_components(sim, design, (mp.Ez,))
            == expected
        )
    finally:
        sim.reset_meep()


@pytest.mark.parametrize(
    "location", ["fixed_medium", "geometry_grid", "default_grid", "extra_grid"]
)
def test_eps_averaging_distinguishes_fixed_geometry_from_isotropic_grids(location):
    design, spec = _scene(eps_only=True)
    material = mp.Medium(epsilon=3)
    if location != "fixed_medium":
        material = mp.MaterialGrid(
            mp.Vector3(2, 2), mp.Medium(), material, do_averaging=True, beta=8
        )
    sim = spec.make()
    if location == "default_grid":
        sim.default_material = material
    elif location == "extra_grid":
        sim.extra_materials = [material]
    else:
        sim.geometry.append(
            mp.Block(
                center=mp.Vector3(0.7), size=mp.Vector3(0.1, 0.2), material=material
            )
        )
    try:
        fixed_geometry = location == "fixed_medium"
        assert (
            sampling_grid._requires_all_material_components(sim, design)
            == fixed_geometry
        )
        expected = (mp.Ex, mp.Ey, mp.Ez) if fixed_geometry else (mp.Ez,)
        assert (
            sampling_grid._native_design_components(sim, design, (mp.Ez,))
            == expected
        )
    finally:
        sim.reset_meep()


@pytest.mark.parametrize("eps_averaging", [None, True])
@pytest.mark.parametrize("do_averaging", [False, True])
def test_multiple_isotropic_grids_keep_polarization_with_default_averaging(
    eps_averaging, do_averaging
):
    design, spec = _scene(eps_only=not do_averaging, both_on=do_averaging)
    spec.eps_averaging = eps_averaging
    low, high = mp.Medium(), mp.Medium(epsilon=3)
    second = tm.DesignGrid(
        mp.MaterialGrid(
            mp.Vector3(2, 2), low, high, do_averaging=do_averaging, beta=8
        ),
        mp.Vector3(0.7),
        mp.Vector3(0.1, 0.2),
        (2, 2),
        background=low,
        design_material=high,
    )
    spec.geometry.append(
        mp.Block(center=second.center, size=second.size, material=second.material_grid)
    )
    designs = design_collection._DesignCollection((design, second))
    sim = spec.make()
    try:
        assert sim.eps_averaging
        for simulation in (spec, sim):
            assert not design_collection._collection_requires_all_material_components(
                simulation, designs
            )
            assert design_collection._collection_components(
                simulation, designs, (mp.Ez,)
            ) == ((0, mp.Ez), (1, mp.Ez))
    finally:
        sim.reset_meep()


@pytest.mark.mpi2
@pytest.mark.parametrize("kind", ["tda", "multi"])
@pytest.mark.parametrize("component", [mp.Ez, mp.Hz], ids=["ez", "hz"])
@pytest.mark.parametrize("beta", [1, 64])
@pytest.mark.parametrize(
    "eps_only,both_on",
    [(False, False), (True, False), (False, True)],
    ids=["do-only", "eps-only", "both-on"],
)
def test_projected_polarization_matches_full_components_and_two_step_fd(
    monkeypatch, kind, component, beta, eps_only, both_on
):
    mp.verbosity(0)
    objective = _objective(
        kind, component, beta, eps_only=eps_only, both_on=both_on
    )
    expected = (mp.Ez,) if component == mp.Ez else (mp.Ex, mp.Ey)
    assert objective.gradient_components == expected
    rng = np.random.default_rng(7301)
    weights = rng.uniform(0.2, 0.8, 20)
    direction = rng.uniform(-1, 1, 20)
    direction /= np.linalg.norm(direction)
    value, gradient = objective.fom_and_grad(weights)
    assert np.isfinite(value) and np.all(np.isfinite(gradient))

    with monkeypatch.context() as forced:
        for module in (sampling_grid, design_collection):
            forced.setattr(
                module, "_requires_all_material_components", lambda sim, design: True
            )
        baseline = _objective(
            kind, component, beta, eps_only=eps_only, both_on=both_on
        )
        assert baseline.gradient_components == (mp.Ex, mp.Ey, mp.Ez)
        full_value, full_gradient = baseline.fom_and_grad(weights)
    np.testing.assert_allclose(value, full_value, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(gradient, full_gradient, rtol=1e-9, atol=1e-11)

    adjoint = float(gradient @ direction)
    differences = [
        (
            objective.fom(weights + h * direction)
            - objective.fom(weights - h * direction)
        )
        / (2 * h)
        for h in (2e-4, 1e-4)
    ]
    print(
        "Projected polarization FD:",
        kind,
        component,
        beta,
        eps_only,
        both_on,
        adjoint,
        differences,
    )
    assert abs(differences[-1]) > 1e-9
    np.testing.assert_allclose(differences[0], differences[1], rtol=0.001, atol=1e-10)
    assert np.sign(adjoint) == np.sign(differences[-1])
    np.testing.assert_allclose(adjoint, differences[-1], rtol=0.005, atol=1e-10)


@pytest.mark.mpi2
def test_averaging_changes_hz_forward_and_retains_xy_coupling():
    mp.verbosity(0)
    weights = np.linspace(0.2, 0.8, 20)
    averaged = _objective("tda", mp.Hz, 64, both_on=True)
    unaveraged = _objective("tda", mp.Hz, 64, eps_only=True)
    assert averaged.gradient_components == (mp.Ex, mp.Ey)
    averaged_value = averaged.fom(weights)
    unaveraged_value = unaveraged.fom(weights)
    assert np.isfinite(averaged_value) and np.isfinite(unaveraged_value)
    assert not np.isclose(averaged_value, unaveraged_value, rtol=1e-4, atol=1e-12)

    design, spec = _scene(mp.Hz, 64, both_on=True)
    design.material_grid.update_weights(weights)
    sim = spec.make()
    try:
        sampling_grid._prepare_native_material(sim, design)
        assert not sim.force_all_components
        sim.init_sim()
        tensors = np.asarray(
            [
                [
                    [
                        sim.fields.get_chi1inv(component, direction, mp.vec(x, y))
                        for direction in (mp.X, mp.Y, mp.Z)
                    ]
                    for component in (mp.Ex, mp.Ey, mp.Ez)
                ]
                for x in np.linspace(-0.2, 0.2, 5)
                for y in np.linspace(-0.2, 0.2, 5)
            ]
        )
        assert np.all(np.isfinite(tensors))
        assert np.max(np.abs(tensors[:, 0, 1])) > 1e-5
        np.testing.assert_allclose(tensors[:, :2, 2], 0, atol=1e-12)
        np.testing.assert_allclose(tensors[:, 2, :2], 0, atol=1e-12)
    finally:
        sim.reset_meep()
