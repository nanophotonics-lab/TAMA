import meep as mp
import numpy as np
import pytest

import tama as tm


def _make_cylindrical_problem(
    *,
    m,
    objective_kind="tda",
    axis_design=False,
    reuse_simulation=False,
    sampling_interval=1,
    target_component=None,
    target_radius=None,
    regional_target=False,
    surface_target=None,
    surface_normal="z",
):
    resolution = 10
    air = mp.Medium(epsilon=1.0)
    dielectric = mp.Medium(epsilon=2.25)
    shape = (3, 3)
    center = mp.Vector3(0.3 if axis_design else 1.0, 0.0, 0.0)
    size = mp.Vector3(0.6, 0.0, 0.6)
    material_grid = mp.MaterialGrid(
        mp.Vector3(shape[0], 1, shape[1]),
        air,
        dielectric,
    )
    design = tm.DesignGrid(
        material_grid=material_grid,
        center=center,
        size=size,
        shape=shape,
        background=air,
        design_material=dielectric,
        coordinate_system="cylindrical",
    )

    if m == 0:
        component = mp.Ep if target_component is None else target_component
        pulse = mp.GaussianSource(
            frequency=0.8,
            fwidth=0.6,
            cutoff=3,
        )
        sources = [
            mp.Source(
                pulse,
                component=mp.Ep,
                center=mp.Vector3(0.3, 0.0, -0.8),
            ),
            mp.Source(
                pulse,
                component=mp.Er,
                center=mp.Vector3(0.35, 0.0, -0.8),
                amplitude=0.4 + 0.25j,
            ),
        ]
        radius = 1.15 if target_radius is None else target_radius
        target_position = mp.Vector3(radius, 0.0, 0.8)
    else:
        component = mp.Er if target_component is None else target_component
        pulse = mp.GaussianSource(
            frequency=0.8,
            fwidth=0.6,
            cutoff=3,
        )
        sources = [
            mp.Source(
                pulse,
                component=mp.Er,
                center=mp.Vector3(0.85, 0.0, -0.8),
            ),
            mp.Source(
                pulse,
                component=mp.Ep,
                center=mp.Vector3(0.8, 0.0, -0.8),
                amplitude=0.35 + 0.45j,
            ),
        ]
        radius = 0.85 if target_radius is None else target_radius
        target_position = mp.Vector3(radius, 0.0, 0.8)

    simulation = tm.SimulationSpec(
        cell_size=mp.Vector3(2.0, 0.0, 3.0),
        boundary_layers=[
            mp.PML(0.4, direction=mp.R),
            mp.PML(0.4, direction=mp.Z),
        ],
        geometry=[
            mp.Block(
                center=center,
                size=size,
                material=material_grid,
            )
        ],
        sources=sources,
        resolution=resolution,
        dimensions=mp.CYLINDRICAL,
        m=m,
        eps_averaging=False,
    )
    if surface_target is not None:
        if surface_normal == "z":
            normal = mp.Vector3(0.0, 0.0, 1.0)
            positions = (
                target_position + mp.Vector3(-0.10, 0.0, 0.0),
                target_position + mp.Vector3(+0.12, 0.0, 0.0),
            )
        else:
            normal = mp.Vector3(1.0, 0.0, 0.0)
            positions = (
                target_position + mp.Vector3(0.0, 0.0, -0.08),
                target_position + mp.Vector3(0.0, 0.0, +0.11),
            )
        base_weights = (0.09, 0.11)
        spatial_weights = tuple(
            2.0 * np.pi * position.x * base_weight
            for position, base_weight in zip(positions, base_weights)
        )
        if surface_target == "flux":
            target = tm.FluxTarget(
                positions=positions,
                normal=normal,
                spatial_weights=spatial_weights,
            )
        else:
            first_electric = (1.0 + 0.2j, 0.8 - 0.1j)
            second_electric = (0.45 - 0.3j, 0.65 + 0.2j)
            reference_mode_fields = (
                {
                    mp.Er: first_electric,
                    mp.Hp: first_electric,
                    mp.Ep: second_electric,
                    mp.Hr: tuple(-value for value in second_electric),
                }
                if surface_normal == "z"
                else {
                    mp.Ep: first_electric,
                    mp.Hz: first_electric,
                    mp.Ez: second_electric,
                    mp.Hp: tuple(-value for value in second_electric),
                }
            )
            target = tm.EigenmodeCoefficientTarget(
                positions=positions,
                normal=normal,
                spatial_weights=spatial_weights,
                reference_mode_fields=reference_mode_fields,
                reference_m=m,
            )
    elif regional_target:
        target = tm.FieldRegionTarget(
            positions=(
                target_position + mp.Vector3(-0.08, 0.0, -0.06),
                target_position + mp.Vector3(0.07, 0.0, 0.05),
            ),
            component=component,
            sample_shape=(2,),
            spatial_weights=(0.7, 1.3),
        )
    else:
        target = tm.PointTarget(
            position=target_position,
            component=component,
        )
    common = dict(
        design=design,
        simulation=simulation,
        t_final=12.0,
        dt=0.5 / resolution,
        sampling_interval=sampling_interval,
        reuse_simulation=reuse_simulation,
    )
    if objective_kind == "tda":
        return tm.TDAObjective(target=target, **common)
    return tm.MultiTDAObjective(
        targets=[target],
        wavelength_bands=[(0.9, 2.0)],
        weights=[1.0],
        kernel_length=31,
        pixel_chunk=3,
        **common,
    )


def _directional_derivatives(objective):
    design = np.linspace(0.25, 0.75, 9)
    if objective._simulation_spec.m == 0:
        direction = np.array([1, 1, 1, -1, -1, 1, -1, -1, -1]) / 3.0
    else:
        direction = np.array([0.31, -0.47, 0.19, 0.53, -0.29, 0.41, -0.23, 0.17, -0.27])
        direction /= np.linalg.norm(direction)

    _, gradient = objective.fom_and_grad(design)
    step = 1.0e-3
    finite_difference = (
        objective.fom(design + step * direction)
        - objective.fom(design - step * direction)
    ) / (2.0 * step)
    adjoint_derivative = float(gradient @ direction)
    relative_error = abs(adjoint_derivative - finite_difference) / abs(
        finite_difference
    )
    return adjoint_derivative, finite_difference, relative_error


@pytest.mark.parametrize(
    ("m", "axis_design"),
    [(0, False), (1, True), (-1, True)],
)
def test_cylindrical_tda_gradient_matches_directional_finite_difference(
    m,
    axis_design,
):
    mp.verbosity(0)
    objective = _make_cylindrical_problem(
        m=m,
        axis_design=axis_design,
    )

    adjoint_derivative, finite_difference, relative_error = _directional_derivatives(
        objective
    )

    assert np.sign(adjoint_derivative) == np.sign(finite_difference)
    assert relative_error < 0.02


@pytest.mark.parametrize("objective_kind", ["tda", "multi"])
def test_cylindrical_field_region_matches_directional_fd(objective_kind):
    mp.verbosity(0)
    objective = _make_cylindrical_problem(
        m=0,
        objective_kind=objective_kind,
        regional_target=True,
    )

    adjoint_derivative, finite_difference, relative_error = _directional_derivatives(
        objective
    )

    assert np.sign(adjoint_derivative) == np.sign(finite_difference)
    assert relative_error < 0.02


@pytest.mark.parametrize(
    (
        "objective_kind",
        "surface_target",
        "surface_normal",
        "m",
        "reuse_simulation",
    ),
    [
        ("tda", "flux", "z", 1, False),
        ("multi", "flux", "r", 0, True),
        ("tda", "eigenmode", "r", -1, True),
        ("multi", "eigenmode", "z", 1, False),
    ],
)
def test_cylindrical_surface_targets_match_directional_fd(
    objective_kind,
    surface_target,
    surface_normal,
    m,
    reuse_simulation,
):
    mp.verbosity(0)
    objective = _make_cylindrical_problem(
        m=m,
        objective_kind=objective_kind,
        axis_design=m != 0,
        reuse_simulation=reuse_simulation,
        surface_target=surface_target,
        surface_normal=surface_normal,
    )

    adjoint_derivative, finite_difference, relative_error = _directional_derivatives(
        objective
    )

    assert set(objective.gradient_components) == {mp.Er, mp.Ep, mp.Ez}
    assert abs(finite_difference) > 1.0e-10
    assert np.sign(adjoint_derivative) == np.sign(finite_difference)
    assert relative_error < 0.02


@pytest.mark.parametrize(
    ("objective_kind", "m", "target_component"),
    [
        ("tda", 0, mp.Er),
        ("tda", 0, mp.Ep),
        ("tda", 0, mp.Ez),
        ("tda", 0, mp.Hr),
        ("tda", 0, mp.Hz),
        ("tda", 0, mp.Hp),
        ("tda", 1, mp.Er),
        ("tda", 1, mp.Ep),
        ("tda", 1, mp.Ez),
        ("tda", 1, mp.Hr),
        ("tda", 1, mp.Hp),
        ("tda", 1, mp.Hz),
        ("tda", -1, mp.Er),
        ("tda", -1, mp.Ep),
        ("tda", -1, mp.Ez),
        ("tda", -1, mp.Hr),
        ("tda", -1, mp.Hp),
        ("tda", -1, mp.Hz),
        ("multi", 0, mp.Hz),
        ("multi", 1, mp.Ep),
        ("multi", -1, mp.Hp),
    ],
)
def test_cylindrical_eh_targets_match_directional_finite_difference(
    objective_kind,
    m,
    target_component,
):
    mp.verbosity(0)
    objective = _make_cylindrical_problem(
        m=m,
        objective_kind=objective_kind,
        axis_design=m != 0,
        target_component=target_component,
    )

    adjoint_derivative, finite_difference, relative_error = _directional_derivatives(
        objective
    )

    assert np.sign(adjoint_derivative) == np.sign(finite_difference)
    assert relative_error < 0.02


@pytest.mark.parametrize(
    ("objective_kind", "m", "target_component"),
    [
        ("tda", 0, mp.Ez),
        ("tda", 0, mp.Hz),
        ("tda", 1, mp.Er),
        ("tda", 1, mp.Ep),
        ("tda", 1, mp.Hr),
        ("tda", 1, mp.Hp),
        ("tda", -1, mp.Er),
        ("tda", -1, mp.Ep),
        ("tda", -1, mp.Hr),
        ("tda", -1, mp.Hp),
        ("multi", 0, mp.Hz),
        ("multi", 1, mp.Ep),
        ("multi", -1, mp.Hr),
    ],
)
def test_cylindrical_on_axis_targets_match_directional_finite_difference(
    objective_kind,
    m,
    target_component,
):
    mp.verbosity(0)
    objective = _make_cylindrical_problem(
        m=m,
        objective_kind=objective_kind,
        axis_design=m != 0,
        target_component=target_component,
        target_radius=0.0,
    )

    adjoint_derivative, finite_difference, relative_error = _directional_derivatives(
        objective
    )

    assert np.sign(adjoint_derivative) == np.sign(finite_difference)
    assert relative_error < 0.02


@pytest.mark.parametrize(
    ("objective_kind", "m", "target_component"),
    [
        ("tda", 0, mp.Ep),
        ("tda", 1, mp.Ez),
        ("tda", 1, mp.Ep),
        ("multi", -1, mp.Hr),
    ],
)
def test_cylindrical_near_axis_targets_match_directional_finite_difference(
    objective_kind,
    m,
    target_component,
):
    mp.verbosity(0)
    objective = _make_cylindrical_problem(
        m=m,
        objective_kind=objective_kind,
        axis_design=True,
        target_component=target_component,
        target_radius=0.02,
    )

    adjoint_derivative, finite_difference, relative_error = _directional_derivatives(
        objective
    )

    assert np.sign(adjoint_derivative) == np.sign(finite_difference)
    assert relative_error < 0.02


@pytest.mark.parametrize(
    ("objective_kind", "m", "target_component"),
    [
        ("tda", 0, mp.Er),
        ("tda", 0, mp.Ep),
        ("tda", 0, mp.Hr),
        ("tda", 0, mp.Hp),
        ("tda", 1, mp.Ez),
        ("tda", 1, mp.Hz),
        ("tda", -1, mp.Ez),
        ("tda", -1, mp.Hz),
        ("multi", 0, mp.Ep),
        ("multi", 1, mp.Ez),
    ],
)
def test_cylindrical_on_axis_target_rejects_vanishing_component(
    objective_kind,
    m,
    target_component,
):
    mp.verbosity(0)
    objective = _make_cylindrical_problem(
        m=m,
        objective_kind=objective_kind,
        axis_design=m != 0,
        target_component=target_component,
        target_radius=0.0,
    )

    with pytest.raises(ValueError, match="vanishes at r=0"):
        objective.fom_and_grad(np.full(9, 0.5))


@pytest.mark.parametrize(
    ("objective_kind", "m", "target_component", "target_radius"),
    [
        ("tda", 1, mp.Ep, None),
        ("multi", -1, mp.Hp, None),
        ("tda", 1, mp.Ep, 0.0),
        ("multi", -1, mp.Hr, 0.0),
    ],
)
def test_cylindrical_reuse_matches_two_simulation_path(
    objective_kind,
    m,
    target_component,
    target_radius,
):
    mp.verbosity(0)
    design = np.linspace(0.25, 0.75, 9)
    separate = _make_cylindrical_problem(
        m=m,
        objective_kind=objective_kind,
        axis_design=True,
        target_component=target_component,
        target_radius=target_radius,
    )
    reused = _make_cylindrical_problem(
        m=m,
        objective_kind=objective_kind,
        axis_design=True,
        reuse_simulation=True,
        target_component=target_component,
        target_radius=target_radius,
    )

    separate_fom, separate_gradient = separate.fom_and_grad(design)
    reused_fom, reused_gradient = reused.fom_and_grad(design)

    assert np.isclose(reused_fom, separate_fom, rtol=2e-12, atol=2e-12)
    assert np.allclose(
        reused_gradient,
        separate_gradient,
        rtol=2e-10,
        atol=2e-12,
    )


@pytest.mark.parametrize(
    ("objective_kind", "m", "target_component", "target_radius"),
    [
        ("tda", 1, mp.Er, None),
        ("multi", -1, mp.Hp, None),
        ("tda", 0, mp.Hz, 0.0),
        ("multi", -1, mp.Hr, 0.0),
    ],
)
def test_cylindrical_nyquist_gradient_matches_directional_finite_difference(
    objective_kind,
    m,
    target_component,
    target_radius,
):
    mp.verbosity(0)
    objective = _make_cylindrical_problem(
        m=m,
        objective_kind=objective_kind,
        axis_design=True,
        sampling_interval=2,
        target_component=target_component,
        target_radius=target_radius,
    )

    adjoint_derivative, finite_difference, relative_error = _directional_derivatives(
        objective
    )

    assert np.sign(adjoint_derivative) == np.sign(finite_difference)
    assert relative_error < 0.03
