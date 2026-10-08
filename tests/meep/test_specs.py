from typing import Optional, get_type_hints

import numpy as np
import meep as mp
import pytest

import tama as tm


def test_design_grid_bundles_material_grid_coordinates_and_updates():
    air = mp.Medium(epsilon=1.0)
    dielectric = mp.Medium(epsilon=4.0)
    material_grid = mp.MaterialGrid(mp.Vector3(2, 3), air, dielectric)

    design = tm.DesignGrid(
        material_grid=material_grid,
        center=mp.Vector3(0.5, -0.5),
        size=mp.Vector3(2.0, 3.0),
        shape=(2, 3),
        background=air,
        design_material=dielectric,
    )

    design.update_weights(np.linspace(0.0, 1.0, 6))

    assert len(design.coords_x) == 2
    assert len(design.coords_y) == 3
    assert design.cell_area == 1.0
    assert design.material_factor == 3.0


def test_cylindrical_design_grid_maps_rz_shape_to_material_grid():
    air = mp.Medium(epsilon=1.0)
    dielectric = mp.Medium(epsilon=4.0)
    material_grid = mp.MaterialGrid(mp.Vector3(2, 1, 3), air, dielectric)
    design = tm.DesignGrid(
        material_grid=material_grid,
        center=mp.Vector3(1.5, 0.0, -0.5),
        size=mp.Vector3(2.0, 0.0, 3.0),
        shape=(2, 3),
        background=air,
        design_material=dielectric,
        coordinate_system="cylindrical",
    )
    weights = np.linspace(0.0, 1.0, 6)

    design.update_weights(weights)

    assert design.is_cylindrical
    assert design.dimensions == 2
    assert design.spacing == (1.0, 1.0)
    assert design.coords_r == [1.0, 2.0]
    assert design.coords_x == design.coords_r
    assert design.coords_y is None
    assert design.coords_z == [-1.5, -0.5, 0.5]
    assert design.cell_area == 1.0
    assert design.cell_volume is None
    np.testing.assert_allclose(
        np.asarray(material_grid.weights).reshape(2, 1, 3),
        weights.reshape(2, 1, 3),
    )


def test_cylindrical_design_grid_requires_degenerate_y_material_grid():
    air = mp.Medium(epsilon=1.0)
    dielectric = mp.Medium(epsilon=4.0)

    with np.testing.assert_raises_regex(
        ValueError,
        "cylindrical DesignGrid shape must match",
    ):
        tm.DesignGrid(
            material_grid=mp.MaterialGrid(
                mp.Vector3(2, 3, 1),
                air,
                dielectric,
            ),
            center=mp.Vector3(1.0, 0.0, 0.0),
            size=mp.Vector3(2.0, 0.0, 3.0),
            shape=(2, 3),
            background=air,
            design_material=dielectric,
            coordinate_system="cylindrical",
        )


def test_cylindrical_design_grid_rejects_negative_radius():
    air = mp.Medium(epsilon=1.0)
    dielectric = mp.Medium(epsilon=4.0)

    with np.testing.assert_raises_regex(ValueError, "must not extend below r=0"):
        tm.DesignGrid(
            material_grid=mp.MaterialGrid(
                mp.Vector3(2, 1, 3),
                air,
                dielectric,
            ),
            center=mp.Vector3(0.25, 0.0, 0.0),
            size=mp.Vector3(1.0, 0.0, 1.5),
            shape=(2, 3),
            background=air,
            design_material=dielectric,
            coordinate_system="cylindrical",
        )


def test_simulation_spec_uses_default_or_override_sources():
    default_sources = [
        mp.Source(
            mp.GaussianSource(frequency=1.0, fwidth=0.1),
            component=mp.Ez,
            center=mp.Vector3(),
        )
    ]
    override_sources = [
        mp.Source(
            mp.GaussianSource(frequency=1.2, fwidth=0.1),
            component=mp.Ez,
            center=mp.Vector3(),
        )
    ]
    spec = tm.SimulationSpec(
        cell_size=mp.Vector3(1.0, 1.0),
        boundary_layers=[],
        geometry=[],
        sources=lambda: default_sources,
        resolution=10,
    )

    default_sim = spec.make()
    override_sim = spec.make(override_sources)

    assert default_sim.sources is default_sources
    assert override_sim.sources is override_sources


def test_simulation_spec_resolves_callable_sources_once_for_explicit_reuse():
    calls = []

    def make_sources():
        calls.append(None)
        return [
            mp.Source(
                mp.GaussianSource(frequency=1.0, fwidth=0.1),
                component=mp.Ez,
                center=mp.Vector3(0.25, 0.0),
            )
        ]

    spec = tm.SimulationSpec(
        cell_size=mp.Vector3(1.0, 1.0),
        sources=make_sources,
    )

    resolved = spec.resolve_sources()
    simulation = spec.make(resolved)

    assert len(calls) == 1
    assert simulation.sources is resolved


def test_simulation_spec_forwards_default_and_custom_courant():
    default_spec = tm.SimulationSpec(cell_size=mp.Vector3(1.0, 1.0))
    custom_spec = tm.SimulationSpec(
        cell_size=mp.Vector3(1.0, 1.0),
        resolution=10,
        courant=0.3,
    )

    assert default_spec.courant == 0.5
    assert default_spec.make().Courant == 0.5
    custom_sim = custom_spec.make()
    assert custom_sim.Courant == 0.3
    custom_sim.init_sim()
    np.testing.assert_allclose(custom_sim.fields.dt, 0.03)
    custom_sim.reset_meep()


def test_simulation_spec_distinguishes_metal_and_gamma_periodic_boundaries():
    metal_spec = tm.SimulationSpec(cell_size=mp.Vector3(1.0, 1.0))
    periodic_spec = tm.SimulationSpec(
        cell_size=mp.Vector3(1.0, 1.0),
        k_point=mp.Vector3(),
    )

    metal_sim = metal_spec.make()
    periodic_forward = periodic_spec.make()
    periodic_adjoint = periodic_spec.make([])

    assert metal_spec.k_point is False
    assert metal_sim.k_point is False
    assert periodic_spec.k_point is not False
    assert periodic_forward.k_point is not False
    assert periodic_adjoint.k_point is not False
    assert tuple(periodic_forward.k_point) == (0.0, 0.0, 0.0)
    assert tuple(periodic_adjoint.k_point) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "k_point",
    (True, (0.0, 0.0, 0.0), mp.Vector3(0.1), mp.Vector3(np.nan)),
)
def test_simulation_spec_rejects_unsupported_k_point(k_point):
    with pytest.raises(
        (TypeError, ValueError),
        match="k_point",
    ):
        tm.SimulationSpec(
            cell_size=mp.Vector3(1.0, 1.0),
            k_point=k_point,
        )


def test_simulation_spec_rejects_invalid_courant():
    for courant in (True, 0.0, -0.1, np.nan, np.inf):
        with np.testing.assert_raises_regex(
            ValueError,
            "courant must be a positive finite number",
        ):
            tm.SimulationSpec(
                cell_size=mp.Vector3(1.0, 1.0),
                courant=courant,
            )


@pytest.mark.parametrize("resolution", (True, 0.0, -1.0, np.nan, np.inf))
def test_simulation_spec_rejects_invalid_resolution(resolution):
    with pytest.raises(ValueError, match="resolution must be a positive finite"):
        tm.SimulationSpec(
            cell_size=mp.Vector3(1.0, 1.0),
            resolution=resolution,
        )


@pytest.mark.parametrize("dimensions", (True, 0, 4, -3, "2"))
def test_simulation_spec_rejects_invalid_dimensions(dimensions):
    with pytest.raises(ValueError, match="dimensions must be"):
        tm.SimulationSpec(
            cell_size=mp.Vector3(1.0, 1.0),
            dimensions=dimensions,
        )


def test_cylindrical_simulation_spec_uses_stored_or_overridden_integer_m():
    spec = tm.SimulationSpec(
        cell_size=mp.Vector3(2.0, 0.0, 3.0),
        sources=[],
        resolution=10,
        dimensions=mp.CYLINDRICAL,
        m=1,
    )

    forward_sim = spec.make()
    adjoint_sim = spec.make([], m=-1)

    assert spec.is_cylindrical
    assert spec.m == 1
    assert forward_sim.m == 1
    assert adjoint_sim.m == -1


def test_simulation_spec_rejects_noninteger_or_cartesian_nonzero_m():
    with np.testing.assert_raises_regex(ValueError, "m must be an integer"):
        tm.SimulationSpec(cell_size=mp.Vector3(1.0, 1.0), m=1.5)

    with np.testing.assert_raises_regex(
        ValueError,
        "nonzero m requires dimensions=mp.CYLINDRICAL",
    ):
        tm.SimulationSpec(cell_size=mp.Vector3(1.0, 1.0), m=1)


def test_simulation_spec_chunk_layout_annotation_is_parameterized():
    assert (
        get_type_hints(tm.SimulationSpec)["chunk_layout"]
        == Optional[mp.BinaryPartition]
    )


def test_tda_objective_accepts_bundled_design_simulation_and_target():
    air = mp.Medium(epsilon=1.0)
    dielectric = mp.Medium(epsilon=4.0)
    design = tm.DesignGrid(
        material_grid=mp.MaterialGrid(mp.Vector3(2, 2), air, dielectric),
        center=mp.Vector3(),
        size=mp.Vector3(2.0, 2.0),
        shape=(2, 2),
        background=air,
        design_material=dielectric,
    )
    simulation = tm.SimulationSpec(
        cell_size=mp.Vector3(1.0, 1.0),
        boundary_layers=[],
        geometry=[],
        sources=[],
        resolution=10,
    )
    target = tm.PointTarget(
        position=mp.Vector3(0.0, 0.25),
        component=mp.Ez,
        adjoint_source_size=mp.Vector3(),
        adjoint_source_amplitude=2.0,
    )

    obj = tm.TDAObjective(
        design=design,
        simulation=simulation,
        target=target,
        t_final=1.0,
        dt=0.05,
    )

    assert obj.design is design
    assert obj.update_design == design.update_weights
    assert obj.objective.monitor_position == target.position
    assert obj.objective.component == mp.Ez
    assert obj.design.material_factor == design.material_factor
    assert obj.objective.adjoint_source_amplitude == 2.0


@pytest.mark.parametrize(
    "position",
    (
        mp.Vector3(np.nan, 0.0, 0.0),
        mp.Vector3(np.inf, 0.0, 0.0),
        mp.Vector3(1.0j, 0.0, 0.0),
    ),
)
def test_point_target_rejects_invalid_position(position):
    with pytest.raises((TypeError, ValueError), match="position must"):
        tm.PointTarget(position, mp.Ez)


@pytest.mark.parametrize(
    "source_size",
    (
        mp.Vector3(-0.1, 0.0, 0.0),
        mp.Vector3(np.nan, 0.0, 0.0),
        mp.Vector3(1.0j, 0.0, 0.0),
    ),
)
def test_point_target_rejects_invalid_adjoint_source_size(source_size):
    with pytest.raises((TypeError, ValueError), match="adjoint_source_size must"):
        tm.PointTarget(
            mp.Vector3(),
            mp.Ez,
            adjoint_source_size=source_size,
        )


@pytest.mark.parametrize("amplitude", (np.nan, np.inf, 1.0j * np.inf, "bad"))
def test_targets_reject_invalid_adjoint_source_amplitude(amplitude):
    for target_factory in (
        lambda: tm.PointTarget(
            mp.Vector3(),
            mp.Ez,
            adjoint_source_amplitude=amplitude,
        ),
        lambda: tm.FieldRegionTarget(
            (mp.Vector3(),),
            mp.Ez,
            adjoint_source_amplitude=amplitude,
        ),
    ):
        with pytest.raises((TypeError, ValueError), match="amplitude must"):
            target_factory()


def test_field_region_target_normalizes_shape_and_spatial_weights():
    positions = [
        mp.Vector3(-0.2, 0.1),
        mp.Vector3(0.0, 0.1),
        mp.Vector3(0.2, 0.1),
        mp.Vector3(0.4, 0.1),
    ]

    target = tm.FieldRegionTarget(
        positions=positions,
        component=mp.Ez,
        sample_shape=(2, 2),
        spatial_weights=[1.0, 2.0, 3.0, 4.0],
        adjoint_source_amplitude=2j,
    )

    assert target.positions == tuple(positions)
    assert target.sample_shape == (2, 2)
    assert target.spatial_weights == (1.0, 2.0, 3.0, 4.0)
    assert target.adjoint_source_amplitude == 2j


def test_field_region_target_validates_positions_shape_and_weights():
    with np.testing.assert_raises_regex(ValueError, "must not be empty"):
        tm.FieldRegionTarget([], mp.Ez)

    positions = [mp.Vector3(), mp.Vector3(0.1)]
    with np.testing.assert_raises_regex(ValueError, "sample_shape must match"):
        tm.FieldRegionTarget(positions, mp.Ez, sample_shape=(3,))

    with np.testing.assert_raises_regex(
        ValueError,
        "spatial_weights must match",
    ):
        tm.FieldRegionTarget(
            positions,
            mp.Ez,
            spatial_weights=[1.0],
        )

    with np.testing.assert_raises_regex(
        ValueError,
        "finite, nonnegative",
    ):
        tm.FieldRegionTarget(
            positions,
            mp.Ez,
            spatial_weights=[1.0, -1.0],
        )

    with np.testing.assert_raises_regex(
        TypeError,
        "spatial_weights must be real",
    ):
        tm.FieldRegionTarget(
            positions,
            mp.Ez,
            spatial_weights=np.asarray([1.0, 2.0j]),
        )

    with np.testing.assert_raises_regex(
        ValueError,
        "positions must have finite coordinates",
    ):
        tm.FieldRegionTarget(
            [mp.Vector3(float("nan"), 0.0, 0.0)],
            mp.Ez,
        )


def test_flux_target_normalizes_positions_normal_and_spatial_weights():
    positions = [mp.Vector3(0.1, -0.2), mp.Vector3(0.1, 0.2)]

    target = tm.FluxTarget(
        positions=positions,
        normal=mp.Vector3(-1.0, 0.0, 0.0),
        spatial_weights=[0.25, 0.75],
    )

    assert target.positions == tuple(positions)
    assert (target.normal.x, target.normal.y, target.normal.z) == (
        -1.0,
        0.0,
        0.0,
    )
    assert target.spatial_weights == (0.25, 0.75)
    assert tm.FluxTarget(
        positions,
        mp.Vector3(0.0, 1.0),
    ).spatial_weights == (1.0, 1.0)


def test_flux_target_validates_positions_normal_and_spatial_weights():
    with np.testing.assert_raises_regex(ValueError, "must not be empty"):
        tm.FluxTarget([], mp.Vector3(1.0))

    positions = [mp.Vector3(), mp.Vector3(0.1)]
    for invalid_normal in (
        mp.Vector3(),
        mp.Vector3(0.5),
        mp.Vector3(1.0, 1.0),
    ):
        with np.testing.assert_raises_regex(
            ValueError,
            "signed coordinate-axis unit vector",
        ):
            tm.FluxTarget(positions, invalid_normal)

    with np.testing.assert_raises_regex(
        ValueError,
        "spatial_weights must match",
    ):
        tm.FluxTarget(
            positions,
            mp.Vector3(0.0, 0.0, 1.0),
            spatial_weights=[1.0],
        )

    with np.testing.assert_raises_regex(
        ValueError,
        "finite, nonnegative",
    ):
        tm.FluxTarget(
            positions,
            mp.Vector3(0.0, 1.0),
            spatial_weights=[0.0, 0.0],
        )

    with np.testing.assert_raises_regex(
        TypeError,
        "spatial_weights must be real",
    ):
        tm.FluxTarget(
            positions,
            mp.Vector3(0.0, -1.0),
            spatial_weights=[1.0, 2.0j],
        )


def test_eigenmode_coefficient_target_normalizes_public_settings():
    positions = [mp.Vector3(0.2, -0.5), mp.Vector3(0.2, 0.5)]
    mode_region = mp.Volume(
        center=mp.Vector3(0.2, 0.0),
        size=mp.Vector3(0.0, 1.0),
    )

    target = tm.EigenmodeCoefficientTarget(
        positions=positions,
        normal=mp.Vector3(-1.0),
        mode_region=mode_region,
        frequency=np.float64(0.75),
        mode=np.int64(2),
        spatial_weights=[0.25, 0.75],
        eig_parity=np.int64(mp.ODD_Z),
        eig_resolution=np.float64(24.0),
        eig_tolerance=np.float64(1.0e-10),
    )

    assert target.positions == tuple(positions)
    assert (target.normal.x, target.normal.y, target.normal.z) == (
        -1.0,
        0.0,
        0.0,
    )
    assert target.mode_region is mode_region
    assert target.frequency == 0.75
    assert target.mode == 2
    assert target.spatial_weights == (0.25, 0.75)
    assert target.eig_parity == mp.ODD_Z
    assert target.eig_resolution == 24.0
    assert target.eig_tolerance == 1.0e-10
    assert "EigenmodeCoefficientTarget" in tm.__all__

    default_target = tm.EigenmodeCoefficientTarget(
        positions,
        mp.Vector3(1.0),
        mode_region,
        0.75,
    )
    assert default_target.mode == 1
    assert default_target.spatial_weights == (1.0, 1.0)
    assert default_target.eig_parity == mp.NO_PARITY
    assert default_target.eig_resolution == 0.0
    assert default_target.eig_tolerance == 1.0e-12


def test_eigenmode_coefficient_target_accepts_fixed_reference_fields():
    positions = [mp.Vector3(0.4, 0.0, -0.2), mp.Vector3(0.8, 0.0, -0.2)]
    reference_mode_fields = {
        mp.Er: [1.0 + 0.5j, 2.0 - 0.25j],
        mp.Hp: [0.75 - 0.1j, 1.5 + 0.2j],
    }

    target = tm.EigenmodeCoefficientTarget(
        positions=positions,
        normal=mp.Vector3(0.0, 0.0, 1.0),
        spatial_weights=[0.3, 0.7],
        reference_mode_fields=reference_mode_fields,
        reference_m=np.int64(1),
    )

    assert target.mode_region is None
    assert target.frequency is None
    assert target.reference_m == 1
    assert set(target.reference_mode_fields) == {mp.Er, mp.Hp}
    np.testing.assert_array_equal(
        target.reference_mode_fields[mp.Er],
        np.asarray(reference_mode_fields[mp.Er]),
    )


def test_eigenmode_coefficient_target_validates_fixed_reference_fields():
    positions = [mp.Vector3(0.4), mp.Vector3(0.8)]

    with pytest.raises(ValueError, match="mode_region and frequency or"):
        tm.EigenmodeCoefficientTarget(
            positions=positions,
            normal=mp.Vector3(0.0, 0.0, 1.0),
        )

    with pytest.raises(ValueError, match="match positions"):
        tm.EigenmodeCoefficientTarget(
            positions=positions,
            normal=mp.Vector3(0.0, 0.0, 1.0),
            reference_mode_fields={mp.Er: [1.0]},
            reference_m=0,
        )

    with pytest.raises(ValueError, match="finite"):
        tm.EigenmodeCoefficientTarget(
            positions=positions,
            normal=mp.Vector3(0.0, 0.0, 1.0),
            reference_mode_fields={mp.Er: [1.0, np.nan]},
            reference_m=0,
        )

    with pytest.raises(ValueError, match="reference_m must be an integer"):
        tm.EigenmodeCoefficientTarget(
            positions=positions,
            normal=mp.Vector3(0.0, 0.0, 1.0),
            reference_mode_fields={mp.Er: [1.0, 2.0]},
            reference_m=0.5,
        )


def test_eigenmode_coefficient_target_validates_surface_geometry():
    mode_region = mp.Volume(
        center=mp.Vector3(),
        size=mp.Vector3(0.0, 1.0),
    )

    with np.testing.assert_raises_regex(ValueError, "must not be empty"):
        tm.EigenmodeCoefficientTarget(
            [],
            mp.Vector3(1.0),
            mode_region,
            1.0,
        )

    with np.testing.assert_raises_regex(
        ValueError,
        "signed coordinate-axis unit vector",
    ):
        tm.EigenmodeCoefficientTarget(
            [mp.Vector3()],
            mp.Vector3(1.0, 1.0),
            mode_region,
            1.0,
        )

    with np.testing.assert_raises_regex(TypeError, "must be a Meep Volume"):
        tm.EigenmodeCoefficientTarget(
            [mp.Vector3()],
            mp.Vector3(1.0),
            object(),
            1.0,
        )

    with np.testing.assert_raises_regex(
        ValueError,
        "must be normal to normal",
    ):
        tm.EigenmodeCoefficientTarget(
            [mp.Vector3()],
            mp.Vector3(1.0),
            mp.Volume(size=mp.Vector3(0.2, 1.0)),
            1.0,
        )

    with np.testing.assert_raises_regex(
        ValueError,
        "positive tangential extent",
    ):
        tm.EigenmodeCoefficientTarget(
            [mp.Vector3()],
            mp.Vector3(1.0),
            mp.Volume(),
            1.0,
        )

    with np.testing.assert_raises_regex(
        ValueError,
        "positions must lie in mode_region",
    ):
        tm.EigenmodeCoefficientTarget(
            [mp.Vector3(0.0, 0.6)],
            mp.Vector3(1.0),
            mode_region,
            1.0,
        )


def test_eigenmode_coefficient_target_validates_mode_and_solver_settings():
    positions = [mp.Vector3(0.0, -0.5), mp.Vector3(0.0, 0.5)]
    mode_region = mp.Volume(size=mp.Vector3(0.0, 1.0))

    for invalid_frequency in (0.0, -1.0, float("nan")):
        with np.testing.assert_raises_regex(
            ValueError,
            "frequency must be finite and positive",
        ):
            tm.EigenmodeCoefficientTarget(
                positions,
                mp.Vector3(1.0),
                mode_region,
                invalid_frequency,
            )

    for invalid_mode in (True, 0, -1, 1.5):
        with np.testing.assert_raises_regex(
            ValueError,
            "mode must be a positive integer",
        ):
            tm.EigenmodeCoefficientTarget(
                positions,
                mp.Vector3(1.0),
                mode_region,
                1.0,
                mode=invalid_mode,
            )

    for name, value, message in (
        ("eig_parity", -1, "nonnegative integer"),
        ("eig_resolution", -1.0, "finite and nonnegative"),
        ("eig_tolerance", 0.0, "finite and positive"),
    ):
        with np.testing.assert_raises_regex(ValueError, message):
            tm.EigenmodeCoefficientTarget(
                positions,
                mp.Vector3(1.0),
                mode_region,
                1.0,
                **{name: value},
            )

    with np.testing.assert_raises_regex(
        ValueError,
        "spatial_weights must match",
    ):
        tm.EigenmodeCoefficientTarget(
            positions,
            mp.Vector3(1.0),
            mode_region,
            1.0,
            spatial_weights=[1.0],
        )

    with np.testing.assert_raises_regex(
        ValueError,
        "finite, nonnegative",
    ):
        tm.EigenmodeCoefficientTarget(
            positions,
            mp.Vector3(1.0),
            mode_region,
            1.0,
            spatial_weights=[0.0, 0.0],
        )


if __name__ == "__main__":
    test_design_grid_bundles_material_grid_coordinates_and_updates()
    test_cylindrical_design_grid_maps_rz_shape_to_material_grid()
    test_cylindrical_design_grid_requires_degenerate_y_material_grid()
    test_cylindrical_design_grid_rejects_negative_radius()
    test_simulation_spec_uses_default_or_override_sources()
    test_cylindrical_simulation_spec_uses_stored_or_overridden_integer_m()
    test_simulation_spec_rejects_noninteger_or_cartesian_nonzero_m()
    test_simulation_spec_chunk_layout_annotation_is_parameterized()
    test_tda_objective_accepts_bundled_design_simulation_and_target()
    test_field_region_target_normalizes_shape_and_spatial_weights()
    test_field_region_target_validates_positions_shape_and_weights()
    test_flux_target_normalizes_positions_normal_and_spatial_weights()
    test_flux_target_validates_positions_normal_and_spatial_weights()
    test_eigenmode_coefficient_target_normalizes_public_settings()
    test_eigenmode_coefficient_target_validates_surface_geometry()
    test_eigenmode_coefficient_target_validates_mode_and_solver_settings()
