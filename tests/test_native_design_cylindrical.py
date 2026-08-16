import meep as mp
import numpy as np
import pytest

import tama as tm
from tama.sampling_grid import NativeDesignField


def _make_cylindrical_design(*, center_r=0.75, size_r=0.8):
    air = mp.Medium(epsilon=1.0)
    dielectric = mp.Medium(epsilon=4.0)
    shape = (3, 4)
    center = mp.Vector3(center_r, 0.0, 0.05)
    size = mp.Vector3(size_r, 0.0, 0.6)
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
    simulation = mp.Simulation(
        cell_size=mp.Vector3(2.0, 0.0, 2.0),
        boundary_layers=[mp.PML(0.25)],
        geometry=[
            mp.Block(
                center=center,
                size=size,
                material=material_grid,
            )
        ],
        dimensions=mp.CYLINDRICAL,
        m=1,
        resolution=10,
        eps_averaging=False,
        force_all_components=True,
        force_complex_fields=True,
    )
    return simulation, design


def test_native_cylindrical_plans_have_rz_signatures():
    mp.verbosity(0)
    simulation, design = _make_cylindrical_design()
    simulation.init_sim()
    fields = []
    try:
        for component in (mp.Er, mp.Ep, mp.Ez):
            field = NativeDesignField(simulation, component, design)
            fields.append(field)

            assert field.local_size > 0
            assert field.signature.shape == (field.local_size, 2)
            assert np.unique(field.signature, axis=0).shape[0] == field.local_size
            assert np.ptp(field.signature[:, 0]) > 0
            assert np.ptp(field.signature[:, 1]) > 0

            accumulator = np.zeros(design.shape, dtype=np.complex128)
            field.accumulate(
                np.zeros(field.local_size, dtype=np.complex128),
                accumulator,
            )
            assert accumulator.shape == design.shape
    finally:
        for field in fields:
            field.release()
        simulation.reset_meep()


def test_native_cylindrical_plan_rejects_cartesian_component():
    mp.verbosity(0)
    simulation, design = _make_cylindrical_design()
    simulation.init_sim()
    try:
        with pytest.raises(ValueError, match="require Er, Ep, or Ez"):
            NativeDesignField(simulation, mp.Ex, design)
    finally:
        simulation.reset_meep()


def test_native_cylindrical_design_may_touch_axis():
    mp.verbosity(0)
    simulation, design = _make_cylindrical_design(
        center_r=0.2,
        size_r=0.4,
    )
    simulation.init_sim()
    field = None
    try:
        field = NativeDesignField(simulation, mp.Ez, design)
        assert field.local_size > 0
    finally:
        if field is not None:
            field.release()
        simulation.reset_meep()


def test_native_design_rejects_coordinate_system_mismatch():
    simulation, design = _make_cylindrical_design()
    design.coordinate_system = "cartesian"

    with pytest.raises(ValueError, match="coordinate system"):
        NativeDesignField(simulation, mp.Ez, design)


def test_native_cylindrical_design_rejects_high_radial_pml_overlap():
    simulation, design = _make_cylindrical_design(
        center_r=1.8,
        size_r=0.4,
    )

    with pytest.raises(ValueError, match="overlapping PML or absorber"):
        NativeDesignField(simulation, mp.Ez, design)
