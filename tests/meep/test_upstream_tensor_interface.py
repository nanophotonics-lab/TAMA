"""Compare the custom operator with Meep's geometric Kottke interface average.

The independent oracle is geom_epsilon::eff_chi1inv_matrix in Meep v1.34.0:
https://github.com/NanoComp/meep/blob/v1.34.0/src/meepgeom.cpp
This is a derived regression.
"""

import meep as mp
import numpy as np
import pytest

import tama as tm
from tama.backends.meep.sampling_grid import _prepare_native_material


def _plane_coefficients(mode, axis):
    resolution = 8
    normal = np.array((0.36, 0.48, 0.8))
    tangent = np.cross(normal, (0, 0, 1))
    tangent /= np.linalg.norm(tangent)
    third = np.cross(normal, tangent)
    offset = 0.0 if axis is None else normal[axis] / (2 * resolution)
    first = mp.Medium(
        epsilon_diag=mp.Vector3(
            1.404643166919161, 1.6796316050221778, 2.015725228058661
        ),
        epsilon_offdiag=mp.Vector3(
            -0.10853364362110643, -0.1749479662432673, -0.15474877160277223
        ),
    )
    second = mp.Medium(
        epsilon_diag=mp.Vector3(
            3.5465172935340075, 4.241277040167499, 4.512205666298495
        ),
        epsilon_offdiag=mp.Vector3(
            0.31489724341775416, 0.8659358464962967, 0.11352059706786331
        ),
    )
    # These SPD tensors have eigenvalues (1.3, 1.7, 2.1) and (3, 4.2, 5.1),
    # with different principal axes and all three off-diagonal entries nonzero.
    if mode == "geometry":
        geometry = [
            mp.Block(
                size=mp.Vector3(4, mp.inf, mp.inf),
                center=mp.Vector3(*((2 + offset) * normal)),
                e1=mp.Vector3(*normal),
                e2=mp.Vector3(*tangent),
                e3=mp.Vector3(*third),
                material=second,
            )
        ]
    else:
        # MaterialGrid samples are cell-centered. This affine density projects
        # to precisely the same half-space n.r >= offset as the rotated Block.
        coords = (np.arange(5) + 0.5) / 5 - 0.5
        xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
        weights = 0.5 + 0.5 * (
            normal[0] * xx + normal[1] * yy + normal[2] * zz - offset
        )
        grid = mp.MaterialGrid(
            mp.Vector3(5, 5, 5),
            first,
            second,
            weights=weights,
            beta=np.inf,
            eta=0.5,
            do_averaging=True,
        )
        design = tm.DesignGrid(
            material_grid=grid,
            center=mp.Vector3(),
            size=mp.Vector3(1, 1, 1),
            shape=(5, 5, 5),
            background=first,
            design_material=second,
        )
        geometry = [mp.Block(size=design.size, material=grid)]
    sim = mp.Simulation(
        cell_size=mp.Vector3(1.5, 1.5, 1.5),
        geometry=geometry,
        default_material=first,
        resolution=resolution,
        force_all_components=True,
        eps_averaging=True,
        subpixel_tol=1e-10,
    )
    try:
        if mode == "tama":
            _prepare_native_material(sim, design)
        sim.init_sim()
        values = []
        for i, component in enumerate((mp.Ex, mp.Ey, mp.Ez)):
            point = np.zeros(3)
            point[i] = 0.5 / resolution
            for j, direction in enumerate((mp.X, mp.Y, mp.Z)):
                if (axis is None and i != j) or (axis == i and i == j):
                    values.append(
                        sim.fields.get_chi1inv(component, direction, mp.vec(*point))
                    )
        return np.asarray(values)
    finally:
        sim.reset_meep()


@pytest.mark.parametrize("axis", [None, 0, 1, 2], ids=["offdiagonal", "xx", "yy", "zz"])
def test_oblique_tensor_interface_matches_geometric_meep(axis):
    mp.verbosity(0)
    # Meep stores diagonal rows at electric Yee points and off-diagonal rows
    # half a cell behind along the component axis. Center the interface on the
    # corresponding voxel: both geometric box and MaterialGrid sphere fill=1/2.
    # The box and sphere kernels give different off-center fill fractions.
    expected = _plane_coefficients("geometry", axis)
    actual = _plane_coefficients("tama", axis)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    if axis is None:
        assert np.min(np.abs(expected)) > 1e-3
        stock_grid = _plane_coefficients("stock_grid", axis)
        # Stock Meep ignores anisotropic MaterialGrid averaging. This catches
        # an accidentally bypassed custom operator even if its own FD passes.
        assert np.max(np.abs(stock_grid - expected)) > 1e-2
