"""Evaluate a cylindrical m=+1 TDA objective and gradient.

Run from the repository root, for example:

    mpirun -np 2 python examples/tda_cylindrical_mode.py
    mpirun -np 2 python examples/tda_cylindrical_mode.py --target regional
    mpirun -np 2 python examples/tda_cylindrical_mode.py --target flux
    mpirun -np 2 python examples/tda_cylindrical_mode.py --target eigenmode
"""

import argparse

import meep as mp
import numpy as np

import tama as tm


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=("point", "regional", "flux", "eigenmode"),
        default="point",
        help="objective target to evaluate (default: point)",
    )
    return parser.parse_args()


def make_target(kind):
    if kind == "point":
        return tm.PointTarget(
            position=mp.Vector3(0.85, 0.0, 0.8),
            component=mp.Er,
            adjoint_source_size=mp.Vector3(),
        )

    dr = 0.2
    radii = np.array([0.7, 0.9, 1.1, 1.3])
    positions = tuple(mp.Vector3(radius, 0.0, 0.8) for radius in radii)
    surface_weights = tuple(2.0 * np.pi * radii * dr)
    if kind == "regional":
        return tm.FieldRegionTarget(
            positions=positions,
            component=mp.Er,
            sample_shape=(len(positions),),
            spatial_weights=surface_weights,
        )
    if kind == "flux":
        return tm.FluxTarget(
            positions=positions,
            normal=mp.Vector3(0.0, 0.0, 1.0),
            spatial_weights=surface_weights,
        )

    # Analytic +z vacuum plane wave in Meep's m=+1 convention. In unit
    # impedance, Ep=i*Er, Hr=-i*Er, and Hp=Er. The samples form a fixed
    # reference profile on the axial quadrature surface.
    er = np.ones(len(positions), dtype=np.complex128)
    return tm.EigenmodeCoefficientTarget(
        positions=positions,
        normal=mp.Vector3(0.0, 0.0, 1.0),
        spatial_weights=surface_weights,
        reference_mode_fields={
            mp.Er: er,
            mp.Hp: er,
            mp.Ep: 1j * er,
            mp.Hr: -1j * er,
        },
        reference_m=1,
    )


def main():
    args = parse_args()
    mp.verbosity(0)

    air = mp.Medium(epsilon=1.0)
    dielectric = mp.Medium(epsilon=2.25)
    design_shape = (3, 3)
    design_center = mp.Vector3(0.3, 0.0, 0.0)
    design_size = mp.Vector3(0.6, 0.0, 0.6)

    material_grid = mp.MaterialGrid(
        mp.Vector3(design_shape[0], 1, design_shape[1]),
        air,
        dielectric,
    )
    design = tm.DesignGrid(
        material_grid=material_grid,
        center=design_center,
        size=design_size,
        shape=design_shape,
        background=air,
        design_material=dielectric,
        coordinate_system="cylindrical",
    )
    simulation = tm.SimulationSpec(
        cell_size=mp.Vector3(2.0, 0.0, 3.0),
        boundary_layers=[
            mp.PML(0.4, direction=mp.R),
            mp.PML(0.4, direction=mp.Z),
        ],
        geometry=[
            mp.Block(
                center=design_center,
                size=design_size,
                material=material_grid,
            )
        ],
        sources=[
            mp.Source(
                mp.GaussianSource(frequency=0.8, fwidth=0.6, cutoff=3),
                component=mp.Er,
                center=mp.Vector3(0.85, 0.0, -0.8),
            )
        ],
        resolution=10,
        dimensions=mp.CYLINDRICAL,
        m=1,
        eps_averaging=False,
    )
    target = make_target(args.target)
    objective = tm.TDAObjective(
        design=design,
        simulation=simulation,
        target=target,
        t_final=12.0,
    )

    fom, gradient = objective.fom_and_grad(np.full(np.prod(design_shape), 0.5))
    if mp.am_master():
        print(f"target={args.target}")
        print(f"FoM={fom:.8e}")
        print(f"|gradient|_2={np.linalg.norm(gradient):.8e}")


if __name__ == "__main__":
    main()
