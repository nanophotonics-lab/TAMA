"""Small 3D TAMA optimization.

Run from the repository root, for example:

    mpirun -np 8 python examples/tda_3d_projected_gradient.py

The script intentionally contains no timing or memory instrumentation so that
the process can be monitored externally without adding measurement collectives.
"""

import argparse

import meep as mp
import numpy as np

import tama as tm


RESOLUTION = 20
DESIGN_SHAPE = (8, 8, 8)
DESIGN_CENTER = mp.Vector3(0.04, -0.03, 0.02)
DESIGN_SIZE = mp.Vector3(1.0, 1.0, 1.0)
CELL_SIZE = mp.Vector3(4.0, 3.5, 3.0)
PML_THICKNESS = 0.5
SOURCE_POSITION = mp.Vector3(-1.2, -0.25, 0.15)
TARGET_POSITION = mp.Vector3(1.2, 0.25, -0.20)
T_FINAL = 100.0
MAX_UPDATE = 0.05
RANDOM_SEED = 17


def make_objective():
    air = mp.Medium(epsilon=1.0)
    dielectric = mp.Medium(epsilon=4.0)
    material_grid = mp.MaterialGrid(
        mp.Vector3(*DESIGN_SHAPE),
        air,
        dielectric,
        do_averaging=False,
    )
    design = tm.DesignGrid(
        material_grid=material_grid,
        center=DESIGN_CENTER,
        size=DESIGN_SIZE,
        shape=DESIGN_SHAPE,
        background=air,
        design_material=dielectric,
    )
    simulation = tm.SimulationSpec(
        cell_size=CELL_SIZE,
        boundary_layers=[mp.PML(PML_THICKNESS)],
        geometry=[
            mp.Block(
                center=DESIGN_CENTER,
                size=DESIGN_SIZE,
                material=material_grid,
            )
        ],
        sources=[
            mp.Source(
                mp.GaussianSource(frequency=0.7, fwidth=0.5),
                component=mp.Ex,
                center=SOURCE_POSITION,
            )
        ],
        resolution=RESOLUTION,
        dimensions=3,
        eps_averaging=False,
    )
    target = tm.PointTarget(
        position=TARGET_POSITION,
        component=mp.Ex,
    )
    return tm.TDAObjective(
        design=design,
        simulation=simulation,
        target=target,
        t_final=T_FINAL,
        reuse_simulation=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="number of projected-gradient iterations (default: 20)",
    )
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    return args


def main():
    args = parse_args()
    mp.verbosity(0)

    objective = make_objective()
    rng = np.random.default_rng(RANDOM_SEED)
    design = rng.uniform(0.4, 0.6, np.prod(DESIGN_SHAPE))

    if mp.am_master():
        print(
            f"3D TDA optimization: ranks={mp.count_processors()}, "
            f"design={DESIGN_SHAPE}, resolution={RESOLUTION}",
            flush=True,
        )

    for iteration in range(1, args.iterations + 1):
        fom, gradient = objective.fom_and_grad(design)
        gradient_scale = max(
            float(np.max(np.abs(gradient))),
            np.finfo(float).tiny,
        )
        design = np.clip(
            design + MAX_UPDATE * gradient / gradient_scale,
            0.0,
            1.0,
        )

        if mp.am_master():
            print(
                f"iteration={iteration:03d} "
                f"FoM={fom:.8e} "
                f"|gradient|_2={np.linalg.norm(gradient):.3e}",
                flush=True,
            )
            report = objective.last_chunk_balance
            if report is not None:
                rank_work = ", ".join(
                    f"{seconds:.3f}" for seconds in report.rank_work_seconds
                )
                print(
                    "    Adaptive chunks: "
                    f"topology={report.topology}->{report.next_topology}, "
                    f"critical={report.critical_path_seconds:.3f}s, "
                    f"max/mean={report.max_to_mean:.3f}, "
                    f"moved={report.moved_boundaries}, "
                    f"applied={report.applied}, "
                    f"exploring={report.exploring}, "
                    f"reason={report.reason}",
                    flush=True,
                )
                print(f"    Rank work [s]: {rank_work}", flush=True)


if __name__ == "__main__":
    main()
