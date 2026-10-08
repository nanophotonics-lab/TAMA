"""Reduced 3D normal-incidence Gamma-periodic RGB-router optimization.

The example evaluates every one of the 3 wavelength bands at every one of the
3 detector patches. Before optimization, one homogeneous no-device forward
run uses the identical source, time grid, temporal FIRs, and full-cell flux
quadrature to measure the three incident band fluxes. Those values remain
fixed throughout optimization. The scalar objective is

    mean_b(desired_efficiency[b]
           - CROSSTALK_PENALTY
             * sum_{d != b} signed_efficiency[b, d]**2).

Squaring the off-diagonal incident-normalized signed flux penalizes either
forward leakage or local backflow instead of rewarding a negative value.

This is a scale-reduced API example, not a converged color-router design.

Run from the repository root, for example:

    python examples/tda_3d_gamma_rgb_router.py --dry-run
    mpirun -np 8 python examples/tda_3d_gamma_rgb_router.py --iterations 5
"""

import argparse
from pathlib import Path

import meep as mp
import numpy as np

import tama as tm

RESOLUTION = 8
COURANT = 0.125
CELL_SIZE = mp.Vector3(1.75, 1.5, 3.0)
PML_THICKNESS = 0.35
DESIGN_SHAPE = (6, 6, 2)
DESIGN_SIZE = mp.Vector3(CELL_SIZE.x, CELL_SIZE.y, 0.4)
DESIGN_CENTER = mp.Vector3()
SOURCE_Z = 0.75
DETECTOR_Z = -0.65
T_FINAL = 16.0
KERNEL_LENGTH = 101
WAVELENGTH_BANDS = (
    (1.10, 1.35),
    (1.35, 1.65),
    (1.65, 2.00),
)
CHANNEL_NAMES = ("blue", "green", "red")
DETECTOR_CENTERS_X = (-0.55, 0.0, 0.55)
DETECTOR_WIDTH = 0.34
DETECTOR_HEIGHT = 0.60
FILTER_RADIUS = 0.36
PROJECTION_BETA = 4.0
PROJECTION_ETA = 0.5
CROSSTALK_PENALTY = 0.25
MAX_UPDATE = 0.04
RANDOM_SEED = 20260816


def _broadband_source():
    return mp.Source(
        mp.GaussianSource(frequency=0.70, fwidth=0.80),
        component=mp.Ex,
        center=mp.Vector3(0.0, 0.0, SOURCE_Z),
        size=mp.Vector3(CELL_SIZE.x, CELL_SIZE.y, 0.0),
    )


def _flux_patch(center_x, width, height, nx, ny):
    dx = width / nx
    dy = height / ny
    positions = [
        mp.Vector3(
            center_x - 0.5 * width + (ix + 0.5) * dx,
            -0.5 * height + (iy + 0.5) * dy,
            DETECTOR_Z,
        )
        for ix in range(nx)
        for iy in range(ny)
    ]
    return tm.FluxTarget(
        positions=positions,
        normal=mp.Vector3(0.0, 0.0, -1.0),
        spatial_weights=[dx * dy] * len(positions),
    )


def _common_simulation(geometry):
    return tm.SimulationSpec(
        cell_size=CELL_SIZE,
        boundary_layers=[mp.PML(PML_THICKNESS, direction=mp.Z)],
        geometry=geometry,
        sources=[_broadband_source()],
        resolution=RESOLUTION,
        courant=COURANT,
        dimensions=3,
        eps_averaging=False,
        k_point=mp.Vector3(),
    )


def make_reference_objective():
    """Construct the homogeneous incident-flux calibration objective."""
    full_plane = _flux_patch(
        0.0,
        CELL_SIZE.x,
        CELL_SIZE.y,
        nx=4,
        ny=4,
    )
    return tm.MultiTDAObjective(
        update_design=lambda _: None,
        simulation=_common_simulation([]),
        targets=[full_plane] * len(WAVELENGTH_BANDS),
        t_final=T_FINAL,
        dt=COURANT / RESOLUTION,
        wavelength_bands=WAVELENGTH_BANDS,
        weights=np.ones(len(WAVELENGTH_BANDS)),
        kernel_length=KERNEL_LENGTH,
        chunk_balancer=None,
    )


def make_routing_scalarization(incident_band_fluxes):
    """Return the incident-normalized desired/crosstalk scalarization."""
    incident_band_fluxes = np.asarray(incident_band_fluxes, dtype=float)
    if incident_band_fluxes.shape != (len(WAVELENGTH_BANDS),):
        raise ValueError("incident_band_fluxes must have one value per band")
    if not np.all(np.isfinite(incident_band_fluxes)) or np.any(
        incident_band_fluxes <= 0.0
    ):
        raise ValueError("incident_band_fluxes must be finite and positive")

    matrix_shape = (
        len(WAVELENGTH_BANDS),
        len(DETECTOR_CENTERS_X),
    )
    diagonal = np.eye(*matrix_shape, dtype=bool)

    def scalarization(band_detector_fluxes):
        flux_matrix = np.asarray(band_detector_fluxes).reshape(matrix_shape)
        efficiency_matrix = flux_matrix / incident_band_fluxes[:, np.newaxis]
        desired_efficiency = np.diag(efficiency_matrix)
        off_diagonal_efficiency = np.where(
            diagonal,
            0.0,
            efficiency_matrix,
        )
        crosstalk_penalty = np.sum(off_diagonal_efficiency**2, axis=1)
        total = np.mean(desired_efficiency - CROSSTALK_PENALTY * crosstalk_penalty)
        coefficients = (
            -2.0
            * CROSSTALK_PENALTY
            * off_diagonal_efficiency
            / (len(WAVELENGTH_BANDS) * incident_band_fluxes[:, np.newaxis])
        )
        diagonal_indices = np.diag_indices_from(coefficients)
        coefficients[diagonal_indices] = 1.0 / (
            len(WAVELENGTH_BANDS) * incident_band_fluxes
        )
        return (
            float(total),
            coefficients.ravel(),
            {
                "efficiency_matrix": efficiency_matrix.copy(),
                "desired_efficiency": desired_efficiency.copy(),
                "off_diagonal_efficiency": (off_diagonal_efficiency.copy()),
                "crosstalk_penalty": crosstalk_penalty.copy(),
            },
        )

    return scalarization


def make_device_objective(incident_band_fluxes):
    air = mp.Medium(epsilon=1.0)
    substrate = mp.Medium(epsilon=2.1)
    dielectric = mp.Medium(epsilon=6.0)
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
    geometry = [
        mp.Block(
            center=mp.Vector3(0.0, 0.0, -0.75),
            size=mp.Vector3(CELL_SIZE.x, CELL_SIZE.y, 1.1),
            material=substrate,
        ),
        mp.Block(
            center=DESIGN_CENTER,
            size=DESIGN_SIZE,
            material=material_grid,
        ),
    ]
    detector_targets = [
        _flux_patch(
            center_x,
            DETECTOR_WIDTH,
            DETECTOR_HEIGHT,
            nx=2,
            ny=2,
        )
        for center_x in DETECTOR_CENTERS_X
    ]
    # MultiTDA pairs one target with one band, so repeat each wavelength band
    # across all detector patches to evaluate the complete 3 x 3 matrix.
    repeated_bands = [band for band in WAVELENGTH_BANDS for _ in detector_targets]
    repeated_targets = [target for _ in WAVELENGTH_BANDS for target in detector_targets]
    objective = tm.MultiTDAObjective(
        design=design,
        simulation=_common_simulation(geometry),
        targets=repeated_targets,
        t_final=T_FINAL,
        dt=COURANT / RESOLUTION,
        wavelength_bands=repeated_bands,
        weights=np.ones(len(repeated_bands)),
        kernel_length=KERNEL_LENGTH,
        scalarization_fn=make_routing_scalarization(incident_band_fluxes),
        sampling_interval=1,
        chunk_balancer=None,
    )
    return objective, design


def _physical_design(raw_design, design):
    filtered = tm.periodic_conic_filter(
        raw_design.reshape(design.shape),
        radius=FILTER_RADIUS,
        spacing=design.spacing,
        periodic_axes=(0, 1),
    )
    projected = tm.tanh_projection(
        filtered,
        beta=PROJECTION_BETA,
        eta=PROJECTION_ETA,
    )
    # The analytic map stays in [0, 1]; clipping only removes FFT roundoff at
    # an exact bound before MaterialGrid receives the weights.
    projected = np.clip(projected, 0.0, 1.0)
    return filtered, projected


def _raw_design_gradient(filtered, material_gradient, design):
    filtered_gradient = tm.tanh_projection_vjp(
        filtered,
        material_gradient.reshape(design.shape),
        beta=PROJECTION_BETA,
        eta=PROJECTION_ETA,
    )
    return tm.periodic_conic_filter_vjp(
        filtered_gradient,
        radius=FILTER_RADIUS,
        spacing=design.spacing,
        periodic_axes=(0, 1),
    ).ravel()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="number of projected-gradient iterations (default: 5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build and validate both objectives without running FDTD",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional .npz path for the final evaluated density and metrics",
    )
    args = parser.parse_args(argv)
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    mp.verbosity(0)

    reference_objective = make_reference_objective()
    if args.dry_run:
        objective, design = make_device_objective(np.ones(3))
        if mp.am_master():
            print(
                "dry-run: Gamma-periodic 3D RGB router; "
                f"design={design.shape}; reference_bands="
                f"{len(reference_objective.wavelength_bands)}; "
                f"device_band_detector_terms={len(objective.wavelength_bands)}",
                flush=True,
            )
        return

    reference_objective.fom(np.empty(0))
    incident_band_fluxes = np.asarray(
        reference_objective.last_band_objectives,
        dtype=float,
    )
    if np.any(incident_band_fluxes <= 0.0):
        raise RuntimeError(
            "incident calibration returned nonpositive signed flux; "
            "increase t_final or refine the full-plane quadrature"
        )
    objective, design = make_device_objective(incident_band_fluxes)
    rng = np.random.default_rng(RANDOM_SEED)
    raw_design = rng.uniform(0.4, 0.6, np.prod(design.shape))

    if mp.am_master():
        calibration = ", ".join(
            f"{name}={value:.6e}"
            for name, value in zip(CHANNEL_NAMES, incident_band_fluxes)
        )
        print(f"fixed incident band fluxes: {calibration}", flush=True)

    for iteration in range(1, args.iterations + 1):
        filtered, material_weights = _physical_design(raw_design, design)
        value, material_gradient = objective.fom_and_grad(material_weights.ravel())
        raw_gradient = _raw_design_gradient(
            filtered,
            material_gradient,
            design,
        )
        gradient_scale = max(
            float(np.max(np.abs(raw_gradient))),
            np.finfo(float).tiny,
        )
        raw_design = np.clip(
            raw_design + MAX_UPDATE * raw_gradient / gradient_scale,
            0.0,
            1.0,
        )

        if mp.am_master():
            info = objective.last_scalarization_info
            desired = ", ".join(
                f"{name}={efficiency:.4f}"
                for name, efficiency in zip(
                    CHANNEL_NAMES,
                    info["desired_efficiency"],
                )
            )
            crosstalk = ", ".join(
                f"{name}={efficiency:.4f}"
                for name, efficiency in zip(
                    CHANNEL_NAMES,
                    np.sqrt(info["crosstalk_penalty"]),
                )
            )
            print(
                f"iteration={iteration:03d} FoM={value:.8e} "
                f"desired=[{desired}] crosstalk_l2=[{crosstalk}]",
                flush=True,
            )

    if args.output is not None:
        filtered, material_weights = _physical_design(raw_design, design)
        final_value = objective.fom(material_weights.ravel())
        info = objective.last_scalarization_info
        if mp.am_master():
            np.savez(
                args.output,
                raw_design=raw_design.reshape(design.shape),
                filtered_design=filtered,
                material_weights=material_weights,
                incident_band_fluxes=incident_band_fluxes,
                efficiency_matrix=info["efficiency_matrix"],
                desired_efficiency=info["desired_efficiency"],
                off_diagonal_efficiency=info["off_diagonal_efficiency"],
                crosstalk_penalty=info["crosstalk_penalty"],
                fom=final_value,
                wavelength_bands=np.asarray(WAVELENGTH_BANDS),
                detector_centers_x=np.asarray(DETECTOR_CENTERS_X),
                filter_radius=FILTER_RADIUS,
                projection_beta=PROJECTION_BETA,
                projection_eta=PROJECTION_ETA,
            )
            print(f"saved final evaluated design to {args.output}", flush=True)


if __name__ == "__main__":
    main()
