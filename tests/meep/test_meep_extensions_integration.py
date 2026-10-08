"""Far-field gradients with independent regions and actual Mirror reduction."""

import meep as mp
import numpy as np
import pytest

import tama as tm


def _problem(mirror):
    low, high = mp.Medium(epsilon=1), mp.Medium(epsilon=3)
    designs, geometry = [], []
    for x in (-0.3, 0.3):
        grid = mp.MaterialGrid(mp.Vector3(3, 3), low, high)
        design = tm.DesignGrid(
            grid,
            mp.Vector3(x, 0),
            mp.Vector3(0.3, 0.5),
            (3, 3),
            low,
            high,
        )
        designs.append(design)
        geometry.append(mp.Block(center=design.center, size=design.size, material=grid))
    regions = [
        mp.Near2FarRegion(center=mp.Vector3(x, 0), size=mp.Vector3(0, 2), weight=x)
        for x in (-1, 1)
    ] + [
        mp.Near2FarRegion(center=mp.Vector3(0, y), size=mp.Vector3(2, 0), weight=y)
        for y in (-1, 1)
    ]
    return tm.TDAObjective(
        designs=designs,
        simulation=tm.SimulationSpec(
            cell_size=mp.Vector3(3, 3),
            resolution=10,
            courant=0.25,
            boundary_layers=[mp.PML(0.3)],
            geometry=geometry,
            sources=[
                mp.Source(
                    mp.GaussianSource(0.7, fwidth=0.4),
                    component=mp.Ez,
                    center=mp.Vector3(-0.75, 0),
                )
            ],
            eps_averaging=False,
            symmetries=[mp.Mirror(mp.Y)] if mirror else [],
        ),
        target=tm.Near2FarTarget(
            regions, [0.6, 0.8], [mp.Vector3(8, 2), mp.Vector3(8, -2)]
        ),
        t_final=35,
        chunk_balancer=None,
    )


@pytest.mark.mpi2
def test_near2far_multiple_regions_and_mirror_match_full_domain_and_fd():
    mp.verbosity(0)
    weights = np.linspace(0.3, 0.6, 18).reshape(2, 3, 3)
    weights = 0.5 * (weights + weights[:, :, ::-1])
    direction = np.random.default_rng(274).normal(size=weights.shape)
    direction = 0.5 * (direction + direction[:, :, ::-1])
    direction /= np.linalg.norm(direction)
    weights, direction = weights.ravel(), direction.ravel()
    full, reduced = _problem(False), _problem(True)
    full_value, full_gradient = full.fom_and_grad(weights)
    value, gradient = reduced.fom_and_grad(weights)
    assert gradient.shape == (18,)
    assert np.all(np.isfinite(gradient))
    np.testing.assert_allclose(value, full_value, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(
        gradient @ direction, full_gradient @ direction, rtol=0.002, atol=1e-10
    )
    differences = []
    for h in (1e-4, 5e-5):
        differences.append(
            (
                reduced.fom(weights + h * direction)
                - reduced.fom(weights - h * direction)
            )
            / (2 * h)
        )
    np.testing.assert_allclose(differences[0], differences[1], rtol=1e-4, atol=1e-10)
    np.testing.assert_allclose(gradient @ direction, differences, rtol=0.01, atol=1e-10)
    print("Combined near2far/regions/Mirror:", value, gradient @ direction, differences)
