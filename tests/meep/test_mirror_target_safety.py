"""Mirror gradients require the validated indexed monitor transpose."""

import meep as mp
import numpy as np
import pytest

import tama as tm


@pytest.mark.parametrize("objective_type", [tm.TDAObjective, tm.MultiTDAObjective])
@pytest.mark.parametrize("mode", ["factory", "finite_source"])
def test_mirror_rejects_ordinary_adjoint_sources_but_allows_value(objective_type, mode):
    mp.verbosity(0)
    low, high = mp.Medium(epsilon=1), mp.Medium(epsilon=3)
    grid = mp.MaterialGrid(mp.Vector3(3, 3), low, high)
    design = tm.DesignGrid(grid, mp.Vector3(), mp.Vector3(0.4, 0.4), (3, 3), low, high)
    simulation = tm.SimulationSpec(
        cell_size=mp.Vector3(2.4, 2.4),
        resolution=10,
        courant=0.25,
        boundary_layers=[mp.PML(0.3)],
        eps_averaging=False,
        geometry=[mp.Block(size=design.size, material=grid)],
        sources=[
            mp.Source(
                mp.GaussianSource(0.7, fwidth=0.4),
                component=mp.Ez,
                center=mp.Vector3(-0.7, 0),
            )
        ],
        symmetries=[mp.Mirror(mp.Y)],
    )
    target = tm.PointTarget(
        mp.Vector3(0.7, -0.2),
        mp.Ez,
        adjoint_source_size=mp.Vector3(0.08, 0.08) if mode == "finite_source" else None,
    )
    kwargs = dict(design=design, t_final=20, dt=0.025, chunk_balancer=None)
    if mode == "factory":
        kwargs["sim_factory"] = lambda sources=None: simulation.make(sources)
    else:
        kwargs["simulation"] = simulation
    if objective_type is tm.MultiTDAObjective:
        kwargs.update(
            targets=[target],
            weights=[1.0],
            wavelength_bands=[(1.2, 1.7)],
            kernel_length=31,
        )
    else:
        kwargs["target"] = target
    objective = objective_type(**kwargs)
    weights = np.full(9, 0.4)
    assert np.isfinite(objective.fom(weights))
    with pytest.raises(
        ValueError, match="Mirror gradients require direct SimulationSpec"
    ):
        objective.fom_and_grad(weights)
