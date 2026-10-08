# SPDX-License-Identifier: GPL-2.0-or-later
# Meep fixture copyright (C) 2005-2026 Massachusetts Institute of Technology.
"""TAMA adaptation of Meep 1.34's off-diagonal adjoint regression.

Upstream: NanoComp/meep@3e7b7fee0da4a9b91b07acd9f71d35d674396563,
python/tests/test_adjoint_solver.py::TestAdjointSolver.test_offdiagonal.
The dielectric tensor, geometry, resolution, design mesh, random seed,
conic filter, perturbation, and 0.002 single-frequency tolerance are retained.
The objective is TDA point-field energy instead of frequency-domain mode
coefficients; source bandwidth is 0.2*fcen instead of 0.05*fcen, with a fixed
120-unit recording and stride 2. Central differences at dp and dp/2 also
check step stability. The averaging-on case adds beta=8 and do_averaging=True;
it is an extension of the upstream fixture, not an unchanged upstream test.
"""

import json

import meep as mp
import meep.adjoint as mpa
import numpy as np
import pytest
from autograd import tensor_jacobian_product

import tama as tm
from tama_meep_native import native_sampler


def _problem(averaging):
    sapphire = mp.Medium(
        epsilon_diag=(10.225, 10.225, 9.95),
        epsilon_offdiag=(-0.825, -0.55 * np.sqrt(1.5), 0.55 * np.sqrt(1.5)),
    )
    grid = mp.MaterialGrid(
        mp.Vector3(91, 91),
        mp.air,
        sapphire,
        weights=np.ones((91, 91)),
        do_averaging=averaging,
        beta=8 if averaging else 0,
    )
    design = tm.DesignGrid(
        material_grid=grid,
        center=mp.Vector3(),
        size=mp.Vector3(1.5, 1.5),
        shape=(91, 91),
        background=mp.air,
        design_material=sapphire,
    )
    fcen = 1 / 1.55
    simulation = tm.SimulationSpec(
        cell_size=mp.Vector3(5, 5),
        boundary_layers=[mp.PML(1)],
        geometry=[
            mp.Block(
                size=mp.Vector3(mp.inf, 1, mp.inf), material=mp.Medium(epsilon=12)
            ),
            mp.Block(center=design.center, size=design.size, material=grid),
        ],
        sources=[
            mp.EigenModeSource(
                src=mp.GaussianSource(fcen, fwidth=0.2 * fcen),
                center=mp.Vector3(-1.5, 0),
                size=mp.Vector3(0, 3),
                eig_parity=mp.EVEN_Y + mp.ODD_Z,
            )
        ],
        resolution=30,
        courant=0.5,
        eps_averaging=True,
    )
    return tm.TDAObjective(
        design=design,
        simulation=simulation,
        target=tm.PointTarget(position=mp.Vector3(1.5, 0), component=mp.Ez),
        t_final=120,
        dt=0.5 / 30,
        sampling_interval=2,
        chunk_balancer=None,
    )


def _filter(density):
    return mpa.conic_filter(density.reshape((91, 91)), 0.25, 1.5, 1.5, 60).flatten()


@pytest.mark.parametrize("averaging", [False, True], ids=["off", "on"])
def test_upstream_offdiagonal_fixture_tama_gradient(averaging, monkeypatch):
    """Catch tensor contraction/normal derivatives against actual forward FD."""
    mp.verbosity(0)
    configured = []
    original = native_sampler.configure_native_material_operator

    def configure(*args):
        result = original(*args)
        configured.append(args[2:])
        return result

    monkeypatch.setattr(native_sampler, "configure_native_material_operator", configure)
    objective = _problem(averaging)
    rng = np.random.RandomState(9861548)
    density = 0.5 * rng.rand(91 * 91)
    perturbation = 1e-5 * rng.rand(91 * 91)
    value, gradient = objective.fom_and_grad(_filter(density))
    raw_gradient = tensor_jacobian_product(_filter, 0)(density, gradient)
    adjoint = float(perturbation @ raw_gradient)
    finite_differences = [
        (
            objective.fom(_filter(density + scale * perturbation))
            - objective.fom(_filter(density - scale * perturbation))
        )
        / (2 * scale)
        for scale in (1.0, 0.5)
    ]
    denominator = max(abs(adjoint), abs(finite_differences[-1]))
    error = abs(adjoint - finite_differences[-1]) / denominator
    if mp.am_master():
        print(
            json.dumps(
                dict(
                    fixture="meep-1.34-offdiagonal",
                    averaging=averaging,
                    value=float(value),
                    adjoint=adjoint,
                    central_fd=finite_differences,
                    relative_error=error,
                    material_operator_installations=len(configured),
                ),
                sort_keys=True,
            )
        )
    assert np.isfinite(value) and np.all(np.isfinite(gradient))
    assert denominator > 1e-10
    if averaging:
        assert configured, "test must exercise TAMA's tensor averaging operator"
        assert all(
            enabled and tolerance <= 1e-8 for enabled, tolerance, _ in configured
        )
    else:
        assert not configured
    np.testing.assert_allclose(
        finite_differences[0], finite_differences[1], rtol=0.001, atol=1e-12
    )
    assert error <= 0.002
