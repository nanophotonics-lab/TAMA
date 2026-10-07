"""Nondispersive adaptation of Meep's analytic anisotropic Bloch-wave test.

Source: https://github.com/NanoComp/meep/blob/
3e7b7fee0da4a9b91b07acd9f71d35d674396563/tests/aniso_disp.cpp
Copyright (C) 2005-2026 Massachusetts Institute of Technology.
SPDX-License-Identifier: GPL-2.0-or-later

Keep the static tensor from the analytic-reference comment, wavevector,
resolution, and 1e-4 frequency tolerance.
Remove the Lorentz susceptibility and compute NEW nondispersive eigenfrequencies.
This exercises forward material setup only; TAMA's Bloch adjoint is unsupported.
"""

import meep as mp
import numpy as np
import pytest

import tama as tm
from tama.backends.meep.sampling_grid import _prepare_native_material


@pytest.mark.parametrize("material", ["medium", "grid_off", "grid_on"])
def test_anisotropic_bloch_frequencies_match_maxwell_eigenproblem(material):
    mp.verbosity(0)
    epsilon = np.array([
        [2.41104, 0.48709, 0.41226],
        [0.48709, 2.43172, 1.62060],
        [0.41226, 1.62060, 3.61498],
    ])
    medium = mp.Medium(
        epsilon_diag=mp.Vector3(*np.diag(epsilon)),
        epsilon_offdiag=mp.Vector3(epsilon[0, 1], epsilon[0, 2], epsilon[1, 2]),
    )
    k = 0.813
    cross = np.array([[0, 0, 0], [0, 0, -k], [0, k, 0]])
    eigenvalues = np.linalg.eigvalsh(-cross @ np.linalg.inv(epsilon) @ cross)
    reference = np.sort(np.sqrt(eigenvalues[eigenvalues > 1e-12]))
    assert reference.shape == (2,)
    grid = mp.MaterialGrid(
        mp.Vector3(2, 2, 2), medium, medium, weights=np.full(8, 0.37),
        do_averaging=material == "grid_on", beta=16,
    )
    design = tm.DesignGrid(
        material_grid=grid, center=mp.Vector3(), size=mp.Vector3(1, 1, 1),
        shape=(2, 2, 2), background=medium, design_material=medium,
    )
    sim = mp.Simulation(
        # A nonzero one-cell z extent prevents Python's automatic 2D reduction.
        cell_size=mp.Vector3(0, 0, 1/200), dimensions=3, resolution=200,
        geometry=[mp.Block(size=design.size, material=grid)] if material != "medium" else [],
        default_material=medium, k_point=mp.Vector3(k, 0, 0),
        eps_averaging=material != "grid_off", force_all_components=True,
        sources=[mp.Source(mp.GaussianSource(0.5, fwidth=1), mp.Ez, mp.Vector3())],
    )
    try:
        if material != "medium":
            _prepare_native_material(sim, design)
        if material == "grid_on":
            assert sim._tama_tensor_structure is sim.structure
        mode = mp.Harminv(mp.Ez, mp.Vector3(), 0.5, 1.0)
        sim.run(mp.after_sources(mode), until_after_sources=200)
        assert sim.dimensions == 3
        observed = np.sort([m.freq for m in mode.modes if m.freq > 0])
        print(f"anisotropic wave {material}: observed={observed.tolist()}, "
              f"reference={reference.tolist()}")
        assert len(observed) == 2
        np.testing.assert_allclose(observed, reference, rtol=1e-4, atol=0)
        inverse = np.array([
            [sim.fields.get_chi1inv(c, d, mp.vec(0, 0, 0)).real
             for d in (mp.X, mp.Y, mp.Z)] for c in (mp.Ex, mp.Ey, mp.Ez)
        ])
        np.testing.assert_allclose(inverse, np.linalg.inv(epsilon), rtol=1e-12, atol=1e-12)
    finally:
        sim.reset_meep()
