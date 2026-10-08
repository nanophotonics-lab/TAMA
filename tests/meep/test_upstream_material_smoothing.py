"""Forward-only adaptations of Meep v1.34.0 MaterialGrid regressions.

Source: https://github.com/NanoComp/meep/blob/
3e7b7fee0da4a9b91b07acd9f71d35d674396563/python/tests/test_material_grid.py
Source SHA256: f587dc3820db82c0b809af07f77ec1e52a6cb505a9bdeef16262676e03ff14c7
Adapted from Meep, Copyright (C) Massachusetts Institute of Technology,
GPL-2.0-or-later. These low-level forward tests retain upstream Bloch vectors;
they do not imply Bloch support in TAMA's high-level adjoint interface.
"""

import json
import time
from types import SimpleNamespace

import meep as mp
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from tama.backends.meep.sampling_grid import _prepare_native_material


def _prepare(sim, grid, mode):
    if mode == "stock":
        return
    _prepare_native_material(sim, SimpleNamespace(material_grid=grid))
    assert sim.subpixel_tol <= 1e-8
    if mode == "near_isotropic":
        # Nonzero xy enters the custom tensor branch, unlike exact isotropy.
        assert grid.medium2.epsilon_offdiag.x == 1e-6
        assert sim._tama_tensor_structure is sim.structure
        assert sim.fields is None


def _mode_frequency(sim, component, position, fcen, df, label):
    start = time.perf_counter()
    harminv = mp.Harminv(component, position, fcen, df)
    try:
        sim.run(mp.after_sources(harminv), until_after_sources=200)
        assert harminv.modes, "No resonant modes found in the upstream frequency window"
        frequency = harminv.modes[0].freq
        assert np.isfinite(frequency)
        print(
            "upstream_smoothing: "
            + json.dumps(
                {
                    "case": label,
                    "resolution": sim.resolution,
                    "frequency": frequency,
                    "elapsed_seconds": time.perf_counter() - start,
                }
            ),
            flush=True,
        )
        return frequency
    finally:
        sim.reset_meep()


@pytest.fixture(scope="module")
def circle_weights():
    coordinates = np.linspace(-0.5, 0.5, 1200)
    x, y = np.meshgrid(coordinates, coordinates)
    return gaussian_filter(
        np.sqrt(x * x + y * y) < 0.301943, sigma=3.0, output=np.double
    )


def _circle_frequency(resolution, weights, mode):
    medium = mp.Medium(index=3.5)
    if mode == "near_isotropic":
        medium.epsilon_offdiag = mp.Vector3(1e-6, 0, 0)
    grid = mp.MaterialGrid(
        mp.Vector3(1200, 1200),
        mp.air,
        medium,
        weights=weights,
        do_averaging=True,
        beta=1000,
        eta=0.5,
    )
    fcen, df = 0.3, 0.06
    sim = mp.Simulation(
        resolution=resolution,
        cell_size=mp.Vector3(1, 1),
        geometry=[mp.Block(size=mp.Vector3(1, 1), material=grid)],
        sources=[
            mp.Source(
                mp.GaussianSource(fcen, fwidth=df),
                component=mp.Hz,
                center=mp.Vector3(-0.1057, 0.2094),
            )
        ],
        k_point=mp.Vector3(0.3892, 0.1597),
    )
    _prepare(sim, grid, mode)
    return _mode_frequency(
        sim,
        mp.Hz,
        mp.Vector3(0.3718, -0.2076),
        fcen,
        df,
        "circle_" + mode,
    )


@pytest.mark.parametrize("mode", ["stock", "prepared", "near_isotropic"])
def test_upstream_circle_subpixel_convergence(circle_weights, mode):
    mp.verbosity(0)
    # Upstream's reference is a resolution-300 numerical result, not an exact solution.
    reference = 0.29826813873225283
    frequencies = [_circle_frequency(r, circle_weights, mode) for r in (25, 50)]
    for frequency in frequencies:
        assert round(abs(frequency - reference), 2) == 0
    # Preserve the upstream better-than-linear criterion without claiming order two.
    assert 2 * abs(frequencies[1] - reference) < abs(frequencies[0] - reference)


def _sphere_frequency(mode):
    resolution, radius = 25, 0.34
    silicon, silica = mp.Medium(index=3.45), mp.Medium(index=1.45)
    if mode == "near_isotropic":
        silicon.epsilon_offdiag = mp.Vector3(1e-6, 0, 0)
    if mode == "geometry":
        grid = None
        geometry = [mp.Sphere(radius=radius, material=silicon)]
    else:
        coordinates = np.linspace(-0.5, 0.5, 2 * resolution)
        x, y, z = np.meshgrid(coordinates, coordinates, coordinates)
        weights = gaussian_filter(
            np.sqrt(x * x + y * y + z * z) < radius,
            sigma=4 / resolution,
            output=np.double,
        )
        grid = mp.MaterialGrid(
            mp.Vector3(50, 50, 50),
            silica,
            silicon,
            weights=weights,
            do_averaging=True,
            beta=1000,
            eta=0.5,
        )
        geometry = [mp.Block(size=mp.Vector3(1, 1, 1), material=grid)]
    fcen = 1 / 1.27
    df = 0.02 * fcen
    sim = mp.Simulation(
        resolution=resolution,
        cell_size=mp.Vector3(1, 1, 1),
        geometry=geometry,
        default_material=silica,
        sources=[
            mp.Source(
                mp.GaussianSource(fcen, fwidth=df),
                component=mp.Ez,
                center=mp.Vector3(0.13, 0.25, 0.06),
            )
        ],
        k_point=mp.Vector3(0.23, -0.17, 0.35),
    )
    if grid is not None:
        _prepare(sim, grid, mode)
    return _mode_frequency(
        sim,
        mp.Ez,
        mp.Vector3(-0.2684, 0.1185, 0.0187),
        fcen,
        df,
        "sphere_" + mode,
    )


@pytest.fixture(scope="module")
def sphere_frequencies():
    mp.verbosity(0)
    return {
        mode: _sphere_frequency(mode)
        for mode in ("geometry", "stock", "prepared", "near_isotropic")
    }


@pytest.mark.parametrize("mode", ["stock", "prepared", "near_isotropic"])
def test_upstream_material_grid_sphere_matches_geometry(sphere_frequencies, mode):
    # As upstream: agreement to two decimal places at one spatial resolution.
    assert round(abs(sphere_frequencies[mode] - sphere_frequencies["geometry"]), 2) == 0


def test_sphere_tensor_averaging_has_continuous_isotropic_limit(sphere_frequencies):
    # A 1e-6 xy perturbation must not switch to a differently normalized kernel.
    assert (
        abs(sphere_frequencies["near_isotropic"] - sphere_frequencies["prepared"])
        < 1e-7
    )
