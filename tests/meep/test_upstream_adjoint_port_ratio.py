# SPDX-License-Identifier: GPL-2.0-or-later
# Meep fixture copyright (C) 2005-2026 Massachusetts Institute of Technology.
"""Port the six physical gradient cases in Meep's JAX-wrapper regression.

Source: NanoComp/meep@3e7b7fee0da4a9b91b07acd9f71d35d674396563,
python/tests/test_adjoint_jax.py::WrapperTest::test_wrapper_gradients.
The geometry, materials, resolution 20, design shape (20, 10), six frequency,
source and port conditions, initial density 0.5, five perturbations of scale
1e-4, mean modal-power ratio, and original infinity-norm tolerance 0.025 are
retained (0.1 in single precision). TAMA MultiTDAObjective and scalarization
replace MeepJaxWrapper;
both the numerator and the design-dependent denominator are differentiated.

TAMA records 500 time units instead of using upstream adaptive DFT decay,
with sampling_interval=1 and explicit 51-point trapezoidal quadrature over
each original 2.5-unit port. Each frequency has its own fixed reference mode. A
normalized one-tap temporal filter is the identity. These modal Fourier
functionals are not bitwise replacements for Meep's DFT mode monitors. This
does not test JAX tracing, custom VJP plumbing, or distributed DFT pointers.

The data fixture contains the exact upstream expression
1e-4*jax.random.normal(jax.random.PRNGKey(seed),(20,10)), seeds 0 through 4,
generated with the already installed JAX 0.11.2 on CPU, jax_enable_x64=True,
jax_default_prng_impl='threefry2x32', jax_threefry_partitionable=True.
Upstream does not pin JAX's version or partitionable setting; these settings
make this port reproducible without a runtime JAX dependency. Fixture SHA256:
33aedfecd8e84a778aa893a6ec26e774e6db2dce6b5770e0447d15b458d8a22e.
"""
import hashlib
import json
from pathlib import Path

import meep as mp
import numpy as np
import pytest
from autograd import numpy as npa

import tama as tm


CASES = [
    ("1500_1550bw_01relative_gaussian_port1", (1 / 1.50, 1 / 1.55), 3, 0.1, 0),
    ("1550_1600bw_02relative_gaussian_port1", (1 / 1.55, 1 / 1.60), 3, 0.2, 0),
    ("1500_1600bw_03relative_gaussian_port1", (1 / 1.50, 1 / 1.60), 4, 0.3, 0),
    ("1500_1550bw_01relative_gaussian_port2", (1 / 1.50, 1 / 1.55), 3, 0.1, 1),
    ("1550_1600bw_02relative_gaussian_port2", (1 / 1.55, 1 / 1.60), 3, 0.2, 1),
    ("1500_1600bw_03relative_gaussian_port2", (1 / 1.50, 1 / 1.60), 4, 0.3, 1),
]


def _modal_power(frequency):
    def objective(history, dt):
        phase = npa.exp(2j * npa.pi * frequency * npa.arange(history.shape[0]) * dt)
        coefficient = npa.sum(history * phase) * dt / npa.sqrt(2 * npa.pi)
        return npa.abs(coefficient) ** 2
    return objective


def _problem(frequencies, width, excite_port):
    silicon, silica = mp.Medium(index=3.4), mp.Medium(index=1.44)
    grid = mp.MaterialGrid(mp.Vector3(20, 10), silica, silicon, grid_type="U_DEFAULT")
    geometry = [
        mp.Block(center=mp.Vector3(-1.5), size=mp.Vector3(2, 0.5, 0), material=silicon),
        mp.Block(center=mp.Vector3(1.5), size=mp.Vector3(2, 0.5, 0), material=silicon),
        mp.Block(center=mp.Vector3(), size=mp.Vector3(1, 0.5, 0), material=grid),
    ]
    source = mp.EigenModeSource(
        mp.GaussianSource(frequency=float(np.mean(frequencies)),
                          fwidth=float(np.mean(frequencies)) * width),
        eig_band=1, direction=mp.NO_DIRECTION,
        eig_kpoint=mp.Vector3(1 if excite_port == 0 else -1),
        size=mp.Vector3(0, 2.5, 0),
        center=mp.Vector3(-1 if excite_port == 0 else 1),
    )
    simulation = tm.SimulationSpec(
        cell_size=mp.Vector3(5, 4.5), boundary_layers=[mp.PML(1)],
        geometry=geometry, sources=[source], resolution=20,
    )
    design = tm.DesignGrid(grid, mp.Vector3(), mp.Vector3(1, 0.5, 0),
                           (20, 10), silica, silicon)
    targets, callbacks = [], []
    weights = np.full(51, 1 / 20)
    weights[[0, -1]] *= 0.5
    # Upstream order is s1p,s1m,s2p,s2m. Preserve even the unused channels.
    for center in (-0.9, 0.9):
        for forward in (True, False):
            for frequency in frequencies:
                targets.append(tm.EigenmodeCoefficientTarget(
                    positions=tuple(mp.Vector3(center, y) for y in np.linspace(-1.25, 1.25, 51)),
                    normal=mp.Vector3(1 if forward else -1), mode=1,
                    mode_region=mp.Volume(center=mp.Vector3(center), size=mp.Vector3(0, 2.5)),
                    frequency=float(frequency), eig_parity=mp.NO_PARITY,
                    spatial_weights=tuple(weights),
                ))
                callbacks.append(_modal_power(float(frequency)))
    nfreq = len(frequencies)
    numerator, denominator = (2, 0) if excite_port == 0 else (1, 3)

    def ratio(values):
        powers = npa.reshape(values, (4, nfreq))
        return npa.mean(powers[numerator] / powers[denominator])

    return tm.MultiTDAObjective(
        design=design, simulation=simulation, targets=targets,
        t_final=500, sampling_interval=1, chunk_balancer=None,
        wavelength_bands=[(1.5, 1.6)] * len(targets), weights=[1.0] * len(targets),
        kernel_length=1, fom_fn=callbacks, scalarization_fn=ratio,
    )


@pytest.mark.parametrize("name,bounds,nfreq,width,excite_port", CASES,
                         ids=[case[0] for case in CASES])
def test_upstream_port_ratio_gradient(name, bounds, nfreq, width, excite_port):
    mp.verbosity(0)
    fixture = Path(__file__).with_name("data") / "upstream_adjoint_port_ratio_directions.npy"
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == (
        "33aedfecd8e84a778aa893a6ec26e774e6db2dce6b5770e0447d15b458d8a22e"
    )
    perturbations = np.load(fixture, allow_pickle=False)
    assert perturbations.shape == (5, 20, 10)
    frequencies = np.linspace(*bounds, nfreq)
    problem = _problem(frequencies, width, excite_port)
    x = np.full((20, 10), 0.5)
    value, gradient = problem.evaluate(x.ravel(), need_gradient=True)
    projections, differences = [], []
    for perturbation in perturbations:
        shifted = problem.fom((x + perturbation).ravel())
        projections.append(float(perturbation.ravel() @ gradient.ravel()))
        differences.append(float(shifted - value))
    projected, finite_difference = np.asarray(projections), np.asarray(differences)
    error = np.linalg.norm(projected - finite_difference, ord=np.inf)
    scale = max(np.linalg.norm(projected, ord=np.inf),
                np.linalg.norm(finite_difference, ord=np.inf))
    tolerance = 0.1 if mp.is_single_precision() else 0.025
    record = dict(case=name, value=float(value), projected=projections,
                  finite_difference=differences, relative_error=float(error / scale),
                  tolerance=tolerance, sampling_interval=1, t_final=500)
    print("UPSTREAM_PORT_RATIO " + json.dumps(record), flush=True)
    assert error <= tolerance * scale, record
