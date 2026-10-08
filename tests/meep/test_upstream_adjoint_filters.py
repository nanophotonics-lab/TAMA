# SPDX-License-Identifier: GPL-2.0-or-later
# Meep fixture copyright (C) 2005-2026 Massachusetts Institute of Technology.
"""TAMA filters in Meep 1.34's fully periodic conic-filter regression.

Upstream: NanoComp/meep@3e7b7fee0da4a9b91b07acd9f71d35d674396563,
python/tests/test_adjoint_solver.py::TestAdjointSolver.test_periodic_design.
The mesh, seed, radius, projection, solid/void constraints, shifts, and
15-place scalar and gradient-L1 comparisons are retained. Meep supplies
the constraint functions; TAMA supplies the filter, projection, and VJPs.

Only the conic branch with both axes periodic has matching semantics.
Meep filters nonperiodic axes with edge padding, whereas TAMA leaves axes
outside periodic_axes uncoupled. The single-periodic-axis, unequal-resolution,
and nonperiodic zero-phase fixtures therefore are not ported here.
"""

import meep.adjoint as mpa
import numpy as np
import pytest
from autograd import grad
from autograd.extend import defvjp, primitive

import tama as tm


@pytest.mark.parametrize(
    "constraint, eta",
    [(mpa.constraint_solid, 0.75), (mpa.constraint_void, 0.25)],
    ids=["solid", "void"],
)
@pytest.mark.parametrize("axis, fraction", [(0, 0.19), (1, 0.28)], ids=["x", "y"])
def test_upstream_fully_periodic_conic_constraints(constraint, eta, axis, fraction):
    # These callbacks expose TAMA's explicit VJPs to Meep's Autograd objective.
    @primitive
    def filter_f(weights):
        return tm.periodic_conic_filter(
            weights, radius=0.3, spacing=1 / 60, periodic_axes=(0, 1)
        )

    defvjp(
        filter_f,
        lambda result, weights: lambda cotangent: tm.periodic_conic_filter_vjp(
            cotangent, radius=0.3, spacing=1 / 60, periodic_axes=(0, 1)
        ),
    )

    @primitive
    def threshold_f(weights):
        return tm.tanh_projection(weights, beta=10, eta=0.5)

    defvjp(
        threshold_f,
        lambda result, weights: lambda cotangent: tm.tanh_projection_vjp(
            weights, cotangent, beta=10, eta=0.5
        ),
    )

    weights = 0.5 * np.random.RandomState(9861548).rand(91, 91)
    shift = int(fraction * weights.shape[axis])
    shifted_weights = np.roll(weights, -shift, axis=axis)
    tama_args = (400, eta, filter_f, threshold_f, 60, (0, 1))
    meep_args = (
        400,
        eta,
        lambda x: mpa.conic_filter(x, 0.3, 1.5, 1.5, 60, (0, 1)),
        lambda x: mpa.tanh_projection(x, 10, 0.5),
        60,
        (0, 1),
    )

    value = constraint(weights, *tama_args)
    gradient = grad(constraint)(weights, *tama_args)
    expected_value = constraint(weights, *meep_args)
    expected_gradient = grad(constraint)(weights, *meep_args)
    assert np.isfinite(value) and np.all(np.isfinite(gradient))
    assert np.max(np.abs(gradient)) > 0

    # Match unittest.assertAlmostEqual(..., places=15), including its L1 norm.
    assert round(abs(value - expected_value), 15) == 0
    assert round(np.sum(np.abs(gradient - expected_gradient)), 15) == 0
    shifted_value = constraint(shifted_weights, *tama_args)
    shifted_gradient = grad(constraint)(shifted_weights, *tama_args)
    assert round(abs(value - shifted_value), 15) == 0
    assert round(
        np.sum(np.abs(np.roll(gradient, -shift, axis=axis) - shifted_gradient)), 15
    ) == 0
