# SPDX-License-Identifier: GPL-2.0-or-later
# Meep fixture copyright (C) 2005-2026 Massachusetts Institute of Technology.
"""Seven Meep adjoint solver regressions, evaluated through installed TAMA.

Source: NanoComp/meep@3e7b7fee0da4a9b91b07acd9f71d35d674396563,
python/tests/test_adjoint_solver.py::TestAdjointSolver. The geometry, source,
random designs, perturbation sizes, spectral objectives, and tolerances are
retained. Parameterization separates frequency groups, modes, and bandwidths.

TAMA records 500 time units at its default sampling_interval=1 instead of
Meep's DFT-decay stopping condition. Fourier transforms use dt/sqrt(2*pi).
Fixed reference modes use 91 trapezoidal points across each original port.
These temporal and spatial conversions are not bitwise Meep equivalence.
Meep supplies the incident-flux normalization and differentiable input filter;
TAMA supplies the objective values and design gradients. No Meep adjoint solve
is used. The mapped-gradient check uses a second-order forward difference
to reduce reference truncation error. All other finite-difference formulas
are unchanged.
"""

import meep as mp
import meep.adjoint as mpa
import numpy as np
import pytest
from autograd import numpy as npa
from autograd import tensor_jacobian_product

import tama as tm


def _fourier(history, dt, frequency):
    phase = npa.exp(2j * npa.pi * frequency * npa.arange(history.shape[0]) * dt)
    return npa.sum(history * phase) * dt / npa.sqrt(2 * npa.pi)


def _assert_close(actual, expected, tolerance):
    """Retain upstream utils.ApproxComparisonTestCase.assertClose's norm."""
    actual, expected = np.atleast_1d(actual).ravel(), np.atleast_1d(expected).ravel()
    difference = np.linalg.norm(actual - expected, ord=np.inf)
    scale = max(
        np.linalg.norm(actual, ord=np.inf), np.linalg.norm(expected, ord=np.inf)
    )
    assert difference <= tolerance * scale, (
        f"relative error={difference / scale:.12g}, tolerance={tolerance}, "
        f"adjoint={actual}, reference={expected}"
    )


def _assert_six_places(actual, expected):
    # unittest.assertAlmostEqual(..., places=6) rounds the absolute difference.
    assert round(abs(actual - expected), 6) == 0, (actual, expected)


class _Solver:
    def __init__(self, source_bandwidth=0.05):
        mp.verbosity(0)
        self.resolution = 30
        self.silicon = mp.Medium(epsilon=12)
        self.sapphire = mp.Medium(
            epsilon_diag=(10.225, 10.225, 9.95),
            epsilon_offdiag=(-0.825, -0.55 * np.sqrt(3 / 2), 0.55 * np.sqrt(3 / 2)),
        )
        self.cell_size = mp.Vector3(5, 5)
        self.design_size = mp.Vector3(1.5, 1.5)
        self.shape = (91, 91)
        self.eig_parity = mp.EVEN_Y + mp.ODD_Z
        self.fcen = 1 / 1.55
        self.df = 0.05 * self.fcen
        self.frequency_groups = (
            (self.fcen,),
            (self.fcen - 0.09 * self.df, self.fcen, self.fcen + 0.06 * self.df),
        )
        rng = np.random.RandomState(9861548)
        self.p = 0.5 * rng.rand(np.prod(self.shape))
        self.dp = 1e-5 * rng.rand(np.prod(self.shape))
        self.mode_source = [
            mp.EigenModeSource(
                src=mp.GaussianSource(self.fcen, fwidth=source_bandwidth * self.fcen),
                center=mp.Vector3(-1.5),
                size=mp.Vector3(0, 3),
                eig_parity=self.eig_parity,
            )
        ]
        # Keep only scalar results/gradients, never the large field histories.
        self.results = {}
        self.two_results = {}
        self.references = {}
        self.input_fluxes = {}
        self.dft_point = None

    def simulation(self, grid=None):
        geometry = [mp.Block(material=self.silicon, size=mp.Vector3(mp.inf, 1, mp.inf))]
        if grid is not None:
            geometry.append(
                mp.Block(center=mp.Vector3(), size=self.design_size, material=grid)
            )
        return tm.SimulationSpec(
            resolution=self.resolution,
            cell_size=self.cell_size,
            boundary_layers=[mp.PML(1)],
            sources=self.mode_source,
            geometry=geometry,
            courant=0.5,
            eps_averaging=True,
        )

    def design(self, medium):
        grid = mp.MaterialGrid(
            mp.Vector3(*self.shape), mp.air, medium, weights=np.ones(self.shape)
        )
        return grid, tm.DesignGrid(
            grid, mp.Vector3(), self.design_size, self.shape, mp.air, medium
        )

    def target(self, kind, frequency, mode=1, parity=None):
        if kind == "DFT":
            if self.dft_point is None:
                sim = self.simulation().make()
                monitor = sim.add_dft_fields(
                    [mp.Ez],
                    [frequency],
                    where=mp.Volume(center=mp.Vector3(1.25), size=mp.Vector3(0.25, 1)),
                    yee_grid=False,
                )
                sim.init_sim()
                x, y, _, _ = sim.get_array_metadata(dft_cell=monitor)
                self.dft_point = mp.Vector3(x[4], y[10])
                sim.reset_meep()
            return tm.PointTarget(self.dft_point, mp.Ez)
        center = mp.Vector3(-1 if kind == "reflection" else 1.5)
        weights = np.full(91, 1 / self.resolution)
        weights[[0, -1]] *= 0.5
        return tm.EigenmodeCoefficientTarget(
            positions=tuple(
                mp.Vector3(center.x, y) for y in np.linspace(-1.5, 1.5, 91)
            ),
            normal=mp.Vector3(-1 if kind == "reflection" else 1),
            mode_region=mp.Volume(center=center, size=mp.Vector3(0, 3)),
            frequency=float(frequency),
            mode=mode,
            eig_parity=self.eig_parity if parity is None else parity,
            spatial_weights=tuple(weights),
        )

    def incident(self, frequency):
        if frequency not in self.references:
            captured = []

            def capture(history, dt):
                captured.append(complex(_fourier(history, dt, frequency)))
                return 0.0

            tm.TDAObjective(
                update_design=lambda unused: None,
                simulation=self.simulation(),
                target=self.target("reflection", frequency),
                t_final=500,
                sampling_interval=1,
                chunk_balancer=None,
                fom_fn=capture,
            ).fom(np.array([]))
            self.references[frequency] = captured[-1]
        return self.references[frequency]

    def scalar(
        self,
        params,
        frequency,
        kind,
        need_gradient,
        mode=1,
        parity=None,
        subtract=False,
        medium=None,
    ):
        medium = self.silicon if medium is None else medium
        parity = self.eig_parity if parity is None else parity
        key = (
            np.asarray(params).tobytes(),
            float(frequency),
            kind,
            mode,
            parity,
            subtract,
            tuple(medium.epsilon_diag),
            tuple(medium.epsilon_offdiag),
        )
        if key in self.results and (
            not need_gradient or self.results[key][1] is not None
        ):
            return self.results[key]
        reference = self.incident(frequency) if subtract else 0j
        grid, design = self.design(medium)

        def objective(history, dt):
            return npa.abs(_fourier(history, dt, frequency) - reference) ** 2

        problem = tm.TDAObjective(
            design=design,
            simulation=self.simulation(grid),
            target=self.target(kind, frequency, mode, parity),
            t_final=500,
            sampling_interval=1,
            chunk_balancer=None,
            fom_fn=objective,
        )
        self.results[key] = problem.evaluate(params, need_gradient=need_gradient)
        return self.results[key]

    def evaluate(
        self, params, frequencies, kind="eigenmode", need_gradient=True, medium=None
    ):
        values, gradients = [], []
        for frequency in frequencies:
            if kind == "DFT":
                value, gradient = self.scalar(params, frequency, "DFT", need_gradient)
            elif len(frequencies) == 1:
                reflection, reflection_gradient = self.scalar(
                    params,
                    frequency,
                    "reflection",
                    need_gradient,
                    subtract=medium is None,
                    medium=medium,
                )
                transmission, transmission_gradient = self.scalar(
                    params,
                    frequency,
                    "transmission",
                    need_gradient,
                    mode=2,
                    medium=medium,
                )
                value = transmission - reflection
                gradient = (
                    transmission_gradient - reflection_gradient
                    if need_gradient
                    else None
                )
            else:
                value, gradient = self.scalar(
                    params,
                    frequency,
                    "transmission",
                    need_gradient,
                    medium=medium,
                )
            values.append(value)
            gradients.append(gradient)
        return np.asarray(values), (
            np.stack(gradients, axis=-1) if need_gradient else None
        )

    def two_objectives(self, params, frequencies, need_gradient=True):
        frequencies = tuple(frequencies)
        key = (np.asarray(params).tobytes(), frequencies)
        if key in self.two_results and (
            not need_gradient or self.two_results[key][1] is not None
        ):
            return self.two_results[key]
        if frequencies not in self.input_fluxes:
            ref = self.simulation().make()
            monitor = ref.add_mode_monitor(
                list(frequencies),
                mp.ModeRegion(center=mp.Vector3(-1), size=mp.Vector3(0, 3)),
                yee_grid=True,
            )
            ref.run(until_after_sources=20)
            self.input_fluxes[frequencies] = np.asarray(mp.get_fluxes(monitor))
            ref.reset_meep()
        input_flux = self.input_fluxes[frequencies]
        if len(frequencies) > 1:
            result = self.multimode(params, frequencies, input_flux, need_gradient)
        else:
            values, gradients = [], []
            for kind in ("reflection", "transmission"):
                value, gradient = self.scalar(
                    params,
                    frequencies[0],
                    kind,
                    need_gradient,
                    mode=1 if kind == "reflection" else 2,
                    parity=self.eig_parity if kind == "reflection" else mp.ODD_Z,
                    subtract=kind == "reflection",
                )
                sign = 1 if kind == "reflection" else -1
                values.append((0 if sign == 1 else 1) + sign * value / input_flux[0])
                gradients.append(
                    sign * gradient / input_flux[0] if need_gradient else None
                )
            result = np.asarray(values), (
                np.stack(gradients, axis=-1) if need_gradient else None
            )
        self.two_results[key] = result
        return result

    def multimode(self, params, frequencies, input_flux, need_gradient):
        targets, callbacks = [], []
        for kind in ("reflection", "transmission"):
            for index, frequency in enumerate(frequencies):
                targets.append(
                    self.target(
                        kind,
                        frequency,
                        mode=1 if kind == "reflection" else 2,
                        parity=self.eig_parity if kind == "reflection" else mp.ODD_Z,
                    )
                )
                reference = self.incident(frequency) if kind == "reflection" else 0j
                sign = 1 if kind == "reflection" else -1

                def callback(
                    history,
                    dt,
                    f=float(frequency),
                    r=reference,
                    s=sign,
                    norm=float(input_flux[index]),
                ):
                    return (0 if s == 1 else 1) + s * npa.abs(
                        _fourier(history, dt, f) - r
                    ) ** 2 / norm

                callbacks.append(callback)
        grid, design = self.design(self.silicon)
        selected = [0]
        problem = tm.MultiTDAObjective(
            design=design,
            simulation=self.simulation(grid),
            targets=targets,
            t_final=500,
            sampling_interval=1,
            chunk_balancer=None,
            wavelength_bands=[(1.5, 1.6)] * len(targets),
            weights=[1.0] * len(targets),
            kernel_length=1,
            fom_fn=callbacks,
            scalarization_fn=lambda values: values[selected[0]],
        )
        # A normalized one-tap filter is the identity. Select each actual target
        # derivative from MultiTDAObjective without replacing its forward solve.
        gradients = []
        for selected[0] in range(len(targets) if need_gradient else 1):
            _, gradient = problem.evaluate(params, need_gradient=need_gradient)
            gradients.append(gradient)
        return (
            np.asarray(problem.last_band_objectives),
            np.stack(gradients, axis=-1) if need_gradient else None,
        )


@pytest.fixture(scope="module")
def solver():
    return _Solver()


@pytest.fixture(scope="module", params=[0, 1], ids=["single", "three_frequencies"])
def frequencies(request, solver):
    return solver.frequency_groups[request.param]


def test_DFT_fields(solver, frequencies):
    value, gradient = solver.evaluate(solver.p, frequencies, kind="DFT")
    perturbed, _ = solver.evaluate(
        solver.p + solver.dp, frequencies, kind="DFT", need_gradient=False
    )
    _assert_close(
        solver.dp @ gradient,
        perturbed - value,
        0.075 if mp.is_single_precision() else 0.002,
    )


def test_eigenmode(solver, frequencies):
    value, gradient = solver.evaluate(solver.p, frequencies)
    perturbed, _ = solver.evaluate(
        solver.p + solver.dp, frequencies, need_gradient=False
    )
    tolerance = (
        (0.006 if mp.is_single_precision() else 5e-5)
        if len(frequencies) == 1
        else (0.008 if mp.is_single_precision() else 0.0024)
    )
    _assert_close(solver.dp @ gradient, perturbed - value, tolerance)


def test_gradient_backpropagation(solver, frequencies):
    def mapping(params):
        filtered = mpa.conic_filter(params, 0.21985, 1.5, 1.5, 60)
        return mpa.tanh_projection(filtered, 4.0698, 0.49093).flatten()

    value, gradient = solver.evaluate(mapping(solver.p), frequencies)
    raw_gradient = np.column_stack(
        [
            tensor_jacobian_product(mapping, 0)(solver.p, gradient[:, index])
            for index in range(len(frequencies))
        ]
    )
    perturbed, _ = solver.evaluate(
        mapping(solver.p + solver.dp), frequencies, need_gradient=False
    )
    twice_perturbed, _ = solver.evaluate(
        mapping(solver.p + 2 * solver.dp), frequencies, need_gradient=False
    )
    # Cancel first-order FD truncation without reducing dp or crossing bounds.
    finite_difference = (4 * (perturbed - value) - (twice_perturbed - value)) / 2
    tolerance = (
        (6e-4 if mp.is_single_precision() else 5e-6)
        if len(frequencies) == 1
        else (0.005 if mp.is_single_precision() else 0.002)
    )
    _assert_close(solver.dp @ raw_gradient, finite_difference, tolerance)


def test_offdiagonal(solver, frequencies):
    def mapping(params):
        return mpa.conic_filter(
            params.reshape(solver.shape), 0.25, 1.5, 1.5, 60
        ).flatten()

    value, gradient = solver.evaluate(
        mapping(solver.p), frequencies, medium=solver.sapphire
    )
    raw_gradient = np.column_stack(
        [
            tensor_jacobian_product(mapping, 0)(solver.p, gradient[:, index])
            for index in range(len(frequencies))
        ]
    )
    perturbed, _ = solver.evaluate(
        mapping(solver.p + solver.dp),
        frequencies,
        need_gradient=False,
        medium=solver.sapphire,
    )
    tolerance = (
        (0.04 if mp.is_single_precision() else 0.002)
        if len(frequencies) == 1
        else (0.05 if mp.is_single_precision() else 0.005)
    )
    _assert_close(solver.dp @ raw_gradient, perturbed - value, tolerance)


@pytest.mark.parametrize("objective", [0, 1], ids=["reflection", "transmission"])
def test_two_objfunc(solver, frequencies, objective):
    value, gradient = solver.two_objectives(solver.p, frequencies)
    perturbed, _ = solver.two_objectives(
        solver.p + solver.dp, frequencies, need_gradient=False
    )
    nfrq = len(frequencies)
    selection = slice(objective * nfrq, (objective + 1) * nfrq)
    tolerance = (
        (0.05 if mp.is_single_precision() else 0.0001)
        if nfrq == 1
        else (0.15 if mp.is_single_precision() else 0.001)
    )
    _assert_close(
        solver.dp @ gradient[:, selection],
        perturbed[selection] - value[selection],
        tolerance,
    )


@pytest.fixture(scope="module")
def multifrequency(solver):
    frequencies = np.linspace(
        solver.fcen - 0.2 * solver.df, solver.fcen + 0.2 * solver.df, 5
    )
    return frequencies, solver.two_objectives(solver.p, frequencies)


@pytest.mark.parametrize("frequency_index", range(5))
@pytest.mark.parametrize("objective", [0, 1], ids=["reflection", "transmission"])
def test_multifreq_monitor(solver, multifrequency, frequency_index, objective):
    frequencies, (multi_value, multi_gradient) = multifrequency
    value, gradient = solver.two_objectives(solver.p, [frequencies[frequency_index]])
    index = frequency_index + objective * len(frequencies)
    _assert_six_places(value[objective], multi_value[index])
    _assert_close(
        gradient[:, objective],
        multi_gradient[:, index],
        0.005 if mp.is_single_precision() else 0.004,
    )


@pytest.fixture(
    scope="module",
    params=np.linspace(0.05, 0.25, 5),
    ids=["width_0.05", "width_0.10", "width_0.15", "width_0.20", "width_0.25"],
)
def bandwidth_solver(request, solver):
    return solver if request.param == 0.05 else _Solver(source_bandwidth=request.param)


@pytest.mark.parametrize("objective", [0, 1], ids=["reflection", "transmission"])
def test_mode_source_bandwidth(solver, bandwidth_solver, objective):
    value, _ = solver.two_objectives(solver.p, [solver.fcen], need_gradient=False)
    perturbed, _ = solver.two_objectives(
        solver.p + solver.dp, [solver.fcen], need_gradient=False
    )
    width_value, width_gradient = bandwidth_solver.two_objectives(
        solver.p, [solver.fcen]
    )
    _assert_six_places(value[objective], width_value[objective])
    adjoint = solver.dp @ width_gradient[:, objective]
    finite_difference = perturbed[objective] - value[objective]
    relative_error = abs((finite_difference - adjoint) / finite_difference)
    assert relative_error <= 0.05, relative_error
