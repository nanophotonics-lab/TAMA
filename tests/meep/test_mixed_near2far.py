"""Temporal and near-to-far targets share a forward and adjoint simulation."""

from unittest.mock import patch

import autograd.numpy as npa
from autograd import grad
import meep as mp
import numpy as np
import pytest

import tama as tm
from tama.backends.meep import multi_tda_objective as multi_module
from tama.backends.meep import near2far as far_module


def _problem(kind="field"):
    component = mp.Hz if kind == "magnetic" else mp.Ez
    low, high = mp.Medium(epsilon=1), mp.Medium(epsilon=3)
    grid = mp.MaterialGrid(mp.Vector3(3, 3), low, high)
    design = tm.DesignGrid(grid, mp.Vector3(), mp.Vector3(0.4, 0.4), (3, 3), low, high)
    simulation = tm.SimulationSpec(
        cell_size=mp.Vector3(3, 3),
        dimensions=2,
        resolution=12,
        courant=0.25,
        boundary_layers=[mp.PML(0.4)],
        eps_averaging=False,
        geometry=[mp.Block(size=design.size, material=grid)],
        sources=[
            mp.Source(
                mp.GaussianSource(0.7, fwidth=0.4),
                component=component,
                center=mp.Vector3(-0.55, 0.09),
            )
        ],
    )
    positions = (mp.Vector3(0.7, -0.15), mp.Vector3(0.7, 0.15))
    if kind in ("field", "magnetic"):
        temporal = tm.FieldRegionTarget(positions, component)
    elif kind == "flux":
        temporal = tm.FluxTarget(positions, mp.Vector3(1), (0.3, 0.3))
    else:
        temporal = tm.EigenmodeCoefficientTarget(
            positions,
            mp.Vector3(1),
            spatial_weights=(0.3, 0.3),
            reference_mode_fields={
                mp.Ez: (1.0, 1.0),
                mp.Hy: (-1.0, -1.0),
                mp.Ey: (0.0, 0.0),
                mp.Hz: (0.0, 0.0),
            },
        )
    far = tm.Near2FarTarget(
        [mp.Near2FarRegion(center=mp.Vector3(0.8), size=mp.Vector3(0, 1.6))],
        (0.65, 0.75),
        (mp.Vector3(3, 0.4), mp.Vector3(2, -2)),
    )
    common = dict(design=design, simulation=simulation, t_final=24, chunk_balancer=None)
    return temporal, far, common


def _scalarize(values):
    return 0.4 * values[0] + 0.6 * values[1] + 0.2 * values[0] * values[1]


@pytest.mark.mpi2
@pytest.mark.parametrize("kind", ["field", "magnetic", "flux", "eigenmode"])
def test_mixed_targets_match_separate_nonlinear_objectives_and_fd(kind):
    mp.verbosity(0)
    temporal, far, common = _problem(kind)
    mixed = tm.MultiTDAObjective(
        targets=[temporal, far],
        wavelength_bands=[(1.1, 2.0), None],
        weights=[1.0, 1.0],
        kernel_length=31,
        scalarization_fn=_scalarize,
        **common,
    )
    temporal_only = tm.MultiTDAObjective(
        targets=[temporal],
        wavelength_bands=[(1.1, 2.0)],
        weights=[1.0],
        kernel_length=31,
        **common,
    )
    far_only = tm.TDAObjective(target=far, **common)
    x = np.random.default_rng(812).uniform(0.25, 0.75, 9)
    temporal_value, temporal_gradient = temporal_only.fom_and_grad(x)
    far_value, far_gradient = far_only.fom_and_grad(x)
    value, gradient = mixed.fom_and_grad(x)
    expected_values = np.array([temporal_value, far_value])
    expected_gradient = (0.4 + 0.2 * far_value) * temporal_gradient + (
        0.6 + 0.2 * temporal_value
    ) * far_gradient
    np.testing.assert_allclose(
        mixed.last_band_objectives, expected_values, rtol=1e-11, atol=1e-13
    )
    np.testing.assert_allclose(value, _scalarize(expected_values), rtol=1e-11)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=0.005, atol=1e-8)
    direction = np.random.default_rng(813).normal(size=x.size)
    direction /= np.linalg.norm(direction)
    differences = [
        (mixed.fom(x + h * direction) - mixed.fom(x - h * direction)) / (2 * h)
        for h in (2e-4, 1e-4)
    ]
    assert abs(differences[-1]) > 1e-8
    np.testing.assert_allclose(differences[0], differences[1], rtol=0.001, atol=1e-10)
    np.testing.assert_allclose(
        gradient @ direction, differences[-1], rtol=0.005, atol=1e-9
    )
    print(
        "Mixed target FD:",
        kind,
        "adjoint",
        gradient @ direction,
        "finite_difference",
        differences[-1],
        "relative_error",
        abs((gradient @ direction - differences[-1]) / differences[-1]),
    )


def test_mixed_targets_share_one_native_forward_and_adjoint_loop():
    mp.verbosity(0)
    temporal, far, common = _problem()
    mixed = tm.MultiTDAObjective(
        targets=[temporal, far],
        wavelength_bands=[(1.1, 2.0), None],
        weights=[1.0, 1.0],
        kernel_length=31,
        scalarization_fn=_scalarize,
        **common,
    )
    with (
        patch.object(
            multi_module,
            "_run_native_forward_loop",
            wraps=multi_module._run_native_forward_loop,
        ) as forward,
        patch.object(
            multi_module,
            "_run_native_adjoint_loop",
            wraps=multi_module._run_native_adjoint_loop,
        ) as adjoint,
        patch.object(
            far_module,
            "_run_native_forward_loop",
            wraps=far_module._run_native_forward_loop,
        ) as far_forward,
        patch.object(
            far_module,
            "_run_native_adjoint_loop",
            wraps=far_module._run_native_adjoint_loop,
        ) as far_adjoint,
    ):
        value, gradient = mixed.fom_and_grad(np.linspace(0.25, 0.75, 9))
    assert np.isfinite(value) and np.all(np.isfinite(gradient))
    assert forward.call_count + far_forward.call_count == 1
    assert adjoint.call_count + far_adjoint.call_count == 1


@pytest.mark.mpi4
def test_mixed_targets_use_the_active_process_group():
    if mp.comm.Get_size() != 4:
        pytest.skip("requires exactly four MPI ranks")
    mp.verbosity(0)
    group = mp.divide_parallel_processes(2)
    try:
        temporal, far, common = _problem("flux")
        far = tm.Near2FarTarget(
            far.near_regions, far.frequencies, (mp.Vector3(3 + group, 0.4),)
        )
        mixed = tm.MultiTDAObjective(
            targets=[far, temporal],
            wavelength_bands=[None, (1.1, 2.0)],
            weights=[1.0, 1.0],
            kernel_length=31,
            **common,
        )
        x = np.linspace(0.25, 0.65, 9) + 0.1 * group
        direction = np.ones_like(x) / np.sqrt(x.size)
        _, gradient = mixed.fom_and_grad(x)
        h = 1e-4
        fd = (mixed.fom(x + h * direction) - mixed.fom(x - h * direction)) / (2 * h)
        assert abs(fd) > 1e-8
        np.testing.assert_allclose(gradient @ direction, fd, rtol=0.005, atol=1e-9)
    finally:
        mp.end_divide_parallel()


def test_reversed_mixed_targets_keep_callbacks_and_scalarization_order():
    mp.verbosity(0)
    temporal, far, common = _problem()
    temporal_fn = lambda fields, dt: 1.3 * dt * npa.sum(npa.abs(fields) ** 2)
    far_fn = lambda fields, dt: npa.real((0.7 + 0.3j) * fields[0, 0, 2])
    mixed = tm.MultiTDAObjective(
        targets=[far, temporal],
        wavelength_bands=[None, (1.1, 2.0)],
        weights=[1.0, 1.0],
        kernel_length=31,
        fom_fn=[far_fn, temporal_fn],
        scalarization_fn=_scalarize,
        **common,
    )
    temporal_only = tm.MultiTDAObjective(
        targets=[temporal],
        wavelength_bands=[(1.1, 2.0)],
        weights=[1.0],
        kernel_length=31,
        fom_fn=temporal_fn,
        **common,
    )
    far_only = tm.TDAObjective(target=far, fom_fn=far_fn, **common)
    x = np.random.default_rng(812).uniform(0.25, 0.75, 9)
    temporal_value, temporal_gradient = temporal_only.fom_and_grad(x)
    far_value, far_gradient = far_only.fom_and_grad(x)
    value, gradient = mixed.fom_and_grad(x)
    np.testing.assert_allclose(
        mixed.last_band_objectives, [far_value, temporal_value], rtol=1e-11, atol=1e-13
    )
    np.testing.assert_allclose(
        value, _scalarize([far_value, temporal_value]), rtol=1e-11
    )
    expected = (0.4 + 0.2 * temporal_value) * far_gradient + (
        0.6 + 0.2 * far_value
    ) * temporal_gradient
    np.testing.assert_allclose(gradient, expected, rtol=0.005, atol=1e-8)


def test_sparse_mixed_targets_keep_value_and_gradient_endpoints_equal():
    mp.verbosity(0)
    temporal, far, common = _problem()
    common.update(t_final=23.98, sampling_interval=3)
    mixed = tm.MultiTDAObjective(
        targets=[temporal, far],
        wavelength_bands=[(1.1, 2.0), None],
        weights=[1.0, 1.0],
        kernel_length=31,
        scalarization_fn=_scalarize,
        **common,
    )
    x = np.random.default_rng(812).uniform(0.25, 0.75, 9)
    value, gradient = mixed.fom_and_grad(x)
    actual_time = mixed.last_actual_time
    np.testing.assert_allclose(mixed.fom(x), value, rtol=1e-12)
    assert mixed.last_actual_time == actual_time
    assert 23.98 <= actual_time < 23.98 + 3 * 0.25 / 12
    direction = np.ones_like(x) / np.sqrt(x.size)
    h = 1e-4
    difference = (mixed.fom(x + h * direction) - mixed.fom(x - h * direction)) / (2 * h)
    np.testing.assert_allclose(gradient @ direction, difference, rtol=0.005, atol=1e-9)


@pytest.mark.mpi2
@pytest.mark.parametrize("failure", ["value", "signal", "scalarization"])
@pytest.mark.parametrize("mixed_targets", [True, False])
def test_far_callbacks_fail_collectively(failure, mixed_targets):
    if mp.count_processors() != 2:
        pytest.skip("requires exactly two MPI ranks")
    mp.verbosity(0)
    temporal, far, common = _problem()

    def far_fn(fields, dt):
        if failure == "value" and mp.my_rank() == 0:
            return np.nan
        return 0.5 * npa.sum(npa.abs(fields[..., :3]) ** 2)

    def far_signal(fields, dt):
        if mp.my_rank() == 0:
            raise ValueError("rank-local manual covector failure")
        signal = np.zeros_like(fields)
        signal[..., :3] = np.conjugate(fields[..., :3]) / dt
        return signal

    def scalarize(values):
        if mp.my_rank() == 0:
            raise ValueError("rank-local scalarization failure")
        return npa.sum(values)

    options = (
        dict(
            targets=[temporal, far],
            wavelength_bands=[(1.1, 2.0), None],
            weights=[1.0, 1.0],
            kernel_length=31,
            fom_fn=[None, far_fn],
        )
        if mixed_targets
        else dict(targets=[far], fom_fn=[far_fn])
    )
    mixed = tm.MultiTDAObjective(
        **options,
        adjoint_signal_fn=(
            ([None, far_signal] if mixed_targets else [far_signal])
            if failure == "signal"
            else None
        ),
        scalarization_fn=scalarize if failure == "scalarization" else None,
        **common,
    )
    with pytest.raises((ValueError, RuntimeError)):
        mixed.evaluate(np.linspace(0.25, 0.75, 9), need_gradient=failure == "signal")


@pytest.mark.parametrize(
    "invalid, message",
    [
        ("bands_length", "wavelength_bands"),
        ("far_band", "None"),
        ("weights_length", "weights"),
        ("far_weight", "unit filter weights"),
        ("adaptive", "adaptive chunk balancing"),
        ("frequency", "near-to-far target frequency"),
        ("factory", "direct simulation"),
    ],
)
def test_mixed_target_configuration_guards(invalid, message):
    temporal, far, options = _problem()
    options.update(
        targets=[temporal, far],
        wavelength_bands=[(1.5, 2.0), None],
        weights=[1.0, 1.0],
        kernel_length=31,
    )
    if invalid == "bands_length":
        options["wavelength_bands"] = [(1.5, 2.0)]
    elif invalid == "far_band":
        options["wavelength_bands"][1] = (1.5, 2.0)
    elif invalid == "weights_length":
        options["weights"] = [1.0]
    elif invalid == "far_weight":
        options["weights"][1] = 0.5
    elif invalid == "adaptive":
        options["chunk_balancer"] = tm.AdaptiveAdjointChunkBalancer()
    elif invalid == "frequency":
        options["max_frequency"] = 0.7
    else:
        simulation = options.pop("simulation")
        options["sim_factory"] = lambda sources=None: simulation.make(sources)
    with pytest.raises(ValueError, match=message):
        tm.MultiTDAObjective(**options)


def test_mixed_manual_covectors_keep_temporal_and_spectral_dt_contract():
    mp.verbosity(0)
    temporal, far, common = _problem()
    callbacks = [
        lambda fields, dt: 0.7 * dt * npa.sum(npa.abs(fields) ** 2),
        lambda fields, dt: npa.real((0.7 + 0.3j) * fields[0, 0, 2]),
    ]
    mixed = tm.MultiTDAObjective(
        targets=[temporal, far],
        wavelength_bands=[(1.1, 2.0), None],
        weights=[1.0, 1.0],
        kernel_length=31,
        fom_fn=callbacks,
        scalarization_fn=_scalarize,
        **common,
    )
    x = np.linspace(0.25, 0.75, 9)
    automatic = mixed.fom_and_grad(x)
    mixed.adjoint_signal_fn = [
        lambda fields, dt, fn=fn: grad(fn, 0)(fields, dt) / dt for fn in callbacks
    ]
    manual = mixed.fom_and_grad(x)
    np.testing.assert_allclose(manual[0], automatic[0], rtol=1e-13)
    np.testing.assert_allclose(manual[1], automatic[1], rtol=1e-12, atol=1e-14)


@pytest.mark.mpi2
@pytest.mark.parametrize("include_electric", [False, True])
def test_mixed_distributed_magnetic_and_electric_histories_match_fd(include_electric):
    mp.verbosity(0)
    magnetic, far, common = _problem("magnetic")
    targets, bands = [magnetic, far], [(1.1, 2.0), None]
    if include_electric:
        targets.append(tm.FieldRegionTarget(magnetic.positions, mp.Ey))
        bands.append((1.2, 1.8))
    mixed = tm.MultiTDAObjective(
        targets=targets,
        wavelength_bands=bands,
        weights=[1.0] * len(targets),
        kernel_length=31,
        target_history_block_size=1,
        **common,
    )
    assert mixed._uses_distributed_builtin_target_history()
    x = np.random.default_rng(812).uniform(0.25, 0.75, 9)
    with patch.object(
        mixed, "_distributed_adjoint_sources", wraps=mixed._distributed_adjoint_sources
    ) as sources:
        _, gradient = mixed.fom_and_grad(x)
    assert sources.call_count == 1
    direction = np.ones_like(x) / np.sqrt(x.size)
    h = 1e-4
    difference = (mixed.fom(x + h * direction) - mixed.fom(x - h * direction)) / (2 * h)
    assert abs(difference) > 1e-8
    np.testing.assert_allclose(gradient @ direction, difference, rtol=0.005, atol=1e-9)
    print(
        "Mixed distributed FD:",
        include_electric,
        "relative_error",
        abs((gradient @ direction - difference) / difference),
    )


@pytest.mark.mpi2
def test_mixed_three_dimensional_averaged_tensor_regions_with_mirror_match_fd():
    mp.verbosity(0)
    low = mp.Medium(epsilon=1)
    high = mp.Medium(
        epsilon_diag=mp.Vector3(2.4, 2.7, 3.0), epsilon_offdiag=mp.Vector3(0, 0.1, 0)
    )
    designs, geometry = [], []
    for center in (mp.Vector3(-0.3), mp.Vector3(0.3)):
        grid = mp.MaterialGrid(
            mp.Vector3(2, 2, 2), low, high, beta=4, do_averaging=True
        )
        design = tm.DesignGrid(
            grid, center, mp.Vector3(0.35, 0.5, 0.5), (2, 2, 2), low, high
        )
        designs.append(design)
        geometry.append(mp.Block(center=center, size=design.size, material=grid))
    simulation = tm.SimulationSpec(
        cell_size=mp.Vector3(3, 2.5, 2.5),
        dimensions=3,
        resolution=8,
        courant=0.15,
        boundary_layers=[mp.PML(0.375)],
        geometry=geometry,
        eps_averaging=True,
        symmetries=[mp.Mirror(mp.Y)],
        sources=[
            mp.Source(
                mp.GaussianSource(0.7, fwidth=0.4, cutoff=3),
                component=mp.Ez,
                center=mp.Vector3(-0.75, 0, 0.07),
            )
        ],
    )
    temporal = tm.FieldRegionTarget(
        (mp.Vector3(0.7, -0.15, 0.2), mp.Vector3(0.7, 0.15, 0.2)), mp.Ez
    )
    far = tm.Near2FarTarget(
        [mp.Near2FarRegion(center=mp.Vector3(0.8), size=mp.Vector3(0, 1.2, 1.2))],
        (0.65, 0.75),
        (mp.Vector3(3, 0.4, 0.2),),
    )
    mixed = tm.MultiTDAObjective(
        designs=designs,
        simulation=simulation,
        targets=[temporal, far],
        wavelength_bands=[(1.1, 2.0), None],
        weights=[1.0, 1.0],
        kernel_length=31,
        scalarization_fn=_scalarize,
        t_final=20,
        chunk_balancer=None,
    )
    rng = np.random.default_rng(814)
    x = rng.uniform(0.25, 0.75, (2, 2, 2, 2))
    x = 0.5 * (x + np.flip(x, axis=2))
    direction = rng.normal(size=x.shape)
    direction = 0.5 * (direction + np.flip(direction, axis=2))
    direction /= np.linalg.norm(direction)
    x, direction = x.ravel(), direction.ravel()
    value, gradient = mixed.fom_and_grad(x)
    assert np.isfinite(value) and gradient.shape == (16,)
    np.testing.assert_allclose(
        gradient.reshape(2, 2, 2, 2), np.flip(gradient.reshape(2, 2, 2, 2), axis=2)
    )
    h = 1e-4
    difference = (mixed.fom(x + h * direction) - mixed.fom(x - h * direction)) / (2 * h)
    assert abs(difference) > 1e-8
    np.testing.assert_allclose(gradient @ direction, difference, rtol=0.005, atol=1e-9)
    print(
        "Mixed 3D tensor/regions/Mirror FD: relative_error",
        abs((gradient @ direction - difference) / difference),
    )
