"""Small real-Meep checks of full FIR output and its material gradient.

Run serially or with two MPI ranks. The geometry and gradient tolerances
(0.6% for field energy; 1% for flux/modes) come from existing native tests.
"""

import json

import meep as mp
import numpy as np
import pytest

import tama as tm


def _make_objective(path):
    resolution = 20
    air, dielectric = mp.Medium(epsilon=1), mp.Medium(epsilon=4)
    center, size = mp.Vector3(0.013, -0.017), mp.Vector3(0.237, 0.263)
    material = mp.MaterialGrid(mp.Vector3(5, 5), air, dielectric)
    design = tm.DesignGrid(
        material_grid=material, center=center, size=size, shape=(5, 5),
        background=air, design_material=dielectric,
    )
    simulation = tm.SimulationSpec(
        cell_size=mp.Vector3(6, 6), boundary_layers=[mp.PML(1)],
        geometry=[mp.Block(center=center, size=size, material=material)],
        sources=[mp.Source(
            mp.GaussianSource(frequency=0.7, fwidth=0.4), component=mp.Ez,
            center=mp.Vector3(-1.5, center.y), size=mp.Vector3(0, 3.5),
        )],
        resolution=resolution, eps_averaging=False,
        chunk_layout=(
            mp.BinaryPartition(data=[(mp.X, 0.0), 0, 1])
            if mp.count_processors() == 2 else None
        ),
    )
    positions = (mp.Vector3(1.5, center.y - 0.1),
                 mp.Vector3(1.5, center.y + 0.1))
    target_count = 2 if path.endswith(("dedup", "distributed", "multi")) else 1
    targets = []
    for index in range(target_count):
        if path.startswith("field"):
            target = tm.FieldRegionTarget(
                positions, mp.Ez, sample_shape=(2,), spatial_weights=(0.6, 1.4),
            )
        elif path.startswith("flux"):
            target = tm.FluxTarget(
                positions, normal=mp.Vector3(1), spatial_weights=(0.6, 1.4),
            )
        else:
            target = tm.EigenmodeCoefficientTarget(
                positions=positions, normal=mp.Vector3(1),
                mode_region=mp.Volume(center=mp.Vector3(1.5, center.y),
                                      size=mp.Vector3(0, 3.5)),
                frequency=0.7 + 0.05 * index, spatial_weights=(0.6, 1.4),
                eig_parity=mp.ODD_Z,
            )
        targets.append(target)
    return tm.MultiTDAObjective(
        design=design, simulation=simulation, targets=targets,
        t_final=100.0, dt=0.5 / resolution, sampling_interval=1,
        wavelength_bands=[(1.1, 2.0)] if target_count == 1
        else [(1.1, 1.5), (1.5, 2.0)],
        weights=[1.0] if target_count == 1 else [0.7, 1.3],
        kernel_length=1001, pixel_chunk=5, reuse_simulation=False,
        target_history_block_size=1 if path.endswith("distributed") else None,
    )


PATHS = (
    "field", "field_dedup", "field_distributed",
    "flux", "flux_dedup", "flux_distributed",
    "eigenmode", "eigenmode_multi",
)


@pytest.mark.parametrize("path", PATHS)
def test_filtered_material_gradient_matches_central_fd(path, monkeypatch):
    mp.verbosity(0)
    objective = _make_objective(path)
    calls = {}
    for name in (
        "_distributed_builtin_band_objectives",
        "_distributed_deduplicated_band_objectives",
        "_eigenmode_fom_values_and_overlap_covectors",
        "_filter_transpose_eigenmode_overlap_covectors",
        "_band_fom_values_and_adjoint_signals",
    ):
        original = getattr(objective, name)

        def record(*args, _name=name, _original=original, **kwargs):
            calls[_name] = calls.get(_name, 0) + 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(objective, name, record)

    rng = np.random.default_rng(20260801 + int(mp.Ez))
    density = rng.uniform(0.15, 0.85, 25)
    directions = np.stack((rng.uniform(0.2, 1.0, 25), rng.normal(size=25)))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    value, gradient = objective.fom_and_grad(density)
    assert np.isfinite(value) and value > 0
    assert gradient.shape == (25,) and np.all(np.isfinite(gradient))
    assert objective.run_time == 100.0
    assert objective.filter_time == pytest.approx(1001 * 0.025)
    rows = []
    step = 2e-4
    for index, direction in enumerate(directions):
        plus = objective.fom(density + step * direction)
        minus = objective.fom(density - step * direction)
        fd = float((plus - minus) / (2 * step))
        adjoint = float(gradient @ direction)
        relative = abs(adjoint - fd) / abs(fd)
        rows.append(dict(direction=index, h=step, adjoint=adjoint,
                         finite_difference=fd, relative_error=relative))
        assert abs(fd) > 1e-10
        assert np.sign(adjoint) == np.sign(fd)
    expected_path = (
        "_distributed_builtin_band_objectives" if path.endswith("distributed")
        else "_distributed_deduplicated_band_objectives" if path.endswith("dedup")
        else "_eigenmode_fom_values_and_overlap_covectors" if path.startswith("eigenmode")
        else "_band_fom_values_and_adjoint_signals"
    )
    assert calls[expected_path] == 5
    if path.startswith("eigenmode"):
        assert calls["_filter_transpose_eigenmode_overlap_covectors"] == 1
    if mp.am_master():
        print(json.dumps(dict(path=path, ranks=mp.count_processors(),
                              value=float(value), results=rows), sort_keys=True))
    tolerance = 0.006 if path.startswith("field") else 0.01
    assert max(row["relative_error"] for row in rows) < tolerance


@pytest.mark.parametrize("kind", ("field", "flux"))
def test_filtered_distributed_matches_deduplicated_history(kind):
    mp.verbosity(0)
    density = np.linspace(0.15, 0.85, 25)
    reference = _make_objective(kind + "_dedup")
    distributed = _make_objective(kind + "_distributed")
    reference_value, reference_gradient = reference.fom_and_grad(density)
    value, gradient = distributed.fom_and_grad(density)
    assert reference._uses_deduplicated_monitors
    assert reference.last_target_history_mode == "legacy"
    assert distributed.last_target_history_mode == "distributed"
    np.testing.assert_allclose(value, reference_value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gradient, reference_gradient, rtol=1e-10, atol=1e-12)
