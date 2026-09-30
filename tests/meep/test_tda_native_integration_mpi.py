import meep as mp
import numpy as np
import pytest

import tama as tm


def _make_objective(
    *,
    reuse_simulation=False,
    use_simulation_spec=False,
    target_on_boundary=False,
    regional_target=False,
    flux_target=False,
    eigenmode_target=False,
    sampling_interval=1,
    max_frequency=None,
    t_final=1.0,
    reconstruction_window="kaiser",
):
    resolution = 8
    dt = 0.5 / resolution
    cell_size = mp.Vector3(6, 6)
    source_position = mp.Vector3(-1.1, -0.3)
    monitor_position = mp.Vector3(
        0.0 if target_on_boundary and not eigenmode_target else 0.7,
        0.2,
    )
    layout = (
        mp.BinaryPartition(data=[(mp.X, 0.0), 0, 1])
        if mp.count_processors() == 2
        else None
    )
    forward_sources = [
        mp.Source(
            mp.GaussianSource(frequency=1.5, fwidth=1.0),
            component=mp.Ez,
            center=source_position,
        )
    ]
    air = mp.Medium(epsilon=1.0)
    dielectric = mp.Medium(epsilon=2.0)
    design_center = mp.Vector3()
    design_size = mp.Vector3(1.0, 1.0)
    material_grid = mp.MaterialGrid(
        mp.Vector3(2, 2),
        air,
        dielectric,
        do_averaging=False,
    )
    design = tm.DesignGrid(
        material_grid=material_grid,
        center=design_center,
        size=design_size,
        shape=(2, 2),
        background=air,
        design_material=dielectric,
    )
    geometry = [
        mp.Block(
            center=design_center,
            size=design_size,
            material=material_grid,
        )
    ]

    def sim_factory(sources=None):
        return mp.Simulation(
            cell_size=cell_size,
            resolution=resolution,
            chunk_layout=layout,
            geometry=geometry,
            sources=forward_sources if sources is None else sources,
            eps_averaging=False,
        )

    simulation_args = (
        {
            "simulation": tm.SimulationSpec(
                cell_size=cell_size,
                geometry=geometry,
                sources=forward_sources,
                resolution=resolution,
                chunk_layout=layout,
                eps_averaging=False,
            )
        }
        if (
            reuse_simulation
            or use_simulation_spec
            or regional_target
            or flux_target
            or eigenmode_target
        )
        else {"sim_factory": sim_factory}
    )
    target_args = (
        {
            "target": tm.EigenmodeCoefficientTarget(
                positions=(
                    monitor_position + mp.Vector3(0.0, -0.1),
                    monitor_position + mp.Vector3(0.0, 0.1),
                ),
                normal=mp.Vector3(1.0),
                mode_region=mp.Volume(
                    center=monitor_position,
                    size=mp.Vector3(0.0, 4.0),
                ),
                frequency=1.5,
                spatial_weights=(0.75, 1.25),
                eig_parity=mp.ODD_Z,
            )
        }
        if eigenmode_target
        else {
            "target": tm.FluxTarget(
                positions=(
                    monitor_position + mp.Vector3(0.0, -0.1),
                    monitor_position + mp.Vector3(0.0, 0.1),
                ),
                normal=mp.Vector3(1.0),
                spatial_weights=(0.75, 1.25),
            )
        }
        if flux_target
        else {
            "target": tm.FieldRegionTarget(
                positions=(
                    monitor_position + mp.Vector3(0.0, -0.1),
                    monitor_position + mp.Vector3(0.0, 0.1),
                ),
                component=mp.Ez,
                sample_shape=(2,),
                spatial_weights=(0.75, 1.25),
            )
        }
        if regional_target
        else {
            "monitor_position": monitor_position,
            "component": mp.Ez,
        }
    )
    return tm.TDAObjective(
        design=design,
        t_final=t_final,
        dt=dt,
        sampling_interval=sampling_interval,
        max_frequency=max_frequency,
        reconstruction_window=reconstruction_window,
        reuse_simulation=reuse_simulation,
        **(
            {}
            if flux_target or eigenmode_target
            else {
                "fom_fn": lambda history, sample_dt: float(
                    0.5 * np.sum(np.abs(history) ** 2) * sample_dt
                ),
                "adjoint_signal_fn": (
                    lambda history, sample_dt: np.ones_like(history)
                ),
            }
        ),
        **simulation_args,
        **target_args,
    )


@pytest.mark.mpi2
def test_tda_simulation_reuse_matches_separate_simulations():
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")

    for regional_target, flux_target, eigenmode_target in (
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ):
        separate_value, separate_gradient = _make_objective(
            use_simulation_spec=True,
            target_on_boundary=True,
            regional_target=regional_target,
            flux_target=flux_target,
            eigenmode_target=eigenmode_target,
        ).evaluate(
            np.zeros(4),
            need_gradient=True,
        )
        reused_value, reused_gradient = _make_objective(
            reuse_simulation=True,
            target_on_boundary=True,
            regional_target=regional_target,
            flux_target=flux_target,
            eigenmode_target=eigenmode_target,
        ).evaluate(
            np.zeros(4),
            need_gradient=True,
        )

        assert np.allclose(
            reused_value,
            separate_value,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        assert np.allclose(
            reused_gradient,
            separate_gradient,
            rtol=1.0e-11,
            atol=1.0e-12,
        )


@pytest.mark.mpi2
def test_tda_nyquist_frequency_matches_explicit_interval():
    if mp.count_processors() < 2:
        pytest.skip("requires at least two MPI ranks")
    assert tm.native_sampler_available()

    explicit_objective = _make_objective(sampling_interval=2)
    explicit_value, explicit_gradient = explicit_objective.evaluate(
        np.zeros(4),
        need_gradient=True,
    )
    nyquist_objective = _make_objective(max_frequency=3.0)
    nyquist_value, nyquist_gradient = nyquist_objective.evaluate(
        np.zeros(4),
        need_gradient=True,
    )

    assert explicit_objective.last_sampling_interval == 2
    assert nyquist_objective.last_sampling_interval == 2
    assert np.allclose(nyquist_value, explicit_value, rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(
        nyquist_gradient,
        explicit_gradient,
        rtol=1.0e-11,
        atol=1.0e-12,
    )


def test_tda_nyquist_alignment_preserves_objective_value():
    design = np.zeros(4)
    full_value, _ = _make_objective(t_final=0.26).evaluate(
        design,
        need_gradient=False,
    )
    nyquist_objective = _make_objective(
        t_final=0.26,
        max_frequency=2.5,
    )
    nyquist_value_with_gradient, _ = nyquist_objective.evaluate(
        design,
        need_gradient=True,
    )
    nyquist_value_only, _ = nyquist_objective.evaluate(
        design,
        need_gradient=False,
    )

    assert nyquist_objective.last_sampling_interval == 3
    assert np.allclose(
        nyquist_value_with_gradient,
        full_value,
        rtol=1.0e-13,
        atol=1.0e-15,
    )
    assert np.allclose(
        nyquist_value_only,
        full_value,
        rtol=1.0e-13,
        atol=1.0e-15,
    )


def test_tda_nyquist_gradient_matches_full_rate_after_field_decay():
    design = np.zeros(4)
    _, full_gradient = _make_objective(
        sampling_interval=1,
        t_final=8.0,
    ).evaluate(design, need_gradient=True)
    _, sparse_gradient = _make_objective(
        sampling_interval=2,
        t_final=8.0,
        reconstruction_window="hann",
    ).evaluate(design, need_gradient=True)

    relative_error = np.linalg.norm(
        sparse_gradient - full_gradient
    ) / np.linalg.norm(full_gradient)
    assert relative_error < 1.0e-3


if __name__ == "__main__":
    test_tda_simulation_reuse_matches_separate_simulations()
    test_tda_nyquist_frequency_matches_explicit_interval()
    test_tda_nyquist_alignment_preserves_objective_value()
    test_tda_nyquist_gradient_matches_full_rate_after_field_decay()
