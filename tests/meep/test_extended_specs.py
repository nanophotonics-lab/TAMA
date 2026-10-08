import meep as mp
import numpy as np
import pytest

import tama as tm


def test_mirror_settings_reach_forward_and_adjoint_simulations():
    mirrors = [mp.Mirror(mp.Y, phase=-1)]
    spec = tm.SimulationSpec(cell_size=mp.Vector3(2, 2), symmetries=mirrors)
    for sim in (spec.make(), spec.make([])):
        assert len(sim.symmetries) == 1
        assert sim.symmetries[0].direction == mp.Y
        assert sim.symmetries[0].phase == -1


@pytest.mark.parametrize("symmetry", [mp.Rotate2(mp.Z), mp.Mirror(mp.Y, phase=0)])
def test_unsupported_symmetry_cannot_silently_change_the_physics(symmetry):
    with pytest.raises(ValueError, match="Mirror|phase"):
        tm.SimulationSpec(cell_size=mp.Vector3(2, 2), symmetries=[symmetry])


def _near_region():
    return mp.Near2FarRegion(center=mp.Vector3(0, 0.5), size=mp.Vector3(1, 0))


def test_near2far_target_preserves_frequency_and_observation_order():
    target = tm.Near2FarTarget(
        near_regions=[_near_region()],
        frequencies=[0.8, 0.6],
        far_points=[mp.Vector3(0, 10), mp.Vector3(1, 10)],
    )
    assert target.frequencies == (0.8, 0.6)
    assert target.far_points[1].x == 1
    assert len(target.near_regions) == 1
    assert tm.get_backend("meep").Near2FarTarget is tm.Near2FarTarget


@pytest.mark.parametrize(
    "frequencies", [[], [0], [-1], [np.nan], [np.inf], np.array([0.6 + 0.1j])]
)
def test_near2far_target_rejects_invalid_frequencies(frequencies):
    with pytest.raises(ValueError, match="frequencies"):
        tm.Near2FarTarget([_near_region()], frequencies, [mp.Vector3(0, 10)])


@pytest.mark.parametrize(
    "points", [[], [mp.Vector3(np.nan)], [mp.Vector3(1j)], [(0, 10, 0)]]
)
def test_near2far_target_rejects_invalid_observation_points(points):
    with pytest.raises((TypeError, ValueError), match="far_points"):
        tm.Near2FarTarget([_near_region()], [0.6], points)


@pytest.mark.parametrize("regions", [[], [object()]])
def test_near2far_target_requires_near_surface_regions(regions):
    with pytest.raises((TypeError, ValueError), match="near_regions"):
        tm.Near2FarTarget(regions, [0.6], [mp.Vector3(0, 10)])
