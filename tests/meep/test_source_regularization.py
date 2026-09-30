import meep as mp
import numpy as np

import tama as tm


def test_finite_source_regularization_preserves_complex_phase():
    amplitude = 0.6 - 0.8j

    source_size, regularized_amplitude, changed_axes = (
        tm.regularize_source_size_and_amplitude(
            mp.Vector3(),
            amplitude,
            dimensions=2,
            resolution=10,
            width_cells=2,
            axes=(mp.X, mp.Y),
        )
    )

    assert source_size == mp.Vector3(0.2, 0.2)
    assert changed_axes == ("x", "y")
    assert np.isclose(regularized_amplitude, amplitude / 0.04)


def test_legacy_slab_helpers_are_not_public():
    removed_names = (
        "SourceBoundaryDecision",
        "SourceBoundaryPolicy",
        "adjoint_source_boundary_workaround",
        "resolve_source_boundary_workaround",
        "safe_chunk_layout",
    )

    assert all(not hasattr(tm, name) for name in removed_names)
