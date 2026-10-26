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
