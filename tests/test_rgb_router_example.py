import importlib.util
from pathlib import Path

import numpy as np


EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "tda_3d_gamma_rgb_router.py"
)


def _load_example():
    spec = importlib.util.spec_from_file_location("tama_rgb_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rgb_routing_scalarization_uses_all_nine_incident_normalized_terms():
    example = _load_example()
    incident = np.array([2.0, 4.0, 5.0])
    flux = np.array(
        [
            [1.0, 0.2, 0.1],
            [0.4, 2.0, 0.2],
            [0.5, 0.25, 2.5],
        ]
    )
    value, coefficients, info = example.make_routing_scalarization(
        incident
    )(flux.ravel())
    efficiency = flux / incident[:, np.newaxis]
    desired = np.diag(efficiency)
    off_diagonal = efficiency.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    crosstalk_penalty = np.sum(off_diagonal**2, axis=1)
    expected = np.mean(
        desired - example.CROSSTALK_PENALTY * crosstalk_penalty
    )

    assert value == expected
    assert coefficients.shape == (9,)
    np.testing.assert_allclose(info["efficiency_matrix"], efficiency)
    np.testing.assert_allclose(info["desired_efficiency"], desired)
    np.testing.assert_allclose(
        info["off_diagonal_efficiency"],
        off_diagonal,
    )
    np.testing.assert_allclose(
        info["crosstalk_penalty"],
        crosstalk_penalty,
    )

    step = 1e-7
    for index in range(flux.size):
        direction = np.zeros(flux.size)
        direction[index] = 1.0
        plus = example.make_routing_scalarization(incident)(
            flux.ravel() + step * direction
        )[0]
        minus = example.make_routing_scalarization(incident)(
            flux.ravel() - step * direction
        )[0]
        np.testing.assert_allclose(
            coefficients[index],
            (plus - minus) / (2.0 * step),
            rtol=5e-8,
            atol=5e-10,
        )


def test_rgb_router_dry_run_builds_reference_and_nine_term_objective(
    capsys,
    tmp_path,
):
    example = _load_example()
    output_path = tmp_path / "must-not-exist.npz"
    example.main([
        "--dry-run",
        "--iterations",
        "1",
        "--output",
        str(output_path),
    ])

    output = capsys.readouterr().out
    assert "reference_bands=3" in output
    assert "device_band_detector_terms=9" in output
    assert not output_path.exists()
