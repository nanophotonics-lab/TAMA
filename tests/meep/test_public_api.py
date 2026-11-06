"""Root exports and direct modules expose the same implementations."""

import importlib
from pathlib import Path

import tama


def test_root_exports_match_direct_modules():
    exports = {
        "adaptive_chunking": (
            "AdaptiveAdjointChunkBalancer", "AdaptiveChunkBalanceReport",
            "AdaptiveSourceBoundaryDecision",
        ),
        "chunking": ("regularize_source_size_and_amplitude",),
        "coords": ("centered_grid_coords",),
        "fabrication": (
            "periodic_conic_filter", "periodic_conic_filter_vjp",
            "tanh_projection", "tanh_projection_vjp",
        ),
        "sampling_grid": (
            "FastFieldGrid", "FastGradientGrid", "native_sampler_available",
        ),
        "specs": (
            "DesignGrid", "EigenmodeCoefficientTarget", "FieldRegionTarget",
            "FluxTarget", "Near2FarTarget", "PointTarget", "SimulationSpec",
        ),
        "tda_objective": ("TDAObjective",),
        "multi_tda_objective": ("MultiTDAObjective", "auto_pixel_chunk"),
        "nyquist": ("nyquist_sampling_interval",),
    }
    package_dir = Path(tama.__file__).parent
    names = {"__version__"}
    for module_name, attributes in exports.items():
        module = importlib.import_module(f"tama.{module_name}")
        assert Path(module.__file__).parent == package_dir
        for name in attributes:
            assert getattr(tama, name) is getattr(module, name)
            names.add(name)
    assert set(tama.__all__) == names


def test_native_sampler_module_identity():
    native = importlib.import_module("tama_meep_native.native_sampler")
    assert importlib.import_module("tama.native_sampler") is native
    assert tama.native_sampler_available()
