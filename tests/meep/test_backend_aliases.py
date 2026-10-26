"""Compatibility aliases share classes, state, and native module identity."""

import importlib

import tama


def test_legacy_module_aliases_and_backend_exports():
    backend = tama.get_backend("meep")
    for name in (
        "adaptive_chunking",
        "chunking",
        "coords",
        "eigenmode",
        "fabrication",
        "fastmeep_grid",
        "flux",
        "multi_tda_objective",
        "native_design",
        "objectives",
        "sampling_grid",
        "specs",
        "tda_objective",
    ):
        legacy = importlib.import_module(f"tama.{name}")
        relocated = importlib.import_module(f"tama.backends.meep.{name}")
        assert legacy is relocated
    for name in tama.__all__:
        if name != "get_backend":
            assert getattr(tama, name) is getattr(backend, name)
    native = importlib.import_module("tama_meep_native.native_sampler")
    assert importlib.import_module("tama.native_sampler") is native
    assert importlib.import_module("tama.backends.meep.native_sampler") is native
    assert tama.native_sampler_available()
