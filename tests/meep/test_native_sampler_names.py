from pathlib import Path
import importlib
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "tama" / "backends" / "meep"
NATIVE = ROOT / "native" / "meep"


def test_native_sampler_import_guard_raises_clear_missing_extension_error():
    namespace = {
        "__name__": "tama.native_sampler",
        "__package__": "tama",
    }
    missing = ModuleNotFoundError("missing test extension", name="tama_meep_native")
    with patch.object(importlib, "import_module", side_effect=missing):
        try:
            exec((SRC / "native_sampler.py").read_text(encoding="utf-8"), namespace)
        except ImportError as exc:
            message = str(exc)
        else:
            raise AssertionError("expected native sampler import guard to raise ImportError")

    assert "native sampler extension is not built" in message
    assert "fastmeep_sample" not in message


def test_native_sampler_allreduced_names_replace_local_sum_names():
    checked_files = [
        NATIVE / "native_sampler.cpp",
        NATIVE / "src" / "tama_meep_native" / "native_sampler.pyi",
        SRC / "sampling_grid.py",
    ]

    combined = "\n".join(path.read_text(encoding="utf-8") for path in checked_files)

    for old_name in (
        "sample_component_grid_plan_local_sum",
        "sample_component_grid_local_sum",
        "accumulate_component_product_local_sum",
    ):
        assert old_name not in combined

    for new_name in (
        "sample_component_grid_plan_allreduced",
        "sample_component_grid_allreduced",
        "accumulate_component_product_allreduced",
    ):
        assert new_name in combined


if __name__ == "__main__":
    test_native_sampler_import_guard_raises_clear_missing_extension_error()
    test_native_sampler_allreduced_names_replace_local_sum_names()
