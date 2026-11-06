from importlib.metadata import PackageNotFoundError, files, metadata, version
from pathlib import Path

import pytest

import tama


def test_package_and_distribution_versions_match():
    assert tama.__version__ == version("tama")


@pytest.mark.parametrize("distribution", ["tama", "tama-meep-native"])
def test_distribution_uses_gpl3_only_and_preserves_original_component_license(distribution):
    try:
        package_metadata = metadata(distribution)
    except PackageNotFoundError:
        if distribution == "tama-meep-native":
            pytest.skip("The native sampler is an optional distribution")
        raise

    assert package_metadata["License-Expression"] == "GPL-3.0-only"
    assert any(
        str(path).endswith("/licenses/licenses/GPL-2.0-or-later.txt")
        for path in files(distribution)
    )


def test_wheel_contains_direct_modules_and_license_files():
    package_dir = Path(tama.__file__).parent
    assert (package_dir / "py.typed").is_file()
    assert not (package_dir / "backends").exists()
    for name in (
        "tda_objective", "multi_tda_objective", "near2far",
        "design_collection", "sampling_grid", "native_sampler",
    ):
        assert (package_dir / f"{name}.py").is_file()
    distribution_files = [str(path) for path in files("tama")]
    assert any(path.endswith("/licenses/LICENSE") for path in distribution_files)
    assert any(
        path.endswith("/licenses/THIRD_PARTY_NOTICES.txt")
        for path in distribution_files
    )
    assert not any(path.endswith((".so", ".pyd")) for path in distribution_files)


def test_native_dependency_is_optional():
    assert metadata("tama").get_all("Provides-Extra") == ["meep"]
    requirements = metadata("tama").get_all("Requires-Dist")
    native_requirement = f"tama-meep-native=={version('tama')};"
    assert any(
        requirement.startswith(native_requirement) for requirement in requirements
    )
    for requirement in requirements:
        if requirement.startswith("tama-meep-native"):
            assert "extra ==" in requirement
