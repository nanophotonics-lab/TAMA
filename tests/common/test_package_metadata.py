from importlib.metadata import files, metadata, version
from pathlib import Path

import tama


def test_package_and_distribution_versions_match():
    assert tama.__version__ == version("tama")


def test_distribution_uses_gpl3_and_preserves_original_component_license():
    package_metadata = metadata("tama")

    assert package_metadata["License-Expression"] == "GPL-3.0-or-later"
    assert any(
        str(path).endswith("/licenses/licenses/GPL-2.0-or-later.txt")
        for path in files("tama")
    )


def test_wheel_contains_meep_backend_and_license_files():
    package_dir = Path(tama.__file__).parent
    assert (package_dir / "py.typed").is_file()
    assert {
        path.name
        for path in (package_dir / "backends").iterdir()
        if path.is_dir() and path.name != "__pycache__"
    } == {"meep"}
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
