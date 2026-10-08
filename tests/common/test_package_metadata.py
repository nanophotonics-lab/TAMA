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


def test_wheel_contains_embedded_backend_and_provenance():
    package_dir = Path(tama.__file__).parent
    assert (package_dir / "py.typed").is_file()
    assert not (package_dir / "backends/fdtdx/fdtdx_source_sha256.json").exists()
    assert (package_dir / "backends/fdtdx/LICENSE").is_file()
    assert (package_dir / "backends/fdtdx/THIRD_PARTY_NOTICES.txt").is_file()
    distribution_files = [str(path) for path in files("tama")]
    assert any(path.endswith("/licenses/LICENSE") for path in distribution_files)
    assert any(
        path.endswith("/licenses/THIRD_PARTY_NOTICES.txt")
        for path in distribution_files
    )
    assert not any(path.endswith((".so", ".pyd")) for path in distribution_files)
    assert not any(path.startswith("tama_fdtdx/") for path in distribution_files)


def test_backend_dependencies_are_optional():
    requirements = metadata("tama").get_all("Requires-Dist")
    native_requirement = f"tama-meep-native=={version('tama')};"
    assert any(
        requirement.startswith(native_requirement) for requirement in requirements
    )
    for requirement in requirements:
        if requirement.startswith(
            ("fdtdx", "jax", "jaxlib", "equinox", "tama-meep-native")
        ):
            assert "extra ==" in requirement
        assert not requirement.startswith("tama-fdtdx")
