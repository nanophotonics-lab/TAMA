from importlib.metadata import metadata, version

import tama


def test_package_and_distribution_versions_match():
    assert tama.__version__ == version("tama")


def test_distribution_declares_gpl_v2_or_later():
    package_metadata = metadata("tama")

    assert package_metadata["License"] == "GPL-2.0-or-later"
    assert (
        "License :: OSI Approved :: GNU General Public License v2 or later "
        "(GPLv2+)"
    ) in package_metadata.get_all("Classifier")
