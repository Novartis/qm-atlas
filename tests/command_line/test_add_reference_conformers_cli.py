"""Tests for the add_reference_conformers CLI page (YAML config round-trip)."""

from conftest import RESOURCES  # pylint: disable=import-error

from qm_atlas.command_line.pages import add_reference_conformers

TEST_SDF = RESOURCES / "test_geometry.sdf"


def test_config_roundtrip(config_roundtrip, tmp_path):
    """add_reference_conformers config (top-level input_file / results_directory)
    round-trips through YAML."""
    _cfg, dump = config_roundtrip(
        add_reference_conformers.load_config,
        {
            "input_file": [str(TEST_SDF.resolve())],
            "results_directory": str(tmp_path),
            "verbose": True,
        },
        inject_results_dir=False,
    )
    assert dump["input_file"] == [str(TEST_SDF.resolve())]
    assert dump["verbose"] is True
