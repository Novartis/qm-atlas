"""Tests for the ``--print-output-names`` helper (output_names.py).

These are pure-python tests: they turn calculation task-options objects into a
preview of the SDF property tags a run would produce, without executing any
external software.
"""

import pytest
import yaml

from qm_atlas.command_line.utils import output_names
from qm_atlas.tasks.calculate_properties import TurbomoleSinglePointOptions
from qm_atlas.tasks.cosmo_properties import CosmoDeltaGConfig, CosmoLogPConfig


def test_collect_output_names_returns_tag_type_and_label():
    task = TurbomoleSinglePointOptions()
    rows = output_names.collect_output_names([task])

    tags = {row[0] for row in rows}
    assert tags == {
        "tm_sp_HOMO(eV)",
        "tm_sp_LUMO(eV)",
        "tm_sp_HLgap(eV)",
        "tm_sp_TotalEnergy(Ht)",
    }
    # Every tag is a ScalarProperty produced by the turbomole single point.
    for _tag, type_name, label in rows:
        assert type_name == "ScalarProperty"
        assert label == "turbomole_single_point"


def test_collect_output_names_across_multiple_tasks():
    tasks = [CosmoLogPConfig(), CosmoDeltaGConfig(solvent_name="h2o")]
    rows = output_names.collect_output_names(tasks)

    tags = {row[0] for row in rows}
    assert "cosmo_logp" in tags
    assert "h2o_cosmo_delta_g" in tags


def test_format_output_names_produces_aligned_table():
    table = output_names.format_output_names([TurbomoleSinglePointOptions()])

    lines = table.splitlines()
    assert lines[0].split() == ["PROPERTY", "TYPE", "TASK"]
    assert set(lines[1]) == {"-"}  # separator rule
    assert any("tm_sp_HOMO(eV)" in line for line in lines[2:])
    # Columns are left-aligned to a shared width, so the TYPE token starts at
    # the same offset on every data row.
    type_offsets = {line.index("ScalarProperty") for line in lines[2:]}
    assert len(type_offsets) == 1


def test_format_output_names_empty():
    assert output_names.format_output_names([]) == "No output properties configured."


def test_extract_config_path_space_separated():
    argv = ["conformer-properties", "--config", "run.yaml", "--input.task_id", "1"]
    assert output_names.extract_config_path(argv) == output_names.Path("run.yaml")


def test_extract_config_path_equals_form():
    argv = ["conformer-properties", "--config=nested/run.yaml"]
    assert output_names.extract_config_path(argv) == output_names.Path("nested/run.yaml")


def test_extract_config_path_absent():
    assert output_names.extract_config_path(["--print-output-names"]) is None


def test_extract_config_path_trailing_flag_without_value():
    # ``--config`` as the final token has no value to consume.
    assert output_names.extract_config_path(["--config"]) is None


def test_load_options_section_reads_subsection(tmp_path):
    config = {
        "conformer_property_options": {"calculation_tasks": [{"backend": "xtb_single_point"}]},
        "input": {"results_directory": "somewhere"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    section = output_names.load_options_section(config_path, "conformer_property_options")
    assert section == config["conformer_property_options"]


def test_load_options_section_missing_key_returns_empty(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"input": {}}), encoding="utf-8")

    assert output_names.load_options_section(config_path, "not_here") == {}


def test_load_options_section_non_mapping_value_returns_empty(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"conformer_property_options": [1, 2, 3]}), encoding="utf-8"
    )

    assert output_names.load_options_section(config_path, "conformer_property_options") == {}


def test_print_output_names_flags_constant():
    assert output_names.PRINT_OUTPUT_NAMES_FLAGS == (
        "--print-output-names",
        "--print_output_names",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
