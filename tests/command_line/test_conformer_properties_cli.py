"""Tests for the conformer_properties CLI page (YAML config round-trip)."""

from qm_atlas.command_line.pages import conformer_properties


def test_config_roundtrip_multi_backend(config_roundtrip):
    """A multi-task config with non-default options round-trips through YAML."""
    _cfg, dump = config_roundtrip(
        conformer_properties.load_config,
        {
            "conformer_property_options": {
                "calculation_tasks": [
                    {
                        "backend": "xtb_single_point",
                        "solvation_model": "alpb",
                        "solvent": "methanol",
                        "calculate_fukui": True,
                    },
                    {
                        "backend": "turbomole_single_point",
                        "basis": "def2-mTZVP",
                        "functional": ["b97-3c"],
                        "use_cosmo": False,
                    },
                    {"backend": "turbomole_fukui_indices", "grid": "m4"},
                ]
            }
        },
    )
    tasks = dump["conformer_property_options"]["calculation_tasks"]
    assert [t["backend"] for t in tasks] == [
        "xtb_single_point",
        "turbomole_single_point",
        "turbomole_fukui_indices",
    ]


def test_config_roundtrip_single_task(config_roundtrip):
    """A single-task config with a custom property prefix round-trips."""
    _cfg, dump = config_roundtrip(
        conformer_properties.load_config,
        {
            "conformer_property_options": {
                "calculation_tasks": [
                    {"backend": "turbomole_vcd", "property_prefix": "vcd_custom_", "grid": "5"},
                ]
            }
        },
    )
    (task,) = dump["conformer_property_options"]["calculation_tasks"]
    assert task["backend"] == "turbomole_vcd"
    assert task["property_prefix"] == "vcd_custom_"
