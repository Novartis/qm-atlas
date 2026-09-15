"""Tests for the cosmotherm_properties CLI page (YAML config round-trip)."""

from qm_atlas.command_line.pages import cosmotherm_properties


def test_config_roundtrip_all_backends(config_roundtrip):
    """All four COSMO task backends with non-default options round-trip."""
    _cfg, dump = config_roundtrip(
        cosmotherm_properties.load_config,
        {
            "cosmo_property_options": {
                "calculation_tasks": [
                    {"backend": "cosmo_logp", "is_woctanol": False},
                    {"backend": "cosmo_delta_g", "solvent_name": "methanol"},
                    {"backend": "cosmo_descriptors", "solvent_name": "chcl3", "level": "bp-tzvp"},
                    {"backend": "cosmo_psa", "smoothen": False, "on_charges": True},
                ]
            }
        },
    )
    tasks = dump["cosmo_property_options"]["calculation_tasks"]
    assert [t["backend"] for t in tasks] == [
        "cosmo_logp",
        "cosmo_delta_g",
        "cosmo_descriptors",
        "cosmo_psa",
    ]


def test_config_roundtrip_custom_prefix(config_roundtrip):
    """A custom property prefix survives the round-trip."""
    _cfg, dump = config_roundtrip(
        cosmotherm_properties.load_config,
        {
            "cosmo_property_options": {
                "calculation_tasks": [
                    {"backend": "cosmo_delta_g", "solvent_name": "self", "property_prefix": "dg_"},
                ]
            }
        },
    )
    (task,) = dump["cosmo_property_options"]["calculation_tasks"]
    assert task["solvent_name"] == "self"
    assert task["property_prefix"] == "dg_"
