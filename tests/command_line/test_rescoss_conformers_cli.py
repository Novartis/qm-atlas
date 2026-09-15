"""Tests for the rescoss_conformers CLI page (YAML config round-trip)."""

from qm_atlas.command_line.pages import rescoss_conformers


def test_config_roundtrip(config_roundtrip):
    """Conformer generators + xtb/DFT optimization configs round-trip with
    non-default options."""
    _cfg, dump = config_roundtrip(
        rescoss_conformers.load_config,
        {
            "rescoss_options": {
                "conformer_generation_options": {
                    "conf_gens": [
                        {
                            "backend": "macromodel",
                            "force_field": 16,
                            "solvent_num": 9,
                            "max_conformers": 500,
                        },
                        {"backend": "omega", "start_from_corina": True, "rms_threshold": 0.5},
                        {"backend": "rdkit", "method": "ETKDGv3", "random_seed": 7},
                    ],
                    "combination_name": "joint_set",
                    "rms_threshold": 0.25,
                },
                "xtb_optimization_config": [
                    {"backend": "xtb", "method": "1", "opt_level": "tight"}
                ],
                "final_optimization_config": [
                    {"backend": "xtb_turbomole", "basis": "def2-SVP", "functional": ["pbe"]}
                ],
                "num_clusters": 5,
                "num_per_cluster": 2,
            }
        },
    )
    rescoss = dump["rescoss_options"]
    conf_gens = rescoss["conformer_generation_options"]["conf_gens"]
    assert [c["backend"] for c in conf_gens] == ["macromodel", "omega", "rdkit"]
    assert rescoss["final_optimization_config"][0]["backend"] == "xtb_turbomole"
    assert rescoss["num_clusters"] == 5
