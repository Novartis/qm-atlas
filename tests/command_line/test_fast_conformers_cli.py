"""Tests for the fast_conformers CLI page (YAML config round-trip)."""

from qm_atlas.command_line.pages import fast_conformers


def test_config_roundtrip(config_roundtrip):
    """Fallback conformer generation + a two-step xtb optimization round-trip."""
    _cfg, dump = config_roundtrip(
        fast_conformers.load_config,
        {
            "fast_conformers_options": {
                "conformer_generation_options": {
                    "conf_gens": [
                        {"backend": "omega", "max_conformers": 250},
                        {"backend": "rdkit", "method": "ETKDGv2"},
                    ],
                    "combination_name": "fallback",
                },
                "optimization_config": [
                    {"backend": "xtb", "method": "2", "opt_level": "lax", "max_num_steps": 250},
                    {
                        "backend": "xtb",
                        "method": "0",
                        "solvation_model": "gbsa",
                        "solvent": "water",
                    },
                ],
                "final_optimization_config": [],
                "expand_conformers": False,
            }
        },
    )
    opts = dump["fast_conformers_options"]
    assert [o["backend"] for o in opts["optimization_config"]] == ["xtb", "xtb"]
    assert opts["optimization_config"][0]["opt_level"] == "lax"
    assert opts["expand_conformers"] is False
