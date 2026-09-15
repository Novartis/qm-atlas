"""Tests for the optimize_reference_conformers CLI page (YAML config round-trip)."""

from qm_atlas.command_line.pages import optimize_reference_conformers


def test_config_roundtrip_nested_fallback(config_roundtrip):
    """The nested ``list[list[OptimizationConfig]]`` (per-step fallback chains,
    mixing all three optimization backends) round-trips with non-default options."""
    _cfg, dump = config_roundtrip(
        optimize_reference_conformers.load_config,
        {
            "constrained_optimization_config": [
                [
                    {
                        "backend": "xtb_turbomole",
                        "constrain": True,
                        "force_constant": 0.02,
                        "basis": "def2-SVP",
                    }
                ],
                [
                    {
                        "backend": "xtb",
                        "constrain": True,
                        "force_constant": 0.005,
                        "opt_level": "tight",
                    },
                    {"backend": "jobex", "max_num_steps": 50},
                ],
            ]
        },
    )
    steps = dump["constrained_optimization_config"]
    assert [[c["backend"] for c in step] for step in steps] == [
        ["xtb_turbomole"],
        ["xtb", "jobex"],
    ]
    assert steps[0][0]["force_constant"] == 0.02
