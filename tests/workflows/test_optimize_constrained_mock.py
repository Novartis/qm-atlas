"""Mocked tests for workflows/optimize_constrained.py.

The real optimization drivers (xtb / turbomole via ``optimize.optimize_mol`` and
``run_parallel``) are mocked, so these cover the input validation, property
annotation and orchestration logic without external software.
"""

from unittest.mock import MagicMock

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from qm_atlas.tasks import optimize
from qm_atlas.workflows import optimize_constrained

ETHANOL_SMILES = "CCO"  # CHEMBL545


def _ethanol_3d():
    mol = Chem.AddHs(Chem.MolFromSmiles(ETHANOL_SMILES))
    AllChem.EmbedMolecule(mol, randomSeed=11)
    return mol


def _one_config():
    return [
        optimize.XtbTurbomoleOptions(
            max_num_steps=1,
            opt_level="normal",
            basis="def2-TZVP",
            functional=["b-p"],
            constrain=True,
            force_constant=0.1,
            torsions_fixed=True,
            angles_fixed=False,
            bonds_fixed=False,
        )
    ]


# ---------------------------------------------------------------------------
# optimize_fixed_mol
# ---------------------------------------------------------------------------


def test_optimize_fixed_mol_requires_mol():
    with pytest.raises(ValueError, match="Input molecule must be provided"):
        optimize_constrained.optimize_fixed_mol(_one_config(), mol=None)


def test_optimize_fixed_mol_annotates_settings_and_rmsd(monkeypatch):
    mol_in = _ethanol_3d()
    mol_opt = _ethanol_3d()  # already has exactly one conformer

    monkeypatch.setattr(
        optimize_constrained.optimize, "optimize_mol", lambda *a, **k: (mol_opt, None)
    )
    monkeypatch.setattr(
        optimize_constrained.conformer_geometry,
        "align_mol_to_reference",
        lambda opt, ref: (opt, {"conf0": 0.42}),
    )

    result = optimize_constrained.optimize_fixed_mol(_one_config(), mol=mol_in)

    assert result is mol_opt
    # The optimization settings are serialized onto the result ...
    assert result.HasProp("Settings_Constrained_Optimization")
    # ... and the RMSD to the input geometry is attached.
    assert result.GetDoubleProp(optimize_constrained.OPTIMIZATION_RMSD_PROP) == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# run_constrained_optimization
# ---------------------------------------------------------------------------


def test_run_constrained_optimization_requires_single_conformer():
    mol = Chem.MolFromSmiles(ETHANOL_SMILES)  # no conformer
    with pytest.raises(ValueError, match="exactly one conformation"):
        optimize_constrained.run_constrained_optimization(mol)


def test_run_constrained_optimization_delegates_to_run_parallel(monkeypatch):
    mol = _ethanol_3d()
    optimized = [MagicMock(name="opt")]
    captured = {}

    def fake_run_parallel(func, args, **kwargs):
        captured["func"] = func
        captured["args"] = args
        captured["kwargs"] = kwargs
        return optimized

    monkeypatch.setattr(optimize_constrained, "run_parallel", fake_run_parallel)

    config = [_one_config()]  # a single group of configs
    result = optimize_constrained.run_constrained_optimization(mol, optimization_config=config)

    assert result is optimized
    assert captured["func"] is optimize_constrained.optimize_fixed_mol
    # One parallel arg tuple per config group.
    assert captured["args"] == [(config[0],)]
    assert captured["kwargs"]["mol"] is mol


def test_run_constrained_optimization_uses_default_config(monkeypatch):
    mol = _ethanol_3d()
    captured = {}

    def fake_run_parallel(func, args, **kwargs):
        captured["args"] = args
        return []

    monkeypatch.setattr(optimize_constrained, "run_parallel", fake_run_parallel)

    optimize_constrained.run_constrained_optimization(mol)  # optimization_config=None
    # Falls back to the two-step DEFAULT_OPTIMIZATION_CONFIG.
    assert len(captured["args"]) == len(optimize_constrained.DEFAULT_OPTIMIZATION_CONFIG)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
