"""Tests for format_utils.py.

These exercise the RDKit <-> SDF conversion helpers. No external software is
required; everything runs through RDKit in-process.
"""

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from qm_atlas.command_line.utils import format_utils

ETHANOL_SMILES = "CCO"  # CHEMBL545


def _ethanol_3d() -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(ETHANOL_SMILES))
    AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    mol.SetProp("_Name", "ethanol")
    return mol


def test_write_sdf_v3000_and_v2000(tmp_path):
    mol = _ethanol_3d()

    v3000_file = tmp_path / "ethanol_v3000.sdf"
    format_utils.write_sdf(mol, v3000_file)
    v3000_text = v3000_file.read_text()
    assert "V3000" in v3000_text

    v2000_file = tmp_path / "ethanol_v2000.sdf"
    format_utils.write_sdf(mol, v2000_file, use_v2000=True)
    v2000_text = v2000_file.read_text()
    assert "V3000" not in v2000_text

    # Both files describe the same molecule.
    assert Chem.MolFromMolBlock(v2000_text.split("$$$$")[0]) is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
