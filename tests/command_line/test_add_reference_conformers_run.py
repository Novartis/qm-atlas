"""Mocked tests for add_reference_conformers registration + entry point.

These verify that reference molecules are matched to their owning compound
directory and registered, and that ``main`` collects the input molecules and
forwards them. Molecule matching and file I/O are mocked, so nothing external
runs.

The epimer regression tests at the bottom use the public compound
L-histidine (CHEMBL17962) and its D-enantiomer: two stereoisomers that share a
single stereo-insensitive SMILES, reproducing the collision that dropped
references in the strain run.
"""

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from context import create_homedir_tmp_path  # pylint: disable=import-error
from rdkit import Chem
from rdkit.Chem import AllChem

from qm_atlas.command_line.file_interface import compound_dir
from qm_atlas.command_line.pages import add_reference_conformers as page

ETHANOL_SMILES = "CCO"  # CHEMBL545
L_HISTIDINE_SMILES = "N[C@@H](Cc1cnc[nH]1)C(=O)O"  # L-histidine, CHEMBL17962
D_HISTIDINE_SMILES = "N[C@H](Cc1cnc[nH]1)C(=O)O"  # D-histidine (enantiomer of CHEMBL17962)
HISTIDINE_FLAT_SMILES = "NC(Cc1cnc[nH]1)C(=O)O"  # stereo stripped


def _ethanol_3d(name="ethanol"):
    mol = Chem.AddHs(Chem.MolFromSmiles(ETHANOL_SMILES))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    mol.SetProp("_Name", name)
    return mol


# ---------------------------------------------------------------------------
# register_reference_conformers
# ---------------------------------------------------------------------------


def test_register_matches_single_owner_and_registers(monkeypatch, tmp_path):
    mol = _ethanol_3d()
    cpd_dir = MagicMock()
    cpd_dir.find_state_by_mol.return_value = "ethanol"  # matches (does not raise)
    cpd_dir.add_reference_input.return_value = Path("ref.sdf")
    monkeypatch.setattr(page, "_find_cpd_dirs", lambda results_dir: [cpd_dir])

    page.register_reference_conformers(tmp_path, [mol])

    cpd_dir.add_reference_input.assert_called_once()
    _args, kwargs = cpd_dir.add_reference_input.call_args
    assert kwargs["conf_id"] == 0


def test_register_skips_when_no_match(monkeypatch, tmp_path):
    mol = _ethanol_3d()
    cpd_dir = MagicMock()
    cpd_dir.find_state_by_mol.side_effect = ValueError("no match")
    monkeypatch.setattr(page, "_find_cpd_dirs", lambda results_dir: [cpd_dir])

    page.register_reference_conformers(tmp_path, [mol])

    cpd_dir.add_reference_input.assert_not_called()


def test_register_skips_when_ambiguous(monkeypatch, tmp_path):
    mol = _ethanol_3d()
    cpd_a, cpd_b = MagicMock(), MagicMock()
    cpd_a.find_state_by_mol.return_value = "ethanol"
    cpd_b.find_state_by_mol.return_value = "ethanol"  # two owners -> ambiguous
    monkeypatch.setattr(page, "_find_cpd_dirs", lambda results_dir: [cpd_a, cpd_b])

    page.register_reference_conformers(tmp_path, [mol])

    cpd_a.add_reference_input.assert_not_called()
    cpd_b.add_reference_input.assert_not_called()


def _register_smiles_state(cpd_dir, name, smiles):
    """Register a state from a SMILES string without needing 3D structures."""
    mol = Chem.MolFromSmiles(smiles)
    cpd_dir.registry_handler.register_state(
        name=name,
        smiles=compound_dir.canonical_smiles(mol),
        charge=Chem.GetFormalCharge(mol),
        input_file_path=cpd_dir.input_dir / f"{name}.sdf",
    )


def _histidine_3d(smiles, name):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    mol.SetProp("_Name", name)
    return mol


def test_reference_registers_to_exact_match_despite_epimer_sibling():
    """A reference with a perfect isomeric match is registered to its compound
    even when an epimer of the same 2D structure is a separate compound."""
    tmp_path = create_homedir_tmp_path()
    results = tmp_path / "results"
    dir_l = compound_dir.create_cpd_dir(results / "his_L", create_new=True)
    dir_d = compound_dir.create_cpd_dir(results / "his_D", create_new=True)
    _register_smiles_state(dir_l, "his_L", L_HISTIDINE_SMILES)
    _register_smiles_state(dir_d, "his_D", D_HISTIDINE_SMILES)

    page.register_reference_conformers(results, [_histidine_3d(L_HISTIDINE_SMILES, "his_L")])

    # Re-read from disk so the registry reflects the just-written reference.
    dir_l = compound_dir.create_cpd_dir(results / "his_L", create_new=False)
    dir_d = compound_dir.create_cpd_dir(results / "his_D", create_new=False)
    assert dir_l.get_reference_inputs("his_L")  # registered, not skipped
    assert not dir_d.get_reference_inputs("his_D")  # epimer sibling untouched

    shutil.rmtree(tmp_path)


def test_reference_falls_back_to_unique_compound_without_stereo():
    """With no exact match, a stereo-insensitive reference still registers to a
    constitutionally-unique compound directory."""
    tmp_path = create_homedir_tmp_path()
    results = tmp_path / "results"
    dir_l = compound_dir.create_cpd_dir(results / "his_L", create_new=True)
    _register_smiles_state(dir_l, "his_L", L_HISTIDINE_SMILES)

    page.register_reference_conformers(results, [_histidine_3d(HISTIDINE_FLAT_SMILES, "his_flat")])

    dir_l = compound_dir.create_cpd_dir(results / "his_L", create_new=False)
    assert dir_l.get_reference_inputs("his_L")

    shutil.rmtree(tmp_path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_reads_inputs_and_registers(monkeypatch, tmp_path):
    sdf = tmp_path / "refs.sdf"
    with Chem.SDWriter(str(sdf)) as writer:
        writer.write(_ethanol_3d())
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    mols = [_ethanol_3d()]
    monkeypatch.setattr(page, "apply_software_config", lambda _p: None)
    monkeypatch.setattr(page, "setup_logging", lambda **_k: None)
    monkeypatch.setattr(page.input_file, "read_all_molecules", lambda files, cfg: mols)
    registered = {}
    monkeypatch.setattr(
        page,
        "register_reference_conformers",
        lambda results_directory, reference_conformers: registered.update(
            dir=results_directory, mols=reference_conformers
        ),
    )

    config = page.ReferenceConformersInputConfig(input_file=[sdf], results_directory=results_dir)
    page.main(config=config)

    assert registered["dir"] == results_dir
    assert registered["mols"] is mols


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
