from context import require_software  # pylint: disable=import-error
from ppqm import chembridge
from rdkit import Chem

from qm_atlas.command_line.file_interface import input_check
from qm_atlas.tasks.calculate_properties import (
    XTB_SP_SOLVENT_CONFIGS,
    get_xtb_energies_and_weights,
)
from qm_atlas.tasks.protonation import MokaBlabberConfig, run_protonation
from qm_atlas.tasks.tautomer_generation import (
    TAUTOMER_CONFORMER_OPTIONS,
    XTB_SP_CONFIG,
    generate_tautomer_conformers,
    generate_tautomers,
    select_stable_tautomers,
)
from qm_atlas.utils import remove_salt

CHEMBL508094 = "C[C@H](O)[C@H](C)[C@H](N)C(=O)O"


def _make_molobj(smiles: str = CHEMBL508094) -> Chem.Mol:
    """Helper: parse, validate, and neutralize a SMILES string."""
    clean_smiles = remove_salt(smiles)
    molobj = chembridge.smiles_to_molobj(clean_smiles)
    ignore_reason = input_check.ignore_molecule(molobj)
    if ignore_reason is not None:
        raise ValueError(f"Molecule rejected: {ignore_reason}")
    return chembridge.neutralize_molobj(molobj)


@require_software("unicon")
def test_generate_tautomers(tmp_path):
    """generate_tautomers returns at least the original molecule."""
    molobj = _make_molobj()
    tautomers = generate_tautomers(molobj, scr=tmp_path)
    assert isinstance(tautomers, list)
    assert len(tautomers) >= 1
    for t in tautomers:
        assert isinstance(t, Chem.Mol)


@require_software("unicon")
def test_generate_tautomer_conformers(tmp_path):
    """generate_tautomer_conformers embeds at least one conformer per tautomer."""
    molobj = _make_molobj()
    tautomers = generate_tautomers(molobj, scr=tmp_path)
    tautomers_3d = generate_tautomer_conformers(
        tautomers,
        options=TAUTOMER_CONFORMER_OPTIONS,
        scr=tmp_path,
        show_progress=False,
    )
    assert len(tautomers_3d) == len(tautomers)
    for t in tautomers_3d:
        assert t.GetNumConformers() > 0


@require_software("unicon", "xtb")
def test_select_stable_tautomers(tmp_path):
    """select_stable_tautomers returns graph-level mols with no conformers."""
    molobj = _make_molobj()
    tautomers = generate_tautomers(molobj, scr=tmp_path)
    tautomers_3d = generate_tautomer_conformers(
        tautomers,
        options=TAUTOMER_CONFORMER_OPTIONS,
        scr=tmp_path,
        show_progress=False,
    )
    stable = select_stable_tautomers(
        tautomers_3d,
        xtb_sp_config=XTB_SP_CONFIG,
        threshold_kcal_mol=7.0,
        n_cores=1,
        scr=tmp_path,
        show_progress=False,
    )
    assert isinstance(stable, list)
    assert len(stable) >= 1
    for mol in stable:
        assert isinstance(mol, Chem.Mol)
        assert mol.GetNumConformers() == 0


@require_software("moka")
def test_run_protonation(tmp_path):
    """run_protonation returns a ProtonationResult with at least one protomer."""
    molobj = _make_molobj()
    config = MokaBlabberConfig()
    result = run_protonation(molobj, config=config, scr=tmp_path)
    assert hasattr(result, "protomers")
    assert hasattr(result, "transitions")
    assert isinstance(result.protomers, list)
    assert len(result.protomers) >= 1
    for entry in result.protomers:
        assert isinstance(entry.mol, Chem.Mol)
    # blabber does not emit pKa transitions
    assert result.transitions == []


@require_software("xtb")
def test_get_xtb_energies_and_weights(tmp_path):
    """get_xtb_energies_and_weights returns one dict per conformer with expected keys."""
    from ppqm import tasks

    mol = Chem.MolFromSmiles(CHEMBL508094)
    mol = tasks.generate_conformers(mol, n_conformers=2)
    rows = get_xtb_energies_and_weights(
        mol,
        final_xtb_sp_configs=XTB_SP_SOLVENT_CONFIGS,
        scr=tmp_path,
        show_progress=False,
    )
    assert len(rows) == mol.GetNumConformers()
    for row in rows:
        for solvent in ["water", "woctanol", "chcl3"]:
            assert f"energy_{solvent}" in row
            assert f"weight_{solvent}" in row
