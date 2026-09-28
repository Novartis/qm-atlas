import pytest
from context import require_software  # pylint: disable=import-error
from rdkit import Chem
from rdkit.Chem import AllChem

from qm_atlas.wrappers import cresset


def _make_multi_conformer_mol():
    """Ethanol with three embedded conformers (conformer 0 doubles as reference)."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMultipleConfs(mol, numConfs=3, randomSeed=0xF00D)
    return mol


@require_software("cresset")
def test_align_mol_to_reference(tmp_path):
    mol = _make_multi_conformer_mol()
    reference = Chem.Mol(mol)

    aligned, rmsd_dict = cresset.align_mol_to_reference(mol, reference, ref_cid=0, scr=tmp_path)

    conf_ids = [conf.GetId() for conf in aligned.GetConformers()]
    assert conf_ids == [conf.GetId() for conf in mol.GetConformers()]
    assert set(rmsd_dict) == set(conf_ids)
    assert all(rmsd >= 0 for rmsd in rmsd_dict.values())

    # The reference conformer aligned to itself must overlay almost perfectly.
    assert rmsd_dict[0] == pytest.approx(0.0, abs=1e-2)

    # Cresset writes an overall similarity score (0-1) that we carry per conformer.
    for conf in aligned.GetConformers():
        assert conf.HasProp(cresset.SIMILARITY_CONF_PROP)
        sim = conf.GetDoubleProp(cresset.SIMILARITY_CONF_PROP)
        assert 0.0 <= sim <= 1.0
    assert aligned.GetConformer(0).GetDoubleProp(cresset.SIMILARITY_CONF_PROP) == pytest.approx(
        1.0, abs=1e-2
    )
