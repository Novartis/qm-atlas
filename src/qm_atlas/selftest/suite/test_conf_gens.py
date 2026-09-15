import copy

import pytest
from context import require_software  # pylint: disable=import-error
from rdkit import Chem

from qm_atlas.tasks import conformer_generation
from qm_atlas.utils import conformer_is_3d

SMILES = [
    "CCC",  # CHEMBL135416
    "CCCO",  # CHEMBL14687
]

CHEMBL4566196 = "C1COCCOCCOCCO1"  # CHEMBL4566196

# A compound with a single, unassigned stereo center. Conformer generators must
# not change the molecular graph, i.e. the stereo center must remain unassigned
# after conformer generation.
UNASSIGNED_STEREO_SMILES = "CCC(C)O"  # CHEMBL45462

MOLECULES = [Chem.MolFromSmiles(smi) for smi in SMILES]


def _get_options(tmpdir):
    options = copy.deepcopy({})
    options["scr"] = tmpdir
    return options


@require_software("openeye")
@pytest.mark.parametrize("mol", MOLECULES)
def test_openeye_conformers(mol, tmpdir):

    options = _get_options(tmpdir)

    # Build molobj with conformers
    mol_3d = conformer_generation.generate_omega_conformers(mol, **options)

    # Assert that some conformers were generated
    assert mol_3d.GetNumConformers() > 0
    conf_ids = [conf.GetId() for conf in mol_3d.GetConformers()]
    for conf_id in conf_ids:
        assert conformer_is_3d(mol_3d, conf_id=conf_id)


@require_software("corina")
@pytest.mark.parametrize("mol", MOLECULES)
def test_corina_conformers(mol, tmp_path):

    # Build molobj with conformers
    mol_3d = conformer_generation.generate_corina_conformers(mol, scr=tmp_path)

    # Assert that some conformers were generated
    assert mol_3d.GetNumConformers() > 0
    conf_ids = [conf.GetId() for conf in mol_3d.GetConformers()]
    for conf_id in conf_ids:
        assert conformer_is_3d(mol_3d, conf_id=conf_id)


@require_software("macromodel")
@pytest.mark.parametrize("mol", MOLECULES)
def test_macromodel_conformers(mol, tmpdir):

    options = _get_options(tmpdir)

    # Build molobj with conformers
    mol_3d = conformer_generation.generate_macromodel_conformers(mol, **options)

    # Assert that some conformers were generated
    assert mol_3d.GetNumConformers() > 0
    conf_ids = [conf.GetId() for conf in mol_3d.GetConformers()]
    for conf_id in conf_ids:
        assert conformer_is_3d(mol_3d, conf_id=conf_id)


@require_software("moe")
@pytest.mark.parametrize("mol", MOLECULES)
def test_moe_conformers(mol, tmpdir):

    options = _get_options(tmpdir)

    # Build molobj with conformers
    mol_3d = conformer_generation.generate_moe_conformers(mol, **options)

    # Assert that some conformers were generated
    assert mol_3d.GetNumConformers() > 0
    conf_ids = [conf.GetId() for conf in mol_3d.GetConformers()]
    for conf_id in conf_ids:
        assert conformer_is_3d(mol_3d, conf_id=conf_id)


@pytest.mark.parametrize("mol", MOLECULES)
def test_rdkit_conformers(mol):

    mol_3d = conformer_generation.generate_rdkit_conformers(mol, max_conformers=20)
    # Assert that some conformers were generated
    assert mol_3d.GetNumConformers() > 0
    conf_ids = [conf.GetId() for conf in mol_3d.GetConformers()]
    for conf_id in conf_ids:
        assert conformer_is_3d(mol_3d, conf_id=conf_id)


@pytest.mark.parametrize("method", list(conformer_generation.RDKIT_METHODS.keys()))
def test_rdkit_conformers_methods(method):

    mol = MOLECULES[0]

    # Build molobj with conformers
    mol_3d = conformer_generation.generate_rdkit_conformers(mol, method=method, max_conformers=20)

    # Assert that some conformers were generated
    assert mol_3d.GetNumConformers() > 0
    conf_ids = [conf.GetId() for conf in mol_3d.GetConformers()]
    for conf_id in conf_ids:
        assert conformer_is_3d(mol_3d, conf_id=conf_id)


def test_rdkit_conformers_invalid_method():

    mol = MOLECULES[0]
    non_existing_method = "ETKDG_v25.2"
    with pytest.raises(ValueError):
        _ = conformer_generation.generate_rdkit_conformers(
            mol, method=non_existing_method, max_conformers=20
        )


@require_software("macromodel")
def test_generate_macrocycle_conformers(tmpdir):

    options = _get_options(tmpdir)
    options["max_conformers"] = 10
    mol = Chem.MolFromSmiles(CHEMBL4566196)
    mol_3d = conformer_generation.generate_macrocycle_conformers(mol, **options)

    # Assert that some conformers were generated
    assert mol_3d.GetNumConformers() > 0
    conf_ids = [conf.GetId() for conf in mol_3d.GetConformers()]
    for conf_id in conf_ids:
        assert conformer_is_3d(mol_3d, conf_id=conf_id)


@require_software("macromodel")
def test_generate_macrocycle_conformers_non_macrocycle(tmpdir):

    options = _get_options(tmpdir)
    mol = MOLECULES[0]  # non-macrocycle molecule
    # assert that a RuntimeError is raised when trying to generate macrocycle conformers for a non-macrocycle molecule
    with pytest.raises(RuntimeError):
        _ = conformer_generation.generate_macrocycle_conformers(mol, **options)


@require_software("openeye")
def test_generate_omega_conformers_macrocycle(tmpdir):

    options = _get_options(tmpdir)
    mol = Chem.MolFromSmiles(CHEMBL4566196)
    mol_3d = conformer_generation.generate_omega_conformers(mol, **options)

    # Assert that some conformers were generated
    assert mol_3d.GetNumConformers() > 0
    conf_ids = [conf.GetId() for conf in mol_3d.GetConformers()]
    for conf_id in conf_ids:
        assert conformer_is_3d(mol_3d, conf_id=conf_id)


def test_generate_multi_conformer_set(tmp_path):

    # enable "write_intermediates"
    test_scr = tmp_path / "test_scr"
    test_scr.mkdir(parents=True, exist_ok=False)

    test_mol = MOLECULES[0]
    conf_gens = [conformer_generation.RdkitOptions()]

    mol_3d = conformer_generation.generate_joint_conformer_set(
        test_mol,
        conf_gens,
        trace_dir=test_scr,
        write_intermediates=True,
    )

    assert mol_3d.GetNumConformers() > 0
    conf_ids = [conf.GetId() for conf in mol_3d.GetConformers()]
    for conf_id in conf_ids:
        assert conformer_is_3d(mol_3d, conf_id=conf_id)

    sdf_files = [file for file in test_scr.iterdir() if file.suffix == ".sdf"]
    assert len(sdf_files) == len(conf_gens) + 1


def _num_unassigned_stereocenters(mol: Chem.Mol) -> int:
    """Return the number of unassigned stereo centers in the molecular graph,
    ignoring any 3D coordinates that may be present."""

    graph_mol = Chem.Mol(mol)
    graph_mol.RemoveAllConformers()
    centers = Chem.FindMolChiralCenters(
        graph_mol, includeUnassigned=True, useLegacyImplementation=False
    )
    return sum(1 for _, label in centers if label == "?")


def test_unassigned_stereo_smiles_sanity():
    # The input compound must have exactly one unassigned stereo center.
    mol = Chem.MolFromSmiles(UNASSIGNED_STEREO_SMILES)
    assert _num_unassigned_stereocenters(mol) == 1


@require_software("openeye")
def test_openeye_preserves_unassigned_stereo(tmp_path):
    mol = Chem.MolFromSmiles(UNASSIGNED_STEREO_SMILES)
    mol_3d = conformer_generation.generate_omega_conformers(mol, scr=tmp_path)

    assert mol_3d.GetNumConformers() > 0
    assert _num_unassigned_stereocenters(mol_3d) == 1


@require_software("corina")
def test_corina_preserves_unassigned_stereo(tmp_path):
    mol = Chem.MolFromSmiles(UNASSIGNED_STEREO_SMILES)
    mol_3d = conformer_generation.generate_corina_conformers(mol, scr=tmp_path)

    assert mol_3d.GetNumConformers() > 0
    assert _num_unassigned_stereocenters(mol_3d) == 1


@require_software("macromodel")
def test_macromodel_preserves_unassigned_stereo(tmp_path):
    mol = Chem.MolFromSmiles(UNASSIGNED_STEREO_SMILES)
    mol_3d = conformer_generation.generate_macromodel_conformers(mol, scr=tmp_path)

    assert mol_3d.GetNumConformers() > 0
    assert _num_unassigned_stereocenters(mol_3d) == 1


@require_software("moe")
def test_moe_preserves_unassigned_stereo(tmp_path):
    mol = Chem.MolFromSmiles(UNASSIGNED_STEREO_SMILES)
    mol_3d = conformer_generation.generate_moe_conformers(mol, scr=tmp_path)

    assert mol_3d.GetNumConformers() > 0
    assert _num_unassigned_stereocenters(mol_3d) == 1
