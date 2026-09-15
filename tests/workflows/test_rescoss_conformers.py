import context  # pylint: disable=import-error
from context import TEST_CONF_GEN_OPTIONS, require_software  # pylint: disable=import-error
from rdkit import Chem

from qm_atlas.workflows import rescoss_conformers

GLYCINE = "C(C(=O)O)N"


@require_software("xtb", "turbomole", "cosmotherm")
def test_exp_conformers(tmp_path):

    test_scr = tmp_path / "exp_conf_scr"
    test_scr.mkdir(exist_ok=False, parents=True)

    glycine = Chem.MolFromSmiles(GLYCINE)
    glycine.SetProp("Test", "test_value")
    glycine.SetProp("_Name", "glycine")
    n_cores = context.get_n_cores()

    glycine_3d = rescoss_conformers.generate_rescoss_conformers(
        glycine,
        n_cores=n_cores,
        trace_dir=test_scr,
        write_intermediates=True,
        conformer_generation_options=TEST_CONF_GEN_OPTIONS,
    )

    assert glycine_3d.GetNumConformers() > 1
    assert glycine_3d.GetProp("Test") == "test_value"
    assert glycine_3d.GetProp("_Name") == "glycine"

    sdf_files = [file for file in test_scr.iterdir() if file.suffix == ".sdf"]
    assert len(sdf_files) >= 1
