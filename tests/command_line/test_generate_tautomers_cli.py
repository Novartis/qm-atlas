"""Tests for the generate_tautomers CLI page.

The scoring workflow (`workflows.score_tautomers.generate_tautomers`) is
monkey-patched in every test, so no external QM software is required.
"""

import shutil
from unittest.mock import patch

import pytest
from context import create_homedir_tmp_path  # pylint: disable=import-error
from rdkit import Chem

# ===========================================================================
# SMILES Test Strings - Global Constants
# ===========================================================================
SMILES_ETHANE = "CC"
SMILES_ACETAMIDE = "O=C(N)C"
SMILES_ACETAMIDE_TAUTOMER_1 = "OC(=N)C"
SMILES_ACETAMIDE_TAUTOMER_2 = "OC(N)=C"
SMILES_ETHANOL = "CCO"

from qm_atlas.command_line.base_config import CalculationInput
from qm_atlas.command_line.file_interface.compound_dir import create_cpd_dir
from qm_atlas.command_line.file_interface.input_file import InputConfig
from qm_atlas.command_line.pages import generate_tautomers
from qm_atlas.command_line.pages.generate_tautomers import (
    GenerateTautomersInterfaceConfig,
    GenerateTautomersOptions,
    GenerateTautomersSubmissionConfig,
    run_local,
)
from qm_atlas.command_line.pages.read_input import ReadInputConfig
from qm_atlas.command_line.pages.read_input import run as read_input_run


@pytest.fixture
def temp_work_dir():
    tmp_path = create_homedir_tmp_path()
    yield tmp_path
    if tmp_path.exists():
        # ignore_errors: on NFS the per-task log FileHandler left open by
        # run_local() causes .nfsXXXX silly-rename leftovers that break rmdir.
        shutil.rmtree(tmp_path, ignore_errors=True)


def _make_compound_dir(temp_work_dir, smiles: str = SMILES_ETHANE, name: str = "ethane"):
    """Create a results directory with a single compound + single input state."""
    results_dir = temp_work_dir / "results"
    csv_file = temp_work_dir / "in.csv"
    csv_file.write_text(f"name,smiles\n{name},{smiles}\n")
    read_input_run(
        ReadInputConfig(
            input_file=[csv_file],
            results_directory=results_dir,
            input_config=InputConfig(),
        )
    )
    return results_dir, results_dir / name


def _build_config(cpd_dir_path) -> GenerateTautomersInterfaceConfig:
    return GenerateTautomersInterfaceConfig(
        input=CalculationInput(
            results_directory=cpd_dir_path.parent,
            compound_directories=[cpd_dir_path],
        ),
        submission_config=GenerateTautomersSubmissionConfig(submit=False),
        generate_tautomers_options=GenerateTautomersOptions(),
    )


def _mols_from_smiles(*smiles_list: str) -> list[Chem.Mol]:
    return [Chem.MolFromSmiles(s) for s in smiles_list]


def test_new_tautomers_are_registered_as_states(temp_work_dir):
    """Survived tautomers are added to the compound dir with `_T{n}` names."""
    _, cpd_dir_path = _make_compound_dir(temp_work_dir, smiles=SMILES_ACETAMIDE, name="acetamide")

    new_tautomers = _mols_from_smiles(SMILES_ACETAMIDE_TAUTOMER_1, SMILES_ACETAMIDE_TAUTOMER_2)

    with patch.object(
        generate_tautomers.score_tautomers,
        "generate_tautomers",
        return_value=new_tautomers,
    ):
        run_local(_build_config(cpd_dir_path))

    cpd = create_cpd_dir(cpd_dir_path, create_new=False)
    names = set(cpd.registry_handler.get_registered_names())
    assert "acetamide" in names
    assert "acetamide_T1" in names
    assert "acetamide_T2" in names

    assert (cpd_dir_path / "input" / "acetamide_T1.sdf").is_file()
    assert (cpd_dir_path / "input" / "acetamide_T2.sdf").is_file()


def test_duplicate_smiles_is_skipped(temp_work_dir):
    """A returned tautomer matching an already-registered SMILES is not added."""
    parent_smiles = SMILES_ACETAMIDE
    _, cpd_dir_path = _make_compound_dir(temp_work_dir, smiles=parent_smiles, name="acetamide")

    # First "tautomer" is identical to the parent (canonical SMILES match) and
    # must be skipped; second is genuinely new.
    duplicate = Chem.MolFromSmiles(parent_smiles)
    fresh = Chem.MolFromSmiles(SMILES_ACETAMIDE_TAUTOMER_1)

    with patch.object(
        generate_tautomers.score_tautomers,
        "generate_tautomers",
        return_value=[duplicate, fresh],
    ):
        run_local(_build_config(cpd_dir_path))

    cpd = create_cpd_dir(cpd_dir_path, create_new=False)
    names = set(cpd.registry_handler.get_registered_names())
    assert names == {"acetamide", "acetamide_T1"}


def test_name_collision_skips_to_next_index(temp_work_dir):
    """If `{parent}_T1` is already taken, the new tautomer becomes `_T2`."""
    _, cpd_dir_path = _make_compound_dir(temp_work_dir, smiles=SMILES_ACETAMIDE, name="acetamide")

    # Register a placeholder state directly via the registry (without writing an
    # input SDF file) so worker-path iteration does not pick it up as an
    # additional state to process.
    cpd = create_cpd_dir(cpd_dir_path, create_new=False)
    cpd.registry_handler.register_state(
        name="acetamide_T1",
        smiles=SMILES_ETHANOL,
        charge=0,
        input_file_path=cpd_dir_path / "input" / "acetamide_T1.sdf",
        description="placeholder",
    )

    new_tautomer = Chem.MolFromSmiles(SMILES_ACETAMIDE_TAUTOMER_1)
    with patch.object(
        generate_tautomers.score_tautomers,
        "generate_tautomers",
        return_value=[new_tautomer],
    ):
        run_local(_build_config(cpd_dir_path))

    cpd = create_cpd_dir(cpd_dir_path, create_new=False)
    names = set(cpd.registry_handler.get_registered_names())
    assert "acetamide_T2" in names
    # _T1 still belongs to the placeholder
    assert cpd.registry_handler.get_entry("acetamide_T1").smiles == "CCO"


def test_empty_result_list_is_a_noop(temp_work_dir):
    _, cpd_dir_path = _make_compound_dir(temp_work_dir, smiles=SMILES_ACETAMIDE, name="acetamide")

    with patch.object(
        generate_tautomers.score_tautomers,
        "generate_tautomers",
        return_value=[],
    ):
        run_local(_build_config(cpd_dir_path))

    cpd = create_cpd_dir(cpd_dir_path, create_new=False)
    assert set(cpd.registry_handler.get_registered_names()) == {"acetamide"}


# ===========================================================================
# YAML config round-trip
# ===========================================================================


def test_config_roundtrip(config_roundtrip):
    """Non-default scoring settings + mixed optimization backends round-trip."""
    _cfg, dump = config_roundtrip(
        generate_tautomers.load_config,
        {
            "generate_tautomers_options": {
                "energy_threshold": 4.5,
                "solvent_name": "methanol",
                "optimization_config": [{"backend": "xtb", "method": "2", "opt_level": "tight"}],
                "final_optimization_config": [{"backend": "xtb_turbomole", "basis": "def2-SVP"}],
            }
        },
    )
    opts = dump["generate_tautomers_options"]
    assert opts["energy_threshold"] == 4.5
    assert opts["optimization_config"][0]["backend"] == "xtb"
    assert opts["final_optimization_config"][0]["backend"] == "xtb_turbomole"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
