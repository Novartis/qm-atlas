import json

import numpy as np
import pytest
from context import require_software  # pylint: disable=import-error
from ppqm.utils.files import WorkDir
from rdkit import Chem

from qm_atlas.wrappers import xtb

TEST_MOLBLOCK = """test molecule
     RDKit          3D

  5  4  0  0  0  0  0  0  0  0999 V2000
    0.1279   -0.0772   -0.0040 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0216   -1.1047    0.8417 F   0  0  0  0  0  0  0  0  0  0  0  0
    0.3965    1.4141    0.9229 Cl  0  0  0  0  0  0  0  0  0  0  0  0
   -1.4841    0.0407   -1.0993 Br  0  0  0  0  0  0  0  0  0  0  0  0
    0.9814   -0.2728   -0.6612 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
  1  4  1  0
  1  5  1  0
M  END
"""

# Expected values from xtb calculations on test molecule
EXP_SP_RESULTS = {
    "total energy": -15.99589516,
    "HOMO-LUMO gap / eV": 4.69397778,
    "electronic energy": -16.03122411,
}

EXP_OPT_COORDINATES = np.array(
    [
        [0.12310027, -0.07769139, 0.00135135],
        [-0.01746619, -1.10716848, 0.84353837],
        [0.40056634, 1.41450131, 0.92230253],
        [-1.47746794, 0.03991191, -1.10440973],
        [0.97136752, -0.26945335, -0.66268251],
    ]
)


def test_get_num_unpaired_electrons_closed_shell():
    mol = Chem.MolFromSmiles("C")
    assert xtb.get_num_unpaired_electrons(mol) == 0


def test_get_num_unpaired_electrons_radical():
    mol = Chem.MolFromSmiles("[CH3]")
    assert xtb.get_num_unpaired_electrons(mol) == 1


def test_get_num_unpaired_electrons_biradical_like():
    mol = Chem.MolFromSmiles("[CH]")
    assert xtb.get_num_unpaired_electrons(mol) == 3


def test_write_xcontrol_file_optimization_template(tmp_path):
    mol = Chem.MolFromMolBlock(TEST_MOLBLOCK, removeHs=False)
    temp = WorkDir(dir=tmp_path, prefix="test_xcontrol_", keep=False)

    xcontrol_path = xtb.write_xcontrol_file(
        mol,
        temp,
        xtb.XCONTROL_TEMPLATE_OPT,
        max_num_steps=100,
        opt_level="tight",
    )

    assert xcontrol_path.exists()
    content = xcontrol_path.read_text()
    assert "$chrg 0" in content
    assert "$spin 0" in content
    assert "$opt" in content
    assert "maxcycle=100" in content
    assert "optlevel=tight" in content


def test_write_xcontrol_file_charged_molecule(tmp_path):
    mol = Chem.MolFromSmiles("[NH4+]")
    temp = WorkDir(dir=tmp_path, prefix="test_xcontrol_charged_", keep=False)

    xcontrol_path = xtb.write_xcontrol_file(
        mol,
        temp,
        xtb.XCONTROL_TEMPLATE_SP,
    )

    content = xcontrol_path.read_text()
    assert "$chrg 1" in content


def test_write_xcontrol_file_radical_molecule(tmp_path):
    mol = Chem.MolFromSmiles("[CH3]")
    temp = WorkDir(dir=tmp_path, prefix="test_xcontrol_radical_", keep=False)

    xcontrol_path = xtb.write_xcontrol_file(
        mol,
        temp,
        xtb.XCONTROL_TEMPLATE_SP,
    )

    content = xcontrol_path.read_text()
    assert "$spin 1" in content


def test_write_xtb_input_creates_files(tmp_path):
    mol = Chem.MolFromMolBlock(TEST_MOLBLOCK, removeHs=False)

    temp, xyz_input, xcontrol_input = xtb.write_xtb_input(
        mol,
        0,
        xtb.XCONTROL_TEMPLATE_SP,
        scr=tmp_path,
    )

    assert xyz_input.exists()
    assert xcontrol_input.exists()
    assert temp.get_path().exists()


def test_write_xtb_input_with_constrain(tmp_path):
    mol = Chem.MolFromMolBlock(TEST_MOLBLOCK, removeHs=False)

    temp, _, xcontrol_input = xtb.write_xtb_input(
        mol,
        0,
        xtb.XCONTROL_TEMPLATE_OPT,
        scr=tmp_path,
        constrain=True,
        torsions_fixed=True,
        force_constant=0.001,
    )

    assert temp.get_path().exists()
    content = xcontrol_input.read_text()
    assert "$constrain" in content
    assert "all torsions=true" in content


def test_check_xtb_run_normal_termination():
    stdout = "this is normal output\nnormal termination of xtb\n"
    xtb.check_xtb_run(stdout, "")


def test_read_xtb_output_json_output(tmp_path):
    json_data = {
        "total energy": -15.99589516,
        "HOMO-LUMO gap / eV": 4.69397778,
        "electronic energy": -16.03122411,
    }

    temp = WorkDir(dir=tmp_path, prefix="test_read_output_", keep=False)
    json_file = temp.get_path() / xtb.JSON_OUTPUT_FILENAME
    json_file.write_text(json.dumps(json_data))

    results = xtb.read_xtb_output(temp, "", "", read_json=True)

    assert "total energy" in results
    assert "HOMO-LUMO gap / eV" in results
    assert results["total energy"].get_property_value() == -15.99589516


def test_get_optimization_converged_true(tmp_path):
    temp = WorkDir(dir=tmp_path, prefix="test_converged_", keep=False)
    opt_flag = temp.get_path() / xtb.OPTIMIZATION_CONVERGED_FILE
    opt_flag.touch()

    is_converged = xtb.get_optimization_converged(temp, "", "")
    assert is_converged is True


def test_get_optimization_converged_false(tmp_path):
    temp = WorkDir(dir=tmp_path, prefix="test_not_converged_", keep=False)
    is_converged = xtb.get_optimization_converged(temp, "", "")
    assert is_converged is False


@require_software("xtb")
def test_single_point_basic(tmp_path):
    mol = Chem.MolFromMolBlock(TEST_MOLBLOCK, removeHs=False)
    results = xtb.calculate_single_point(mol, 0, scr=tmp_path)

    assert results is not None
    assert "total energy" in results
    assert "HOMO-LUMO gap / eV" in results
    assert "dipole / a.u." in results


@require_software("xtb")
def test_single_point_energy_value(tmp_path):
    mol = Chem.MolFromMolBlock(TEST_MOLBLOCK, removeHs=False)
    results = xtb.calculate_single_point(mol, 0, scr=tmp_path)

    energy = results["total energy"].get_property_value()
    assert np.isclose(energy, EXP_SP_RESULTS["total energy"], atol=0.01)


@require_software("xtb")
def test_single_point_bond_orders(tmp_path):
    mol = Chem.MolFromMolBlock(TEST_MOLBLOCK, removeHs=False)
    results = xtb.calculate_single_point(mol, 0, scr=tmp_path)

    assert "bondorders" in results
    bond_orders = results["bondorders"].get_property_value()
    assert len(bond_orders) == 4


@require_software("xtb")
def test_single_point_with_solvation(tmp_path):
    mol = Chem.MolFromMolBlock(TEST_MOLBLOCK, removeHs=False)
    results = xtb.calculate_single_point(
        mol, 0, scr=tmp_path, solvation_model="alpb", solvent="water"
    )

    assert "total energy" in results
    assert results["total energy"].get_property_value() is not None


@require_software("xtb")
def test_single_point_solvation_requires_solvent(tmp_path):
    mol = Chem.MolFromMolBlock(TEST_MOLBLOCK, removeHs=False)
    with pytest.raises(ValueError):
        xtb.calculate_single_point(mol, 0, scr=tmp_path, solvation_model="alpb", solvent=None)


@require_software("xtb")
def test_optimize_geometry_basic(tmp_path):
    mol = Chem.MolFromMolBlock(TEST_MOLBLOCK, removeHs=False)
    opt_coords, opt_results = xtb.optimize_geometry(mol, 0, scr=tmp_path, max_num_steps=1)

    assert opt_coords is not None
    assert opt_coords.shape == (mol.GetNumAtoms(), 3)
    assert opt_results is not None


@require_software("xtb")
def test_optimize_geometry_coordinates(tmp_path):
    mol = Chem.MolFromMolBlock(TEST_MOLBLOCK, removeHs=False)
    opt_coords, _ = xtb.optimize_geometry(mol, 0, scr=tmp_path, max_num_steps=1)

    assert np.allclose(opt_coords, EXP_OPT_COORDINATES, atol=0.01)


@require_software("xtb")
def test_optimize_geometry_converged(tmp_path):
    mol = Chem.MolFromMolBlock(TEST_MOLBLOCK, removeHs=False)
    _, opt_results = xtb.optimize_geometry(mol, 0, scr=tmp_path, max_num_steps=1)
    assert opt_results is not None
    assert xtb.OPTIMIZATION_CONVERGED_KEY in opt_results
    is_converged = opt_results[xtb.OPTIMIZATION_CONVERGED_KEY].get_property_value()
    assert not is_converged  # The optimization should not converge in 1 step for this molecule.


@require_software("xtb")
def test_optimize_geometry_with_solvation(tmp_path):
    mol = Chem.MolFromMolBlock(TEST_MOLBLOCK, removeHs=False)
    opt_coords, _ = xtb.optimize_geometry(
        mol,
        0,
        scr=tmp_path,
        max_num_steps=1,
        solvation_model="alpb",
        solvent="water",
    )

    assert opt_coords is not None
    assert opt_coords.shape == (mol.GetNumAtoms(), 3)


@require_software("xtb")
def test_optimize_geometry_invalid_solvent_without_model(tmp_path):
    mol = Chem.MolFromMolBlock(TEST_MOLBLOCK, removeHs=False)
    with pytest.raises(ValueError):
        xtb.optimize_geometry(
            mol,
            0,
            scr=tmp_path,
            solvation_model="alpb",
            solvent=None,
        )


def test_read_properties_from_stdout_maps_keys(monkeypatch):
    """ppqm summary-table keys are mapped to the canonical JSON names/types."""
    monkeypatch.setattr(
        xtb.ppqm_xtb,
        "read_properties_sp",
        lambda lines: {
            "total_energy": -15.99589516,
            "homo_lumo_gap": 4.69397778,
            "scc_energy": -16.03122411,
            "dipole": 0.1234,  # not part of the canonical map -> dropped
        },
    )

    props = xtb.read_properties_from_stdout("some stdout", "")

    assert set(props) == {"total energy", "HOMO-LUMO gap / eV", "electronic energy"}
    assert props["total energy"].get_property_value() == -15.99589516
    assert props["HOMO-LUMO gap / eV"].get_property_value() == 4.69397778
    assert props["electronic energy"].get_property_value() == -16.03122411


def test_read_properties_from_stdout_raises_when_unparsable(monkeypatch):
    """A RuntimeError is raised when no properties can be parsed from stdout."""
    monkeypatch.setattr(xtb.ppqm_xtb, "read_properties_sp", lambda lines: None)

    with pytest.raises(RuntimeError):
        xtb.read_properties_from_stdout("", "")


@require_software("xtb")
def test_single_point_no_json_falls_back_to_stdout(tmp_path):
    """Running without --json still yields properties, parsed from stdout."""
    mol = Chem.MolFromMolBlock(TEST_MOLBLOCK, removeHs=False)
    results = xtb.calculate_single_point(mol, 0, scr=tmp_path, use_json=False)

    assert "total energy" in results
    assert np.isclose(
        results["total energy"].get_property_value(),
        EXP_SP_RESULTS["total energy"],
        atol=0.01,
    )
    assert "HOMO-LUMO gap / eV" in results
    assert "electronic energy" in results
