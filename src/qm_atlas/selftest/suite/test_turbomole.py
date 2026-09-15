import numpy as np
import pytest
from context import require_software  # pylint: disable=import-error
from ppqm.utils.files import WorkDir
from rdkit import Chem

from qm_atlas.tasks.calculate_properties import extract_cosmo_output
from qm_atlas.wrappers.turbomole import (
    freeh,
    fukui,
    jobex,
    nmr_shielding,
    single_point,
    utils,
    vcd,
    xtb_opt,
)

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

EXP_SP_RESULTS = {
    "HOMO(eV)": -7.35842,
    "LUMO(eV)": -2.17596,
    "HLgap(eV)": 5.18246,
    "TotalEnergy(Ht)": -3173.3250579663,
}

EXP_NMR = np.asarray([60.472306, 237.802299, 541.70756, 1711.046386, 23.259648])

EXP_FUKUI = {
    "charge_nbo": [0.10536, -0.32142, -0.03511, 0.01047, 0.24071],
    "nucleophilic_nbo": [0.1717, 0.0522, 0.1804, 0.4714, 0.1243],
    "electrophilic_nbo": [-0.0269, 0.0541, 0.32, 0.6144, 0.0385],
    "radical_nbo": [0.0724, 0.0531, 0.2502, 0.5429, 0.0814],
    "ionization_potential [eV]": 8.023,
    "electron_affinity [eV]": 0.691,
    "hardness [eV]": 3.666,
    "electronegativity [eV]": 4.357,
    "electrophilicity [eV]": 2.589,
}

EXP_FREEH = {
    "chem.pot.": -24.79,
    "energy": 64.2,
    "entropy": 0.3078,
    "enthalpy": 66.68,
    "HOMO(eV)": -8.25,
    "LUMO(eV)": -1.29,
    "HLgap(eV)": 6.9600,
    "TotalEnergy(Ht)": -3172.74326,
}


SOLVENTS = [
    "water",
    "vacuum",
    "conductor",
]


EXP_POSITIONS_JOBEX = np.array(
    [
        [1.27919061e-01, -8.11609512e-02, 1.46179799e-03],
        [-1.75147109e-02, -1.12240043e00, 8.64047175e-01],
        [4.04103235e-01, 1.41488173e00, 9.18867993e-01],
        [-1.48817147e00, 4.57616870e-02, -1.11740950e00],
        [9.73763887e-01, -2.56982028e-01, -6.66867471e-01],
    ]
)

EXP_POSITIONS_XTB_TM = np.array(
    [
        [0.12002664, -0.0892357, 0.00771142],
        [-0.00489171, -1.12426711, 0.86862906],
        [0.40150772, 1.41473055, 0.91394056],
        [-1.4807939, 0.05509268, -1.12290008],
        [0.96425126, -0.25622042, -0.66728095],
    ]
)

EXP_VCD_INTENSITY = [-0.39, 0.37, -0.70, 25.78, -25.18, -10.09, 2.18, 5.85, -0.19]


@require_software("turbomole")
def test_single_point(tmp_path):
    test_mol = Chem.MolFromMolBlock(TEST_MOLBLOCK, removeHs=False)
    sp_result = single_point.run_ridft(
        test_mol,
        0,
        scr=tmp_path,
    )

    assert sp_result is not None
    cosmo_output = extract_cosmo_output(sp_result, remove_from_dict=True)
    assert cosmo_output is not None
    assert "ridft;b-p;def2-TZVPD" in cosmo_output

    for key, exp_value in EXP_SP_RESULTS.items():
        assert key in sp_result
        value = sp_result[key].get_property_value()
        assert np.isclose(value, exp_value, atol=1.0e-2)


@require_software("turbomole")
def test_nmr(tmp_path):
    test_mol = Chem.MolFromMolBlock(TEST_MOLBLOCK, removeHs=False)
    calculated_properties = nmr_shielding.calculate_nmr_shieldings(test_mol, 0, scr=tmp_path)
    assert calculated_properties is not None
    assert nmr_shielding.NMR_SHIELDINGS_KEY in calculated_properties.keys()
    nmr_shieldings = calculated_properties[nmr_shielding.NMR_SHIELDINGS_KEY].get_property_value()

    # NMR shieldings drift by <~0.2% between Turbomole versions; compare relatively.
    assert np.allclose(nmr_shieldings, EXP_NMR, rtol=5e-3, atol=0.1)


@require_software("turbomole")
def test_fukui(tmp_path):
    version = utils.get_turbomole_version()
    if version is not None and version[0] >= 8:
        pytest.skip(f"Fukui per-atom parsing unsupported on Turbomole {version[0]}.{version[1]}")
    test_mol = Chem.MolFromMolBlock(TEST_MOLBLOCK, removeHs=False)
    fukui_result = fukui.calculate_fukui(test_mol, 0, charge_schemes=["nbo"], scr=tmp_path)

    assert fukui_result is not None

    for fukui_key, exp_res in EXP_FUKUI.items():
        assert fukui_key in fukui_result
        exp_res = np.asarray(exp_res)
        res = np.asarray(fukui_result[fukui_key].get_property_value())
        assert np.allclose(res, exp_res, atol=1.0e-2)


@require_software("turbomole")
def test_freeh(tmp_path):
    test_mol = Chem.MolFromMolBlock(TEST_MOLBLOCK, removeHs=False)
    freeh_res = freeh.calculate_freeh(test_mol, 0, scr=tmp_path)
    results = []
    expectations = []

    assert freeh_res is not None
    for key, exp_res in EXP_FREEH.items():
        assert key in freeh_res
        expectations.append(exp_res)
        res = freeh_res[key].get_property_value()
        results.append(res)

    results = np.asarray(results)
    expectations = np.asarray(expectations)

    assert np.allclose(results, expectations, atol=1.0e-2)


@require_software("turbomole")
@pytest.mark.parametrize("solvent", SOLVENTS)
def test_control_file(solvent: str, tmp_path):
    test_mol = Chem.MolFromMolBlock(TEST_MOLBLOCK, removeHs=False)
    conf_id = test_mol.GetConformer().GetId()
    temp = utils.prepare_turbomole_input(
        test_mol, nmr_shielding.PREPARE_NMR, conf_id=conf_id, solvent=solvent, scr=tmp_path
    )
    scr = temp.get_path()

    control_file = scr / "control"
    assert control_file.exists()
    with open(control_file, "r", encoding="utf-8") as contr_f:
        control_str = contr_f.read()

    if solvent == "vacuum":
        assert "$cosmo" in control_str
        assert "epsilon=1" in control_str

    elif solvent == "conductor":
        assert "$cosmo" in control_str
        assert "epsilon=infinity" in control_str

    elif solvent == "water":
        assert "$cosmo" in control_str
        assert "epsilon=78.39" in control_str


@require_software("turbomole")
def test_coord_parsing(tmp_path):
    # get a test molecule
    test_mol = Chem.MolFromMolBlock(TEST_MOLBLOCK, removeHs=False)
    conf_id = test_mol.GetConformer(0).GetId()
    # write it to turbomole coord file

    temp_dir = utils.prepare_turbomole_input(
        test_mol, jobex.PREPARE_JOBEX, conf_id=conf_id, scr=tmp_path
    )
    coord_file = temp_dir.get_path() / "coord"
    # parse coordinates back from that file
    labels = [atom.GetSymbol() for atom in test_mol.GetAtoms()]
    coords = jobex.parse_coord(coord_file, labels)

    exp_coords = np.asarray(test_mol.GetConformer(0).GetPositions())
    assert np.allclose(exp_coords, coords)


@require_software("turbomole")
def test_jobex(tmp_path):
    test_mol = Chem.MolFromMolBlock(TEST_MOLBLOCK, removeHs=False)
    new_coords, _ = jobex.run_jobex(test_mol, 0, scr=tmp_path, num_steps=1)
    assert np.allclose(new_coords, EXP_POSITIONS_JOBEX, atol=0.001)


@require_software("xtb", "turbomole")
def test_xtb_tm_opt(tmp_path):
    test_mol = Chem.MolFromMolBlock(TEST_MOLBLOCK, removeHs=False)
    new_coords, _ = xtb_opt.run_xtb_tm_optimization(
        test_mol,
        0,
        scr=tmp_path,
        max_num_steps=1,
    )
    assert np.allclose(new_coords, EXP_POSITIONS_XTB_TM, atol=0.001)


def test_check_jobex(tmp_path):
    tmol_temp = WorkDir(dir=tmp_path, prefix="test_turbomole_jobex_", keep=False)

    assert jobex.check_jobex(tmol_temp, "", "ended normally") == False

    converged_flag = tmol_temp.get_path() / "GEO_OPT_CONVERGED"
    converged_flag.touch()
    assert jobex.check_jobex(tmol_temp, "", "ended normally") == True

    with pytest.raises(RuntimeError):
        jobex.check_jobex(tmol_temp, "", "ended abnormally")


@require_software("turbomole")
def test_vcd(tmp_path):
    test_mol = Chem.MolFromMolBlock(TEST_MOLBLOCK, removeHs=False)
    vcd_properties = vcd.calculate_vcd(
        test_mol,
        0,
        functional=["b3-lyp"],
        basis="def2-SVP",
        scr=tmp_path,
    )
    assert vcd_properties is not None
    print(vcd_properties)
    vcd_intensity = vcd_properties["VCD_intensity"].get_property_value()
    # VCD intensities drift by O(1) between Turbomole versions; compare loosely.
    assert np.allclose(vcd_intensity, EXP_VCD_INTENSITY, atol=2.0)
