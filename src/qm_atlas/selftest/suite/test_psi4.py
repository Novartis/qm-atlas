import json
import shlex
from unittest.mock import Mock

import numpy as np
import pytest
from context import require_software  # pylint: disable=import-error
from pydantic import ValidationError
from rdkit import Chem
from rdkit.Chem import AllChem

from qm_atlas.tasks import calculate_properties
from qm_atlas.tasks.calculate_properties import Psi4SinglePointOptions, calculate_property
from qm_atlas.wrappers import psi4


def molecule(smiles="O"):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    return mol


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"method": "hf"},
        {"method": "hf", "basis": ""},
        {"method": "hf", "basis": "sto-3g", "memory_mb": 0},
        {"method": "hf", "basis": "sto-3g", "multiplicity": 0},
        {"method": "hf", "basis": "sto-3g", "solvent": "water"},
    ],
)
def test_single_point_options_reject_invalid_settings(options):
    with pytest.raises(ValidationError):
        Psi4SinglePointOptions(**options)


def test_single_point_property_prefix():
    options = Psi4SinglePointOptions(method="hf", basis="sto-3g", property_prefix="comparison_")
    assert options.get_property_names() == {"total_energy": "comparison_total_energy"}


def test_selected_conformer_units_and_unchanged_geometry():
    mol = molecule()
    second = Chem.Conformer(mol.GetConformer())
    second.SetId(7)
    second.SetAtomPosition(0, (1.0, 2.0, 3.0))
    mol.AddConformer(second, assignId=False)
    original = mol.GetConformer(7).GetPositions().copy()
    data = psi4.build_input(mol, 7, "pbe0", "def2-SVP")
    assert data["driver"] == "energy"
    assert data["model"] == {"method": "pbe0", "basis": "def2-SVP"}
    assert data["molecule"]["symbols"] == ["O", "H", "H"]
    assert np.asarray(data["molecule"]["geometry"]).reshape(-1, 3) == pytest.approx(
        original / 0.529177210544
    )
    assert data["molecule"]["fix_com"] is True
    assert data["molecule"]["fix_orientation"] is True
    assert np.array_equal(mol.GetConformer(7).GetPositions(), original)


@pytest.mark.parametrize(
    "smiles,method,multiplicity,reference,charge,spin",
    [
        ("O", "hf", None, "rhf", 0, 1),
        ("O", "pbe0", None, "rks", 0, 1),
        ("[CH3]", "hf", None, "uhf", 0, 2),
        ("[CH3]", "pbe0", None, "uks", 0, 2),
        ("[NH4+]", "hf", None, "rhf", 1, 1),
        ("[CH2]", "hf", 1, "rhf", 0, 1),
    ],
)
def test_charge_spin_and_reference(smiles, method, multiplicity, reference, charge, spin):
    data = psi4.build_input(molecule(smiles), 0, method, "sto-3g", multiplicity)
    assert data["molecule"]["molecular_charge"] == charge
    assert data["molecule"]["molecular_multiplicity"] == spin
    assert data["keywords"]["reference"] == reference


def test_explicit_reference():
    data = psi4.build_input(molecule("[CH3]"), 0, "hf", "sto-3g", reference="rohf")
    assert data["keywords"]["reference"] == "rohf"


@pytest.mark.parametrize("multiplicity", [0, 2, 99])
def test_invalid_spin(multiplicity):
    with pytest.raises(ValueError, match="multiplicity"):
        psi4.build_input(molecule(), 0, "hf", "sto-3g", multiplicity)


def test_open_shell_restricted_reference_rejected():
    with pytest.raises(ValueError, match="open-shell"):
        psi4.build_input(molecule("[CH3]"), 0, "hf", "sto-3g", reference="rhf")


def test_missing_hydrogens_rejected():
    mol = Chem.MolFromSmiles("O")
    with pytest.raises(ValueError, match="explicit hydrogen"):
        psi4.build_input(mol, 0, "hf", "sto-3g")


def test_missing_conformer_rejected():
    with pytest.raises(ValueError, match="does not exist"):
        psi4.build_input(molecule(), 4, "hf", "sto-3g")


@pytest.mark.parametrize("bad_coordinates", [False, True])
def test_invalid_coordinates_rejected(bad_coordinates):
    mol = molecule()
    if bad_coordinates:
        mol.GetConformer().SetAtomPosition(0, (float("nan"), 0, 0))
    else:
        mol.GetConformer().Set3D(False)
    with pytest.raises(ValueError, match="finite 3D"):
        psi4.build_input(mol, 0, "hf", "sto-3g")


@pytest.mark.parametrize(
    "content",
    [
        None,
        "not json",
        "[]",
        "{}",
        '{"success": false, "error": {"error_message": "SCF failed"}}',
        '{"success": true}',
        '{"success": true, "return_result": NaN}',
        '{"success": true, "return_result": Infinity}',
        '{"success": true, "return_result": [1, 2]}',
        '{"success": true, "return_result": true}',
        '{"success": true, "return_result": "-1.0"}',
    ],
)
def test_invalid_results(tmp_path, content):
    path = tmp_path / "result.json"
    if content is not None:
        path.write_text(content)
    with pytest.raises(RuntimeError):
        psi4.read_output(path)


def test_valid_energy(tmp_path):
    path = tmp_path / "result.json"
    path.write_text('{"success": true, "return_result": -75.0}')
    assert psi4.read_output(path)["total_energy"].get_property_value() == -75.0


@pytest.fixture
def fake_environment(monkeypatch):
    manager = Mock()
    manager.get_command.return_value = "/opt/Conda Environments/psi4/bin/psi4"
    manager.get_run_environment.return_value = {"PATH": "/fake/bin"}
    monkeypatch.setattr(psi4.SOFTWARE_CONFIG, "get_environment_manager", lambda: manager)
    return manager


@pytest.mark.parametrize("keep", [False, True])
def test_execution_resources_and_scratch(tmp_path, monkeypatch, fake_environment, keep):
    paths = []

    def run(command, *, env, cwd):
        args = shlex.split(command)
        assert args[0] == fake_environment.get_command.return_value
        assert "--qcschema" in args
        assert args[args.index("-n") + 1] == "3"
        assert args[args.index("--memory") + 1] == "768 MB"
        assert args[args.index("-s") + 1] == str(cwd)
        assert ("--messy" in args) == keep
        assert env == {"PATH": "/fake/bin"}
        data = json.loads((cwd / "input.json").read_text())
        assert data["model"]["method"] == "pbe0"
        (cwd / "result.json").write_text('{"success": true, "return_result": -75.0}')
        paths.append(cwd)
        return "calculated", ""

    monkeypatch.setattr(psi4, "run_command", run)
    for _ in range(2):
        result = psi4.calculate_single_point(
            molecule(),
            0,
            "pbe0",
            "sto-3g",
            scr=tmp_path,
            n_cores=3,
            memory_mb=768,
            keep_files=keep,
        )
        assert result["total_energy"].get_property_value() == -75.0
    assert paths[0] != paths[1]
    assert all(path.exists() == keep for path in paths)
    if keep:
        assert (paths[0] / "stdout.log").read_text() == "calculated"
    fake_environment.get_run_environment.assert_called_with("psi4", n_cores=3)


def test_process_failure_and_task_suppression(tmp_path, monkeypatch, fake_environment):
    def fail(*args, **kwargs):
        raise RuntimeError("Executable failed")

    monkeypatch.setattr(psi4, "run_command", fail)
    mol = molecule()
    options = calculate_properties.Psi4SinglePointOptions(method="hf", basis="sto-3g")
    assert calculate_properties.calculate_property(mol, 0, options, scr=tmp_path) is None
    with pytest.raises(RuntimeError, match="Executable failed"):
        calculate_properties.calculate_property(
            mol, 0, options, scr=tmp_path, suppress_errors=False
        )


def test_disabled_software_failure(tmp_path, fake_environment):
    fake_environment.get_command.side_effect = OSError("Software environment not available")
    with pytest.raises(RuntimeError, match="not available"):
        psi4.calculate_single_point(molecule(), 0, "hf", "sto-3g", scr=tmp_path)


@require_software("psi4")
@pytest.mark.parametrize(
    "symbols,method,expected",
    [
        (["H", "H"], "hf", -1.116783318),
        (["H", "H"], "pbe0", -1.154371810),
        (["H"], "hf", -0.466581850),
    ],
)
def test_psi4_energy(symbols, method, expected, tmp_path):
    # H2 at 0.74 Angstrom and atomic H, using explicit atoms throughout.
    editable = Chem.RWMol()
    for symbol in symbols:
        atom = Chem.Atom(symbol)
        atom.SetNoImplicit(True)
        editable.AddAtom(atom)
    if len(symbols) == 2:
        editable.AddBond(0, 1, Chem.BondType.SINGLE)
    mol = editable.GetMol()
    Chem.SanitizeMol(mol)
    conformer = Chem.Conformer(len(symbols))
    for idx in range(len(symbols)):
        conformer.SetAtomPosition(idx, (0, 0, idx * 0.74))
    conformer.Set3D(True)
    mol.AddConformer(conformer)
    original = mol.GetConformer().GetPositions().copy()
    props = calculate_property(
        mol,
        0,
        Psi4SinglePointOptions(method=method, basis="sto-3g"),
        scr=tmp_path,
        suppress_errors=False,
    )
    assert props["psi4_total_energy"].get_property_value() == pytest.approx(expected, abs=2e-6)
    assert np.array_equal(mol.GetConformer().GetPositions(), original)
