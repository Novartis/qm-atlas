import json
import math
import shlex
from pathlib import Path
from typing import Literal

import numpy as np
from ppqm.utils.files import WorkDir
from rdkit import Chem
from scipy.constants import physical_constants

from qm_atlas.constants import DEFAULT_SCR
from qm_atlas.software_environment import SOFTWARE_CONFIG
from qm_atlas.tasks.common import ScalarProperty
from qm_atlas.wrappers.common import run_command

Reference = Literal["rhf", "uhf", "rohf", "rks", "uks"]
ENERGY_KEY = "total_energy"
ANGSTROM_TO_BOHR = 1.0 / (physical_constants["Bohr radius"][0] * 1e10)


def build_input(
    mol: Chem.Mol,
    conf_id: int,
    method: str,
    basis: str,
    multiplicity: int | None = None,
    reference: Reference | None = None,
) -> dict:
    """Builds the QCSchema input for a Psi4 single-point energy calculation.

    Args:
        mol (Chem.Mol):
            RDKit molecule with explicit hydrogen atoms and finite 3D coordinates.
            The molecule and its conformers are not modified.
        conf_id (int):
            The conformer ID to consider.
        method (str):
            Psi4 calculation method, e.g. "hf" or "pbe0".
        basis (str):
            Basis set for the calculation, e.g. "def2-SVP".
        multiplicity (int | None, optional):
            Spin multiplicity. Defaults to None, which uses the radical-electron
            count plus one. Override this high-spin assumption for other states.
        reference (Reference | None, optional):
            SCF reference: "rhf", "uhf", "rohf", "rks", or "uks".
            Defaults to None, which selects RHF/UHF for HF or SCF methods and
            RKS/UKS for DFT, depending on multiplicity.

    Returns:
        dict: A QCSchema v1 energy input containing the model, molecular charge,
        spin multiplicity, and atom coordinates converted from Angstrom to Bohr.
        Molecular translation and reorientation are disabled.

    Raises:
        ValueError: If method or basis is empty, the molecule is empty or lacks
            explicit hydrogen atoms, the conformer is missing or has invalid 3D
            coordinates, or the multiplicity or reference is incompatible.
    """
    if not method.strip() or not basis.strip():
        raise ValueError("Psi4 method and basis must be non-empty")
    if not mol.GetNumAtoms() or any(atom.GetTotalNumHs() for atom in mol.GetAtoms()):
        raise ValueError("Psi4 requires a non-empty molecule with explicit hydrogen atoms")
    try:
        conformer = mol.GetConformer(conf_id)
    except ValueError as exc:
        raise ValueError(f"Psi4 conformer {conf_id} does not exist") from exc
    coordinates = conformer.GetPositions()
    if not conformer.Is3D() or not np.isfinite(coordinates).all():
        raise ValueError("Psi4 requires finite 3D coordinates")

    charge = Chem.GetFormalCharge(mol)
    if multiplicity is None:
        multiplicity = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()) + 1
    electrons = sum(atom.GetAtomicNum() for atom in mol.GetAtoms()) - charge
    unpaired = multiplicity - 1
    if multiplicity < 1 or unpaired > electrons or (electrons - unpaired) % 2:
        raise ValueError("Psi4 multiplicity is incompatible with the electron count")
    if reference is None:
        hf = method.lower() in {"hf", "scf", "rhf", "uhf", "rohf"}
        reference = ("rhf" if hf else "rks") if multiplicity == 1 else ("uhf" if hf else "uks")
    if reference not in {"rhf", "uhf", "rohf", "rks", "uks"}:
        raise ValueError(f"Unsupported Psi4 reference: {reference}")
    if multiplicity > 1 and reference in {"rhf", "rks"}:
        raise ValueError("An open-shell Psi4 calculation requires an open-shell reference")

    return {
        "schema_name": "qcschema_input",
        "schema_version": 1,
        "driver": "energy",
        "model": {"method": method, "basis": basis},
        "molecule": {
            "symbols": [atom.GetSymbol() for atom in mol.GetAtoms()],
            "geometry": (coordinates * ANGSTROM_TO_BOHR).ravel().tolist(),
            "molecular_charge": charge,
            "molecular_multiplicity": multiplicity,
            "fix_com": True,
            "fix_orientation": True,
            "fix_symmetry": "c1",
        },
        "keywords": {"reference": reference},
    }


def read_output(path: Path) -> dict[str, ScalarProperty]:
    """Reads and parses a Psi4 QCSchema JSON energy output file.

    Args:
        path (Path):
            Path to the JSON output file produced by Psi4.

    Returns:
        dict[str, ScalarProperty]: A dictionary containing the total energy in
        Hartree under the key "total_energy".

    Raises:
        RuntimeError: If the file cannot be read or parsed, the calculation did
            not succeed, or the returned energy is missing or is not a finite
            numeric scalar.
    """
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict) or result.get("success") is not True:
            error = result.get("error", {}) if isinstance(result, dict) else result
            raise RuntimeError(f"Psi4 calculation failed: {error}")
        energy = result["return_result"]
        if isinstance(energy, bool) or not isinstance(energy, (float, int)):
            raise ValueError("Energy must be a numeric scalar")
        if not math.isfinite(energy):
            raise ValueError("Energy must be finite")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Invalid or missing Psi4 energy result: {path}") from exc
    return {ENERGY_KEY: ScalarProperty(float(energy))}


def calculate_single_point(
    mol: Chem.Mol,
    conf_id: int,
    method: str,
    basis: str,
    multiplicity: int | None = None,
    reference: Reference | None = None,
    memory_mb: int = 512,
    scr: Path = DEFAULT_SCR,
    n_cores: int = 1,
    keep_files: bool = False,
) -> dict[str, ScalarProperty]:
    """Runs a gas-phase Psi4 single-point calculation on a given conformer.

    Args:
        mol (Chem.Mol):
            RDKit molecule with explicit hydrogen atoms and finite 3D coordinates.
            The calculation does not optimize or modify the input geometry.
        conf_id (int):
            The conformer ID to consider.
        method (str):
            Psi4 calculation method, e.g. "hf" or "pbe0". No method fallback
            is performed if the calculation fails.
        basis (str):
            Basis set for the calculation, e.g. "def2-SVP".
        multiplicity (int | None, optional):
            Spin multiplicity. Defaults to None, which uses the radical-electron
            count plus one. Override this high-spin assumption for other states.
        reference (Reference | None, optional):
            SCF reference: "rhf", "uhf", "rohf", "rks", or "uks".
            Defaults to None, which selects RHF/UHF for HF or SCF methods and
            RKS/UKS for DFT, depending on multiplicity.
        memory_mb (int, optional):
            Memory allowance in MB per calculation, shared across its threads.
            Defaults to 512.
        scr (Path, optional):
            Parent directory for the isolated calculation directory.
            Defaults to DEFAULT_SCR.
        n_cores (int, optional):
            Number of CPU threads allocated to the calculation. Defaults to 1.
            Passed to the Psi4 executable through its -n argument.
        keep_files (bool, optional):
            Whether to retain the calculation directory and Psi4 scratch files
            for debugging. Defaults to False.

    Returns:
        dict[str, ScalarProperty]: A dictionary containing the total energy in
        Hartree under the key "total_energy".

    Raises:
        ValueError: If the molecular input, method, basis, multiplicity, or
            reference is invalid, or memory_mb or n_cores is not positive.
        OSError: If the calculation directory or input file cannot be created.
        RuntimeError: If Psi4 is unavailable, execution fails, or its energy
            result is missing, malformed, unsuccessful, or nonfinite.
    """
    data = build_input(mol, conf_id, method, basis, multiplicity, reference)
    if memory_mb < 1 or n_cores < 1:
        raise ValueError("Psi4 memory_mb and n_cores must be positive")
    temp = WorkDir(dir=scr, prefix="psi4_", keep=keep_files)
    workdir = temp.get_path().resolve()
    (workdir / "input.json").write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    try:
        manager = SOFTWARE_CONFIG.get_environment_manager()
        command = manager.get_command("psi4", "psi4")
        env = manager.get_run_environment("psi4", n_cores=n_cores)
        # Psi4's CLI thread flag takes precedence over OMP/MKL environment vars.
        args = [
            "--qcschema",
            "-i",
            "input.json",
            "-o",
            "result.json",
            "-n",
            str(n_cores),
            "--memory",
            f"{memory_mb} MB",
            "-s",
            str(workdir),
        ]
        if keep_files:
            args.append("--messy")
        stdout, stderr = run_command(shlex.join([command, *args]), env=env, cwd=workdir)
        (workdir / "stdout.log").write_text(stdout, encoding="utf-8")
        (workdir / "stderr.log").write_text(stderr, encoding="utf-8")
        return read_output(workdir / "result.json")
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"Psi4 single-point calculation failed in {workdir}: {exc}") from exc
