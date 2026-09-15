import logging
import subprocess
from pathlib import Path

import numpy as np
from rdkit import Chem

_logger = logging.getLogger(__name__)


def write_xyz(mol: Chem.Mol, output_xyz: Path | str, conf_id: int = -1):
    """Writes an xyz file from a molecule.

    Args:
        mol (Chem.Mol): The molecule
        output_xyz (Path | str): The xyz file to be created
        conf_id (int, optional): The conformer ID to consider. Defaults to -1.
    """
    if isinstance(output_xyz, Path):
        output_xyz = str(output_xyz.resolve())

    Chem.MolToXYZFile(mol, output_xyz, confId=conf_id)


def read_xyz_coordinates(xyz_file: Path) -> tuple[list[str], np.ndarray]:
    """Read atomic symbols and coordinates from an XYZ file.

    Parses a standard XYZ format file and extracts the atomic symbols and their
    3D coordinates.

    Args:
        xyz_file: Path to XYZ file to read

    Returns:
        tuple[list[str], np.ndarray]:
            A tuple ``(symbols, coordinates)`` where ``symbols`` is the list of
            element symbols (strings) in order of appearance and ``coordinates``
            is a numpy array of shape (n_atoms, 3) with atomic coordinates in
            Angstrom units.

    Examples:
        >>> symbols, coords = read_xyz_coordinates("molecule.xyz")
        >>> print(len(symbols))  # Number of atoms
        >>> print(coords.shape)  # (n_atoms, 3)
        >>> print(coords[0])  # Coordinates of first atom in Angstrom
    """
    mol_without_connectivity = Chem.MolFromXYZFile(str(xyz_file.resolve()))
    coordinates = mol_without_connectivity.GetConformer().GetPositions()
    symbols = [atom.GetSymbol() for atom in mol_without_connectivity.GetAtoms()]

    return symbols, coordinates


def run_command(
    command: str,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    cwd: Path | str | None = None,
) -> tuple[str, str]:
    """Runs a shell command

    Args:
        command (str): The command to run
        env (dict[str, str], optional): Environment variables to set. Defaults to None.
        timeout (int, optional): Timeout in seconds. Defaults to None.
        cwd (Path | str, optional): Working directory for the command. Defaults to None.

    Raises:
        RuntimeError: If the command fails or times out

    Returns:
        tuple[str | None, str | None]: stdout and stderr from the command
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
            cwd=cwd,
        )
        return result.stdout, result.stderr

    except subprocess.CalledProcessError as exc:
        _logger.error(f"Command '{command}' failed with error: {exc.stderr}")
        _logger.error(f"stdout: {exc.stdout}")
        raise RuntimeError(f"Command '{command}' failed") from exc

    except subprocess.TimeoutExpired as exc:
        _logger.error(f"Command '{command}' timed out after {timeout} seconds")
        raise RuntimeError(f"Command '{command}' timed out after {timeout} seconds") from exc
