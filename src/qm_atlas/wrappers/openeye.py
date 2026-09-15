#!/usr/bin/env python

# Simple parameter list
#     Execute Options
#       -param : A parameter file
#
#     File Options
#       -in : Input filename
#       -out : Output filename
#       -prefix : Prefix to use to name output files
#       -progress : Method of showing job progress. "none","dots","log","percent".
#       -sdEnergy : Writes conformer energies to the SD tag field
#       -verbose : Triggers copious logging output
#
#     3D Construction Parameters
#       -fromCT : Generate structures from connection-table only.
#
#     Torsion Driving Parameters
#       -ewindow : Energy window used for conformer selection [Defaults: dense:
#                  15.0]
#       -maxconfs : Maximum number of conformations to be saved. [Defaults: rocs:
#                   50, fastrocs: 10, dense: 20000]
#       -rms : RMS threshold used to determine duplicate conformations [Defaults:
#              dense: 0.3]
#
#     Stereo Parameters
#       -strictstereo : Requires that all chiral atoms and bonds have specified
#                       stereo [Defaults: dense: false]
#
#     General
#       -strict : A convenience flag to set -strictstereo, -strictatomtyping, and
#                 -strictfrags to true or false and override previous settings.
#
#
# Additional help functions:
#    --help simple      : Get a list of simple parameters (as seen above)
#    --help all         : Get a complete list of parameters
#    --help defaults    : List the defaults for all parameters
#    --help <parameter> : Get detailed help on a parameter
#    --help html        : Create an html help file for this program
#    --help versions    : List the toolkits and versions used in the application

import logging
from pathlib import Path

from ppqm.utils import WorkDir
from rdkit import Chem
from rdkit.Chem import AllChem

from qm_atlas import constants, utils
from qm_atlas.software_environment import SOFTWARE_CONFIG
from qm_atlas.wrappers import corina
from qm_atlas.wrappers.common import run_command
from qm_atlas.wrappers.helpers import (
    map_conformers_onto_template,
    read_conformers_as_mols,
    store_atom_map_numbers,
)

OMEGA_CMD_MACRO = ["oeomega", "macrocycle"]
OMEGA_CMD = "omega2"
OMEGA_CLASSIC_FILES = [
    # "oeomega_classic.log",
    # "oeomega_classic.parm",
    "oeomega_classic.rpt",
    "oeomega_classic_status.txt",
    "oeomega_classic.fail",
]
OMEGA_MACRO_FILES = [
    "oeomega_macrocycle_log.txt",
    "oeomega_macrocycle_parm.txt",
    "oeomega_macrocycle_rpt.csv",
]


_logger = logging.getLogger(__name__)


def get_omega_command(
    is_macrocycle: bool = False,
    software_env_manager=None,
) -> str:
    """Determine which omega command to use based on whether the molecule
    contains a macrocycle.

    Args:
        is_macrocycle (bool): Whether to use the macrocycle-specific omega command.
            Defaults to False.
        software_env_manager: An optional environment manager; resolved from the
            software config when None.

    Returns:
        list[str]: The command to execute omega.
    """

    if software_env_manager is None:
        software_env_manager = SOFTWARE_CONFIG.get_environment_manager()
    if is_macrocycle:
        return software_env_manager.get_command("openeye", "oeomega_macrocycle")

    return software_env_manager.get_command("openeye", "omega")


def generate_input_mol(
    molobj: Chem.Mol,
    start_from_corina: bool = False,
) -> tuple[Chem.Mol, bool]:
    """Generate input molecule for omega conformer generation.  If the input
    molecule has hydrogens and 3D coordinates, we use it directly.  Otherwise,
    if start_from_corina is True, we use corina to generate 3D coordinates.  If
    start_from_corina is False, we use rdkit to add hydrogens and generate 2D
    coordinates.

    Args:
        molobj (Chem.Mol):
            The rdkit Mol
        start_from_corina (bool):
            Whether to use corina to generate a 3d input conformation. Note that corina
            is only used if the input is not already 3d. Defaults to False, in this
            case, we use rdkit to generate a 3d input conformation

    Returns:
        tuple[Chem.Mol, bool]:
            The molecule to be used as input to omega, and a boolean indicating
            whether the molecule is 2D (True) or 3D (False).
    """

    if utils.has_hydrogens(molobj) and utils.conformer_is_3d(molobj):
        return Chem.Mol(molobj), False

    if start_from_corina:
        _mol = Chem.AddHs(molobj)
        return corina.generate_conformers_molobj(_mol), False

    _mol = Chem.AddHs(molobj)
    AllChem.Compute2DCoords(_mol)
    return _mol, True


def generate_conformers_molobj(
    molobj: Chem.Mol,
    scr: Path = constants.DEFAULT_SCR,
    keep_files: bool = False,
    start_from_corina: bool = False,
    max_conformers: int | None = None,
    rms_threshold: float | None = None,
    n_cores: int = 1,
) -> Chem.Mol:
    """Main Wrapper for OpenEye Omega Conformer Generation. The function will call
    oeomega classic or oeomega macrocycle (if appropriate) to generate conformations
    for the input molecule.

    Args:
        molobj (Chem.Mol):
            The rdkit Mol
        scr (Path):
            The scratch directory to use for temporary files. Defaults to
            constants.DEFAULT_SCR.
        keep_files (bool):
            Whether to keep the temporary files. Defaults to False. Useful for debugging.
        start_from_corina (bool):
            Whether to use corina to generate a 3d input conformation. Note that corina
            is only used if the input is not already 3d. Defaults to False, in this
            case, we use rdkit to generate a 3d input conformation
        max_conformers (int | None):
            The maximum number of conformers to generate. If None, omega's default
            is used.
        rms_threshold (float | None):
            The RMS threshold for conformer filtering. If None, omega's default is used.
        n_cores (int):
            The number of cores to use for omega. Defaults to 1.

    Raises:
        RuntimeError: If the omega conformer generation fails.

    Returns:
        Chem.Mol: The molecule with all the conformers generated by Openeye Omega.
    """

    return_mol, from_ct = generate_input_mol(molobj, start_from_corina=start_from_corina)
    input_mol = store_atom_map_numbers(return_mol)

    software_env_manager = SOFTWARE_CONFIG.get_environment_manager()

    temp = WorkDir(dir=scr, prefix="omega_", keep=keep_files)
    scr = temp.get_path()

    # write input mol to sdf
    input_sdf = scr / "omega_input.sdf"
    with Chem.SDWriter(str(input_sdf.resolve())) as writer:
        writer.write(input_mol)
    output_sdf = scr / "omega_output.sdf"

    # Check if this is a macrocycle
    is_macrocycle = utils.has_macrocycle(input_mol, ring_threshold=10)

    # build command
    cmd = get_omega_command(
        is_macrocycle=is_macrocycle,
        software_env_manager=software_env_manager,
    )
    cmd_list = [cmd]

    cmd_list.append(f"-in {input_sdf.name}")
    cmd_list.append(f"-out {output_sdf.name}")

    if max_conformers is not None:
        cmd_list.append(f"-maxconfs {max_conformers}")

    # Note: If max_conformers is set to 0, this implies that rms is set to 0 by omega.
    if rms_threshold is not None and max_conformers != 0:
        cmd_list.append(f"-rms {rms_threshold}")

    # oeomega macrocycle does not accept the optional argument -strictstereo
    # and does NOT accept molecules with unspecified stereochemistry.
    if not is_macrocycle:
        cmd_list += [
            "-strictstereo false",
            "-sampleHydrogens true",
            # "-canonOrder false",
        ]
        if not from_ct:
            cmd_list.append("-fromCT false")

    # Use multiple cores
    cmd_list.append(f"-mpi_np {n_cores}")
    cmd = " ".join(cmd_list)

    _logger.debug(cmd)

    env = software_env_manager.get_run_environment("openeye")
    _, stderr = run_command(cmd, env=env, cwd=scr)

    # check if out filename exists
    if not output_sdf.is_file():
        _logger.error("Omega failed: Output file not found")
        lines = stderr.split("\n")
        for line in lines:
            line = line.strip().lower()
            if "error" in line or "license" in line or "warning" in line:
                _logger.error(line)
        raise RuntimeError("Omega conformer generation failed")

    # check if output file is empty
    if output_sdf.stat().st_size == 0:
        _logger.error("Omega failed: Output file is empty")
        if is_macrocycle:
            for log_name in OMEGA_MACRO_FILES:
                _logger.debug(check_error(scr / log_name))
        else:
            for log_name in OMEGA_CLASSIC_FILES:
                _logger.debug(check_error(scr / log_name))
        raise RuntimeError("Omega conformer generation failed: output file is empty")

    generated_mols = read_conformers_as_mols(output_sdf)
    if not generated_mols:
        _logger.error("Omega failed: could not read conformers from output file")
        raise RuntimeError("Omega conformer generation failed: could not read output")

    return map_conformers_onto_template(return_mol, generated_mols)


def check_error(filename: Path) -> str:
    """Reads the log file and returns any error messages found.

    Args:
        filename (Path): The path to the log file to be checked.

    Returns:
        str: The error messages found in the log file, prefixed with "log:".
    """

    if isinstance(filename, str):
        filename = Path(filename)

    if not filename.is_file():
        return ""

    with utils.open_utf8(filename, "r") as f:
        lines = f.read()

    return "log:" + lines
