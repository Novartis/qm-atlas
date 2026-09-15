import logging
import math
import re
from pathlib import Path

from jinja2 import Template
from ppqm.utils.files import WorkDir
from rdkit import Chem

from qm_atlas.constants import DEFAULT_SCR
from qm_atlas.software_environment import SOFTWARE_CONFIG
from qm_atlas.units import hartree_to_kcalmol
from qm_atlas.utils import open_utf8, remove_conformers
from qm_atlas.wrappers.common import run_command, write_xyz

_logger = logging.getLogger(__name__)


# eps values according to https://www.scm.com/doc/ADF/Input/COSMO.html
SOLVENT_TO_EPS = {
    "acetone": 20.7,
    "acetonitrile": 37.5,
    "aniline": 6.8,
    "benzene": 2.3,
    "chloroform": 4.8,
    "chcl3": 4.8,  # also chloroform
    "dmf": 37.0,
    "dmso": 46.7,
    "dioxane": 2.2,
    "ethylacetate": 6.02,
    "hexane": 1.88,
    "methanol": 32.6,
    "nitromethane": 35.87,
    "thf": 7.58,
    "toluene": 2.38,
    "water": 78.39,
    "h2o": 78.39,
    "vacuum": 1,
    "conductor": "infinity",
}


def prepare_turbomole_input(
    mol: Chem.Mol,
    template_str: str,
    conf_id: int = -1,
    use_cosmo: bool = True,
    solvent: str = "conductor",
    keep_coords_fixed: list[int] | None = None,
    scr: Path = DEFAULT_SCR,
    keep_files: bool = False,  # disable for debugging
    **template_kwargs,
) -> WorkDir:
    """Creates a directory for the turbomole input and creates the necessary
    control and coord files.

    Args:
        mol (Chem.Mol):
            The molecule to work with.
        template_str (str):
            The template for the control file to use. Cf. e.g. PREPARE_NMR for
            an example. Using a different control file allows to modify the
            turbomole runs (e.g. different functionals or basis sets). The function
            will render variables called "charge" and "epsilon_value".
        conf_id (int):
            The ID of the conformation to work with.
        use_cosmo (bool, optional):
            If set False, it toggles the line "$cosmo" in the turbomole control file.
            Defaults to True.
        solvent (str, optional):
            The implicit solvent model to use. Translates to an epsilon value
            in turbomole. Defaults to "conductor", other solvents are available if
            they have been added to SOLVENT_TO_EPS
        keep_coords_fixed (list[int], optional):
            List of atom indices, for which the positions should be kept fixed during
            the geometry optimization. Defaults to None, i.e. all positions can be
            modified.
        scr (Path, optional):
            A scratch directory to use for the turbomole calculations.
            Defaults to to DEFAULT_SCR, defined in qm_atlas.constants
        keep_files (bool, optional):
            Whether to keep all files created in the process of the calculation.
            Defaults to False.
        **template_kwargs:
            Additional keyword arguments used to fill out the template.

    Raises:
        RuntimeError:
            If turbomole preparations fail.
        ValueError:
            If an unknown solvent is provided.

    Returns:
        temp (WorkDir):
            A scratch directory for the turbomole calculations.
    """

    # ensure the scratch directory exists
    scr.mkdir(parents=True, exist_ok=True)

    temp = WorkDir(dir=scr, prefix="turbomole_", keep=keep_files)
    scr = temp.get_path()
    scr.mkdir(parents=True, exist_ok=True)

    # convert input to xyz file
    input_xyz = (scr / "input.xyz").resolve()
    write_xyz(mol, input_xyz, conf_id=conf_id)

    # convert to turbomole coordinate input
    software_env_manager = SOFTWARE_CONFIG.get_environment_manager()
    cmd_x2t = software_env_manager.get_command("turbomole", "x2t")
    env = software_env_manager.get_run_environment("turbomole")
    stdout, stderr = run_command(f"{cmd_x2t} {input_xyz}", env=env, cwd=scr)
    if stdout is None:
        _logger.error("Error transforming xyz to turbomole:")
        _logger.error(stderr)
        raise RuntimeError("Error transforming xyz to turbomole")
    if "coord" not in stdout:
        _logger.error("Error transforming xyz to turbomole:")
        _logger.error(stderr)
        raise RuntimeError("Error transforming xyz to turbomole")

    if keep_coords_fixed is not None:
        coord_str = fix_coordinates(stdout, keep_coords_fixed)
    else:
        coord_str = stdout

    with open_utf8(scr / "coord", "w") as turbomole_coord:
        turbomole_coord.write(coord_str)

    # prepare the turbomole control file
    charge = Chem.GetFormalCharge(mol)

    # extract epsilon value for solvent
    if solvent.lower() not in SOLVENT_TO_EPS:
        raise ValueError(f"Unknown solvent {solvent}. Available solvents: {SOLVENT_TO_EPS.keys()}")
    epsilon_value = SOLVENT_TO_EPS[solvent.lower()]

    # render the control file
    template = Template(template_str)
    msg = template.render(
        charge=charge, use_cosmo=use_cosmo, epsilon_value=epsilon_value, **template_kwargs
    )
    with open_utf8(scr / "control", "w") as turbomole_control:
        turbomole_control.write(msg)

    _logger.debug(f"wrote turbomole control file: {msg}")
    return temp


def fix_coordinates(coord_str: str, keep_coords_fixed: list[int]) -> str:
    """Adapts the turbomole coord file to inform turbomole on atoms positions which
        are supposed to be kept fixed (in a geometry optimization).

    Args:
        coord_str (str):
            The contents of the turbomole coord file, as a string
        keep_coords_fixed (list[int]):
            indices of the atoms which are supposed to be kept fixed

    Returns:
        str: The adapted coord file.
    """

    coord_lines = coord_str.split("\n")
    new_lines = []
    line_idx = 0
    for line in coord_lines:
        line = f"{line}\n"
        if "$" in line:
            new_lines.append(line)
            continue
        if line_idx in keep_coords_fixed:
            line = line.strip("\n")
            line = f"{line} f\n"
        new_lines.append(line)
        line_idx += 1

    return "".join(new_lines)


def get_turbomole_version() -> tuple[int, int] | None:
    """Return the installed Turbomole ``(major, minor)`` version.

    Parsed from the ``TURBODIR`` path of the configured turbomole environment
    (e.g. ``.../Turbomole/8.0`` -> ``(8, 0)``). Returns ``None`` if it cannot be
    determined.
    """
    env = SOFTWARE_CONFIG.get_environment_manager().get_run_environment("turbomole")
    turbodir = env.get("TURBODIR", "")
    match = re.search(r"[Tt]urbomole/(\d+)\.(\d+)", turbodir)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def check_tmol_generic(stderr: str, log_file: Path, prog_name: str):
    """Searches the output of a turbomole program for signs of an error and raises an
        exception if it finds one.

    Args:
        stderr (str):
            The output to stderr.
        log_file (Path):
            The path to the turbomole logfile.
        prog_name (str):
            The name of the program that is checked, used for logging.

    Raises:
        RuntimeError:
            If the program claims it ended abnormally.
    """

    with open_utf8(log_file, "r") as tmol_log:
        tmol_log_str = tmol_log.read()

    # Turbomole programs exit 0 even on failure, and newer versions may print
    # nothing to stderr on success, so detect the explicit abnormal-termination
    # marker rather than requiring a positive "ended normally" message.
    if "ended abnormally" in stderr or "ended abnormally" in tmol_log_str:
        _logger.error(f"Turbomole {prog_name} ended abnormally")
        _logger.info(tmol_log_str)
        raise RuntimeError(f"Turbomole {prog_name} ended abnormally")


def seek_match(input_lines, match_str, line_splits):
    """Find matching string in file lines, and grab strings in specified subsequent lines
    This function is sufficient for parsing many non-tabular values in QM output files

    Args:
        input_lines (List):
            file read into lines and stripped of \n characters
        match_str (Str):
            string to find among lines
        line_splits (List of tuples):
            first element in pair specifies how many lines after match_str line,
            second element specifies index of line.split() to append to values

    Returns:
        values (List):
            List of parsed values
    """
    values = []
    for i, line in enumerate(input_lines):
        if match_str in line:
            for line_displace, split_num in line_splits:
                values.append(input_lines[i + line_displace].split()[split_num])
            return values


def extract_energy(cosmo_output: str, unit: str = "kcal/mol") -> float:  # type: ignore
    """extracts the (epsilon=infinity) energy from a .cosmo file.

    Args:
        cosmo_output (str):
            The content of a cosmo file as a string.
        unit (str, optional):
            The unit to use. Defaults to "kcal/mol", other option is "Hartree" or "a.u.",
            resolving to the same unit (Hartree).

    Raises:
        ValueError: If the unit is not recognized.

    Returns:
        float: The energy noted in the cosmo file for Total energy + OC corr.
    """

    if unit not in ["kcal/mol", "Hartree", "a.u."]:
        raise ValueError(f"Unknown unit {unit}")

    for line in cosmo_output.split("\n"):
        if "Total energy + OC corr. [a.u.]" in line:
            energy = float(line.split("=")[1])
            break
    else:
        raise ValueError("Invalid Cosmo File")

    if unit == "kcal/mol":
        return energy * hartree_to_kcalmol

    elif unit == "Hartree" or unit == "a.u.":
        return energy


def extract_converged(
    mol: Chem.Mol,
    cosmo_outputs: list[str],
) -> tuple[Chem.Mol, list[str], set[int]]:
    """extracts converged runs for single-point calculations on all conformers of
        a molecule.

    Args:
        mol (Chem.Mol): The molecule
        cosmo_outputs (list[str]): The output of the single-point calculations

    Returns:
        new_mol (Chem.Mol):
            A copy of the molecule with thse conformers removed, which correspond to
            unconverged single-point calculations.
        converged_outputs (list[str]):
            The cosmo outputs of the converged runs.
        unconverged_idcs (set[int]):
            The conformer indices for which the calculaton failed to converge.
    """

    new_mol = Chem.Mol(mol)
    num_conformers = new_mol.GetNumConformers()
    if num_conformers != len(cosmo_outputs):
        raise ValueError(
            f"Number of conformers ({num_conformers}) does not match "
            f"number of cosmo outputs ({len(cosmo_outputs)})"
        )

    unconverged_idcs = set()
    for idx, cosmo_output in enumerate(cosmo_outputs):
        if cosmo_output is None:
            unconverged_idcs.add(idx)
            continue
        energy = extract_energy(cosmo_output, unit="Hartree")
        if math.isnan(energy):
            unconverged_idcs.add(idx)

    converged_outputs = [
        cosmo_outputs[idx] for idx in range(num_conformers) if idx not in unconverged_idcs
    ]

    remove_conformers(new_mol, unconverged_idcs)

    return new_mol, converged_outputs, unconverged_idcs
