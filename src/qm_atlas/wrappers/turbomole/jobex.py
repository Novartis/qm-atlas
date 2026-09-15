import logging
from collections.abc import Sequence
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from ppqm.utils.files import WorkDir
from rdkit import Chem

from qm_atlas import utils as mp_utils
from qm_atlas.software_environment import SOFTWARE_CONFIG
from qm_atlas.tasks import common as calculated_properties
from qm_atlas.units import BOHR_TO_ANGSTROM
from qm_atlas.wrappers import xtb
from qm_atlas.wrappers.common import run_command
from qm_atlas.wrappers.turbomole import utils

_logger = logging.getLogger(__name__)

PREPARE_JOBEX = """$atoms
basis= {{ basis }}
$coord file=coord
$symmetry c1
$eht charge={{ charge }} {% if scfiterlimit %}
$scfiterlimit {{ scfiterlimit }} {% endif %}
$dft
{% for func in functional %}functional {{ func }}\n{% endfor %}gridsize {{ grid }}
{% if use_disp %}$disp3 bj {% endif %}
$scfconv {{ scf_conv }}
$rij
$marij {% if use_cosmo %}
$cosmo
  epsilon={{ epsilon_value }} {% endif %}
$grad file=gradient
$end
"""

# default level for rescoss 100 cycles of BP86/TZVP
DEFAULT_BASIS = "def2-TZVP"
DEFAULT_FUNCTIONAL = ("b-p",)
DEFAULT_STEPS = 100
DEFAULT_ENERGY_CONV = 7
DEFAULT_SCF_CONV = 7
DEFAULT_GEOM_CONV = 3
DEFAULT_GRID = "m4"
DEFAULT_SCFITERLIMIT = 500
DEFAULT_SOLVENT = "conductor"


def run_jobex(
    mol: Chem.Mol,
    conf_id: int,
    control_template: str = PREPARE_JOBEX,
    keep_files: bool = False,  # disable for debugging
    max_num_steps: int = DEFAULT_STEPS,
    energy_conv: int = DEFAULT_ENERGY_CONV,
    scf_conv: int = DEFAULT_SCF_CONV,
    geom_conv: int = DEFAULT_GEOM_CONV,
    grid: str = DEFAULT_GRID,
    basis: str = DEFAULT_BASIS,
    functional: Sequence[str] = DEFAULT_FUNCTIONAL,
    use_cosmo: bool = True,
    solvent: str = DEFAULT_SOLVENT,
    use_disp: bool = False,
    scfiterlimit: int = DEFAULT_SCFITERLIMIT,
    keep_coords_fixed: list[int] | str | None = None,
    log_information: bool = True,
    n_cores: int = 1,
    **kwargs,
) -> tuple[np.ndarray | None, dict[str, calculated_properties.Property] | None]:
    """Runs the turbomole jobex script to optimize coordinates of a conformer

    Args:
        mol (Chem.Mol):
            The molecule to work with.
        conf_id (int):
            The ID of the conformation to work with.
        control_template (str, optional):
            The template defining the turbomole control file; this sets the functional,
            basis set etc. Defaults to :data:`PREPARE_JOBEX`.
        keep_files (bool, optional):
            Whether to keep the intermediate files produced by turbomole.
            Defaults to False.
        max_num_steps (int, optional):
            The maximum number of optimization steps to perform. Defaults to :data:`DEFAULT_STEPS` (100).
        energy_conv (int, optional):
            Sets geometry iteration energy convergence threshold to 10.**(-energy_conv).
            Higher values lead to tighter convergence. Defaults to :data:`DEFAULT_ENERGY_CONV`,
            note that the turbomole jobex default is 6.
        scf_conv (int, optional):
            Sets scf energy convergence threshold to 10.**(-scf_conv). Higher values lead to
            tighter convergence. scf_conv should be at least as tight as geom_conv, otherwise
            geometry iterations will have trouble converging. Defaults to :data:`DEFAULT_SCF_CONV`.
        geom_conv (int, optional):
            Sets coordinate change convergence threshold to 10.**(-geom_conv). Higher values
            lead to tighter convergence. This value should be set higher for geometries to be
            used in IR or VCD spectra calculations to avoid inaccurate intensities and negative
            frequencies. Defaults to :data:`DEFAULT_GEOM_CONV`, which is also the turbomole default.
        grid (str, optional):
            Sets grid to use for turbomole calculation. Options can be m3, m4, etc... or 3, 4, etc...
            Higher values lead to denser grids with greater computational cost; denser grids are
            necessary for some calculations, but the error for the default is typically small
            relative to DFT error in general. Defaults to :data:`DEFAULT_GRID`.
        basis (str, optional):
            The basis set to specify in the turbomole control file. Defaults to :data:`DEFAULT_BASIS`.
        functional (Sequence[str], optional):
            The functional to specify for the turbomole control file. If the list has several
            entries, the control file will have several lines "functional ..."
            Defaults to :data:`DEFAULT_FUNCTIONAL`.
        use_cosmo (bool, optional):
            Whether to use COSMO solvation model. Defaults to True.
        solvent (str, optional):
            The solvent to use for COSMO calculations. Defaults to :data:`DEFAULT_SOLVENT` ("conductor").
        use_disp (bool, optional):
            If True, adds dispersion corrections ($disp3 bj) to the control file.
            Defaults to False.
        scfiterlimit (int, optional):
            The maximum number of SCF iterations to perform. Defaults to :data:`DEFAULT_SCFITERLIMIT`.
        keep_coords_fixed (list[int] | str | None, optional):
            List of atom indices, for which the positions should be kept fixed during
            the geometry optimization. Defaults to None (all positions can be modified).
            Additionally, you may use the string "heavy_atoms" to indicate that only
            the hydrogen positions may be adapted.
        log_information (bool, optional):
            Whether to log information about the settings used for the run.
            Defaults to True.
        n_cores (int, optional):
            The number of CPU cores to use for the calculation. Defaults to 1.
        **kwargs:
            Additional keyword arguments passed to :func:`~qm_atlas.wrappers.turbomole.utils.prepare_turbomole_input`.

    Raises:
        RuntimeError: If the calculation fails or turbomole ends abnormally.

    Returns:
        tuple[np.ndarray | None, dict[str, Property] | None]:
            A tuple containing:

            - new_coords: A numpy array of shape (num_atoms, 3) with optimized atom
              coordinates, or None if optimization failed.
            - opt_results: A dictionary with optimization results including convergence
              status, or None if optimization failed.
    """

    template_options = {
        "basis": basis,
        "functional": functional,
        "use_disp": use_disp,
        "scf_conv": scf_conv,
        "scfiterlimit": scfiterlimit,
        "grid": grid,
    }

    if log_information:
        _logger.info(f"Running turbomole geometry optimization with options {template_options}")

    labels = [atom.GetSymbol() for atom in mol.GetAtoms()]

    if isinstance(keep_coords_fixed, str):
        if keep_coords_fixed == "heavy_atoms":
            keep_coords_fixed = [idx for idx, label in enumerate(labels) if label != "H"]
        else:
            raise ValueError()

    temp = utils.prepare_turbomole_input(
        mol,
        control_template,
        conf_id=conf_id,
        keep_files=keep_files,
        keep_coords_fixed=keep_coords_fixed,
        use_cosmo=use_cosmo,
        solvent=solvent,
        **template_options,
        **kwargs,
    )
    scr = temp.get_path()

    # execute turbomole geometry optimization script
    software_env_manager = SOFTWARE_CONFIG.get_environment_manager()
    cmd = software_env_manager.get_command("turbomole", "jobex")
    env = software_env_manager.get_run_environment("turbomole", n_cores=n_cores)
    stdout, stderr = run_command(
        f"{cmd} -ri -c {max_num_steps} -energy {energy_conv} -gcart {geom_conv}",
        env=env,
        cwd=scr,
    )
    has_converged = check_jobex(temp, stdout, stderr)
    opt_results: dict[str, calculated_properties.Property] = {}
    opt_results[xtb.OPTIMIZATION_CONVERGED_KEY] = calculated_properties.BoolProperty(has_converged)

    coord_file = scr / "coord"
    new_coords = parse_coord(coord_file, labels, keep_coords_fixed=keep_coords_fixed)

    return new_coords, opt_results


def check_jobex(
    temp: WorkDir,
    stdout: str,
    stderr: str,
) -> bool:
    """Checks if a turbomole jobex job failed. Raises a RuntimeError in case it failed.

    Args:
        temp (WorkDir):
            The directory the job was run in
        stdout (str):
            The stdout output of the job
        stderr (str):
            The output of stderr

    Raises:
        RuntimeError: If the job has failed.

    Returns:
        bool: Whether the jobex optimization converged.
    """

    if "ended normally" not in stderr:
        _logger.error("Turbomole jobex ended abnormally")
        log_debug_information(stdout, stderr)
        raise RuntimeError("Turbomole jobex ended abnormally")

    converged_flag = temp.get_path() / "GEO_OPT_CONVERGED"
    return converged_flag.exists()


def log_debug_information(stdout: str | None, stderr: str | None):
    """Logs debug information from the turbomole jobex run.

    Args:
        stdout (str | None): The stdout output from the jobex run.
        stderr (str | None): The stderr output from the jobex run.
    """
    if stderr is not None:
        _logger.debug(f"Turbomole Message: {stderr}")

    stdout_log_lines = stdout.splitlines() if stdout is not None else []

    if len(stdout_log_lines) <= 200:
        _logger.debug("log file")
        _logger.debug(stdout)

    else:
        _logger.debug("log file is very long, reducing output")
        _logger.debug("First 100 lines of log file:")
        _logger.debug("\n".join(stdout_log_lines[:100]))
        _logger.debug("Last 100 lines of log file:")
        _logger.debug("".join(stdout_log_lines[-100:]))


def parse_coord(
    coord_file: Path,
    expected_atom_labels: list[str] | None = None,
    keep_coords_fixed: list[int] | None = None,
) -> np.ndarray:
    """parses coordinates from a turbomole coord file.

    Args:
        coord_file (Path):
            Path to a turbomole coord file
        expected_atom_labels (list[str], optional):
            The list of element symbol corresponding to the expected indexing of
            the molecule. If provided, this is used as a cross-check. Defaults to None,
            in this case no checking of the indexing is performed.
        keep_coords_fixed (list[int] | None):
            List of atom indices, for which the positions should be kept fixed during
            the geometry optimization.

    Raises:
        RuntimeError:
            If the checking of the order of element symbols fails

    Returns:
        new_coords (np.ndarray):
            An array of shape (n_atoms, 3) listing the coordinates of the nuclei.
    """

    with mp_utils.open_utf8(coord_file, "r") as tmol_coord:
        coord_lines = tmol_coord.readlines()

    marked_lines = []
    for idx, line in enumerate(coord_lines):
        if "$" in line:
            marked_lines.append(idx)
        if len(marked_lines) >= 2:
            break
    if len(marked_lines) != 2:
        _logger.error("Error parsing tmol coord file")
        raise RuntimeError
    line_1, line_2 = marked_lines[0], marked_lines[1]

    if keep_coords_fixed is None:
        col_names = ["X", "Y", "Z", "label"]
    else:
        col_names = ["X", "Y", "Z", "label", "fixed"]
    coord_df = pd.read_csv(
        StringIO("".join(coord_lines[line_1 + 1 : line_2])),
        engine="python",
        sep=r"\s+",
        names=col_names,
        skipinitialspace=True,
        on_bad_lines="skip",
    )

    # check if atom labels match
    if expected_atom_labels is not None:
        new_labels = coord_df["label"].to_list()
        for new_label, old_label in zip(new_labels, expected_atom_labels):
            if new_label.lower() != old_label.lower():
                _logger.error("Label Mismatch when parsing jobex output")
                raise RuntimeError("Label Mismatch when parsing jobex output")

    if keep_coords_fixed is not None:
        for atom_idx in keep_coords_fixed:
            if coord_df.loc[atom_idx, "fixed"] != "f":
                raise RuntimeError(f"Atom {atom_idx} was expected to be fixed but is not")

    new_coords = coord_df.loc[:, ["X", "Y", "Z"]].to_numpy() * BOHR_TO_ANGSTROM

    return new_coords
