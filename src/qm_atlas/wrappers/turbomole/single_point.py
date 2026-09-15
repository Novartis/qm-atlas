import logging
import math
from collections.abc import Sequence
from pathlib import Path

from rdkit import Chem

from qm_atlas.constants import COSMO_OUTPUT_KEY, DEFAULT_SCR
from qm_atlas.software_environment import SOFTWARE_CONFIG
from qm_atlas.tasks.common import Property, ScalarProperty, StringProperty
from qm_atlas.wrappers.common import run_command
from qm_atlas.wrappers.turbomole import utils

_logger = logging.getLogger(__name__)

# check for def2-mTZVP basis is necessary to make sure basis sets for iodine can be found.
# might be fixed in turbomole versions > 7.7
PREPARE_SP = """$atoms
basis ={{ basis }} {% if basis=="def2-mTZVP" %}
jbas =def2-TZVP{% else %}
jbas ={{ basis }} {% endif %}
$coord file=coord
$symmetry c1
$eht charge={{ charge }} {% if scfiterlimit %}
$scfiterlimit {{ scfiterlimit }} {% endif %}
$scfconv {{ scf_conv }} {% if ricore %}
$ricore {{ ricore }} {% endif %}
$dft
{% for func in functional %}functional {{ func }}\n{% endfor %}
gridsize {{ grid }} {% if use_disp %}
$disp3 bj {% endif %}
$rij
$marij {% if use_cosmo %}
{% if klamt %}
$klamt {% endif %}
$cosmo
  epsilon={{ epsilon_value }}
rsolv = 1.30
$cosmo_atoms
$cosmo_out file={{ name }}.cosmo
$cosmo_isorad {% endif %}
$end
"""


# default COSMO-FINE level
DEFAULT_BASIS = "def2-TZVPD"
DEFAULT_FUNCTIONAL = ("b-p",)
DEFAULT_GRID = "m4"
DEFAULT_SCF_CONV = 7
DEFAULT_SCFITERLIMIT = 500
DEFAULT_RICORE = None
DEFAULT_SOLVENT = "conductor"


def run_ridft(
    mol: Chem.Mol,
    conf_id: int,
    control_template: str | None = PREPARE_SP,
    basis: str = DEFAULT_BASIS,
    functional: Sequence[str] = DEFAULT_FUNCTIONAL,
    grid: str = DEFAULT_GRID,
    scf_conv: int = DEFAULT_SCF_CONV,
    scfiterlimit: int = DEFAULT_SCFITERLIMIT,
    ricore: int | None = DEFAULT_RICORE,
    use_disp: bool = False,
    use_cosmo: bool = True,
    klamt: bool = True,
    solvent: str = DEFAULT_SOLVENT,
    scr: Path = DEFAULT_SCR,
    keep_files: bool = False,  # disable for debugging
    n_cores: int = 1,
    name: str = "Molecule",
    **kwargs,
) -> dict[str, Property] | None:
    """Runs turbomole ridft single-point calculations

    Args:
        mol (Chem.Mol):
            The molecule to work with.
        conf_id (int):
            The ID of the conformation to work with.
        control_template (str | None, optional):
            The template for the control file to use. Using a different template
            allows modification of all turbomole settings, beyond the ones explicitly listed.
            Defaults to :data:`PREPARE_SP`, which is also used if control_template=None.
        basis (str, optional):
            The basis set to specify in the turbomole control file.
            Defaults to :data:`DEFAULT_BASIS` ("def2-TZVPD").
        functional (Sequence[str], optional):
            The functional to specify for the turbomole control file. If the list has
            several entries, the control file will have several lines "functional ..."
            Defaults to :data:`DEFAULT_FUNCTIONAL` ("b-p").
        grid (str, optional):
            Sets grid to use for turbomole calculation. Options can be m3, m4, etc...
            Higher values lead to denser grids with greater computational cost.
            Defaults to :data:`DEFAULT_GRID` ("m4").
        scf_conv (int, optional):
            SCF energy convergence criterion as exponent (10^-scf_conv). Higher values
            lead to tighter convergence. Defaults to :data:`DEFAULT_SCF_CONV` (7).
        scfiterlimit (int, optional):
            The maximum number of SCF iterations to perform. Defaults to :data:`DEFAULT_SCFITERLIMIT` (500).
        ricore (int | None, optional):
            RICORE memory parameter. Defaults to :data:`DEFAULT_RICORE` (None).
        use_disp (bool, optional):
            If True, adds dispersion corrections ($disp3 bj) to the control file.
            Defaults to False.
        use_cosmo (bool, optional):
            Whether to use COSMO solvation model. Defaults to True.
        klamt (bool, optional):
            Whether to use Klamt's COSMO variant. Defaults to True.
        solvent (str, optional):
            The solvent to use for COSMO calculations. Defaults to :data:`DEFAULT_SOLVENT` ("conductor").
        scr (Path, optional):
            The scratch directory for temporary calculation files. Defaults to DEFAULT_SCR.
        keep_files (bool, optional):
            Whether to keep temporary files created during the calculation.
            Defaults to False.
        n_cores (int, optional):
            The number of CPU cores to use for the calculation. Defaults to 1.
        name (str, optional):
            Filename prefix for COSMO output file (within the temp directory that is created).
            Defaults to "Molecule".
        **kwargs:
            Additional keyword arguments passed to :func:`~qm_atlas.wrappers.turbomole.utils.prepare_turbomole_input`.

    Raises:
        RuntimeError:
            If turbomole preparations fail, if calculations end abnormally, or if
            turbomole output file cannot be found.

    Returns:
        dict[str, Property] | None:
            A dictionary with turbomole single point properties including
            HOMO-LUMO gap, total energy, and (if use_cosmo=True) COSMO output.
            Returns None if the calculation failed.
    """

    if control_template is None:
        control_template = PREPARE_SP

    # set up turbomole input
    template_options = {
        "basis": basis,
        "functional": functional,
        "grid": grid,
        "scf_conv": scf_conv,
        "scfiterlimit": scfiterlimit,
        "ricore": ricore,
        "use_disp": use_disp,
        "use_cosmo": use_cosmo,
        "klamt": klamt,
        "name": name,
    }

    temp = utils.prepare_turbomole_input(
        mol,
        control_template,
        conf_id=conf_id,
        scr=scr,
        keep_files=keep_files,
        solvent=solvent,
        **template_options,
        **kwargs,
    )
    scr = temp.get_path()

    software_env_manager = SOFTWARE_CONFIG.get_environment_manager()
    env = software_env_manager.get_run_environment("turbomole", n_cores=n_cores)
    ridft_cmd = software_env_manager.get_command("turbomole", "ridft")
    _, stderr = run_command(f"{ridft_cmd} > ridft.log", env=env, cwd=scr)
    check_ridft(stderr, scr / "ridft.log")
    single_point_functions = parse_ridft(scr / "ridft.log")

    if use_cosmo:
        cosmo_output = read_cosmo_file(scr, "Molecule.cosmo", basis)
        single_point_functions[COSMO_OUTPUT_KEY] = StringProperty(
            cosmo_output if cosmo_output is not None else ""
        )

    return single_point_functions


def read_cosmo_file(scr: Path, filename: str, basis: str) -> str:
    """Reads the contents of the cosmo file and returns it as a string. Tries to
    fix an error in the info line of the cosmo file.
    """

    file = scr / filename

    if not file.is_file():
        raise RuntimeError(f"Could not find cosmo output file at {file}")

    with open(file, "r", encoding="utf-8") as f:
        cosmo_lines = f.readlines()

    # fix info line in cosmo file
    if not basis in cosmo_lines[1]:
        cosmo_lines[1] = cosmo_lines[1].replace("ridft;b-p", f"ridft;b-p;{basis}")
    cosmo_output = "".join(cosmo_lines)

    return cosmo_output


def check_ridft(stderr: str, log_file: Path | str):
    """Searches turbomole ridft output for signs of an error and raises an
        exception if it finds one.

    Args:
        stderr (str):
            The ridft output to stderr.
        log_file (Path | str):
            The path to the turbomole logfile.

    Raises:
        RuntimeError:
            If ridft claims it ended abnormally or did not converge.
    """

    with open(log_file, encoding="utf-8") as ridft_log:
        ridft_log_str = ridft_log.read()

    # Turbomole exits 0 even on failure, and newer versions may print nothing to
    # stderr on success, so detect the explicit abnormal-termination marker
    # rather than requiring a positive "ended normally" message.
    if "ended abnormally" in stderr or "ended abnormally" in ridft_log_str:
        _logger.error("Turbomole ridft ended abnormally")
        _logger.info(ridft_log_str)
        raise RuntimeError("Turbomole ridft ended abnormally")

    # check log file, raise and log errors
    if "ridft did not converge!" in ridft_log_str:
        _logger.error("Turbomole ridft did not converge")
        _logger.info(ridft_log_str)
        raise RuntimeError("Turbomole ridft did not converge")


def parse_ridft(log_file: Path | str) -> dict[str, Property]:
    """Parses the output of a turbomole single point calculation

    Args:
        log_file (Path | str):
            The Path to the ridft.log output file produced by turbomole.

    Raises:
        RuntimeError:
            If the output cannot be parsed or doesn't pass consistency checks.

    Returns:
        single_point_functions (Dict[str, float]):
            A dictionary with turbomole single point properties
    """
    with open(log_file, "r", encoding="utf-8") as f_out:
        lines = f_out.readlines()

    hl_raw = utils.seek_match(lines, "HOMO-LUMO Separation", [[1, -2], [2, -2], [3, -2]])
    if hl_raw is None:
        raise RuntimeError("Cannot find ridft HOMO-LUMO separation")

    total_energy = utils.seek_match(lines, "Total energy + OC corr.", [[0, -1]])
    if total_energy is None:
        total_energy = utils.seek_match(lines, "|  total energy", [[0, -2]])
    if total_energy is None:
        raise RuntimeError("Cannot find ridft total energy")

    try:
        sp_functions = {
            "HOMO(eV)": float(hl_raw[0]),
            "LUMO(eV)": float(hl_raw[1]),
            "HLgap(eV)": float(hl_raw[2]),
            "TotalEnergy(Ht)": float(total_energy[0]),
        }
    except ValueError as exc:
        raise RuntimeError("Could not parse ridft output") from exc

    single_point_functions = {}
    for key, energy in sp_functions.items():
        if math.isnan(energy):
            _logger.error("Turbomole single point calculation produced NaN for energy")
            raise RuntimeError("Turbomole single point calculation produced NaN for energy")
        single_point_functions[key] = ScalarProperty(energy)

    return single_point_functions
