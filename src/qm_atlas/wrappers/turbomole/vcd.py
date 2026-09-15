import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from qm_atlas.software_environment import SOFTWARE_CONFIG
from qm_atlas.tasks.common import TensorProperty
from qm_atlas.utils import open_utf8
from qm_atlas.wrappers.common import run_command
from qm_atlas.wrappers.turbomole import single_point, utils

_logger = logging.getLogger(__name__)

PREPARE_VCD = """$atoms
basis={{ basis }} {% if basis=="def2-mTZVP" %}
jbas=def2-TZVP{% else %}
jbas={{ basis }} {% endif %}
$coord file=coord
$symmetry c1
$eht charge={{ charge }} {% if scfiterlimit %}
$scfiterlimit {{ scfiterlimit }} {% endif %} {% if scfdamp %}
$scfdamp   {{ scfdamp }}{% endif %}
$scfconv {{ scf_conv }} {% if ricore %}
$ricore {{ ricore }} {% endif %}
$dft
{% for func in functional %}functional {{ func }}\n{% endfor %}gridsize {{ grid }} {% if use_disp %}
$disp3 bj {% endif %}
$rij
$marij {% if use_cosmo %}
$cosmo {% if klamt %}
  $klamt {% endif %}
  epsilon={{ epsilon_value }} {% endif %}
$end
"""

DEFAULT_BASIS = "def2-SVP"
DEFAULT_FUNCTIONAL = ("b3-lyp",)
DEFAULT_GRID = "4"
DEFAULT_SCF_CONV = 8
DEFAULT_SCFITERLIMIT = 500
DEFAULT_SCFDAMP = "start=0.300  step=0.050  min=0.100"
DEFAULT_RICORE = None
DEFAULT_SOLVENT = "water"


def calculate_vcd(
    mol: Chem.Mol,
    conf_id: int,
    basis: str = DEFAULT_BASIS,
    functional: Sequence[str] = DEFAULT_FUNCTIONAL,
    grid: str = DEFAULT_GRID,
    scf_conv: int = DEFAULT_SCF_CONV,
    scfiterlimit: int = DEFAULT_SCFITERLIMIT,
    scfdamp: str = DEFAULT_SCFDAMP,
    ricore: int | None = DEFAULT_RICORE,
    use_disp: bool = False,
    use_cosmo: bool = True,
    klamt: bool = False,
    solvent: str = DEFAULT_SOLVENT,
    control_template: str | None = PREPARE_VCD,
    keep_files: bool = False,  # disable for debugging
    n_cores: int = 1,
    **kwargs,
) -> dict[str, TensorProperty] | None:
    """Calculates IR & VCD (vibrational circular dichroism) spectra using turbomole
        for a given set of conditions. VCD is typically used to distinguish enantiomers,
        which have spectra that are mirror images. Note that IR & VCD should be run on geometries
        that are converged to tighter thresholds (energy 8, gcart 4 for turbomole jobex).
        Note that turbomole VCD can't use turbomole's mgrids, so "gridsize 4" is the default rather than m4.

    Args:
        mol (Chem.Mol):
            The molecule to work with.
        conf_id (int):
            The ID of the conformation to work with.
        basis (str, optional):
            The basis set to specify in the turbomole control file.
            Defaults to "def2-SVP".
        functional (Sequence[str], optional):
            The functional to specify for the turbomole control file. If the list has
            several entries, the control file will have several lines
            "functional ..."
            Defaults to ("b3-lyp", ).
        grid (str, optional):
            Sets grid to use for turbomole calculation (numeric format like 4, 5, etc).
            Defaults to "4".
        scf_conv (int, optional):
            Sets scf energy convergence threshold to 10.**(-scf_conv). Higher values lead to tigher convergence.
            Defaults to 8.
        scfiterlimit (int, optional):
            The maximum number of SCF iterations to perform. Defaults to 500.
        scfdamp (str, optional):
            The SCF damping settings written to the control file. Defaults to DEFAULT_SCFDAMP.
        ricore (int | None, optional):
            RICORE memory parameter. Defaults to None.
        use_disp (bool, optional):
            If set, this will add a line $disp3 bj to the control file. Defaults to False.
        use_cosmo (bool, optional):
            Whether to use COSMO solvation model. Defaults to True.
        klamt (bool, optional):
            Whether to use Klamt's COSMO variant. Defaults to False.
        solvent (str, optional):
            The solvent to use for COSMO calculations. Defaults to "conductor".
        control_template (str, optional):
            The template for the control file to use. Cf. e.g. :data:`~qm_atlas.wrappers.turbomole.nmr_shielding.PREPARE_NMR` for
            an example. Using a different control file allows to modify the
            turbomole runs (e.g. different functionals or basis sets). The function
            will render variables called "charge" and "epsilon_value". Defaults to
            :data:`PREPARE_VCD`.
        keep_files (bool, optional):
            Whether to use a keep all files created in the process of the calculation.
            Defaults to False.
        n_cores (int, optional):
            The number of CPU cores to use for the calculation. Defaults to 1.
        **kwargs:
            Additional keyword arguments passed to :func:`~qm_atlas.wrappers.turbomole.utils.prepare_turbomole_input`.

    Raises:
        RuntimeError:
            If turbomole preparations fail (cannot be turned off)
            If turbomole calculations end abnormally, or turbomole output
            file can't be found.

    Returns:
        vcd_results (Dict[str, float]):
            A dictionary containing the IR and VCD spectra energies and intensities
    """
    if control_template is None:
        control_template = PREPARE_VCD

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
        "scfdamp": scfdamp,
    }

    _logger.info(f"Calculating turbomole vcd with options {template_options}")

    temp = utils.prepare_turbomole_input(
        mol,
        control_template,
        conf_id=conf_id,
        keep_files=keep_files,
        solvent=solvent,
        **template_options,
        **kwargs,
    )
    scr = temp.get_path()

    # execute turbomole calculations
    software_env_manager = SOFTWARE_CONFIG.get_environment_manager()
    env = software_env_manager.get_run_environment("turbomole", n_cores=n_cores)
    ridft_cmd = software_env_manager.get_command("turbomole", "ridft")
    _, stderr = run_command(f"{ridft_cmd} > ridft.log", env=env, cwd=scr)  # sp may be unnecessary
    single_point.check_ridft(stderr, scr / "ridft.log")

    vcd_cmd = software_env_manager.get_command("turbomole", "vcd")
    _, stderr = run_command(f"{vcd_cmd} > vcd.log", env=env, cwd=scr)
    vcd_results = parse_vcd(scr / "vibspectrum", scr / "vcd.log")

    return vcd_results


def parse_vcd(out_file: Path, log_file: Path) -> dict[str, TensorProperty] | None:
    """Extracts the IR and VCD spectrum from a vcd calculation exspectrum file output.

    Args:
        out_file (Path):
            Path to the turbomole vcd vibspectrum output file.
        log_file (Path):
            Path to the vcd log file (used for error reporting).

    Returns:
        Dict[str, float]:
            A dictionary mapping the keywords to the respective result.
    """

    if not out_file.is_file():
        with open_utf8(log_file) as vcd_log:
            vcd_log_str = vcd_log.read()
        _logger.error("turbomole vcd did not converge")
        _logger.info(vcd_log_str)
        raise RuntimeError("turbomole vcd did not converge")

    col_names = [
        "mode",
        "symmetry",
        "wave_number(cm-1)",
        "IR intensity(km/mol)",
        "selection_IR",
        "selection_raman",
        "VCD_intensity",
        "arccos_Deg",
    ]

    # It's tempting to try reading vibspectrum with pd.read_csv, but the format is often broken by edge cases
    # I found reading the file line-by-line to better handle missing values.
    try:
        with open_utf8(out_file, "r") as f:
            lines = f.readlines()

        # Validate file has minimum required structure
        if len(lines) < 4:
            raise ValueError(
                f"vibspectrum file has insufficient lines (expected at least 4, got {len(lines)})"
            )

        # Check header line for arccos column
        if "arccos" not in lines[1]:  # arccos_Deg sometimes missing for vcd
            col_names.pop(-1)

        spectrum_lines = lines[3:-1]  # header rows
        freq_lines = []
        for l in spectrum_lines:
            spline = l.split()
            if len(spline) == len(col_names):  # 0 frequency rows are missing the symmetry column
                freq_lines.append(spline)

        if not freq_lines:
            raise ValueError("No valid frequency lines found in vibspectrum output")

        df_vcd = pd.DataFrame(
            freq_lines, columns=col_names
        )  # convert collected lines to pandas datafram

        for col in ["wave_number(cm-1)", "IR intensity(km/mol)", "VCD_intensity"]:
            df_vcd[col] = pd.to_numeric(df_vcd[col], errors="coerce")
            # Check for NaN values after conversion
            if df_vcd[col].isna().any():
                _logger.warning(
                    f"Column '{col}' contains non-numeric values that were converted to NaN"
                )

        df_vcd.drop(
            columns=["mode", "symmetry", "selection_IR", "selection_raman"], inplace=True
        )  # not meaningful information to keep
        vcd_results = df_vcd.to_dict(orient="list")

        # Ensure correct typing for mypy and static checkers
        vcd_results = {
            str(k): TensorProperty(np.array(list(map(float, v)))) for k, v in vcd_results.items()
        }
        return vcd_results

    except (ValueError, KeyError, pd.errors.ParserError) as exc:
        _logger.error(f"Failed to parse vibspectrum file: {exc}")
        with open_utf8(log_file) as vcd_log:
            vcd_log_str = vcd_log.read()
        _logger.info(vcd_log_str)
        raise RuntimeError("turbomole vcd output file format is invalid") from exc
