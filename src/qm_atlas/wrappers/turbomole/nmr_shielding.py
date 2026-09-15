"""Turbomole wrapper for NMR Shielding calculation"""
import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from qm_atlas.software_environment import SOFTWARE_CONFIG
from qm_atlas.tasks.common import AtomBasedProperty, Property
from qm_atlas.utils import open_utf8
from qm_atlas.wrappers.common import run_command
from qm_atlas.wrappers.turbomole import single_point, utils

_logger = logging.getLogger(__name__)


PREPARE_NMR = """$atoms
basis={{ basis }} {% if basis=="def2-mTZVP" %}
jbas=def2-TZVP{% else %}
jbas={{ basis }} {% endif %}
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
$cosmo {% if klamt %}
  $klamt {% endif %}
  epsilon={{ epsilon_value }} {% endif %}
$nmr dft shielding constants file=shielding
$end
"""

DEFAULT_BASIS = "def2-TZVP"
DEFAULT_FUNCTIONAL = (
    "xcfun set-gga",
    "xcfun kt3 1.0",
)
DEFAULT_GRID = "3"
DEFAULT_SCF_CONV = 7
DEFAULT_SCFITERLIMIT = 200
DEFAULT_RICORE = None

NMR_SHIELDINGS_KEY = "nmr_shieldings_isotropic"


def calculate_nmr_shieldings(
    mol: Chem.Mol,
    conf_id: int,
    control_template: str | None = PREPARE_NMR,
    keep_files: bool = False,  # disable for debugging
    basis: str = DEFAULT_BASIS,
    functional: Sequence[str] = DEFAULT_FUNCTIONAL,
    grid: str = DEFAULT_GRID,
    scf_conv: int = DEFAULT_SCF_CONV,
    scfiterlimit: int = DEFAULT_SCFITERLIMIT,
    ricore: int | None = DEFAULT_RICORE,
    use_disp: bool = False,
    use_cosmo: bool = True,
    klamt: bool = False,
    solvent: str = "water",
    n_cores: int = 1,
    **kwargs,
) -> dict[str, Property] | None:
    """Calculates isotropic nmr shieldings for all atoms in a conformer using turbomole.

    Args:
        mol (Chem.Mol):
            The molecule to work with.
        conf_id (int):
            The ID of the conformation to work with.
        control_template (str, optional):
            The template for the control file to use. Cf. e.g. :data:`PREPARE_NMR` for
            an example. Using a different control file allows to modify the
            turbomole runs (e.g. different functionals or basis sets). The function
            will render variables called "charge" and "epsilon_value". Defaults to
            :data:`PREPARE_NMR`.
        keep_files (bool, optional):
            Whether to use a keep all files created in the process of the calculation.
            Defaults to False.
        basis (str, optional):
            The basis set to specify in the turbomole control file.
            Defaults to "def2-TZVP".
        functional (Sequence[str], optional):
            The functional to specify for the turbomole control file. If the list has
            several entries, the control file will have several lines
            "functional ..."
            Defaults to ["xcfun set-gga", "xcfun kt3 1.0"].
        grid (str, optional):
            Sets grid to use for turbomole calculation. Options can be 3, 4, m3, m4, etc...
            Defaults to "3".
        scf_conv (int, optional):
            SCF energy convergence criterion as exponent (10^-scf_conv). Higher values
            lead to tighter convergence. Defaults to 7.
        scfiterlimit (int, optional):
            The maximum number of SCF iterations to perform. Defaults to 200.
        ricore (int | None, optional):
            RICORE memory parameter. Defaults to None.
        use_disp (bool, optional):
            If True, adds dispersion corrections ($disp3 bj) to the control file.
            Defaults to False.
        use_cosmo (bool, optional):
            Whether to use COSMO solvation model. Defaults to True.
        klamt (bool, optional):
            Whether to use Klamt's COSMO variant. Defaults to False.
        solvent (str, optional):
            The solvent to use for COSMO calculations. Defaults to "water".
        n_cores (int, optional):
            The number of CPU cores to use for the calculation. Defaults to 1.
        **kwargs:
            cf. the docstring of :func:`~qm_atlas.wrappers.turbomole.utils.prepare_turbomole_input` for more keyword arguments.

    Raises:
        RuntimeError:
            If turbomole preparations fail, turbomole calculations end abnormally,
            or turbomole output file can't be found.

    Returns:
        dict[str, AtomBasedProperty] | None:
            A dictionary mapping a single key to an atom-based property.
            nmr_shieldings[idx] is the isotropic nmr shielding (trace of the nmr shielding tensor)
            for the atom with index idx in the molecule.
    """
    if control_template is None:
        control_template = PREPARE_NMR

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
    }

    _logger.info(f"Calculating turbomole nmr shifts with options {template_options}")

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
    ridft_cmd = software_env_manager.get_command("turbomole", "ridft")
    mpshift_cmd = software_env_manager.get_command("turbomole", "mpshift")
    env = software_env_manager.get_run_environment("turbomole", n_cores=n_cores)

    _, stderr = run_command(f"{ridft_cmd} > ridft.log", env=env, cwd=scr)
    single_point.check_ridft(stderr, scr / "ridft.log")

    _, stderr = run_command(f"{mpshift_cmd} > mpshift.log", env=env, cwd=scr)
    utils.check_tmol_generic(stderr, scr / "mpshift.log", "mpshift")

    # read in the nmr shifts from the output file
    nmr_shieldings = parse_nmr_isotropic(scr / "shielding", mol)
    return {NMR_SHIELDINGS_KEY: AtomBasedProperty(nmr_shieldings)}


def parse_nmr_isotropic(turbomole_output_file: Path, mol: Chem.Mol) -> np.ndarray:
    """Parses the isotropic nmr shifts from a turbomole output file.

    Args:
        turbomole_output_file (Path):
            The Path to the output file produced by turbomole.
        mol (Chem.Mol):
            The molecule the shieldings are computed for (defines atom ordering).

    Raises:
        RuntimeError:
            If the file does not exist at the specified location.

    Returns:
        nmr_shieldings (np.ndarray):
            nmr_shieldings[idx] is the isotropic nmr shielding (trace of the nmr shielding tensor)
            for the atom with index idx in the molecule.

    """

    if not turbomole_output_file.is_file():
        raise RuntimeError(f"Output file {turbomole_output_file} not found")

    # remove first and last line, remove # from second line
    with tempfile.TemporaryFile(mode="w+") as pandas_input:
        with open_utf8(turbomole_output_file, "r") as tmol_out_orig:
            for line in tmol_out_orig.readlines()[1:-1]:
                new_line = line.replace("#", "")
                pandas_input.write(new_line)
        pandas_input.seek(0)

        # read in adjusted turbomole output file
        tmol_df: pd.DataFrame = pd.read_csv(
            pandas_input, engine="python", sep=" ", skipinitialspace=True, on_bad_lines="skip"
        )  # type: ignore

    # check if the element labels are as expected
    labels = [atom.GetSymbol() for atom in mol.GetAtoms()]
    tmol_labels = tmol_df.loc[:, "TYPE"].to_list()
    for orig_label, tmol_label in zip(labels, tmol_labels):
        tmol_label = tmol_label.replace("*", "")
        if orig_label.lower() != tmol_label.lower():
            raise RuntimeError("Error parsing turbomole output file")

    return tmol_df.loc[:, "ISOTROPIC"].to_numpy()  # pylint: disable=no-member
