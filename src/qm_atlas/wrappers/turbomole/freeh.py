from collections.abc import Sequence
from io import StringIO
from pathlib import Path

import pandas as pd
from rdkit import Chem

from qm_atlas.software_environment import SOFTWARE_CONFIG
from qm_atlas.tasks.common import Property, ScalarProperty
from qm_atlas.utils import open_utf8
from qm_atlas.wrappers.common import run_command
from qm_atlas.wrappers.turbomole import single_point, utils

PREPARE_FREEH = """$atoms
basis ={{ basis }} {% if basis=="def2-mTZVP" %}
jbas =def2-TZVP{% else %}
jbas ={{ basis }} {% endif %}
$coord file=coord
$symmetry c1
$eht charge={{ charge }} {% if scfiterlimit %}
$scfiterlimit {{ scfiterlimit }} {% endif %} {% if scfdamp %}
$scfdamp {{ scfdamp }}{% endif %}
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
  epsilon={{ epsilon_value }} {% endif %}
$end
"""

DEFAULT_BASIS = "def2-TZVP"
DEFAULT_FUNCTIONAL = ("b3-lyp",)
DEFAULT_GRID = "m3"
DEFAULT_SCF_CONV = 7
DEFAULT_SCFITERLIMIT = 200
DEFAULT_SCFDAMP = "start=0.300  step=0.050  min=0.100"
DEFAULT_RICORE = None
DEFAULT_SOLVENT = "conductor"

FREEH_SETTINGS = """
0.97
tstart=298.15 tend=298.15 numt=1 pstart=0.1 pend=0.1 nump=1
q
"""


def calculate_freeh(
    mol: Chem.Mol,
    conf_id: int,
    control_template: str | None = PREPARE_FREEH,
    freeh_settings: str = FREEH_SETTINGS,
    keep_files: bool = False,  # disable for debugging
    basis: str = DEFAULT_BASIS,
    functional: Sequence[str] = DEFAULT_FUNCTIONAL,
    grid: str = DEFAULT_GRID,
    scf_conv: int = DEFAULT_SCF_CONV,
    scfiterlimit: int = DEFAULT_SCFITERLIMIT,
    scfdamp: str = DEFAULT_SCFDAMP,
    ricore: int | None = DEFAULT_RICORE,
    use_disp: bool = False,
    use_cosmo: bool = False,
    klamt: bool = True,
    solvent: str = DEFAULT_SOLVENT,
    n_cores: int = 1,
    **kwargs,
) -> dict[str, Property] | None:
    """Calculates energy, entropy, enthalpy and chemical potential using turbomole
        for a given set of conditions.

    Args:
        mol (Chem.Mol):
            The molecule to work with.
        conf_id (int):
            The ID of the conformation to work with.
        control_template (str, optional):
            The template for the control file to use. Cf. e.g. :data:`~qm_atlas.wrappers.turbomole.nmr_shielding.PREPARE_NMR` for
            an example. Using a different control file allows to modify the
            turbomole runs (e.g. different functionals or basis sets). The function
            will render variables called "charge" and "epsilon_value". Defaults to
            :data:`PREPARE_FREEH`.
        freeh_settings (str):
            A string specifying the conditions for the freeh calculation.
            Defaults to :data:`FREEH_SETTINGS`.
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
            Defaults to ["b3-lyp"].
        grid (str, optional):
            Sets grid to use for turbomole calculation. Options can be m3, m4, etc...
            Defaults to "m3".
        scf_conv (int, optional):
            SCF energy convergence criterion as exponent (10^-scf_conv). Higher values
            lead to tighter convergence. Defaults to 7.
        scfiterlimit (int, optional):
            The maximum number of SCF iterations to perform. Defaults to 200.
        scfdamp (str, optional):
            SCF damping parameters. Defaults to "start=0.300  step=0.050  min=0.100".
        ricore (int | None, optional):
            RICORE memory parameter. Defaults to None.
        use_disp (bool, optional):
            If True, adds dispersion corrections ($disp3 bj) to the control file.
            Defaults to False.
        use_cosmo (bool, optional):
            Whether to use COSMO solvation model. Defaults to False.
        klamt (bool, optional):
            Whether to use Klamt's COSMO variant. Defaults to True.
        solvent (str, optional):
            The solvent to use for COSMO calculations. Defaults to "conductor".
        n_cores (int, optional):
            The number of CPU cores to use for the calculation. Defaults to 1.
        **kwargs:
            Additional keyword arguments to be passed to the function
            :func:`~qm_atlas.wrappers.turbomole.utils.prepare_turbomole_input`, cf. the respective docstring
            for more details

    Raises:
        RuntimeError:
            If turbomole preparations fail, turbomole calculations end abnormally,
            or turbomole output file can't be found.

    Returns:
        freeh_results (dict[str, float]):
            A dictionary containing the chemical potential, energy, entropy and
            enthalpy at the conditions specified in freeh_settings.
    """
    if control_template is None:
        control_template = PREPARE_FREEH

    # set up turbomole input
    template_options = {
        "basis": basis,
        "functional": functional,
        "grid": grid,
        "scf_conv": scf_conv,
        "scfiterlimit": scfiterlimit,
        "scfdamp": scfdamp,
        "ricore": ricore,
        "use_disp": use_disp,
        "use_cosmo": use_cosmo,
        "klamt": klamt,
    }

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
    _, stderr = run_command(f"{ridft_cmd} > ridft.log", env=env, cwd=scr)
    single_point.check_ridft(stderr, scr / "ridft.log")
    single_point_functions = single_point.parse_ridft(scr / "ridft.log")

    aoforce_cmd = software_env_manager.get_command("turbomole", "aoforce")
    _, stderr = run_command(f"{aoforce_cmd} > aoforce.log", env=env, cwd=scr)
    utils.check_tmol_generic(stderr, scr / "aoforce.log", "aoforce")

    freeh_inp = "freeE.inp"
    freeh_out = "freeE.out"
    # write freeh input file
    with open_utf8(scr / freeh_inp, "w") as freeh_inp_f:
        freeh_inp_f.write(freeh_settings)

    freeh_cmd_exec = software_env_manager.get_command("turbomole", "freeh")
    freeh_cmd = f"{freeh_cmd_exec} < {freeh_inp} > {freeh_out}"
    _, stderr = run_command(freeh_cmd, env=env, cwd=scr)
    utils.check_tmol_generic(stderr, scr / freeh_out, "freeh")

    # read in the nmr shifts from the output file
    freeh_results = parse_freeh(scr / freeh_out)
    freeh_results = {**freeh_results, **single_point_functions}

    return freeh_results


def parse_freeh(freeh_output: Path) -> dict[str, Property]:
    """Extracts the chemical potential, energy, entropy and enthalpy from the output
        file of a turbomole freeh calculation.

    Args:
        freeh_output (Path):
            Path to the turbomole freeh output file

    Returns:
        Dict[str, float]:
            A dictionary mapping the keywords to the respective result.
    """

    freeh_results = dict()

    line_1_marker = "ln(qtrans)"
    line_2_marker = "enthalpy"

    with open_utf8(freeh_output, "r") as file_in:
        lines = file_in.readlines()

    line_1 = [idx for idx, line in enumerate(lines) if line_1_marker in line][0]
    line_2 = [idx for idx, line in enumerate(lines) if line_2_marker in line][0]

    table_1 = pd.read_csv(
        StringIO("".join(lines[line_1 : line_1 + 4])),
        engine="python",
        sep=" ",
        skipinitialspace=True,
        on_bad_lines="skip",
    )

    table_2 = pd.read_csv(
        StringIO("".join(lines[line_2 : line_2 + 3])),
        engine="python",
        sep=" ",
        skipinitialspace=True,
        on_bad_lines="skip",
    )

    table_1_keys = ["chem.pot.", "energy", "entropy"]
    table_2_keys = ["enthalpy"]

    for key in table_1_keys:
        freeh_results[key] = ScalarProperty(float(table_1.loc[1, key]))
    for key in table_2_keys:
        freeh_results[key] = ScalarProperty(float(table_2.loc[1, key]))

    return freeh_results
