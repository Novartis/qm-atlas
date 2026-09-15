import logging
import tempfile
import traceback
from collections.abc import Sequence
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from qm_atlas.software_environment import SOFTWARE_CONFIG
from qm_atlas.tasks.common import AtomBasedProperty, Property, ScalarProperty
from qm_atlas.utils import open_utf8
from qm_atlas.wrappers.common import run_command
from qm_atlas.wrappers.turbomole import single_point, utils

_logger = logging.getLogger(__name__)

PREPARE_FUKUI = """$atoms
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
{% for func in functional %}  functional {{ func }}\n{% endfor %}
  gridsize {{ grid }} {% if use_disp %}
$disp3 bj {% endif %}
$rij
$marij {% if use_cosmo %}
$cosmo {% if klamt %}
  $klamt {% endif %}
  epsilon={{ epsilon_value }} {% endif %}
$end
"""

DEFAULT_BASIS = "def2-SV(P)"
DEFAULT_FUNCTIONAL = ("pbe",)
DEFAULT_GRID = "m3"
DEFAULT_SCF_CONV = 7
DEFAULT_SCFITERLIMIT = 100
DEFAULT_SCFDAMP = "start=2.000  step=0.050  min=0.100"
DEFAULT_RICORE = None
DEFAULT_SOLVENT = "water"


def calculate_fukui(
    mol: Chem.Mol,
    conf_id: int,
    control_template: str | None = PREPARE_FUKUI,
    charge_schemes: list | None = None,
    keep_files: bool = False,  # disable for debugging
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
    n_cores: int = 1,
    **kwargs,
) -> dict[str, Property] | None:
    """Calculates Fukui indices for all atoms in a conformer using turbomole.

    Args:
        mol (Chem.Mol):
            The molecule to work with.
        conf_id (int):
            The ID of the conformation to work with.
        control_template (str, optional):
            The template for the control file to use. Cf. e.g. :data:`PREPARE_FUKUI` for
            an example. Using a different control file allows to modify the
            turbomole runs (e.g. different functionals or basis sets). The function
            will render variables called "charge" and "epsilon_value". Defaults to
            :data:`PREPARE_FUKUI`.
        charge_schemes (list, optional):
            The charge schemes to use the list can contain the turbomole charge
            schemes nbo, mulliken, loewdin, paboon.
            Defaults to all of them, see :func:`get_charge_schemes`.
        keep_files (bool, optional):
            Whether to use a keep all files created in the process of the calculation.
            Defaults to False.
        basis (str, optional):
            The basis set to specify in the turbomole control file.
            Defaults to "def2-SV(P)".
        functional (Sequence[str], optional):
            The functional to specify for the turbomole control file. If the list has
            several entries, the control file will have several lines
            "functional ..."
            Defaults to ["pbe"].
        grid (str, optional):
            Sets grid to use for turbomole calculation. Options can be m3, m4, etc...
            Defaults to "m3".
        scf_conv (int, optional):
            SCF energy convergence criterion as exponent (10^-scf_conv). Higher values
            lead to tighter convergence. Defaults to 7.
        scfiterlimit (int, optional):
            The maximum number of SCF iterations to perform. Defaults to 100.
        scfdamp (str, optional):
            SCF damping parameters. Defaults to "start=2.000  step=0.050  min=0.100".
        ricore (int | None, optional):
            RICORE memory parameter. Defaults to None.
        use_disp (bool, optional):
            If True, adds dispersion corrections ($disp3 bj) to the control file.
            Defaults to False.
        use_cosmo (bool, optional):
            Whether to use COSMO solvation model. Defaults to False.
        klamt (bool, optional):
            Whether to use Klamt's COSMO variant. Defaults to False.
        solvent (str, optional):
            The solvent to use for COSMO calculations. Defaults to "conductor".
        n_cores (int, optional):
            The number of CPU cores to use for the calculation. Defaults to 1.
        **kwargs:
            Additional keyword arguments passed to :func:`~qm_atlas.wrappers.turbomole.utils.prepare_turbomole_input`.

    Raises:
        RuntimeError:
            If turbomole preparations fail or if calculations end abnormally, or turbomole output
            file can't be found.

    Returns:
        fukui_functions (dict[str, np.ndarray]):
            A dictionary mapping a keyword to the respective fukui indices. For
            each of the arrays, array[idx] is the value of the respective fukui index
            for the atom with index idx. The dictionary also contains some scalar properties
            related to ionization such as the ionization potential or electron affinity.
    """
    if control_template is None:
        control_template = PREPARE_FUKUI

    # Turbomole 8.0 dropped the '-v' verbose output, so the 'Fukui' script no
    # longer emits the per-atom condensed Fukui tables this wrapper parses.
    turbomole_version = utils.get_turbomole_version()
    if turbomole_version is not None and turbomole_version[0] >= 8:
        raise RuntimeError(
            f"Fukui calculation is not supported for Turbomole {turbomole_version[0]}."
            f"{turbomole_version[1]}: the 'Fukui' script no longer emits the per-atom "
            "condensed Fukui tables (the '-v' verbose output was removed in 8.0). "
            "Use Turbomole 7.x for Fukui calculations."
        )

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

    _logger.info(f"Calculating turbomole fukui indices with options {template_options}")

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

    fukui_cmd = software_env_manager.get_command("turbomole", "Fukui")
    _, stderr = run_command(f"{fukui_cmd} -v > fukui.log", env=env, cwd=scr)
    check_fukui(scr / "fukui.log", scr)

    # read in the fukui functions from the output file
    fukui_functions = parse_fukui(scr / "fukui.log", mol, charge_schemes=charge_schemes)

    return fukui_functions


def check_fukui(log_file: Path, cwd: Path):
    """Searches turbomole Fukui output for signs of an error and raises an
        exception if it finds one.

    Args:
        log_file (Path):
            The Fukui output file
        cwd (Path):
            The directory the turbomole calculations were performed in

    Raises:
        RuntimeError:
            If the turbomole Fukui script did not converge
    """

    fukui_converged = cwd / "FUKUI_CONVERGED"
    if not fukui_converged.is_file():
        with open(log_file, encoding="utf-8") as fukui_log:
            fukui_log_str = fukui_log.read()
        _logger.error("turbomole Fukui did not converge")
        _logger.info(fukui_log_str)
        raise RuntimeError("turbomole Fukui did not converge")


def get_charge_schemes() -> list[str]:
    """returns the available charge schemes."""
    return ["nbo", "mulliken", "loewdin", "paboon"]


def parse_fukui(
    fukui_output: Path,
    mol: Chem.Mol,
    charge_schemes: list | None = None,
    tolerance: float = 1.0e-2,
) -> dict[str, Property]:
    """Parses the output of a turbomole Fukui run

    Args:
        fukui_output (Path):
            The Path to the output file produced by turbomole.
        mol (Chem.Mol):
            The molecule that was used, used for consistency checks.
        charge_schemes (list, optional):
            The charge schemes to use. The list can contain the turbomole charge
            schemes nbo, mulliken, loewdin, paboon.
            Defaults to all of them, see get_charge_schemes
        tolerance (float, optional):
            The tolerance to use when checking if the fukui indices sum up to 1.
            Defaults to 0.01.

    Raises:
        RuntimeError:
            If the output cannot be parsed or doesn't pass consistency checks.

    Returns:
        fukui_functions (dict[str, list[float]]):
            A dictionary with keys of the form {type}_{charge_scheme} where the
            type covers nucleophilic, electrophilic and radical Fukui functions
            and the charge schemes are the ones requested. All values are lists
            following the rdkit indexing of the atoms in the molecule.
    """

    if charge_schemes is None:
        charge_schemes = get_charge_schemes()

    if not fukui_output.is_file():
        raise RuntimeError("Output file not found")

    fukui_types = ["nucleophilic", "electrophilic", "radical"]

    labels = [atom.GetSymbol() for atom in mol.GetAtoms()]
    num_atoms = len(labels)

    # read output
    with open_utf8(fukui_output, "r") as f_out:
        lines = f_out.readlines()

    # find positions of tables in turbomole output
    marked_lines = []
    for j, line in enumerate(lines):
        if "-----------------" in line:
            marked_lines.append(j)

    if len(marked_lines) != 3:
        _logger.error("Error parsing turbomole Fukui output. Output file was:")
        with open_utf8(fukui_output, "r") as f_out:
            fukui_log_str = f_out.read()
        _logger.error(fukui_log_str)
        raise RuntimeError("Error parsing turbomole Fukui output.")

    fukui_functions = {}

    # iterate over fukui functions
    for fukui_key, line_index in zip(fukui_types, marked_lines):
        # Locate the section header by scanning upward: the number of descriptive
        # lines between the header and its table varies across Turbomole versions,
        # so a fixed offset is not reliable.
        if not any(fukui_key in lines[k] for k in range(max(0, line_index - 6), line_index)):
            raise RuntimeError("Error parsing turbomole Fukui output")

        # read the output to a pandas DataFrame
        with tempfile.TemporaryFile(mode="w+") as pandas_input:
            for line in lines[line_index - 1 : line_index + num_atoms + 1]:
                if "-------------" in line:
                    continue
                pandas_input.write(line)
            pandas_input.seek(0)
            fukui_df = pd.read_csv(
                pandas_input,
                engine="python",
                sep=" ",
                skipinitialspace=True,
                on_bad_lines="skip",
            )

            # consistency check on atom ordering
            tmol_labels = fukui_df.loc[:, "at"].to_list()
            if len(tmol_labels) != len(labels):
                _logger.error("Error parsing output file")
                raise RuntimeError("Error parsing output file")
            for orig_label, tmol_label in zip(labels, tmol_labels):
                tmol_label = tmol_label.replace("*", "")
                if orig_label.lower() != tmol_label.lower():
                    raise RuntimeError("Error parsing output file")

            # extract fukui functions for the relevant charge schemes
            for charge_scheme in charge_schemes:
                key = f"{fukui_key}_{charge_scheme}"
                res = [float(x) for x in fukui_df.loc[:, charge_scheme]]
                if abs(sum(res) - 1) > tolerance:
                    _logger.error(f"Sum {sum(res)} of fukui indices out of tolerance")
                    raise RuntimeError("Sum of fukui indices out of tolerance")
                fukui_functions[key] = AtomBasedProperty(np.array(res))

    charges_dict = extract_charges_fukui(fukui_output, num_atoms, charge_schemes=charge_schemes)

    ionization_energies = extract_energies_fukui(fukui_output)
    combined_results: dict[str, Property] = {
        **fukui_functions,
        **charges_dict,
        **ionization_energies,
    }

    return combined_results


def extract_charges_fukui(
    fukui_output: Path,
    num_atoms: int,
    charge_schemes: list | None = None,
) -> dict[str, AtomBasedProperty]:
    """Extracts partial charges from the output of a turbomole Fukui run

    Args:
        fukui_output (Path):
            The Path to the output file produced by turbomole.
        num_atoms (int):
            The number of atoms of the molecule, used for consistency checks.
        charge_schemes (list, optional):
            The charge schemes to use. The list can contain the turbomole charge
            schemes nbo, mulliken, loewdin, paboon.
            Defaults to all of them, see get_charge_schemes

    Returns:
        charges_dict (dict[str, list[float]]):
            A dictionary with keys of the form charge_{scheme} where the
            charge schemes are the ones requested. All values are lists
            following the rdkit indexing of the atoms in the molecule.
    """

    if charge_schemes is None:
        charge_schemes = get_charge_schemes()

    # read in file
    with open_utf8(fukui_output, "r") as f_out:
        lines = f_out.readlines()

    # find positions of partial charge results in the file
    start_marker = None
    end_marker = None
    for line_idx, line in enumerate(lines):
        if "fukui0 energy" in line and start_marker is None:
            start_marker = line_idx + 1
        if "++++++++++++++++" in line and end_marker is None:
            end_marker = line_idx
            break

    if start_marker is None or end_marker is None:
        raise RuntimeError("Could not find charge extraction markers in Fukui output")
    if end_marker <= start_marker:
        raise RuntimeError("Could not find charge extraction markers in Fukui output")

    # read into pandas DataFrame
    charge_df = pd.read_csv(
        StringIO("".join(lines[start_marker:end_marker])),
        engine="python",
        sep=" ",
        skipinitialspace=True,
        on_bad_lines="skip",
        names=["f_index", "charge_type", "atom", "atom_idx", "value", "unnamed"],
    )

    # extract values for partial charges (named fukui0 in turbomole)
    f0_filter = charge_df["f_index"] == "fukui0"
    charge_s = charge_df.loc[f0_filter, :].groupby("charge_type")["value"].apply(list)
    charges_dict = {
        f"charge_{scheme}": [float(x) for x in charge_s[scheme]] for scheme in charge_schemes
    }

    # consistency check on the parsing
    for partial_charge_list in charges_dict.values():
        if len(partial_charge_list) != num_atoms:
            raise RuntimeError("Error parsing partial charges from Fukui output")

    return_dict = {key: AtomBasedProperty(np.array(value)) for key, value in charges_dict.items()}

    return return_dict


def extract_energies_fukui(
    fukui_output: Path,
) -> dict[str, ScalarProperty]:
    """extracts different ionization energies from the output of a turbomole Fukui run.

    Args:
        fukui_output (Path):
            The Path to the output file produced by turbomole.

    Returns:
        energies_dict (dict[str, float]):
            A dictionary describing the different ionization energies.
            Contains the keys ionization_potential [eV], electron_affinity [eV],
            hardness [eV], electronegativity [eV], electrophilicity [eV].
    """

    energies_dict = dict()

    with open_utf8(fukui_output, "r") as f_out:
        lines = f_out.readlines()

    line_idx = None
    for idx, line in enumerate(lines):
        if "condensed Fukui functions" in line:
            line_idx = idx
            break

    if line_idx is None:
        _logger.error("Parsing of ionization energies failed")
        return energies_dict

    try:
        # ionization potential
        ip = float(lines[line_idx - 8].split("eV")[0][-10:])
        # electron affinity
        ea = float(lines[line_idx - 7].split("eV")[0][-10:])
        # hardness
        eta = float(lines[line_idx - 6].split("eV")[0][-10:])
        # electronegativity
        chi = float(lines[line_idx - 5].split("eV")[0][-10:])
        # electrophilicity
        omega = float(lines[line_idx - 4].split("eV")[0][-10:])

        # check if results are consistent
        condition = (
            np.isclose(ip, ea + 2 * eta, atol=2.0e-3)
            and np.isclose(ea, chi - eta, atol=2.0e-3)
            and np.isclose(omega, chi**2 / (2 * eta), atol=2.0e-3)
        )
        if not condition:
            raise RuntimeError("Inconsistent ionization energies parsed")

        energies_dict["ionization_potential [eV]"] = ScalarProperty(ip)
        energies_dict["electron_affinity [eV]"] = ScalarProperty(ea)
        energies_dict["hardness [eV]"] = ScalarProperty(eta)
        energies_dict["electronegativity [eV]"] = ScalarProperty(chi)
        energies_dict["electrophilicity [eV]"] = ScalarProperty(omega)
    except RuntimeError:
        _logger.error("Parsing of ionization energies failed")
        return energies_dict

    except (ValueError, TypeError) as exc:
        _logger.error(f"Parsing of ionization energies failed: {exc}")
        _logger.error(f"Exception traceback: {traceback.format_exc()}")

    return energies_dict
