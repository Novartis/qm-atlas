import json
import logging
import time
import traceback
from collections.abc import Sequence
from pathlib import Path

from jinja2 import Template
from rdkit import Chem

from qm_atlas.constants import DEFAULT_SCR
from qm_atlas.software_environment import SOFTWARE_CONFIG
from qm_atlas.tasks import common as calculated_properties
from qm_atlas.utils import open_utf8
from qm_atlas.wrappers.common import run_command
from qm_atlas.wrappers.helpers import wait_for_license
from qm_atlas.wrappers.jaguar import common
from qm_atlas.wrappers.jaguar.common import FEATURE_NAME, SOFTWARE_NAME

DEFAULT_OPTIONS = {
    "-WAIT": "",
}

NUM_LICS_PER_CORE = 2

_logger = logging.getLogger(__name__)

CONFIG_TEMPLATE = """
OutputFormats:
    - sdf
BasisSets:
    - {{ basis }}
Functionals:
{% for func in functional %}    - {{ func }}\n{% endfor %}
GenKeyvals:
{% for key, value in atom_prop_dict.items() %}    {{ key }}: {{ value }}\n{% endfor %}
Properties:
    - all_descriptor_numeric
"""


DEFAULT_BASIS = "LACVP*"
DEFAULT_FUNCTIONAL = ("b3lyp-d3",)


ATOM_PROPERTY_DICT = {
    "fukui": (
        1,
        [
            "s_j_Atom_Fukui_Index_f_NN_HOMO",
            "s_j_Atom_Fukui_Index_f_NS_HOMO",
            "s_j_Atom_Fukui_Index_f_SN_HOMO",
            "s_j_Atom_Fukui_Index_f_SS_HOMO",
            "s_j_Atom_Fukui_Index_f_NN_LUMO",
            "s_j_Atom_Fukui_Index_f_NS_LUMO",
            "s_j_Atom_Fukui_Index_f_SN_LUMO",
            "s_j_Atom_Fukui_Index_f_SS_LUMO",
        ],
    ),
    "lowdin": (1, ["s_j_Lowdin_Atom_Charge"]),
    "nmr": (
        1,
        [
            "s_j_Atom_NMR_Isotropic_Shielding",
            "s_j_NMR_Atomic_Absolute_Shifts",
        ],
    ),
}

DEFAULT_PROPERTIES = ("fukui", "lowdin", "nmr")


def calculate_descriptors(
    mol: Chem.Mol,
    conf_id: int,
    properties: Sequence[str] = DEFAULT_PROPERTIES,
    basis: str = DEFAULT_BASIS,
    functional: Sequence[str] = DEFAULT_FUNCTIONAL,
    jaguar_options: dict[str, str] | None = None,
    n_cores: int = 1,
    scr: Path = DEFAULT_SCR,
    keep_files: bool = False,
    license_buffer: int = 40,
    license_timeout: int = 2880,  # wait for up to 2 days
) -> dict[str, calculated_properties.AtomBasedProperty]:
    """Wrapper function for jaguar property calculations, runs the qm_descriptors.py script in jaguar.
    The Fukui index calculations are done according to the method described in References [1], [2].

    Args:
        mol (Chem.Mol):
            The molecule to work on.
        properties (list[str]):
            The properties to calculate, options are any combination of
            "fukui", "nmr", and "lowdin" (for Lowdin partial charges).
        basis (str, optional):
            The basis set to use for the calculation.
            Defaults to :data:`DEFAULT_BASIS` (LACVP*).
        functional (list[str], optional):
            The functional to use. Defaults to :data:`DEFAULT_FUNCTIONAL` (b3lyp-d3).
        conf_id (int, optional):
            The conformer to consider, specify the confId for the mol. Defaults to 0.
        jaguar_options (Dict[str, str], optional):
            Additional options for the jaguar run. Defaults to None.
        n_cores (int, optional):
            How many cpus to use. Defaults to 1.
        scr (Path, optional):
            A scratch directory to use. Defaults to DEFAULT_SCR.
        keep_files (bool, optional):
            whether to keep that scratch directory after the calculation has run.
            Defaults to False.
        license_buffer (int, optional):
            A buffer of licenses to leave. Defaults to 40.
        license_timeout (int, optional):
            How long to wait for a license (in minutes). Defaults to 2880 (2 days).

    Raises:
        RuntimeError: If the calculation fails or the output file cannot be found.

    Returns:
        dict[str, calculated_properties.AtomBasedProperty]:
            A dictionary mapping the keys for the calculated properties to the
            arrays holding these (atom-based) properties.
    References:
        [1]: Contreras, R. R.; Fuentealba, P.; Galvan, M.; Perez, P.
        "A direct evaluation of regional Fukui functions in molecules",
        Chem. Phys. Lett. 1999, 304, 405, https://doi.org/10.1016/S0009-2614(99)00325-5

        [2]: Chamorro, E.; Perez, P.
        "Condensed-to-atoms electronic Fukui functions within the framework of spin-polarized density-functional theory",
        J. Chem. Phys. 2005, 123, 114107, https://doi.org/10.1063/1.2033689
    """

    if jaguar_options is None:
        jaguar_options = {}

    temp, mae_file = common.prepare_jaguar_input(
        mol, keep_files=keep_files, scr=scr, conf_id=conf_id
    )
    scr = temp.get_path()

    # create config file
    atom_prop_dict = dict()
    sdf_prop_keys = []
    for key in properties:
        if not key in ATOM_PROPERTY_DICT:
            _logger.warning(f"Unknown Property {key}. Skipping...")
            continue
        atom_prop_dict[key] = ATOM_PROPERTY_DICT[key][0]
        sdf_prop_keys.extend(ATOM_PROPERTY_DICT[key][1])

    template = Template(CONFIG_TEMPLATE)
    config_str = template.render(basis=basis, functional=functional, atom_prop_dict=atom_prop_dict)
    config_file = scr / "configfile.yml"
    with open_utf8(config_file, "w") as c_file:
        c_file.write(config_str)

    software_env_manager = SOFTWARE_CONFIG.get_environment_manager()
    jaguar_cmd = software_env_manager.get_command("jaguar", "jaguar")
    cmd = f"{jaguar_cmd} run qm_descriptors.py -maes {mae_file.resolve()} -config {config_file.resolve()} "

    options = dict(DEFAULT_OPTIONS)
    options.update(jaguar_options)
    options["-PARALLEL"] = n_cores
    options_str = " ".join([f"{key} {val}" for key, val in options.items()])
    cmd += options_str

    # check if license is available
    try:
        buffer = license_buffer + n_cores * NUM_LICS_PER_CORE - 1
        wait_for_license(SOFTWARE_NAME, FEATURE_NAME, buffer=buffer, timeout=license_timeout)
    except TimeoutError:
        _logger.error("No Jaguar license available. Exiting BDE calculation")
        return dict()
    except Exception as exc:
        _logger.error(f"got exception {exc} while waiting for license")
        _logger.error(traceback.format_exc())
        return dict()

    # Run Jaguar QM descriptors script
    env = software_env_manager.get_run_environment("jaguar")
    stdout, stderr = run_command(cmd, env=env, cwd=scr)

    _logger.debug(f"Jaguar Output was: {stdout}")
    if stderr:
        _logger.warning(f"Jaguar run reported error: {stderr}")

    # give a bit of buffer for the out file to be ready
    time.sleep(10)

    # collect output file
    output_file = scr / "all_calcs.sdf"

    if not output_file.is_file():
        _logger.error("Jaguar Calculation failed: No output found")
        raise RuntimeError("Jaguar Calculation failed: No output found")

    descriptors = parse_descriptors(output_file, sdf_prop_keys)
    return descriptors


def parse_descriptors(
    sdf_file: Path,
    property_keys: list[str],
) -> dict[str, calculated_properties.AtomBasedProperty]:
    """Read per-atom descriptor properties from an SDF file into AtomBasedProperty objects."""

    desc_dict = dict()
    with Chem.SDMolSupplier(str(sdf_file.resolve()), removeHs=False) as supplier:
        ret_mol = supplier[0]

    for prop_key in property_keys:
        prop_list = json.loads(ret_mol.GetProp(prop_key))
        desc_dict[prop_key] = calculated_properties.AtomBasedProperty(prop_list)

    return desc_dict
