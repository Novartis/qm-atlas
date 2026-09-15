import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from jinja2 import Template
from ppqm import xtb as ppqm_xtb
from ppqm.utils import linesio
from ppqm.utils.files import WorkDir
from rdkit import Chem

from qm_atlas import utils as mp_utils
from qm_atlas.constants import DEFAULT_SCR
from qm_atlas.software_environment import SOFTWARE_CONFIG
from qm_atlas.tasks import common as calculated_properties
from qm_atlas.wrappers import torsion_constraints
from qm_atlas.wrappers.common import read_xyz_coordinates, run_command, write_xyz

_logger = logging.getLogger(__name__)

DEFAULT_XTB_STEPS = 500
XCONTROL_FILENAME = "xcontrol.inp"
INPUT_FILENAME = "input.xyz"
#: Force constant for the ``$constrain`` block, in Hartree/rad^2 for angles and torsions and
#: Hartree/Bohr^2 for distances. 0.5 holds a restrained torsion to a few tenths of a degree,
#: which is what makes "the input pose was held" a checkable statement. The previous value of
#: 0.001 dated from when the block also restrained every C/N/O/S pairwise distance, so the
#: force constant was not the only thing holding the geometry; with the explicit ``dihedral:``
#: restraints it is, and 0.001 leaves a torsion almost as free as no restraint at all.
DEFAULT_FORCE_CONSTANT = 0.5
JSON_OUTPUT_FILENAME = "xtbout.json"
COORDINATES_OUTPUT_FILENAME = "xtbopt.xyz"
OPTIMIZATION_CONVERGED_FILE = ".xtboptok"
OPTIMIZATION_CONVERGED_KEY = "optimization_converged"
OPTIMIZATION_CONVERGED_PATTERN = "GEOMETRY OPTIMIZATION CONVERGED"
OPTIMIZATION_FAILED_PATTERN = "FAILED TO CONVERGE GEOMETRY OPTIMIZATION"

#: Constraint section shared by the xtb and xtb/Turbomole optimization templates.
#: ``torsions_fixed``, ``angles_fixed`` and ``bonds_fixed`` switch xtb's own global
#: restraints on directly; ``heavy_atom_torsions_fixed`` instead emits the explicit
#: ``dihedral:`` restraints derived by :func:`resolve_dihedral_constraints`.
CONSTRAIN_BLOCK_TEMPLATE = """

$constrain{% if torsions_fixed %}
  all torsions=true{% endif %}{% if heavy_atom_torsions_fixed %}{% for atom_indices in dihedral_constraints %}
  dihedral: {{ atom_indices | join(",") }},auto{% endfor %}{% endif %}{% if angles_fixed %}
  all angles=true{% endif %}{% if bonds_fixed %}
  all bonds=true{% endif %}
  force constant={{ force_constant | default(0.5) }} """


# The ``$gfn`` method is taken from ``method`` (the name used by
# :class:`~qm_atlas.tasks.optimize.XtbOptions`); ``gfn_method`` is still accepted because
# it used to be the only variable name these templates understood.
XCONTROL_TEMPLATE_OPT = (
    """$chrg {{ charge | default(0) }}
$spin {{ num_unpaired_electrons | default(0) }}

$gfn
  method={{ method | default(gfn_method) | default(2) }}
  scc=true

$opt
  maxcycle={{ max_num_steps | default(500) }}
  optlevel={{ opt_level | default("normal") }} {% if constrain %}"""
    + CONSTRAIN_BLOCK_TEMPLATE
    + """{% endif %}
"""
)


XCONTROL_TEMPLATE_SP = """$chrg {{ charge | default(0) }}
$spin {{ num_unpaired_electrons | default(0) }}

$gfn
  method={{ method | default(gfn_method) | default(2) }}
  scc=true
"""

XTB_ENERGY_KEY = "total energy"

XTB_PROPERTY_TYPES = {
    XTB_ENERGY_KEY: calculated_properties.ScalarProperty,
    "HOMO-LUMO gap / eV": calculated_properties.ScalarProperty,
    "electronic energy": calculated_properties.ScalarProperty,
    "dipole / a.u.": calculated_properties.TensorProperty,
    "atomic dipole moments": calculated_properties.TensorProperty,
    "atomic quadrupole moments": calculated_properties.TensorProperty,
    "orbital energies / eV": calculated_properties.TensorProperty,
    "fractional occupation": calculated_properties.TensorProperty,
    "bondorders": calculated_properties.BondBasedProperty,
    "partial charges": calculated_properties.AtomBasedProperty,
}

XTB_FUKUI_PROPERTY_TYPES = {
    "f_plus": calculated_properties.AtomBasedProperty,
    "f_minus": calculated_properties.AtomBasedProperty,
    "f_zero": calculated_properties.AtomBasedProperty,
}

# Maps the summary-table keys produced by ppqm's stdout parser to the canonical
# JSON property names/types. Used as a fallback when xtb does not write
# xtbout.json but still prints the properties to stdout.
XTB_STDOUT_PROPERTY_MAP: dict[str, type[calculated_properties.Property]] = {
    "total_energy": calculated_properties.ScalarProperty,
    "homo_lumo_gap": calculated_properties.ScalarProperty,
    "scc_energy": calculated_properties.ScalarProperty,
}

XTB_STDOUT_KEY_TO_JSON_KEY = {
    "total_energy": XTB_ENERGY_KEY,
    "homo_lumo_gap": "HOMO-LUMO gap / eV",
    "scc_energy": "electronic energy",
}


def get_num_unpaired_electrons(mol: Chem.Mol) -> int:
    """Calculate the number of unpaired electrons in a molecule.

    Counts the total number of radical (unpaired) electrons across all atoms
    in the molecule. This is used to set the $spin parameter in xtb calculations.

    Args:
        mol: RDKit molecule object

    Returns:
        Total number of unpaired electrons (integer >= 0)

    Examples:
        >>> mol = Chem.MolFromSmiles("[CH3]")  # Methyl radical
        >>> get_num_unpaired_electrons(mol)
        1
        >>> mol = Chem.MolFromSmiles("C")  # Methane
        >>> get_num_unpaired_electrons(mol)
        0
    """
    total_num_unpaired = 0
    for atom in mol.GetAtoms():
        total_num_unpaired += atom.GetNumRadicalElectrons()
    return total_num_unpaired


def resolve_dihedral_constraints(
    mol: Chem.Mol,
    conf_id: int = -1,
    constrain: bool = False,
    heavy_atom_torsions_fixed: bool = False,
) -> list[tuple[int, ...]]:
    """Derive the explicit ``dihedral:`` restraints for :data:`CONSTRAIN_BLOCK_TEMPLATE`.

    The global ``all torsions`` / ``all angles`` / ``all bonds`` switches are driven
    straight from the matching boolean template variables, so the only part of the
    constraint block that needs code is the ``dihedral_constraints`` list. When
    ``heavy_atom_torsions_fixed`` is requested, one restraint is selected per
    heavy-atom torsion with
    :func:`~qm_atlas.wrappers.torsion_constraints.find_heavy_atom_torsions`.

    Args:
        mol (Chem.Mol):
            Molecule that is about to be optimized. Only used when
            ``heavy_atom_torsions_fixed`` is True to pick the torsions.
        conf_id (int, optional):
            Conformation the torsions are selected for. Defaults to -1.
        constrain (bool, optional):
            Master switch for constraints. Defaults to False, in which case no
            dihedral restraints are produced.
        heavy_atom_torsions_fixed (bool, optional):
            Whether to restrain one explicit dihedral per heavy-atom torsion.
            Defaults to False.

    Returns:
        list[tuple[int, ...]]: One ``(i, j, k, l)`` tuple of **1-based** atom
        indices per restrained torsion, as xtb expects them, or an empty list when
        no explicit dihedral restraints are requested.
    """
    if not (constrain and heavy_atom_torsions_fixed):
        return []

    torsions = torsion_constraints.find_heavy_atom_torsions(mol, conf_id=conf_id)
    if not torsions:
        _logger.warning(
            "heavy_atom_torsions_fixed was requested but no heavy-atom torsion could "
            "be restrained; those constraints will have no effect."
        )
    return [tuple(index + 1 for index in torsion) for torsion in torsions]


def write_xcontrol_file(
    mol: Chem.Mol,
    temp: WorkDir,
    template_str: str,
    conf_id: int = -1,
    **template_kwargs,
) -> Path:
    """Render the xcontrol template with the molecule's charge/multiplicity and write it to *temp*.

    The ``heavy_atom_torsions_fixed`` keyword argument is resolved into the
    ``dihedral_constraints`` template variable by
    :func:`resolve_dihedral_constraints` before rendering.
    """
    context: dict[str, Any] = {
        "charge": Chem.GetFormalCharge(mol),
        "num_unpaired_electrons": get_num_unpaired_electrons(mol),
        **template_kwargs,
    }
    context["dihedral_constraints"] = resolve_dihedral_constraints(
        mol,
        conf_id=conf_id,
        constrain=bool(context.get("constrain", False)),
        heavy_atom_torsions_fixed=bool(context.get("heavy_atom_torsions_fixed", False)),
    )

    content = Template(template_str).render(**context)

    xcontrol_input = temp.get_path() / XCONTROL_FILENAME
    with mp_utils.open_utf8(xcontrol_input, "w") as f:
        f.write(content)

    return xcontrol_input


def write_xtb_input(
    mol: Chem.Mol,
    conf_id: int,
    template_str: str,
    scr: Path = DEFAULT_SCR,
    keep_files: bool = False,
    temp: WorkDir | None = None,
    **template_kwargs,
) -> tuple[WorkDir, Path, Path]:
    """Writes the input files for an xtb calculation.

    Args:
        mol (Chem.Mol):
            The molecule to write the input for.
        conf_id (int):
            The conformer ID to consider.
        template_str (str):
            The Jinja2 template string for the xcontrol input file.
        scr (Path, optional):
            Directory to use for scratch files. Defaults to DEFAULT_SCR.
        keep_files (bool, optional):
            Whether to keep temporary files. Defaults to False.
        temp (WorkDir | None, optional):
            Optional working directory object. Defaults to None.
        **template_kwargs:
            Additional keyword arguments used to render the xcontrol template.

    Returns:
        tuple[WorkDir, Path, Path]: A tuple containing the working directory object,
        the path to the XYZ input file, and the path to the xcontrol input file.
    """

    scr.mkdir(parents=True, exist_ok=True)

    if temp is None:
        temp = WorkDir(dir=scr, prefix="xtb_", keep=keep_files)
    scratch = temp.get_path()
    scratch.mkdir(parents=True, exist_ok=True)

    xyz_input = scratch / INPUT_FILENAME
    write_xyz(mol, xyz_input, conf_id=conf_id)

    xcontrol_input = write_xcontrol_file(
        mol,
        temp,
        template_str,
        conf_id=conf_id,
        **template_kwargs,
    )

    return temp, xyz_input, xcontrol_input


def run_xtb(
    mol: Chem.Mol,
    conf_id: int,
    template_str: str,
    keep_files: bool = False,
    scr: Path = DEFAULT_SCR,
    n_cores: int = 1,
    xtb_command_add: list[str] | None = None,
    solvation_model: str | None = None,
    solvent: str | None = None,
    use_json: bool = True,
    **template_kwargs,
) -> tuple[WorkDir, str | None, str | None]:
    """Runs an xtb calculation on a given molecule and conformer.

    Args:
        mol (Chem.Mol):
            The molecule to write the input for.
        conf_id (int):
            The conformer ID to consider.
        template_str (str):
            The Jinja2 template string for the xcontrol input file.
        scr (Path, optional):
            Directory to use for scratch files. Defaults to DEFAULT_SCR.
        keep_files (bool, optional):
            Whether to keep temporary files. Defaults to False.
        n_cores (int, optional):
            Number of CPU cores to use. Defaults to 1. Sets OMP_NUM_THREADS
            and other parallelization environment variables.
        xtb_command_add (list[str] | None, optional):
            Additional command-line arguments to pass to xtb. Defaults to None (empty list).
        solvation_model (str | None, optional):
            The solvation model to use. Defaults to None.
        solvent (str | None, optional):
            The solvent to use with the solvation model. Defaults to None.
        use_json (bool, optional):
            Whether to pass ``--json`` to xtb so it writes ``xtbout.json``.
            Defaults to True. When False, properties are parsed from stdout.
        **template_kwargs:
            Additional keyword arguments used to render the xcontrol template.

    Raises:
        ValueError: If solvent is not specified when solvation_model is provided.
        RuntimeError: If the xtb calculation fails.

    Returns:
        tuple[WorkDir, str | None, str | None]: A tuple containing the working directory object,
        the standard output, and the standard error from the xtb calculation.
    """

    if xtb_command_add is None:
        xtb_command_add = list()

    if solvation_model is not None:
        if solvent is None:
            raise ValueError("xtb: Must specify solvent if solvation is requested")
        xtb_command_add.extend([f"--{solvation_model}", solvent])

    temp, input_file, xcontrol_input = write_xtb_input(
        mol,
        conf_id=conf_id,
        scr=scr,
        keep_files=keep_files,
        template_str=template_str,
        **template_kwargs,
    )

    software_env_manager = SOFTWARE_CONFIG.get_environment_manager()
    xtb_cmd = software_env_manager.get_command("xtb", "xtb")
    cmd_list = [
        xtb_cmd,
        str(input_file.name),
        "--input",
        str(xcontrol_input.name),
    ]
    if use_json:
        cmd_list.append("--json")

    cmd_list.extend(xtb_command_add)
    cmd = " ".join(cmd_list)

    env = software_env_manager.get_run_environment("xtb", n_cores=n_cores)
    _logger.debug(f"Running xtb with command: {cmd} in {temp.get_path()}")
    stdout, stderr = run_command(cmd, env=env, cwd=temp.get_path())
    check_xtb_run(stdout, stderr)
    return temp, stdout, stderr


def check_xtb_run(stdout: str, stderr: str):
    """Checks whether an xtb calculation finished successfully. Logs any issues that
    occurred during the calculation.

    Args:
        stdout (str): Standard output from the xtb calculation.
        stderr (str): Standard error from the xtb calculation.

    Raises:
        RuntimeError: If the xtb calculation failed.
    """
    lines = stdout.splitlines() + stderr.splitlines()

    keywords = [
        "Program stopped due to fatal error",
        "abnormal termination of xtb",
    ]
    stoppattern = "normal termination of xtb"

    idcs = linesio.get_rev_indices_patterns(lines, keywords, stoppattern=stoppattern)

    for idx in idcs:
        if idx is not None:
            error_idx = linesio.get_rev_index(lines, "ERROR")
            if error_idx is None:
                _logger.error("xtb failed, but error message could not be read")
                raise RuntimeError("xtb failed, but error message could not be read")

            for line in lines[error_idx:-2]:
                _logger.error(f"xtb error: {line}")
            raise RuntimeError("xtb failed, error messages logged")


def read_properties_from_stdout(
    stdout: str,
    stderr: str,
) -> dict[str, calculated_properties.Property]:
    """Parse xtb properties from stdout as a fallback when no JSON is written.

    Uses ppqm's summary-table parser and maps the recognised keys to the
    canonical JSON property names/types used throughout the wrapper.

    Args:
        stdout (str): Standard output from the xtb calculation.
        stderr (str): Standard error from the xtb calculation.

    Returns:
        A dictionary of the parsed properties, keyed by their canonical names.

    Raises:
        RuntimeError: If the properties could not be parsed from stdout.
    """

    lines = stdout.splitlines() + stderr.splitlines()
    parsed = ppqm_xtb.read_properties_sp(lines)
    if not parsed:
        raise RuntimeError("Could not parse xtb properties from stdout")

    properties: dict[str, calculated_properties.Property] = {}
    for stdout_key, prop_class in XTB_STDOUT_PROPERTY_MAP.items():
        value = parsed.get(stdout_key)
        if value is not None:
            json_key = XTB_STDOUT_KEY_TO_JSON_KEY[stdout_key]
            properties[json_key] = prop_class(value)  # type: ignore
    return properties


def read_xtb_output(
    temp: WorkDir,
    stdout: str,
    stderr: str,
    read_json: bool = True,
    read_bondorders: bool = False,
    read_fukui: bool = False,
) -> dict[str, calculated_properties.Property]:
    """Reads and parses an xtb JSON output file.

    Args:
        temp (WorkDir):
            WorkDir object containing the scratch directory path.
        stdout (str):
            Standard output from the xtb calculation.
        stderr (str):
            Standard error from the xtb calculation.
        read_json (bool, optional):
            Whether to read the JSON output file. Defaults to True.
        read_bondorders (bool, optional):
            Whether to read bond orders from the output. Defaults to False.
        read_fukui (bool, optional):
            Whether to read Fukui indices from the output. Defaults to False.
    Returns:
        A dictionary containing the parsed properties.
    Raises:
        json.JSONDecodeError: If the JSON file exists but is malformed.
        RuntimeError: If the JSON file is missing and properties cannot be
            parsed from stdout either.
    """

    return_properties = {}

    if read_json:
        json_file = temp.get_path() / JSON_OUTPUT_FILENAME
        if json_file.exists():
            with mp_utils.open_utf8(json_file, "r") as f:
                json_data = json.load(f)

            for key, val in json_data.items():
                if key in XTB_PROPERTY_TYPES:
                    prop_class = XTB_PROPERTY_TYPES[key]
                    return_properties[key] = prop_class(val)
        else:
            # xtb occasionally omits the JSON file while still printing the
            # properties to stdout; fall back to parsing those.
            _logger.warning(
                f"xtb JSON output file not found: {json_file}. "
                "Falling back to parsing properties from stdout."
            )
            return_properties.update(read_properties_from_stdout(stdout, stderr))

    if read_bondorders:
        bonds, bondorders = ppqm_xtb.get_wbo(temp.get_path())
        bondorder_dict = {(a1, a2): bo for (a1, a2), bo in zip(bonds, bondorders)}
        bondorder_prop = calculated_properties.BondBasedProperty(bondorder_dict)
        return_properties["bondorders"] = bondorder_prop

    if read_fukui:
        lines = stdout.splitlines() + stderr.splitlines()
        fukui_indices = ppqm_xtb.read_properties_fukui(lines)
        if fukui_indices is None:
            _logger.error("Could not parse xtb fukui indices from stdout")
            raise RuntimeError("Could not parse xtb fukui indices from stdout")
        for key, val in fukui_indices.items():
            if key in XTB_FUKUI_PROPERTY_TYPES:
                prop_class = XTB_FUKUI_PROPERTY_TYPES[key]
                return_properties[key] = prop_class(val)

    return return_properties


def optimize_geometry(
    mol: Chem.Mol,
    conf_id: int,
    keep_files: bool = False,
    scr: Path = DEFAULT_SCR,
    n_cores: int = 1,
    template_str: str | None = None,
    xtb_command_add: list[str] | None = None,
    solvation_model: str | None = None,
    solvent: str | None = None,
    use_json: bool = True,
    **template_kwargs,
) -> tuple[np.ndarray | None, dict[str, calculated_properties.Property] | None]:
    """Optimize the geometry of a conformation using xtb.

    Args:
        mol (Chem.Mol):
            RDKit molecule object with 3D coordinates for the specified conformer
        conf_id (int):
            ID of the conformer to optimize (typically 0 for single conformer)
        keep_files (bool):
            If True, keep intermediate files in scratch directory. Defaults to False.
            Can be useful for debugging.
        scr (Path):
            Scratch directory path for temporary files. Defaults to DEFAULT_SCR.
        n_cores (int):
            Number of CPU cores to use. Defaults to 1. Sets OMP_NUM_THREADS
            and other parallelization environment variables.
        template_str (str):
            Jinja2 template string for xcontrol input file.
            Defaults to :data:`XCONTROL_TEMPLATE_OPT`.
        xtb_command_add (list[str] | None):
            Additional command-line arguments to pass to xtb.
            Defaults to None (empty list).
        solvation_model (str | None):
            Solvation model to use ("alpb", "gbsa", "cosmo" or "cpmcx"). Defaults to None (no solvation).
        solvent (str | None):
            Solvent name (e.g., "water", "methanol") if solvation_model is specified. Defaults to None.
        use_json (bool):
            Whether to request xtb's ``--json`` output. If False (or if the JSON
            file is missing), properties are parsed from stdout. Defaults to True.
        **template_kwargs: Additional keyword arguments passed to the xcontrol template
                 (e.g., max_num_steps, optlevel, constrain, torsions_fixed, etc.)

    Returns:
        coordinates: Numpy array of shape (n_atoms, 3) containing optimized atomic
                    coordinates in Angstrom units.
        properties: Dictionary containing xtb properties calculated in the final single-point
                    calculation after geometry optimization.

    Raises:
        RuntimeError: If any error occurs during optimization.
    """
    if xtb_command_add is None:
        xtb_command_add = list()
    if "--opt" not in xtb_command_add:
        xtb_command_add.append("--opt")

    if template_str is None:
        template_str = XCONTROL_TEMPLATE_OPT

    temp, stdout, stderr = run_xtb(
        mol,
        conf_id,
        template_str,
        keep_files=keep_files,
        scr=scr,
        n_cores=n_cores,
        xtb_command_add=xtb_command_add,
        solvation_model=solvation_model,
        solvent=solvent,
        use_json=use_json,
        **template_kwargs,
    )

    # Read properties from json file
    try:
        opt_results = read_xtb_output(temp, stdout, stderr, read_json=True)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise RuntimeError("Error reading xtb optimization output") from exc

    optimized_xyz_path = temp.get_path() / COORDINATES_OUTPUT_FILENAME
    if not optimized_xyz_path.exists():
        raise RuntimeError(
            f"xtb optimization did not produce expected output file: {optimized_xyz_path}"
        )

    # Read coordinates from XYZ file
    symbols, coordinates = read_xyz_coordinates(optimized_xyz_path)

    # Validate that atom symbols match
    if len(symbols) != mol.GetNumAtoms():
        error_msg = (
            f"Geometry optimization failed for conformer {conf_id}: "
            f"Optimized geometry has {len(symbols)} atoms but input molecule has "
            f"{mol.GetNumAtoms()} atoms"
        )
        raise RuntimeError(error_msg)

    for i, (xyz_symbol, mol_atom) in enumerate(zip(symbols, mol.GetAtoms())):
        mol_symbol = mol_atom.GetSymbol()
        if xyz_symbol.lower() != mol_symbol.lower():
            error_msg = (
                f"Geometry optimization failed for conformer {conf_id}: "
                f"Atom {i}: optimized geometry has {xyz_symbol} but input molecule "
                f"has {mol_symbol}"
            )
            raise RuntimeError(error_msg)

    # check if optimization has succeeded
    has_converged = get_optimization_converged(temp, stdout, stderr)
    opt_results[OPTIMIZATION_CONVERGED_KEY] = calculated_properties.BoolProperty(has_converged)

    return coordinates, opt_results


def get_optimization_converged(
    temp: WorkDir,
    stdout: str,
    stderr: str,
) -> bool:
    """Checks whether an xtb geometry optimization converged successfully.

    Parses stdout/stderr for xtb's convergence status messages. If neither a
    "converged" nor a "failed to converge" message is found, falls back to
    checking for the presence of the ``.xtboptok`` flag file.

    Args:
        temp: WorkDir object containing the scratch directory path.
        stdout: Standard output from the xtb calculation.
        stderr: Standard error from the xtb calculation.

    Returns:
        bool: True if the optimization converged, False otherwise.
    """
    lines = stdout.splitlines() + stderr.splitlines()
    for line in lines:
        if OPTIMIZATION_CONVERGED_PATTERN in line:
            return True
        if OPTIMIZATION_FAILED_PATTERN in line:
            return False

    opt_flag_file = temp.get_path() / OPTIMIZATION_CONVERGED_FILE
    return opt_flag_file.exists()


def calculate_single_point(
    mol: Chem.Mol,
    conf_id: int,
    keep_files: bool = False,
    scr: Path = DEFAULT_SCR,
    n_cores: int = 1,
    template_str: str | None = None,
    xtb_command_add: list[str] | None = None,
    solvation_model: str | None = None,
    solvent: str | None = None,
    calculate_fukui: bool = False,
    use_json: bool = True,
    **template_kwargs,
) -> dict[str, calculated_properties.Property]:
    """Calculate single-point properties using xtb.

    Args:
        mol (Chem.Mol):
            RDKit molecule object with 3D coordinates for the specified conformer
        conf_id (int):
            ID of the conformer to optimize (typically 0 for single conformer)
        keep_files (bool):
            If True, keep intermediate files in scratch directory. Defaults to False.
            Can be useful for debugging.
        scr (Path):
            Scratch directory path for temporary files. Defaults to DEFAULT_SCR.
        n_cores (int):
            Number of CPU cores to use. Defaults to 1. Sets OMP_NUM_THREADS
            and other parallelization environment variables.
        template_str (str):
            Jinja2 template string for xcontrol input file.
            Defaults to :data:`XCONTROL_TEMPLATE_SP`.
        xtb_command_add (list[str] | None):
            Additional command-line arguments to pass to xtb.
            Defaults to None (empty list).
        solvation_model (str | None):
            Solvation model to use ("alpb", "gbsa", "cosmo" or "cpmcx"). Defaults to None (no solvation).
        solvent (str | None):
            Solvent name (e.g., "water", "methanol") if solvation_model is specified. Defaults to None.
        calculate_fukui (bool):
            If True, calculate Fukui indices and include them in the output. This will add the flag
            --vfukui to the xtb_command_add if not already present.
            Defaults to False.
        use_json (bool):
            Whether to request xtb's ``--json`` output. If False (or if the JSON
            file is missing), properties are parsed from stdout. Defaults to True.
        **template_kwargs: Additional keyword arguments passed to the xcontrol template
                 (e.g., max_num_steps, optlevel, constrain, torsions_fixed, etc.)

    Returns:
        dict[str, calculated_properties.Property]:
            Dictionary of single-point properties from xtb.

    Raises:
        RuntimeError: If any error occurs during calculation.
        ValueError: If solvent is not specified when solvation_model is provided.
    """

    if xtb_command_add is None:
        xtb_command_add = list()

    if calculate_fukui and "--vfukui" not in xtb_command_add:
        xtb_command_add.append("--vfukui")

    if template_str is None:
        template_str = XCONTROL_TEMPLATE_SP

    temp, stdout, stderr = run_xtb(
        mol,
        conf_id,
        template_str,
        keep_files=keep_files,
        scr=scr,
        n_cores=n_cores,
        xtb_command_add=xtb_command_add,
        solvation_model=solvation_model,
        solvent=solvent,
        use_json=use_json,
        **template_kwargs,
    )

    try:
        sp_results = read_xtb_output(
            temp,
            stdout,
            stderr,
            read_json=True,
            read_fukui=calculate_fukui,
            read_bondorders=True,
        )
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise RuntimeError("Error reading xtb single-point output") from exc

    return sp_results
