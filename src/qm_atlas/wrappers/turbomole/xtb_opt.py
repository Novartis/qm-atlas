import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

from qm_atlas.constants import DEFAULT_SCR
from qm_atlas.software_environment import SOFTWARE_CONFIG
from qm_atlas.tasks import common as calculated_properties
from qm_atlas.wrappers import xtb
from qm_atlas.wrappers.common import run_command
from qm_atlas.wrappers.turbomole import jobex, utils

_logger = logging.getLogger(__name__)


DEFAULT_XTB_TM_STEPS = 100
DEFAULT_OPT_LEVEL = "normal"
OUT_FILENAME = "tm_opt.log"

XCONTROL_TEMPLATE_CSTR = (
    """$chrg {{ charge | default(0) }}
$spin {{ num_unpaired_electrons | default(0) }}

$opt
  maxcycle={{ max_num_steps | default(100) }}
  optlevel={{ opt_level | default("normal") }} {% if constrain %}"""
    + xtb.CONSTRAIN_BLOCK_TEMPLATE
    + """{% endif %}
"""
)


def run_xtb_tm_optimization(
    mol: Chem.Mol,
    conf_id: int,
    keep_files: bool = False,
    scr: Path = DEFAULT_SCR,
    n_cores: int = 1,
    tm_control_template: str = jobex.PREPARE_JOBEX,
    xtb_command_add: list[str] | None = None,
    xcontrol_template: str | None = XCONTROL_TEMPLATE_CSTR,
    scf_conv: int = 7,
    grid: str = "m4",
    basis: str = jobex.DEFAULT_BASIS,
    functional: Sequence[str] = jobex.DEFAULT_FUNCTIONAL,
    use_disp: bool = True,
    scfiterlimit: int | None = None,
    use_cosmo: bool = True,
    solvent: str = "conductor",
    max_num_steps: int = DEFAULT_XTB_TM_STEPS,
    opt_level: str = DEFAULT_OPT_LEVEL,
    constrain: bool = False,
    torsions_fixed: bool = False,
    heavy_atom_torsions_fixed: bool = True,
    angles_fixed: bool = False,
    bonds_fixed: bool = False,
    force_constant: float = xtb.DEFAULT_FORCE_CONSTANT,
    **kwargs,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    """Optimize molecule geometry using xtb-driven turbomole DFT calculations.

    Uses xtb to drive the geometry optimization while turbomole calculates the
    gradients at the DFT level. This combines the efficiency of xtb's optimizer
    with the accuracy of DFT gradients.

    Args:
        mol:
            RDKit molecule object with 3D coordinates for the specified conformer
        conf_id:
            ID of the conformer to optimize (typically 0 for single conformer)
        keep_files:
            If True, keep intermediate files in scratch directory. Defaults to False.
        scr:
            Scratch directory path for temporary files. Defaults to DEFAULT_SCR.
        n_cores:
            Number of CPU cores to use. Defaults to 1. Sets OMP_NUM_THREADS
                 and other parallelization environment variables.
        tm_control_template: Jinja2 template string for turbomole control file.
                            Defaults to :data:`~qm_atlas.wrappers.turbomole.jobex.PREPARE_JOBEX`.
        xtb_command_add: Additional command-line arguments to pass to xtb.
                        Defaults to None (empty list).
        xcontrol_template: Jinja2 template string for xtb xcontrol input file.
                          Defaults to :data:`XCONTROL_TEMPLATE_CSTR`.
        scf_conv:
            SCF energy convergence threshold as 10**(-scf_conv). Defaults to 7.
        grid:
            Grid size for turbomole DFT calculation (m3, m4, etc.). Defaults to "m4".
        basis:
            Basis set for turbomole calculation. Defaults to "def2-TZVP".
        functional:
            DFT functional(s) for turbomole. Defaults to ("b-p",).
        use_disp:
            If True, adds dispersion correction ($disp3 bj). Defaults to True.
        scfiterlimit:
            Maximum number of SCF iterations. Defaults to None (not set).
        use_cosmo:
            If True, uses the COSMO implicit solvent model. Defaults to True.
        solvent:
            Implicit solvent model. Defaults to "conductor".
        max_num_steps:
            Maximum number of optimization steps. Defaults to :data:`DEFAULT_XTB_TM_STEPS` (100).
        opt_level:
            Optimization level for xtb (e.g., "loose", "normal", "tight").
            Defaults to :data:`DEFAULT_OPT_LEVEL` ("normal").
        constrain:
            Master switch: if True, applies the geometry constraints selected by the
            boolean flags below. Defaults to False.
        torsions_fixed:
            Fix all torsions via xtb's global ``all torsions`` switch
            (only applies if ``constrain``). Defaults to False.
        heavy_atom_torsions_fixed:
            Fix heavy-atom torsions by writing one explicit ``dihedral: i,j,k,l,auto``
            restraint per heavy-atom torsion, leaving bonds and angles free
            (only applies if ``constrain``). Defaults to True.
        angles_fixed:
            Fix all bond angles via xtb's global ``all angles`` switch
            (only applies if ``constrain``). Defaults to False.
        bonds_fixed:
            Fix all bond lengths via xtb's global ``all bonds`` switch
            (only applies if ``constrain``). Defaults to False.
        force_constant:
            Force constant for constraints. Defaults to :data:`~qm_atlas.wrappers.xtb.DEFAULT_FORCE_CONSTANT` (0.5).
        **kwargs:
            Additional keyword arguments passed to turbomole control template and the
            template for the xcontrol file. Note that the filling of templates can handle
            keyword arguments that are not used in the templates.

    Returns:
        coordinates:
            Numpy array of shape (n_atoms, 3) containing optimized atomic
            coordinates in Angstrom units. Returns None if optimization failed.
        properties:
            Dictionary containing optimization properties (convergence status, etc.).
            Returns None if optimization failed.

    Raises:
        RuntimeError:
            If xtb-turbomole calculation fails.
        ValueError:
            If optimized atom symbols don't match input molecule
    """

    if xtb_command_add is None:
        xtb_command_add = []
    if "--opt" not in xtb_command_add:
        xtb_command_add.append("--opt")

    # Prepare turbomole input files
    template_options = {
        "basis": basis,
        "functional": functional,
        "use_disp": use_disp,
        "scf_convergence": scf_conv,
        "scfiterlimit": scfiterlimit,
        "grid": grid,
    }

    temp = utils.prepare_turbomole_input(
        mol,
        tm_control_template,
        conf_id=conf_id,
        keep_files=keep_files,
        scr=scr,
        use_cosmo=use_cosmo,
        solvent=solvent,
        **template_options,
        **kwargs,
    )

    scr_path = temp.get_path()

    # Build xtb command with turbomole-specific flags
    software_env_manager = SOFTWARE_CONFIG.get_environment_manager()
    xtb_cmd = software_env_manager.get_command("xtb", "xtb")
    cmd_parts = [xtb_cmd, "--tm", "coord"]

    # Write xcontrol file for xtb
    constraint_file = xtb.write_xcontrol_file(
        mol,
        temp,
        xcontrol_template,
        conf_id=conf_id,
        max_num_steps=max_num_steps,
        opt_level=opt_level,
        constrain=constrain,
        torsions_fixed=torsions_fixed,
        heavy_atom_torsions_fixed=heavy_atom_torsions_fixed,
        angles_fixed=angles_fixed,
        bonds_fixed=bonds_fixed,
        force_constant=force_constant,
        **kwargs,
    )
    cmd_parts.extend(["--input", str(constraint_file.name)])

    # Add additional command arguments
    cmd_parts.extend(xtb_command_add)
    cmd = " ".join(cmd_parts)

    # Execute xtb-driven turbomole optimization
    env = software_env_manager.get_run_environment("xtb_turbomole", n_cores=n_cores)
    stdout, stderr = run_command(cmd, env=env, cwd=scr_path)

    check_xtb_tm_optimization(stdout, stderr, conf_id)

    coord_file = scr_path / "coord"
    labels = [atom.GetSymbol() for atom in mol.GetAtoms()]
    new_coords = jobex.parse_coord(coord_file, labels)

    # Validate atom symbols match
    if len(new_coords) != mol.GetNumAtoms():
        error_msg = (
            f"Geometry optimization failed for conformer {conf_id}: "
            f"Optimized geometry has {len(new_coords)} atoms but input molecule has "
            f"{mol.GetNumAtoms()} atoms"
        )
        raise RuntimeError(error_msg)

    for i, (coord_label, mol_atom) in enumerate(zip(labels, mol.GetAtoms())):
        mol_symbol = mol_atom.GetSymbol()
        if coord_label.lower() != mol_symbol.lower():
            error_msg = (
                f"Geometry optimization failed for conformer {conf_id}: "
                f"Atom {i}: optimized geometry has {coord_label} but input molecule "
                f"has {mol_symbol}"
            )
            raise RuntimeError(error_msg)

    # Check convergence and prepare results
    has_converged = xtb.get_optimization_converged(temp, stdout, stderr)
    opt_results = {
        xtb.OPTIMIZATION_CONVERGED_KEY: calculated_properties.BoolProperty(has_converged)
    }

    # Note: Unlike pure xtb, we don't have a JSON output, but we can add other properties later

    return new_coords, opt_results


def check_xtb_tm_optimization(
    stdout: str | None,
    stderr: str | None,
    conf_id: int,
):
    """Checks whether an xtb-driven turbomole geometry optimization finished successfully.
        Raises a RuntimeError in case it failed.

    Args:
        stdout (str | None):
            The standard output from the xtb-driven turbomole optimization.
        stderr (str | None):
            The standard error output from the optimization.
        conf_id (int):
            The ID of the conformation that was considered. Needed for logging.

    Raises:
        RuntimeError: If the job has failed.
    """

    xtb_tm_log_str = stdout if stdout is not None else ""

    if "GEOMETRY OPTIMIZATION CONVERGED" not in xtb_tm_log_str:
        _logger.warning(f"geometry optimization did not converge for conformer {conf_id}")
        jobex.log_debug_information(stdout, stderr)

    if stderr is not None and "ended normally" not in stderr:
        _logger.error(
            f"xtb-driven turbomole optimization ended abnormally for conformer {conf_id}"
        )
        jobex.log_debug_information(stdout, stderr)
        raise RuntimeError("Turbomole jobex ended abnormally")
