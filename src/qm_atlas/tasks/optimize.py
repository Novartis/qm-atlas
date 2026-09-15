"""Geometry optimization workflows supporting multiple drivers.

This module provides a unified interface for geometry optimization using different
computational chemistry backends. Currently, the following optimization drivers are
supported:

- **Turbomole/jobex**: DFT-level geometry optimization.
- **xtb-driven Turbomole**: DFT-level geometry optimization using the xtb optimizer
  and turbomole for the energy/gradient calculations.
- **xtb only**: semi-empirical geometry optimization (fast).

All functions follow the same interface and support parallel execution.
"""

import logging
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rdkit import Chem, Geometry
from rdkit.Numerics import rdAlignment

from qm_atlas.constants import DEFAULT_SCR
from qm_atlas.tasks import common as calculated_properties
from qm_atlas.tasks.utils import conformer_geometry
from qm_atlas.tasks.utils.parallel import run_parallel
from qm_atlas.wrappers import xtb
from qm_atlas.wrappers.turbomole import jobex, xtb_opt

_logger = logging.getLogger(__name__)

JOBEX_NAME = "jobex"
XTBTM_NAME = "xtb_turbomole"
XTB_NAME = "xtb"

XTB_OPTIMIZATION_LEVELS = Literal[
    "crude",
    "sloppy",
    "loose",
    "lax",
    "normal",
    "tight",
    "verytight",
    "extreme",
]

OPTIMIZER_FUNCTIONS = {
    JOBEX_NAME: jobex.run_jobex,
    XTBTM_NAME: xtb_opt.run_xtb_tm_optimization,
    XTB_NAME: xtb.optimize_geometry,
}


class OptimizationOptions(BaseModel):
    """Base class for geometry optimization backends.

    Each concrete backend declares a unique ``backend`` literal that acts as the
    discriminator for (de)serialization within :data:`OptimizationConfig`.
    """


class XtbOptions(OptimizationOptions):
    """Options specific to the xtb optimization driver."""

    # Every field only fills the xcontrol template, so reject unknown keys as typos.
    model_config = ConfigDict(extra="forbid")

    backend: Literal["xtb"] = Field(
        default="xtb",
        description="Optimization backend to use",
    )
    max_num_steps: int = Field(
        default=100,
        description="Maximum number of optimization steps for xtb",
        ge=1,
    )
    method: str = Field(
        default="2",
        description="xtb method to use, e.g. '2' for GFN2-xTB, '1' for GFN1-xTB, '0' for GFN0-xTB",
    )
    opt_level: XTB_OPTIMIZATION_LEVELS = Field(
        default="normal",
        description="xtb optimization level",
    )
    solvation_model: str | None = Field(
        default="alpb",
        description="solvation model for xtb (e.g. 'alpb', 'gbsa', 'cosmo', or None for gas phase)",
    )
    solvent: str | None = Field(
        default="water",
        description="Solvent to use if a solvation model is specified",
    )
    constrain: bool = Field(
        default=False,
        description="Master switch: whether to apply any geometry constraints during optimization",
    )
    torsions_fixed: bool = Field(
        default=False,
        description="Fix all torsions via xtb's global 'all torsions' switch (only applies if constrain=True)",
    )
    heavy_atom_torsions_fixed: bool = Field(
        default=True,
        description=(
            "Fix heavy-atom torsions by writing one explicit 'dihedral: i,j,k,l,auto' "
            "restraint per heavy-atom torsion, leaving bonds and angles free "
            "(only applies if constrain=True)"
        ),
    )
    angles_fixed: bool = Field(
        default=False,
        description="Fix all angles via xtb's global 'all angles' switch (only applies if constrain=True)",
    )
    bonds_fixed: bool = Field(
        default=False,
        description="Fix all bonds via xtb's global 'all bonds' switch (only applies if constrain=True)",
    )
    force_constant: float = Field(
        default=xtb.DEFAULT_FORCE_CONSTANT,
        description="Force constant for constraints in xtb optimization (in Hartree/rad^2 for angles or torsions, Hartree/Bohr^2 for distances)",
    )

    @model_validator(mode="after")
    def check_solvation(self) -> "XtbOptions":
        if self.solvation_model is not None and self.solvent is None:
            raise ValueError("If solvation_model is specified, solvent must also be specified")
        return self


class JobexOptions(OptimizationOptions):
    """Options specific to the Turbomole jobex optimization driver."""

    # Every field only fills the control template, so reject unknown keys as typos.
    model_config = ConfigDict(extra="forbid")

    backend: Literal["jobex"] = Field(
        default="jobex",
        description="Optimization backend to use",
    )
    max_num_steps: int = Field(
        default=100,
        description="Maximum number of optimization steps for jobex",
        ge=1,
    )
    energy_conv: int = Field(
        default=jobex.DEFAULT_ENERGY_CONV,
        description="Sets geometry iteration energy convergence threshold to 10.**(-energy_conv). Higher values lead to tigher convergence.",
    )
    geom_conv: int = Field(
        default=jobex.DEFAULT_GEOM_CONV,
        description="Sets coordinate change convergence threshold to 10.**(-geom_conv). Higher values lead to tighter convergence.",
    )
    functional: list[str] = Field(
        default=["b-p"],
        description="Functional for geometry optimization (e.g., 'b-p' for BP86)",
    )
    basis: str = Field(
        default="def2-TZVP",
        description="Basis set for geometry optimization",
    )
    scf_conv: int = Field(
        default=jobex.DEFAULT_SCF_CONV,
        description="Sets scf energy convergence threshold to 10.**(-scf_conv), should be at least as tight as geom_conv, otherwise geometry iterations will have trouble converging.",
    )
    scfiterlimit: int = Field(
        default=jobex.DEFAULT_SCFITERLIMIT,
        description="The maximum number of SCF iterations to perform.",
    )
    use_disp: bool = Field(
        default=False,
        description="Whether to add dispersion correction ($disp3 bj) in jobex optimization",
    )
    use_cosmo: bool = Field(
        default=True,
        description="Whether to use the COSMO implicit solvent model",
    )
    solvent: str = Field(
        default="conductor",
        description="Solvent environment (conductor, water, etc.). Only used if use_cosmo is True. Translates to an epsilon value in turbomole.",
    )


class XtbTurbomoleOptions(OptimizationOptions):
    """Options specific to the xtb-driven Turbomole optimization driver."""

    # Every field only fills the control/xcontrol templates, so reject unknown keys as typos.
    model_config = ConfigDict(extra="forbid")

    backend: Literal["xtb_turbomole"] = Field(
        default="xtb_turbomole",
        description="Optimization backend to use",
    )
    max_num_steps: int = Field(
        default=100,
        description="Maximum number of optimization steps for jobex",
        ge=1,
    )
    opt_level: XTB_OPTIMIZATION_LEVELS = Field(
        default="normal",
        description="xtb optimization level",
    )
    functional: list[str] = Field(
        default=["b-p"],
        description="Functional for geometry optimization (e.g., 'b-p' for BP86)",
    )
    basis: str = Field(
        default="def2-TZVP",
        description="Basis set for geometry optimization",
    )
    scf_conv: int = Field(
        default=jobex.DEFAULT_SCF_CONV,
        description="Sets scf energy convergence threshold to 10.**(-scf_conv), should be at least as tight as geom_conv, otherwise geometry iterations will have trouble converging.",
    )
    scfiterlimit: int = Field(
        default=jobex.DEFAULT_SCFITERLIMIT,
        description="The maximum number of SCF iterations to perform.",
    )
    use_disp: bool = Field(
        default=False,
        description="Whether to add dispersion correction ($disp3 bj) in jobex optimization",
    )
    use_cosmo: bool = Field(
        default=True,
        description="Whether to use the COSMO implicit solvent model",
    )
    solvent: str = Field(
        default="conductor",
        description="Solvent environment (conductor, water, etc.). Only used if use_cosmo is True. Translates to an epsilon value in turbomole.",
    )
    constrain: bool = Field(
        default=False,
        description="Master switch: whether to apply any geometry constraints during optimization",
    )
    torsions_fixed: bool = Field(
        default=False,
        description="Fix all torsions via xtb's global 'all torsions' switch (only applies if constrain=True)",
    )
    heavy_atom_torsions_fixed: bool = Field(
        default=True,
        description=(
            "Fix heavy-atom torsions by writing one explicit 'dihedral: i,j,k,l,auto' "
            "restraint per heavy-atom torsion, leaving bonds and angles free "
            "(only applies if constrain=True)"
        ),
    )
    angles_fixed: bool = Field(
        default=False,
        description="Fix all angles via xtb's global 'all angles' switch (only applies if constrain=True)",
    )
    bonds_fixed: bool = Field(
        default=False,
        description="Fix all bonds via xtb's global 'all bonds' switch (only applies if constrain=True)",
    )
    force_constant: float = Field(
        default=xtb.DEFAULT_FORCE_CONSTANT,
        description="Force constant for constraints in xtb optimization (in Hartree/rad^2 for angles or torsions, Hartree/Bohr^2 for distances)",
    )


OPTIONS_CLASS_REGISTRY = {
    XTB_NAME: XtbOptions,
    JOBEX_NAME: JobexOptions,
    XTBTM_NAME: XtbTurbomoleOptions,
}

# Discriminated union of all optimization config types. The ``backend`` literal
# field acts as the discriminator so Pydantic can select the concrete class
# automatically when deserializing from a dict, and serialize all fields when
# calling model_dump().
OptimizationConfig = Annotated[
    XtbOptions | JobexOptions | XtbTurbomoleOptions,
    Field(discriminator="backend"),
]


def optimize_with_fallback(
    mol: Chem.Mol,
    conf_id: int,
    config: list[XtbOptions | JobexOptions | XtbTurbomoleOptions],
    treat_unconverged_as_failure: bool = True,
    suppress_errors: bool = True,
    log_errors: bool = True,
    n_cores: int = 1,
    scr: Path = DEFAULT_SCR,
    keep_files: bool = False,
) -> tuple[np.ndarray | None, dict[str, calculated_properties.Property] | None]:
    """Optimize a conformer with automatic fallback to alternative drivers on failure.

    Attempts to optimize a single conformer using the specified optimization driver.
    If optimization fails (raises an exception) or does not converge (when
    treat_unconverged_as_failure=True), automatically attempts fallback drivers in
    the order specified. This provides robustness by allowing optimization to
    fall back to less sophisticated but more stable methods.

    Args:
        mol (Chem.Mol):
            RDKit molecule containing the conformer to optimize.
        conf_id (int):
            ID of the conformer within the molecule to optimize.
        config (Sequence[XtbOptions | JobexOptions | XtbTurbomoleOptions]):
            Primary optimization configuration or list of configurations to try in order.
        treat_unconverged_as_failure (bool):
            If True, treats unconverged optimizations as failures and attempts fallbacks.
            If False, returns the results of the primary optimization even if unconverged.
            Defaults to True.
        suppress_errors (bool):
            If True, returns (None, None) on failure instead of raising an exception.
            If False, raises RuntimeError on optimization failure. Defaults to True.
        log_errors (bool):
            If True, logs an error when all optimization attempts fail. Set to False
            in bulk/parallel runs where a summary warning is emitted instead.
            Defaults to True.
        n_cores (int):
            Number of CPU cores to use for the optimization. Defaults to 1.
        scr (Path):
            Path to the scratch directory for temporary files. Defaults to DEFAULT_SCR.
        keep_files (bool):
            If True, keeps intermediate files generated during optimization. Defaults to False.

    Returns:
        tuple[np.ndarray | None, dict[str, Property] | None]:
            The optimized coordinates and the dict of calculated properties, or
            (None, None) if optimization failed and suppress_errors=True.
    """

    new_coords = mol.GetConformer(conf_id).GetPositions()
    properties: dict[str, calculated_properties.Property] = {
        xtb.OPTIMIZATION_CONVERGED_KEY: calculated_properties.BoolProperty(True)
    }
    optimization_successful = True

    for current_config in config:

        optimizer = OPTIMIZER_FUNCTIONS[current_config.backend]
        current_kwargs = current_config.model_dump(exclude={"backend"})
        try:
            new_coords, properties = optimizer(
                mol, conf_id, n_cores=n_cores, scr=scr, keep_files=keep_files, **current_kwargs
            )
        except RuntimeError as exc:
            optimization_successful = False
            _logger.warning(
                f"Optimization with backend '{current_config.backend}' failed: {exc}. "
                "Trying next fallback if available..."
            )
            continue

        converged = properties.get(xtb.OPTIMIZATION_CONVERGED_KEY, None)
        if treat_unconverged_as_failure and (
            converged is None or converged.get_property_value() is False
        ):
            optimization_successful = False
            continue

        optimization_successful = True
        break

    if not optimization_successful:
        if log_errors:
            _logger.error("All optimization attempts failed.")
        if suppress_errors:
            return None, None
        raise RuntimeError("All optimization attempts failed.")

    return new_coords, properties


def optimize_conformer(
    conformer: Chem.Conformer,
    config: list[XtbOptions | JobexOptions | XtbTurbomoleOptions],
    treat_unconverged_as_failure: bool = True,
    suppress_errors: bool = True,
    n_cores: int = 1,
    scr: Path = DEFAULT_SCR,
    keep_files: bool = False,
) -> tuple[Chem.Conformer | None, dict[str, calculated_properties.Property] | None]:
    """Run geometry optimization on a single conformer.

    Optimizes the 3D coordinates of a conformer using the specified computational
    chemistry backend. The conformer's parent molecule must have proper connectivity
    information (bonds, charges, etc.).

    Args:
        conformer: RDKit conformer to optimize. Must have a parent molecule with
            proper connectivity information.
        config: Optimization configuration(s) to try in order (each selects a
            backend and its options).
        n_cores: Number of CPU cores to use for the optimization. Defaults to 1.
        scr: Path to the scratch directory for temporary files. Defaults to DEFAULT_SCR.
        treat_unconverged_as_failure: If True, treats unconverged optimizations as failures and attempts fallbacks.
            If False, returns the results of the primary optimization even if unconverged. Defaults to True.
        suppress_errors: If True, returns (None, None) on failure instead of raising an exception.
            If False, raises RuntimeError on optimization failure. Defaults to True.
        keep_files: If True, keeps intermediate files generated during optimization. Defaults to False.

    Returns:
        RDKit Conformer: The conformer with optimized coordinates, or None if
            optimization failed and suppress_errors=True.

    Raises:
        RuntimeError: If optimization fails and suppress_errors=False.
        KeyError: If an invalid driver is specified.

    Examples:
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> mol = Chem.AddHs(mol)
        >>> AllChem.EmbedMolecule(mol)
        >>> conformer = mol.GetConformer()
        >>> opt_conf = optimize_conformer(conformer, config=[XtbTurbomoleOptions()], n_cores=4)
    """

    conf_id = conformer.GetId()
    mol = conformer.GetOwningMol()

    # Run optimization using the appropriate driver
    new_coords, properties = optimize_with_fallback(
        mol,
        conf_id,
        config=config,
        n_cores=n_cores,
        treat_unconverged_as_failure=treat_unconverged_as_failure,
        suppress_errors=suppress_errors,
        scr=scr,
        keep_files=keep_files,
    )
    try:
        new_conformer = adapt_positions(conformer, new_coords)
        return new_conformer, properties
    except (RuntimeError, ValueError) as exc:
        if suppress_errors:
            return None, None
        raise RuntimeError from exc


def optimize_mol(
    mol: Chem.Mol,
    config: list[XtbOptions | JobexOptions | XtbTurbomoleOptions],
    treat_unconverged_as_failure: bool = True,
    n_cores: int = 1,
    scr: Path = DEFAULT_SCR,
    keep_files: bool = False,
    show_progress: bool = True,
) -> tuple[Chem.Mol, list[dict[str, calculated_properties.Property]]]:
    """Run geometry optimization on all conformers of a molecule in parallel.

    Optimizes all conformers in a molecule using the specified driver. Conformers
    are processed in parallel to utilize multiple CPU cores efficiently.

    Args:
        mol (Chem.Mol):
            RDKit molecule with one or more conformers to optimize.
        config (list[XtbOptions | JobexOptions | XtbTurbomoleOptions]):
            Optimization configuration(s) to try in order (each selects a backend
            and its options; later entries act as fallbacks).
        treat_unconverged_as_failure (bool):
            If True, treats unconverged optimizations as failures and attempts the
            next fallback configuration. Defaults to True.
        n_cores (int):
            Number of CPU cores to use for parallel processing. Defaults to 1 (sequential).
        scr (Path):
            Path to the scratch directory for temporary files. Defaults to DEFAULT_SCR.
        keep_files (bool):
            If True, keeps intermediate files generated during optimization. Defaults to False.
        show_progress (bool):
            If True, displays a progress bar. Defaults to True.

    Returns:
        RDKit Molecule: New molecule with optimized conformers. Failed conformers
            are either removed or kept with original coordinates, depending on
            keep_failures setting.

    Raises:
        ValueError: If an invalid driver is specified.

    Examples:
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> mol = Chem.AddHs(mol)
        >>> AllChem.EmbedMultipleConfs(mol, 5)
        >>> opt_mol = optimize_mol(mol, config=[XtbTurbomoleOptions()], n_cores=4)
    """

    optimized_mols, calculated_properties_list = optimize_mols(
        [mol],
        n_cores=n_cores,
        show_progress=show_progress,
        config=config,
        treat_unconverged_as_failure=treat_unconverged_as_failure,
        scr=scr,
        keep_files=keep_files,
    )

    return optimized_mols[0], calculated_properties_list[0]


def optimize_mols(
    molecules: list[Chem.Mol],
    config: list[XtbOptions | JobexOptions | XtbTurbomoleOptions],
    n_cores: int = 1,
    treat_unconverged_as_failure: bool = True,
    scr: Path = DEFAULT_SCR,
    keep_files: bool = False,
    show_progress: bool = True,
) -> tuple[list[Chem.Mol], list[list[dict[str, calculated_properties.Property]]]]:
    """Run geometry optimization on all conformers of multiple molecules in parallel.

    Optimizes all conformers in a list of molecules using the specified driver.
    All conformers are processed in parallel, regardless of which molecule they
    belong to. This is most efficient when optimizing many conformers (>10).

    Args:
        molecules: List of RDKit molecules with conformers to optimize. All
            molecules should have the same structure (connectivity, charges, etc.)
            for optimal performance.
        config: Optimization configuration(s) to try in order (each selects a
            backend and its options; later entries act as fallbacks).
        n_cores: Number of CPU cores to use for parallel processing. The actual
            number of processes spawned will be min(n_cores, total_num_conformers).
            Defaults to 1 (sequential processing).
        treat_unconverged_as_failure: If True, treats unconverged optimizations as
            failures and attempts the next fallback configuration. Defaults to True.
        scr: Path to the scratch directory for temporary files. Defaults to DEFAULT_SCR.
        keep_files: If True, keeps intermediate files generated during optimization.
            Defaults to False.
        show_progress: If True, displays a progress bar. Defaults to True.

    Returns:
        List of molecules with optimized conformers. The output list has the same
        length as the input list, with conformers reorganized by molecule.

    Raises:
        ValueError: if molecules have no conformers.
        KeyError: If an invalid driver is specified.

    Examples:
        >>> mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CC")]
        >>> for mol in mols:
        ...     mol = Chem.AddHs(mol)
        ...     AllChem.EmbedMultipleConfs(mol, 3)
        >>> opt_mols = optimize_mols(mols, config=[XtbTurbomoleOptions()], n_cores=8)
    """

    # Extract conformer information
    n_confs = [mol.GetNumConformers() for mol in molecules]
    conf_ids, mols = [], []

    for mol in molecules:
        for conformer in mol.GetConformers():
            conf_ids.append(conformer.GetId())
            mols.append(mol)

    if not conf_ids:
        raise ValueError("No conformers found in input molecules")

    result_list = run_parallel(
        optimize_with_fallback,
        list(zip(mols, conf_ids)),
        config=config,
        treat_unconverged_as_failure=treat_unconverged_as_failure,
        n_cores=n_cores,
        show_progress=show_progress,
        func_has_ncores_arg=True,
        scr=scr,
        rdkit_pickle_properties=Chem.PropertyPickleOptions.AllProps,
        suppress_errors=True,  # Set suppress_errors=True to handle failures gracefully
        log_errors=False,  # Rely on the aggregated warning in update_mol instead
        keep_files=keep_files,
    )

    # Reorganize results by molecule
    updated_mols = []
    calculated_properties_list = []
    start_idx = 0
    for n_conf, mol in zip(n_confs, molecules):
        stop_idx = start_idx + n_conf
        new_coords_lst = [result[0] for result in result_list[start_idx:stop_idx]]
        updated_mol, failed_idcs = update_mol(mol, new_coords_lst)
        updated_mols.append(updated_mol)
        calculated_properties_lst = [
            result[1]
            for idx, result in enumerate(result_list[start_idx:stop_idx])
            if idx not in failed_idcs
        ]
        calculated_properties_list.append(calculated_properties_lst)
        start_idx = stop_idx

    return updated_mols, calculated_properties_list


def update_mol(
    mol: Chem.Mol,
    new_coordinates_lst: list[np.ndarray | None],
) -> tuple[Chem.Mol, list[int]]:
    """Update a molecule with new optimized coordinates.

    Creates a new molecule with updated conformers, replacing the original ones.
    Failed optimizations (None coordinates) are either removed or kept with original
    coordinates based on the keep_failures flag.

    Args:
        mol: RDKit molecule with conformers to update.
        new_coordinates_lst: List of coordinate arrays (one per original conformer).
            Each element is either a (n_atoms, 3) numpy array or None if optimization
            failed. The list length must match mol.GetNumConformers().

    Returns:
        New RDKit Molecule with updated conformers. The output molecule is a copy
        of the input, with all conformers replaced.

    Raises:
        ValueError: If the number of coordinate arrays doesn't match the number
            of conformers, or if coordinate shapes are invalid.

    Examples:
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> mol = Chem.AddHs(mol)
        >>> AllChem.EmbedMultipleConfs(mol, 2)
        >>> coords1 = np.array([[0,0,0], [1,1,1], ...])  # 8 atoms
        >>> coords2 = None  # Failed optimization
        >>> opt_mol = update_mol(mol, [coords1, coords2])
    """
    if len(new_coordinates_lst) != mol.GetNumConformers():
        raise ValueError(
            f"Number of coordinate arrays ({len(new_coordinates_lst)}) must match "
            f"number of conformers ({mol.GetNumConformers()})"
        )

    new_conformers = []
    failed_idcs = []

    for idx, (new_coords, conformer) in enumerate(zip(new_coordinates_lst, mol.GetConformers())):
        try:
            new_conformer = adapt_positions(conformer, new_coords)
            new_conformers.append(new_conformer)
        except (RuntimeError, ValueError):
            failed_idcs.append(idx)

    # Log warning if optimizations failed
    if failed_idcs:
        num_failures = len(failed_idcs)
        num_total = len(new_conformers) + num_failures
        mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else Chem.MolToSmiles(mol)
        _logger.warning(
            f"{num_failures} out of {num_total} conformers could not be optimized for {mol_name}"
        )

    # Create new molecule with updated conformers
    new_mol = Chem.Mol(mol)
    new_mol.RemoveAllConformers()
    for new_conf in new_conformers:
        new_mol.AddConformer(new_conf)

    return new_mol, failed_idcs


def adapt_positions(
    conformer: Chem.Conformer,
    coordinates: np.ndarray | None,
    rmsd_warning_threshold: float = 3.0,
    contact_scale: float | None = conformer_geometry.NONBONDED_CONTACT_SCALE,
) -> Chem.Conformer:
    """Create a conformer with updated atom positions.

    Takes a template conformer and new coordinate values, creating a new conformer
    with updated atom positions. Checks that the new coordinates are still consistent
    with the bond table of the molecule, i.e. that all declared bonds are intact and
    that no new bond was formed
    (see :func:`~qm_atlas.tasks.utils.conformer_geometry.check_geometry_consistency`).

    Args:
        conformer: Template RDKit conformer to use as basis. Its parent molecule
            must have the same number of atoms as the coordinates array.
        coordinates: Array of shape (n_atoms, 3) with new atomic coordinates in
            Angstroms. If None, raises RuntimeError.
        rmsd_warning_threshold: If the RMSD between old and new coordinates exceeds
            this value (in Angstroms), logs a warning. This can indicate unphysical
            geometry changes. Defaults to 3.0.
        contact_scale: Scale factor applied to the sum of the covalent radii below
            which two atoms that are not bonded are considered to have formed a bond.
            Pass None to skip that test. Defaults to
            :data:`~qm_atlas.tasks.utils.conformer_geometry.NONBONDED_CONTACT_SCALE`.

    Returns:
        New RDKit Conformer with updated atom positions.

    Raises:
        RuntimeError: If coordinates is None or if the new coordinates are not
            consistent with the bond table of the molecule.
        ValueError: If the number of coordinates doesn't match the number of atoms.

    Examples:
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> mol = Chem.AddHs(mol)
        >>> AllChem.EmbedMolecule(mol)
        >>> conf = mol.GetConformer()
        >>> new_coords = np.random.rand(mol.GetNumAtoms(), 3)  # Optimized coords
        >>> new_conf = adapt_positions(conf, new_coords)
    """
    if coordinates is None:
        raise RuntimeError("Optimization failed: no coordinates returned")

    num_atoms = conformer.GetOwningMol().GetNumAtoms()
    new_conformer = Chem.Conformer(conformer)

    if (
        len(coordinates.shape) != 2
        or coordinates.shape[1] != 3
        or coordinates.shape[0] != num_atoms
    ):
        raise ValueError(f"Coordinates must have shape (n_atoms, 3), got {coordinates.shape}")

    # Update atom positions
    for atom_idx in range(num_atoms):
        new_position = Geometry.Point3D(*coordinates[atom_idx, :])
        new_conformer.SetAtomPosition(atom_idx, new_position)

    # Check the new geometry against the bond table of the molecule
    violations = _check_conformer(conformer.GetOwningMol(), new_conformer, contact_scale)

    if violations:
        message = (
            "Optimized geometry is inconsistent with the bond table of the input molecule: "
            + ", ".join(violations)
        )
        _logger.error(message)
        raise RuntimeError(message)

    check_displacement(conformer, new_conformer, rmsd_warning_threshold)
    return new_conformer


def _check_conformer(
    mol: Chem.Mol,
    conformer: Chem.Conformer,
    contact_scale: float | None,
) -> list[conformer_geometry.GeometryViolationKind]:
    """Run the geometry consistency check for a single conformer of a molecule.

    Args:
        mol: The molecule whose bond table defines the expected connectivity.
        conformer: The conformer holding the coordinates to check.
        contact_scale: Scale factor for the non-bonded contact test, None skips it.

    Returns:
        The detected inconsistencies, empty if the geometry is consistent.
    """
    _mol = Chem.Mol(mol)
    _mol.RemoveAllConformers()
    conf_id = _mol.AddConformer(conformer, assignId=True)
    return conformer_geometry.check_geometry_consistency(
        _mol, conf_id, contact_scale=contact_scale
    )


def check_displacement(
    conformer: Chem.Conformer,
    new_conformer: Chem.Conformer,
    rmsd_warning_threshold: float = 3.0,
) -> None:
    """Check and log the RMSD displacement between two conformers.

    Calculates the root-mean-square displacement (RMSD) between the original
    and optimized geometries. Logs informational or warning messages if the
    displacement is unusually large, which can indicate unphysical geometry changes.

    Args:
        conformer: Original RDKit conformer (before optimization).
        new_conformer: Optimized RDKit conformer (after optimization).
        rmsd_warning_threshold: Displacement threshold (in Angstroms) above which
            a warning is logged. Values > 3.0 Å may indicate the optimization
            converged to a different tautomer or isomer. Defaults to 3.0.

    Returns:
        None. Results are logged using the logging module.

    Notes:
        Uses RDKit's alignment algorithm to calculate best-fit RMSD, which accounts
        for rigid-body transformations (translation/rotation).

    Examples:
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> mol = Chem.AddHs(mol)
        >>> AllChem.EmbedMolecule(mol)
        >>> conf = mol.GetConformer()
        >>> # ... optimize ...
        >>> new_conf = Chem.Conformer(conf)
        >>> check_displacement(conf, new_conf)  # Logs RMSD
    """
    num_atoms = conformer.GetNumAtoms()
    old_coords = [conformer.GetAtomPosition(idx) for idx in range(num_atoms)]
    new_coords = [new_conformer.GetAtomPosition(idx) for idx in range(num_atoms)]

    try:
        ssr, _ = rdAlignment.GetAlignmentTransform(new_coords, old_coords)
        displacement = np.sqrt(ssr / num_atoms)
        _logger.debug(f"Optimization RMSD displacement: {displacement:.3f} Angstroms")

        if displacement > rmsd_warning_threshold:
            _logger.warning(
                f"Large geometry displacement detected: RMSD = {displacement:.3f} Angstroms."
            )
    except RuntimeError as exc:
        _logger.warning(f"Could not calculate RMSD displacement: {exc}")
