import json
from collections.abc import Sequence

from rdkit import Chem

from qm_atlas.tasks import optimize
from qm_atlas.tasks.utils import conformer_geometry
from qm_atlas.tasks.utils.parallel import run_parallel
from qm_atlas.wrappers import xtb
from qm_atlas.wrappers.turbomole import xtb_opt

#: The second of the two force constants the default configuration runs, so that a caller can
#: see whether the answer depends on how tightly the pose was held. Both settings hold the
#: restrained torsions: 0.5 to a few tenths of a degree, 0.1 to about two degrees. There is no
#: longer any reason to go softer than this - the soft end of the old 0.5 / 0.1 / 0.01 ladder
#: existed only to let the input geometry's bond lengths and bond angles relax out of the
#: restraint, and those are now unrestrained at every force constant.
DEFAULT_SECOND_FORCE_CONSTANT = 0.1

OPTIMIZATION_RMSD_PROP = "Optimization_RMSD"


DEFAULT_OPTIMIZATION_CONFIG = [
    [
        optimize.XtbTurbomoleOptions(
            max_num_steps=xtb_opt.DEFAULT_XTB_TM_STEPS,
            opt_level="normal",
            basis="def2-TZVP",
            functional=["b-p"],
            constrain=True,
            force_constant=xtb.DEFAULT_FORCE_CONSTANT,
            torsions_fixed=False,
            heavy_atom_torsions_fixed=True,
            angles_fixed=False,
            bonds_fixed=False,
        ),
    ],
    [
        optimize.XtbTurbomoleOptions(
            max_num_steps=xtb_opt.DEFAULT_XTB_TM_STEPS,
            opt_level="normal",
            basis="def2-TZVP",
            functional=["b-p"],
            constrain=True,
            force_constant=DEFAULT_SECOND_FORCE_CONSTANT,
            torsions_fixed=False,
            heavy_atom_torsions_fixed=True,
            angles_fixed=False,
            bonds_fixed=False,
        ),
    ],
]


def optimize_fixed_mol(
    config_list: Sequence[
        optimize.XtbTurbomoleOptions | optimize.XtbOptions | optimize.JobexOptions
    ],
    n_cores: int = 1,
    mol: Chem.Mol | None = None,
    treat_unconverged_as_failure: bool = False,
) -> Chem.Mol:
    """Run constrained optimization with a list of configuration options.

    Each configuration in the list is tried sequentially with fallback behavior.

    Args:
        config_list: Sequence of optimization configurations to try in order.
        n_cores: Number of cores to allocate for this optimization.
        mol: Molecule to optimize.
        treat_unconverged_as_failure: Whether unconverged results should trigger fallback.

    Returns:
        Optimized molecule.
    """
    if mol is None:
        raise ValueError("Input molecule must be provided for optimization.")

    mol_opt, _ = optimize.optimize_mol(
        mol,
        config=config_list,
        n_cores=n_cores,
        treat_unconverged_as_failure=treat_unconverged_as_failure,
    )

    # Store the settings for the first config in the list
    mol_opt.SetProp(
        "Settings_Constrained_Optimization", json.dumps([cfg.model_dump() for cfg in config_list])
    )

    # Compute the RMSD between the input geometry and the optimized geometry and
    # attach it as a property.
    if mol_opt.GetNumConformers() == 1:
        _, rmsd_dict = conformer_geometry.align_mol_to_reference(mol_opt, mol)
        optimization_rmsd = next(iter(rmsd_dict.values()))
        mol_opt.SetDoubleProp(OPTIMIZATION_RMSD_PROP, optimization_rmsd)

    return mol_opt


def run_constrained_optimization(
    mol: Chem.Mol,
    n_cores: int = 1,
    optimization_config: Sequence[
        Sequence[optimize.XtbTurbomoleOptions | optimize.XtbOptions | optimize.JobexOptions]
    ]
    | None = None,
    treat_unconverged_as_failure: bool = False,
) -> list[Chem.Mol]:
    """Run constrained optimization for a molecule with multiple configuration options.

    Args:
        mol (Chem.Mol):
            Molecule to optimize. Must have exactly one conformation.
        n_cores (int, optional):
            Number of cores to allocate for this optimization. Defaults to 1.
        optimization_config (Sequence[Sequence[optimize.XtbTurbomoleOptions  |  optimize.XtbOptions  |  optimize.JobexOptions]], optional):
            List of optimization configurations to run. Each one is run separately and different molecules are returned for each.
            Each entry can be a sublist of configurations to try sequentially with fallback behavior.
            Defaults to :data:`~qm_atlas.workflows.optimize_constrained.DEFAULT_OPTIMIZATION_CONFIG`
            (after resolving the default value of None),
            which corresponds to two runs with different force constants for the constraints
            (0.5 and 0.1).
        treat_unconverged_as_failure (bool, optional): Whether unconverged results should trigger fallback. Defaults to False.

    Raises:
        ValueError: If the input molecule does not have exactly one conformation.

    Returns:
        list[Chem.Mol]: List of optimized molecules.
    """

    if optimization_config is None:
        optimization_config = DEFAULT_OPTIMIZATION_CONFIG

    if mol.GetNumConformers() != 1:
        raise ValueError(
            "Input molecule must have exactly one conformation for constrained optimization."
        )

    # Convert each group of configurations into a combined config list
    args_for_run_parallel = [(config_sublist,) for config_sublist in optimization_config]
    optimized_mols = run_parallel(
        optimize_fixed_mol,
        args_for_run_parallel,
        mol=mol,
        n_cores=n_cores,
        treat_unconverged_as_failure=treat_unconverged_as_failure,
        show_progress=False,
        func_has_ncores_arg=True,
        rdkit_pickle_properties=Chem.PropertyPickleOptions.AllProps,
    )

    return optimized_mols
