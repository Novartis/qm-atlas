"""
Implements the ReSCoSS conformer workflow described in:

[1] Udvarhelyi, A., Rodde, S. & Wilcken, R. ReSCoSS: a flexible quantum chemistry workflow
identifying relevant solution conformers of drug-like molecules.
J Comput Aided Mol Des 35, 399-415 (2021). https://doi.org/10.1007/s10822-020-00337-7
"""

import heapq
import logging
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from hpc_funcs.files import generate_name
from ppqm.utils import functools as ppqm_functools
from rdkit import Chem

from qm_atlas import constants
from qm_atlas.resources import COSMO_SOLVENTS_DIR
from qm_atlas.tasks import calculate_properties, conformer_generation, optimize
from qm_atlas.tasks.utils import conformer_geometry
from qm_atlas.utils import remove_conformers
from qm_atlas.workflows.utils import filters, properties
from qm_atlas.wrappers.cosmotherm import cosmo_tasks
from qm_atlas.wrappers.turbomole import utils as tm_utils

_logger = logging.getLogger(__name__)


RESCOSS_DEFAULT_CONF_GENS = conformer_generation.ConformerGenerationOptions(
    conf_gens=[
        conformer_generation.MacromodelOptions(  # Macromodel OPLS4/water
            rms_threshold=0.3,
            max_conformers=0,
            mode="MCMM",
            force_field=16,
            solvent_num=1,
        ),
        conformer_generation.MacromodelOptions(  # Macromodel OPLS4/octanol
            rms_threshold=0.3,
            max_conformers=0,
            mode="MCMM",
            force_field=16,
            solvent_num=9,
        ),
        conformer_generation.MacromodelOptions(  # Macromodel OPLS2005/water
            rms_threshold=0.3,
            max_conformers=0,
            mode="MCMM",
            force_field=14,
            solvent_num=1,
        ),
        conformer_generation.OmegaOptions(
            rms_threshold=0.5,
            max_conformers=1000,
            start_from_corina=True,
        ),
        conformer_generation.MoeOptions(
            rms_threshold=0.5,
            max_conformers=1000,
            mode="'LowModeMD'",
        ),
        conformer_generation.RdkitOptions(rms_threshold=0.3, max_conformers=1000, method="KDG"),
    ],
    combination_name="joint_set",
    rms_threshold=0.3,
    rmsd_method="tradeoff",
)


RESCOSS_SOLVENTS = [
    "h2o",
    "dimethylsulfoxide",
    "cyclohexane",
    "1-octanol",
    "methanol",
    "chcl3",
    "propanone",
    "acetonitrile",
    "vacuum",
    "perfluoropyrrole",
]


XTB_OPTIMIZATION_CONFIG: list[optimize.OptimizationConfig] = [
    optimize.XtbOptions(
        method="2",
        opt_level="normal",
        max_num_steps=100,
        solvation_model="alpb",
        solvent="water",
    ),
]

INT_TM_SP_CONFIG = calculate_properties.TurbomoleSinglePointOptions(
    basis="def2-mTZVP", functional=("b97-3c",), control_template=None
)


DEFAULT_FINAL_OPTIMIZATION_CONFIG: list[optimize.OptimizationConfig] = [
    optimize.XtbTurbomoleOptions(
        max_num_steps=100,
        opt_level="normal",
        solvent="conductor",
        basis="def2-TZVP",
        functional=["b-p"],
    ),
]


RMSD_THRESHOLD = 0.3

DEFAULT_NUM_CLUSTERS = 3
DEFAULT_NUM_PER_CLUSTER = 3


def generate_rescoss_conformers(
    mol: Chem.Mol,
    conformer_generation_options: conformer_generation.ConformerGenerationOptions = RESCOSS_DEFAULT_CONF_GENS,
    expand_conformers: bool = True,
    xtb_optimization_config: list[
        optimize.XtbOptions | optimize.XtbTurbomoleOptions | optimize.JobexOptions
    ] = XTB_OPTIMIZATION_CONFIG,
    rmsd_threshold: float = RMSD_THRESHOLD,
    solvents_dirs: list[Path] | None = None,
    use_config_cosmo_dir: bool = True,
    num_clusters: int = DEFAULT_NUM_CLUSTERS,
    num_per_cluster: int = DEFAULT_NUM_PER_CLUSTER,
    final_optimization_config: list[
        optimize.XtbOptions | optimize.XtbTurbomoleOptions | optimize.JobexOptions
    ]
    | None = DEFAULT_FINAL_OPTIMIZATION_CONFIG,
    final_rmsd_threshold: float = RMSD_THRESHOLD / 5,
    scr: Path = constants.DEFAULT_SCR,
    write_intermediates: bool = False,  # debugging option
    trace_dir: Path = constants.DEFAULT_SCR,  # debugging option
    keep_files: bool = False,  # debugging option
    n_cores: int = 1,
) -> Chem.Mol:
    """The ReSCoSS conformer workflow. Generates a list of conformers that aims to
    represent relevant conformations in different solvents. See Reference [1] for
    a detailed description. Here is a brief summary of the steps the workflow takes:

        1) Initial force-field based conformer generation.

        2) xtb-level geometry optimization of all obtained structures.

        3) Deduplication of the conformers after xtb-level geometry optimization
            based on their RMSD.

        4) Single-Point Calculations (B97-3c/mTZVP) and subsequent extraction of
            cosmotherm properties in 10 different solvents.

        5) k-means clustering based on the cosmotherm properties in water.

        6) From each cluster, keep all conformers with the lowest N energies
            in any of the 10 solvents (and discard all others).

        7) DFT-level geometry optimization (default BP86/TZVP, can be changed).

    Args:
        mol (Chem.Mol):
            The molecule to work on.
        conformer_generation_options (ConformerGenerationOptions):
            Specifies which conformer generators to use for the initial conformer
            expansion. Defaults to
            :data:`~qm_atlas.workflows.rescoss_conformers.RESCOSS_DEFAULT_CONF_GENS`, which uses a
            combination of macromodel, omega, moe, and rdkit.
        expand_conformers (bool):
            If False, the workflow skips force-field based conformer expansion and
            uses the conformers provided with the molecule. Defaults to True.
        xtb_optimization_config (OptimizationConfig):
            Configuration for xtb-level geometry optimization in step 2.
            Defaults to :data:`~qm_atlas.workflows.rescoss_conformers.XTB_OPTIMIZATION_CONFIG`.
        rmsd_threshold (float):
            RMSD value below which two conformers are considered duplicates in the
            deduplication carried out after xtb-level geometry optimization.
            Defaults to :data:`~qm_atlas.workflows.rescoss_conformers.RMSD_THRESHOLD`.
        solvents_dirs (list[Path] | None):
            Additional directories in which to look for the solvent .cosmo files used
            in the cosmotherm property calculations. Defaults to None, i.e. relying on
            the cosmotherm database and the configured directories.
        use_config_cosmo_dir (bool):
            Whether to prepend the solvent directories from the software configuration
            file to ``solvents_dirs``. Defaults to True.
        final_optimization_config (OptimizationConfig):
            Configuration for DFT-level geometry optimization in step 7.
            Defaults to :data:`~qm_atlas.workflows.rescoss_conformers.DEFAULT_FINAL_OPTIMIZATION_CONFIG`.
        final_rmsd_threshold (float):
            RMSD value below which two conformers are considered duplicates in the
            deduplication carried out after the final DFT-level optimization.
            Defaults to :data:`~qm_atlas.workflows.rescoss_conformers.RMSD_THRESHOLD` / 5.
        num_clusters (int):
            The number of clusters to use in k-means clustering based on cosmotherm
            properties in water. Defaults to :data:`~qm_atlas.workflows.rescoss_conformers.DEFAULT_NUM_CLUSTERS`.
        num_per_cluster (int):
            The number of conformations to keep per cluster and solvent.
            Defaults to :data:`~qm_atlas.workflows.rescoss_conformers.DEFAULT_NUM_PER_CLUSTER`.
        scr (Path):
            The scratch directory for intermediate calculations.
            Defaults to constants.DEFAULT_SCR.
        write_intermediates (bool):
            If True, writes sdf files with conformers at intermediate workflow steps
            to the trace directory. Debugging option. Defaults to False.
        trace_dir (Path):
            The directory to write intermediate sdf files for debugging purposes.
            Defaults to constants.DEFAULT_SCR.
        keep_files (bool):
            If True, keeps intermediate files from calculations. Debugging option.
            Defaults to False.
        n_cores (int):
            The number of available cores for parallel processing.
            Defaults to 1.

    Returns:
        Chem.Mol:
            The molecule with final, DFT-optimized geometries and representative
            conformers selected through the clustering and filtering workflow.

    References:
        [1] Udvarhelyi, A., Rodde, S. & Wilcken, R. ReSCoSS: a flexible quantum
        chemistry workflow identifying relevant solution conformers of drug-like
        molecules. *J. Comput. Aided Mol. Des.* **35**, 399-415 (2021).
        https://doi.org/10.1007/s10822-020-00337-7
    """

    # store name and properties, to keep them
    mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else "UNNAMED"
    _logger.info(f"Working on Molecule {mol_name}")
    _logger.debug(f"Scratch Directory is {scr}")

    mol_props = properties.read_properties(mol)

    # **Step 1**: generate starting conformations
    if expand_conformers:

        mol_ff = conformer_generation.generate_conformers(
            mol,
            conformer_generation_options=conformer_generation_options,
            num_cores=n_cores,
            write_intermediates=write_intermediates,
            trace_dir=trace_dir,
        )
        num_conformers = mol_ff.GetNumConformers()
        _logger.info(f"Found {num_conformers} in force-field conformer generation")

    else:
        mol_ff = mol
        num_conformers = mol_ff.GetNumConformers()
        _logger.info("Skipping conformer generation Step.")
        _logger.info(f"Starting with {num_conformers} conformers")

    _logger.info("Running xtb geometry optimization.")

    # **Step 2**: optimize them using xtb
    mol_xtb, _ = optimize.optimize_mol(
        mol_ff,
        config=xtb_optimization_config,
        show_progress=False,
        n_cores=n_cores,
        scr=scr,
    )

    if write_intermediates:
        random_name = generate_name()
        sdf_file = trace_dir / f"confs_xtb_{random_name}.sdf"
        _logger.debug(f"Writing xtb-optimized conformer set to {sdf_file.resolve()}")

        conf_ids = [conformer.GetId() for conformer in mol_xtb.GetConformers()]
        with Chem.SDWriter(str(sdf_file.resolve())) as writer:
            for conf_id in conf_ids:
                writer.write(mol_xtb, confId=conf_id)

    # **Step 3**: deduplicate again
    conformer_geometry.deduplicate_conformers(
        mol_xtb, rms_threshold=rmsd_threshold, method="tradeoff", num_cores=n_cores
    )

    num_conformers = mol_xtb.GetNumConformers()
    _logger.info(f"Keeping {num_conformers} conformers after RMSD-based deduplication.")

    # **Step4**: do single point calculations and extract properties

    # run Single-Point calculation at B97-3c level

    single_point_properties = calculate_properties.calculate_property_mol(
        mol_xtb,
        INT_TM_SP_CONFIG,
        n_cores=n_cores,
        scr=scr,
        keep_files=keep_files,
    )
    cosmo_outputs = [
        calculate_properties.extract_cosmo_output(sp_prop, remove_from_dict=True)
        for sp_prop in single_point_properties
    ]

    # remove unconverged calculations
    mol_xtb, cosmo_outputs, unconverged_idcs = tm_utils.extract_converged(mol_xtb, cosmo_outputs)

    num_unconverged = len(unconverged_idcs)
    if num_unconverged > 0:
        _logger.warning("Single-Point calculations failed for xtb-optimized geometries")
        _logger.warning(f"Failure Rate: {num_unconverged} / {num_conformers}")
        _logger.warning("Unconverged Conformers were removed.")
    num_converged = len(cosmo_outputs)

    # extract cosmotherm properties
    get_cosmo_parallel = partial(
        get_cosmo_descriptors,
        cosmo_outputs=cosmo_outputs,
        scr=scr,
        keep_files=keep_files,
        solvents_dirs=solvents_dirs,
        use_config_cosmo_dir=use_config_cosmo_dir,
    )

    n_procs = min([n_cores, len(RESCOSS_SOLVENTS)])
    desc_dfs = ppqm_functools.func_parallel(
        get_cosmo_parallel,
        RESCOSS_SOLVENTS,
        n_cores=n_procs,
        title="Cosmo_Descriptors",
        show_progress=False,
    )

    # **Step 5**: Cluster based on cosmo properties in water
    # properties for clustering
    prop_cols = ["Dipol(t)", "Moment_HBacc", "Moment_HBdon", "Area"]
    desc_df_water = desc_dfs[0]
    prop_arr = desc_df_water.loc[:, prop_cols].to_numpy()

    # now, extract representatives using k-means clustering
    label_to_idcs = filters.cluster_kmeans(prop_arr, n_clusters=num_clusters)

    # **Step 6**: extract representatives from cluster, based on energies in different solvents
    solvent_energies = dict()
    for idx, solvent in enumerate(RESCOSS_SOLVENTS):
        desc_df = desc_dfs[idx]
        solvent_energies[solvent] = desc_df["E_COSMO+dE+Mu"].to_numpy()

    conf_idcs_to_keep = select_from_clusters(
        label_to_idcs, solvent_energies, num_per_cluster=num_per_cluster
    )
    idcs_to_remove = [j for j in range(num_converged) if j not in conf_idcs_to_keep]
    remove_conformers(mol_xtb, idcs_to_remove)

    num_conformers = mol_xtb.GetNumConformers()
    _logger.info(f"Keeping {num_conformers} conformers after clustering")

    # **Step 7**: DFT-level geometry optimization
    if final_optimization_config:
        _logger.info(
            f"Running final geometry optimization with settings {final_optimization_config}"
        )
        mol_dft_opt, _ = optimize.optimize_mol(
            mol_xtb,
            config=final_optimization_config,
            n_cores=n_cores,
            show_progress=False,
            scr=scr,
        )
    else:
        mol_dft_opt = mol_xtb

    # remove duplicates again after dft-level geometry optimization
    conformer_geometry.deduplicate_conformers(
        mol_dft_opt, rms_threshold=final_rmsd_threshold, num_cores=n_cores
    )
    num_conformers = mol_dft_opt.GetNumConformers()
    _logger.info(f"Keeping {num_conformers} conformers after final geometry optimization")

    # restore properties of the molecule
    properties.restore_properties(mol_dft_opt, mol_props)

    return mol_dft_opt


def get_cosmo_descriptors(
    solvent_name: str,
    cosmo_outputs: list[str],
    solvents_dirs: list[Path] | None = None,
    use_config_cosmo_dir: bool = True,
    scr: Path = constants.DEFAULT_SCR,
    keep_files: bool = False,
) -> pd.DataFrame:
    """Helper function to calculate conformer cosmo descriptors for a given solvent.

    Args:
        solvent_name (str):
            The name of the solvent
        cosmo_outputs (list[str], optional):
            The cosmo outputs for the conformers.
        solvents_dirs (list[Path] | None, optional):
            The directory (or list of directories) in which the solvent cosmo file is located.
            Defaults to the COSMO solvents directories (TZVP) from config.
        use_config_cosmo_dir (bool, optional):
            If True, also include the COSMO solvents directory configured via the
            software config. Defaults to True.
        scr (Path, optional):
            The scratch directory to use for the calculation.
            Defaults to constants.DEFAULT_SCR.
        keep_files:
            whether to keep the generated files. Defaults to False.

    Returns:
        desc_df (pd.DataFrame):
            A dataframe containing the descriptors for the solvent.
    """

    # Fall back on the bundled .cosmo files for solvents (e.g. perfluoropyrrole)
    # that are not part of a standard cosmotherm installation.
    if solvents_dirs is None:
        solvents_dirs = [COSMO_SOLVENTS_DIR]
    else:
        solvents_dirs = list(solvents_dirs) + [COSMO_SOLVENTS_DIR]

    return cosmo_tasks.calculate_descriptors_df(
        cosmo_outputs,
        solvent_name=solvent_name,
        level="bp-tzvp",
        solvents_dirs=solvents_dirs,
        use_config_cosmo_dir=use_config_cosmo_dir,
        scr=scr,
        keep_files=keep_files,
    )


def select_from_clusters(
    label_to_idcs: list[np.ndarray],
    solvent_energies: dict[str, np.ndarray],
    num_per_cluster: int = 3,
) -> set[int]:
    """For all supplied clusters, the code selects all indices corresponding
        to the N conformers with lowest energy in each of the solvents
        for which the energy is supplied.

    Args:
        label_to_idcs (List[np.array]):
            A list of numpy arrays. label_to_idcs[cluster_idx] contains the indices of
            all conformers belonging to the cluster with index cluster_idx.
        solvent_energies (Dict[str, np.array]):
            A dictionary mapping solvent names to the energies of all conformers in
            these solvents. For each solvent, solvent_energies[solvent_name][idx] is
            the energy of the conformer with index idx in the specified solvent.
        num_per_cluster (int):
            The number of molecules to keep per cluster and solvent. Defaults to 3.

    Returns:
        selected_idcs(Set[int]):
            A set of all conformer indices selected. Note that we could in principle
            select, for each cluster, N * (number of solvents) indices. However, the
            indices typically overlap and we select much fewer.
    """
    selected_idcs = set()

    for conf_idcs in label_to_idcs:
        for solvent_eng in solvent_energies.values():
            n_smallest = heapq.nsmallest(
                num_per_cluster, conf_idcs, key=lambda x, se=solvent_eng: se[x]
            )
            selected_idcs.update(n_smallest)

    return selected_idcs
