import logging
from pathlib import Path

import numpy as np
from ppqm import chembridge
from rdkit import Chem

from qm_atlas import constants
from qm_atlas.tasks import conformer_generation, optimize
from qm_atlas.utils import remove_conformers
from qm_atlas.workflows.utils import filters, properties
from qm_atlas.wrappers import xtb

_logger = logging.getLogger(__name__)


FAST_DEFAULT_OPTIONS = conformer_generation.ConformerGenerationOptions(
    conf_gens=[
        conformer_generation.OmegaOptions(
            rms_threshold=0.3,
            max_conformers=1000,
            start_from_corina=True,
        ),
        conformer_generation.RdkitOptions(rms_threshold=0.3, max_conformers=1000, method="KDG"),
    ],
    combination_name="fallback",
)


DEFAULT_OPTIMIZATION_CONFIG: list[optimize.OptimizationConfig] = [
    optimize.XtbOptions(
        method="2",
        opt_level="normal",
        max_num_steps=100,
        solvation_model="alpb",
        solvent="water",
    ),
    optimize.XtbOptions(
        method="0",
        opt_level="normal",
        max_num_steps=200,
        solvation_model="alpb",
        solvent="water",
    ),
    optimize.XtbOptions(
        method="2",
        opt_level="lax",
        max_num_steps=500,
        solvation_model="alpb",
        solvent="water",
    ),
]

DEFAULT_FINAL_OPTIMIZATION_CONFIG: list[optimize.OptimizationConfig] = []


def reindex_conformers(molobj: Chem.Mol) -> Chem.Mol:
    """Renumber the conformer ids of *molobj* to ``0..N-1``, keeping their order.

    Both :mod:`ppqm` helpers used in this module address conformers by *position*
    where rdkit expects an *id*: ``chembridge.get_sasa`` iterates
    ``range(GetNumConformers())`` and passes each counter as ``confIdx``, and
    ``chembridge.molobj_select_conformers`` is called here with the positional
    indices that :mod:`~qm_atlas.workflows.utils.filters` produces but looks them up
    with ``GetConformer(id=...)``. Both are correct only while ids and positions
    coincide.

    They stop coinciding after :func:`~qm_atlas.tasks.optimize.optimize_mol`, which
    drops conformers whose optimization failed and keeps the ids of the survivors -
    four conformers with one failure leave ids ``0, 2, 3``, and the next SASA call
    raises ``ValueError: Bad Conformer Id``. Calling this after every optimization
    restores the invariant those helpers rely on.

    Args:
        molobj (Chem.Mol): Molecule whose conformers should be renumbered.

    Returns:
        Chem.Mol: ``molobj`` itself when the ids are already contiguous, otherwise a
            copy carrying the same conformers in the same order with ids ``0..N-1``.
    """
    conf_ids = [conformer.GetId() for conformer in molobj.GetConformers()]
    if conf_ids == list(range(len(conf_ids))):
        return molobj

    _logger.debug(f"Renumbering non-contiguous conformer ids {conf_ids} to 0..{len(conf_ids) - 1}")
    molobj_reindexed = Chem.Mol(molobj)
    remove_conformers(molobj_reindexed, [], reset_idcs=True)
    return molobj_reindexed


def get_conformer_shapes(molobj: Chem.Mol) -> np.ndarray:
    """
    Get a list of descriptors for each conformer, namely;

    - SASA
    - Dipole moment

    Note: recoss clusters on
    - dipole moment (xtb)
    - hydrogen-acceptor moment (COSMO)
    - hydrogen-donor moment (COSMO)
    - Solvent accessible surface Area (COSMO)

    """

    # Get surface accessible solvent area
    sasas = chembridge.get_sasa(molobj, extra_radius=1.0)

    # Dipole moments
    dipole_moments = chembridge.get_dipole_moments(molobj)

    # Package in one numpy array
    values = [sasas, dipole_moments]
    values = np.asarray(values, dtype=np.float64)

    return values


def filter_molobj_shapes(molobj: Chem.Mol, pick_n: int = 1) -> Chem.Mol:
    """
    Filter conformers by shape descriptors, keeping one representative per cluster.

    Computes SASA and dipole moment for each conformer, clusters them using the
    combined IsolationForest -> DBSCAN -> KMeans search, and picks one conformer
    per cluster.

    Args:
        molobj (Chem.Mol):
            Input molecule with at least one conformer.
        pick_n (int):
            How many conformers per cluster. Defaults to 1.

    Returns:
        Chem.Mol:
            Molecule containing N conformer per shape cluster.
    """

    values = get_conformer_shapes(molobj)

    # Select subset
    groups = filters.combination(values)
    selection = filters.pick_n_from_groups(groups, pick_n)
    molobj_select: Chem.Mol = chembridge.molobj_select_conformers(molobj, selection)  # type: ignore

    return molobj_select


def fast_conformer_filtering(
    molobj: Chem.Mol,
    optimization_config: list[
        optimize.XtbOptions | optimize.JobexOptions | optimize.XtbTurbomoleOptions
    ] = DEFAULT_OPTIMIZATION_CONFIG,
    show_progress: bool = False,
    n_cores: int = 1,
    scr: Path = constants.DEFAULT_SCR,
) -> Chem.Mol:
    """
    Given pre-generated conformers, filter them using shape descriptors and xTB optimization.

    Applies two rounds of shape-based filtering (SASA + dipole moment) around a
    geometry optimization step:

    1. **First filter**: clusters conformers by shape descriptors and picks one
       representative per cluster to reduce the set before optimization.
    2. **xTB optimization**: relaxes the filtered set using ``optimization_config``
       and collects per-conformer energies.
    3. **Second filter**: re-clusters the optimized conformers by shape. Within each
       cluster the lowest-energy conformer is kept. If fewer than 10 conformers
       remain and shape variance is very low (SASA std < 5.0 Å**2, dipole std < 1.0 D),
       all are treated as one cluster.

    Args:
        molobj (Chem.Mol):
            Input molecule with at least 2 conformers.
        optimization_config (list[XtbOptions | JobexOptions | XtbTurbomoleOptions], optional):
            Sequence of xTB optimization steps. Defaults to
            :data:`~qm_atlas.workflows.fast_conformers.DEFAULT_OPTIMIZATION_CONFIG`
            (GFN2-xTB normal → GFN0-xTB normal → GFN2-xTB lax, all with ALPB water).
        show_progress (bool, optional):
            Show a progress bar during optimization. Defaults to ``False``.
        n_cores (int, optional):
            Number of CPU cores to use. Defaults to 1.
        scr (Path, optional):
            Scratch directory for xTB calculations. Defaults to ``constants.DEFAULT_SCR``.

    Returns:
        Chem.Mol:
            Molecule with a reduced, shape-diverse set of xTB-optimized conformers.
            Each conformer is the lowest-energy representative of its shape cluster.
    """

    # First filter
    molobj_select = filter_molobj_shapes(molobj)

    # Optimize structures
    molobj_xtb, properties = optimize.optimize_mol(
        molobj_select,
        config=optimization_config,
        show_progress=show_progress,
        n_cores=n_cores,
        scr=scr,
    )

    # A failed optimization leaves a gap in the conformer ids, which the ppqm helpers
    # below cannot address. `properties` already covers only the surviving conformers,
    # in order, so it stays aligned with the renumbered ids.
    molobj_xtb = reindex_conformers(molobj_xtb)

    # Second filter
    energies = np.array(
        [property_dict[xtb.XTB_ENERGY_KEY].get_property_value() for property_dict in properties]
    )

    n_filter2 = molobj_xtb.GetNumConformers()

    values = get_conformer_shapes(molobj_xtb)

    # rows: sasa and dipole moment
    stds = values.std(axis=1)

    # If only 10 conformers are left and the shapes are very similar,
    # assume this is the same group and select one conformer
    # TODO Collect all that falls into this filter
    if stds[0] < 5.0 and stds[1] < 1.0 and n_filter2 < 10:
        groups_again = [list(range(n_filter2))]
        _logger.warning("molecule has low shape diversity")

    else:
        groups_again = filters.combination(values, eps=0.3)

    # Select again, but with energies
    selection = []

    for group in groups_again:
        group_energies = energies[group]
        idx_min_energy = np.argmin(group_energies)
        idx = group[idx_min_energy]
        selection.append(idx)

    # Embed conformers into molobj and return
    molobj_unique: Chem.Mol = chembridge.molobj_select_conformers(molobj_xtb, selection)  # type: ignore

    return molobj_unique


def generate_fast_conformers(
    molobj: Chem.Mol,
    conformer_generation_options: conformer_generation.ConformerGenerationOptions = FAST_DEFAULT_OPTIONS,
    expand_conformers: bool = True,
    optimization_config: list[
        optimize.XtbOptions | optimize.JobexOptions | optimize.XtbTurbomoleOptions
    ] = DEFAULT_OPTIMIZATION_CONFIG,
    final_optimization_config: list[
        optimize.XtbOptions | optimize.JobexOptions | optimize.XtbTurbomoleOptions
    ] = DEFAULT_FINAL_OPTIMIZATION_CONFIG,
    scr: Path = constants.DEFAULT_SCR,
    show_progress: bool = False,
    n_cores: int = 1,
    trace_dir: Path = constants.DEFAULT_SCR,
    write_intermediates: bool = False,
) -> Chem.Mol:
    """
    Generate a representative set of conformers for a molecule using a multi-stage
    force-field + xTB pipeline.

    Workflow:

    1. **Conformer generation**: Runs force-field based conformer generators
    2. **Shape filtering + xTB optimization**: run :func:`~qm_atlas.workflows.fast_conformers.fast_conformer_filtering`
    3. **Optional final optimization**: the unique conformers are re-optimized.

    Args:
        molobj (Chem.Mol):
            Input RDKit molecule (2D or 3D; conformers are always generated from scratch).
        conformer_generation_options (conformer_generation.ConformerGenerationOptions, optional):
            Force-field conformer generator settings. Defaults to
            :data:`~qm_atlas.workflows.fast_conformers.FAST_DEFAULT_OPTIONS`
            (OpenEye Omega, rms_threshold=0.3, max_conformers=1000, with RDKit KDG fallback).
        expand_conformers (bool, optional):
            If False, skips force-field based conformer generation and uses the
            conformers already present on ``molobj``. Defaults to ``True``.
        optimization_config (list[XtbOptions | JobexOptions | XtbTurbomoleOptions], optional):
            Sequence of xTB optimization steps applied after the first shape filter.
            Defaults to :data:`~qm_atlas.workflows.fast_conformers.DEFAULT_OPTIMIZATION_CONFIG`
            (GFN2-xTB normal → GFN0-xTB normal → GFN2-xTB lax, all with ALPB water solvation).
        final_optimization_config (list[XtbOptions | JobexOptions | XtbTurbomoleOptions], optional):
            Optional sequence of optimization steps applied to the final unique
            conformers. Pass an empty list
            (default, :data:`~qm_atlas.workflows.fast_conformers.DEFAULT_FINAL_OPTIMIZATION_CONFIG`)
            to skip this step.
        scr (Path, optional):
            Scratch directory for xTB calculations. Defaults to ``constants.DEFAULT_SCR``.
        show_progress (bool, optional):
            Show a progress bar during optimization. Defaults to ``False``.
        n_cores (int, optional):
            Number of CPU cores to use. Defaults to 1.
        trace_dir (Path, optional):
            Directory for intermediate SDF files written when
            ``write_intermediates=True``. Defaults to ``constants.DEFAULT_SCR``.
        write_intermediates (bool, optional):
            Write SDF files at each workflow step for debugging. Defaults to ``False``.

    Returns:
        Chem.Mol:
            Molecule with a reduced, shape-diverse set of xTB-optimized conformers.
            Each conformer is the lowest-energy representative of its shape cluster
            (defined by SASA and dipole moment). Returns the input ``molobj`` if
            conformer generation fails or produces fewer than 2 conformers.
    """
    mol_props = properties.read_properties(molobj)

    if expand_conformers:
        molobj_3d = conformer_generation.generate_conformers(
            molobj,
            conformer_generation_options=conformer_generation_options,
            num_cores=n_cores,
            trace_dir=trace_dir,
            write_intermediates=write_intermediates,
        )
    else:
        molobj_3d = molobj
        _logger.info("Skipping conformer generation step.")
        _logger.info(f"Starting with {molobj_3d.GetNumConformers()} conformers")

    n_conf = molobj_3d.GetNumConformers()

    if n_conf == 0:
        _logger.error("Unable to generate conformers")
        properties.restore_properties(molobj_3d, mol_props)
        return molobj_3d

    if n_conf == 1:
        _logger.warning("Only 1 conformer generated")
        properties.restore_properties(molobj_3d, mol_props)
        return molobj_3d

    molobj_unique = fast_conformer_filtering(
        molobj_3d,
        optimization_config=optimization_config,
        show_progress=show_progress,
        n_cores=n_cores,
        scr=scr,
    )

    if final_optimization_config:
        _logger.info(
            f"Running final geometry optimization with settings {final_optimization_config}"
        )
        molobj_unique, _ = optimize.optimize_mol(
            molobj_unique,
            show_progress=show_progress,
            n_cores=n_cores,
            scr=scr,
            config=final_optimization_config,
        )
        molobj_unique = reindex_conformers(molobj_unique)

    properties.restore_properties(molobj_unique, mol_props)

    return molobj_unique
