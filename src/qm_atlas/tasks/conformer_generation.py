"""Module collecting methods for force-field based conformer generation
"""
import logging
import traceback
from copy import copy
from pathlib import Path
from typing import Annotated, Literal

from hpc_funcs.files import generate_name
from pydantic import BaseModel, ConfigDict, Field, field_validator
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from qm_atlas import constants
from qm_atlas.command_line.utils.format_utils import write_sdf
from qm_atlas.tasks.utils import conformer_geometry
from qm_atlas.wrappers import corina, macromodel, moe, openeye

_logger = logging.getLogger(__name__)


DEFAULT_RMSD_THRESHOLD: float = 0.3


def generate_corina_conformers(mol: Chem.Mol, **kwargs):
    """Generate a 3D conformer for *mol* using CORINA (explicit Hs are added first)."""
    return corina.generate_conformers_molobj(Chem.AddHs(mol), **kwargs)


def generate_omega_conformers(mol: Chem.Mol, **kwargs):
    """Generate conformers for *mol* using OpenEye OMEGA."""
    return openeye.generate_conformers_molobj(mol, **kwargs)


def generate_moe_conformers(mol: Chem.Mol, **kwargs):
    """Generate conformers for *mol* using MOE."""
    return moe.generate_conformers(mol, **kwargs)


def generate_macromodel_conformers(mol: Chem.Mol, **kwargs):
    """Generate conformers for *mol* using MacroModel."""
    return macromodel.generate_conformers(mol, **kwargs)


def generate_macrocycle_conformers(mol: Chem.Mol, **kwargs):
    """Generate macrocycle conformers for *mol* using MacroModel."""
    return macromodel.generate_macrocycle_conformers(mol, **kwargs)


RDKIT_METHODS = {
    "ETDG": rdDistGeom.ETDG,
    "ETDGv2": rdDistGeom.ETDGv2,
    "ETKDG": rdDistGeom.ETKDG,
    "ETKDGv2": rdDistGeom.ETKDGv2,
    "ETKDGv3": rdDistGeom.ETKDGv3,
    "KDG": rdDistGeom.KDG,
    "srETKDGv3": rdDistGeom.srETKDGv3,
}


def generate_rdkit_conformers(
    mol: Chem.Mol,
    method: str = "KDG",
    max_conformers: int = 1000,
    rms_threshold: float = DEFAULT_RMSD_THRESHOLD,
    random_seed: int = 1,
    scr: Path | None = None,
) -> Chem.Mol | None:
    """Calls rdkit to generate conformers and deduplicates them according to
        the RMSD between them.

    Args:
        mol (Chem.Mol):
            The input graph.
        method (str, optional):
            The RDKit embedding method (a key of RDKIT_METHODS, e.g. "ETKDGv3").
            Defaults to "KDG".
        max_conformers (int, optional):
            The maximal number of conformers to generate. Defaults to 1000.
        rms_threshold (float, optional):
            The threshold value for the RMSD to use for the deduplication.
            Defaults to DEFAULT_RMSD_THRESHOLD.
        random_seed (int, optional):
            A random seed to get reproducible results from the rdkit conformer
            generator. Defaults to 1.
        scr (Path | None, optional):
            Unused; accepted for API compatibility with the other generators.
            Defaults to None.

    Returns:
        Optional[Chem.Mol]:
            The molecule with conformers. Returns None if the code fails.
    """
    del scr  # un-used but API-required

    if method not in RDKIT_METHODS:
        _logger.error(f"rdkit conformer generation method {method} not recognized")
        raise ValueError(f"rdkit conformer generation method {method} not recognized")

    rdk_conf_gen = RDKIT_METHODS[method]()
    rdk_conf_gen.verbose = False
    rdk_conf_gen.randomSeed = random_seed
    rdk_conf_gen.pruneRmsThresh = rms_threshold

    # fresh start
    mol_3d = Chem.AddHs(mol)
    mol_3d.RemoveAllConformers()

    try:
        rdDistGeom.EmbedMultipleConfs(mol_3d, numConfs=max_conformers, params=rdk_conf_gen)

    except Exception as exc:  # pylint: disable=broad-except
        _logger.error(f"rdkit conformer generation failed with Exception {exc}")
        _logger.error(f"Exception traceback: {traceback.format_exc()}")
        return None

    return mol_3d


class ConformerGeneratorOptions(BaseModel):
    """Base class for conformer generator backends.

    Each concrete backend declares a unique ``backend`` literal that acts as the
    discriminator for (de)serialization within :data:`ConformerGeneratorConfig`.
    """


class MacromodelOptions(ConformerGeneratorOptions):
    """Options for Macromodel conformer generation."""

    # Every field only fills the MacroModel input template, so reject unknown keys as typos.
    model_config = ConfigDict(extra="forbid")

    backend: Literal["macromodel"] = Field(
        default="macromodel",
        description="Conformer generator backend to use",
    )
    rms_threshold: float = Field(
        default=0.3,
        description="RMSD threshold for duplicate detection",
    )
    max_conformers: int = Field(
        default=1000,
        description="Maximum number of conformers to generate (0=unlimited)",
    )
    mode: str = Field(
        default="MCMM",
        description="Conformer search type: MCMM or LMCS",
    )
    force_field: int = Field(
        default=14,
        description="Force field: 10=MMFF94s, 14=OPLS2005, 16=OPLS4",
    )
    solvent_num: int = Field(
        default=1,
        description="Solvent number: 1=water, 9=octanol",
    )
    license_buffer: int = Field(
        default=1,
        description="Number of licenses to leave available",
    )
    license_timeout: int = Field(
        default=60,
        description="Timeout in minutes for waiting on license",
    )


class OmegaOptions(ConformerGeneratorOptions):
    """Options for OpenEye Omega conformer generation."""

    # The Omega wrapper takes only explicit parameters, so reject unknown keys as typos.
    model_config = ConfigDict(extra="forbid")

    backend: Literal["omega"] = Field(
        default="omega",
        description="Conformer generator backend to use",
    )
    rms_threshold: float = Field(
        default=0.3,
        description="RMSD threshold for duplicate detection",
    )
    max_conformers: int = Field(
        default=1000,
        description="Maximum number of conformers to generate",
    )
    start_from_corina: bool = Field(
        default=False,
        description="Use CORINA to generate 3D input conformation",
    )
    n_cores: int = Field(
        default=1,
        description="Number of cores to use",
    )


class MoeOptions(ConformerGeneratorOptions):
    """Options for MOE conformer generation."""

    # Every field only fills the MOE input template, so reject unknown keys as typos.
    model_config = ConfigDict(extra="forbid")

    backend: Literal["moe"] = Field(
        default="moe",
        description="Conformer generator backend to use",
    )
    rms_threshold: float = Field(
        default=0.3,
        description="RMSD threshold for duplicate detection",
    )
    max_conformers: int = Field(
        default=1000,
        description="Maximum number of conformers to generate",
    )
    mode: str = Field(
        default="'LowModeMD'",
        description="Conformer search mode (wrapped in quotes for MOE)",
    )


class CorinaOptions(ConformerGeneratorOptions):
    """Options for CORINA conformer generation."""

    # The CORINA wrapper takes only explicit parameters, so reject unknown keys as typos.
    model_config = ConfigDict(extra="forbid")

    backend: Literal["corina"] = Field(
        default="corina",
        description="Conformer generator backend to use",
    )


class RdkitOptions(ConformerGeneratorOptions):
    """Options for RDKit conformer generation."""

    # The RDKit generator takes only explicit parameters, so reject unknown keys as typos.
    model_config = ConfigDict(extra="forbid")

    backend: Literal["rdkit"] = Field(
        default="rdkit",
        description="Conformer generator backend to use",
    )
    method: str = Field(
        default="KDG",
        description="Embedding method: ETDG, ETKDG, ETKDGv2, ETKDGv3, KDG, srETKDGv3",
    )
    max_conformers: int = Field(
        default=1000,
        description="Maximum number of conformers to generate",
    )
    rms_threshold: float = Field(
        default=0.3,
        description="RMSD threshold for deduplication",
    )
    random_seed: int = Field(
        default=1,
        description="Random seed for reproducibility",
    )

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        if v not in RDKIT_METHODS:
            allowed = ", ".join(RDKIT_METHODS.keys())
            raise ValueError(f"method must be one of: {allowed}")
        return v


class MacrocycleOptions(ConformerGeneratorOptions):
    """Options for Macrocycle conformer generation."""

    # The macrocycle wrapper forwards extra kwargs to the script as CLI arguments.
    model_config = ConfigDict(extra="allow")

    backend: Literal["macrocycle"] = Field(
        default="macrocycle",
        description="Conformer generator backend to use",
    )
    max_conformers: int = Field(
        default=1000,
        description="Maximum number of conformers to generate",
    )
    energy_cutoff: float = Field(
        default=30.0,
        description="Energy cutoff in kcal/mol",
    )
    rms_threshold: float = Field(
        default=0.3,
        description="RMS threshold in Angstroms",
    )
    num_cores: int = Field(
        default=1,
        description="Number of cores to use",
    )
    license_buffer: int = Field(
        default=1,
        description="Number of licenses to leave available",
    )
    license_timeout: int = Field(
        default=60,
        description="Timeout in minutes for waiting on license",
    )


GENERATOR_FUNCTIONS_AND_OPTIONS = {
    "corina": (generate_corina_conformers, CorinaOptions),
    "macrocycle": (generate_macrocycle_conformers, MacrocycleOptions),
    "macromodel": (generate_macromodel_conformers, MacromodelOptions),
    "moe": (generate_moe_conformers, MoeOptions),
    "omega": (generate_omega_conformers, OmegaOptions),
    "rdkit": (generate_rdkit_conformers, RdkitOptions),
}


def get_generator_function(name: str):
    """Get the conformer generator function for a given generator name."""
    if name not in GENERATOR_FUNCTIONS_AND_OPTIONS:
        available = ", ".join(GENERATOR_FUNCTIONS_AND_OPTIONS.keys())
        raise ValueError(
            f"Unknown conformer generator '{name}'. " f"Available generators: {available}"
        )
    return GENERATOR_FUNCTIONS_AND_OPTIONS[name][0]


# Discriminated union of all conformer generator config types. The ``backend``
# literal field acts as the discriminator so Pydantic selects the concrete class
# automatically when deserializing from a dict, and serializes all fields during
# model_dump().
ConformerGeneratorConfig = Annotated[
    MacromodelOptions
    | OmegaOptions
    | MoeOptions
    | CorinaOptions
    | RdkitOptions
    | MacrocycleOptions,
    Field(discriminator="backend"),
]


def deserialize_conformer_generator_config(v: object) -> object:
    """Convert a dict into the appropriate conformer generator options class."""
    if isinstance(v, dict):
        backend = v.get("backend")
        if backend not in GENERATOR_FUNCTIONS_AND_OPTIONS:
            available = ", ".join(GENERATOR_FUNCTIONS_AND_OPTIONS.keys())
            raise ValueError(
                f"Unknown conformer generator '{backend}'. Available generators: {available}"
            )
        options_cls = GENERATOR_FUNCTIONS_AND_OPTIONS[backend][1]
        return options_cls(**v)
    return v


def join_conformers(
    mol_1: Chem.Mol | None,
    mol_2: Chem.Mol | None,
) -> Chem.Mol | None:
    """Merges the conformers of two molecule objects representing the same graph.
        For convenience either of the input molecules can be None, in this case the
        other is returned.

    Args:
        mol_1 (Optional[Chem.Mol]):
            An rdkit molecule or None.
        mol_2 (Optional[Chem.Mol]):
            An rdkit molecule or None.

    Returns:
        ret_mol (Optional[Chem.Mol]):
            A molecule representing the same graph, with all conformers set on either
            of the input molecules.
    """

    if mol_1 is None:
        if mol_2 is None:
            return None
        return mol_2

    if mol_2 is None:
        return mol_1

    ret_mol = Chem.Mol(mol_1)
    for conformer in mol_2.GetConformers():
        ret_mol.AddConformer(conformer, assignId=True)

    return ret_mol


def generate_conformers_with_fallback(
    mol: Chem.Mol,
    conf_gens: list[ConformerGeneratorConfig],
) -> Chem.Mol | None:
    """Allows to specify fall-backs for failed conformer generations.
        The conformer generators are called in order until one succeeds.

    Args:
        mol (Chem.Mol):
            The rdkit molecule.
        conf_gens (list[ConformerGeneratorConfig]):
            The conformer generator option objects to use, in order. Each is a
            typed options object whose ``backend`` field selects the generator.

    Returns:
        mol_3d (Optional[Chem.Mol]):
            The molecule with all conformers. Returns None if all generators fail.
    """

    for conf_gen_entry in copy(conf_gens):
        conf_gen = get_generator_function(conf_gen_entry.backend)
        options = conf_gen_entry.model_dump(exclude={"backend"})
        try:
            mol_3d = conf_gen(mol, **options)
            if mol_3d is not None and mol_3d.GetNumConformers() > 0:
                _logger.info(f"Used {conf_gen_entry.backend} to generate conformers")
                return mol_3d
        except Exception as exc:  # pylint: disable=broad-except
            _logger.warning(
                f"Conformer generation with {conf_gen_entry.backend} failed with Exception {exc}"
            )
            _logger.warning("Relying on Falback method.")

    _logger.error("All conformer generators failed")
    raise RuntimeError("All conformer generators failed")


def generate_joint_conformer_set(
    mol: Chem.Mol,
    conf_gens: list[ConformerGeneratorConfig],
    rms_threshold: float = DEFAULT_RMSD_THRESHOLD,
    rmsd_method: str = "tradeoff",
    num_cores: int = 1,
    trace_dir: Path = constants.DEFAULT_SCR,
    write_intermediates: bool = False,
) -> Chem.Mol | None:
    """Calls different conformer generators, merges their results and deduplicates
        according to RMSD between conformers.

    Args:
        mol (Chem.Mol):
            The rdkit molecule.
        conf_gens (list[ConformerGeneratorConfig]):
            The conformer generator option objects to use. Each is a typed options
            object whose ``backend`` field selects the generator.
        rms_threshold (float, optional):
            The rmsd threshold to use for the deduplication. Defaults to
            DEFAULT_RMSD_THRESHOLD. If you pass None, no deduplication will be performed.
        rmsd_method (str, optional):
            The RMSD algorithm used during deduplication. Defaults to "tradeoff".
        num_cores (int, optional):
            Number of CPU cores to use for the deduplication. Defaults to 1.
        trace_dir (Path, optional):
            Directory to write per-generator intermediate SDFs to when
            *write_intermediates* is True. Defaults to constants.DEFAULT_SCR.
        write_intermediates (bool, optional):
            If True, writes each generator's conformers to *trace_dir* for
            debugging. Defaults to False.

    Returns:
        mol_3d (Optional[Chem.Mol]):
            The molecule with all conformers. Returns None if all generators fail.
    """

    mol_3d = None

    for conf_gen_entry in copy(conf_gens):
        conf_gen = get_generator_function(conf_gen_entry.backend)
        options = conf_gen_entry.model_dump(exclude={"backend"})
        try:
            new_mol_3d = conf_gen(mol, **options)
        except Exception as exc:  # pylint: disable=broad-except
            _logger.warning(
                f"Conformer generation with {conf_gen_entry.backend} failed with Exception {exc}"
            )
            _logger.warning("Skipping this conformer generator.")
            continue

        if new_mol_3d is None or new_mol_3d.GetNumConformers() == 0:
            _logger.warning(f"Conformer generator {conf_gen_entry.backend} returned no conformers")
            _logger.warning("Skipping this conformer generator.")
            continue

        if write_intermediates:
            random_name = generate_name()
            sdf_file = trace_dir / f"confs_{conf_gen_entry.backend}_{random_name}.sdf"
            _logger.debug(f"Writing {conf_gen_entry.backend} results to {sdf_file.resolve()}")
            write_sdf(new_mol_3d, sdf_file)

        mol_3d = join_conformers(mol_3d, new_mol_3d)

    if mol_3d is None:
        _logger.error("All conformer generators failed")
        raise RuntimeError("All conformer generators failed")

    if rms_threshold is not None:
        conformer_geometry.deduplicate_conformers(
            mol_3d, rms_threshold=rms_threshold, method=rmsd_method, num_cores=num_cores
        )

    if write_intermediates:
        random_name = generate_name()
        sdf_file = trace_dir / f"confs_joint_{random_name}.sdf"
        _logger.debug(f"Writing joint conformer set to {sdf_file.resolve()}")
        write_sdf(mol_3d, sdf_file)

    return mol_3d


class ConformerGenerationOptions(BaseModel):
    """Options for the generate_conformers task."""

    conf_gens: list[ConformerGeneratorConfig] = Field(
        ...,
        description="List of conformer generator option objects to use. Each is a typed "
        "options object whose 'backend' field selects the generator.",
    )
    combination_name: Literal["joint_set", "fallback"] = Field(
        default="joint_set",
        description="Method to combine multiple conformer generators: 'joint_set' or 'fallback'",
    )
    rms_threshold: float = Field(
        default=DEFAULT_RMSD_THRESHOLD,
        description="RMSD threshold for deduplication of joint conformer set. Ignored if combination_name is 'fallback'.",
    )
    rmsd_method: str = Field(
        default="tradeoff",
        description="RMSD method for deduplication of joint conformer set. Ignored if combination_name is 'fallback'.",
    )

    @field_validator("conf_gens", mode="before")
    @classmethod
    def deserialize_conf_gens(cls, v: object) -> object:
        """Deserialize conformer generator configs from dicts."""
        if isinstance(v, list):
            return [deserialize_conformer_generator_config(item) for item in v]
        return v


def generate_conformers(
    mol: Chem.Mol,
    conformer_generation_options: ConformerGenerationOptions,
    num_cores: int = 1,
    trace_dir: Path = constants.DEFAULT_SCR,
    write_intermediates: bool = False,
) -> Chem.Mol | None:
    """Main entry point for conformer generation. Dispatches to the appropriate method based on the combination_name."""
    if conformer_generation_options.combination_name == "joint_set":
        return generate_joint_conformer_set(
            mol=mol,
            conf_gens=conformer_generation_options.conf_gens,
            rms_threshold=conformer_generation_options.rms_threshold,
            rmsd_method=conformer_generation_options.rmsd_method,
            num_cores=num_cores,
            trace_dir=trace_dir,
            write_intermediates=write_intermediates,
        )
    elif conformer_generation_options.combination_name == "fallback":
        return generate_conformers_with_fallback(
            mol=mol,
            conf_gens=conformer_generation_options.conf_gens,
        )
