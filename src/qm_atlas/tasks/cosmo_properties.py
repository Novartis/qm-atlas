from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from qm_atlas.constants import DEFAULT_SCR
from qm_atlas.tasks.common import ScalarProperty
from qm_atlas.wrappers.cosmotherm import cosmo_tasks, cpsa
from qm_atlas.wrappers.cosmotherm.cosmo_tasks import (
    COSMO_DELTA_G_KEY,
    COSMO_DESCRIPTORS_COLUMNS,
    COSMO_LOGP_KEY,
    COSMO_PERM_KEY,
)
from qm_atlas.wrappers.cosmotherm.cpsa import COSMO_PSA_KEY

# calculations to allow: cosmo_descriptors, cosmo_psa, cosmo_logp, cosmo_delta_g
# create seperate interfaces for cocrystal screening and solvent screening

# Keys to exclude when converting CosmoConfig objects to kwargs for passing to calculator functions
COSMO_CONFIG_EXCLUDE_KEYS = {"backend", "property_prefix"}


class CosmoConfig(BaseModel):
    # Extra kwargs are forwarded to the cosmotherm output reader at runtime, and allowing
    # them also lets subclass fields survive base-class pre-instantiation by config loaders.
    model_config = ConfigDict(extra="allow")

    level: Literal["bp-tzvpd", "bp-tzvp", "bp-tzvp", "dmol3-pbe"] = Field(
        default="bp-tzvpd",
        description="Level of theory to use for the COSMO calculation.",
    )
    temperature_Celsius: float = Field(
        default=25.0,
        description="Temperature in Celsius to use for the COSMO calculation. Default is 25.0 °C.",
    )
    solvents_dirs: list[Path] = Field(
        default_factory=list,
        description="Additional directories to find the .cosmo files for the solvents. By default, only the cosmotherm database is used.",
    )
    use_config_cosmo_dir: bool = Field(
        default=True,
        description="Whether to prepend the solvent directories from the software configuration file to the user-provided solvents_dirs. If False, only the user-provided directories are used.",
    )
    property_prefix: str | None = Field(
        default=None,
        description="Prefix for property names. Default is set in subclasses if None.",
    )

    def get_property_prefix(self) -> str:
        """Get the property prefix. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement get_property_prefix method")

    def get_property_names(self) -> dict[str, str]:
        raise NotImplementedError("to be implemented in subclasses")

    def get_property_types(self) -> dict[str, type]:
        return {prop_name: ScalarProperty for prop_name in self.get_property_names().values()}


class CosmoLogPConfig(CosmoConfig):
    backend: Literal["cosmo_logp"] = Field(
        default="cosmo_logp", description="Type identifier for this COSMO calculation"
    )

    is_woctanol: bool = Field(
        default=True,
        description="Whether to calculate logP for wet octanol, i.e taking into account the presence of water in the octanol phase. If False, logP is calculated for pure octanol.",
    )
    solvent_names: list[str] = Field(
        default_factory=lambda: ["h2o", "octanol"],
        description="Names of the solvents to use for the COSMO calculation. The first solvent in the list will be used as the reference solvent for the logP calculation (e.g. water for logP).",
    )
    eq_phases: bool = Field(
        default=False,
        description="Compute the phase equilibrium between the given two solvents. For water-octanol, the default options have an experimentally determined phase equilibrium, so this option is not needed.",
    )
    property_prefix: str | None = Field(
        default=None,
        description="Prefix for property names. If None, defaults to ``cosmo_logp_``",
    )

    def get_property_prefix(self) -> str:
        """Determine property prefix."""
        if self.property_prefix is not None:
            return self.property_prefix
        else:
            return ""

    def get_property_names(self) -> dict[str, str]:

        property_prefix = self.get_property_prefix()
        return {COSMO_LOGP_KEY: f"{property_prefix}{COSMO_LOGP_KEY}"}


class CosmoDeltaGConfig(CosmoConfig):
    backend: Literal["cosmo_delta_g"] = Field(
        default="cosmo_delta_g", description="Type identifier for this COSMO calculation"
    )

    solvent_name: str = Field(
        default="h2o",
        description="Name of the solvent to use for the COSMO calculation, e.g. 'h2o' for water, 'self' can be used the compound itself as solvent.",
    )
    property_prefix: str | None = Field(
        default=None,
        description="Prefix for property names. If None, defaults to ``cosmo_delta_g_``",
    )

    def get_property_prefix(self) -> str:
        """Determine property prefix."""
        if self.property_prefix is not None:
            return self.property_prefix
        else:
            return f"{self.solvent_name}_"

    def get_property_names(self) -> dict[str, str]:
        property_prefix = self.get_property_prefix()
        prop_name = f"{property_prefix}{COSMO_DELTA_G_KEY}"
        return {COSMO_DELTA_G_KEY: prop_name}


class CosmoDescriptorsConfig(CosmoConfig):
    backend: Literal["cosmo_descriptors"] = Field(
        default="cosmo_descriptors", description="Type identifier for this COSMO calculation"
    )

    solvent_name: str = Field(
        default="h2o",
        description="Name of the solvent to use for the COSMO calculation, e.g. 'h2o' for water.",
    )
    property_prefix: str | None = Field(
        default=None,
        description="Prefix for property names. If None, defaults to ``cosmo_``",
    )

    def get_property_prefix(self) -> str:
        """Determine property prefix."""
        if self.property_prefix is not None:
            return self.property_prefix
        else:
            return f"cosmo_{self.solvent_name}_"

    def get_property_names(self) -> dict[str, str]:

        property_prefix = self.get_property_prefix()
        name_dict = {}
        for descriptor in COSMO_DESCRIPTORS_COLUMNS:
            if self.level != "bp-tzvpd" and "Gdisp3" in descriptor:
                # skip Gdisp3 descriptor for lower levels of theory
                continue
            prop_name = f"{property_prefix}{descriptor}"
            name_dict[descriptor] = prop_name
        return name_dict


class CosmoPsaConfig(BaseModel):
    # The PSA wrapper takes only explicit parameters, so reject unknown keys as typos.
    model_config = ConfigDict(extra="forbid")

    backend: Literal["cosmo_psa"] = Field(
        default="cosmo_psa", description="Type identifier for this COSMO calculation"
    )
    smoothen: bool = Field(
        default=True,
        description="Whether to smoothen the sigma charge density before calculating the PSA. Smoothing is done according to eq. 11 in J. Phys. Chem. A 1998, 102, 5074-5085, https://doi.org/10.1021/jp980017s",
    )
    on_charges: bool = Field(
        default=False,
        description="Whether to calculate the PSA based on the charge density (sigma) or the charge. If False, the PSA is calculated based on the charge density, if True, it is calculated based on the charge.",
    )
    lower_limit: float = Field(
        default=cpsa.MLIMIT,
        description="Lower limit for the charge density (sigma) to consider a surface segment as polar. Should be adapted from the default value if on_charges is True.",
    )
    upper_limit: float = Field(
        default=cpsa.PLIMIT,
        description="Upper limit for the charge density (sigma) to consider a surface segment as polar. Should be adapted from the default value if on_charges is True.",
    )
    property_prefix: str | None = Field(
        default=None,
        description="Prefix for property names. If None, defaults to ``cosmo_psa_``",
    )

    def get_property_prefix(self) -> str:
        """Determine property prefix."""
        if self.property_prefix is not None:
            return self.property_prefix
        else:
            return ""

    def get_property_names(self) -> dict[str, str]:

        property_prefix = self.get_property_prefix()
        prop_name = f"{property_prefix}{COSMO_PSA_KEY}"
        return {COSMO_PSA_KEY: prop_name}

    def get_property_types(self) -> dict[str, type]:
        return {prop_name: ScalarProperty for prop_name in self.get_property_names().values()}


class CosmoPermConfig(CosmoConfig):
    backend: Literal["cosmo_perm"] = Field(
        default="cosmo_perm", description="Type identifier for this COSMO calculation"
    )
    micelle_file: Path | None = Field(
        default=None, description="Path to the micelle file to use for the COSMOperm calculation."
    )
    micelle_name: str = Field(
        default="dmpc",
        description="Name of the micelle to use (e.g., 'dmpc'). The system will look for 'COSMOmic-{micelle_name}.mic' in the database.",
    )
    pH: float = Field(
        default=7.4,
        description="pH value for the COSMOperm calculation. Default is 7.4.",
    )
    property_prefix: str | None = Field(
        default=None,
        description="Prefix for property names. If None, defaults to 'cosmo_perm_'",
    )

    def get_property_prefix(self) -> str:
        """Determine property prefix."""
        if self.property_prefix is not None:
            return self.property_prefix
        else:
            return ""

    def get_property_names(self) -> dict[str, str]:
        property_prefix = self.get_property_prefix()
        prop_name = f"{property_prefix}{COSMO_PERM_KEY}"
        return {COSMO_PERM_KEY: prop_name}


# Discriminated union of all COSMO task config types. The ``backend`` literal field
# acts as the discriminator so Pydantic selects the concrete class automatically
# when deserializing from a dict, and serializes all fields during model_dump().
CosmoTaskConfig = Annotated[
    CosmoLogPConfig
    | CosmoDeltaGConfig
    | CosmoDescriptorsConfig
    | CosmoPsaConfig
    | CosmoPermConfig,
    Field(discriminator="backend"),
]


def calculate_cosmo_psa(
    compound_conformers: list[Path],
    config: CosmoPsaConfig | CosmoConfig,
    scr: Path = DEFAULT_SCR,
    keep_files: bool = False,
    n_cores: int = 1,
) -> list[dict[str, ScalarProperty]]:
    """Thin wrapper around :func:`~qm_atlas.wrappers.cosmotherm.cpsa.extract_psa_from_cosmo_file`
    that unpacks *config* and renames the output property.
    """
    # scr, keep_files and n_cores are accepted for a uniform calculator interface
    del scr, keep_files, n_cores  # unused but API-required
    if not isinstance(config, CosmoPsaConfig):
        raise ValueError(
            f"Config for COSMO PSA calculation must be of type CosmoPsaConfig, but got {type(config)}"
        )

    psa_results = []
    kwargs = config.model_dump(exclude=COSMO_CONFIG_EXCLUDE_KEYS)
    prop_name_dict = config.get_property_names()

    for conformer_cosmo_file in compound_conformers:
        if conformer_cosmo_file.suffix == ".cosmo":
            psa_result = cpsa.extract_psa_from_cosmo_file(conformer_cosmo_file, **kwargs)
            psa_results.append({prop_name_dict[COSMO_PSA_KEY]: psa_result[COSMO_PSA_KEY]})

    return psa_results


def calculate_logp(
    compound_conformers: list[Path],
    config: CosmoLogPConfig,
    scr: Path = DEFAULT_SCR,
    keep_files: bool = False,
    n_cores: int = 1,
) -> dict[str, ScalarProperty]:
    """Thin wrapper around :func:`~qm_atlas.wrappers.cosmotherm.cosmo_tasks.calculate_logp`
    that unpacks *config* and renames the output property.
    """
    kwargs = config.model_dump(exclude=COSMO_CONFIG_EXCLUDE_KEYS)
    logp_result = cosmo_tasks.calculate_logp(
        compound_conformers=compound_conformers,
        scr=scr,
        keep_files=keep_files,
        n_cores=n_cores,
        **kwargs,
    )
    prop_name_dict = config.get_property_names()
    return {prop_name_dict[COSMO_LOGP_KEY]: logp_result[COSMO_LOGP_KEY]}


def calculate_delta_g(
    compound_conformers: list[Path],
    config: CosmoDeltaGConfig,
    scr: Path = DEFAULT_SCR,
    keep_files: bool = False,
    n_cores: int = 1,
) -> dict[str, ScalarProperty]:
    """Thin wrapper around :func:`~qm_atlas.wrappers.cosmotherm.cosmo_tasks.calculate_delta_g`
    that unpacks *config* and renames the output property.
    """
    kwargs = config.model_dump(exclude=COSMO_CONFIG_EXCLUDE_KEYS)
    delta_g_result = cosmo_tasks.calculate_delta_g(
        compound_conformers=compound_conformers,
        scr=scr,
        keep_files=keep_files,
        n_cores=n_cores,
        **kwargs,
    )
    prop_name_dict = config.get_property_names()
    return {prop_name_dict[COSMO_DELTA_G_KEY]: delta_g_result[COSMO_DELTA_G_KEY]}


def calculate_descriptors(
    compound_conformers: list[Path],
    config: CosmoDescriptorsConfig,
    scr: Path = DEFAULT_SCR,
    keep_files: bool = False,
    n_cores: int = 1,
) -> list[dict[str, ScalarProperty]]:
    """Thin wrapper around :func:`~qm_atlas.wrappers.cosmotherm.cosmo_tasks.calculate_descriptors`
    that unpacks *config* and renames the output properties.
    """
    kwargs = config.model_dump(exclude=COSMO_CONFIG_EXCLUDE_KEYS)
    descriptors_results = cosmo_tasks.calculate_descriptors(
        compound_conformers=compound_conformers,
        scr=scr,
        keep_files=keep_files,
        n_cores=n_cores,
        **kwargs,
    )
    prop_name_dict = config.get_property_names()
    results_with_new_names = []
    for conf_result in descriptors_results:
        conf_result_with_new_names = {}
        # Only iterate over descriptors that are actually in the result
        for descriptor in conf_result:
            conf_result_with_new_names[prop_name_dict[descriptor]] = conf_result[descriptor]
        results_with_new_names.append(conf_result_with_new_names)
    return results_with_new_names


def calculate_perm(
    compound_conformers: list[Path],
    config: CosmoPermConfig,
    scr: Path = DEFAULT_SCR,
    keep_files: bool = False,
    n_cores: int = 1,
) -> dict[str, ScalarProperty]:
    """Thin wrapper around :func:`~qm_atlas.wrappers.cosmotherm.cosmo_tasks.calculate_cosmoperm`
    that unpacks *config* and renames the output property.
    """
    kwargs = config.model_dump(exclude=COSMO_CONFIG_EXCLUDE_KEYS)
    perm_result = cosmo_tasks.calculate_cosmoperm(
        compound_conformers=compound_conformers,
        scr=scr,
        keep_files=keep_files,
        n_cores=n_cores,
        **kwargs,
    )
    prop_name_dict = config.get_property_names()
    return {prop_name_dict[COSMO_PERM_KEY]: perm_result[COSMO_PERM_KEY]}


CALCULATOR_REGISTRY = {
    "cosmo_logp": calculate_logp,
    "cosmo_delta_g": calculate_delta_g,
    "cosmo_descriptors": calculate_descriptors,
    "cosmo_psa": calculate_cosmo_psa,
    "cosmo_perm": calculate_perm,
}

OPTIONS_CLASS_REGISTRY = {
    "cosmo_logp": CosmoLogPConfig,
    "cosmo_delta_g": CosmoDeltaGConfig,
    "cosmo_descriptors": CosmoDescriptorsConfig,
    "cosmo_psa": CosmoPsaConfig,
    "cosmo_perm": CosmoPermConfig,
}


def run_cosmo_calculation(
    conformer_cosmo_files: list[Path],
    config: CosmoTaskConfig,
    scr: Path = DEFAULT_SCR,
    keep_files: bool = False,
    n_cores: int = 1,
) -> dict[str, ScalarProperty] | list[dict[str, ScalarProperty]]:
    """Dispatch to the COSMO calculator selected by ``config.backend`` (see
    :data:`CALCULATOR_REGISTRY`) and return its named properties.
    """
    calculator = CALCULATOR_REGISTRY[config.backend]
    calculation_result = calculator(
        conformer_cosmo_files,
        config,
        scr=scr,
        keep_files=keep_files,
        n_cores=n_cores,
    )
    return calculation_result
