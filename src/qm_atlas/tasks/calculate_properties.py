import logging
from itertools import product
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from ppqm import chembridge
from pydantic import BaseModel, ConfigDict, Field
from rdkit import Chem

from qm_atlas.constants import COSMO_OUTPUT_KEY, DEFAULT_SCR
from qm_atlas.tasks.common import AtomBasedProperty, Property, ScalarProperty, TensorProperty
from qm_atlas.tasks.utils.parallel import run_parallel
from qm_atlas.wrappers import xtb
from qm_atlas.wrappers.jaguar import bond_dissociation, qm_descriptors
from qm_atlas.wrappers.turbomole import freeh, fukui, nmr_shielding, single_point, vcd
from qm_atlas.wrappers.xtb import XTB_ENERGY_KEY, XTB_PROPERTY_TYPES

_logger = logging.getLogger(__name__)


XTB_SP_KEY = "xtb_single_point"
TM_SP_KEY = "turbomole_single_point"
TM_FH_KEY = "turbomole_freeh"
TM_FUKUI_KEY = "turbomole_fukui_indices"
TM_NMR_KEY = "turbomole_nmr_shielding"
TM_VCD_KEY = "turbomole_vcd"
JAGUAR_HABS_KEY = "jaguar_h_abstraction_energies"
JAGUAR_PROPS_KEY = "jaguar_properties"

# Keys to exclude when converting PropertyCalculatorOptions objects to kwargs for passing to calculator functions
CALCULATOR_OPTIONS_EXCLUDE_KEYS = {"backend", "property_prefix"}

CALCULATOR_REGISTRY = {
    XTB_SP_KEY: xtb.calculate_single_point,
    TM_SP_KEY: single_point.run_ridft,
    TM_FH_KEY: freeh.calculate_freeh,
    TM_FUKUI_KEY: fukui.calculate_fukui,
    TM_NMR_KEY: nmr_shielding.calculate_nmr_shieldings,
    TM_VCD_KEY: vcd.calculate_vcd,
    JAGUAR_HABS_KEY: bond_dissociation.get_hydrogen_abstraction_energies,
    JAGUAR_PROPS_KEY: qm_descriptors.calculate_descriptors,
}


class PropertyCalculatorOptions(BaseModel):
    """Base class for property calculator options. Not used directly, but serves as a common parent for all calculator option classes."""

    # Allow extra fields so that subclass-specific options (e.g. basis, functional)
    # survive when config loaders (jsonargparse) pre-instantiate a list item as this
    # base class before re-dispatching to the concrete subclass. Without this, those
    # fields would be silently dropped and the calculation would fall back to defaults.
    model_config = ConfigDict(extra="allow")

    property_prefix: str | None = Field(
        default=None,
        description="Prefix for property names. If None, defaults to a calculator-specific value",
    )

    def get_property_prefix(self) -> str:
        raise NotImplementedError("Subclasses must implement get_property_prefix method")

    def get_property_names(self) -> dict[str, str]:
        """Generate property names based on non-default settings. Must be implemented by each calculator options class."""
        raise NotImplementedError("Subclasses must implement get_property_names method")

    def get_property_types(self) -> dict[str, type]:
        """Map output keys to Property types. Must be implemented by each calculator options class."""
        raise NotImplementedError("Subclasses must implement get_property_types method")


class XtbSinglePointOptions(PropertyCalculatorOptions):
    """Options for xtb.calculate_single_point (exposed CLI parameters only).

    To customise the input beyond the exposed fields, supply a fully rendered
    ``template_str``; unknown keyword arguments are rejected.
    """

    model_config = ConfigDict(extra="forbid")

    backend: Literal["xtb_single_point"] = Field(
        default=XTB_SP_KEY, description="Type identifier for this calculator configuration"
    )
    solvation_model: str | None = Field(
        default=None, description="Solvation model (alpb, gbe, gbsa, cosmo, cpmcx)"
    )
    solvent: str | None = Field(default=None, description="Solvent name (e.g., water, methanol)")
    calculate_fukui: bool = Field(default=False, description="Calculate Fukui indices")
    xtb_command_add: list[str] | None = Field(
        default=None, description="Additional command-line arguments to pass to xtb"
    )
    template_str: str | None = Field(
        default=None,
        description="Jinja2 template for xcontrol input file (if None, uses xtb default)",
    )
    property_prefix: str | None = Field(
        default=None,
        description="Prefix for property names. If None, defaults to a calculator-specific value",
    )

    def get_property_prefix(self) -> str:
        """Determine property prefix based on solvation settings."""
        if self.property_prefix is not None:
            return self.property_prefix
        elif self.solvation_model is not None:
            return f"xtb_{self.solvation_model}_{self.solvent}_"
        else:
            return "xtb_gas_phase_"

    def get_property_names(self) -> dict[str, str]:
        """Generate property names based on non-default settings."""

        property_prefix = self.get_property_prefix()

        name_dict = {}
        for key in XTB_PROPERTY_TYPES:
            name_dict[key] = f"{property_prefix}{key}"

        if self.calculate_fukui:
            for key in xtb.XTB_FUKUI_PROPERTY_TYPES:
                name_dict[key] = f"{property_prefix}{key}"

        return name_dict

    def get_property_types(self) -> dict[str, type]:
        """Map output keys to Property types."""

        property_names = self.get_property_names()
        type_dict = {}
        all_property_types = {**XTB_PROPERTY_TYPES, **xtb.XTB_FUKUI_PROPERTY_TYPES}

        for key, prop_name in property_names.items():
            # Get the descriptive name from get_property_names
            prop_type = all_property_types.get(key, ScalarProperty)
            type_dict[prop_name] = prop_type

        return type_dict


class TurbomoleSinglePointOptions(PropertyCalculatorOptions):
    """Options for turbomole.single_point.run_ridft.

    To customise the input beyond the exposed fields, supply a fully rendered
    ``control_template``; unknown keyword arguments are rejected.
    """

    model_config = ConfigDict(extra="forbid")

    backend: Literal["turbomole_single_point"] = Field(
        default=TM_SP_KEY, description="Type identifier for this calculator configuration"
    )
    basis: str = Field(default="def2-TZVPD", description="Basis set for the calculation")
    functional: list[str] = Field(default=["b-p"], description="DFT functional(s) to use")
    grid: str = Field(default="m4", description="Grid size for integration (e.g., m3, m4)")
    scf_conv: int = Field(
        default=7,
        description="SCF energy convergence criterion (10^-scf_conv). Higher = tighter convergence.",
    )
    scfiterlimit: int = Field(default=500, description="Maximum number of SCF iterations")
    ricore: int | None = Field(default=None, description="RICORE memory parameter")
    use_disp: bool = Field(
        default=False, description="Whether to include dispersion corrections (D3 BJ)"
    )
    use_cosmo: bool = Field(default=True, description="Whether to use COSMO solvation model")
    klamt: bool = Field(default=True, description="Whether to use Klamt's COSMO variant")
    solvent: str = Field(default="conductor", description="Solvent for COSMO calculations")
    control_template: str | None = Field(
        default=None,
        description="Jinja2 template for control file (if None, uses turbomole default)",
    )
    property_prefix: str | None = Field(
        default=None,
        description="Prefix for property names. If None, defaults to ``tm_sp_``",
    )

    def get_property_prefix(self) -> str:
        """Determine property prefix."""
        if self.property_prefix is not None:
            return self.property_prefix
        else:
            return "tm_sp_"

    def get_property_names(self) -> dict[str, str]:
        """Generate property names based on non-default settings."""
        property_prefix = self.get_property_prefix()

        # Output keys from parse_ridft
        output_keys = ["HOMO(eV)", "LUMO(eV)", "HLgap(eV)", "TotalEnergy(Ht)"]
        name_dict = {key: f"{property_prefix}{key}" for key in output_keys}
        return name_dict

    def get_property_types(self) -> dict[str, type]:
        """Map output keys to Property types."""
        name_dict = self.get_property_names()
        return {name: ScalarProperty for name in name_dict.values()}


class TurbomoleFreehOptions(PropertyCalculatorOptions):
    """Options for turbomole.freeh.calculate_freeh.

    To customise the input beyond the exposed fields, supply a fully rendered
    ``control_template``; unknown keyword arguments are rejected.
    """

    model_config = ConfigDict(extra="forbid")

    backend: Literal["turbomole_freeh"] = Field(
        default=TM_FH_KEY, description="Type identifier for this calculator configuration"
    )

    basis: str = Field(default="def2-TZVP", description="Basis set for the calculation")
    functional: list[str] | tuple[str, ...] = Field(
        default=("b3-lyp",), description="DFT functional(s) to use"
    )
    grid: str = Field(default="m3", description="Grid size for integration (e.g., m3, m4)")
    scf_conv: int = Field(
        default=7,
        description="SCF energy convergence criterion (10^-scf_conv). Higher = tighter convergence.",
    )
    scfiterlimit: int = Field(default=200, description="Maximum number of SCF iterations")
    scfdamp: str = Field(
        default="start=0.300  step=0.050  min=0.100", description="SCF damping parameters"
    )
    ricore: int | None = Field(default=None, description="RICORE memory parameter")
    use_disp: bool = Field(
        default=False, description="Whether to include dispersion corrections (D3 BJ)"
    )
    use_cosmo: bool = Field(default=False, description="Whether to use COSMO solvation model")
    klamt: bool = Field(default=True, description="Whether to use Klamt's COSMO variant")
    solvent: str = Field(default="conductor", description="Solvent for COSMO calculations")
    freeh_settings: str = Field(
        default=freeh.FREEH_SETTINGS,
        description="Settings string specifying temperature, pressure conditions for freeh",
    )
    control_template: str | None = Field(
        default=None,
        description="Jinja2 template for control file (if None, uses turbomole default)",
    )
    property_prefix: str | None = Field(
        default=None,
        description="Prefix for property names. If None, defaults to ``tm_freeh_``",
    )

    def get_property_prefix(self) -> str:
        """Determine property prefix."""
        if self.property_prefix is not None:
            return self.property_prefix
        else:
            return "tm_freeh_"

    def get_property_names(self) -> dict[str, str]:
        """Generate property names based on non-default settings."""
        property_prefix = self.get_property_prefix()

        # Output keys from parse_freeh and ridft
        output_keys = [
            "HOMO(eV)",
            "LUMO(eV)",
            "HLgap(eV)",
            "TotalEnergy(Ht)",
            "chem.pot.",
            "energy",
            "entropy",
            "enthalpy",
        ]
        name_dict = {key: f"{property_prefix}{key}" for key in output_keys}
        return name_dict

    def get_property_types(self) -> dict[str, type]:
        """Map output keys to Property types."""
        name_dict = self.get_property_names()
        return {name: ScalarProperty for name in name_dict.values()}


class TurbomoleFukuiOptions(PropertyCalculatorOptions):
    """Options for turbomole.fukui.calculate_fukui.

    To customise the input beyond the exposed fields, supply a fully rendered
    ``control_template``; unknown keyword arguments are rejected.
    """

    model_config = ConfigDict(extra="forbid")

    backend: Literal["turbomole_fukui_indices"] = Field(
        default=TM_FUKUI_KEY, description="Type identifier for this calculator configuration"
    )

    basis: str = Field(default="def2-SV(P)", description="Basis set for the calculation")
    functional: list[str] | tuple[str, ...] = Field(
        default=("pbe",), description="DFT functional(s) to use"
    )
    grid: str = Field(default="m3", description="Grid size for integration (e.g., m3, m4)")
    scf_conv: int = Field(
        default=7,
        description="SCF energy convergence criterion (10^-scf_conv). Higher = tighter convergence.",
    )
    scfiterlimit: int = Field(default=100, description="Maximum number of SCF iterations")
    scfdamp: str = Field(
        default="start=2.000  step=0.050  min=0.100", description="SCF damping parameters"
    )
    ricore: int | None = Field(default=None, description="RICORE memory parameter")
    use_disp: bool = Field(
        default=False, description="Whether to include dispersion corrections (D3 BJ)"
    )
    use_cosmo: bool = Field(default=False, description="Whether to use COSMO solvation model")
    klamt: bool = Field(default=True, description="Whether to use Klamt's COSMO variant")
    solvent: str = Field(default="conductor", description="Solvent for COSMO calculations")
    charge_schemes: list[str] | None = Field(
        default=None,
        description="Charge schemes to use (nbo, mulliken, loewdin, paboon). If None, all are used.",
    )
    control_template: str | None = Field(
        default=None,
        description="Jinja2 template for control file (if None, uses turbomole default)",
    )
    property_prefix: str | None = Field(
        default=None,
        description="Prefix for property names. If None, defaults to ``tm_fukui_``",
    )

    def get_property_prefix(self) -> str:
        """Determine property prefix."""
        if self.property_prefix is not None:
            return self.property_prefix
        else:
            return "tm_fukui_"

    def get_atom_property_names(self) -> dict[str, str]:

        property_prefix = self.get_property_prefix()

        # Charge schemes: nbo, mulliken, loewdin, paboon
        # These are the output keys
        schemes = ["nbo", "mulliken", "loewdin", "paboon"]
        if self.charge_schemes is not None:
            schemes = self.charge_schemes
        index_names = ["nucleophilic", "electrophilic", "radical", "charge"]

        name_dict = {}
        for index_name, scheme in product(index_names, schemes):
            full_name = f"{property_prefix}fukui_{index_name}_{scheme}"
            name_dict[f"{index_name}_{scheme}"] = full_name

        return name_dict

    def get_scalar_property_names(self) -> dict[str, str]:
        property_prefix = self.get_property_prefix()

        single_keys = [
            "ionization_potential [eV]",
            "electron_affinity [eV]",
            "hardness [eV]",
            "electronegativity [eV]",
            "electrophilicity [eV]",
        ]
        name_dict = {}
        for key in single_keys:
            name_dict[key] = f"{property_prefix}{key}"
        return name_dict

    def get_property_names(self) -> dict[str, str]:
        """Generate property names based on non-default settings."""
        name_dict = self.get_atom_property_names()
        name_dict.update(self.get_scalar_property_names())
        return name_dict

    def get_property_types(self) -> dict[str, type]:
        """Map output keys to Property types."""
        atom_based_names = self.get_atom_property_names()
        scalar_names = self.get_scalar_property_names()
        type_dict = {}
        type_dict.update({name: AtomBasedProperty for name in atom_based_names.values()})
        type_dict.update({name: ScalarProperty for name in scalar_names.values()})
        return type_dict


class TurbomoleNmrShieldingOptions(PropertyCalculatorOptions):
    """Options for turbomole.nmr_shielding.calculate_nmr_shieldings (exposed CLI parameters only).

    To customise the input beyond the exposed fields, supply a fully rendered
    ``control_template``; unknown keyword arguments are rejected.
    """

    model_config = ConfigDict(extra="forbid")

    backend: Literal["turbomole_nmr_shielding"] = Field(
        default=TM_NMR_KEY,
        description="Type identifier for this calculator configuration",
    )

    basis: str = Field(default="def2-TZVP", description="Basis set for the calculation")
    functional: list[str] | tuple[str, ...] = Field(
        default=("xcfun set-gga", "xcfun kt3 1.0"), description="DFT functional(s) to use"
    )
    grid: str = Field(default="3", description="Grid size for integration (e.g., 3, 4, m3, m4)")
    scf_conv: int = Field(
        default=7,
        description="SCF energy convergence criterion (10^-scf_conv). Higher = tighter convergence.",
    )
    scfiterlimit: int = Field(default=200, description="Maximum number of SCF iterations")
    ricore: int | None = Field(default=None, description="RICORE memory parameter")
    use_disp: bool = Field(
        default=False, description="Whether to include dispersion corrections (D3 BJ)"
    )
    use_cosmo: bool = Field(default=False, description="Whether to use COSMO solvation model")
    klamt: bool = Field(default=True, description="Whether to use Klamt's COSMO variant")
    solvent: str = Field(default="conductor", description="Solvent for COSMO calculations")
    control_template: str | None = Field(
        default=None,
        description="Jinja2 template for control file (if None, uses turbomole default)",
    )
    property_prefix: str | None = Field(
        default=None,
        description="Prefix for property names. If None, defaults to ``tm_nmr_shieldings_``",
    )

    def get_property_prefix(self) -> str:
        """Determine property prefix."""
        if self.property_prefix is not None:
            return self.property_prefix
        else:
            return "tm_nmr_shieldings_"

    def get_property_names(self) -> dict[str, str]:
        """Generate property names based on non-default settings."""
        property_prefix = self.get_property_prefix()

        # Output key from nmr_shielding module
        prop_name = f"{property_prefix}isotropic"
        return {"nmr_shieldings_isotropic": prop_name}

    def get_property_types(self) -> dict[str, type]:
        """Map output keys to Property types."""
        name_dict = self.get_property_names()
        return {name: AtomBasedProperty for name in name_dict.values()}


class TurbomoleVcdOptions(PropertyCalculatorOptions):
    """Options for turbomole.vcd.calculate_vcd (exposed CLI parameters only).

    To customise the input beyond the exposed fields, supply a fully rendered
    ``control_template``; unknown keyword arguments are rejected.
    """

    model_config = ConfigDict(extra="forbid")

    backend: Literal["turbomole_vcd"] = Field(
        default=TM_VCD_KEY, description="Type identifier for this calculator configuration"
    )

    basis: str = Field(default="def2-SVP", description="Basis set for the calculation")
    functional: list[str] | tuple[str, ...] = Field(
        default=("b3-lyp",), description="DFT functional(s) to use"
    )
    grid: str = Field(
        default="4", description="Grid size for integration (numeric format like 3, 4, 5)"
    )
    scf_conv: int = Field(
        default=8,
        description="SCF energy convergence criterion (10^-scf_conv). Higher = tighter convergence.",
    )
    scfiterlimit: int = Field(default=500, description="Maximum number of SCF iterations")
    ricore: int | None = Field(default=None, description="RICORE memory parameter")
    use_disp: bool = Field(
        default=False, description="Whether to include dispersion corrections (D3 BJ)"
    )
    use_cosmo: bool = Field(default=False, description="Whether to use COSMO solvation model")
    klamt: bool = Field(default=True, description="Whether to use Klamt's COSMO variant")
    solvent: str = Field(default="conductor", description="Solvent for COSMO calculations")
    control_template: str | None = Field(
        default=None,
        description="Jinja2 template for control file (if None, uses turbomole default)",
    )
    property_prefix: str | None = Field(
        default=None,
        description="Prefix for property names. If None, defaults to ``tm_vcd_``",
    )

    def get_property_prefix(self) -> str:
        """Determine property prefix."""
        if self.property_prefix is not None:
            return self.property_prefix
        else:
            return "tm_vcd_"

    def get_property_names(self) -> dict[str, str]:
        """Generate property names based on non-default settings."""
        property_prefix = self.get_property_prefix()
        # VCD returns spectral data as TensorProperty
        prop_name = f"{property_prefix}spectrum"
        return {"vcd_spectrum": prop_name}

    def get_property_types(self) -> dict[str, type]:
        """Map output keys to Property types."""
        name_dict = self.get_property_names()
        return {name: TensorProperty for name in name_dict.values()}


class JaguarHydrogenAbstractionOptions(PropertyCalculatorOptions):
    """Options for jaguar.bond_dissociation.get_hydrogen_abstraction_energies.

    Note: Internal parameters like keep_files, scr, and n_cores are not exposed via CLI.
    """

    model_config = ConfigDict(extra="forbid")

    backend: Literal["jaguar_h_abstraction_energies"] = Field(
        default=JAGUAR_HABS_KEY,
        description="Type identifier for this calculator configuration",
    )

    atom_idcs: list[int] | str = Field(
        default="C",
        description="Atom indices (list of int) or element symbol(s) ('C', 'N', 'C,N')",
    )
    geo_opt_settings: dict[str, str] | None = Field(
        default=None,
        description="Settings for geometry optimization (e.g., {'dftname': 'b3lyp', 'basis': 'lacvp*'})",
    )
    single_point_settings: dict[str, str] | None = Field(
        default=None, description="Settings for single point energy calculations"
    )
    jaguar_options: dict[str, str] | None = Field(
        default=None, description="Additional jaguar command-line options"
    )
    license_buffer: int = Field(default=40, description="Number of licenses to keep in reserve")
    license_timeout: int = Field(
        default=2880, description="Timeout for license acquisition in minutes (default: 2 days)"
    )
    property_prefix: str | None = Field(
        default=None,
        description="Prefix for property names. If None, defaults to 'jaguar_h_abstraction_energies'",
    )

    def get_property_prefix(self) -> str:
        """Determine property prefix."""
        if self.property_prefix is not None:
            return self.property_prefix
        else:
            return ""

    def get_property_names(self) -> dict[str, str]:
        """Generate property names based on non-default settings."""
        property_prefix = self.get_property_prefix()

        # Returns per-atom hydrogen abstraction energies
        prop_name = f"{property_prefix}"
        return {"h_abstraction_energies": prop_name}

    def get_property_types(self) -> dict[str, type]:
        """Map output keys to Property types."""
        name_dict = self.get_property_names()
        return {name: AtomBasedProperty for name in name_dict.values()}


class JaguarQmDescriptorsOptions(PropertyCalculatorOptions):
    """Options for jaguar.qm_descriptors.calculate_descriptors (exposed CLI parameters only).

    Note: Internal parameters like keep_files, scr, and n_cores are not exposed via CLI.
    """

    model_config = ConfigDict(extra="forbid")

    backend: Literal["jaguar_properties"] = Field(
        default=JAGUAR_PROPS_KEY,
        description="Type identifier for this calculator configuration",
    )

    properties: list[str] | tuple[str, ...] = Field(
        default=("fukui", "lowdin", "nmr"),
        description="Properties to calculate (choices: fukui, lowdin, nmr)",
    )
    basis: str = Field(default="LACVP*", description="Basis set for the calculation")
    functional: list[str] | tuple[str, ...] = Field(
        default=("b3lyp-d3",), description="DFT functional(s) to use"
    )
    jaguar_options: dict[str, str] | None = Field(
        default=None, description="Additional jaguar command-line options"
    )
    license_buffer: int = Field(default=40, description="Number of licenses to keep in reserve")
    license_timeout: int = Field(
        default=2880, description="Timeout for license acquisition in minutes (default: 2 days)"
    )
    property_prefix: str | None = Field(
        default=None,
        description="Prefix for property names. If None, defaults to ``jaguar_``",
    )

    def get_property_prefix(self) -> str:
        """Determine property prefix."""
        if self.property_prefix is not None:
            return self.property_prefix
        else:
            return "jaguar_"

    def get_property_names(self) -> dict[str, str]:
        """Generate property names based on non-default settings."""
        property_prefix = self.get_property_prefix()

        property_names = {
            "fukui": [
                "s_j_Atom_Fukui_Index_f_NN_HOMO",
                "s_j_Atom_Fukui_Index_f_NS_HOMO",
                "s_j_Atom_Fukui_Index_f_SN_HOMO",
                "s_j_Atom_Fukui_Index_f_SS_HOMO",
                "s_j_Atom_Fukui_Index_f_NN_LUMO",
                "s_j_Atom_Fukui_Index_f_NS_LUMO",
                "s_j_Atom_Fukui_Index_f_SN_LUMO",
                "s_j_Atom_Fukui_Index_f_SS_LUMO",
            ],
            "lowdin": ["s_j_Atom_Lowdin_Charge"],
            "nmr": [
                "s_j_Atom_NMR_Isotropic_Shielding",
                "s_j_NMR_Atomic_Absolute_Shifts",
            ],
        }

        name_dict = {}
        for prop in self.properties:
            keys = property_names.get(prop, [])
            for key in keys:
                name_dict[key] = f"{property_prefix}{key}"

        return name_dict

    def get_property_types(self) -> dict[str, type]:
        """Map output keys to Property types."""
        name_dict = self.get_property_names()
        return {name: AtomBasedProperty for name in name_dict.values()}


CONFIG_CLASS_REGISTRY = {
    XTB_SP_KEY: XtbSinglePointOptions,
    TM_SP_KEY: TurbomoleSinglePointOptions,
    TM_FH_KEY: TurbomoleFreehOptions,
    TM_FUKUI_KEY: TurbomoleFukuiOptions,
    TM_NMR_KEY: TurbomoleNmrShieldingOptions,
    TM_VCD_KEY: TurbomoleVcdOptions,
    JAGUAR_HABS_KEY: JaguarHydrogenAbstractionOptions,
    JAGUAR_PROPS_KEY: JaguarQmDescriptorsOptions,
}


# Discriminated union of all property-calculator config types. The ``backend``
# literal field acts as the discriminator so Pydantic selects the concrete class
# automatically when deserializing from a dict, and serializes all fields during
# model_dump().
CalculationConfig = Annotated[
    XtbSinglePointOptions
    | TurbomoleSinglePointOptions
    | TurbomoleFreehOptions
    | TurbomoleFukuiOptions
    | TurbomoleNmrShieldingOptions
    | TurbomoleVcdOptions
    | JaguarHydrogenAbstractionOptions
    | JaguarQmDescriptorsOptions,
    Field(discriminator="backend"),
]


def create_calculation_config(backend: str, **kwargs) -> CalculationConfig:
    """Convenience function to create a CalculationConfig object based on the backend and provided options."""
    config_class = CONFIG_CLASS_REGISTRY.get(backend)

    if config_class is None:
        raise ValueError(f"Unknown calculator name: {backend}")

    return config_class(**kwargs)


def calculate_property(
    mol: Chem.Mol,
    conf_id: int,
    config: CalculationConfig,
    suppress_errors: bool = True,
    scr: Path = DEFAULT_SCR,
    n_cores: int = 1,
    keep_files: bool = False,
) -> dict[str, Property] | None:
    """Calculate a molecular property for a single conformer using a specified calculator.

    Dispatches to the appropriate calculator backend based on the backend
    specified in the config object. Operates on a single conformer identified by its ID
    within a molecule.

    Args:
        mol (Chem.Mol):
            RDKit molecule containing the conformer to analyze.
        conf_id (int):
            ID of the conformer within the molecule to calculate the property for.
        config (CalculationConfig):
            Configuration object specifying the calculator to use and its options.
            The backend attribute determines which backend is used.
            Available config options are:
            - XtbSinglePointOptions for xtb single-point calculations
            - TurbomoleSinglePointOptions for turbomole single-point calculations
            - TurbomoleFreehOptions for turbomole freeh calculations
            - TurbomoleFukuiOptions for turbomole Fukui index calculations
            - TurbomoleNmrShieldingOptions for turbomole NMR shielding calculations
            - TurbomoleVcdOptions for turbomole VCD calculations
            - JaguarHydrogenAbstractionOptions for Jaguar hydrogen abstraction energy calculations
            - JaguarQmDescriptorsOptions for Jaguar QM descriptor calculations
        suppress_errors (bool, optional):
            If True, suppresses exceptions during calculation and returns None
            on failure. If False, exceptions are propagated. Defaults to True.
        scr (Path, optional):
            Scratch directory path for temporary calculation files. Defaults to DEFAULT_SCR.
        n_cores (int, optional):
            Number of CPU cores to use for the calculation. Defaults to 1.
        keep_files (bool, optional):
            If True, preserves temporary calculation files after completion.
            If False, cleans up temporary files. Defaults to False.

    Returns:
        dict[str, Property] | None:
            Dictionary mapping property names to Property objects. Each Property
            object contains the calculated value and associated metadata.
            Returns None if the calculation failed and suppress_errors is True.

    Raises:
        ValueError:
            If config.backend is not found in CALCULATOR_REGISTRY.
        RuntimeError:
            If the underlying calculator function raises an error and suppress_errors is False.

    Examples:
        >>> from rdkit import Chem, AllChem
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> mol = Chem.AddHs(mol)
        >>> AllChem.EmbedMolecule(mol)
        >>> config = XtbSinglePointOptions()
        >>> props = calculate_property(mol, 0, config)
        >>> energy = props["energy"].get_property_value()
    """
    if config.backend not in CALCULATOR_REGISTRY:
        raise ValueError(f"Unknown property calculator: {config.backend}")

    calculator = CALCULATOR_REGISTRY[config.backend]
    try:
        kwargs = config.model_dump(exclude=CALCULATOR_OPTIONS_EXCLUDE_KEYS)
        calculation_result = calculator(
            mol, conf_id, scr=scr, n_cores=n_cores, keep_files=keep_files, **kwargs
        )
        renamed_calculation_result = {}
        property_names = config.get_property_names()
        for key, value in calculation_result.items():
            if key in property_names:
                renamed_calculation_result[property_names[key]] = value
            else:
                renamed_calculation_result[key] = value
        return renamed_calculation_result

    except RuntimeError:
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else "UNNAMED"
        _logger.error(
            f"Calculation {config.backend} failed for molecule: {name}, Conformer ID: {conf_id}"
        )
        if not suppress_errors:
            raise
        return None


def calculate_property_conformer(
    conf: Chem.Conformer,
    config: CalculationConfig,
    suppress_errors: bool = True,
    scr: Path = DEFAULT_SCR,
    n_cores: int = 1,
    keep_files: bool = False,
) -> dict[str, Property] | None:
    """Calculate a molecular property for a conformer using a specified calculator.

    Convenience wrapper around calculate_property that accepts an RDKit Conformer
    object directly instead of requiring the separate molecule and conformer ID.
    Extracts the owning molecule and conformer ID automatically.

    Args:
        conf (Chem.Conformer):
            RDKit conformer object to calculate the property for. The conformer
            must have a parent molecule with proper connectivity information.
        config (CalculationConfig):
            Configuration object specifying the calculator to use and its options.
            The backend attribute determines which backend is used.
        suppress_errors (bool, optional):
            If True, suppresses exceptions during calculation and returns None
            on failure. If False, exceptions are propagated. Defaults to True.
        scr (Path, optional):
            Scratch directory path for temporary calculation files. Defaults to DEFAULT_SCR.
        n_cores (int, optional):
            Number of CPU cores to use for the calculation. Defaults to 1.
        keep_files (bool, optional):
            If True, preserves temporary calculation files after completion.
            If False, cleans up temporary files. Defaults to False.

    Returns:
        dict[str, Property] | None:
            Dictionary mapping property names to Property objects containing
            calculated values and metadata.

    Raises:
        ValueError:
            If config.backend is not found in CALCULATOR_REGISTRY.
        RuntimeError:
            If the underlying calculator function raises an error and suppress_errors is False.

    Examples:
        >>> from rdkit import Chem, AllChem
        >>> mol = Chem.MolFromSmiles("CC")
        >>> mol = Chem.AddHs(mol)
        >>> AllChem.EmbedMolecule(mol)
        >>> conformer = mol.GetConformer()
        >>> config = XtbSinglePointOptions()
        >>> props = calculate_property_conformer(conformer, config)
    """
    mol = conf.GetOwningMol()
    conf_id = conf.GetId()
    return calculate_property(
        mol,
        conf_id,
        config,
        suppress_errors=suppress_errors,
        scr=scr,
        n_cores=n_cores,
        keep_files=keep_files,
    )


def calculate_property_mol(
    mol: Chem.Mol,
    config: CalculationConfig,
    n_cores: int = 1,
    show_progress: bool = False,
    scr: Path = DEFAULT_SCR,
    keep_files: bool = False,
) -> list[dict[str, Property] | None]:
    """Calculate a molecular property for all conformers of a molecule. Uses
    multiprocessing if multiple CPU cores are specified.

    Automatically handles conformer iteration and returns results organized
    by conformer.

    Args:
        mol (Chem.Mol):
            RDKit molecule with one or more conformers to calculate properties for.
        config (CalculationConfig):
            Configuration object specifying the calculator to use and its options.
            The backend attribute determines which backend is used.
        n_cores (int, optional):
            Number of CPU cores to use for parallel processing. The actual number
            of processes spawned will be min(n_cores, total_num_conformers).
            Defaults to 1 (sequential processing).
        show_progress (bool, optional):
            If True, displays a progress bar during calculation. Defaults to False.
        scr (Path, optional):
            Scratch directory path for temporary calculation files. Defaults to DEFAULT_SCR.
        keep_files (bool, optional):
            If True, preserves temporary calculation files after completion.
            If False, cleans up temporary files. Defaults to False.

    Returns:
        list[dict[str, Property] | None]:
            List of property dictionaries, one for each conformer in the input
            molecule. Each element is either a dict of calculated Property objects
            or None if the calculation failed. The list length matches the number
            of conformers in the input molecule.

    Raises:
        ValueError:
            If config.backend is not found in CALCULATOR_REGISTRY.

    Examples:
        >>> from rdkit import Chem, AllChem
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> mol = Chem.AddHs(mol)
        >>> AllChem.EmbedMultipleConfs(mol, 3)
        >>> config = XtbSinglePointOptions()
        >>> all_props = calculate_property_mol(mol, config)
        >>> # all_props is a list of 3 property dicts, one for each conformer
        >>> energy_conf_0 = all_props[0]["energy"].get_property_value()
    """
    calculated_properties = calculate_property_mols(
        [mol],
        config,
        n_cores=n_cores,
        show_progress=show_progress,
        scr=scr,
        keep_files=keep_files,
    )
    return calculated_properties[0]


def calculate_property_mols(
    mols: list[Chem.Mol],
    config: CalculationConfig,
    n_cores: int = 1,
    show_progress: bool = False,
    scr: Path = DEFAULT_SCR,
    keep_files: bool = False,
) -> list[list[dict[str, Property] | None]]:
    """Calculate a molecular property for all conformers across multiple molecules in parallel.

    High-level function for batch property calculation. All conformers from all
    molecules are processed in parallel to maximize computational efficiency.
    Results are automatically reorganized by molecule in the output.

    Args:
        mols (list[Chem.Mol]):
            List of RDKit molecules with conformers to calculate properties for.
        config (CalculationConfig):
            Configuration object specifying the calculator to use and its options.
            The backend attribute determines which backend is used.
        n_cores (int, optional):
            Number of CPU cores to use for parallel processing. The actual number
            of processes spawned will be min(n_cores, total_num_conformers).
            Defaults to 1 (sequential processing).
        show_progress (bool, optional):
            If True, displays a progress bar during calculation. Defaults to False.
        scr (Path, optional):
            Scratch directory path for temporary calculation files. Defaults to DEFAULT_SCR.
        keep_files (bool, optional):
            If True, preserves temporary calculation files after completion.
            If False, cleans up temporary files. Defaults to False.

    Returns:
        list[list[dict[str, Property] | None]]:
            Nested list organized as: output[mol_idx][conf_idx] = properties_dict
            Each properties_dict contains calculated Property objects or None if
            the calculation failed. Output list length matches input list length.
            Inner list lengths match the number of conformers for each molecule.

    Raises:
        ValueError:
            If config.backend is not found in CALCULATOR_REGISTRY.

    Notes:
        - All conformers from all molecules are flattened for parallel processing,
          then reorganized into the original molecule structure before returning.
        - Failed calculations return None in the output while other calculations
          continue.
        - Progress title in the progress bar is derived from the calculator name.

    Examples:
        >>> from rdkit import Chem, AllChem
        >>> mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CC")]
        >>> for mol in mols:
        ...     mol = Chem.AddHs(mol)
        ...     AllChem.EmbedMultipleConfs(mol, 2)
        >>> config = XtbSinglePointOptions()
        >>> results = calculate_property_mols(
        ...     mols, config,
        ...     n_cores=4, show_progress=True
        ... )
        >>> # results[0] contains property dicts for 2 conformers of first molecule
        >>> # results[1] contains property dicts for 2 conformers of second molecule
    """

    if config.backend not in CALCULATOR_REGISTRY:
        raise ValueError(f"Unknown property calculator: {config.backend}")

    n_confs = [mol.GetNumConformers() for mol in mols]
    _conf_ids, _mols = list(), list()
    for mol in mols:
        for conformer in mol.GetConformers():
            _conf_ids.append(conformer.GetId())
            _mols.append(mol)

    calculated_properties = run_parallel(
        calculate_property,
        list(zip(_mols, _conf_ids)),
        n_cores=n_cores,
        show_progress=show_progress,
        title=config.backend,
        func_has_ncores_arg=True,
        config=config,
        scr=scr,
        keep_files=keep_files,
        rdkit_pickle_properties=Chem.PropertyPickleOptions.AllProps,
    )

    return_properties = list()
    start_idx = 0

    # re-arrange results into promised form
    for n_conf in n_confs:
        stop_idx = start_idx + n_conf
        ret_props = list()
        for res in calculated_properties[start_idx:stop_idx]:
            ret_props.append(res)
        return_properties.append(ret_props)
        start_idx = stop_idx

    return return_properties


def extract_cosmo_output(
    calculated_properties: dict[str, Property] | None,
    remove_from_dict: bool = False,
) -> str | None:
    """Extract and optionally remove COSMO output string from calculated properties.

    Utility function to retrieve COSMO (conductor-like screening model) output
    data from a property dictionary. COSMO outputs are typically large text
    blocks containing detailed solvation calculations that are stored separately
    from scalar properties.

    Args:
        calculated_properties (dict[str, Property] | None):
            Dictionary of calculated Property objects, typically returned from
            a calculator function. Can be None, in which case None is returned.
        remove_from_dict (bool, optional):
            If True, removes the COSMO output from the dictionary after extraction
            (destructive operation). If False, leaves the dictionary unchanged.
            Defaults to False.

    Returns:
        str | None:
            COSMO output string if found in the properties dictionary, or None if:
            - calculated_properties is None
            - The COSMO_OUTPUT_KEY is not present in the dictionary

    Notes:
        - This function is useful for processing and storing COSMO data separately
          from other property data when needed for archival or memory management.
        - The COSMO_OUTPUT_KEY constant is imported from qm_atlas.constants.
        - When remove_from_dict=True, the dictionary is modified in-place.

    Examples:
        >>> from qm_atlas.tasks.calculate_properties import (
        ...     calculate_property, extract_cosmo_output
        ... )
        >>> props = calculate_property(mol, 0, "turbomole_single_point")
        >>> cosmo_str = extract_cosmo_output(props)  # Retrieve without removing
        >>> if cosmo_str:
        ...     print("COSMO output found")
        >>> # To extract and clean up the dictionary:
        >>> cosmo_str = extract_cosmo_output(props, remove_from_dict=True)
        >>> # props now no longer contains COSMO data
    """
    if calculated_properties is None:
        return None

    if remove_from_dict:
        cosmo_output = calculated_properties.pop(COSMO_OUTPUT_KEY, None)
    else:
        cosmo_output = calculated_properties.get(COSMO_OUTPUT_KEY, None)
    if cosmo_output is None:
        return None
    return cosmo_output.get_property_value()


XTB_SP_CONFIG = XtbSinglePointOptions(
    solvation_model="alpb",
    solvent="water",
    calculate_fukui=False,
    xtb_command_add=list(),
)

XTB_SP_SOLVENT_CONFIGS = [
    XtbSinglePointOptions(
        solvation_model="alpb",
        solvent="water",
        calculate_fukui=False,
        xtb_command_add=list(),
    ),
    XtbSinglePointOptions(
        solvation_model="alpb",
        solvent="woctanol",
        calculate_fukui=False,
        xtb_command_add=list(),
    ),
    XtbSinglePointOptions(
        solvation_model="alpb",
        solvent="chcl3",
        calculate_fukui=False,
        xtb_command_add=list(),
    ),
]


def get_xtb_energies_and_weights(
    molobj: Chem.Mol,
    final_xtb_sp_configs: list[XtbSinglePointOptions] | None = None,
    show_progress: bool = True,
    scr: Path = DEFAULT_SCR,
) -> list[dict]:
    """Calculate XTB solvation energies and Boltzmann weights for all conformers.

    Runs single-point XTB calculations for each solvation config on every
    conformer of *molobj*, then computes normalized Boltzmann weights per solvent.

    Args:
        molobj: Molecule with one or more conformers.
        final_xtb_sp_configs: List of XTB single-point configs (one per solvent).
            Defaults to :data:`XTB_SP_SOLVENT_CONFIGS` (water, woctanol, chcl3).
        show_progress: Show tqdm progress bar.
        scr: Scratch directory for xTB calculations.

    Returns:
        List of dicts, one per conformer. Each dict contains
        ``energy_{solvent}`` and ``weight_{solvent}`` keys for every config.
    """
    if final_xtb_sp_configs is None:
        final_xtb_sp_configs = XTB_SP_SOLVENT_CONFIGS

    rows: list[dict] = [dict() for _ in molobj.GetConformers()]

    for config in final_xtb_sp_configs:
        xtb_property_names = config.get_property_names()
        energy_key = xtb_property_names[XTB_ENERGY_KEY]
        xtb_results = calculate_property_mol(
            molobj,
            config=config,
            n_cores=1,
            scr=scr,
            show_progress=show_progress,
        )
        solvent = config.solvent if config.solvent is not None else "vacuum"

        energies: list[float] = []
        for result in xtb_results:
            if result is None or energy_key not in result:
                energy = float("nan")
            else:
                energy = result[energy_key].get_property_value()
            energies.append(energy)

        energies_arr = np.array(energies)
        weights = chembridge.get_boltzmann_weights(energies_arr)
        weights = weights / np.sum(weights)

        for row, energy, weight in zip(rows, energies_arr, weights):
            row[f"energy_{solvent}"] = float(energy)
            row[f"weight_{solvent}"] = float(weight)

    return rows
