import math
import re
from dataclasses import dataclass
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from more_itertools import batched
from ppqm.utils import functools as ppqm_functools

from qm_atlas.wrappers.cosmotherm import cosmo_utils, interface

SOLUBILITY_SETTINGS = cosmo_utils.CosmoSettings(
    action_lines=[],  # overwritten in code
    command_lines=["unit notempty ehfile wtln"],
    comment_line="!! solubility calculation !!",
)

NUM_STEPS_BINARY_MIXTURE = 11
NUM_STEPS_TERNARY_MIXTURE = 7

# global variables for parsing of multi-solvent cosmotherm output
SOLVENT_COL = "Solvent"
MOL_WEIGHT_COL = "Solvent_MolWeight"
NEW_SOLVENT_COL = "Solvent {}"
MOLAR_FRAC_COL = "Molar Fraction {}"
MASS_FRAC_COL = "Mass Fraction {}"


@dataclass
class ReferenceSolubility:
    """class to hold data about measured solubility in a solvent or solvent mixture."""

    value: float
    solvent_names: list[str]
    temperature: float
    mass_fractions: None | list[float] = None
    solubility_unit: str = "c"

    def __post_init__(self):

        if len(self.solvent_names) == 1 and self.mass_fractions is None:
            self.mass_fractions = [1.0]

        if self.mass_fractions is None:
            raise ValueError(
                "Mass fractions must be provided for reference solubilities in multiple solvents"
            )

        if len(self.solvent_names) != len(self.mass_fractions):
            raise ValueError("Number of solvents and mass fractions must match")


class SolventInputTable:
    """A class to handle the input solvents provided to cosmotherm solubility calculations.
    While the API is not handled here, it is assumed that the first compound to input
    is the API, such that the first solvent has the second position in the list of
    input structures.
    """

    def __init__(self, cosmo_dirs: tuple[Path, ...]):

        self.cosmo_dirs = cosmo_dirs

        self.name_to_cosmo_files: dict[str, list[Path]] = dict()
        self.name_to_position: dict[str, int] = dict()
        self.num_solvents = 0
        self.ready = False

    def add_solvents(self, solvent_names: list[str]):
        for solvent_name in solvent_names:
            if solvent_name in self.name_to_position:
                continue
            cosmo_files = cosmo_utils.find_cosmo_files(self.cosmo_dirs, solvent_name)
            self.name_to_cosmo_files[solvent_name] = cosmo_files
            self.name_to_position[solvent_name] = self.num_solvents + 1
            self.num_solvents += 1

    def finish_setup(self):
        if "h2o" not in self.name_to_position and "water" not in self.name_to_position:
            self.add_solvents(["h2o"])
        self.ready = True

    def get_cosmo_files(self, name: str):
        return self.name_to_cosmo_files[name]

    def get_position(self, name: str):
        return self.name_to_position[name]

    def iterate_solvents(self):
        idx_to_solvent = {idx: name for name, idx in self.name_to_position.items()}
        for idx in range(1, self.num_solvents + 1):
            name = idx_to_solvent[idx]
            yield name, self.get_cosmo_files(name)

    def get_concentration_string(
        self, solvent_names: list[str], mass_fractions: list[float]
    ) -> str:
        fractions_all_solvents = np.zeros(self.num_solvents + 1)
        for name, frac in zip(solvent_names, mass_fractions):
            idx = self.get_position(name)
            fractions_all_solvents[idx] = frac
        return "{" + " ".join([str(mass) for mass in fractions_all_solvents]) + "}"

    def get_reference_string(self, references: list[ReferenceSolubility]):

        if len(references) == 0:
            return ""

        elif len(references) == 1:
            ref = references[0]
            ref_str = f"ref_sol_{ref.solubility_unit}={ref.value} "
            c_str = self.get_concentration_string(ref.solvent_names, ref.mass_fractions)
            ref_str += f"c_ref_sol={c_str} "
            ref_str += f"t_ref_sol_c={ref.temperature} "
            return ref_str

        else:
            reference_strings = [self.get_reference_string([ref]) for ref in references]
            return " ".join(reference_strings)

    def get_solubility_action_line(
        self,
        solvent_names: list[str],
        cosmo_key: str,
        references: list[ReferenceSolubility],
        temperature: float,
        output_units: str,
    ):

        if not self.ready:
            raise RuntimeError("Must run finish_setup before getting action line")

        action_line = f"tc={temperature} solub solute=1 {cosmo_key} screening "
        solvents = " ".join([f"nx_pure={solvent_name}" for solvent_name in solvent_names])
        reference_string = self.get_reference_string(references)
        end_line = f" {reference_string} {output_units}"

        return action_line + solvents + end_line

    def get_solubility_action_line_mixture(
        self,
        solvent_names: list[str],
        cosmo_key: str,
        references: list[ReferenceSolubility],
        temperature: float,
        output_units: str,
        num_steps_binary: int = NUM_STEPS_BINARY_MIXTURE,
        num_steps_ternary: int = NUM_STEPS_TERNARY_MIXTURE,
        optimize: bool = False,
    ):

        if not self.ready:
            raise RuntimeError("Must run finish_setup before getting action line")

        action_line = f"tc={temperature} solub solute=1 {cosmo_key} screening "
        reference_string = self.get_reference_string(references)
        end_line = f" {reference_string} {output_units}"

        if optimize:
            s_string = " ".join(solvent_names)
            action_line += f"nopt={{{s_string}}} "
            solvents_line = ""

        elif len(solvent_names) == 2:
            line = np.linspace(0, 1, num_steps_binary + 1)
            grid = [[elem, 1 - elem] for elem in line]
            frac_strings = [self.get_concentration_string(solvent_names, mf) for mf in grid]
            solvents_line = "c=" + " c=".join(frac_strings)

        elif len(solvent_names) == 3:
            line = np.linspace(0, 1, num_steps_ternary + 1)
            grid = [
                [elem_1, elem_2, 1 - elem_1 - elem_2]
                for elem_1, elem_2 in product(line, line)
                if 1 - elem_1 - elem_2 >= -1.0e-10
            ]
            frac_strings = [self.get_concentration_string(solvent_names, mf) for mf in grid]
            solvents_line = "c=" + " c=".join(frac_strings)

        else:
            raise ValueError("Must use two or three solvents for mixture calculation")

        return action_line + solvents_line + end_line


def calculate_solubility(
    compound_conformers: list[str | Path],
    solvent_names: list[str],
    melting_temperature_C: float | None = None,
    melting_enthalpy_kJ_per_mol: float | None = None,
    level: str = "bp-tzvpd",
    references: list[ReferenceSolubility] | None = None,
    options: cosmo_utils.CosmoSettings = SOLUBILITY_SETTINGS,
    solvents_dirs: list[Path] | None = None,
    temperature: float = 25.0,
    output_units: str = "wsolout_c lsolout_gl",
    n_cores: int = 1,
    use_config_cosmo_dir: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    This function calculates the solubility of compounds in pure solvents
    using Cosmotherm. The method and example applications are described in reference [1].

    Args:
        compound_conformers (list[str | Path]):
            List of compound conformers, either specifying the path to the file or
            providing the contents as a string. In the latter case, the file must be
            a .cosmo file, otherwise it can be any file that cosmotherm accepts.
        solvent_names (list[str]):
            List of solvents, specified by their names. Corresponding .cosmo files must
            be found in the database, either the standard cosmotherm one or an
            additional, user-provided one.
        melting_temperature_C (float | None):
            The experimental melting temperature of the compound in degree Celsius. This
            can be used to determine the heat of fusion. Defaults to None, in this case
            no information on the melting temperature is passed to cosmotherm.
        melting_enthalpy_kJ_per_mol (float | None):
            The experimental melting enthalpy of the compound. This
            can be used to determine the heat of fusion. Defaults to None, in this case
            no information on the melting enthalpy is passed to cosmotherm.
        level (str, optional):
            Level of theory. Defaults to "bp-tzvpd".
        references (list[ReferenceSolubility], optional):
            List of reference solubilities. Defaults to None.
        options (CosmoSettings, optional):
            Settings for the Cosmo task. Defaults to :data:`SOLUBILITY_SETTINGS`.
        solvents_dirs (list[Path] | None, optional):
            Additional directories to find the .cosmo files for the solvents.
            Defaults to None, i.e. just relying on the cosmotherm database. If
            additional directories are considered, the cosmotherm database is searched
            first, the the solvents_dirs.

        temperature (float, optional):
            Temperature for the calculation in °C. Defaults to 25.
        output_units (str, optional):
            Units for the output. Defaults to "wsolout_c lsolout_gl".
            Check the cosmotherm documentation to see which strings can be used.
        n_cores (int, optional):
            Number of cores to use for the calculation. Defaults to 1.
        use_config_cosmo_dir (bool, optional):
            Whether to prepend the solvent directories from the software configuration
            file to the user-provided solvents_dirs. If False, only the user-provided
            directories are used. Defaults to True.
        **kwargs: Additional keyword arguments for the output reader.

    Returns:
        pd.DataFrame: A DataFrame containing the results of the solubility calculation.

    References:
        [1] Loschen, C. & Klamt, A. Solubility prediction, solvate and cocrystal
        screening as tools for rational crystal engineering.
        *J. Pharm. Pharmacol.* **67**, 803-811 (2015).
        https://doi.org/10.1111/jphp.12376
    """

    if (melting_temperature_C is not None) != (melting_enthalpy_kJ_per_mol is not None):
        raise ValueError("Need to provide both melting temperature and enthalpy")

    n_procs = min(n_cores, len(solvent_names))

    # split solvents in chunks
    batch_size = math.ceil(len(solvent_names) / n_procs)
    solvent_batches = list(batched(solvent_names, batch_size))
    calculator_function = partial(
        _calculate_solubility,
        compound_conformers=compound_conformers,
        melting_temperature_C=melting_temperature_C,
        melting_enthalpy_kJ_per_mol=melting_enthalpy_kJ_per_mol,
        level=level,
        references=references,
        options=options,
        solvents_dirs=solvents_dirs,
        temperature=temperature,
        output_units=output_units,
        use_config_cosmo_dir=use_config_cosmo_dir,
        **kwargs,
    )

    if len(solvent_batches) == 1:
        dataframes = [calculator_function(solvent_batches[0])]

    else:
        dataframes = ppqm_functools.func_parallel(
            calculator_function,
            solvent_batches,
            n_cores=n_procs,
            title="Cosmo_Solubility",
        )

    return pd.concat(dataframes, ignore_index=True)


def _calculate_solubility(
    solvent_names: list[str],
    compound_conformers: list[str | Path],
    melting_temperature_C: float | None = None,
    melting_enthalpy_kJ_per_mol: float | None = None,
    level: str = "bp-tzvpd",
    references: list[ReferenceSolubility] | None = None,
    options: cosmo_utils.CosmoSettings = SOLUBILITY_SETTINGS,
    solvents_dirs: list[Path] | None = None,
    temperature: float = 25.0,
    output_units: str = "wsolout_c lsolout_gl",
    use_config_cosmo_dir: bool = True,
    **kwargs,
) -> pd.DataFrame:

    if references is None:
        references = list()

    # determine mode of calculation. Will get overwritten to iterative
    # if melting enthalpy and temperature are provided
    cosmo_key = "relative" if len(references) == 0 else "iterative"

    cosmo_utils.check_input_conformers(compound_conformers)

    cosmo_paths = interface.get_cosmo_paths(level=level)
    cosmo_dirs = cosmo_utils.get_databases(
        cosmo_paths,
        level=level,
        other_databases=solvents_dirs,
        use_config_cosmo_dir=use_config_cosmo_dir,
    )

    # organize solvents
    input_table = SolventInputTable(cosmo_dirs)
    input_table.add_solvents(solvent_names)
    for reference in references:
        input_table.add_solvents(reference.solvent_names)
    input_table.finish_setup()

    conformer_names: list[str | None] = [None]
    api_prop = ""

    if melting_temperature_C is not None and melting_enthalpy_kJ_per_mol is not None:
        api_prop += f"Tmelt_C={melting_temperature_C}"
        api_prop += f" DHfus_SI={melting_enthalpy_kJ_per_mol}"
        cosmo_key = "iterative"

    additional_properties_list = [api_prop]
    conformers: list[list[str | Path]] = [compound_conformers]
    for solvent_name, cosmo_files in input_table.iterate_solvents():
        conformers.append(cosmo_files)
        conformer_names.append(solvent_name)
        additional_properties_list.append("DGfus=0")

    action_line = input_table.get_solubility_action_line(
        solvent_names, cosmo_key, references, temperature, output_units
    )

    command_lines = options.get_command_lines(cosmo_paths)
    comment_line = options.comment_line

    res_df_list = interface.calculate_cosmotherm(
        conformers,
        [action_line],
        command_lines=command_lines,
        comment_line=comment_line,
        table_start_patterns=["Nr Solvent"],
        conformer_names=conformer_names,
        additional_properties_list=additional_properties_list,
        **kwargs,
    )

    return res_df_list[0]


def calculate_solubility_mixture(
    compound_conformers: list[str | Path],
    solvent_combinations: list[list[str]],
    melting_temperature_C: float | None = None,
    melting_enthalpy_kJ_per_mol: float | None = None,
    level: str = "bp-tzvpd",
    references: list[ReferenceSolubility] | None = None,
    options: cosmo_utils.CosmoSettings = SOLUBILITY_SETTINGS,
    solvents_dirs: list[Path] | None = None,
    temperature: float = 25.0,
    output_units: str = "wsolout_c lsolout_gl",
    num_steps_binary: int = NUM_STEPS_BINARY_MIXTURE,
    num_steps_ternary: int = NUM_STEPS_TERNARY_MIXTURE,
    optimize: bool = False,
    n_cores: int = 1,
    use_config_cosmo_dir: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    This function calculates the solubility of compounds in binary or ternary mixtures
    of solvents using Cosmotherm. For each solvent combination, it goes through
    equally spaced mixture ratios (in terms of mass fractions). The number of
    steps for this spacing can be controlled via the arguments
    num_steps_binary and num_steps ternary.
    The method and example applications are described in reference [1].

    Args:
        compound_conformers (list[str | Path]):
            List of compound conformers, either specifying the path to the file or
            providing the contents as a string. In the latter case, the file must be
            a .cosmo file, otherwise it can be any file that cosmotherm accepts.
        solvent_combinations (list[list[str]]):
            List of solvent combinations, each combination is a list of solvents,
            specified by their names. Corresponding .cosmo files must be found
            in the database, either the standard cosmotherm one or an additional,
            user-provided one.
        melting_temperature_C (float | None):
            The experimental melting temperature of the compound in degree Celsius. This
            can be used to determine the heat of fusion. Defaults to None, in this case
            no information on the melting temperature is passed to cosmotherm.
        melting_enthalpy_kJ_per_mol (float | None):
            The experimental melting enthalpy of the compound. This
            can be used to determine the heat of fusion. Defaults to None, in this case
            no information on the melting enthalpy is passed to cosmotherm.
        level (str, optional):
            Level of theory. Defaults to "bp-tzvpd".
        references (list[ReferenceSolubility], optional):
            List of reference solubilities. Defaults to None.
        options (CosmoSettings, optional):
            Settings for the Cosmo task. Defaults to :data:`SOLUBILITY_SETTINGS`.
        solvents_dirs (list[Path] | None, optional):
            Additional directories to find the .cosmo files for the solvents.
            Defaults to None, i.e. just relying on the cosmotherm database. If
            additional directories are considered, the cosmotherm database is searched
            first, the the solvents_dirs.

        temperature (float, optional):
            Temperature for the calculation in °C. Defaults to 25.
        output_units (str, optional):
            Units for the output. Defaults to "wsolout_c lsolout_gl".
            Check the cosmotherm documentation to see which strings can be used.
        num_steps_binary (int, optional):
            Number of steps for binary mixtures. Defaults to :data:`NUM_STEPS_BINARY_MIXTURE`.
        num_steps_ternary (int, optional):
            Number of steps for ternary mixtures. Defaults to :data:`NUM_STEPS_TERNARY_MIXTURE`.
        optimize (bool):
            Whether to ask cosmotherm to ptimize the mixture ratios. Note that in this
            case no mixture ratios will be screened. Defaults to False.
        n_cores (int, optional):
            Number of cores to use for the calculation. Defaults to 1.
        use_config_cosmo_dir (bool, optional):
            Whether to prepend the solvent directories from the software configuration
            file to the user-provided solvents_dirs. If False, only the user-provided
            directories are used. Defaults to True.
        **kwargs: Additional keyword arguments for the output reader.

    Returns:
        pd.DataFrame: A DataFrame containing the results of the solubility calculation.

    References:
        [1] Loschen, C. & Klamt, A. Solubility prediction, solvate and cocrystal
        screening as tools for rational crystal engineering.
        *J. Pharm. Pharmacol.* **67**, 803-811 (2015).
        https://doi.org/10.1111/jphp.12376
    """

    if (melting_temperature_C is not None) != (melting_enthalpy_kJ_per_mol is not None):
        raise ValueError("Need to provide both melting temperature and enthalpy")

    n_procs = min(n_cores, len(solvent_combinations))

    calculator_function = partial(
        _calculate_solubility_mixture,
        compound_conformers=compound_conformers,
        melting_temperature_C=melting_temperature_C,
        melting_enthalpy_kJ_per_mol=melting_enthalpy_kJ_per_mol,
        level=level,
        references=references,
        options=options,
        solvents_dirs=solvents_dirs,
        temperature=temperature,
        output_units=output_units,
        num_steps_binary=num_steps_binary,
        num_steps_ternary=num_steps_ternary,
        optimize=optimize,
        use_config_cosmo_dir=use_config_cosmo_dir,
        **kwargs,
    )

    if n_procs == 1:
        dataframes = [
            calculator_function(solvents, n_cores=n_cores) for solvents in solvent_combinations
        ]

    else:
        dataframes = ppqm_functools.func_parallel(
            calculator_function,
            solvent_combinations,
            n_cores=n_procs,
            title="Cosmo_Solubility",
        )

    return pd.concat(dataframes, ignore_index=True)


def _calculate_solubility_mixture(
    solvent_names: list[str],
    compound_conformers: list[str | Path],
    melting_temperature_C: float | None = None,
    melting_enthalpy_kJ_per_mol: float | None = None,
    level: str = "bp-tzvpd",
    references: list[ReferenceSolubility] | None = None,
    options: cosmo_utils.CosmoSettings = SOLUBILITY_SETTINGS,
    solvents_dirs: list[Path] | None = None,
    temperature: float = 25.0,
    output_units: str = "wsolout_c lsolout_gl",
    num_steps_binary: int = NUM_STEPS_BINARY_MIXTURE,
    num_steps_ternary: int = NUM_STEPS_TERNARY_MIXTURE,
    optimize: bool = False,
    use_config_cosmo_dir: bool = True,
    **kwargs,
) -> pd.DataFrame:

    if references is None:
        references = list()

    if len(solvent_names) not in [2, 3]:
        raise ValueError("Mixtures must have two or three components")

    # determine mode of calculation. Will get overwritten to iterative
    # if melting enthalpy and temperature are provided
    cosmo_key = "relative" if len(references) == 0 else "iterative"

    cosmo_utils.check_input_conformers(compound_conformers)
    cosmo_paths = interface.get_cosmo_paths(level=level)
    cosmo_dirs = cosmo_utils.get_databases(
        cosmo_paths,
        level=level,
        other_databases=solvents_dirs,
        use_config_cosmo_dir=use_config_cosmo_dir,
    )

    # organize solvents
    input_table = SolventInputTable(cosmo_dirs)
    input_table.add_solvents(solvent_names)
    for reference in references:
        input_table.add_solvents(reference.solvent_names)
    input_table.finish_setup()

    conformer_names = [None]
    api_prop = ""

    if melting_temperature_C is not None and melting_enthalpy_kJ_per_mol is not None:
        api_prop += f"Tmelt_C={melting_temperature_C}"
        api_prop += f" DHfus_SI={melting_enthalpy_kJ_per_mol}"
        cosmo_key = "iterative"

    additional_properties_list = [api_prop]
    conformers = [compound_conformers]
    for solvent_name, cosmo_files in input_table.iterate_solvents():
        conformers.append(cosmo_files)
        conformer_names.append(solvent_name)
        additional_properties_list.append("DGfus=0")

    action_line = input_table.get_solubility_action_line_mixture(
        solvent_names,
        cosmo_key,
        references,
        temperature,
        output_units,
        num_steps_binary=num_steps_binary,
        num_steps_ternary=num_steps_ternary,
        optimize=optimize,
    )

    command_lines = options.get_command_lines(cosmo_paths)
    comment_line = options.comment_line

    res_df_list = interface.calculate_cosmotherm(
        conformers,
        [action_line],
        command_lines=command_lines,
        comment_line=comment_line,
        conformer_names=conformer_names,
        additional_properties_list=additional_properties_list,
        table_start_patterns=["Nr Solvent"],
        multi_solvent=True,
        **kwargs,
    )

    return res_df_list[0]


def extract_molar_weights(sol_df: pd.DataFrame) -> dict[str, float]:
    """Extracts molar weights of pure solvents from the output dataframe of a
        Cosmotherm solubility calculation for multiple solvents. The molar weights
        are read from those rows where a pure solvent is used.

    Args:
        sol_df (pd.DataFrame): The output dataframe of a solubility calculation

    Returns:
        dict[str, float]: Mapping solvent names to their molar weights
    """
    pure_solvents = ~sol_df[SOLVENT_COL].str.contains("X=")
    help_df = sol_df.loc[pure_solvents, [SOLVENT_COL, MOL_WEIGHT_COL]].drop_duplicates()
    help_df.set_index(SOLVENT_COL, inplace=True)
    return help_df[MOL_WEIGHT_COL].to_dict()


def parse_solvent_row(row_entry: str) -> tuple[list[str], list[float]]:
    """Parses a row entry from a Cosmotherm solubility calculation output.
    The row entry has the form
    "solvent_1 solvent_2 X ={ frac_1 frac_2 }"
    for a binary mixture, but ternatry mixtures or pure solvents are handled as
    well.

    Args:
        row_entry (str): A single solvent row from the cosmotherm solubility output.

    Returns:
        list[str]: A list of solvent names
        list[float]: The corresponding molar fractions
    """

    data_list = re.findall(r"[\w-]+", row_entry)
    molar_fractions = re.findall(r"\d+\.\d+", row_entry)

    # If it's a pure solvent
    if not molar_fractions:
        molar_fractions = [1.0]

    else:
        molar_fractions = list(map(float, molar_fractions))

    return data_list[: len(molar_fractions)], molar_fractions


def calculate_mass_fractions(
    molar_fractions: list[float],
    molar_weights: list[float],
) -> list[float]:
    """Converts molar fractions to mass fractions given the molar weights of the components.

    Args:
        molar_fractions (list[float]): Molar fractions of the components
        molar_weights (list[float]): Molar weights of the components

    Returns:
        list[float]: Mass fractions of the components
    """
    total_mass = sum(mf * mw for mf, mw in zip(molar_fractions, molar_weights))
    return [mf * mw / total_mass for mf, mw in zip(molar_fractions, molar_weights)]


def parse_solubility_data(
    sol_df: pd.DataFrame,
    num_solvents: int = 2,
) -> pd.DataFrame:
    """Parses the output dataframe of a cosmotherm solubility calculation to obtain
        a more readable version for binary or ternary mixtures.
        The cosmotherm solvent row has the form
        "solvent_1 solvent_2 X ={ frac_1 frac_2 }"
        this is converted to two (or more) additional columns containing the names,
        molar fractions and mass fractions of the mixtures.

    Args:
        sol_df (pd.DataFrame):
            The output dataframe of a solubility calculation
        num_solvents (int, optional):
            The maximum number of pure solvents in the mixtures. Defaults to 2.

    Raises:
        ValueError: If there are more solvents in a mixture than the specified maximum.

    Returns:
        pd.DataFrame: The parsed dataframe
    """
    # Initialize new columns
    parsed_df = sol_df.copy()

    new_col_names = [NEW_SOLVENT_COL, MOLAR_FRAC_COL, MASS_FRAC_COL]
    for col_name, i in product(new_col_names, range(1, num_solvents + 1)):
        parsed_df[col_name.format(i)] = None

    # Create a dictionary to store the molar weights of the solvents
    solvent_molar_weights = extract_molar_weights(sol_df)

    # Calculate molar and mass fractions
    for index, row in sol_df.iterrows():
        row_entry = row[SOLVENT_COL]
        solvents, molar_fractions = parse_solvent_row(row_entry)
        if len(solvents) > num_solvents:
            raise ValueError(f"Line {index} contains more than {num_solvents} solvents")

        molar_weights = [solvent_molar_weights[solvent_name] for solvent_name in solvents]
        mass_fractions = calculate_mass_fractions(molar_fractions, molar_weights)

        for idx, (solvent_name, molar_fraction, mass_fraction) in enumerate(
            zip(solvents, molar_fractions, mass_fractions), start=1
        ):
            parsed_df.at[index, NEW_SOLVENT_COL.format(idx)] = solvent_name
            parsed_df.at[index, MOLAR_FRAC_COL.format(idx)] = molar_fraction
            parsed_df.at[index, MASS_FRAC_COL.format(idx)] = mass_fraction

    return parsed_df
