import logging
from pathlib import Path

import pandas as pd

from qm_atlas.tasks import common as calculated_properties
from qm_atlas.wrappers.cosmotherm import cosmo_utils, interface

_logger = logging.getLogger(__name__)


LOGP_WATER_OCTANOL_SETTINGS = cosmo_utils.CosmoSettings(
    action_lines=[
        "LogP xl1={{ 1 0 0 }} xl2={{ 0.27272727 0.72727273 0 }} tc={temperature_Celsius} vq=0.1505"
    ],
    command_lines=["unit notempty ehfile wtln dipoleref=cop"],
    comment_line="!! logP calculation !!",
)

LOGP_GENERIC_SETTINGS = cosmo_utils.CosmoSettings(
    action_lines=["LogP={{1 2}} tc={temperature_Celsius}"],
    command_lines=["unit notempty ehfile wtln dipoleref=cop"],
    comment_line="!! logP calculation !!",
)

DG_SETTINGS = cosmo_utils.CosmoSettings(
    action_lines=["henry=1 tc={temperature_Celsius} GSOLV"],
    command_lines=["unit notempty ehfile wtln"],
    comment_line="!! dG solv calculation !!",
)

DESCRIPTORS_SETTINGS = cosmo_utils.CosmoSettings(
    action_lines=["x_pure=1 tc={temperature_Celsius} ctab wconf"],
    command_lines=["unit notempty ehfile wtln pri2 pri1 dipoleref=cop"],
    comment_line="!! mixture calculation !!",
)

DESCRIPTORS_SETTINGS_SIMPLIFIED = cosmo_utils.CosmoSettings(
    action_lines=["x_pure=1 tc={temperature_Celsius} ctab wconf"],
    command_lines=["unit notempty ehfile wtln dipoleref=cop"],
    comment_line="!! mixture calculation !!",
)

PKA_SETTINGS = cosmo_utils.CosmoSettings(
    action_lines=["tc={temperature_Celsius} pka={{1 2 3}} {solvent}-{form}"],
    command_lines=["unit notempty ehfile wtln dipoleref=cop"],
    comment_line="!! pka calculation !!",
)

COSMO_PERM_SETTINGS = cosmo_utils.CosmoSettings(
    action_lines=["tc={temperature_Celsius} MICELLE PERMEABILITY centersig2 pH={pH}"],
    command_lines=["RMIC={micelle_file}"],
    comment_line="!! COSMOperm calculation !!",
)

PKA_SOLVENTS = {
    "h2o": "WATER",
    "dimethylsulfoxide": "DMSO",
    "acetonitrile": "ACETONITRILE",
    "n-heptane": "HEPTANE",
    "thf": "THF",
}

COSMO_LOGP_KEY = "cosmo_logp"
COSMO_DELTA_G_KEY = "cosmo_delta_g"
COSMO_PKA_KEY = "cosmo_pka"
COSMO_PERM_KEY = "cosmo_perm"


COSMO_DESCRIPTORS_COLUMNS = [
    "mu",
    "log10(p)",
    "E_COSMO+dE+Mu",
    "H_int",
    "H_MF",
    "H_HB",
    "H_vdW",
    "H_glt",
    "E_Ring",
    "comp",
    "conf",
    "w(liq)",
    "w(gas)",
    "E_COSMO+dE",
    "E_gas",
    "E_COS+dE-E_gas",
    "E_diel",
    "dE",
    "Area",
    "Volume",
    "MolWeight",
    "Charge",
    "Dipol(t)",
    "Moment_HBacc",
    "Moment_HBdon",
    "Moment_sig2",
    "Moment_sig3",
    "Moment_sig4",
    "Moment_sig5",
    "Moment_sig6",
    "Polmom_norm",
    "charge_p",
    "charge_n",
    "area_p",
    "area_n",
    "Gdisp3_cosmo",
    "Gdisp3_gas",
]


def calculate_logp(
    compound_conformers: list[str | Path],
    level: str = "bp-tzvpd",
    is_woctanol: bool = True,
    options: cosmo_utils.CosmoSettings | None = None,
    solvent_names: tuple[str, str] = ("h2o", "1-octanol"),
    temperature_Celsius: float = 25.0,
    solvents_dirs: list[Path] | None = None,
    eq_phases: bool = False,
    use_config_cosmo_dir: bool = True,
    **kwargs,
) -> dict[str, calculated_properties.ScalarProperty]:
    """
    This function calculates the logP of a compound using Cosmotherm.

    Args:
        compound_conformers (list[str | Path]):
            List of compound conformers, either specifying the path to the file or
            providing the contents as a string. In the latter case, the file must be
            a .cosmo file, otherwise it can be any file that cosmotherm accepts.
        level (str, optional):
            Level of theory. Defaults to "bp-tzvpd". Other allowed options are
            "bp-tzvp", "bp-svp" and "dmol3-pbe".
        is_woctanol (bool, optional):
            Whether the logP calculation is for water-octanol. Defaults to True. The value has
            an effect on the settings for the calculation. If it is set True and the options
            argument is not provided, the default settings for water-octanol logP calculation
            are used. These take into account that there is some octanol in the water and vice-versa.
        options (cosmo_utils.CosmoSettings, optional):
            Settings for the Cosmo task. Defaults to None, in which case default options are used,
            these are different for water-octanol and any other mixture.
        solvent_names (list[str], optional):
            List of two solvents, specified by their names. Corresponding .cosmo files
            must be found in the database, either the standard cosmotherm one or an
            additional, user-provided one. Defaults to ["h2o", "1-octanol"].
        temperature_Celsius (float, optional):
            Temperature (in Celsius) at which the calculation is run. Defaults to 25.0.
        solvents_dirs (list[Path], optional):
            Additional directories to find the .cosmo files for the solvents.
            Defaults to None, i.e. just relying on the cosmotherm database.
        eq_phases (bool):
            Cosmotherm option for logp calculations: Compute the phase equilibrium
            between the given two solvents. For water-octanol, the default options have
            an experimentally determined phase equilibrium, so this option is not needed.
            Default to False.
        use_config_cosmo_dir (bool, optional):
            Whether to prepend the solvent directories from the software configuration
            file to the user-provided solvents_dirs. If False, only the user-provided
            directories are used. Defaults to True.
        **kwargs: Additional keyword arguments for the output reader.

    Returns:
        dict[str, calculated_properties.ScalarProperty]:
            A dictionary mapping a single key to a scalar property.
            The key is "cosmo_logp" and the value is the calculated logP value of the compound.
    """

    if options is None:
        if is_woctanol:
            options = LOGP_WATER_OCTANOL_SETTINGS
        else:
            options = LOGP_GENERIC_SETTINGS

    if not len(solvent_names) == 2:
        raise ValueError("LogP calculation must be done using two solvents.")

    cosmo_utils.check_input_conformers(compound_conformers)

    cosmo_paths = interface.get_cosmo_paths(level=level)

    solvents_databases = cosmo_utils.get_databases(
        cosmo_paths,
        level=level,
        other_databases=solvents_dirs,
        use_config_cosmo_dir=use_config_cosmo_dir,
    )

    solvents: list[list[Path]] = [
        cosmo_utils.find_cosmo_files(
            solvents_databases,
            solvent_names[0],
            fail_on_error=True,
        )
    ]
    solvents.append(cosmo_utils.find_cosmo_files(solvents_databases, solvent_names[1]))

    conformers = solvents + [compound_conformers]

    action_lines = options.action_lines
    if action_lines is None:
        raise ValueError("options.action_lines must not be None")
    if eq_phases:
        if len(action_lines) != 1:
            raise ValueError("eq_phases can only be used with one action line.")
        action_lines = [action_lines[0] + " eq_phases"]

    formatted_action_lines = [
        line.format(temperature_Celsius=temperature_Celsius) for line in action_lines
    ]
    command_lines = options.get_command_lines(cosmo_paths)
    comment_line = options.comment_line

    res_df_list = interface.calculate_cosmotherm(
        conformers,
        formatted_action_lines,
        command_lines=command_lines,
        comment_line=comment_line,
        **kwargs,
    )

    res_df = res_df_list[0]
    log_p = res_df.loc[res_df.index[-1], "log10(P)"]  # type: ignore

    return {COSMO_LOGP_KEY: calculated_properties.ScalarProperty(float(log_p))}


def calculate_delta_g(
    compound_conformers: list[str | Path],
    level: str = "bp-tzvpd",
    options: cosmo_utils.CosmoSettings = DG_SETTINGS,
    solvent_name: str = "h2o",
    temperature_Celsius: float = 25.0,
    solvents_dirs: list[Path] | None = None,
    use_config_cosmo_dir: bool = True,
    **kwargs,
) -> dict[str, calculated_properties.ScalarProperty]:
    """
    This function calculates the free energy of solvation (delta G)
    of a compound using Cosmotherm.

    Args:
        compound_conformers (list[str | Path]):
            List of compound conformers, either specifying the path to the file or
            providing the contents as a string. In the latter case, the file must be
            a .cosmo file, otherwise it can be any file that cosmotherm accepts.
        level (str, optional):
            Level of theory. Defaults to "bp-tzvpd". Other allowed options are
            "bp-tzvp", "bp-svp" and "dmol3-pbe".
        options (cosmo_utils.CosmoSettings, optional):
            Settings for the Cosmo task. Defaults to :data:`DG_SETTINGS`.
        solvent_name (str, optional):
            The solvent, specified by its name. Corresponding .cosmo file must
            be found in the database, either the standard cosmotherm one or an
            additional, user-provided one. There is an option to consider the
            case of the compound dissolved in itself, i.e. self-solvation.
            In this case, the solvent name should be set to "self". The default
            is "h2o".
        temperature_Celsius (float, optional):
            Temperature (in Celsius) at which the calculation is run. Defaults to 25.0.
        solvents_dirs (list[Path], optional):
            Additional directories to find the .cosmo file for the solvent.
            Defaults to None, i.e. just relying on the cosmotherm database.
        use_config_cosmo_dir (bool, optional):
            Whether to prepend the solvent directories from the software configuration
            file to the user-provided solvents_dirs. If False, only the user-provided
            directories are used. Defaults to True.
        **kwargs: Additional keyword arguments for the output reader.

    Returns:
        dict[str, calculated_properties.ScalarProperty]:
            A dictionary mapping a single key to a scalar property.
            The key is "cosmo_delta_g" and the value is the calculated free energy
            of solvation (delta G) of the compound in kcal/mol.
    """
    cosmo_utils.check_input_conformers(compound_conformers)

    cosmo_paths = interface.get_cosmo_paths(level=level)

    solvents_databases = cosmo_utils.get_databases(
        cosmo_paths,
        level=level,
        other_databases=solvents_dirs,
        use_config_cosmo_dir=use_config_cosmo_dir,
    )

    if solvent_name == "self":
        # self-solvation
        conformers = [compound_conformers]
    else:
        solvent = cosmo_utils.find_cosmo_files(
            solvents_databases,
            solvent_name,
            fail_on_error=True,
        )

        conformers = [solvent] + [compound_conformers]

    action_lines = options.action_lines
    if action_lines is None:
        raise ValueError("options.action_lines must not be None")
    formatted_action_lines = [
        line.format(temperature_Celsius=temperature_Celsius) for line in action_lines
    ]
    command_lines = options.get_command_lines(cosmo_paths)
    comment_line = options.comment_line

    res_df_list = interface.calculate_cosmotherm(
        conformers,
        formatted_action_lines,
        command_lines=command_lines,
        comment_line=comment_line,
        **kwargs,
    )

    res_df = res_df_list[0]
    delta_g = res_df.loc[res_df.index[-1], "Gsolv"]  # type: ignore

    return {COSMO_DELTA_G_KEY: calculated_properties.ScalarProperty(float(delta_g))}


def calculate_descriptors_df(
    compound_conformers: list[str | Path],
    level: str = "bp-tzvpd",
    options: cosmo_utils.CosmoSettings = DESCRIPTORS_SETTINGS,
    solvent_name: str = "h2o",
    temperature_Celsius: float = 25.0,
    solvents_dirs: list[Path] | None = None,
    use_config_cosmo_dir: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    This function calculates the Cosmotherm descriptors of a compound.

    Args:
        compound_conformers (list[str | Path]):
            List of compound conformers, either specifying the path to the file or
            providing the contents as a string. In the latter case, the file must be
            a .cosmo file, otherwise it can be any file that cosmotherm accepts.
        level (str, optional):
            Level of theory. Defaults to "bp-tzvpd". Other allowed options are
            "bp-tzvp", "bp-svp" and "dmol3-pbe".
        options (cosmo_utils.CosmoSettings, optional):
            Settings for the Cosmo task. Defaults to :data:`DESCRIPTORS_SETTINGS`.
        solvent_name (str, optional):
            The solvent, specified by its name. Corresponding .cosmo file must
            be found in the database, either the standard cosmotherm one or an
            additional, user-provided one. Defaults to "h2o".
        temperature_Celsius (float, optional):
            Temperature (in Celsius) at which the calculation is run. Defaults to 25.0.
        solvents_dirs (list[Path], optional):
            Additional directories to find the .cosmo file for the solvent.
            Defaults to None, i.e. just relying on the cosmotherm database.
        use_config_cosmo_dir (bool, optional):
            Whether to prepend the solvent directories from the software configuration file
            to the user-provided solvents_dirs. If False, only the user-provided directories
            are used. Defaults to True.
        **kwargs: Additional keyword arguments for the output reader.

    Returns:
        pd.DataFrame: A DataFrame containing the calculated descriptors of the compound.
    """
    cosmo_utils.check_input_conformers(compound_conformers)

    cosmo_paths = interface.get_cosmo_paths(level=level)

    solvents_databases = cosmo_utils.get_databases(
        cosmo_paths,
        level=level,
        other_databases=solvents_dirs,
        use_config_cosmo_dir=use_config_cosmo_dir,
    )

    solvent = cosmo_utils.find_cosmo_files(
        solvents_databases,
        solvent_name,
        fail_on_error=True,
    )

    conformers = [solvent] + [compound_conformers]

    action_lines = options.action_lines
    if action_lines is None:
        raise ValueError("options.action_lines must not be None")
    formatted_action_lines = [
        line.format(temperature_Celsius=temperature_Celsius) for line in action_lines
    ]
    command_lines = options.get_command_lines(cosmo_paths)
    comment_line = options.comment_line

    res_df_list = interface.calculate_cosmotherm(
        conformers,
        formatted_action_lines,
        command_lines=command_lines,
        comment_line=comment_line,
        table_start_patterns=["Nr Molecule/Conformer"],
        **kwargs,
    )

    desc_df = res_df_list[0]
    desc_df.drop(index=range(len(solvent)), inplace=True)
    desc_df.reset_index(inplace=True, drop=True)
    return desc_df


def calculate_descriptors(
    compound_conformers: list[str | Path],
    level: str = "bp-tzvpd",
    options: cosmo_utils.CosmoSettings = DESCRIPTORS_SETTINGS,
    solvent_name: str = "h2o",
    temperature_Celsius: float = 25.0,
    solvents_dirs: list[Path] | None = None,
    use_config_cosmo_dir: bool = True,
    **kwargs,
) -> list[dict[str, calculated_properties.ScalarProperty]]:
    """
    This function calculates the Cosmotherm descriptors of a compound.

    Args:
        compound_conformers (list[str | Path]):
            List of compound conformers, either specifying the path to the file or
            providing the contents as a string. In the latter case, the file must be
            a .cosmo file, otherwise it can be any file that cosmotherm accepts.
        level (str, optional):
            Level of theory. Defaults to "bp-tzvpd". Other allowed options are
            "bp-tzvp", "bp-svp" and "dmol3-pbe".
        options (cosmo_utils.CosmoSettings, optional):
            Settings for the Cosmo task. Defaults to :data:`DESCRIPTORS_SETTINGS`.
        solvent_name (str, optional):
            The solvent, specified by its name. Corresponding .cosmo file must
            be found in the database, either the standard cosmotherm one or an
            additional, user-provided one. Defaults to "h2o".
        temperature_Celsius (float, optional):
            Temperature (in Celsius) at which the calculation is run. Defaults to 25.0.
        solvents_dirs (list[Path], optional):
            Additional directories to find the .cosmo file for the solvent.
            Defaults to None, i.e. just relying on the cosmotherm database.
        use_config_cosmo_dir (bool, optional):
            Whether to prepend the solvent directories from the software configuration
            file to the user-provided solvents_dirs. If False, only the user-provided
            directories are used. Defaults to True.
        **kwargs: Additional keyword arguments for the output reader.

    Returns:
        list[dict[str, calculated_properties.ScalarProperty]]:
        A list of dictionaries, each containing the calculated descriptors of a compound conformer.
        The list follows the order of the input compound conformers.
        Each dictionary maps descriptor names to their corresponding scalar property values.
    """
    desc_df = calculate_descriptors_df(
        compound_conformers=compound_conformers,
        level=level,
        options=options,
        solvent_name=solvent_name,
        temperature_Celsius=temperature_Celsius,
        solvents_dirs=solvents_dirs,
        use_config_cosmo_dir=use_config_cosmo_dir,
        **kwargs,
    )
    # Only select columns that actually exist in the dataframe
    available_columns = [col for col in COSMO_DESCRIPTORS_COLUMNS if col in desc_df.columns]
    rel_df = desc_df.loc[:, available_columns].copy()

    result_list = []
    for _, row in rel_df.iterrows():
        conf_dict = {}
        for col in available_columns:
            try:
                conf_dict[col] = calculated_properties.ScalarProperty(float(row[col]))
            except ValueError:
                conf_dict[col] = calculated_properties.ScalarProperty(None)

        result_list.append(conf_dict)

    return result_list


def calculate_pka(
    parent_conformers: list[str | Path],
    child_conformers: list[str | Path],
    form: str,
    level: str = "bp-tzvpd",
    options: cosmo_utils.CosmoSettings = PKA_SETTINGS,
    solvent_name: str = "h2o",
    temperature_Celsius: float = 25.0,
    solvents_dirs: list[Path] | None = None,
    use_config_cosmo_dir: bool = True,
    **kwargs,
) -> float:
    """
    This function calculates the pK of a compound using Cosmotherm.

    Args:
        parent_conformers (list[str | Path]):
            List of parent compound conformers, either specifying the path to the file or
            providing the contents as a string. In the latter case, the file must be
            a .cosmo file, otherwise it can be any file that cosmotherm accepts.
        child_conformers (list[str | Path]):
            List of child compound conformers, either specifying the path to the file or
            providing the contents as a string. In the latter case, the file must be
            a .cosmo file, otherwise it can be any file that cosmotherm accepts.
        form (str):
            The form used for the calculation, either "ACID" or "BASE". Note that
            if the form is "ACID", the child conformers should have charge difference -1
            to the parent conformers, and if the form is "BASE", the child conformers
            should have charge difference +1.
        level (str, optional):
            Level of theory. Defaults to "bp-tzvpd". Other allowed options are
            "bp-tzvp", "bp-svp" and "dmol3-pbe".
        options (cosmo_utils.CosmoSettings, optional):
            Settings for the Cosmo task. Defaults to PK_SETTINGS.
        solvent_name (str, optional):
            The solvent, specified by its name. Corresponding .cosmo file must
            be found in the database, either the standard cosmotherm one or an
            additional, user-provided one. Defaults to "h2o".
        temperature_Celsius (float, optional):
            Temperature (in Celsius) at which the calculation is run. Defaults to 25.0.
        solvents_dirs (list[Path], optional):
            Additional directories to find the .cosmo file for the solvent.
            Defaults to None, i.e. just relying on the cosmotherm database.
        use_config_cosmo_dir (bool, optional):
            Whether to prepend the solvent directories from the software configuration
            file to the user-provided solvents_dirs. If False, only the user-provided
            directories are used. Defaults to True.
        **kwargs: Additional keyword arguments for the output reader.

    Raises:
        ValueError:
            If the form is not "ACID" or "BASE" or if the solvent is not available.
            For pKa calculations, only selected solvents are available.

    Returns:
        dict[str, calculated_properties.ScalarProperty]: The calculated pK value of the compound.
    """
    cosmo_utils.check_input_conformers(parent_conformers)
    cosmo_utils.check_input_conformers(child_conformers)

    if form not in ("ACID", "BASE"):
        _logger.error(f"Unknown form: {form}. Must be ACID or BASE.")
        raise ValueError(f"Unknown form: {form}. Must be ACID or BASE.")

    solvent_names_avail = PKA_SOLVENTS.keys()
    if solvent_name not in solvent_names_avail:
        _logger.error(f"Solvent {solvent_name} not available for pK calculation.")
        _logger.error(f"Available solvents are: {solvent_names_avail}")
        raise ValueError(f"Solvent {solvent_name} not available for pK calculation.")

    cosmo_paths = interface.get_cosmo_paths(level=level)

    solvents_databases = cosmo_utils.get_databases(
        cosmo_paths,
        level=level,
        other_databases=solvents_dirs,
        use_config_cosmo_dir=use_config_cosmo_dir,
    )

    solvent = cosmo_utils.find_cosmo_files(
        solvents_databases,
        solvent_name,
        fail_on_error=True,
    )

    conformers = [solvent] + [parent_conformers] + [child_conformers]

    action_lines = options.action_lines
    if action_lines is None:
        raise ValueError("options.action_lines must not be None")
    command_lines = options.get_command_lines(cosmo_paths)
    comment_line = options.comment_line

    # format action_lines
    pka_solvent = PKA_SOLVENTS[solvent_name]
    formatted_action_lines = [
        line.format(temperature_Celsius=temperature_Celsius, solvent=pka_solvent, form=form)
        for line in action_lines
    ]

    res_df_list = interface.calculate_cosmotherm(
        conformers,
        formatted_action_lines,
        command_lines=command_lines,
        comment_line=comment_line,
        **kwargs,
    )

    pka_df = res_df_list[0]

    return float(pka_df.loc[0, "pKa"])


def calculate_cosmoperm(
    compound_conformers: list[str | Path],
    level: str = "bp-tzvpd",
    micelle_file: Path | None = None,
    micelle_name: str = "dmpc",
    temperature_Celsius: float = 37.0,
    pH: float = 7.4,
    options: cosmo_utils.CosmoSettings = COSMO_PERM_SETTINGS,
    solvents_dirs: list[Path] | None = None,
    use_config_cosmo_dir: bool = True,
    **kwargs,
) -> dict[str, calculated_properties.ScalarProperty]:
    """
    This function calculates the permeability of a compound using COSMOperm.

    Args:
        compound_conformers (list[str | Path]):
            List of compound conformers, either specifying the path to the file or
            providing the contents as a string. In the latter case, the file must be
            a .cosmo file, otherwise it can be any file that cosmotherm accepts.
        level (str, optional):
            Level of theory. Defaults to "bp-tzvpd". Other allowed options are
            "bp-tzvp", "bp-svp" and "dmol3-pbe".
        micelle_file (Path | None, optional):
            Path to a custom micelle file (.mic). If provided, this overrides
            micelle_name. Defaults to None (use micelle_name instead).
        micelle_name (str, optional):
            Name of the micelle to use for the calculation (e.g., 'dmpc').
            The system will look for 'COSMOmic-{micelle_name}.mic' in the database.
            Only used if micelle_file is None. Defaults to "dmpc". Other micelle
            names preloaded in COSMOtherm are: "dppc", "popc", "SDS".
        temperature_Celsius (float, optional):
            Temperature in Celsius for the calculation. Defaults to 37.0.
        pH (float, optional):
            pH value for the calculation. Defaults to 7.4.
        options (cosmo_utils.CosmoSettings, optional):
            Settings for the COSMOperm task. Defaults to :data:`COSMO_PERM_SETTINGS`.
        solvents_dirs (list[Path], optional):
            Additional directories to find the .cosmo files (currently unused for COSMOperm).
            Defaults to None.
        use_config_cosmo_dir (bool, optional):
            Whether to use the cosmo directory from the software configuration. Defaults to True.
        **kwargs: Additional keyword arguments for the output reader.

    Returns:
        dict[str, calculated_properties.ScalarProperty]: The calculated logPerm (cm/s) value of the compound.

    Raises:
        ValueError: If the micelle file is not found in the database.
    """
    del solvents_dirs  # unused but API-required (tasks/cosmo_properties.py)
    del use_config_cosmo_dir  # unused but API-required (tasks/cosmo_properties.py)

    cosmo_utils.check_input_conformers(compound_conformers)

    cosmo_paths = interface.get_cosmo_paths(level=level)

    # Find the .mic file in the database
    if micelle_file is None:
        micelle_file = cosmo_paths.database_dir / "c" / f"COSMOmic-{micelle_name}.mic"

    if not micelle_file.exists():
        _logger.error(f"Micelle file not found: {micelle_file}")
        raise ValueError(f"Micelle file not found: {micelle_file}")

    conformers = [compound_conformers]

    action_lines = options.action_lines
    if action_lines is None:
        raise ValueError("COSMO_PERM_SETTINGS.action_lines must not be None")

    formatted_action_lines = [
        line.format(temperature_Celsius=temperature_Celsius, pH=pH) for line in action_lines
    ]

    command_lines = options.get_command_lines(cosmo_paths)

    formated_command_lines = [line.format(micelle_file=micelle_file) for line in command_lines]

    comment_line = options.comment_line

    res_df_list = interface.calculate_cosmotherm(
        conformers,
        formatted_action_lines,
        command_lines=formated_command_lines,
        comment_line=comment_line,
        table_start_patterns=("Nr Solute",),
        **kwargs,
    )

    res_df = res_df_list[0]
    log_perm = res_df.loc[res_df.index[-1], "logPerm(cm/s)"]  # type: ignore

    return {COSMO_PERM_KEY: calculated_properties.ScalarProperty(float(log_perm))}
