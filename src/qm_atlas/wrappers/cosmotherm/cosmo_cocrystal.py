import math
from functools import partial
from pathlib import Path

import pandas as pd
from more_itertools import batched
from ppqm.utils import functools as ppqm_functools

from qm_atlas.wrappers.cosmotherm import cosmo_utils, interface

COCRYSTAL_SETTINGS = cosmo_utils.CosmoSettings(
    action_lines=None,  # overwritten in code
    command_lines=["unit notempty ehfile wtln"],
    comment_line="!! cocrystal screening !!",
)


def screen_cocrystals(
    compound_conformers: list[str | Path],
    coformers: list[str],
    level: str = "bp-tzvpd",
    options: cosmo_utils.CosmoSettings = COCRYSTAL_SETTINGS,
    coformers_dirs: list[Path] | None = None,
    temperature_Celsius: float = 25.0,
    stoichiometry: tuple[int, int] = (1, 1),
    f_fit: bool | tuple[float, float, float] = False,
    f_fit_cof: bool | tuple[float, float] = True,
    f_fit_sol: bool | tuple[float, float, float] = False,
    pZWI: bool = False,
    n_cores: int = 1,
    use_config_cosmo_dir: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    This function performs a cocrystal screening using Cosmotherm, calculating values such as
    the excess enthalpy of the mixture. The method and example applications are described in reference [1].

    Args:
        compound_conformers (list[str | Path]):
            List of compound conformers, either specifying the path to the file or
            providing the contents as a string. In the latter case, the file must be
            a .cosmo file, otherwise it can be any file that cosmotherm accepts.
        coformers (list[str]):
            List of coformers, specified by their names. Corresponding .cosmo files must
            be found in the database, either the standard cosmotherm one or an
            additional, user-provided one.
        level (str, optional):
            Level of theory. Defaults to "bp-tzvpd".
        options (cosmo_utils.CosmoSettings, optional):
            Settings for the Cosmo task. Defaults to :data:`COCRYSTAL_SETTINGS`.
        coformers_dirs (list[Path] | None, optional):
            Additional directories to find the .cosmo files for the coformers.
            Defaults to None, i.e. just relying on the cosmotherm database. If
            additional directories are considered, the cosmotherm database is searched
            first, the the coformers_dirs.

        temperature_Celsius (float, optional):
            Temperature for the calculation in °C. Defaults to 25.
        stoichiometry (tuple[int, int], optional):
            Stoichiometry of the cocrystal. Will be used for all cocrystals in the
            screening. Defaults to (1, 1).
        f_fit (bool | tuple[float, float, float], optional):
            Add the key 'f_fit' to the cosmotherm action line. This will lead to
            additionally calculating a fitted function for a heuristic expression.
            Set to True to use the default cosmotherm parameters, provide a tuple to
            set the parameters. Defaults to False.
        f_fit_cof (bool | tuple[float, float], optional):
            Add the key 'f_fit_cof' to the cosmotherm action line. This will lead to
            additionally calculating a fitted function for a heuristic expression.
            Set to True to use the default cosmotherm parameters, provide a tuple to
            set the parameters. Defaults to True.
        f_fit_sol (bool | tuple[float, float, float], optional):
            Add the key 'f_fit_sol' to the cosmotherm action line. This will lead to
            additionally calculating a fitted function for a heuristic expression.
            Set to True to use the default cosmotherm parameters, provide a tuple to
            set the parameters. Defaults to False.
        pZWI (bool, optional):
            Add the key 'pZWI' to the cosmotherm action line. This is suggested if
            either the API or the coformer is zwitterionic (punish zwittterions).
            Defaults to False.
        n_cores (int, optional):
            Number of cores to use for the calculation. Defaults to 1.
        use_config_cosmo_dir (bool, optional):
            Whether to prepend the solvent directories from the software configuration
            file to the user-provided coformers_dirs. If False, only the user-provided
            directories are used. Defaults to True.
        **kwargs: Additional keyword arguments for the output reader.

    Returns:
        pd.DataFrame: A DataFrame containing the results of the cocrystal screening.

    References:
        [1] Loschen, C. & Klamt, A. Solubility prediction, solvate and cocrystal
        screening as tools for rational crystal engineering.
        *J. Pharm. Pharmacol.* **67**, 803-811 (2015).
        https://doi.org/10.1111/jphp.12376
    """

    n_procs = min(n_cores, len(coformers))

    # split coformers in chunks
    batch_size = math.ceil(len(coformers) / n_procs)
    coformer_batches = list(batched(coformers, batch_size))
    calculator_function = partial(
        _screen_cocrystals,
        compound_conformers=compound_conformers,
        level=level,
        options=options,
        coformers_dirs=coformers_dirs,
        temperature_Celsius=temperature_Celsius,
        stoichiometry=stoichiometry,
        f_fit=f_fit,
        f_fit_cof=f_fit_cof,
        f_fit_sol=f_fit_sol,
        pZWI=pZWI,
        use_config_cosmo_dir=use_config_cosmo_dir,
        **kwargs,
    )

    if len(coformer_batches) == 1:
        dataframes = [calculator_function(coformer_batches[0])]

    else:
        dataframes = ppqm_functools.func_parallel(
            calculator_function,
            coformer_batches,
            n_cores=n_procs,
            title="Cosmo_Cocrystal",
        )

    return pd.concat(dataframes, ignore_index=True)


def _screen_cocrystals(
    coformers: list[str],
    compound_conformers: list[str | Path],
    level: str = "bp-tzvpd",
    options: cosmo_utils.CosmoSettings = COCRYSTAL_SETTINGS,
    coformers_dirs: list[Path] | None = None,
    temperature_Celsius: float = 25.0,
    stoichiometry: tuple[int, int] = (1, 1),
    f_fit: bool | tuple = False,
    f_fit_cof: bool | tuple = False,
    f_fit_sol: bool | tuple = False,
    pZWI: bool = False,
    use_config_cosmo_dir: bool = True,
    **kwargs,
) -> pd.DataFrame:

    cosmo_utils.check_input_conformers(compound_conformers)

    cosmo_paths = interface.get_cosmo_paths(level=level)

    coformers_databases = cosmo_utils.get_databases(
        cosmo_paths,
        level=level,
        other_databases=coformers_dirs,
        use_config_cosmo_dir=use_config_cosmo_dir,
    )

    coformer_files = [
        cosmo_utils.find_cosmo_files(coformers_databases, name) for name in coformers
    ]
    conformers = [compound_conformers] + coformer_files

    if options.action_lines is None:
        action_line = f"tc={temperature_Celsius} cocrystal API=1 coformer=ALL"
        action_line += f" cocrystal_n={{{stoichiometry[0]} {stoichiometry[0]}}}"
        added_fits = ["f_fit", "f_fit_cof", "f_fit_sol", "pZWI"]
        commands = [f_fit, f_fit_cof, f_fit_sol, pZWI]
        for fit, command in zip(added_fits, commands):
            if isinstance(command, tuple):
                action_line += f" {fit}" + "={" + " ".join(command) + "}"
            elif command:
                action_line += f" {fit}"
        action_lines = [action_line]

    else:
        action_lines = options.action_lines

    command_lines = options.get_command_lines(cosmo_paths)
    comment_line = options.comment_line

    res_df_list = interface.calculate_cosmotherm(
        conformers,
        action_lines,
        command_lines=command_lines,
        comment_line=comment_line,
        table_start_patterns=["Nr Coformer"],
        **kwargs,
    )

    return res_df_list[0]
