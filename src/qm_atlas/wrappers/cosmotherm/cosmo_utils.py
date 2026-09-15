import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from qm_atlas.software_environment import get_cosmo_solvents_dirs
from qm_atlas.wrappers.cosmotherm.interface import CONFORMER_FORMATS, CosmoPaths

_logger = logging.getLogger(__name__)


COSMO_SYNONYMS = {
    "acetic acid": "aceticacid",
    "acetic_acid": "aceticacid",
    "acetic-acid": "aceticacid",
    "acetone": "propanone",
    "chloroform": "chcl3",
    "dmso": "dimethylsulfoxide",
    "octanol": "1-octanol",
    "tetrahydrofuran": "thf",
    "water": "h2o",
}


def get_default_command_line(cosmo_paths: CosmoPaths):
    """Return the default cosmotherm command line for the given cosmo paths."""

    default_command_line: str = (
        f'ctd="{cosmo_paths.ctd_file}" cdir="{cosmo_paths.ctd_dir}" '
        f'ldir="{cosmo_paths.license_dir}"'
    )

    return default_command_line


@dataclass
class CosmoSettings:
    action_lines: list[str]
    command_lines: list[str]
    comment_line: str

    def get_command_lines(self, cosmo_paths: CosmoPaths) -> list[str]:
        if len(self.command_lines) >= 2:
            return self.command_lines

        return [get_default_command_line(cosmo_paths)] + self.command_lines


def get_databases(
    cosmo_paths: CosmoPaths,
    level: str = "bp-tzvpd",
    other_databases: list[Path] | None = None,
    use_config_cosmo_dir: bool = True,
) -> tuple[Path, ...]:
    """joins the default cosmo database with other databases if provided. In the returned list,
        other_databases comes first, then cosmo databases from the software config, and last the
        ones included with the cosmotherm installation.

    Args:
        cosmo_paths (CosmoPaths):
            The paths of the cosmotherm module.
        level (str, optional):
            The level of theory, used to locate the configured solvent directories.
            Defaults to "bp-tzvpd".
        other_databases (list[Path], optional):
            The paths to other databases. Defaults to None.
        use_config_cosmo_dir (bool, optional):
            Whether to prepend the cosmo databases from the software config. Defaults to True.

    Returns:
        tuple[Path, ...]: The paths to the databases to use.
    """

    if use_config_cosmo_dir:
        config_cosmo_dirs = get_cosmo_solvents_dirs(level)
        if other_databases is None:
            other_databases = config_cosmo_dirs
        else:
            other_databases = other_databases + config_cosmo_dirs

    if other_databases is None:
        other_databases = []

    return tuple(other_databases + [cosmo_paths.database_dir])


@lru_cache
def get_all_cosmo_files(cosmo_dirs: tuple[Path, ...], max_depth: int = 1) -> dict[str, Path]:
    """collects all cosmo files in a collection of dictionaries and stores them as
        a dictionary mapping the name of the file to the path.

    Args:
        cosmo_dirs (tuple[Path, ...]):
            The directories to search
        max_depth (int, optional):
            How many levels of subdirectories to consider. Defaults to 1.

    Returns:
        dict[str, Path]: a dictionary mapping the name of the file to the path
    """

    cosmo_files = dict()

    for cosmo_dir in cosmo_dirs:

        new_cosmo_files = _get_all_cosmo_files_dir(cosmo_dir, max_depth=max_depth)
        cosmo_files = {**new_cosmo_files, **cosmo_files}

    return cosmo_files


@lru_cache
def _get_all_cosmo_files_dir(cosmo_dir: Path | str, max_depth: int = 1) -> dict[str, Path]:

    cosmo_files = dict()

    if isinstance(cosmo_dir, str):
        cosmo_dir = Path(cosmo_dir)

    for file_or_dir in cosmo_dir.iterdir():

        if file_or_dir.is_dir() and max_depth > 0:
            new_cosmo_files = _get_all_cosmo_files_dir(file_or_dir, max_depth=max_depth - 1)
            cosmo_files = {**new_cosmo_files, **cosmo_files}

        else:
            if file_or_dir.suffix == ".cosmo":
                cosmo_files[file_or_dir.stem] = file_or_dir

    return cosmo_files


def find_cosmo_files(
    cosmo_dirs: tuple[Path, ...],
    compound_name: str,
    search_depth: int = 1,
    fail_on_error: bool = False,
) -> list[Path]:
    """
    Searches for .cosmo files for a given compound in specified directories.

    Args:
        cosmo_dirs (tuple[Path, ...]):
            The directories to search for .cosmo files.
        compound_name (str):
            The name of the compound to search for.
        search_depth (int, optional):
            How many levels of subdirectories to consider. Defaults to 1.
        fail_on_error (bool, optional):
            If True, raises an error when no .cosmo file is found for the compound.
            Defaults to False.

    Raises:
        ValueError: If no .cosmo file is found for the compound and fail_on_error is True.

    Returns:
        list[Path]: A list of Paths to the .cosmo files found for the compound.
    """

    cosmo_files_dict = get_all_cosmo_files(cosmo_dirs, max_depth=search_depth)

    cosmo_files = []
    conformer_idx = 0

    search_name = COSMO_SYNONYMS.get(compound_name.lower(), compound_name.lower())
    while f"{search_name}_c{conformer_idx}" in cosmo_files_dict:
        cosmo_files.append(cosmo_files_dict[f"{search_name}_c{conformer_idx}"])
        conformer_idx += 1

    if len(cosmo_files) > 0:
        return cosmo_files

    search_name = compound_name
    while f"{search_name}_c{conformer_idx}" in cosmo_files_dict:
        cosmo_files.append(cosmo_files_dict[f"{search_name}_c{conformer_idx}"])
        conformer_idx += 1

    if len(cosmo_files) > 0:
        return cosmo_files

    _logger.error(f"No .cosmo file found for {compound_name} in {cosmo_dirs}.")
    if fail_on_error:
        raise ValueError(f"No .cosmo file found for {compound_name}")

    return list()


def check_input_conformers(
    compound_conformers: list[str | Path],
    strict_input: bool = True,
) -> None:
    """Validate that *compound_conformers* are non-empty and in a cosmotherm-accepted format."""

    if len(compound_conformers) == 0:
        _logger.error("No conformers in input for cosmotherm")
        raise ValueError("No conformers in input")

    for compound_conformer in compound_conformers:
        if isinstance(compound_conformer, Path):
            file_suffix = compound_conformer.suffix
            if not file_suffix in CONFORMER_FORMATS:
                _logger.error(f"Unknown file {file_suffix} for cosmotherm")
                if strict_input:
                    raise ValueError(f"Unknown file {file_suffix} for cosmotherm")
            return

        if isinstance(compound_conformer, str):
            if not "$cosmo" in compound_conformer:
                _logger.error("Input string does not appear to be a cosmo file.")
                if strict_input:
                    raise ValueError("Input string does not appear to be a cosmo file.")
            return

        raise ValueError("Unknown input format for conformers")

    return
