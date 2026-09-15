import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from io import StringIO
from itertools import chain
from pathlib import Path

import pandas as pd
from ppqm.utils import WorkDir

from qm_atlas.constants import DEFAULT_SCR
from qm_atlas.software_environment import SOFTWARE_CONFIG
from qm_atlas.utils import open_utf8
from qm_atlas.wrappers.common import run_command

_logger = logging.getLogger(__name__)

ENV_COSMOTHERMDIR = "COSMOTHERMDIR"
CONFORMER_FORMATS = [
    ".cosmo",  # quantum chemistry generated cosmo file, e.g. turbomole
    ".cos",  # cosmo file from a MOPAC calculation
    ".ccf",  # compressed cosmo file in binary format
    ".mcos",  # COSMO-metafile
    ".mix",  # mixture file
]

COSMOTHERM_CTD_DIR = "COSMOtherm/CTDATA-FILES/"
COSMOTHERM_LICENSE_DIR = "licensefiles/"
COSMOTHERM_DATABASE = "COSMOtherm/DATABASE-COSMO"

DEFAULT_COMMENT = "!! generic cosmotherm calculation !!"
DEFAULT_TABLE_START_PATTERNS = ("Nr Compound",)

COSMO_LEVELS = {
    "bp-tzvpd": ("BP_TZVPD_FINE", "BP-TZVPD-FINE"),
    "bp-tzvp": ("BP_TZVP_", "BP-TZVP-COSMO"),
    "bp-svp": ("BP_SVP_AM1", "BP-SVP-AM1"),
    "dmol3-pbe": ("DMOL3_PBE", "DMOL3-PBE"),
}


@dataclass
class CosmoPaths:
    license_dir: Path
    ctd_dir: Path
    ctd_file: str
    database_dir: Path


def get_cosmo_paths(
    cosmotherm_dir: Path | None = None,
    level: str = "bp-tzvpd",
    cosmo_database: Path | None = None,
    cmd: str | None = None,
) -> CosmoPaths:
    """
    Retrieves the paths for COSMOtherm related files and directories.

    Args:
        cosmotherm_dir (Path | None, optional):
            Path to the COSMOtherm directory. If not provided, the function will try to
            find it using the COSMOTHERMDIR environment variable. Defaults to None.
        level (str, optional):
            The level of theory to use for COSMOtherm calculations. Defaults to "bp-tzvpd".
        cosmo_database (Path | None, optional):
            Path to the COSMOtherm database directory. If not provided, the function will
            use the default database directory in the COSMOtherm directory. Defaults to None.
        cmd (str | None, optional):
            Explicit path to the cosmotherm executable. If not provided, it is resolved
            from the environment/config. Defaults to None.

    Raises:
        ValueError:
            If the provided level of theory is not recognized.
        RuntimeError:
            If the COSMOtherm directory or the COSMOtherm database directory cannot be found.

    Returns:
        CosmoPaths:
            A dataclass containing the paths to the license directory,
            CTD directory, CTD file, and database directory.
    """

    if not level in COSMO_LEVELS:
        _logger.error(f"Cosmotherm level {level} unknown. Available levels: {COSMO_LEVELS.keys()}")
        raise ValueError(f"Cosmotherm level {level} unknown.")

    software_env_manager = SOFTWARE_CONFIG.get_environment_manager()
    if cosmotherm_dir is None:
        cosmo_env = software_env_manager.get_run_environment("cosmotherm")
        cosmotherm_dir_str = cosmo_env.get(ENV_COSMOTHERMDIR, None)
        if cosmotherm_dir_str is None:
            if cmd is None:
                cmd = software_env_manager.get_command("cosmotherm", "cosmotherm")
            stdout, _ = run_command(f"which {cmd}", env=cosmo_env)
            command_str = stdout.replace("\n", "").strip()
            cosmotherm_dir = Path(command_str).resolve().parent.parent.parent
        else:
            cosmotherm_dir = Path(cosmotherm_dir_str).parent

    ctd_name, db_name = COSMO_LEVELS[level]

    license_dir = cosmotherm_dir / COSMOTHERM_LICENSE_DIR
    ctd_dir = cosmotherm_dir / COSMOTHERM_CTD_DIR

    if cosmo_database is None:
        database_dir = cosmotherm_dir / COSMOTHERM_DATABASE / db_name
    else:
        database_dir = cosmo_database

    if not database_dir.is_dir():
        _logger.error("Could not find cosmotherm database")
        raise RuntimeError("Unable to locate cosmotherm database")

    ctd_files = [
        file
        for file in ctd_dir.iterdir()
        if file.suffix == ".ctd" and ctd_name in file.stem and not "ELYTE" in file.stem
    ]

    if len(ctd_files) == 1:
        ctd_file = ctd_files[0].name

    else:
        _logger.error(f"Could not find cosmotherm ctd file. Found {ctd_files}")
        raise RuntimeError("Unable to locate cosmotherm ctd file")

    cosmo_paths = CosmoPaths(
        license_dir=license_dir, ctd_dir=ctd_dir, ctd_file=ctd_file, database_dir=database_dir
    )

    return cosmo_paths


def setup_cosmotherm(
    conformers: list[list[str | Path]],
    action_lines: list[str],
    command_lines: list[str] | None = None,
    comment_line: str = DEFAULT_COMMENT,
    scr: Path = DEFAULT_SCR,
    keep_files: bool = False,
    conformer_format: str = ".cosmo",
    conformer_names: list[str] | None = None,
    additional_properties_list: list[str] | None = None,
) -> tuple[WorkDir, Path]:
    """
    Sets up the COSMOtherm calculation by writing the input files.

    Args:
        conformers (list[list[str | Path]]):
            List of conformers for each compound. Each conformer can be a path to a file
            or a string representing the contents of a .cosmo file.
        action_lines (list[str]):
            List of action lines to be included in the input file.
        command_lines (list[str], optional):
            List of command lines to be included in the input file. Defaults to None.
        comment_line (str, optional):
            Comment line to be included in the input file. Defaults to DEFAULT_COMMENT.
        scr (Path, optional):
            Path to the scratch directory where the input files will be written.
            Defaults to DEFAULT_SCR.
        keep_files (bool, optional):
            Whether to keep the input files after the calculation. Defaults to False.
        conformer_format (str, optional):
            Format of the conformer files. Defaults to ".cosmo".
        conformer_names (list[str], optional):
            List of names for the conformers. Defaults to None.
        additional_properties_list (list[str] | None, optional):
            Extra per-compound property lines appended to each compound's input block
            (e.g. "DGfus=0" to mark a liquid). Defaults to None.

    Raises:
        ValueError: If more than two command lines are provided.

    Returns:
        tuple[Path, Path]:
            A tuple containing the path to the temporary directory and the path to the input file.
    """

    temp = WorkDir(dir=scr, prefix="cosmotherm_", keep=keep_files)
    scr = temp.get_path()
    scr.mkdir(parents=True, exist_ok=True)

    if command_lines is None:
        command_lines = [""]
    if len(command_lines) == 0:
        command_lines = [""]  # must have at least one empty command line
    if len(command_lines) >= 3:
        raise ValueError("Cosmotherm only accepts one or two command lines")

    conformer_input_lines = write_conformers(
        conformers,
        scr=scr,
        conformer_format=conformer_format,
        conformer_names=conformer_names,
        additional_properties_list=additional_properties_list,
    )

    all_input_lines = chain(command_lines, [comment_line], conformer_input_lines, action_lines)
    input_str = "\n".join(all_input_lines)

    input_file = scr / "input.txt"
    with open_utf8(input_file, "w") as f:
        f.write(input_str)

    return temp, input_file


def write_conformers(
    conformers: list[list[str | Path]],
    scr: Path = DEFAULT_SCR,
    conformer_format: str = ".cosmo",
    conformer_names: Sequence[str | None] | None = None,
    additional_properties_list: list[str] | None = None,
) -> list[str]:
    """
    Writes the conformers to the specified directory and returns a list of
    input lines for each conformer.

    Args:
        conformers (list[list[str | Path]]):
            List of conformers for each compound. Each conformer can be a path to a file
            or a string representing the contents of a .cosmo file.
        scr (Path, optional):
            Path to the directory where the conformer files will be written. Defaults to DEFAULT_SCR.
        conformer_format (str, optional):
            Format of the conformer files. Defaults to ".cosmo".
        conformer_names (list[str], optional):
            List of names for the conformers. Defaults to None.
        additional_properties_list (list[str] | None, optional):
            Extra per-compound property lines appended to each compound's input block
            (e.g. "DGfus=0" to mark a liquid). Defaults to None.

    Raises:
        RuntimeError:
            If the lengths of conformer_names and conformers, or additional_properties_list
            and conformers, do not match.

    Returns:
        list[str]: List of input lines for each conformer.
    """

    if conformer_names is None:
        conformer_names = [None] * len(conformers)

    if additional_properties_list is None:
        additional_properties_list = [""] * len(conformers)

    if len(conformer_names) != len(conformers):
        raise RuntimeError("Error in setting up cosmotherm input")

    if len(additional_properties_list) != len(conformers):
        raise RuntimeError("Error in setting up cosmotherm input")

    conformer_input = []

    for block_idx, (conformer_block, name, additional_properties) in enumerate(
        zip(conformers, conformer_names, additional_properties_list)
    ):
        block_input = []
        for conf_idx, conformer in enumerate(conformer_block):
            nn = "molecule" if name is None else name
            default_name = f"{nn}{block_idx}_c{conf_idx}"
            conf_path = write_conformer(
                conformer, default_name, scr=scr, conformer_format=conformer_format
            )
            input_line = f"f={conf_path.name}"
            if conf_path.parent != scr:
                input_line += f' fdir="{conf_path.parent}"'
            if conf_idx == 0:
                input_line += f" {additional_properties} "
            if conf_idx == 0 and name is not None:
                input_line += f" comp={name}"
            block_input.append(input_line)

        if len(block_input) >= 2:
            block_input[0] = f"[ {block_input[0]}"
            for input_line in block_input[1:]:
                input_line = f"  {input_line}"
            block_input[-1] = f"{block_input[-1]} ]"

        conformer_input.extend(block_input)

    return conformer_input


def write_conformer(
    conformer: str | Path, name: str, scr: Path = DEFAULT_SCR, conformer_format: str = ".cosmo"
) -> Path:
    """
    Writes a single conformer to a file and returns the path to the file.

    Args:
        conformer (str | Path):
            The conformer to write. Can be a path to a file or a string representing
            the contents of a .cosmo file.
        name (str):
            Name of the conformer.
        scr (Path, optional):
            Path to the directory where the conformer file will be written. Defaults to DEFAULT_SCR.
        conformer_format (str, optional):
            Format of the conformer file. Defaults to ".cosmo".

    Raises:
        ValueError:
            If the provided conformer file format is not recognized, or if the
            provided conformer file does not exist.

    Returns:
        Path: Path to the written conformer file.
    """

    if isinstance(conformer, Path):
        suffix = conformer.suffix
        if suffix not in CONFORMER_FORMATS:
            raise ValueError(f"Unknown file format: {suffix}")
        if not conformer.exists():
            raise ValueError(f"Input file {conformer} does not exist")
        return conformer

    if isinstance(conformer, str):
        if conformer_format not in CONFORMER_FORMATS:
            raise ValueError(f"Unknown input format: {conformer_format}")

        cosmo_file = scr / f"{name}{conformer_format}"
        with open_utf8(cosmo_file, "w") as cf:
            cf.write(conformer)
        return cosmo_file


def run_cosmotherm(
    input_file: Path,
    n_cores: int = 1,
    cmd: str | None = None,
) -> bool:
    """
    Runs the COSMOtherm program with the given input file.

    Args:
        input_file (Path):
            The path to the input file for COSMOtherm.
        n_cores (int, optional):
            The number of cores to use for the COSMOtherm calculation. Defaults to 1.
        cmd (str | None, optional):
            The command to run COSMOtherm. Defaults to None (infer command from config).

    Raises:
        ValueError: If the COSMOtherm command is not found in the system's PATH.

    Returns:
        bool:
            True if the COSMOtherm calculation was successful and the
            expected output file was produced, False otherwise.
    """
    software_env_manager = SOFTWARE_CONFIG.get_environment_manager()
    if cmd is None:
        cmd = software_env_manager.get_command("cosmotherm", "cosmotherm")
    run_dir = input_file.parent
    _logger.debug(f"Running cosmotherm in {run_dir}")

    cmd = f"{cmd} {input_file.name}"
    if n_cores > 1:
        cmd += f" -n {n_cores}"

    env = software_env_manager.get_run_environment("cosmotherm")
    stdout, stderr = run_command(cmd, env=env, cwd=run_dir)

    success = log_output(stdout, "stdout") & log_output(stderr, "stderr")

    if not input_file.with_suffix(".tab").is_file():
        _logger.error("Cosmotherm did not produce the expected output file")
        success = False

    if not success:
        _logger.error("Errors noted in cosmotherm output file:")
        for line in read_errors_output(input_file.with_suffix(".out")):
            _logger.error(line)

    return success


def log_output(
    stdxx: str | None,
    identifier: str,
) -> bool:
    """
    Logs the output from a COSMOtherm call.

    Args:
        stdxx (str | None):
            The output (either stdout or stderr) from the COSMOtherm call.
            If None, the function will return True.
        identifier (str):
            A string identifier to indicate whether the output is stdout or stderr.

    Returns:
        bool: False if there is output to log (i.e., stdxx is not None and not empty), True otherwise.
    """

    if stdxx is None:
        return True

    if len(stdxx) == 0:
        return True

    _logger.error(f"Got {identifier} from cosmotherm call")

    lines = stdxx.split("\n")
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue
        _logger.error(line)

    return False


def read_errors_output(filename: Path) -> list[str]:
    """extracts errors from a cosmotherm .out file"""

    with open_utf8(filename, "r") as f:
        lines = f.readlines()

    errors = []

    key = "error:"

    for line in lines:
        line = line.lower()
        if key in line:
            errors.append(line.replace(key, "").replace("!", "").strip())

    return errors


def read_cosmotherm_output(
    file: Path,
    table_start_patterns: Sequence[str] = DEFAULT_TABLE_START_PATTERNS,
    multi_solvent: bool = False,
) -> list[pd.DataFrame]:
    """
    Reads the COSMOtherm output (.tab) file and extracts data tables.

    Args:
        file (Path):
            The path to the COSMOtherm output file.
        table_start_patterns (Sequence[str], optional):
            A sequence of strings that mark the start of a data table in the output file.
            Defaults to ("Nr Compound",).
        multi_solvent (bool, optional):
            A flag indicating whether the output file includes data for solvent mixtures.
            If True, the function will use a different method to parse the data tables.
            Defaults to False.

    Returns:
        list[pd.DataFrame]:
            A list of pandas DataFrames, each representing a data table in the COSMOtherm output file.
    """

    data_tables = []

    in_table = False
    lines_to_read = []
    with open_utf8(file, "r") as f:
        for line in f:

            stripped_line = line.strip()

            # if inside a table, keep the line
            if in_table:

                if len(stripped_line) == 0:
                    in_table = False

                    table = read_table(lines_to_read, multi_solvent=multi_solvent)
                    data_tables.append(table)
                    lines_to_read = []
                    continue

                lines_to_read.append(stripped_line)

            # check if this line marks the start of a table
            else:
                for pattern in table_start_patterns:
                    if pattern in line:
                        in_table = True
                        break
                if in_table:
                    lines_to_read.append(stripped_line)

    # catch last table in the file
    if len(lines_to_read) > 0:
        table = read_table(lines_to_read, multi_solvent=multi_solvent)
        data_tables.append(table)

    return data_tables


def read_table(
    table_lines: list[str],
    multi_solvent: bool = False,
) -> pd.DataFrame:
    """Reads a table in the cosmother output (.tab) file.

    Args:
        table_lines (list[str]): The lines in the output file that contain the table.
        multi_solvent (bool): If True, parse the wider mixture-solubility layout that
            uses a two-space column delimiter. Defaults to False.

    Returns:
        pd.DataFrame: The cosmotherm output table.
    """

    if not multi_solvent:
        table_block = StringIO("\n".join(table_lines))
        table = pd.read_csv(
            table_block,
            sep=r"\s+",
            engine="python",
            on_bad_lines="skip",
        )

        return table

    # for the solubility calculation including solvent mixtures, a space delimiter
    # does not work. We use two spaces and make sure that all columns are separated
    # by two spaces. This requires some interfering

    adapted_lines = [fix_header_line(table_lines[0])]
    for line in table_lines[1:]:
        adapted_lines.append(fix_line(line))

    table_block = StringIO("\n".join(adapted_lines))
    table = pd.read_csv(table_block, sep=r"\s{2,}", engine="python", index_col=False)
    return table


def fix_header_line(header_line: str) -> str:
    """assures all headers in the cosmotherm table output are separated by two spaces
    by making some explicit replacements.
    """

    line = header_line.replace("Nr Solvent", "Nr  Solvent")
    return line.replace("Solvent_density Solvent_MolWeight", "Solvent_density  Solvent_MolWeight")


def fix_line(line: str) -> str:
    """adapts a line in the cosmotherm output file. Replace an integer followed by one
    space by the same integer followed by two spaces.
    """
    return re.sub(r"^(\d+)\s", r"\1  ", line)


def calculate_cosmotherm(
    conformers: list[list[str | Path]],
    action_lines: list[str],
    command_lines: list[str] | None = None,
    comment_line: str = DEFAULT_COMMENT,
    scr: Path = DEFAULT_SCR,
    keep_files: bool = False,
    cmd: str | None = None,
    n_cores: int = 1,
    output_reader: Callable[[Path], list[pd.DataFrame]] = read_cosmotherm_output,
    conformer_names: list[str] | None = None,
    additional_properties_list: list[str] | None = None,
    conformer_format: str = ".cosmo",
    **reader_kwargs,
) -> list[pd.DataFrame]:
    """
    Sets up and runs a COSMOtherm calculation, then reads the output and returns the data tables.
    This function combines the setup, execution, and output reading of a COSMOtherm calculation, it is
    not specialized to any specific type of calculation.

    Args:
        conformers (list[list[str | Path]]):
            List of conformers for each compound. Each conformer can be a path to a .cosmo file
            or a string with the contents of a .cosmo file.
        action_lines (list[str]):
            List of action lines to be included in the input file.
        command_lines (list[str], optional):
            List of command lines to be included in the input file. Defaults to None.
        comment_line (str, optional):
            Comment line to be included in the input file. Defaults to DEFAULT_COMMENT.
        scr (Path, optional):
            Path to the scratch directory where the input files will be written. Defaults to DEFAULT_SCR.
        keep_files (bool, optional):
            Whether to keep the input files after the calculation. Defaults to False.
        cmd (str | None, optional):
            The command to run COSMOtherm. Defaults to None (infer command from config).
        n_cores (int, optional):
            The number of cores to use for the COSMOtherm calculation. Defaults to 1.
        output_reader (Callable[[Path], list[pd.DataFrame]]):
            A function to read the COSMOtherm output. Defaults to read_cosmotherm_output,
            check there for the signature of the function.
        conformer_names (list[str], optional):
            List of names for the conformers. Defaults to None.
        additional_properties_list (list[str] | None, optional):
            Extra per-compound property lines appended to each compound's input block
            (e.g. "DGfus=0" to mark a liquid). Defaults to None.
        conformer_format (str, optional):
            Format of the conformer files. Defaults to ".cosmo", allowed formats are listed in CONFORMER_FORMATS.
        **reader_kwargs:
            Additional keyword arguments forwarded to *output_reader* (e.g.
            ``table_start_patterns`` or ``multi_solvent``).

    Returns:
        list[pd.DataFrame]:
            A list of pandas DataFrames, each representing a data table in the COSMOtherm output file.
    """

    _, input_file = setup_cosmotherm(
        conformers,
        action_lines,
        command_lines=command_lines,
        comment_line=comment_line,
        scr=scr,
        keep_files=keep_files,
        conformer_names=conformer_names,
        additional_properties_list=additional_properties_list,
        conformer_format=conformer_format,
    )

    success = run_cosmotherm(input_file, n_cores=n_cores, cmd=cmd)

    if not success:
        raise RuntimeError("Cosmotherm calculation failed")

    tables = output_reader(input_file.with_suffix(".tab"), **reader_kwargs)

    return tables
