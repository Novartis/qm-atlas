import logging
from pathlib import Path

from ppqm.utils.files import WorkDir
from rdkit import Chem

from qm_atlas.constants import DEFAULT_SCR
from qm_atlas.software_environment import SOFTWARE_CONFIG
from qm_atlas.wrappers.common import run_command

_logger = logging.getLogger(__name__)

FILE_NAME = "input"
SOFTWARE_NAME = "jaguar"
FEATURE_NAME = "JAGUAR_MAIN"


def convert_to_mae(
    sdf_file: Path,
):
    """Convert an SDF file to Schrödinger .mae format via Jaguar's structconvert."""

    mae_file = sdf_file.with_suffix(".mae")
    software_env_manager = SOFTWARE_CONFIG.get_environment_manager()
    structconvert_cmd = software_env_manager.get_command("jaguar", "structconvert")
    command = f"{structconvert_cmd} {sdf_file.resolve()} {mae_file.resolve()}"

    env = software_env_manager.get_run_environment("jaguar")
    _, stderr = run_command(command, env=env)

    if stderr:
        _logger.error(f"Error converting to mae file: {stderr}")

    if not mae_file.is_file():
        _logger.error("Structure Conversion produced no .mae file")
        raise RuntimeError("Structure Conversion produced no .mae file")

    return mae_file


def prepare_jaguar_input(
    mol: Chem.Mol,
    conf_id: int = 0,
    scr: Path = DEFAULT_SCR,
    keep_files: bool = False,  # disable for debugging
) -> tuple[WorkDir, Path]:
    """Write *mol* to a scratch directory and convert it to a Jaguar .mae input file."""

    # ensure the scratch directory exists
    scr.mkdir(parents=True, exist_ok=True)

    temp = WorkDir(dir=scr, prefix="jaguar_", keep=keep_files)
    scr = temp.get_path()
    scr.mkdir(parents=True, exist_ok=True)

    sdf_file = scr / f"{FILE_NAME}.sdf"
    with Chem.SDWriter(str(sdf_file.resolve())) as writer:
        writer.SetForceV3000(True)
        writer.write(mol, confId=conf_id)

    mae_file = convert_to_mae(sdf_file)

    return temp, mae_file


def extract_log_file(scr: Path) -> None:
    """Log the contents of the single .log file found in *scr* (for error reporting)."""
    log_files = [file for file in scr.iterdir() if file.suffix == ".log"]

    if len(log_files) == 0:
        _logger.error("No log file found")
        return

    if len(log_files) == 1:
        log_file = log_files[0]
        with open(log_file, "r", encoding="utf-8") as log_f:
            log_str = log_f.read()
        _logger.error(log_str)
        return

    _logger.error("Found more than one log file")
    for log_file in log_files:
        _logger.error("New log file:")
        with open(log_file, "r", encoding="utf-8") as log_f:
            log_str = log_f.read()
        _logger.error(log_str)
    return
