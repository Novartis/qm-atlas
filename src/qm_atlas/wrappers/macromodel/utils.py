import logging
from pathlib import Path

from qm_atlas.software_environment import SOFTWARE_CONFIG
from qm_atlas.utils import open_utf8
from qm_atlas.wrappers.common import run_command

_logger = logging.getLogger(__name__)


# Substrings that point to a likely cause of an empty/missing Macromodel output.
MACROMODEL_ERROR_INDICATORS = (
    "Maximum conformation storage exceeded",
    "PROTMP",
    "AUTO SERIAL",
    "license",
    "FATAL",
    "ERROR",
)


def log_macromodel_errors(scr: Path) -> None:
    """Scan the Macromodel log files in a scratch directory for known error
    indicators and log a plausible reason for a failed conformer generation.

    Args:
        scr (Path): The scratch directory containing the Macromodel log files.
    """

    log_files = sorted(scr.glob("*.log"))
    if not log_files:
        _logger.error(f"No Macromodel log file found in {scr} to diagnose the failure")
        return

    found_indicator = False
    storage_exceeded = False
    for log_file in log_files:
        try:
            with open_utf8(log_file, "r") as f:
                lines = f.readlines()
        except OSError:
            continue

        for line in lines:
            stripped = line.strip()
            if any(ind.lower() in stripped.lower() for ind in MACROMODEL_ERROR_INDICATORS):
                _logger.error(f"{log_file.name}: {stripped}")
                found_indicator = True
            if "Maximum conformation storage exceeded" in stripped:
                storage_exceeded = True

    if storage_exceeded:
        _logger.error(
            "Macromodel exceeded its internal conformer storage limit. This happens "
            "when the search finds more conformers than 'max_conformers'. The output "
            "stage then fails. Try increasing the 'max_conformers' option."
        )

    if not found_indicator:
        _logger.error("Could not identify a specific error in the Macromodel log files")


def convert_to_mae(
    sdf_file: Path,
):
    """Convert an SDF file to Schrödinger .mae format via MacroModel's structconvert."""
    software_env_manager = SOFTWARE_CONFIG.get_environment_manager()
    str_conv = software_env_manager.get_command("macromodel", "structconvert")
    mae_file = sdf_file.with_suffix(".mae")
    command = f"{str_conv} {sdf_file} {mae_file}"
    env = software_env_manager.get_run_environment("macromodel")
    _, stderr = run_command(command, env=env)
    if stderr:
        _logger.error(f"Error converting to mae file: {stderr}")
    if not mae_file.is_file():
        _logger.error("Structure Conversion produced no .mae file")
        raise RuntimeError("Structure Conversion produced no .mae file")
    return mae_file
