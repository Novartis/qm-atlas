#!/bin/env python3

import logging
import sys

import tqdm
from rdkit import RDLogger

from qm_atlas import __version__
from qm_atlas.command_line.pages import (
    add_reference_conformers,
    aggregate_conformer_properties,
    align,
    calculate_boltzmann_weights,
    cocrystal_screening,
    collect,
    conformer_properties,
    cosmo_pka,
    cosmotherm_properties,
    describe,
    fast_conformers,
    generate_tautomers,
    optimize_reference_conformers,
    protonate,
    read_input,
    rescoss_conformers,
    solvent_screening,
)
from qm_atlas.command_line.pipeline import main as pipeline_main
from qm_atlas.command_line.utils import matomo
from qm_atlas.selftest import command as test_software_command

try:
    from rich.logging import RichHandler
except ImportError:
    RichHandler = None

logger = logging.getLogger(__name__)


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


__description__ = """
usage: qm-atlas <command> [args]

commands
 read_input                        Read input molecules and create compound directories
 add_reference_conformers          Register reference conformers in compound directories
 align                             Align all result conformations onto a reference conformation
 fast_conformers                   Run fast conformer generation workflow
 rescoss_conformers                Run ReSCoSS conformer generation workflow (xTB-based clustering)
 optimize_reference_conformers     Perform constrained optimization on reference conformers
 conformer_properties              Calculate conformer-based properties (QM descriptors, Fukui, NMR, etc.)
 cosmotherm_properties             Calculate COSMOtherm properties (logP, free energy of solvation, etc.)
 cosmo_pka                         Calculate pKa values with COSMOtherm across registered states (macro/micro)
 generate_tautomers                Generate tautomers and register them as new states in compound directories
 protonate                         Generate protonation states (via MoKa blabber by default) and register them as new states
 solvent_screening                 Run COSMOtherm pure-solvent and mixture solubility screening from previously generated .cosmo files
 cocrystal_screening               Run COSMOtherm cocrystal screening from previously generated .cosmo files
 calculate_boltzmann_weights       Calculate per-conformer Boltzmann weights from a calculated energy property
 aggregate_conformer_properties    Aggregate per-conformer properties into per-state values (mean, Boltzmann-weighted mean, min, max)
 collect                           Collect results into a single directory (merged SDFs, cosmo archives, CSV tables)
 pipeline                          Run a multi-step pipeline from a YAML config file
 describe                          Show available options for calculators, optimizers, generators, etc.
 test_software                     Run the wrapper self-tests against a software config (checks tools/licenses)

global optional arguments:
  --debug               Set logging to debug level
  --silent              Set logging to error-only level
  --version             Print out version and exit
  --help                Print help message and exit

per-command optional arguments:
  --software_config     Path to software configuration YAML file
                        (overrides environment variable QM_ATLAS_SOFTWARE_CONFIG_FILE)

Example commands

  # Get help documentation for the read_input command
  qm-atlas read_input --help

  # Read molecules from SDF file and create compound directories
  qm-atlas read_input --input_file+ ./molecules.sdf --results-directory ./results

  # Run fast conformer generation on compound directories
  qm-atlas fast_conformers --input.results_directory ./results

  # Run ReSCoSS conformer generation with config file
  qm-atlas rescoss_conformers --config ./config.yaml --input.results_directory ./results

  # Specify a custom software config
  qm-atlas fast_conformers --software_config ./my_software_config.yaml --config ./run.yaml --input.results_directory ./results
"""


def main(args: list[str] | None = None) -> int:

    if args is None:
        args = sys.argv

    if len(args) < 2:
        print(__description__)
        return 4

    # Reset rdkit loglevel
    RDLogger.DisableLog("rdApp.*")

    # Set logger format
    logger_format = "%(levelname)-8s - %(message)s"

    # Collect optional args
    debug = False
    silent = False
    if "--debug" in args:
        debug = True
        lg = RDLogger.logger()
        lg.setLevel(RDLogger.ERROR)
        logger_format = "[%(asctime)s] [%(levelname)8s] --- %(message)s (%(filename)s:%(lineno)s)"
        args.remove("--debug")

    if "--silent" in args:
        silent = True
        args.remove("--silent")

    incognito = False
    if "--incognito" in args:
        incognito = True
        args.remove("--incognito")

    if "--version" in args:
        print(__version__)
        sys.exit()

    if RichHandler:
        log_options = {
            "format": "%(message)s",
            "datefmt": "[%X]",
            "handlers": [RichHandler(show_path=False)],
        }
    else:
        log_options = dict(
            format=logger_format,
            datefmt="%X",
            handlers=[TqdmLoggingHandler()],
        )

    # TODO Set only qm_atlas loglevel to info, ignore rest

    # Setup logging
    if not debug and not silent:
        logging.basicConfig(level=logging.INFO, **log_options)
    if not debug and silent:
        logging.basicConfig(level=logging.ERROR, **log_options)
    else:
        logging.basicConfig(level=logging.DEBUG, **log_options)

    if RichHandler:
        logging.getLogger("rich")

    # Disable all loggers not containing qm_atlas
    for log_name, log_obj in logging.Logger.manager.loggerDict.items():
        if "qm_atlas" not in log_name:
            log_obj.disabled = True

    # Call module
    module = args[1]

    if "help" in module:
        print(__description__)
        return 0

    # Track user input
    events = None
    events_pool = None
    try:
        if matomo.health_check() and not incognito:
            events, events_pool = matomo.register_event_subprocess(module, " ".join(args[2:]))
    except Exception as e:
        logger.warning("Unable to initialize Matomo")
        logger.debug(f"Exception for matomo: {e}")

    # Default exit code
    exit_code = 0  # pylint: disable=redefined-outer-name

    # Map module names to command functions
    commands = {
        "read_input": read_input.main,
        "add_reference_conformers": add_reference_conformers.main,
        "align": align.main,
        "fast_conformers": fast_conformers.main,
        "rescoss_conformers": rescoss_conformers.main,
        "optimize_reference_conformers": optimize_reference_conformers.main,
        "conformer_properties": conformer_properties.main,
        "cosmotherm_properties": cosmotherm_properties.main,
        "cosmo_pka": cosmo_pka.main,
        "generate_tautomers": generate_tautomers.main,
        "protonate": protonate.main,
        "solvent_screening": solvent_screening.main,
        "cocrystal_screening": cocrystal_screening.main,
        "calculate_boltzmann_weights": calculate_boltzmann_weights.main,
        "aggregate_conformer_properties": aggregate_conformer_properties.main,
        "collect": collect.main,
        "pipeline": pipeline_main,
        "describe": describe.main,
        "test_software": test_software_command.main,
    }

    call_command = commands.get(module)

    if call_command is None:
        print(f"Unknown command! {module}")
        exit_code = 4

    if exit_code == 0 and call_command is not None:
        try:
            call_command(args[2:])
        except SystemExit:
            exit_code = 1

    # End subprocess
    if events is not None:
        matomo.close_tracking(events, events_pool)

    return exit_code


if __name__ == "__main__":
    exit_code = main()
    # End CLI
    sys.exit(exit_code)
