import logging

import pandas as pd

from qm_atlas.command_line.base_config import worker_path_completed
from qm_atlas.command_line.file_interface import compound_dir
from qm_atlas.command_line.file_interface.submission_files import WorkerPath
from qm_atlas.tasks import common as calculated_properties
from qm_atlas.tasks.calculate_properties import CalculationConfig
from qm_atlas.tasks.cosmo_properties import CosmoDescriptorsConfig, CosmoPsaConfig

_logger = logging.getLogger(__name__)


def check_conformer_expansion(
    worker_paths: list[WorkerPath],
) -> None:
    """Logs Information about the success rate of the conformer expansion

    Args:
        worker_paths (list[WorkerPath]):
            the list of paths that the task array had to work on.
    """
    log_lines = list()
    num_total = len(worker_paths)
    num_succeeded = 0

    for cpd_dir, sdf_file, log_file in worker_paths:
        if not worker_path_completed(cpd_dir, sdf_file, "conformer_expansion"):
            mol_name = cpd_dir.find_state_by_file(sdf_file)
            log_lines.append(f": {cpd_dir.cpd_name}: {mol_name}: Check log file {log_file}")
            continue
        num_succeeded += 1

    _logger.info(f"Conformer expansion succeeded for {num_succeeded}/{num_total} molecules")
    if log_lines:
        _logger.info("Errors were encountered for the following compounds:")
        for line in log_lines:
            _logger.info(line)


def check_reference_optimization(worker_paths: list[WorkerPath]):
    """Logs Information about the success rate of the reference conformer optimization

    Args:
        worker_paths (list[WorkerPath]):
            the list of paths that the task array had to work on.
    """
    log_lines = list()
    num_total = len(worker_paths)
    num_succeeded = 0

    for cpd_dir, sdf_file, log_file in worker_paths:
        if not worker_path_completed(cpd_dir, sdf_file, "reference_optimization"):
            mol_name = cpd_dir.find_state_by_file(sdf_file)
            log_lines.append(f": {cpd_dir.cpd_name}: {mol_name}: Check log file {log_file}")
            continue
        num_succeeded += 1

    _logger.info(
        f"Optimized reference conformations were found for {num_succeeded}/{num_total} conformers"
    )
    if log_lines:
        _logger.info("Errors were encountered for the following compounds:")
        for line in log_lines:
            _logger.info(line)


def check_conformer_property_calculation(
    worker_paths: list[WorkerPath],
    conformer_properties: list[CalculationConfig],
):

    log_lines = list()
    num_total = len(worker_paths) * len(conformer_properties)
    num_succeeded = 0

    for cpd_dir, sdf_file, log_file in worker_paths:
        mol = cpd_dir.extract_mol(sdf_file)
        for prop_config in conformer_properties:
            expected_properties = prop_config.get_property_names()
            property_types = prop_config.get_property_types()
            for prop in expected_properties.values():

                prop_type = property_types.get(prop, None)

                if prop_type == calculated_properties.AtomBasedProperty:
                    # check for atom-based property
                    prop_found = False
                    for atom in mol.GetAtoms():
                        if atom.HasProp(prop):
                            prop_found = True
                            break
                    if not prop_found:
                        log_lines.append(
                            f": {sdf_file.stem}: {prop_config.backend} misseys {prop}: Check log file {log_file}"
                        )
                        break

                elif not mol.HasProp(prop):
                    log_lines.append(
                        f": {sdf_file.stem}: {prop_config.backend} misses {prop}: Check log file {log_file}"
                    )
                    break
            else:
                num_succeeded += 1

    _logger.info(f"Property calculation succeeded for {num_succeeded}/{num_total} tasks")
    if log_lines:
        _logger.info("Errors were encountered for the following tasks:")
        for line in log_lines:
            _logger.info(line)


def check_cosmotherm_calculation(
    worker_paths: list[WorkerPath],
    cosmotherm_properties: list[CalculationConfig],
):

    log_lines = list()
    num_total = len(worker_paths) * len(cosmotherm_properties)
    num_succeeded = 0

    for cpd_dir, sdf_file, log_file in worker_paths:
        mol = cpd_dir.extract_mol(sdf_file)
        for prop_config in cosmotherm_properties:

            if isinstance(prop_config, (CosmoDescriptorsConfig, CosmoPsaConfig)):
                expected_properties = prop_config.get_property_names()
                for prop in expected_properties.values():
                    if not mol.HasProp(prop):
                        log_lines.append(
                            f": {sdf_file.stem}: {prop_config.backend} misses {prop}: Check log file {log_file}"
                        )
                        break
                else:
                    num_succeeded += 1

            else:
                # check whole-molecule csv
                cpd_name = cpd_dir.find_state_by_file(sdf_file)
                whole_mol_csv = cpd_dir.molecule_csv
                df = compound_dir.read_csv(whole_mol_csv)

                for prop in prop_config.get_property_names().values():

                    if (
                        compound_dir.STATE_COLUMN not in df.columns
                        or cpd_name not in df[compound_dir.STATE_COLUMN].values
                        or prop not in df.columns
                    ):
                        log_lines.append(
                            f": {sdf_file.stem}: {prop_config.backend} misses {prop} for {cpd_name} in {whole_mol_csv}: Check log file {log_file}"
                        )
                        break
                    else:
                        value = df.loc[df[compound_dir.STATE_COLUMN] == cpd_name, prop].values[0]
                        if pd.isna(value):  # check for NaN
                            log_lines.append(
                                f": {sdf_file.stem}: {prop_config.backend} has invalid value for {prop} for {cpd_name} in {whole_mol_csv}: Check log file {log_file}"
                            )
                            break
                else:
                    num_succeeded += 1

    _logger.info(
        f"Cosmotherm property calculation succeeded for {num_succeeded}/{num_total} tasks"
    )
    if log_lines:
        _logger.info("Errors were encountered for the following tasks:")
        for line in log_lines:
            _logger.info(line)
