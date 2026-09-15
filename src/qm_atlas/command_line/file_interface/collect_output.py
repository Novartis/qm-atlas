"""Collect the results of a qm_atlas run into a single flat directory.

For every compound directory the collected output contains, per registered
state that has results:

- ``{state}.sdf``: all result conformers merged into a single SDF file.
- ``{state}_cosmo.zip``: the associated ``.cosmo`` files (only if any exist).

In addition, the per-compound bookkeeping files are copied verbatim when
present:

- ``{compound}_properties.csv`` (per-state properties)
- ``{compound}_conformer_properties.csv`` (per-conformer properties)
- ``{compound}_pka.csv`` (pKa values)
- ``{compound}_solubility_screening.csv`` (solvent screening results)
- ``{compound}_cocrystal_screening.csv`` (cocrystal screening results)
"""

import logging
import shutil
from pathlib import Path
from zipfile import ZipFile

from rdkit import Chem

from qm_atlas.command_line.file_interface import compound_dir
from qm_atlas.command_line.utils.extended_precision_sdwriter import ExtendedPrecisionSDWriter

_logger = logging.getLogger(__name__)

OUTPUT_DIR_NAME = "collected_results"


def collect_files(
    cpd_dirs: list[compound_dir.CpdDir],
    target_dir: Path,
    use_v2000: bool = False,
) -> None:
    """Collect results from several compound directories into *target_dir*.

    Args:
        cpd_dirs (list[compound_dir.CpdDir]): Compound directories to collect from.
        target_dir (Path): Directory the collected files are written to. Created
            if it does not exist.
        use_v2000 (bool): Write merged SDF files using the V2000 format instead of
            V3000. Defaults to False.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    for cpd_dir in cpd_dirs:
        if not cpd_dir.has_results():
            _logger.info(f"No results found for {cpd_dir.cpd_name}, skipping")
            continue

        _logger.info(f"Collecting results for {cpd_dir.cpd_name}")
        extract_results(cpd_dir, target_dir, use_v2000=use_v2000)


def extract_results(
    cpd_dir: compound_dir.CpdDir,
    target_dir: Path,
    use_v2000: bool = False,
) -> None:
    """Extract the results of a single compound directory into *target_dir*.

    Args:
        cpd_dir (compound_dir.CpdDir): The compound directory to extract from.
        target_dir (Path): Directory the collected files are written to.
        use_v2000 (bool): Write merged SDF files using the V2000 format instead of
            V3000. Defaults to False.
    """
    for state_name in cpd_dir.registry_handler.get_registered_names():
        if not cpd_dir.has_results_for_name(state_name):
            continue
        _collect_state_results(cpd_dir, state_name, target_dir, use_v2000=use_v2000)

    _copy_bookkeeping_files(cpd_dir, target_dir)


def _collect_state_results(
    cpd_dir: compound_dir.CpdDir,
    state_name: str,
    target_dir: Path,
    use_v2000: bool,
) -> None:
    """Merge conformer SDFs and zip COSMO files for a single state."""
    result_files = cpd_dir.get_result_files(state_name)
    if not result_files:
        return

    merged_sdf_file = target_dir / f"{state_name}.sdf"
    with ExtendedPrecisionSDWriter(str(merged_sdf_file.resolve())) as writer:
        if use_v2000:
            writer.SetForceV3000(False)

        for sdf_file in result_files:
            with Chem.SDMolSupplier(str(sdf_file.resolve()), removeHs=False) as supplier:
                mol = next(supplier, None)
                if mol is None:
                    _logger.warning(f"Could not read molecule from {sdf_file}, skipping")
                    continue
                writer.write(mol)

    cosmo_files = [
        cosmo_file
        for sdf_file in result_files
        if (cosmo_file := cpd_dir.get_associated_cosmo_file(sdf_file)).exists()
    ]
    if cosmo_files:
        cosmo_zip_file = target_dir / f"{state_name}_cosmo.zip"
        with ZipFile(cosmo_zip_file, "w") as zipf:
            for cosmo_file in cosmo_files:
                zipf.write(cosmo_file, arcname=cosmo_file.name)


def _copy_bookkeeping_files(cpd_dir: compound_dir.CpdDir, target_dir: Path) -> None:
    """Copy the per-compound CSV bookkeeping files that exist."""
    bookkeeping_files = [
        cpd_dir.molecule_csv,
        cpd_dir.conformer_csv,
        cpd_dir.pka_csv,
        cpd_dir.solubility_screening_csv,
        cpd_dir.cocrystal_screening_csv,
    ]
    for csv_file in bookkeeping_files:
        if csv_file.is_file():
            shutil.copyfile(csv_file, target_dir / csv_file.name)
