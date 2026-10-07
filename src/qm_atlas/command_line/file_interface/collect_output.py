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

Finally, two combined tables spanning all compounds are written, each with the
compound name and the canonical (hydrogen-free) SMILES of the state as the first
two columns:

- ``all_properties.csv`` (every per-state property row)
- ``all_conformer_properties.csv`` (every per-conformer property row)
"""

import logging
import shutil
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from rdkit import Chem

from qm_atlas.command_line.file_interface import compound_dir
from qm_atlas.command_line.utils.extended_precision_sdwriter import ExtendedPrecisionSDWriter

_logger = logging.getLogger(__name__)

OUTPUT_DIR_NAME = "collected_results"

COMBINED_MOLECULE_CSV = "all_properties.csv"
COMBINED_CONFORMER_CSV = "all_conformer_properties.csv"
COMPOUND_NAME_COLUMN = "Compound_Name"
CANONICAL_SMILES_COLUMN = "Canonical_SMILES"


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

    _write_combined_csvs(cpd_dirs, target_dir)


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


def _state_smiles_map(cpd_dir: compound_dir.CpdDir) -> dict[str, str]:
    """Map each registered state name to its canonical (no-H) SMILES."""
    return {
        name: cpd_dir.registry_handler.get_entry(name).smiles
        for name in cpd_dir.registry_handler.get_registered_names()
    }


def _conformer_smiles_map(cpd_dir: compound_dir.CpdDir) -> dict[str, str]:
    """Map each result conformer file stem to its state's canonical (no-H) SMILES."""
    conformer_smiles: dict[str, str] = {}
    for name in cpd_dir.registry_handler.get_registered_names():
        entry = cpd_dir.registry_handler.get_entry(name)
        for result_file in entry.get_result_files(include_references=True):
            conformer_smiles[result_file.stem] = entry.smiles
    return conformer_smiles


def _tag_frame(
    csv_file: Path,
    cpd_name: str,
    identifier_col: str,
    smiles_map: dict[str, str],
) -> pd.DataFrame | None:
    """Read a per-compound CSV and prepend compound name + canonical SMILES columns."""
    if not csv_file.is_file():
        return None
    df = compound_dir.read_csv(csv_file)
    if df.empty:
        return None
    smiles = df[identifier_col].map(smiles_map) if identifier_col in df.columns else None
    df.insert(0, CANONICAL_SMILES_COLUMN, smiles)
    df.insert(0, COMPOUND_NAME_COLUMN, cpd_name)
    return df


def _write_combined_csvs(cpd_dirs: list[compound_dir.CpdDir], target_dir: Path) -> None:
    """Write one combined molecule-property CSV and one combined conformer-property CSV.

    Each row carries the compound name and the canonical (hydrogen-free) SMILES of
    its state, so the aggregated tables stay self-describing across compounds.
    """
    molecule_frames: list[pd.DataFrame] = []
    conformer_frames: list[pd.DataFrame] = []

    for cpd_dir in cpd_dirs:
        molecule_frame = _tag_frame(
            cpd_dir.molecule_csv,
            cpd_dir.cpd_name,
            compound_dir.STATE_COLUMN,
            _state_smiles_map(cpd_dir),
        )
        if molecule_frame is not None:
            molecule_frames.append(molecule_frame)

        conformer_frame = _tag_frame(
            cpd_dir.conformer_csv,
            cpd_dir.cpd_name,
            compound_dir.CONFORMER_COLUMN,
            _conformer_smiles_map(cpd_dir),
        )
        if conformer_frame is not None:
            conformer_frames.append(conformer_frame)

    if molecule_frames:
        compound_dir.write_csv(
            pd.concat(molecule_frames, ignore_index=True), target_dir / COMBINED_MOLECULE_CSV
        )
    if conformer_frames:
        compound_dir.write_csv(
            pd.concat(conformer_frames, ignore_index=True), target_dir / COMBINED_CONFORMER_CSV
        )
