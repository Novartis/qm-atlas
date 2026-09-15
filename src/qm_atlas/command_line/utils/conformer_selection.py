"""Shared conformer-selection utilities.

Both ``aggregate_conformer_properties`` (which conformers to aggregate over) and
``conformer_properties`` (which conformers to run a calculation on) need to
select a subset of a state's result conformers. This module provides the
configuration models and the selection logic used by both.

Selection has two stages:

- **source**: pick the pool of result files (all results, or reference results
  only).
- **property filters**: keep only conformers whose per-conformer scalar property
  values satisfy every configured :class:`PropertyFilter`.

All filters are combined with logical AND. Values are read directly from the
per-conformer result SDF files, so any per-conformer scalar property (an energy,
a descriptor, ...) can be used — hence "property filter" rather than
"energy filter". Comparisons happen in the property's native units.
"""

import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from rdkit import Chem

from qm_atlas.command_line.file_interface import compound_dir
from qm_atlas.command_line.file_interface.submission_files import WorkerPath

_logger = logging.getLogger(__name__)


class PropertyFilter(BaseModel):
    """Keep conformers whose value of a per-conformer scalar property passes.

    All configured criteria are combined with logical AND. A conformer that is
    missing the property is dropped (with a warning). Comparisons use the
    property's native units.
    """

    property_name: str = Field(
        ...,
        description="Name of the per-conformer scalar property to filter on "
        "(as stored on the result SDF files)",
    )
    window_from_min: float | None = Field(
        default=None,
        description="Keep conformers with value <= min(value) + window_from_min "
        "(e.g. an energy window above the lowest-energy conformer)",
    )
    window_from_max: float | None = Field(
        default=None,
        description="Keep conformers with value >= max(value) - window_from_max",
    )
    min_value: float | None = Field(
        default=None,
        description="Keep conformers with value >= min_value (absolute lower bound)",
    )
    max_value: float | None = Field(
        default=None,
        description="Keep conformers with value <= max_value (absolute upper bound)",
    )
    keep_lowest: int | None = Field(
        default=None,
        ge=1,
        description="Keep only the N conformers with the smallest value",
    )
    keep_highest: int | None = Field(
        default=None,
        ge=1,
        description="Keep only the N conformers with the largest value",
    )

    @model_validator(mode="after")
    def validate_bounds(self) -> "PropertyFilter":
        if self.window_from_min is not None and self.window_from_min < 0:
            raise ValueError("window_from_min must be non-negative")
        if self.window_from_max is not None and self.window_from_max < 0:
            raise ValueError("window_from_max must be non-negative")
        if self.keep_lowest is not None and self.keep_highest is not None:
            raise ValueError("Cannot set both keep_lowest and keep_highest on one filter")
        return self


class ConformerSelection(BaseModel):
    """Selects a subset of a state's result conformers."""

    source: Literal["all_results", "reference_results"] = Field(
        default="all_results",
        description="Pool of conformers to select from: all result conformers, "
        "or only conformers derived from reference inputs",
    )
    property_filters: list[PropertyFilter] = Field(
        default_factory=list,
        description="Per-conformer property filters, combined with logical AND",
    )

    def is_noop(self) -> bool:
        """Return True if this selection keeps every result conformer unchanged."""
        return self.source == "all_results" and not self.property_filters


def _read_scalar_property(sdf_file: Path, property_name: str) -> float | None:
    """Read a scalar property from a result SDF file, or None if unavailable."""
    mol = None
    with Chem.SDMolSupplier(str(sdf_file.resolve()), removeHs=False) as supplier:
        if len(supplier) == 0:
            return None
        mol = supplier[0]
    if mol is None or not mol.HasProp(property_name):
        return None
    try:
        return mol.GetDoubleProp(property_name)
    except (KeyError, ValueError):
        return None


def _source_files(
    cpd_dir: compound_dir.CpdDir,
    state_name: str,
    source: str,
) -> list[Path]:
    """Return the pool of result files for *state_name* according to *source*."""
    if source == "reference_results":
        files: list[Path] = []
        for ref_input in cpd_dir.get_reference_inputs(state_name):
            files.extend(cpd_dir.get_reference_results(state_name, ref_input))
        return files
    return cpd_dir.get_result_files(state_name)


def _apply_filter(
    files: list[Path],
    values: dict[Path, float],
    prop_filter: PropertyFilter,
) -> list[Path]:
    """Return the subset of *files* passing a single :class:`PropertyFilter`."""
    # Conformers missing the property cannot pass the filter.
    available = [f for f in files if f in values]
    missing = [f for f in files if f not in values]
    for f in missing:
        _logger.warning(
            f"Conformer '{f.stem}' has no property '{prop_filter.property_name}'; excluded by filter"
        )

    if not available:
        return []

    numeric = {f: values[f] for f in available}

    if prop_filter.window_from_min is not None:
        lowest = min(numeric.values())
        threshold = lowest + prop_filter.window_from_min
        available = [f for f in available if numeric[f] <= threshold]
    if prop_filter.window_from_max is not None:
        highest = max(numeric.values())
        threshold = highest - prop_filter.window_from_max
        available = [f for f in available if numeric[f] >= threshold]
    if prop_filter.min_value is not None:
        available = [f for f in available if numeric[f] >= prop_filter.min_value]
    if prop_filter.max_value is not None:
        available = [f for f in available if numeric[f] <= prop_filter.max_value]

    if prop_filter.keep_lowest is not None:
        available = sorted(available, key=lambda f: numeric[f])[: prop_filter.keep_lowest]
    elif prop_filter.keep_highest is not None:
        available = sorted(available, key=lambda f: numeric[f], reverse=True)[
            : prop_filter.keep_highest
        ]

    return available


def select_result_files(
    cpd_dir: compound_dir.CpdDir,
    state_name: str,
    selection: ConformerSelection,
) -> list[Path]:
    """Return the result SDF files of *state_name* selected by *selection*.

    Args:
        cpd_dir: The compound directory owning the state.
        state_name: The registered state to select conformers for.
        selection: The selection configuration.

    Returns:
        list[Path]: Selected result SDF files, in the original result order.
    """
    files = _source_files(cpd_dir, state_name, selection.source)
    if not selection.property_filters:
        return files

    kept = set(files)
    for prop_filter in selection.property_filters:
        values: dict[Path, float] = {}
        for f in files:
            value = _read_scalar_property(f, prop_filter.property_name)
            if value is not None:
                values[f] = value
        passing = set(_apply_filter(files, values, prop_filter))
        kept &= passing

    # Preserve original order.
    return [f for f in files if f in kept]


def filter_worker_paths(
    worker_paths: list[WorkerPath],
    selection: ConformerSelection,
) -> list[WorkerPath]:
    """Filter per-conformer worker paths down to the selected conformers.

    Worker paths are grouped by their owning (compound directory, state) so that
    relative filters (e.g. an energy window from the lowest-energy conformer) are
    evaluated per state.

    Args:
        worker_paths: The per-conformer worker paths to filter.
        selection: The selection configuration.

    Returns:
        list[WorkerPath]: The subset of worker paths whose conformer is selected.
    """
    if selection.is_noop():
        return worker_paths

    # Cache selected files per (cpd_dir path, state) to avoid recomputation.
    allowed_cache: dict[tuple[Path, str], set[Path]] = {}
    filtered: list[WorkerPath] = []

    for cpd_dir, sdf_file, log_file in worker_paths:
        try:
            state_name = cpd_dir.find_state_by_file(sdf_file)
        except ValueError:
            _logger.warning(f"Could not resolve owning state for '{sdf_file}'; skipping")
            continue

        cache_key = (cpd_dir.dir, state_name)
        if cache_key not in allowed_cache:
            allowed_cache[cache_key] = {
                f.resolve() for f in select_result_files(cpd_dir, state_name, selection)
            }

        if sdf_file.resolve() in allowed_cache[cache_key]:
            filtered.append((cpd_dir, sdf_file, log_file))

    return filtered
