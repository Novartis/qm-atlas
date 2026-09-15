"""Tests for the shared conformer-selection utilities (conformer_selection.py).

Selection reads per-conformer scalar properties straight from the result SDF
files of the pre-computed ``glycine`` (CHEMBL773) compound directory, so no
external software runs here.
"""

import pytest
from conftest import RESOURCES  # pylint: disable=import-error

from qm_atlas.command_line.file_interface import compound_dir
from qm_atlas.command_line.utils import conformer_selection as cs

GLYCINE_DIR = RESOURCES / "cpd_dir_example" / "glycine"  # glycine, CHEMBL773
ENERGY = "tm_sp_TotalEnergy(Ht)"


@pytest.fixture
def glycine_cpd_dir():
    return compound_dir.create_cpd_dir(GLYCINE_DIR, create_new=False)


# ---------------------------------------------------------------------------
# PropertyFilter / ConformerSelection models
# ---------------------------------------------------------------------------


def test_property_filter_rejects_negative_window():
    with pytest.raises(ValueError):
        cs.PropertyFilter(property_name=ENERGY, window_from_min=-1.0)
    with pytest.raises(ValueError):
        cs.PropertyFilter(property_name=ENERGY, window_from_max=-1.0)


def test_property_filter_rejects_keep_lowest_and_highest_together():
    with pytest.raises(ValueError):
        cs.PropertyFilter(property_name=ENERGY, keep_lowest=1, keep_highest=1)


def test_conformer_selection_is_noop():
    assert cs.ConformerSelection().is_noop() is True
    assert cs.ConformerSelection(source="reference_results").is_noop() is False
    assert (
        cs.ConformerSelection(
            property_filters=[cs.PropertyFilter(property_name=ENERGY, keep_lowest=1)]
        ).is_noop()
        is False
    )


# ---------------------------------------------------------------------------
# _read_scalar_property
# ---------------------------------------------------------------------------


def test_read_scalar_property_present(glycine_cpd_dir):
    sdf = glycine_cpd_dir.results_dir / "glycine_c0.sdf"
    assert cs._read_scalar_property(sdf, ENERGY) == pytest.approx(-284.6017866389)


def test_read_scalar_property_missing_returns_none(glycine_cpd_dir):
    sdf = glycine_cpd_dir.results_dir / "glycine_c0.sdf"
    assert cs._read_scalar_property(sdf, "does_not_exist") is None


# ---------------------------------------------------------------------------
# select_result_files
# ---------------------------------------------------------------------------


def test_select_no_filter_returns_all(glycine_cpd_dir):
    selection = cs.ConformerSelection()
    selected = cs.select_result_files(glycine_cpd_dir, "glycine", selection)
    assert set(selected) == set(glycine_cpd_dir.get_result_files("glycine"))


def test_select_keep_lowest_returns_global_minimum(glycine_cpd_dir):
    selection = cs.ConformerSelection(
        property_filters=[cs.PropertyFilter(property_name=ENERGY, keep_lowest=1)]
    )
    selected = cs.select_result_files(glycine_cpd_dir, "glycine", selection)
    assert len(selected) == 1
    # glycine_c0 is the lowest-energy conformer in the fixture.
    assert selected[0].name == "glycine_c0.sdf"


def test_select_window_from_min_bounds_pool(glycine_cpd_dir):
    all_files = glycine_cpd_dir.get_result_files("glycine")
    values = {f: cs._read_scalar_property(f, ENERGY) for f in all_files}
    lowest = min(values.values())
    window = 0.005  # Hartree

    selection = cs.ConformerSelection(
        property_filters=[cs.PropertyFilter(property_name=ENERGY, window_from_min=window)]
    )
    selected = cs.select_result_files(glycine_cpd_dir, "glycine", selection)

    expected = {f for f, v in values.items() if v <= lowest + window}
    assert set(selected) == expected
    # The lowest is always kept; the very highest one is outside the window.
    assert min(selected, key=lambda f: values[f]).name == "glycine_c0.sdf"
    assert len(selected) < len(all_files)


def test_select_reference_results_source(glycine_cpd_dir):
    selection = cs.ConformerSelection(source="reference_results")
    selected = cs.select_result_files(glycine_cpd_dir, "glycine", selection)
    assert {f.name for f in selected} == {
        "glycine_ref_0_opt_0.sdf",
        "glycine_ref_0_opt_1.sdf",
    }


def test_select_max_value_filter(glycine_cpd_dir):
    # Keep only conformers at or below a cutoff between the min and the rest.
    cutoff = -284.60
    selection = cs.ConformerSelection(
        property_filters=[cs.PropertyFilter(property_name=ENERGY, max_value=cutoff)]
    )
    selected = cs.select_result_files(glycine_cpd_dir, "glycine", selection)
    assert {f.name for f in selected} == {"glycine_c0.sdf"}


# ---------------------------------------------------------------------------
# filter_worker_paths
# ---------------------------------------------------------------------------


def test_filter_worker_paths_noop_returns_input(glycine_cpd_dir):
    files = glycine_cpd_dir.get_result_files("glycine")
    log = glycine_cpd_dir.log_dir / "task.log"
    worker_paths = [(glycine_cpd_dir, f, log) for f in files]

    result = cs.filter_worker_paths(worker_paths, cs.ConformerSelection())
    assert result == worker_paths


def test_filter_worker_paths_applies_selection(glycine_cpd_dir):
    files = glycine_cpd_dir.get_result_files("glycine")
    log = glycine_cpd_dir.log_dir / "task.log"
    worker_paths = [(glycine_cpd_dir, f, log) for f in files]

    selection = cs.ConformerSelection(
        property_filters=[cs.PropertyFilter(property_name=ENERGY, keep_lowest=1)]
    )
    result = cs.filter_worker_paths(worker_paths, selection)

    assert len(result) == 1
    assert result[0][1].name == "glycine_c0.sdf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
