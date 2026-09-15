"""Tests for worker-path / compound-dir routing in base_config.py.

``extract_worker_paths`` decides, for every page and pipeline step, which files
each task operates on. A routing bug here silently processes the wrong molecules,
so this is high-value pure-python coverage. All cases run against the
pre-computed ``glycine`` (CHEMBL773) compound directory.
"""

import json
import shutil

import pytest
from conftest import RESOURCES  # pylint: disable=import-error

from qm_atlas.command_line import base_config
from qm_atlas.command_line.base_config import (
    CalculationInput,
    extract_cpd_dirs,
    extract_worker_paths,
)
from qm_atlas.command_line.file_interface import compound_dir, submission_files

GLYCINE_DIR = RESOURCES / "cpd_dir_example" / "glycine"  # glycine, CHEMBL773


@pytest.fixture
def glycine_cpd_dir():
    return compound_dir.create_cpd_dir(GLYCINE_DIR, create_new=False)


@pytest.fixture
def results_dir(tmp_path):
    """A results directory containing a copy of the glycine compound dir."""
    results = tmp_path / "results"
    results.mkdir()
    shutil.copytree(GLYCINE_DIR, results / "glycine")
    return results


# ---------------------------------------------------------------------------
# Worker-path getters
# ---------------------------------------------------------------------------


def test_get_input_worker_paths(glycine_cpd_dir):
    paths = base_config.get_input_worker_paths(glycine_cpd_dir)
    assert [p[1].name for p in paths] == ["glycine.sdf"]
    # Every worker path is (cpd_dir, sdf, log).
    _cpd, _sdf, log = paths[0]
    assert log.suffix == ".log"


def test_get_reference_worker_paths(glycine_cpd_dir):
    paths = base_config.get_reference_worker_paths(glycine_cpd_dir)
    assert {p[1].name for p in paths} == {"glycine_ref_0.sdf"}


def test_get_conformer_property_worker_paths(glycine_cpd_dir):
    paths = base_config.get_conformer_property_worker_paths(glycine_cpd_dir)
    assert len(paths) == len(glycine_cpd_dir.get_result_files("glycine"))


def test_get_cosmo_property_worker_paths_one_per_state(glycine_cpd_dir):
    paths = base_config.get_cosmo_property_worker_paths(glycine_cpd_dir)
    # Whole-molecule COSMO jobs use a single representative conformer per state.
    assert len(paths) == 1


def test_get_solvent_screening_worker_paths_at_most_one(glycine_cpd_dir):
    paths = base_config.get_solvent_screening_worker_paths(glycine_cpd_dir)
    assert len(paths) == 1


# ---------------------------------------------------------------------------
# extract_worker_paths: compound-directories mode + workflow dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "workflow_type, expected",
    [
        ("conformer_expansion", 1),
        ("reference_optimization", 1),
        ("cosmotherm_properties", 1),
        ("solvent_screening", 1),
    ],
)
def test_extract_worker_paths_compound_dirs_dispatch(workflow_type, expected):
    ci = CalculationInput(results_directory=None, compound_directories=[GLYCINE_DIR])
    paths = extract_worker_paths(ci, workflow_type=workflow_type)
    assert len(paths) == expected


def test_extract_worker_paths_conformer_properties_returns_all_results():
    ci = CalculationInput(results_directory=None, compound_directories=[GLYCINE_DIR])
    paths = extract_worker_paths(ci, workflow_type="conformer_properties")
    cpd = compound_dir.create_cpd_dir(GLYCINE_DIR, create_new=False)
    assert len(paths) == len(cpd.get_result_files("glycine"))


def test_extract_worker_paths_unsupported_workflow_type():
    ci = CalculationInput(results_directory=None, compound_directories=[GLYCINE_DIR])
    with pytest.raises(ValueError, match="Unsupported workflow type"):
        extract_worker_paths(ci, workflow_type="not_a_workflow")


# ---------------------------------------------------------------------------
# extract_worker_paths: skip_completed (resume)
# ---------------------------------------------------------------------------


@pytest.fixture
def input_only_cpd_dir(tmp_path, glycine_cpd_dir):
    """A fresh compound directory holding only the glycine input (no results)."""
    new_dir = compound_dir.create_cpd_dir(tmp_path / "glycine", create_new=True)
    input_sdf = glycine_cpd_dir.get_input_sdf_files()[0]
    new_dir.add_input_structure(glycine_cpd_dir.extract_mol(input_sdf))
    return new_dir


def test_extract_worker_paths_skip_completed_drops_finished():
    # The glycine fixture already carries result conformers, so a resume run skips it.
    ci = CalculationInput(results_directory=None, compound_directories=[GLYCINE_DIR])
    paths = extract_worker_paths(ci, workflow_type="conformer_expansion", skip_completed=True)
    assert paths == []


def test_extract_worker_paths_skip_completed_keeps_pending(input_only_cpd_dir):
    ci = CalculationInput(results_directory=None, compound_directories=[input_only_cpd_dir.dir])
    paths = extract_worker_paths(ci, workflow_type="conformer_expansion", skip_completed=True)
    assert [p[1].name for p in paths] == ["glycine.sdf"]


def test_extract_worker_paths_skip_completed_reference_optimization():
    # The glycine fixture already has optimized reference results.
    ci = CalculationInput(results_directory=None, compound_directories=[GLYCINE_DIR])
    paths = extract_worker_paths(ci, workflow_type="reference_optimization", skip_completed=True)
    assert paths == []


def test_extract_worker_paths_reference_results_do_not_mark_conformers_done(tmp_path):
    # Killed-run scenario: conformer results deleted, reference-opt results kept.
    # Reference results share the results/ dir, so the compound must stay pending.
    cpd = tmp_path / "glycine"
    shutil.copytree(GLYCINE_DIR, cpd)
    for stale in (cpd / "results").glob("glycine_c*.sdf"):
        stale.unlink()
    registry = json.loads((cpd / "registry.json").read_text())
    registry["glycine"]["result_files"] = []
    (cpd / "registry.json").write_text(json.dumps(registry))

    ci = CalculationInput(results_directory=None, compound_directories=[cpd])
    paths = extract_worker_paths(ci, workflow_type="conformer_expansion", skip_completed=True)
    assert [p[1].name for p in paths] == ["glycine.sdf"]


def test_worker_path_completed_predicate(glycine_cpd_dir, input_only_cpd_dir):
    done_sdf = glycine_cpd_dir.get_input_sdf_files()[0]
    pending_sdf = input_only_cpd_dir.get_input_sdf_files()[0]
    assert base_config.worker_path_completed(glycine_cpd_dir, done_sdf, "conformer_expansion")
    assert not base_config.worker_path_completed(
        input_only_cpd_dir, pending_sdf, "conformer_expansion"
    )
    # Workflows without a registered completeness check are never skipped.
    assert not base_config.worker_path_completed(glycine_cpd_dir, done_sdf, "conformer_properties")


# ---------------------------------------------------------------------------
# extract_worker_paths: results-directory mode
# ---------------------------------------------------------------------------


def test_extract_worker_paths_results_directory_mode(results_dir):
    ci = CalculationInput(results_directory=results_dir)
    paths = extract_worker_paths(ci, workflow_type="conformer_expansion")
    assert [p[1].name for p in paths] == ["glycine.sdf"]


# ---------------------------------------------------------------------------
# extract_worker_paths: task-control-file mode
# ---------------------------------------------------------------------------


@pytest.fixture
def task_control_file(tmp_path, glycine_cpd_dir):
    worker_paths = base_config.get_conformer_property_worker_paths(glycine_cpd_dir)
    ctrl = tmp_path / "ctrl.json"
    submission_files.write_task_control(worker_paths, ctrl, jobs_per_task=3)
    return ctrl, len(worker_paths)


def test_extract_worker_paths_task_id_slice(task_control_file):
    ctrl, _total = task_control_file
    ci = CalculationInput(results_directory=None, task_control_file=ctrl, task_id=2)
    paths = extract_worker_paths(ci, workflow_type="conformer_properties")
    assert len(paths) == 3  # 9 conformers / 3-per-task


def test_extract_worker_paths_all_tasks_when_no_task_id(task_control_file):
    ctrl, total = task_control_file
    ci = CalculationInput(results_directory=None, task_control_file=ctrl)
    paths = extract_worker_paths(ci, workflow_type="conformer_properties")
    assert len(paths) == total


def test_extract_worker_paths_unknown_task_id(task_control_file):
    ctrl, _total = task_control_file
    ci = CalculationInput(results_directory=None, task_control_file=ctrl, task_id=999)
    with pytest.raises(ValueError, match="Task ID 999 not found"):
        extract_worker_paths(ci, workflow_type="conformer_properties")


# ---------------------------------------------------------------------------
# extract_cpd_dirs
# ---------------------------------------------------------------------------


def test_extract_cpd_dirs_compound_directories():
    ci = CalculationInput(results_directory=None, compound_directories=[GLYCINE_DIR])
    cpd_dirs = extract_cpd_dirs(ci)
    assert [c.cpd_name for c in cpd_dirs] == ["glycine"]


def test_extract_cpd_dirs_results_directory(results_dir):
    ci = CalculationInput(results_directory=results_dir)
    cpd_dirs = extract_cpd_dirs(ci)
    assert [c.cpd_name for c in cpd_dirs] == ["glycine"]


def test_extract_cpd_dirs_task_control_dedup(task_control_file):
    ctrl, _total = task_control_file
    ci = CalculationInput(results_directory=None, task_control_file=ctrl)
    cpd_dirs = extract_cpd_dirs(ci)
    # All 9 worker paths belong to one compound dir -> deduplicated to one.
    assert len(cpd_dirs) == 1
    assert cpd_dirs[0].cpd_name == "glycine"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
