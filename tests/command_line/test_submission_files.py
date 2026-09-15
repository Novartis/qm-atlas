"""Tests for the UGE task-control file helpers (submission_files.py).

These exercise the worker-path <-> JSON task-control round-trip and the
per-task work-item lookup. They use the ``glycine`` (CHEMBL773) compound
directory so that ``read_task_control`` can reconstruct real ``CpdDir`` objects.
No cluster or external software is involved.
"""

import pytest
from conftest import RESOURCES  # pylint: disable=import-error

from qm_atlas.command_line.file_interface import compound_dir, submission_files

GLYCINE_DIR = RESOURCES / "cpd_dir_example" / "glycine"  # glycine, CHEMBL773


@pytest.fixture
def worker_paths():
    cpd_dir = compound_dir.create_cpd_dir(GLYCINE_DIR, create_new=False)
    log = cpd_dir.log_dir / "task.log"
    return [(cpd_dir, sdf, log) for sdf in cpd_dir.get_result_files("glycine")]


def test_write_task_control_one_job_per_task(tmp_path, worker_paths):
    control_file = tmp_path / "ctrl.json"
    num_tasks = submission_files.write_task_control(worker_paths, control_file, jobs_per_task=1)

    assert control_file.is_file()
    assert num_tasks == len(worker_paths)


def test_write_task_control_groups_jobs_per_task(tmp_path, worker_paths):
    control_file = tmp_path / "ctrl.json"
    num_tasks = submission_files.write_task_control(worker_paths, control_file, jobs_per_task=3)

    # 9 conformers grouped 3-per-task -> 3 tasks.
    assert num_tasks == 3
    data = submission_files.read_task_control(control_file)
    assert set(data.keys()) == {1, 2, 3}
    assert all(len(items) == 3 for items in data.values())


def test_read_task_control_roundtrips_worker_paths(tmp_path, worker_paths):
    control_file = tmp_path / "ctrl.json"
    submission_files.write_task_control(worker_paths, control_file, jobs_per_task=1)

    restored = submission_files.read_task_control(control_file)

    # Same number of tasks, and each item reconstructs a (CpdDir, Path, Path).
    assert len(restored) == len(worker_paths)
    first_cpd, first_sdf, first_log = restored[1][0]
    assert isinstance(first_cpd, compound_dir.CpdDir)
    assert first_sdf.suffix == ".sdf"
    assert first_log.name == "task.log"
    # The reconstructed SDF paths match the originals (order preserved).
    restored_names = [items[0][1].name for items in restored.values()]
    original_names = [sdf.name for _, sdf, _ in worker_paths]
    assert restored_names == original_names


def test_get_task_work_items_returns_task_slice(tmp_path, worker_paths):
    control_file = tmp_path / "ctrl.json"
    submission_files.write_task_control(worker_paths, control_file, jobs_per_task=3)

    items = submission_files.get_task_work_items(control_file, task_id=2)
    assert len(items) == 3
    assert all(isinstance(cpd, compound_dir.CpdDir) for cpd, _, _ in items)


def test_get_task_work_items_unknown_task_raises(tmp_path, worker_paths):
    control_file = tmp_path / "ctrl.json"
    submission_files.write_task_control(worker_paths, control_file, jobs_per_task=1)

    with pytest.raises(ValueError, match="Task ID 999 not found"):
        submission_files.get_task_work_items(control_file, task_id=999)


def test_get_submission_scratch_dir():
    results_dir = RESOURCES / "some_results"
    scratch = submission_files.get_submission_scratch_dir(results_dir)
    assert scratch == results_dir / submission_files.SCRATCH_DIR_NAME


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
