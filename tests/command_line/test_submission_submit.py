"""Tests for the UGE submission orchestration in utils/submission.py.

The ``qsub`` calls are mocked, so nothing is submitted to a real cluster. What
is exercised is the pure-python glue that decides *what* gets submitted: retry
logic, hold-job waiting, task-control chunking and config serialization. A bug
in this glue silently drops or duplicates molecules across the task array.
"""

import subprocess

import pytest
from conftest import RESOURCES  # pylint: disable=import-error

from qm_atlas.command_line.base_config import (
    BaseInterfaceConfig,
    CalculationInput,
    SubmissionConfig,
)
from qm_atlas.command_line.file_interface import compound_dir
from qm_atlas.command_line.utils import submission

GLYCINE_DIR = RESOURCES / "cpd_dir_example" / "glycine"  # glycine, CHEMBL773


# ---------------------------------------------------------------------------
# submit_script_with_retry
# ---------------------------------------------------------------------------


def test_submit_script_with_retry_succeeds_first_try(monkeypatch):
    monkeypatch.setattr(submission, "submit_script", lambda script: "JOB1")
    assert submission.submit_script_with_retry("script.sh") == "JOB1"


def test_submit_script_with_retry_recovers_after_failure(monkeypatch):
    calls = {"n": 0}

    def flaky(script):
        calls["n"] += 1
        if calls["n"] == 1:
            raise subprocess.CalledProcessError(1, "qsub")
        return "JOB2"

    monkeypatch.setattr(submission, "submit_script", flaky)
    monkeypatch.setattr(submission.time, "sleep", lambda _s: None)

    assert submission.submit_script_with_retry("script.sh", wait_time_seconds=0) == "JOB2"
    assert calls["n"] == 2


def test_submit_script_with_retry_exhausts_retries(monkeypatch):
    def always_fail(script):
        raise RuntimeError("qsub down")

    monkeypatch.setattr(submission, "submit_script", always_fail)
    monkeypatch.setattr(submission.time, "sleep", lambda _s: None)

    with pytest.raises(RuntimeError, match="failed after all retries"):
        submission.submit_script_with_retry("script.sh", max_retries=2, wait_time_seconds=0)


# ---------------------------------------------------------------------------
# wait_for_jobs_using_hold_job
# ---------------------------------------------------------------------------


def test_wait_for_jobs_returns_when_finished_file_exists(tmp_path, monkeypatch):
    jobs = ["J1", "J2"]
    monkeypatch.setattr(submission, "generate_script", lambda **kwargs: "#!/bin/bash\n")
    monkeypatch.setattr(
        submission, "write_script", lambda content, directory, filename: directory / filename
    )
    monkeypatch.setattr(submission, "submit_script_with_retry", lambda script: "HOLD1")
    monkeypatch.setattr(submission.time, "sleep", lambda _s: None)

    # Pre-create the sentinel so the polling loop exits immediately.
    finished = (tmp_path / f"hold_job_{'__'.join(jobs)}.finished").resolve()
    finished.write_text("")

    result = submission.wait_for_jobs_using_hold_job(jobs, tmp_path, generate_dirs=False)
    assert result == finished


# ---------------------------------------------------------------------------
# submit: task-control chunking + config serialization
# ---------------------------------------------------------------------------


class _FakeEnvManager:
    def get_submission_config(self):
        return {}


def test_submit_writes_task_control_and_config(tmp_path, monkeypatch):
    cpd_dir = compound_dir.create_cpd_dir(GLYCINE_DIR, create_new=False)
    log = cpd_dir.log_dir / "task.log"
    worker_paths = [(cpd_dir, sdf, log) for sdf in cpd_dir.get_result_files("glycine")]

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    config = BaseInterfaceConfig(
        input=CalculationInput(results_directory=results_dir),
        submission_config=SubmissionConfig(jobs_per_task=3, max_time="1hour", name="qma_test"),
    )

    monkeypatch.setattr(submission, "generate_name", lambda: "uid")
    monkeypatch.setattr(submission, "generate_script", lambda **kwargs: "#!/bin/bash\n")

    scripts = {}

    def fake_write_script(content, directory, filename):
        path = directory / filename
        path.write_text(content)
        scripts["path"] = path
        return path

    monkeypatch.setattr(submission, "write_script", fake_write_script)
    monkeypatch.setattr(submission, "submit_script_with_retry", lambda script: "JOB-XYZ")
    monkeypatch.setattr(
        submission.SOFTWARE_CONFIG, "get_environment_manager", lambda: _FakeEnvManager()
    )

    job_id = submission.submit(worker_paths, config, command="qm_atlas conformer_properties")

    assert job_id == "JOB-XYZ"
    scratch = results_dir / "qm_atlas_submissions"
    # Task control chunks the 9 conformers into ceil(9/3) = 3 tasks.
    control = scratch / "task_control_uid.json"
    assert control.is_file()
    import json

    task_dict = json.loads(control.read_text())
    assert len(task_dict) == 3
    # The per-task config YAML was written for the array job to consume.
    assert (scratch / "config_uid.yaml").is_file()
    assert scripts["path"].is_file()


# ---------------------------------------------------------------------------
# compute_task_concurrency + build_submission_script (real generate_script)
# ---------------------------------------------------------------------------


def test_compute_task_concurrency_floor_divides_core_budget():
    assert submission.compute_task_concurrency(400, 16) == 25
    # A budget equal to one task's cores allows exactly one concurrent task.
    assert submission.compute_task_concurrency(16, 16) == 1


def test_build_submission_script_requests_cores_per_task(tmp_path):
    config = SubmissionConfig(cores_per_task=16, max_time="1hour", name="qma_test")

    script = submission.build_submission_script(
        config, cmd="run.py", num_tasks=100, log_dir=tmp_path
    )

    assert "#$ -pe smp 16" in script


def test_build_submission_script_limits_task_concurrency(tmp_path):
    config = SubmissionConfig(
        cores_per_task=16, max_total_num_cores=400, max_time="1hour", name="qma_test"
    )

    script = submission.build_submission_script(
        config, cmd="run.py", num_tasks=100, log_dir=tmp_path
    )

    # 400 // 16 == 25 concurrent tasks out of the 100-task array.
    assert "#$ -t 1-100:1" in script
    assert "#$ -tc 25" in script


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
