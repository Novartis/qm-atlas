"""Tests for the per-task logging redirection in task_files.py.

The pipeline relies on ``change_log_file`` / ``reset_root_logging`` /
``close_task_log_handler`` to keep each step's output isolated to its own log
file and to restore the pipeline's logging afterwards. A bug here silently
corrupts log files (the exact regression noted in repo memory). These tests
snapshot and restore global logging so they don't disturb the rest of the suite.
"""

import logging

import pytest

from qm_atlas.command_line.file_interface import task_files


@pytest.fixture
def isolated_root_logging():
    root = logging.root
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_task = list(task_files._task_log_handler)
    for h in list(root.handlers):
        root.removeHandler(h)
    task_files._task_log_handler.clear()
    yield root
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    for h in saved_handlers:
        root.addHandler(h)
    root.setLevel(saved_level)
    task_files._task_log_handler[:] = saved_task


def test_change_log_file_redirects_root_to_file(tmp_path, isolated_root_logging):
    log_file = tmp_path / "task1.log"
    task_files.change_log_file(log_file, level=logging.INFO)

    logging.getLogger("qm_atlas.test").info("hello-from-task")
    for h in isolated_root_logging.handlers:
        h.flush()

    assert log_file.is_file()
    assert "hello-from-task" in log_file.read_text()
    # The task handler is tracked so it can be closed later.
    assert task_files._task_log_handler


def test_change_log_file_closes_previous_task_handler(tmp_path, isolated_root_logging):
    first = tmp_path / "task1.log"
    second = tmp_path / "task2.log"
    task_files.change_log_file(first, level=logging.INFO)
    first_handler = task_files._task_log_handler[0]

    task_files.change_log_file(second, level=logging.INFO)

    # Switching tasks replaces (and closes) the previous task handler.
    assert first_handler not in isolated_root_logging.handlers
    logging.getLogger("qm_atlas.test").info("second-task-msg")
    for h in isolated_root_logging.handlers:
        h.flush()
    assert "second-task-msg" in second.read_text()


def test_reset_root_logging_restores_pipeline_handlers(tmp_path, isolated_root_logging):
    pipeline_handler = logging.StreamHandler()
    isolated_root_logging.addHandler(pipeline_handler)
    pipeline_handlers = list(isolated_root_logging.handlers)
    pipeline_level = logging.INFO

    # A step redirects logging to its own file ...
    task_files.change_log_file(tmp_path / "task.log", level=logging.DEBUG)
    assert pipeline_handler not in isolated_root_logging.handlers

    # ... and the pipeline restores its own handlers afterwards.
    task_files.reset_root_logging(pipeline_handlers, pipeline_level)

    assert isolated_root_logging.handlers == pipeline_handlers
    assert isolated_root_logging.level == pipeline_level
    assert not task_files._task_log_handler


def test_close_task_log_handler(tmp_path, isolated_root_logging):
    task_files.change_log_file(tmp_path / "task.log", level=logging.INFO)
    handler = task_files._task_log_handler[0]

    task_files.close_task_log_handler()

    assert handler not in isolated_root_logging.handlers
    assert not task_files._task_log_handler
    # Safe to call again when nothing is installed.
    task_files.close_task_log_handler()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
