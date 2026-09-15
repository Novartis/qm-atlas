"""Task file management for UGE cluster job submission.

This module provides utilities for writing and reading task control files that coordinate
the distribution of work across a task array. Task control files map UGE task IDs to the
specific work items (e.g., molecules, calculations) that should be processed by each task.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any

from qm_atlas.command_line.file_interface import compound_dir

_logger = logging.getLogger(__name__)

SCRATCH_DIR_NAME = "qm_atlas_submissions"


WorkerPath = tuple[compound_dir.CpdDir, Path, Path]


def write_task_control(
    worker_paths: list[WorkerPath], control_file: Path, jobs_per_task: int = 1
) -> int:

    control_file.parent.mkdir(exist_ok=True, parents=True)

    worker_paths_str = [
        (str(cpd_dir), str(sdf_file.resolve()), str(log_file.resolve()))
        for cpd_dir, sdf_file, log_file in worker_paths
    ]

    num_tasks = math.ceil(len(worker_paths) / jobs_per_task)
    job_control_dict = dict()

    for task_id in range(1, num_tasks + 1):  # 1-based indexing for task ids

        start_idx = (task_id - 1) * jobs_per_task
        stop_idx = task_id * jobs_per_task
        paths_subset = worker_paths_str[start_idx:stop_idx]
        job_control_dict[task_id] = paths_subset

    with open(control_file, "w", encoding="utf-8") as json_ctrl:
        json.dump(job_control_dict, json_ctrl)

    return num_tasks


def read_task_control(task_control_file: Path) -> dict:
    with open(task_control_file, "r", encoding="utf-8") as tcf:
        task_control_dict = json.load(tcf)
    task_control_dict = {
        int(task_id): [
            (compound_dir.create_cpd_dir(Path(cpd_dir)), Path(sdf_file), Path(log_file))
            for cpd_dir, sdf_file, log_file in worker_paths
        ]
        for task_id, worker_paths in task_control_dict.items()
    }
    return task_control_dict


def get_task_work_items(control_file: Path, task_id: int) -> list[Any]:
    """Retrieve the work items for a specific task ID from a task control file.

    Args:
        control_file: Path to the task control file.
        task_id: The UGE task ID (1-based indexing).

    Returns:
        List of work items for the given task.

    Raises:
        FileNotFoundError: If the control file doesn't exist.
        KeyError: If the task_id is not in the control file.
    """
    task_data = read_task_control(control_file)

    if task_id not in task_data:
        raise ValueError(
            f"Task ID {task_id} not found in task control file. "
            f"Available task IDs: {sorted(task_data.keys())}"
        )

    return task_data[task_id]


def get_submission_scratch_dir(results_dir: Path) -> Path:
    """Get the scratch directory for submission task control files.

    Args:
        results_dir: The main results directory.

    Returns:
        Path to the submission scratch directory.
    """
    return results_dir / SCRATCH_DIR_NAME
