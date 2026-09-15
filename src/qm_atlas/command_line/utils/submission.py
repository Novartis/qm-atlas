"""UGE cluster job submission configuration and utilities."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path

import yaml
from hpc_funcs.files import generate_name
from hpc_funcs.schedulers.uge.qsub import submit_script, write_script
from hpc_funcs.schedulers.uge.submission import DEFAULT_LOG_DIR, generate_script

from qm_atlas.command_line.base_config import (
    DEFAULT_LOG_DIR_NAME,
    BaseInterfaceConfig,
    SubmissionConfig,
)
from qm_atlas.command_line.file_interface import submission_files
from qm_atlas.command_line.file_interface.submission_files import WorkerPath
from qm_atlas.software_environment import SOFTWARE_CONFIG, SOFTWARE_CONFIG_ENV_VAR

_logger = logging.getLogger(__name__)


def parse_time(timestr):
    """Parse human readable time"""

    if "h" in timestr or "hour" in timestr or "hours" in timestr:

        timestr = timestr.replace("h", "").replace("our", "").replace("s", "")
        timestr = int(timestr)
        hours = timestr
        mins = 0

    elif "m" in timestr or "min" in timestr:
        timestr = timestr.replace("m", "").replace("in", "").replace("s", "")
        timestr = int(timestr)

        hours = timestr // 60
        mins = timestr % 60

    elif "d" in timestr or "day" in timestr or "days" in timestr:
        timestr = timestr.replace("d", "").replace("ay", "").replace("s", "")
        timestr = int(timestr)

        hours = timestr * 24
        mins = 0

    else:
        _logger.error(f"I don't understand the time format {timestr}")
        raise ValueError(f"Time {timestr} could not be parsed")

    return hours, mins


def submit_script_with_retry(
    script: str,
    max_retries: int = 3,
    wait_time_seconds: int = 60,
) -> str:
    """Submit a job script to the UGE cluster.

    Args:
        script: The bash script to submit.
        max_retries: Maximum number of retries for submission.
        wait_time_seconds: Time to wait between retries in seconds.

    Returns:
        Job ID returned by qsub.

    Raises:
        RuntimeError: If qsub submission fails after all retries.
    """

    for attempt in range(1, max_retries + 1):
        try:
            job_id = submit_script(script)
            _logger.info(f"Job submitted successfully with ID: {job_id}")
            return job_id
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            _logger.error(f"Attempt {attempt} - Job submission failed: {exc}")
            if attempt < max_retries:
                _logger.info(f"Retrying in {wait_time_seconds} seconds...")
                time.sleep(wait_time_seconds)

    _logger.error("Max retries reached. Job submission failed.")
    raise RuntimeError("Job submission failed after all retries")


def wait_for_jobs_using_hold_job(
    jobs: list[str],
    scr: Path,
    user_email: str | None = None,
    name: str = "UGEHoldJob",
    log_dir: Path | None = DEFAULT_LOG_DIR,
    generate_dirs: bool = True,
    update_interval: int = 1,
) -> Path:
    """
    Wait for by submitting a job using -hold-jid, which will only start when the other jobs
    are finished. This submitted job creates a file which is used to check if the job is finished.
    This avoids checking qstat, which puts some load on the server.
    """
    job_ids_joined = "__".join(jobs)
    filename = f"hold_job_{job_ids_joined}.finished"
    script_filename = f"hold_job_{job_ids_joined}.sh"
    finished_file = (scr / filename).resolve()
    command = f"touch {finished_file}"

    script_content = generate_script(
        cmd=command,
        name=name,
        cores=1,
        mem=1,
        hours=0,
        mins=1,
        log_dir=log_dir,
        user_email=user_email,
        hold_job_id=",".join(jobs),
        generate_dirs=generate_dirs,
    )
    script_path = write_script(content=script_content, directory=scr, filename=script_filename)
    job_id_hold_job = submit_script_with_retry(script_path)

    _logger.info(f"Submitted job {job_id_hold_job} to wait for jobs {jobs}")
    _logger.info(f"To manually skip waiting, create the file {finished_file}")

    while not finished_file.exists():
        time.sleep(update_interval)

    _logger.info(f"Jobs {jobs} have finished, continuing...")
    return finished_file


def compute_task_concurrency(max_total_num_cores: int, cores_per_task: int) -> int:
    """Number of tasks allowed to run at once given a total core budget."""
    return max_total_num_cores // cores_per_task


def build_submission_script(
    submission_config: SubmissionConfig,
    cmd: str,
    num_tasks: int,
    log_dir: Path,
    env_submission_config: dict | None = None,
) -> str:
    """Render the UGE task-array submission script from the submission config.

    Args:
        submission_config: The ``SubmissionConfig`` describing cores, memory, time, etc.
        cmd: The command executed by each array task.
        num_tasks: Total number of tasks in the array (task-array upper bound).
        log_dir: Directory for UGE job logs.
        env_submission_config: Extra directives from the environment manager.

    Returns:
        The rendered submission script as a string.
    """
    task_concurrent = compute_task_concurrency(
        submission_config.max_total_num_cores, submission_config.cores_per_task
    )
    hours, mins = parse_time(submission_config.max_time)
    user_email = None if submission_config.wait else submission_config.user_email

    return generate_script(
        cmd=cmd,
        name=submission_config.name,
        cores=submission_config.cores_per_task,
        mem=submission_config.mem_per_core,
        hours=hours,
        mins=mins,
        log_dir=log_dir,
        user_email=user_email,
        task_stop=num_tasks,
        task_concurrent=task_concurrent,
        **(env_submission_config or {}),
    )


def submit(
    worker_paths: list[WorkerPath],
    config: BaseInterfaceConfig,
    command: str,
) -> str | None:

    new_config = config.model_copy()
    if new_config.input.results_directory is None:
        raise ValueError(
            "results_directory must be specified in input configuration for submission"
        )

    # Generate task control file
    task_uid = generate_name()
    task_scratch_dir = submission_files.get_submission_scratch_dir(config.input.results_directory)
    task_control_file = task_scratch_dir / f"task_control_{task_uid}.json"
    num_tasks = submission_files.write_task_control(
        worker_paths, task_control_file, config.submission_config.jobs_per_task
    )

    if new_config.input.compound_directories is not None:
        new_config.input.compound_directories = None

    new_config.input.task_control_file = task_control_file
    new_config.submission_config.submit = False
    new_config.n_cores = new_config.submission_config.cores_per_task

    # write config yaml file to same scratch dir for use in job script
    config_file = (task_scratch_dir / f"config_{task_uid}.yaml").expanduser().resolve()
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(new_config.model_dump(mode="json"), f)

    submission_command = f"{command} --config {config_file} --input.task_id $SGE_TASK_ID"
    log_dir = (new_config.input.results_directory / DEFAULT_LOG_DIR_NAME).expanduser().resolve()

    # Export software config env var in submission script if specified
    software_config_env_var_value = os.getenv(SOFTWARE_CONFIG_ENV_VAR, None)
    if software_config_env_var_value is not None:
        export_cmd = f"export {SOFTWARE_CONFIG_ENV_VAR}={software_config_env_var_value}"
        submission_command = f"{export_cmd}\n{submission_command}"

    environment_manager = SOFTWARE_CONFIG.get_environment_manager()
    env_submission_config = environment_manager.get_submission_config().copy()
    command_prepend = env_submission_config.pop("command_prepend", None)
    if command_prepend is not None:
        submission_command = f"{command_prepend} \n{submission_command}"

    submission_script = build_submission_script(
        config.submission_config,
        cmd=submission_command,
        num_tasks=num_tasks,
        log_dir=log_dir,
        env_submission_config=env_submission_config,
    )
    script_path = write_script(
        content=submission_script, directory=task_scratch_dir, filename=f"run_{task_uid}.sh"
    )
    job_id = submit_script_with_retry(script_path)
    _logger.info(f"Job {config.submission_config.name} submitted successfully with ID: {job_id}")

    if new_config.submission_config.wait:
        _logger.info("Waiting for job to complete...")
        wait_for_jobs_using_hold_job(
            [job_id],
            task_scratch_dir,
            user_email=new_config.submission_config.user_email,
            log_dir=log_dir,
            name=f"{config.submission_config.name}_hold",
        )

        _logger.info(f"Job {config.submission_config.name} finished.")

    return job_id
