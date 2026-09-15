import logging
from pathlib import Path

_logger = logging.getLogger(__name__)

SCRATCH_DIR_NAME = "qm_atlas_commands"

# Handler installed by the most recent change_log_file() call, wrapped in a
# single-element list so it can be mutated from helper functions without a
# module-level ``global`` statement. Tracked so that the next redirect (or a
# reset_root_logging() call) can close it and avoid leaking open file handles
# when many tasks are processed in one local run.
_task_log_handler: list[logging.Handler] = []


def change_log_file(new_log_file: Path, **logging_options):
    """Redirect the root logger to a per-task log file.

    Detaches every handler currently on the root logger so that task output is
    written *only* to ``new_log_file``. This mirrors submitted jobs, where each
    task runs in its own process and logs to its own file, keeping any
    surrounding pipeline log clean of per-task chatter.

    Only the task handler installed by a previous ``change_log_file()`` call is
    closed here; handlers installed elsewhere (e.g. a pipeline log handler) are
    left open and merely detached, so the caller can restore them afterwards
    with :func:`reset_root_logging`.

    Args:
        new_log_file (Path): The path for the new log file
    """
    root = logging.root
    previous = _task_log_handler[0] if _task_log_handler else None
    for handler in list(root.handlers):
        root.removeHandler(handler)
        if handler is previous:
            handler.close()

    logging.basicConfig(filename=new_log_file, **logging_options)

    _task_log_handler.clear()
    if root.handlers:
        _task_log_handler.append(root.handlers[-1])


def reset_root_logging(handlers: list[logging.Handler], level: int | None = None) -> None:
    """Restore ``handlers`` as the only handlers on the root logger.

    Used by long-lived callers (such as the pipeline runner) to reinstate their
    own logging after a step has redirected the root logger to per-task files
    via :func:`change_log_file`. Any handler currently attached that is not in
    ``handlers`` (e.g. a leftover task handler) is closed and removed first.

    Args:
        handlers: The handlers to reinstate on the root logger.
        level: Optional log level to restore on the root logger.
    """
    root = logging.root
    for handler in list(root.handlers):
        root.removeHandler(handler)
        if handler not in handlers:
            handler.close()

    for handler in handlers:
        root.addHandler(handler)

    if level is not None:
        root.setLevel(level)

    _task_log_handler.clear()


def close_task_log_handler() -> None:
    """Close and detach the per-task log handler installed by change_log_file().

    Standalone ``run_local()`` loops redirect the root logger to per-task files
    but, unlike the pipeline runner, never call :func:`reset_root_logging`.
    Without this the final task ``FileHandler`` stays open until process exit,
    which on NFS leaves ``.nfsXXXX`` silly-rename files behind and blocks cleanup
    of the log directory. Call this in a ``finally`` after such a loop.
    """
    if not _task_log_handler:
        return

    handler = _task_log_handler[0]
    logging.root.removeHandler(handler)
    handler.close()
    _task_log_handler.clear()
