"""Tests for shared config validation/utilities in base_config.py.

These cover the discriminated input-mode validation on ``CalculationInput``,
submission-config bounds, software-config application, logging setup and the
results-directory validation — all used by every page/pipeline command. No
external software is involved. Uses the ``glycine`` (CHEMBL773) compound
directory as a valid compound directory.
"""

import logging
import os

import pytest
from conftest import RESOURCES  # pylint: disable=import-error

from qm_atlas.command_line import base_config
from qm_atlas.command_line.base_config import CalculationInput, SubmissionConfig
from qm_atlas.software_environment import SOFTWARE_CONFIG_ENV_VAR

GLYCINE_DIR = RESOURCES / "cpd_dir_example" / "glycine"  # glycine, CHEMBL773


# ---------------------------------------------------------------------------
# CalculationInput validators
# ---------------------------------------------------------------------------


def test_compound_directories_single_path_is_coerced_to_list():
    ci = CalculationInput(results_directory=None, compound_directories=GLYCINE_DIR)
    assert ci.compound_directories == [GLYCINE_DIR.resolve()]


def test_compound_directories_nonexistent_rejected(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        CalculationInput(results_directory=None, compound_directories=[tmp_path / "missing"])


def test_compound_directories_file_rejected(tmp_path):
    a_file = tmp_path / "a.txt"
    a_file.write_text("")
    with pytest.raises(ValueError, match="not a directory"):
        CalculationInput(results_directory=None, compound_directories=[a_file])


def test_compound_directories_non_cpd_dir_rejected(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(ValueError, match="not a valid compound directory"):
        CalculationInput(results_directory=None, compound_directories=[plain])


def test_task_control_file_must_exist(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        CalculationInput(results_directory=None, task_control_file=tmp_path / "ctrl.json")


def test_task_control_file_must_be_json(tmp_path):
    not_json = tmp_path / "ctrl.yaml"
    not_json.write_text("{}")
    with pytest.raises(ValueError, match="must be JSON"):
        CalculationInput(results_directory=None, task_control_file=not_json)


def test_compound_directories_and_task_control_mutually_exclusive(tmp_path):
    ctrl = tmp_path / "ctrl.json"
    ctrl.write_text("{}")
    with pytest.raises(ValueError, match="mutually exclusive|Cannot provide both"):
        CalculationInput(
            results_directory=None,
            compound_directories=[GLYCINE_DIR],
            task_control_file=ctrl,
        )


def test_task_id_requires_task_control_file():
    with pytest.raises(ValueError, match="without task_control_file"):
        CalculationInput(results_directory=None, task_id=1)


def test_valid_task_control_input(tmp_path):
    ctrl = tmp_path / "ctrl.json"
    ctrl.write_text("{}")
    ci = CalculationInput(results_directory=None, task_control_file=ctrl, task_id=1)
    assert ci.task_id == 1
    assert ci.task_control_file == ctrl.resolve()


# ---------------------------------------------------------------------------
# SubmissionConfig
# ---------------------------------------------------------------------------


def test_submission_config_rejects_total_cores_below_per_task():
    with pytest.raises(ValueError, match="max_total_num_cores"):
        SubmissionConfig(cores_per_task=8, max_total_num_cores=4)


def test_submission_config_defaults_valid():
    cfg = SubmissionConfig()
    assert cfg.submit is False
    assert cfg.max_total_num_cores >= cfg.cores_per_task


# ---------------------------------------------------------------------------
# apply_software_config
# ---------------------------------------------------------------------------


def test_apply_software_config_none_is_noop(monkeypatch):
    called = {}
    monkeypatch.setattr(
        base_config.SOFTWARE_CONFIG,
        "reload_environment_manager",
        lambda p: called.setdefault("reloaded", p),
    )
    base_config.apply_software_config(None)
    assert "reloaded" not in called


def test_apply_software_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        base_config.apply_software_config(tmp_path / "nope.yaml")


def test_apply_software_config_sets_env_and_reloads(tmp_path, monkeypatch):
    cfg = tmp_path / "software.yaml"
    cfg.write_text("environments: {}\n")
    reloaded = {}
    monkeypatch.setattr(
        base_config.SOFTWARE_CONFIG,
        "reload_environment_manager",
        lambda p: reloaded.setdefault("path", p),
    )
    monkeypatch.delenv(SOFTWARE_CONFIG_ENV_VAR, raising=False)

    base_config.apply_software_config(cfg)

    assert os.environ[SOFTWARE_CONFIG_ENV_VAR] == str(cfg.resolve())
    assert reloaded["path"] == cfg.resolve()


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_root_logging():
    root = logging.root
    saved = list(root.handlers)
    saved_level = root.level
    for h in list(root.handlers):
        root.removeHandler(h)
    yield root
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in saved:
        root.addHandler(h)
    root.setLevel(saved_level)


def test_setup_logging_stream_handler_default(clean_root_logging):
    # basicConfig only acts when the root has no handlers; guarantee that here
    # (pytest's live-log plugin may otherwise leave one attached).
    clean_root_logging.handlers[:] = []
    base_config.setup_logging(verbose=True)
    assert any(isinstance(h, logging.StreamHandler) for h in clean_root_logging.handlers)
    assert clean_root_logging.level == logging.DEBUG


def test_setup_logging_file_handler(tmp_path, clean_root_logging):
    clean_root_logging.handlers[:] = []
    log_file = tmp_path / "run.log"
    base_config.setup_logging(verbose=False, log_file=log_file)
    file_handlers = [h for h in clean_root_logging.handlers if isinstance(h, logging.FileHandler)]
    assert file_handlers
    for h in file_handlers:
        h.close()


# ---------------------------------------------------------------------------
# validate_results_dir
# ---------------------------------------------------------------------------


def test_validate_results_dir_missing_parent(tmp_path):
    with pytest.raises(ValueError, match="Parent directory does not exist"):
        base_config.validate_results_dir(tmp_path / "no_parent" / "results")


def test_validate_results_dir_existing_file(tmp_path):
    a_file = tmp_path / "results"
    a_file.write_text("")
    with pytest.raises(ValueError, match="not a directory"):
        base_config.validate_results_dir(a_file)


def test_validate_results_dir_creates_when_missing(tmp_path):
    target = tmp_path / "results"
    base_config.validate_results_dir(target, create_if_missing=True)
    assert target.is_dir()


def test_validate_results_dir_missing_without_create(tmp_path):
    target = tmp_path / "results"
    with pytest.raises(ValueError, match="does not exist"):
        base_config.validate_results_dir(target, create_if_missing=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
