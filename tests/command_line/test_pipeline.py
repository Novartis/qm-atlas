"""Tests for the pipeline orchestrator (command_line/pipeline.py).

The pipeline is central shared infrastructure: it deserializes each step to a
typed config, injects top-level paths, forces ``wait`` on submitted steps, runs
the steps in order, and restores its own logging after each locally-run step.
These tests exercise that logic with the step ``main`` functions mocked, so no
external software runs.
"""

import logging

import pytest
import yaml
from pydantic import BaseModel, ConfigDict

from qm_atlas.command_line import pipeline
from qm_atlas.command_line.base_config import SubmissionConfig
from qm_atlas.command_line.file_interface import task_files
from qm_atlas.command_line.pages import collect as collect_page
from qm_atlas.command_line.pages import fast_conformers as fast_conformers_page


@pytest.fixture
def restore_logging():
    """Snapshot and restore global logging state mutated by run_pipeline."""
    root = logging.root
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_task = list(task_files._task_log_handler)
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in saved_handlers:
        root.addHandler(handler)
    root.setLevel(saved_level)
    task_files._task_log_handler[:] = saved_task


# ===========================================================================
# PipelineConfig.resolve_steps: deserialization + path injection + wait
# ===========================================================================


def test_steps_deserialized_to_typed_configs(tmp_path):
    config = pipeline.PipelineConfig(
        results_directory=tmp_path,
        steps=[{"command": "collect"}],
    )
    step = config.steps[0]
    assert isinstance(step, collect_page.CollectInterfaceConfig)
    # results_directory is injected into the interface command's input block.
    assert step.input.results_directory == tmp_path.resolve()


def test_direct_input_command_gets_results_and_input_file(tmp_path):
    sdf = tmp_path / "mols.sdf"
    sdf.write_text("")
    config = pipeline.PipelineConfig(
        results_directory=tmp_path,
        input_file=[sdf],
        steps=[{"command": "read_input"}],
    )
    step = config.steps[0]
    assert step.results_directory == tmp_path.resolve()
    assert step.input_file == [sdf.resolve()]


def test_add_reference_conformers_uses_reference_input_file(tmp_path):
    inp = tmp_path / "in.sdf"
    ref = tmp_path / "ref.sdf"
    inp.write_text("")
    ref.write_text("")
    config = pipeline.PipelineConfig(
        results_directory=tmp_path,
        input_file=[inp],
        reference_input_file=[ref],
        steps=[{"command": "add_reference_conformers"}],
    )
    # The reference file wins over the plain input file for this command.
    assert config.steps[0].input_file == [ref.resolve()]


def test_explicit_step_value_not_overwritten(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    # Make it a results dir sibling so validation passes.
    config = pipeline.PipelineConfig(
        results_directory=tmp_path,
        steps=[{"command": "collect", "input": {"results_directory": str(other)}}],
    )
    assert config.steps[0].input.results_directory == other.resolve()


def test_submit_forces_wait_true(tmp_path):
    config = pipeline.PipelineConfig(
        results_directory=tmp_path,
        steps=[{"command": "fast_conformers", "submission_config": {"submit": True}}],
    )
    step = config.steps[0]
    assert isinstance(step, fast_conformers_page.FastConformersInterfaceConfig)
    assert step.submission_config.submit is True
    assert step.submission_config.wait is True


def test_unknown_command_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown pipeline command"):
        pipeline.PipelineConfig(results_directory=tmp_path, steps=[{"command": "does_not_exist"}])


def test_empty_steps_rejected(tmp_path):
    with pytest.raises(ValueError):
        pipeline.PipelineConfig(results_directory=tmp_path, steps=[])


def test_input_file_validation_rejects_missing(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        pipeline.PipelineConfig(
            results_directory=tmp_path,
            input_file=[tmp_path / "nope.sdf"],
            steps=[{"command": "read_input"}],
        )


def test_input_file_validation_rejects_bad_suffix(tmp_path):
    bad = tmp_path / "data.txt"
    bad.write_text("")
    with pytest.raises(ValueError, match="Unsupported input file type"):
        pipeline.PipelineConfig(
            results_directory=tmp_path,
            input_file=[bad],
            steps=[{"command": "read_input"}],
        )


# ===========================================================================
# run_pipeline: ordered dispatch + logging restoration
# ===========================================================================


class _PermissiveStep(BaseModel):
    model_config = ConfigDict(extra="allow")
    command: str


def test_run_pipeline_dispatches_steps_in_order(tmp_path, monkeypatch, restore_logging):
    calls: list[str] = []

    def make_recorder(name):
        def _main(config=None):
            calls.append(name)

        return _main

    fake_registry = {
        "stepA": (make_recorder("stepA"), _PermissiveStep),
        "stepB": (make_recorder("stepB"), _PermissiveStep),
    }
    monkeypatch.setattr(pipeline, "STEP_REGISTRY", fake_registry)

    reset_calls: list[int] = []
    real_reset = task_files.reset_root_logging

    def spy_reset(handlers, level=None):
        reset_calls.append(1)
        real_reset(handlers, level)

    monkeypatch.setattr(pipeline.task_files, "reset_root_logging", spy_reset)

    config = pipeline.PipelineConfig(
        results_directory=tmp_path,
        steps=[{"command": "stepA"}, {"command": "stepB"}],
    )
    pipeline.run_pipeline(config)

    assert calls == ["stepA", "stepB"]
    # Logging is reinstated after every step.
    assert len(reset_calls) == 2
    # A pipeline log file was created in the results directory.
    assert list(tmp_path.glob("pipeline_*.log"))


# ===========================================================================
# main(): submit vs run routing
# ===========================================================================


def test_main_routes_to_run_pipeline_when_not_submitting(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "apply_software_config", lambda _p: None)
    monkeypatch.setattr(pipeline, "setup_logging", lambda **_k: None)
    ran = {}
    monkeypatch.setattr(pipeline, "run_pipeline", lambda cfg: ran.setdefault("run", cfg))
    monkeypatch.setattr(pipeline, "submit", lambda cfg: ran.setdefault("submit", cfg))

    config = pipeline.PipelineConfig(
        results_directory=tmp_path,
        submission_config=SubmissionConfig(submit=False),
        steps=[{"command": "collect"}],
    )
    pipeline.main(config=config)

    assert "run" in ran and "submit" not in ran


def test_main_routes_to_submit_when_submitting(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "apply_software_config", lambda _p: None)
    monkeypatch.setattr(pipeline, "setup_logging", lambda **_k: None)
    ran = {}
    monkeypatch.setattr(pipeline, "run_pipeline", lambda cfg: ran.setdefault("run", cfg))
    monkeypatch.setattr(pipeline, "submit", lambda cfg: ran.setdefault("submit", cfg))

    config = pipeline.PipelineConfig(
        results_directory=tmp_path,
        submission_config=SubmissionConfig(submit=True),
        steps=[{"command": "collect"}],
    )
    pipeline.main(config=config)

    assert "submit" in ran and "run" not in ran


# ===========================================================================
# submit(): resolved-config serialization for cluster re-invocation
# ===========================================================================


class _FakeEnvManager:
    def get_submission_config(self):
        return {}


def test_submit_serializes_resolved_config_and_submits(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "generate_name", lambda: "uid123")
    monkeypatch.setattr(pipeline, "generate_script", lambda **kwargs: "#!/bin/bash\n")
    written = {}

    def fake_write_script(content, directory, filename):
        path = directory / filename
        path.write_text(content)
        written["script"] = path
        return path

    monkeypatch.setattr(pipeline, "write_script", fake_write_script)
    monkeypatch.setattr(pipeline, "submit_script_with_retry", lambda script: "JOB-42")
    monkeypatch.setattr(
        pipeline.SOFTWARE_CONFIG, "get_environment_manager", lambda: _FakeEnvManager()
    )

    config = pipeline.PipelineConfig(
        results_directory=tmp_path,
        submission_config=SubmissionConfig(submit=True, wait=False, max_time="1hour"),
        steps=[{"command": "collect"}],
    )

    pipeline.submit(config)

    # The resolved config was dumped to YAML in the submission scratch dir,
    # with Paths serialized to strings and no submission_config block.
    config_files = list((tmp_path / "qm_atlas_submissions").glob("pipeline_config_*.yaml"))
    assert len(config_files) == 1
    dumped = yaml.safe_load(config_files[0].read_text())
    assert "submission_config" not in dumped
    assert dumped["steps"][0]["command"] == "collect"
    assert written["script"].is_file()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
