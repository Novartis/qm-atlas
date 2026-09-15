"""Tests for the shared page ``main()`` / ``submit()`` routing (point 4).

Every workflow page follows the same structure: ``main()`` routes to ``submit``
(cluster) or ``run_local`` based on ``submission_config.submit``, and ``submit()``
extracts worker paths and hands them to ``submission.submit``. This routing is
pure-python; the software-dependent ``run_local`` bodies are covered elsewhere by
the ``require_software`` suite. ``fast_conformers`` and ``conformer_properties``
are used as representatives of the shared pattern.
"""

import pytest

from qm_atlas.command_line.base_config import CalculationInput
from qm_atlas.command_line.pages import conformer_properties as cp_page
from qm_atlas.command_line.pages import fast_conformers as fc_page


@pytest.fixture
def results_dir(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    return d


def _fc_config(results_dir, submit):
    return fc_page.FastConformersInterfaceConfig(
        input=CalculationInput(results_directory=results_dir),
        submission_config=fc_page.FastConformersSubmissionConfig(submit=submit),
    )


# ---------------------------------------------------------------------------
# main(): submit vs run_local routing
# ---------------------------------------------------------------------------


def test_main_runs_locally_when_not_submitting(results_dir, monkeypatch):
    monkeypatch.setattr(fc_page, "apply_software_config", lambda _p: None)
    monkeypatch.setattr(fc_page, "setup_logging", lambda **_k: None)
    routed = {}
    monkeypatch.setattr(fc_page, "run_local", lambda cfg: routed.setdefault("local", cfg))
    monkeypatch.setattr(fc_page, "submit", lambda cfg: routed.setdefault("submit", cfg))

    fc_page.main(config=_fc_config(results_dir, submit=False))

    assert "local" in routed and "submit" not in routed


def test_main_submits_when_configured(results_dir, monkeypatch):
    monkeypatch.setattr(fc_page, "apply_software_config", lambda _p: None)
    monkeypatch.setattr(fc_page, "setup_logging", lambda **_k: None)
    routed = {}
    monkeypatch.setattr(fc_page, "run_local", lambda cfg: routed.setdefault("local", cfg))
    monkeypatch.setattr(fc_page, "submit", lambda cfg: routed.setdefault("submit", cfg))

    fc_page.main(config=_fc_config(results_dir, submit=True))

    assert "submit" in routed and "local" not in routed


# ---------------------------------------------------------------------------
# submit(): worker-path extraction + delegation + empty guard
# ---------------------------------------------------------------------------


def test_submit_raises_when_no_molecules(results_dir, monkeypatch):
    monkeypatch.setattr(
        fc_page, "extract_worker_paths", lambda _inp, workflow_type, skip_completed=False: []
    )
    with pytest.raises(ValueError, match="No molecules found"):
        fc_page.submit(_fc_config(results_dir, submit=True))


def test_submit_delegates_to_submission_submit(results_dir, monkeypatch):
    sentinel_paths = [("cpd", "sdf", "log")]
    monkeypatch.setattr(
        fc_page,
        "extract_worker_paths",
        lambda _inp, workflow_type, skip_completed=False: sentinel_paths,
    )
    captured = {}
    monkeypatch.setattr(
        fc_page.submission,
        "submit",
        lambda worker_paths, config, command: captured.update(
            worker_paths=worker_paths, command=command
        ),
    )

    fc_page.submit(_fc_config(results_dir, submit=True))

    assert captured["worker_paths"] is sentinel_paths
    assert captured["command"] == fc_page.COMMAND


def test_conformer_properties_submit_uses_conformer_properties_workflow(results_dir, monkeypatch):
    """The property page must route on the ``conformer_properties`` workflow type."""
    seen = {}

    def fake_extract(_inp, workflow_type):
        seen["workflow_type"] = workflow_type
        return [("cpd", "sdf", "log")]

    monkeypatch.setattr(cp_page, "extract_worker_paths", fake_extract)
    monkeypatch.setattr(cp_page.submission, "submit", lambda worker_paths, config, command: None)

    config = cp_page.ConformerPropertyInterfaceConfig(
        input=CalculationInput(results_directory=results_dir),
        submission_config=cp_page.ConformerPropertySubmissionConfig(submit=True),
    )
    cp_page.submit(config)

    assert seen["workflow_type"] == "conformer_properties"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
