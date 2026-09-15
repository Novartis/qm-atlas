"""Tests for the Macromodel log-diagnosis helper (macromodel/utils.py).

``log_macromodel_errors`` scans scratch-directory log files for known failure
indicators. It reads plain text only, so no Macromodel/Schrodinger installation
is needed here.
"""

import pytest

from qm_atlas.wrappers.macromodel import utils as macromodel_utils


def test_no_log_files_reports_missing(tmp_path, caplog):
    with caplog.at_level("ERROR"):
        macromodel_utils.log_macromodel_errors(tmp_path)

    assert any("No Macromodel log file found" in rec.message for rec in caplog.records)


def test_storage_exceeded_gives_actionable_hint(tmp_path, caplog):
    log_file = tmp_path / "job.log"
    log_file.write_text(
        "Starting search\nMaximum conformation storage exceeded\nAborting\n",
        encoding="utf-8",
    )

    with caplog.at_level("ERROR"):
        macromodel_utils.log_macromodel_errors(tmp_path)

    messages = "\n".join(rec.message for rec in caplog.records)
    # The raw indicator line is echoed ...
    assert "Maximum conformation storage exceeded" in messages
    # ... together with the guidance to raise max_conformers.
    assert "max_conformers" in messages


def test_generic_error_indicator_is_logged(tmp_path, caplog):
    log_file = tmp_path / "job.log"
    log_file.write_text("FATAL: license server unreachable\n", encoding="utf-8")

    with caplog.at_level("ERROR"):
        macromodel_utils.log_macromodel_errors(tmp_path)

    messages = "\n".join(rec.message for rec in caplog.records)
    assert "license server unreachable" in messages
    # A specific indicator was found, so the generic fallback must NOT fire.
    assert "Could not identify a specific error" not in messages


def test_no_indicator_reports_unidentified(tmp_path, caplog):
    log_file = tmp_path / "job.log"
    log_file.write_text("Everything went fine.\nDone.\n", encoding="utf-8")

    with caplog.at_level("ERROR"):
        macromodel_utils.log_macromodel_errors(tmp_path)

    assert any("Could not identify a specific error" in rec.message for rec in caplog.records)


def test_unreadable_log_file_is_skipped(tmp_path, caplog):
    # A directory that matches the *.log glob raises OSError on open() and must
    # be skipped without aborting the scan of the remaining files.
    (tmp_path / "broken.log").mkdir()
    (tmp_path / "job.log").write_text("FATAL: boom\n", encoding="utf-8")

    with caplog.at_level("ERROR"):
        macromodel_utils.log_macromodel_errors(tmp_path)

    messages = "\n".join(rec.message for rec in caplog.records)
    assert "boom" in messages


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
