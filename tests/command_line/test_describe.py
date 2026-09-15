"""Tests for the ``describe`` command (pages/describe.py).

Pure-python: it only introspects the option registries and prints tables, so no
external software is involved.
"""

import pytest

from qm_atlas.command_line.pages import describe


def _first_entry(category: str) -> str:
    return next(iter(describe.CATEGORIES[category][1].keys()))


def test_no_args_lists_categories(capsys):
    describe.main([])
    out = capsys.readouterr().out
    for category in describe.CATEGORIES:
        assert category in out


def test_help_lists_categories(capsys):
    describe.main(["--help"])
    out = capsys.readouterr().out
    assert "calculators" in out


@pytest.mark.parametrize("category", list(describe.CATEGORIES.keys()))
def test_category_listing_shows_entries(category, capsys):
    describe.main([category])
    out = capsys.readouterr().out
    # Every registered entry name in the category is listed.
    for name in describe.CATEGORIES[category][1]:
        assert name in out


def test_entry_detail_shows_options(capsys):
    entry = _first_entry("calculators")
    describe.main(["calculators", entry])
    out = capsys.readouterr().out
    assert entry in out
    # jsonargparse renders options as ``--`` flags.
    assert "--" in out


def test_unknown_category_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        describe.main(["not_a_category"])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "Unknown category" in out


def test_unknown_entry_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        describe.main(["calculators", "not_a_real_calculator"])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "Unknown entry" in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
