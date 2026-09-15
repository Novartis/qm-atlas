import pytest

from qm_atlas.command_line.utils import submission


def test_parse_time():
    hours, mins = submission.parse_time("8hours")
    assert hours == 8
    assert mins == 0

    hours, mins = submission.parse_time("1hour")
    assert hours == 1
    assert mins == 0

    hours, mins = submission.parse_time("120min")
    assert hours == 2
    assert mins == 0

    hours, mins = submission.parse_time("30min")
    assert hours == 0
    assert mins == 30

    hours, mins = submission.parse_time("2days")
    assert hours == 48
    assert mins == 0

    hours, mins = submission.parse_time("1day")
    assert hours == 24
    assert mins == 0

    # check wrongly formatted input
    with pytest.raises(ValueError):
        hours, mins = submission.parse_time("30mni")

    with pytest.raises(ValueError):
        hours, mins = submission.parse_time("30unit")
