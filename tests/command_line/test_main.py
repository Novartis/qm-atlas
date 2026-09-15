import logging

import pytest

import qm_atlas.__main__ as qm_atlasmain


@pytest.fixture
def restore_logging():
    """Re-enable loggers and restore root handlers after main() mutates them.

    ``qm_atlasmain.main`` calls ``logging.basicConfig`` and disables every
    non-``qm_atlas`` logger, so we snapshot/restore to avoid leaking that state
    into other test modules.
    """
    root = logging.root
    saved_handlers = list(root.handlers)
    saved_level = root.level
    yield
    for _name, obj in logging.Logger.manager.loggerDict.items():
        if hasattr(obj, "disabled"):
            obj.disabled = False
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in saved_handlers:
        root.addHandler(handler)
    root.setLevel(saved_level)


def test_args_valid():

    args = "qm_atlas --debug --verbose no_module"
    args = args.split()

    return_code = qm_atlasmain.main(args)
    assert return_code != 0


def test_args_error():

    args = "qm_atlas no_module"
    args = args.split()

    return_code = qm_atlasmain.main(args)

    assert return_code != 0


def test_args_empty():

    args = ""
    args = args.split()

    qm_atlasmain.main(args) != 0


def test_args_help():

    args = "qm_atlas help"
    args = args.split()

    return_code = qm_atlasmain.main(args)
    assert return_code == 0


# ---------------------------------------------------------------------------
# Command dispatch
# ---------------------------------------------------------------------------


def test_version_flag_exits(capsys):
    with pytest.raises(SystemExit):
        qm_atlasmain.main(["qm_atlas", "--version"])
    out = capsys.readouterr().out
    assert out.strip()  # the version string is printed


def test_unknown_command_returns_4(restore_logging):
    assert qm_atlasmain.main(["qm_atlas", "totally_unknown"]) == 4


def test_valid_command_is_dispatched(monkeypatch, restore_logging):
    received = {}
    monkeypatch.setattr(
        qm_atlasmain.describe, "main", lambda argv: received.setdefault("argv", argv)
    )

    exit_code = qm_atlasmain.main(["qm_atlas", "describe", "calculators", "xtb_single_point"])

    assert exit_code == 0
    assert received["argv"] == ["calculators", "xtb_single_point"]


def test_command_systemexit_maps_to_exit_code_1(monkeypatch, restore_logging):
    def _boom(argv):
        raise SystemExit(2)

    monkeypatch.setattr(qm_atlasmain.read_input, "main", _boom)

    assert qm_atlasmain.main(["qm_atlas", "read_input"]) == 1


def test_debug_flag_consumed_and_dispatches(monkeypatch, restore_logging):
    called = {}
    monkeypatch.setattr(
        qm_atlasmain.describe, "main", lambda argv: called.setdefault("argv", argv)
    )

    exit_code = qm_atlasmain.main(["qm_atlas", "--debug", "describe"])

    assert exit_code == 0
    # The global flag is stripped before the sub-command sees its arguments.
    assert called["argv"] == []


def test_silent_flag_consumed_and_dispatches(monkeypatch, restore_logging):
    called = {}
    monkeypatch.setattr(
        qm_atlasmain.describe, "main", lambda argv: called.setdefault("argv", argv)
    )

    exit_code = qm_atlasmain.main(["qm_atlas", "--silent", "describe"])

    assert exit_code == 0
    assert called["argv"] == []


# ---------------------------------------------------------------------------
# Matomo usage tracking
# ---------------------------------------------------------------------------


def test_matomo_tracking_registers_and_closes(monkeypatch, restore_logging):
    monkeypatch.setattr(qm_atlasmain.describe, "main", lambda argv: None)
    monkeypatch.setattr(qm_atlasmain.matomo, "health_check", lambda: True)

    registered = {}

    def fake_register(module, args):
        registered["call"] = (module, args)
        return "events", "pool"

    monkeypatch.setattr(qm_atlasmain.matomo, "register_event_subprocess", fake_register)
    closed = {}
    monkeypatch.setattr(
        qm_atlasmain.matomo,
        "close_tracking",
        lambda events, pool: closed.setdefault("call", (events, pool)),
    )

    qm_atlasmain.main(["qm_atlas", "describe", "calculators"])

    assert registered["call"] == ("describe", "calculators")
    assert closed["call"] == ("events", "pool")


def test_incognito_disables_tracking(monkeypatch, restore_logging):
    monkeypatch.setattr(qm_atlasmain.describe, "main", lambda argv: None)
    monkeypatch.setattr(qm_atlasmain.matomo, "health_check", lambda: True)

    registered = {}
    monkeypatch.setattr(
        qm_atlasmain.matomo,
        "register_event_subprocess",
        lambda module, args: registered.setdefault("call", True),
    )

    exit_code = qm_atlasmain.main(["qm_atlas", "--incognito", "describe"])

    assert exit_code == 0
    assert "call" not in registered


def test_matomo_failure_is_swallowed(monkeypatch, restore_logging):
    monkeypatch.setattr(qm_atlasmain.describe, "main", lambda argv: None)

    def _explode():
        raise RuntimeError("matomo unreachable")

    monkeypatch.setattr(qm_atlasmain.matomo, "health_check", _explode)

    # A tracking failure must never break the actual command.
    assert qm_atlasmain.main(["qm_atlas", "describe"]) == 0
