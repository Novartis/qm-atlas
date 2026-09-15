"""Tests for the matomo usage-tracking helpers (matomo.py).

Network access is fully mocked: no request ever leaves the process. The multi
processing helpers are exercised against lightweight fakes so the tests stay
fast and deterministic.
"""

from multiprocessing.context import TimeoutError as MPTimeoutError

import pytest

from qm_atlas.command_line.utils import matomo


def _set_all_env(monkeypatch, value="set"):
    for attr in ("MATOMO_URL", "MATOMO_SITE_ID", "MATOMO_TIMEOUT", "USER", "HOSTNAME"):
        monkeypatch.setattr(matomo, attr, value, raising=False)


def test_health_check_true_when_all_configured(monkeypatch):
    _set_all_env(monkeypatch)
    assert matomo.health_check() is True


@pytest.mark.parametrize(
    "missing",
    ["MATOMO_URL", "MATOMO_SITE_ID", "MATOMO_TIMEOUT", "USER", "HOSTNAME"],
)
def test_health_check_false_when_any_missing(monkeypatch, missing):
    _set_all_env(monkeypatch)
    monkeypatch.setattr(matomo, missing, None, raising=False)
    assert matomo.health_check() is False


def test_register_event_posts_expected_params(monkeypatch):
    monkeypatch.setattr(matomo, "MATOMO_URL", "http://matomo.example/track", raising=False)
    monkeypatch.setattr(matomo, "MATOMO_SITE_ID", "7", raising=False)
    monkeypatch.setattr(matomo, "MATOMO_TIMEOUT", "5", raising=False)
    monkeypatch.setattr(matomo, "USER", "tester", raising=False)
    monkeypatch.setattr(matomo, "HOSTNAME", "node01", raising=False)

    captured = {}

    def fake_post(url, params, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return "ok"

    monkeypatch.setattr(matomo.requests, "post", fake_post)

    result = matomo.register_event("rescoss_properties", "run --config x.yaml")

    assert result == "ok"
    assert captured["url"] == "http://matomo.example/track"
    assert captured["timeout"] == 5
    params = captured["params"]
    assert params["idsite"] == "7"
    assert params["uid"] == "tester"
    assert params["action_name"] == "run --config x.yaml"
    assert params["url"] == "http://qm_atlas/rescoss_properties"
    assert params["urlref"] == "http://node01"


def test_close_tracking_success():
    class FakeResult:
        def get(self, timeout):  # noqa: ARG002 - mirrors AsyncResult.get signature
            return "response"

    class FakePool:
        def __init__(self):
            self.terminated = False

        def terminate(self):
            self.terminated = True

    pool = FakePool()
    matomo.close_tracking(FakeResult(), pool)
    assert pool.terminated is True


def test_close_tracking_handles_timeout():
    class FakeResult:
        def get(self, timeout):  # noqa: ARG002 - mirrors AsyncResult.get signature
            raise MPTimeoutError()

    class FakePool:
        def __init__(self):
            self.terminated = False

        def terminate(self):
            self.terminated = True

    pool = FakePool()
    # A timeout must be swallowed and the pool still terminated.
    matomo.close_tracking(FakeResult(), pool)
    assert pool.terminated is True


def test_register_event_subprocess_dispatches(monkeypatch):
    calls = {}

    class FakeAsyncResult:
        pass

    class FakePool:
        def __init__(self, processes):
            calls["processes"] = processes

        def apply_async(self, func, args):
            calls["func"] = func
            calls["args"] = args
            return FakeAsyncResult()

    monkeypatch.setattr(matomo, "Pool", FakePool)

    result, pool = matomo.register_event_subprocess("collect", "run")

    assert isinstance(result, FakeAsyncResult)
    assert isinstance(pool, FakePool)
    assert calls["processes"] == 1
    assert calls["func"] is matomo.register_event
    assert calls["args"] == ("collect", "run")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
