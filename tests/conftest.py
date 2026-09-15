import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from qm_atlas.software_environment import SOFTWARE_CONFIG


def _get_repo_root_dir() -> Path:
    """
    :return: path to the qm_atlas root directory.
    `tests/` should be in the same folder and this file should be in the root of `tests/`.
    """
    return Path(__file__).parent.parent


ROOT_DIR = _get_repo_root_dir()
RESOURCES = ROOT_DIR / "tests/resources"


def pytest_addoption(parser):
    """Add custom command-line options for pytest."""
    parser.addoption(
        "--software-config",
        action="store",
        default=None,
        help="Path to software configuration YAML file for tests",
    )


def pytest_configure(config):
    """
    Handle pytest configuration before test collection.

    If a custom software config file is provided via --software-config,
    reload the SOFTWARE_CONFIG with that file.
    """
    config_file = config.getoption("--software-config")

    if config_file is not None:
        config_path = Path(config_file)
        if not config_path.is_file():
            raise FileNotFoundError(f"Software config file not found: {config_path}")

        # Reload SOFTWARE_CONFIG with the specified configuration file
        SOFTWARE_CONFIG.reload_environment_manager(config_path)


@pytest.fixture(scope="module")
def home_tmp_path():
    """Make a temporary directory in home

    Home is a globally mounted directory and therefore safe for UGE usage.
    """
    user_homedir = Path.home()
    random_name = next(tempfile._get_candidate_names())

    tmp_path = user_homedir / "tmp" / f"pytest_{random_name}"
    tmp_path.mkdir(parents=True, exist_ok=True)

    yield tmp_path

    # Force clean
    shutil.rmtree(tmp_path)
    assert not tmp_path.is_dir()


@pytest.fixture(scope="module")
def local_resources():
    return RESOURCES


def _assert_preserved(expected, actual, path: str = "") -> None:
    """Assert every value set in ``expected`` appears unchanged in ``actual``.

    ``actual`` may contain additional keys (materialized defaults); only the keys
    present in ``expected`` are checked, recursively.
    """
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path or '<root>'}: expected a mapping, got {actual!r}"
        for key, value in expected.items():
            assert key in actual, f"{path}.{key}: missing from reloaded config"
            _assert_preserved(value, actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected a list, got {actual!r}"
        assert len(actual) == len(
            expected
        ), f"{path}: length changed {len(expected)} -> {len(actual)}"
        for idx, (exp_item, act_item) in enumerate(zip(expected, actual)):
            _assert_preserved(exp_item, act_item, f"{path}[{idx}]")
    else:
        assert actual == expected, f"{path}: value changed {expected!r} -> {actual!r}"


@pytest.fixture
def config_roundtrip(tmp_path):
    """Round-trip a sparse CLI config through a page's ``load_config``.

    Usage::

        cfg, dump = config_roundtrip(page.load_config, {"some_options": {...}})

    Verifies two things:

    1. **Idempotency of the dumped form.** A sparse YAML materializes all
       defaults on load, so its dump cannot equal the source text. Instead we
       require ``load(sparse) -> dump1 -> load(dump1) -> dump2`` with
       ``dump1 == dump2``. This exercises the ``backend`` discriminated-union
       dispatch through jsonargparse.
    2. **Preservation of explicitly-set arguments.** Every value present in the
       sparse config survives unchanged in ``dump1`` (new defaults may appear in
       addition, but nothing that was set may change).

    A valid ``input.results_directory`` (the pytest ``tmp_path``) is injected, so
    callers only supply the page-specific option sections. Pages that do not use
    an ``input`` block (e.g. ``read_input``) pass ``inject_results_dir=False`` and
    provide their own top-level keys. Returns the first loaded config object and
    its ``model_dump(mode="json")``.
    """

    def _run(load_config, options: dict, *, inject_results_dir: bool = True):
        sparse = dict(options)
        if inject_results_dir:
            sparse.setdefault("input", {"results_directory": str(tmp_path)})
        path_a = tmp_path / "config_a.yaml"
        path_a.write_text(yaml.safe_dump(sparse))
        cfg1 = load_config(["--config", str(path_a)])
        dump1 = cfg1.model_dump(mode="json")

        path_b = tmp_path / "config_b.yaml"
        path_b.write_text(yaml.safe_dump(dump1))
        cfg2 = load_config(["--config", str(path_b)])
        dump2 = cfg2.model_dump(mode="json")

        assert dump1 == dump2, "dumped config is not a round-trip fixed point"
        _assert_preserved(options, dump1)
        return cfg1, dump1

    return _run
