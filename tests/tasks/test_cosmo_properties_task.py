"""Tests for the COSMOtherm calculator wrappers (tasks/cosmo_properties.py).

Each wrapper turns a typed config into keyword arguments for a lower-level
``cosmo_tasks`` / ``cpsa`` function and then renames the returned canonical keys
to the config's property names. The lower-level functions are mocked, so no
COSMOtherm run happens; the tests assert that the config is correctly unpacked
and that the result keys are renamed as configured.
"""

import pytest

from qm_atlas.tasks import cosmo_properties
from qm_atlas.tasks.cosmo_properties import (
    CosmoDeltaGConfig,
    CosmoDescriptorsConfig,
    CosmoLogPConfig,
    CosmoPermConfig,
    CosmoPsaConfig,
    run_cosmo_calculation,
)


def _recorder(return_value):
    calls = {}

    def _fn(**kwargs):
        calls.update(kwargs)
        return return_value

    return _fn, calls


# ---------------------------------------------------------------------------
# calculate_logp
# ---------------------------------------------------------------------------


def test_calculate_logp_unpacks_config_and_renames(monkeypatch):
    fn, calls = _recorder({"cosmo_logp": 1.23})
    monkeypatch.setattr(cosmo_properties.cosmo_tasks, "calculate_logp", fn)

    config = CosmoLogPConfig(property_prefix="myprefix_")
    result = cosmo_properties.calculate_logp([cosmo_properties.Path("a.cosmo")], config)

    # Result key renamed to the config's property name.
    assert result == {"myprefix_cosmo_logp": 1.23}
    # Config was unpacked to kwargs without the discriminator/prefix keys.
    assert "backend" not in calls
    assert "property_prefix" not in calls
    assert "is_woctanol" in calls  # a CosmoLogPConfig-specific field
    assert calls["compound_conformers"] == [cosmo_properties.Path("a.cosmo")]


# ---------------------------------------------------------------------------
# calculate_delta_g
# ---------------------------------------------------------------------------


def test_calculate_delta_g_renames_with_solvent_prefix(monkeypatch):
    fn, calls = _recorder({"cosmo_delta_g": -9.9})
    monkeypatch.setattr(cosmo_properties.cosmo_tasks, "calculate_delta_g", fn)

    config = CosmoDeltaGConfig(solvent_name="h2o")
    result = cosmo_properties.calculate_delta_g([cosmo_properties.Path("a.cosmo")], config)

    assert result == {"h2o_cosmo_delta_g": -9.9}
    assert calls["solvent_name"] == "h2o"


# ---------------------------------------------------------------------------
# calculate_descriptors
# ---------------------------------------------------------------------------


def test_calculate_descriptors_renames_each_conformer(monkeypatch):
    fn, _calls = _recorder([{"mu": 1.0, "Area": 2.0}, {"mu": 3.0, "Area": 4.0}])
    monkeypatch.setattr(cosmo_properties.cosmo_tasks, "calculate_descriptors", fn)

    config = CosmoDescriptorsConfig(solvent_name="water")
    result = cosmo_properties.calculate_descriptors(
        [cosmo_properties.Path("a.cosmo"), cosmo_properties.Path("b.cosmo")], config
    )

    assert result == [
        {"cosmo_water_mu": 1.0, "cosmo_water_Area": 2.0},
        {"cosmo_water_mu": 3.0, "cosmo_water_Area": 4.0},
    ]


# ---------------------------------------------------------------------------
# calculate_cosmo_psa
# ---------------------------------------------------------------------------


def test_calculate_cosmo_psa_renames(monkeypatch):
    monkeypatch.setattr(
        cosmo_properties.cpsa,
        "extract_psa_from_cosmo_file",
        lambda cosmo_file, **kwargs: {"cosmo_psa": 42.0},
    )

    config = CosmoPsaConfig(property_prefix="p_")
    result = cosmo_properties.calculate_cosmo_psa([cosmo_properties.Path("a.cosmo")], config)

    assert result == [{"p_cosmo_psa": 42.0}]


def test_calculate_cosmo_psa_rejects_wrong_config():
    with pytest.raises(ValueError, match="must be of type CosmoPsaConfig"):
        cosmo_properties.calculate_cosmo_psa([cosmo_properties.Path("a.cosmo")], CosmoLogPConfig())


def test_calculate_cosmo_psa_skips_non_cosmo_files(monkeypatch):
    monkeypatch.setattr(
        cosmo_properties.cpsa,
        "extract_psa_from_cosmo_file",
        lambda cosmo_file, **kwargs: {"cosmo_psa": 1.0},
    )
    # A non-.cosmo path is silently skipped.
    result = cosmo_properties.calculate_cosmo_psa(
        [cosmo_properties.Path("a.txt")], CosmoPsaConfig()
    )
    assert result == []


# ---------------------------------------------------------------------------
# calculate_perm
# ---------------------------------------------------------------------------


def test_calculate_perm_renames(monkeypatch):
    fn, _calls = _recorder({"cosmo_perm": 7.7})
    monkeypatch.setattr(cosmo_properties.cosmo_tasks, "calculate_cosmoperm", fn)

    config = CosmoPermConfig(property_prefix="perm_")
    result = cosmo_properties.calculate_perm([cosmo_properties.Path("a.cosmo")], config)

    assert result == {"perm_cosmo_perm": 7.7}


# ---------------------------------------------------------------------------
# run_cosmo_calculation dispatch
# ---------------------------------------------------------------------------


def test_run_cosmo_calculation_dispatches_by_backend(monkeypatch):
    fn, calls = _recorder({"cosmo_logp": 0.5})
    monkeypatch.setattr(cosmo_properties.cosmo_tasks, "calculate_logp", fn)

    config = CosmoLogPConfig()
    result = run_cosmo_calculation([cosmo_properties.Path("a.cosmo")], config)

    assert result == {"cosmo_logp": 0.5}
    assert calls  # the logp calculator was invoked


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
