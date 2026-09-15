"""Functional tests for the result-checking helpers (check_results.py).

These verify the success/failure accounting that runs after each pipeline step.
They use the pre-computed ``glycine`` compound directory fixture (glycine,
CHEMBL773) and mock nothing: the checks read real result SDFs, the
compound CSV and the registry. No external software is invoked.
"""

from pathlib import Path

import pytest
from conftest import RESOURCES  # pylint: disable=import-error

from qm_atlas.command_line.file_interface import compound_dir
from qm_atlas.command_line.utils import check_results
from qm_atlas.tasks import common
from qm_atlas.tasks.calculate_properties import TurbomoleSinglePointOptions
from qm_atlas.tasks.cosmo_properties import CosmoDescriptorsConfig, CosmoLogPConfig

GLYCINE_DIR = RESOURCES / "cpd_dir_example" / "glycine"  # glycine, CHEMBL773


@pytest.fixture
def glycine_cpd_dir():
    return compound_dir.create_cpd_dir(GLYCINE_DIR, create_new=False)


@pytest.fixture
def empty_glycine_cpd_dir(tmp_path, glycine_cpd_dir):
    """A fresh compound directory holding only the glycine input (no results)."""
    new_dir = compound_dir.create_cpd_dir(tmp_path / "glycine", create_new=True)
    input_sdf = glycine_cpd_dir.get_input_sdf_files()[0]
    mol = glycine_cpd_dir.extract_mol(input_sdf)
    new_dir.add_input_structure(mol)
    return new_dir


def _dummy_log(cpd_dir: compound_dir.CpdDir) -> Path:
    return cpd_dir.log_dir / "task.log"


# ---------------------------------------------------------------------------
# conformer expansion
# ---------------------------------------------------------------------------


def test_check_conformer_expansion_success(glycine_cpd_dir, caplog):
    input_sdf = glycine_cpd_dir.get_input_sdf_files()[0]
    worker_paths = [(glycine_cpd_dir, input_sdf, _dummy_log(glycine_cpd_dir))]

    with caplog.at_level("INFO"):
        check_results.check_conformer_expansion(worker_paths)

    assert any("succeeded for 1/1" in rec.message for rec in caplog.records)
    assert not any("Errors were encountered" in rec.message for rec in caplog.records)


def test_check_conformer_expansion_failure(empty_glycine_cpd_dir, caplog):
    input_sdf = empty_glycine_cpd_dir.get_input_sdf_files()[0]
    worker_paths = [(empty_glycine_cpd_dir, input_sdf, _dummy_log(empty_glycine_cpd_dir))]

    with caplog.at_level("INFO"):
        check_results.check_conformer_expansion(worker_paths)

    assert any("succeeded for 0/1" in rec.message for rec in caplog.records)
    assert any("Errors were encountered" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# reference optimization
# ---------------------------------------------------------------------------


def test_check_reference_optimization_success(glycine_cpd_dir, caplog):
    ref_input = glycine_cpd_dir.get_reference_inputs("glycine")[0]
    worker_paths = [(glycine_cpd_dir, ref_input, _dummy_log(glycine_cpd_dir))]

    with caplog.at_level("INFO"):
        check_results.check_reference_optimization(worker_paths)

    assert any("found for 1/1" in rec.message for rec in caplog.records)


def test_check_reference_optimization_failure(empty_glycine_cpd_dir, caplog):
    # Register a reference input in the fresh dir but never add optimized results.
    input_sdf = empty_glycine_cpd_dir.get_input_sdf_files()[0]
    mol = empty_glycine_cpd_dir.extract_mol(input_sdf)
    ref_input = empty_glycine_cpd_dir.add_reference_input(
        mol, conf_id=mol.GetConformers()[0].GetId()
    )
    worker_paths = [(empty_glycine_cpd_dir, ref_input, _dummy_log(empty_glycine_cpd_dir))]

    with caplog.at_level("INFO"):
        check_results.check_reference_optimization(worker_paths)

    assert any("found for 0/1" in rec.message for rec in caplog.records)
    assert any("Errors were encountered" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# conformer property calculation
# ---------------------------------------------------------------------------


def test_check_conformer_property_calculation_success(glycine_cpd_dir, caplog):
    result_sdf = glycine_cpd_dir.results_dir / "glycine_c0.sdf"
    worker_paths = [(glycine_cpd_dir, result_sdf, _dummy_log(glycine_cpd_dir))]
    # The glycine result SDFs carry tm_sp_* single-point properties.
    tasks = [TurbomoleSinglePointOptions()]

    with caplog.at_level("INFO"):
        check_results.check_conformer_property_calculation(worker_paths, tasks)

    assert any("succeeded for 1/1" in rec.message for rec in caplog.records)


def test_check_conformer_property_calculation_missing_property(glycine_cpd_dir, caplog):
    result_sdf = glycine_cpd_dir.results_dir / "glycine_c0.sdf"
    worker_paths = [(glycine_cpd_dir, result_sdf, _dummy_log(glycine_cpd_dir))]
    # A prefix that does not exist on the molecule forces a miss.
    tasks = [TurbomoleSinglePointOptions(property_prefix="absent_")]

    with caplog.at_level("INFO"):
        check_results.check_conformer_property_calculation(worker_paths, tasks)

    assert any("succeeded for 0/1" in rec.message for rec in caplog.records)
    assert any("misses" in rec.message for rec in caplog.records)


class _AtomPropertyConfig:
    """Minimal task config exposing a single atom-based property."""

    backend = "stub_atom_property"

    def get_property_names(self):
        return {"charge": "atom_charge"}

    def get_property_types(self):
        return {"atom_charge": common.AtomBasedProperty}


def test_check_conformer_property_calculation_atom_based_found(glycine_cpd_dir, caplog):
    result_sdf = glycine_cpd_dir.results_dir / "glycine_c0.sdf"
    # Atom-level properties are not persisted in the SDF, so drive the check with
    # an in-memory molecule that carries the per-atom tag on one atom.
    mol = glycine_cpd_dir.extract_mol(result_sdf)
    mol.GetAtomWithIdx(0).SetProp("atom_charge", "0.1")
    glycine_cpd_dir.extract_mol = lambda _sdf: mol  # type: ignore[method-assign]

    worker_paths = [(glycine_cpd_dir, result_sdf, _dummy_log(glycine_cpd_dir))]
    with caplog.at_level("INFO"):
        check_results.check_conformer_property_calculation(worker_paths, [_AtomPropertyConfig()])

    assert any("succeeded for 1/1" in rec.message for rec in caplog.records)


def test_check_conformer_property_calculation_atom_based_missing(glycine_cpd_dir, caplog):
    result_sdf = glycine_cpd_dir.results_dir / "glycine_c0.sdf"
    worker_paths = [(glycine_cpd_dir, result_sdf, _dummy_log(glycine_cpd_dir))]

    with caplog.at_level("INFO"):
        check_results.check_conformer_property_calculation(worker_paths, [_AtomPropertyConfig()])

    assert any("succeeded for 0/1" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# cosmotherm property calculation
# ---------------------------------------------------------------------------


def test_check_cosmotherm_calculation_descriptors_success(glycine_cpd_dir, caplog):
    result_sdf = glycine_cpd_dir.results_dir / "glycine_c0.sdf"
    worker_paths = [(glycine_cpd_dir, result_sdf, _dummy_log(glycine_cpd_dir))]
    # cosmo_water_* descriptors are present on the glycine result molecules.
    tasks = [CosmoDescriptorsConfig(solvent_name="water")]

    with caplog.at_level("INFO"):
        check_results.check_cosmotherm_calculation(worker_paths, tasks)

    assert any("succeeded for 1/1" in rec.message for rec in caplog.records)


def test_check_cosmotherm_calculation_descriptors_missing(glycine_cpd_dir, caplog):
    result_sdf = glycine_cpd_dir.results_dir / "glycine_c0.sdf"
    worker_paths = [(glycine_cpd_dir, result_sdf, _dummy_log(glycine_cpd_dir))]
    # A solvent whose descriptors were never computed -> miss.
    tasks = [CosmoDescriptorsConfig(solvent_name="acetone")]

    with caplog.at_level("INFO"):
        check_results.check_cosmotherm_calculation(worker_paths, tasks)

    assert any("succeeded for 0/1" in rec.message for rec in caplog.records)
    assert any("misses" in rec.message for rec in caplog.records)


def test_check_cosmotherm_calculation_whole_molecule_csv(glycine_cpd_dir, caplog):
    input_sdf = glycine_cpd_dir.get_input_sdf_files()[0]
    worker_paths = [(glycine_cpd_dir, input_sdf, _dummy_log(glycine_cpd_dir))]
    # cosmo_logp lives in the compound-level CSV, not on the SDF.
    tasks = [CosmoLogPConfig()]

    with caplog.at_level("INFO"):
        check_results.check_cosmotherm_calculation(worker_paths, tasks)

    assert any("succeeded for 1/1" in rec.message for rec in caplog.records)


def test_check_cosmotherm_calculation_whole_molecule_csv_missing(glycine_cpd_dir, caplog):
    input_sdf = glycine_cpd_dir.get_input_sdf_files()[0]
    worker_paths = [(glycine_cpd_dir, input_sdf, _dummy_log(glycine_cpd_dir))]
    # A prefix not present in the CSV columns -> miss.
    tasks = [CosmoLogPConfig(property_prefix="absent_")]

    with caplog.at_level("INFO"):
        check_results.check_cosmotherm_calculation(worker_paths, tasks)

    assert any("succeeded for 0/1" in rec.message for rec in caplog.records)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
