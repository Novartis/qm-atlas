"""Mocked tests for the workflow pages' ``run_local`` / ``run_job`` glue.

These verify that each page correctly collects its inputs (worker paths, the
molecule/COSMO files) and forwards the config-derived settings to the
lower-level workflow/task functions, then writes the results back to the
compound directory. All heavy workflow/task functions are mocked, so no external
software runs. ``run_local`` is additionally checked for iterating every worker
path and for continuing past a failing job.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from qm_atlas.command_line.base_config import CalculationInput
from qm_atlas.command_line.pages import conformer_properties as cp_page
from qm_atlas.command_line.pages import cosmotherm_properties as cosmo_page
from qm_atlas.command_line.pages import fast_conformers as fc_page
from qm_atlas.command_line.pages import optimize_reference_conformers as ref_page
from qm_atlas.command_line.pages import rescoss_conformers as rescoss_page
from qm_atlas.constants import COSMO_OUTPUT_KEY
from qm_atlas.tasks.calculate_properties import TurbomoleSinglePointOptions
from qm_atlas.tasks.cosmo_properties import CosmoLogPConfig


@pytest.fixture
def results_dir(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    return d


def _neutralize_logging(monkeypatch, page):
    """Stop a page's run_local from touching real per-task log files."""
    monkeypatch.setattr(page.task_files, "change_log_file", lambda *a, **k: None)
    monkeypatch.setattr(page.task_files, "close_task_log_handler", lambda: None)


def _disable_skip(monkeypatch, page):
    """Treat every worker path as incomplete so run_local processes them all."""
    monkeypatch.setattr(page, "worker_path_completed", lambda *a, **k: False)


# ===========================================================================
# fast_conformers
# ===========================================================================


def test_fast_conformers_run_job_forwards_options(results_dir, monkeypatch):
    config = fc_page.FastConformersInterfaceConfig(
        input=CalculationInput(results_directory=results_dir)
    )
    mol_in = MagicMock(name="mol_in")
    mol_3d = MagicMock(name="mol_3d")
    monkeypatch.setattr(fc_page.input_file, "read_input_sdf", lambda sdf: [mol_in])
    gen = MagicMock(return_value=mol_3d)
    monkeypatch.setattr(fc_page.fast_conformers, "generate_fast_conformers", gen)

    cpd_dir = MagicMock()
    cpd_dir.find_state_by_file.return_value = "glycine"
    fc_page.run_job(cpd_dir, Path("glycine.sdf"), config)

    gen.assert_called_once()
    _, kwargs = gen.call_args
    opts = config.fast_conformers_options
    assert kwargs["conformer_generation_options"] == opts.conformer_generation_options
    assert kwargs["n_cores"] == 1
    cpd_dir.add_result_conformers.assert_called_once_with(mol_3d)


def test_fast_conformers_run_local_iterates_and_survives_failures(results_dir, monkeypatch):
    config = fc_page.FastConformersInterfaceConfig(
        input=CalculationInput(results_directory=results_dir)
    )
    worker_paths = [
        (MagicMock(), Path("a.sdf"), Path("a.log")),
        (MagicMock(), Path("b.sdf"), Path("b.log")),
    ]
    monkeypatch.setattr(fc_page, "extract_worker_paths", lambda inp, workflow_type: worker_paths)
    _neutralize_logging(monkeypatch, fc_page)
    _disable_skip(monkeypatch, fc_page)
    run_job = MagicMock(side_effect=[RuntimeError("boom"), None])
    monkeypatch.setattr(fc_page, "run_job", run_job)

    fc_page.run_local(config)

    # Both molecules attempted despite the first raising.
    assert run_job.call_count == 2


def test_fast_conformers_run_local_skips_completed(results_dir, monkeypatch):
    config = fc_page.FastConformersInterfaceConfig(
        input=CalculationInput(results_directory=results_dir)
    )
    worker_paths = [
        (MagicMock(), Path("done.sdf"), Path("done.log")),
        (MagicMock(), Path("todo.sdf"), Path("todo.log")),
    ]
    monkeypatch.setattr(fc_page, "extract_worker_paths", lambda inp, workflow_type: worker_paths)
    _neutralize_logging(monkeypatch, fc_page)
    monkeypatch.setattr(
        fc_page, "worker_path_completed", lambda cpd_dir, sdf_file, wf: sdf_file.stem == "done"
    )
    run_job = MagicMock()
    monkeypatch.setattr(fc_page, "run_job", run_job)

    fc_page.run_local(config)

    # Only the not-yet-completed molecule is processed.
    assert run_job.call_count == 1
    assert run_job.call_args[0][1] == Path("todo.sdf")


# ===========================================================================
# rescoss_conformers
# ===========================================================================


def test_rescoss_run_job_forwards_options(results_dir, monkeypatch):
    config = rescoss_page.RescossInterfaceConfig(
        input=CalculationInput(results_directory=results_dir)
    )
    mol_in = MagicMock(name="mol_in")
    mol_3d = MagicMock(name="mol_3d")
    monkeypatch.setattr(rescoss_page.input_file, "read_input_sdf", lambda sdf: [mol_in])
    gen = MagicMock(return_value=mol_3d)
    monkeypatch.setattr(rescoss_page.rescoss_conformers, "generate_rescoss_conformers", gen)

    cpd_dir = MagicMock()
    cpd_dir.find_state_by_file.return_value = "glycine"
    rescoss_page.run_job(cpd_dir, Path("glycine.sdf"), config)

    gen.assert_called_once()
    _, kwargs = gen.call_args
    opts = config.rescoss_options
    assert kwargs["rmsd_threshold"] == opts.rmsd_threshold
    assert kwargs["num_clusters"] == opts.num_clusters
    cpd_dir.add_result_conformers.assert_called_once_with(mol_3d)


def test_rescoss_run_local_dispatches(results_dir, monkeypatch):
    config = rescoss_page.RescossInterfaceConfig(
        input=CalculationInput(results_directory=results_dir)
    )
    worker_paths = [(MagicMock(), Path("a.sdf"), Path("a.log"))]
    monkeypatch.setattr(
        rescoss_page, "extract_worker_paths", lambda inp, workflow_type: worker_paths
    )
    _neutralize_logging(monkeypatch, rescoss_page)
    _disable_skip(monkeypatch, rescoss_page)
    run_job = MagicMock()
    monkeypatch.setattr(rescoss_page, "run_job", run_job)

    rescoss_page.run_local(config)
    run_job.assert_called_once()


# ===========================================================================
# conformer_properties
# ===========================================================================


def _cp_config(results_dir):
    return cp_page.ConformerPropertyInterfaceConfig(
        input=CalculationInput(results_directory=results_dir),
        conformer_property_options=cp_page.ConformerPropertyOptions(
            calculation_tasks=[TurbomoleSinglePointOptions()]
        ),
    )


def test_conformer_properties_run_job_calculates_and_writes(results_dir, monkeypatch):
    config = _cp_config(results_dir)
    mol = MagicMock(name="mol")
    cpd_dir = MagicMock()
    cpd_dir.extract_mol.return_value = mol

    calc = MagicMock(return_value={"my_prop": 1.0})
    monkeypatch.setattr(cp_page.calculate_properties, "calculate_property", calc)

    cp_page.run_job(cpd_dir, Path("glycine_c0.sdf"), config)

    calc.assert_called_once()
    _, kwargs = calc.call_args
    assert kwargs["mol"] is mol
    assert kwargs["conf_id"] == 0
    assert kwargs["config"] is config.conformer_property_options.calculation_tasks[0]
    cpd_dir.add_properties_to_sdf.assert_called_once()
    cpd_dir.add_properties_to_conformer_csv.assert_called_once()


def test_conformer_properties_run_job_handles_cosmo_output(results_dir, monkeypatch):
    config = _cp_config(results_dir)
    cpd_dir = MagicMock()
    cpd_dir.extract_mol.return_value = MagicMock()

    cosmo_prop = MagicMock()
    cosmo_prop.get_property_value.return_value = "COSMO-TEXT"
    monkeypatch.setattr(
        cp_page.calculate_properties,
        "calculate_property",
        lambda **_k: {COSMO_OUTPUT_KEY: cosmo_prop, "my_prop": 1.0},
    )

    sdf = Path("glycine_c0.sdf")
    cp_page.run_job(cpd_dir, sdf, config)

    # The COSMO output is popped and written as a .cosmo result ...
    cpd_dir.add_cosmo_result.assert_called_once_with(sdf, "COSMO-TEXT", overwrite=True)
    # ... and the remaining scalar property still goes to the SDF/CSV.
    cpd_dir.add_properties_to_sdf.assert_called_once()


def test_conformer_properties_run_job_skips_when_no_properties(results_dir, monkeypatch):
    config = _cp_config(results_dir)
    cpd_dir = MagicMock()
    cpd_dir.extract_mol.return_value = MagicMock()
    monkeypatch.setattr(cp_page.calculate_properties, "calculate_property", lambda **_k: None)

    cp_page.run_job(cpd_dir, Path("glycine_c0.sdf"), config)

    cpd_dir.add_properties_to_sdf.assert_not_called()


def test_conformer_properties_run_local_dispatches(results_dir, monkeypatch):
    config = _cp_config(results_dir)
    worker_paths = [(MagicMock(), Path("a.sdf"), Path("a.log"))]
    monkeypatch.setattr(cp_page, "extract_worker_paths", lambda inp, workflow_type: worker_paths)
    _neutralize_logging(monkeypatch, cp_page)
    run_job = MagicMock()
    monkeypatch.setattr(cp_page, "run_job", run_job)

    cp_page.run_local(config)
    run_job.assert_called_once()


# ===========================================================================
# cosmotherm_properties
# ===========================================================================


def _cosmo_config(results_dir):
    return cosmo_page.CosmoPropertyInterfaceConfig(
        input=CalculationInput(results_directory=results_dir),
        cosmo_property_options=cosmo_page.CosmoPropertyOptions(
            calculation_tasks=[CosmoLogPConfig()]
        ),
    )


def _cpd_dir_with_cosmo(sdf_name="glycine_c0.sdf"):
    cpd_dir = MagicMock()
    cpd_dir.find_state_by_file.return_value = "glycine"
    result_sdf = Path(sdf_name)
    cpd_dir.get_result_files.return_value = [result_sdf]
    cosmo_file = MagicMock()
    cosmo_file.exists.return_value = True
    cpd_dir.get_associated_cosmo_file.return_value = cosmo_file
    return cpd_dir, result_sdf, cosmo_file


def test_cosmotherm_run_job_scalar_result_to_molecule_csv(results_dir, monkeypatch):
    config = _cosmo_config(results_dir)
    cpd_dir, _sdf, cosmo_file = _cpd_dir_with_cosmo()

    run_calc = MagicMock(return_value={"cosmo_logp": -1.1})
    monkeypatch.setattr(cosmo_page.cosmo_properties, "run_cosmo_calculation", run_calc)

    cosmo_page.run_job(cpd_dir, Path("glycine.sdf"), config)

    run_calc.assert_called_once()
    _, kwargs = run_calc.call_args
    assert kwargs["conformer_cosmo_files"] == [cosmo_file]
    cpd_dir.add_properties_to_molecule_csv.assert_called_once_with("glycine", {"cosmo_logp": -1.1})


def test_cosmotherm_run_job_list_result_to_conformer_csv(results_dir, monkeypatch):
    config = _cosmo_config(results_dir)
    cpd_dir, result_sdf, _cosmo = _cpd_dir_with_cosmo()

    monkeypatch.setattr(
        cosmo_page.cosmo_properties,
        "run_cosmo_calculation",
        lambda **_k: [{"cosmo_water_mu": 1.0}],
    )

    cosmo_page.run_job(cpd_dir, Path("glycine.sdf"), config)

    cpd_dir.add_properties_to_sdf.assert_called_once_with(result_sdf, {"cosmo_water_mu": 1.0})
    cpd_dir.add_properties_to_conformer_csv.assert_called_once_with(
        result_sdf, {"cosmo_water_mu": 1.0}
    )


def test_cosmotherm_run_job_continues_when_calc_fails(results_dir, monkeypatch):
    config = _cosmo_config(results_dir)
    cpd_dir, _sdf, _cosmo = _cpd_dir_with_cosmo()

    def _raise(**_k):
        raise RuntimeError("cosmotherm failed")

    monkeypatch.setattr(cosmo_page.cosmo_properties, "run_cosmo_calculation", _raise)

    # A failed task is logged and swallowed; nothing is written.
    cosmo_page.run_job(cpd_dir, Path("glycine.sdf"), config)
    cpd_dir.add_properties_to_molecule_csv.assert_not_called()


def test_cosmotherm_run_local_dispatches(results_dir, monkeypatch):
    config = _cosmo_config(results_dir)
    worker_paths = [(MagicMock(), Path("a.sdf"), Path("a.log"))]
    monkeypatch.setattr(
        cosmo_page, "extract_worker_paths", lambda inp, workflow_type: worker_paths
    )
    _neutralize_logging(monkeypatch, cosmo_page)
    run_job = MagicMock()
    monkeypatch.setattr(cosmo_page, "run_job", run_job)

    cosmo_page.run_local(config)
    run_job.assert_called_once()


# ===========================================================================
# optimize_reference_conformers
# ===========================================================================


def test_optimize_reference_run_job_saves_converged(results_dir, monkeypatch):
    config = ref_page.RefInterfaceConfig(input=CalculationInput(results_directory=results_dir))
    cpd_dir = MagicMock()
    cpd_dir.extract_mol.return_value = MagicMock()

    opt_mol = MagicMock()
    opt_mol.GetNumConformers.return_value = 1
    run_opt = MagicMock(return_value=[opt_mol])
    monkeypatch.setattr(ref_page.optimize_constrained, "run_constrained_optimization", run_opt)

    sdf = Path("glycine_ref_0.sdf")
    ref_page.run_job(cpd_dir, sdf, config)

    run_opt.assert_called_once()
    cpd_dir.add_reference_result.assert_called_once_with(opt_mol, sdf)


def test_optimize_reference_run_job_raises_when_all_failed(results_dir, monkeypatch):
    config = ref_page.RefInterfaceConfig(input=CalculationInput(results_directory=results_dir))
    cpd_dir = MagicMock()
    cpd_dir.extract_mol.return_value = MagicMock()

    failed = MagicMock()
    failed.GetNumConformers.return_value = 0  # optimization produced no conformer
    monkeypatch.setattr(
        ref_page.optimize_constrained,
        "run_constrained_optimization",
        lambda *a, **k: [failed],
    )

    with pytest.raises(RuntimeError, match="All constrained optimizations failed"):
        ref_page.run_job(cpd_dir, Path("glycine_ref_0.sdf"), config)
    cpd_dir.add_reference_result.assert_not_called()


def test_optimize_reference_run_local_dispatches(results_dir, monkeypatch):
    config = ref_page.RefInterfaceConfig(input=CalculationInput(results_directory=results_dir))
    worker_paths = [(MagicMock(), Path("a.sdf"), Path("a.log"))]
    monkeypatch.setattr(ref_page, "extract_worker_paths", lambda inp, workflow_type: worker_paths)
    _neutralize_logging(monkeypatch, ref_page)
    _disable_skip(monkeypatch, ref_page)
    run_job = MagicMock()
    monkeypatch.setattr(ref_page, "run_job", run_job)

    ref_page.run_local(config)
    run_job.assert_called_once()


def test_optimize_reference_run_local_skips_completed(results_dir, monkeypatch):
    config = ref_page.RefInterfaceConfig(input=CalculationInput(results_directory=results_dir))
    worker_paths = [
        (MagicMock(), Path("done_ref_0.sdf"), Path("done.log")),
        (MagicMock(), Path("todo_ref_0.sdf"), Path("todo.log")),
    ]
    monkeypatch.setattr(ref_page, "extract_worker_paths", lambda inp, workflow_type: worker_paths)
    _neutralize_logging(monkeypatch, ref_page)
    monkeypatch.setattr(
        ref_page,
        "worker_path_completed",
        lambda cpd_dir, sdf_file, wf: sdf_file.stem.startswith("done"),
    )
    run_job = MagicMock()
    monkeypatch.setattr(ref_page, "run_job", run_job)

    ref_page.run_local(config)

    assert run_job.call_count == 1
    assert run_job.call_args[0][1] == Path("todo_ref_0.sdf")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
