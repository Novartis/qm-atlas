"""End-to-end integration tests for whole workflows and the CLI.

These are heavier than the unit tests (they drive the top-level ``qm_atlas``
dispatcher and chain several pipeline stages), so they are marked
``@pytest.mark.integration`` and can be deselected with ``-m 'not integration'``.

Most run without any external software. The final test additionally requires a
real conformer/optimization backend and is gated with ``require_software`` on top
of the integration marker.
"""

import shutil

import pandas as pd
import pytest
from conftest import RESOURCES  # pylint: disable=import-error
from context import require_software  # pylint: disable=import-error
from rdkit import Chem
from rdkit.Chem import AllChem

import qm_atlas.__main__ as qm_atlas_cli
from qm_atlas.command_line.base_config import CalculationInput
from qm_atlas.command_line.file_interface import collect_output, compound_dir
from qm_atlas.command_line.pages import (
    aggregate_conformer_properties,
    calculate_boltzmann_weights,
    collect,
)
from qm_atlas.command_line.pages.aggregate_conformer_properties import (
    AggregateConformerPropertiesInterfaceConfig,
    AggregateConformerPropertiesOptions,
    BoltzmannMeanAggregation,
    MeanAggregation,
    PropertyAggregationSpec,
)
from qm_atlas.command_line.pages.calculate_boltzmann_weights import (
    BoltzmannWeightOptions,
    CalculateBoltzmannWeightsInterfaceConfig,
)
from qm_atlas.command_line.pages.collect import CollectInterfaceConfig

GLYCINE_RESOURCE = RESOURCES / "cpd_dir_example" / "glycine"  # glycine, CHEMBL773
ENERGY = "tm_sp_TotalEnergy(Ht)"
HOMO = "tm_sp_HOMO(eV)"

pytestmark = pytest.mark.integration


# ===========================================================================
# CLI: read_input end-to-end (no external software)
# ===========================================================================


def test_cli_read_input_creates_compound_directories(tmp_path):
    """Full top-level CLI dispatch: qm_atlas read_input -> compound dirs.

    Exercises argument parsing, the read_input page, molecule reading and the
    compound-directory/registry file interface in a single real run.
    """
    sdf_file = tmp_path / "mols.sdf"
    with Chem.SDWriter(str(sdf_file)) as writer:
        for name, smiles in [
            ("ethanol", "CCO"),
            ("acetic_acid", "CC(=O)O"),
        ]:  # CHEMBL545, CHEMBL539
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
            mol.SetProp("_Name", name)
            writer.write(mol)

    results_dir = tmp_path / "results"
    exit_code = qm_atlas_cli.main(
        [
            "qm_atlas",
            "read_input",
            "--input_file+",
            str(sdf_file),
            "--results_directory",
            str(results_dir),
        ]
    )

    assert exit_code == 0
    for name in ("ethanol", "acetic_acid"):
        cpd_dir = results_dir / name
        assert cpd_dir.is_dir()
        assert (cpd_dir / "registry.json").is_file()
        assert (cpd_dir / "input" / f"{name}.sdf").is_file()


def test_cli_unknown_command_returns_error_code():
    exit_code = qm_atlas_cli.main(["qm_atlas", "not_a_real_command"])
    assert exit_code == 4


# ===========================================================================
# Workflow: Boltzmann weights -> aggregation -> collect (no external software)
# ===========================================================================


@pytest.fixture
def glycine_results(tmp_path):
    """A writable results directory seeded with the glycine fixture."""
    results_directory = tmp_path / "results"
    results_directory.mkdir(parents=True, exist_ok=True)
    shutil.copytree(GLYCINE_RESOURCE, results_directory / "glycine")
    return results_directory


def test_workflow_boltzmann_aggregate_collect(glycine_results):
    """Chain three post-processing stages on real pre-computed results.

    Boltzmann weighting feeds a Boltzmann-weighted-mean aggregation, and the
    collect stage gathers everything into the output directory — a realistic
    multi-step pipeline that needs no QM software.
    """
    input_block = CalculationInput(results_directory=glycine_results)

    # 1) Per-conformer Boltzmann weights from the single-point energy.
    calculate_boltzmann_weights.main(
        config=CalculateBoltzmannWeightsInterfaceConfig(
            input=input_block,
            boltzmann_options=BoltzmannWeightOptions(
                energy_property=ENERGY,
                energy_unit="hartree",
                temperature=298.15,
                weight_property="boltzmann_weight",
            ),
        )
    )

    # 2) Aggregate HOMO to a plain mean and a Boltzmann-weighted mean.
    aggregate_conformer_properties.main(
        config=AggregateConformerPropertiesInterfaceConfig(
            input=input_block,
            aggregate_options=AggregateConformerPropertiesOptions(
                properties=[
                    PropertyAggregationSpec(
                        property_name=HOMO,
                        operations=[
                            MeanAggregation(),
                            BoltzmannMeanAggregation(weight_property="boltzmann_weight"),
                        ],
                    )
                ],
            ),
        )
    )

    # 3) Collect merged outputs.
    collect.main(config=CollectInterfaceConfig(input=input_block))

    cpd_dir = compound_dir.create_cpd_dir(glycine_results / "glycine")

    # Boltzmann weights are present and normalized.
    conf_df = pd.read_csv(cpd_dir.conformer_csv)
    assert "boltzmann_weight" in conf_df.columns
    assert conf_df["boltzmann_weight"].dropna().sum() == pytest.approx(1.0)

    # Aggregated per-state columns exist in the molecule CSV.
    mol_df = pd.read_csv(cpd_dir.molecule_csv)
    row = mol_df[mol_df[compound_dir.STATE_COLUMN] == "glycine"].iloc[0]
    assert f"{HOMO}_mean" in mol_df.columns
    assert pd.notna(row[f"{HOMO}_mean"])
    boltz_cols = [c for c in mol_df.columns if c.startswith(HOMO) and "boltzmann" in c.lower()]
    assert boltz_cols, "expected a Boltzmann-weighted HOMO aggregation column"

    # Collected output directory holds the merged artifacts.
    collected = glycine_results / collect_output.OUTPUT_DIR_NAME
    assert (collected / "glycine.sdf").is_file()
    assert (collected / "glycine_properties.csv").is_file()


# ===========================================================================
# Workflow: fast conformer generation (requires external software)
# ===========================================================================


@require_software("xtb")
def test_workflow_fast_conformers_generates_geometry(tmp_path):
    """Minimal real conformer-generation run (smallest practical molecule).

    Gated by ``require_software('xtb')`` so it is skipped unless a backend is
    configured; also marked ``integration`` (module-level) so it can be
    deselected in fast CI runs.
    """
    from context import TEST_CONF_GEN_OPTIONS  # pylint: disable=import-error

    from qm_atlas.workflows import fast_conformers

    mol = Chem.MolFromSmiles("CO")  # methanol, CHEMBL14688
    mol.SetProp("_Name", "methanol")

    result = fast_conformers.generate_fast_conformers(
        mol,
        scr=tmp_path,
        show_progress=False,
        conformer_generation_options=TEST_CONF_GEN_OPTIONS,
    )

    assert result is not None
    assert result.GetNumConformers() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
