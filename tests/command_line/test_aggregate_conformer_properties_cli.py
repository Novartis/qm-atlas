"""Tests for the aggregate_conformer_properties CLI module."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest
from conftest import RESOURCES  # pylint: disable=import-error
from context import create_homedir_tmp_path  # pylint: disable=import-error
from rdkit import Chem

from qm_atlas.command_line.base_config import CalculationInput
from qm_atlas.command_line.file_interface import compound_dir
from qm_atlas.command_line.pages import aggregate_conformer_properties
from qm_atlas.command_line.pages.aggregate_conformer_properties import (
    AggregateConformerPropertiesInterfaceConfig,
    AggregateConformerPropertiesOptions,
    BoltzmannMeanAggregation,
    MaxAggregation,
    MeanAggregation,
    MinAggregation,
    PropertyAggregationSpec,
    RepresentativeStructureOptions,
    main,
)
from qm_atlas.command_line.utils.conformer_selection import ConformerSelection, PropertyFilter
from qm_atlas.tasks.common import AtomBasedProperty, ScalarProperty, TensorProperty

GLYCINE_RESOURCE = RESOURCES / "cpd_dir_example" / "glycine"

HOMO = "tm_sp_HOMO(eV)"
ENERGY = "tm_sp_TotalEnergy(Ht)"


@pytest.fixture
def results_dir():
    """Copy the glycine compound directory into a fresh results directory."""
    tmp_path = create_homedir_tmp_path()
    results_directory = tmp_path / "results"
    results_directory.mkdir(parents=True, exist_ok=True)
    shutil.copytree(GLYCINE_RESOURCE, results_directory / "glycine")
    yield results_directory
    shutil.rmtree(tmp_path, ignore_errors=True)


def _sdf_scalar_values(files, property_name):
    values = []
    for sdf_file in files:
        with Chem.SDMolSupplier(str(sdf_file), removeHs=False) as supplier:
            mol = supplier[0]
        values.append(mol.GetDoubleProp(property_name))
    return values


def _molecule_row(cpd_dir):
    mol_df = pd.read_csv(cpd_dir.molecule_csv)
    return mol_df[mol_df[compound_dir.STATE_COLUMN] == "glycine"].iloc[0]


def _make_config(results_dir, options):
    return AggregateConformerPropertiesInterfaceConfig(
        input=CalculationInput(results_directory=results_dir),
        aggregate_options=options,
    )


def test_mean_written_to_molecule_csv(results_dir):
    cpd_dir = compound_dir.create_cpd_dir(results_dir / "glycine")
    files = cpd_dir.get_result_files("glycine")
    expected = float(np.mean(_sdf_scalar_values(files, HOMO)))

    main(
        config=_make_config(
            results_dir,
            AggregateConformerPropertiesOptions(
                properties=[PropertyAggregationSpec(property_name=HOMO)],
            ),
        )
    )

    row = _molecule_row(cpd_dir)
    assert row[f"{HOMO}_mean"] == pytest.approx(expected)


def test_min_max_written_to_molecule_csv(results_dir):
    cpd_dir = compound_dir.create_cpd_dir(results_dir / "glycine")
    files = cpd_dir.get_result_files("glycine")
    values = _sdf_scalar_values(files, HOMO)

    main(
        config=_make_config(
            results_dir,
            AggregateConformerPropertiesOptions(
                properties=[
                    PropertyAggregationSpec(
                        property_name=HOMO,
                        operations=[MinAggregation(), MaxAggregation()],
                    )
                ],
            ),
        )
    )

    row = _molecule_row(cpd_dir)
    assert row[f"{HOMO}_min"] == pytest.approx(min(values))
    assert row[f"{HOMO}_max"] == pytest.approx(max(values))


def test_boltzmann_mean_reads_weights_from_sdf(results_dir):
    cpd_dir = compound_dir.create_cpd_dir(results_dir / "glycine")
    files = cpd_dir.get_result_files("glycine")

    # Set a uniform weight property directly on the result SDF files.
    for sdf_file in files:
        cpd_dir.add_properties_to_sdf(sdf_file, {"test_weight": ScalarProperty(1.0)})

    expected = float(np.mean(_sdf_scalar_values(files, HOMO)))

    main(
        config=_make_config(
            results_dir,
            AggregateConformerPropertiesOptions(
                properties=[
                    PropertyAggregationSpec(
                        property_name=HOMO,
                        operations=[BoltzmannMeanAggregation(weight_property="test_weight")],
                    )
                ],
            ),
        )
    )

    row = _molecule_row(cpd_dir)
    assert row[f"{HOMO}_boltzmann_mean"] == pytest.approx(expected)


def test_property_filter_keep_lowest(results_dir):
    cpd_dir = compound_dir.create_cpd_dir(results_dir / "glycine")
    files = cpd_dir.get_result_files("glycine")
    energies = _sdf_scalar_values(files, ENERGY)
    homos = _sdf_scalar_values(files, HOMO)
    lowest_homo = homos[int(np.argmin(energies))]

    main(
        config=_make_config(
            results_dir,
            AggregateConformerPropertiesOptions(
                selection=ConformerSelection(
                    property_filters=[PropertyFilter(property_name=ENERGY, keep_lowest=1)]
                ),
                properties=[PropertyAggregationSpec(property_name=HOMO)],
            ),
        )
    )

    row = _molecule_row(cpd_dir)
    # Only the lowest-energy conformer contributes, so the mean equals its value.
    assert row[f"{HOMO}_mean"] == pytest.approx(lowest_homo)


def test_reference_results_source(results_dir):
    cpd_dir = compound_dir.create_cpd_dir(results_dir / "glycine")
    ref_files = []
    for ref_input in cpd_dir.get_reference_inputs("glycine"):
        ref_files.extend(cpd_dir.get_reference_results("glycine", ref_input))
    expected = float(np.mean(_sdf_scalar_values(ref_files, HOMO)))

    main(
        config=_make_config(
            results_dir,
            AggregateConformerPropertiesOptions(
                selection=ConformerSelection(source="reference_results"),
                properties=[PropertyAggregationSpec(property_name=HOMO)],
            ),
        )
    )

    row = _molecule_row(cpd_dir)
    assert row[f"{HOMO}_mean"] == pytest.approx(expected)


def test_tensor_property_aggregated_elementwise(results_dir):
    cpd_dir = compound_dir.create_cpd_dir(results_dir / "glycine")
    files = cpd_dir.get_result_files("glycine")

    # Give each conformer a distinct 3-vector tensor property.
    vectors = []
    for i, sdf_file in enumerate(files):
        vector = [float(i), float(i) + 1.0, float(i) + 2.0]
        vectors.append(vector)
        cpd_dir.add_properties_to_sdf(sdf_file, {"vec": TensorProperty(vector)})

    expected = np.mean(np.array(vectors), axis=0).tolist()

    main(
        config=_make_config(
            results_dir,
            AggregateConformerPropertiesOptions(
                properties=[PropertyAggregationSpec(property_name="vec", property_type="tensor")],
            ),
        )
    )

    row = _molecule_row(cpd_dir)
    assert json.loads(row["vec_mean"]) == pytest.approx(expected)


def test_representative_structure_disabled_by_default(results_dir):
    cpd_dir = compound_dir.create_cpd_dir(results_dir / "glycine")
    main(
        config=_make_config(
            results_dir,
            AggregateConformerPropertiesOptions(
                properties=[PropertyAggregationSpec(property_name=HOMO)],
            ),
        )
    )

    assert not (cpd_dir.dir / "glycine_representative.sdf").exists()


def test_representative_structure_carries_atom_based_mean(results_dir):
    cpd_dir = compound_dir.create_cpd_dir(results_dir / "glycine")
    files = cpd_dir.get_result_files("glycine")
    energies = _sdf_scalar_values(files, ENERGY)
    lowest_file = files[int(np.argmin(energies))]

    n_atoms = cpd_dir.extract_mol(files[0]).GetNumAtoms()

    # Give each conformer a distinct per-atom property.
    per_conformer_values = []
    for i, sdf_file in enumerate(files):
        values = [float(atom_idx + i) for atom_idx in range(n_atoms)]
        per_conformer_values.append(values)
        cpd_dir.add_properties_to_sdf(sdf_file, {"charges": AtomBasedProperty(values)})

    expected_mean = np.mean(np.array(per_conformer_values), axis=0)

    main(
        config=_make_config(
            results_dir,
            AggregateConformerPropertiesOptions(
                properties=[
                    PropertyAggregationSpec(property_name="charges", property_type="atom_based")
                ],
                representative_structure=RepresentativeStructureOptions(
                    enabled=True,
                    geometry_selection=ConformerSelection(
                        property_filters=[PropertyFilter(property_name=ENERGY, keep_lowest=1)]
                    ),
                ),
            ),
        )
    )

    representative_file = cpd_dir.dir / "glycine_representative.sdf"
    assert representative_file.exists()

    with Chem.SDMolSupplier(str(representative_file), removeHs=False) as supplier:
        rep_mol = supplier[0]

    # Geometry comes from the lowest-energy conformer.
    assert rep_mol.GetProp("Representative_Geometry_Source") == lowest_file.stem
    lowest_mol = cpd_dir.extract_mol(lowest_file)
    np.testing.assert_allclose(
        rep_mol.GetConformer().GetPositions(),
        lowest_mol.GetConformer().GetPositions(),
    )

    # Atom-based mean is stamped per atom and survives the SDF round-trip.
    read_back = AtomBasedProperty.from_mol(rep_mol, "charges_mean").get_property_value()
    np.testing.assert_allclose(read_back, expected_mean)


def test_missing_property_is_skipped(results_dir):
    cpd_dir = compound_dir.create_cpd_dir(results_dir / "glycine")
    main(
        config=_make_config(
            results_dir,
            AggregateConformerPropertiesOptions(
                properties=[PropertyAggregationSpec(property_name="not_a_real_property")],
            ),
        )
    )

    mol_df = pd.read_csv(cpd_dir.molecule_csv)
    assert "not_a_real_property_mean" not in mol_df.columns


def test_operations_deserialized_from_dicts():
    spec = PropertyAggregationSpec(
        property_name=HOMO,
        operations=[
            {"operation": "mean"},
            {"operation": "min"},
            {"operation": "boltzmann_mean", "weight_property": "w"},
        ],
    )
    assert isinstance(spec.operations[0], MeanAggregation)
    assert isinstance(spec.operations[1], MinAggregation)
    assert isinstance(spec.operations[2], BoltzmannMeanAggregation)
    assert spec.operations[2].weight_property == "w"


# ===========================================================================
# YAML config round-trip
# ===========================================================================


def test_config_roundtrip(config_roundtrip):
    """Per-property aggregation specs with a discriminated ``operation`` union
    round-trip with non-default operations and property types."""
    _cfg, dump = config_roundtrip(
        aggregate_conformer_properties.load_config,
        {
            "aggregate_options": {
                "properties": [
                    {
                        "property_name": HOMO,
                        "property_type": "scalar",
                        "operations": [
                            {"operation": "boltzmann_mean", "weight_property": "w"},
                            {"operation": "min"},
                        ],
                    },
                    {
                        "property_name": "some_tensor",
                        "property_type": "tensor",
                        "operations": [{"operation": "std"}],
                    },
                ]
            }
        },
    )
    props = dump["aggregate_options"]["properties"]
    assert [p["property_type"] for p in props] == ["scalar", "tensor"]
    assert [o["operation"] for o in props[0]["operations"]] == ["boltzmann_mean", "min"]
    assert props[0]["operations"][0]["weight_property"] == "w"
