"""Tests for the align CLI page."""

import shutil

import numpy as np
import pytest
from conftest import RESOURCES  # pylint: disable=import-error
from context import create_homedir_tmp_path  # pylint: disable=import-error

from qm_atlas.command_line.base_config import CalculationInput
from qm_atlas.command_line.file_interface import compound_dir
from qm_atlas.command_line.pages import align
from qm_atlas.command_line.pages.align import (
    AlignInterfaceConfig,
    AlignOptions,
    DynamicReference,
    ReferenceInputConformation,
    main,
)
from qm_atlas.command_line.utils.conformer_selection import ConformerSelection, PropertyFilter

GLYCINE_RESOURCE = RESOURCES / "cpd_dir_example" / "glycine"
ENERGY_PROPERTY = "tm_sp_TotalEnergy(Ht)"


@pytest.fixture
def results_dir():
    """Copy the glycine compound directory into a fresh results directory."""
    tmp_path = create_homedir_tmp_path()
    results_directory = tmp_path / "results"
    results_directory.mkdir(parents=True, exist_ok=True)
    shutil.copytree(GLYCINE_RESOURCE, results_directory / "glycine")
    yield results_directory
    shutil.rmtree(tmp_path, ignore_errors=True)


def _coords_by_file(cpd_dir, state="glycine"):
    """Return {sdf_stem: (n_atoms, 3) coordinate array} for a state's results."""
    coords = {}
    for sdf_file in cpd_dir.get_result_files(state):
        mol = cpd_dir.extract_mol(sdf_file)
        coords[sdf_file.stem] = mol.GetConformer().GetPositions()
    return coords


def _make_config(results_directory, reference, **align_overrides):
    return AlignInterfaceConfig(
        input=CalculationInput(results_directory=results_directory),
        align_options=AlignOptions(reference=reference, **align_overrides),
    )


def test_dynamic_lowest_energy_reference_writes_rmsd(results_dir):
    reference = DynamicReference(
        selection=ConformerSelection(
            property_filters=[PropertyFilter(property_name=ENERGY_PROPERTY, keep_lowest=1)]
        ),
    )
    config = _make_config(results_dir, reference)
    main(config=config)

    cpd_dir = compound_dir.create_cpd_dir(results_dir / "glycine")
    result_files = cpd_dir.get_result_files("glycine")

    rmsds = []
    for sdf_file in result_files:
        mol = cpd_dir.extract_mol(sdf_file)
        assert mol.HasProp("alignment_rmsd")
        rmsds.append(mol.GetDoubleProp("alignment_rmsd"))

    # Exactly one conformer (the reference itself) has ~zero RMSD.
    near_zero = [r for r in rmsds if r < 1e-4]
    assert len(near_zero) == 1
    assert all(r >= 0 for r in rmsds)


def test_dynamic_reference_rmsd_in_conformer_csv(results_dir):
    reference = DynamicReference(
        selection=ConformerSelection(
            property_filters=[PropertyFilter(property_name=ENERGY_PROPERTY, keep_lowest=1)]
        ),
    )
    main(config=_make_config(results_dir, reference))

    cpd_dir = compound_dir.create_cpd_dir(results_dir / "glycine")
    import pandas as pd  # local import keeps the module import light

    conf_df = pd.read_csv(cpd_dir.conformer_csv)
    assert "alignment_rmsd" in conf_df.columns
    assert conf_df["alignment_rmsd"].dropna().shape[0] == len(cpd_dir.get_result_files("glycine"))


def test_alignment_changes_coordinates(results_dir):
    cpd_dir_before = compound_dir.create_cpd_dir(results_dir / "glycine")
    before = _coords_by_file(cpd_dir_before)

    reference = DynamicReference(
        selection=ConformerSelection(
            property_filters=[PropertyFilter(property_name=ENERGY_PROPERTY, keep_lowest=1)]
        ),
    )
    main(config=_make_config(results_dir, reference))

    cpd_dir_after = compound_dir.create_cpd_dir(results_dir / "glycine")
    after = _coords_by_file(cpd_dir_after)

    # At least one non-reference conformer must have moved during alignment.
    moved = [name for name in before if not np.allclose(before[name], after[name], atol=1e-6)]
    assert moved


def test_reference_input_mode(results_dir):
    config = _make_config(results_dir, ReferenceInputConformation())
    # Should complete without raising and align onto the registered reference.
    main(config=config)

    cpd_dir = compound_dir.create_cpd_dir(results_dir / "glycine")
    for sdf_file in cpd_dir.get_result_files("glycine"):
        mol = cpd_dir.extract_mol(sdf_file)
        assert mol.HasProp("alignment_rmsd")


def test_rmsd_property_can_be_disabled(results_dir):
    reference = DynamicReference()
    config = _make_config(results_dir, reference, rmsd_property=None)
    main(config=config)

    cpd_dir = compound_dir.create_cpd_dir(results_dir / "glycine")
    for sdf_file in cpd_dir.get_result_files("glycine"):
        mol = cpd_dir.extract_mol(sdf_file)
        assert not mol.HasProp("alignment_rmsd")


def test_original_properties_are_preserved(results_dir):
    reference = DynamicReference()
    main(config=_make_config(results_dir, reference))

    cpd_dir = compound_dir.create_cpd_dir(results_dir / "glycine")
    for sdf_file in cpd_dir.get_result_files("glycine"):
        mol = cpd_dir.extract_mol(sdf_file)
        # A pre-existing per-conformer property must survive the rewrite.
        assert mol.HasProp(ENERGY_PROPERTY)


def test_missing_reference_input_is_skipped(results_dir):
    config = _make_config(results_dir, ReferenceInputConformation(reference_name="does_not_exist"))
    # No matching reference -> state skipped, no RMSD written, no exception.
    main(config=config)

    cpd_dir = compound_dir.create_cpd_dir(results_dir / "glycine")
    for sdf_file in cpd_dir.get_result_files("glycine"):
        mol = cpd_dir.extract_mol(sdf_file)
        assert not mol.HasProp("alignment_rmsd")


def test_config_roundtrip(config_roundtrip):
    """Align options round-trip through load_config with a dynamic reference.

    Exercises the nested ConformerSelection inside the discriminated reference
    union, which is the case that must survive the jsonargparse round-trip.
    """
    _cfg, dump = config_roundtrip(
        align.load_config,
        {
            "align_options": {
                "reference": {
                    "mode": "dynamic",
                    "selection": {
                        "property_filters": [{"property_name": ENERGY_PROPERTY, "keep_lowest": 1}]
                    },
                },
                "alignment": {"backend": "rdkit"},
                "rmsd_property": "my_rmsd",
            }
        },
    )
    align_opts = dump["align_options"]
    assert align_opts["reference"]["mode"] == "dynamic"
    assert align_opts["alignment"]["backend"] == "rdkit"
    assert align_opts["rmsd_property"] == "my_rmsd"


def test_config_roundtrip_reference_input(config_roundtrip):
    """The reference_input branch of the discriminated union round-trips too."""
    _cfg, dump = config_roundtrip(
        align.load_config,
        {
            "align_options": {
                "reference": {"mode": "reference_input", "reference_name": "glycine_ref_0"},
            }
        },
    )
    reference = dump["align_options"]["reference"]
    assert reference["mode"] == "reference_input"
    assert reference["reference_name"] == "glycine_ref_0"


def test_config_roundtrip_defaults_stay_none(config_roundtrip):
    """Unset reference / alignment stay None at the options level (coalesced downstream)."""
    _cfg, dump = config_roundtrip(align.load_config, {"align_options": {}})
    assert dump["align_options"]["reference"] is None
    assert dump["align_options"]["alignment"] is None


def test_deserialize_reference_via_dict():
    """A YAML dict maps to the right reference subclass via the registry."""
    dynamic = align.deserialize_reference_config({"mode": "dynamic"})
    assert isinstance(dynamic, DynamicReference)
    ref_input = align.deserialize_reference_config(
        {"mode": "reference_input", "reference_name": "foo"}
    )
    assert isinstance(ref_input, ReferenceInputConformation)
    assert ref_input.reference_name == "foo"


def test_deserialize_reference_unknown_mode_raises():
    with pytest.raises(ValueError):
        align.deserialize_reference_config({"mode": "does_not_exist"})
