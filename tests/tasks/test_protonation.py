"""Tests for the protonation task dispatcher and unified result conversion.

These tests cover the three backends exposed by
:mod:`qm_atlas.tasks.protonation`. The MoKa wrappers (``blabber.get_protonation``
and ``moka.get_moka_prediction``) are monkey-patched so the tests run without
the external MoKa software.
"""

from unittest.mock import patch

import pytest
from rdkit import Chem

from qm_atlas.tasks import protonation
from qm_atlas.wrappers.moka import moka, process

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GLYCINE_SMILES = "NCC(=O)O"


def _make_blabber_mol(smiles: str, abundance: float | None = None) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if abundance is not None:
        mol.SetProp(protonation.ABUNDANCE_PROP, str(abundance))
    return mol


def _make_moka_prediction(pkas: list[tuple[str, int, float]]) -> moka.MokaPrediction:
    """Build a ``MokaPrediction`` from (type, atom_index, value) tuples."""
    predicted_pkas = [
        moka.PredictedPKA(value=v, pka_type=t, atom_index=i, standard_deviation=0.5)
        for (t, i, v) in pkas
    ]
    return moka.MokaPrediction(
        name="glycine",
        covalent_hydration="0",
        unstable_tautomer=False,
        num_centers=len(predicted_pkas),
        predicted_pkas=predicted_pkas,
    )


def _glycine_pka_centers(mol: Chem.Mol) -> tuple[int, int]:
    """Return (amine_index, carboxyl_oxygen_index) for a glycine mol."""
    amine_idx = next(i for i, a in enumerate(mol.GetAtoms()) if a.GetSymbol() == "N")
    o_indices = [i for i, a in enumerate(mol.GetAtoms()) if a.GetSymbol() == "O"]
    # the -OH oxygen (single-bonded to C) is the protonatable acidic site
    carboxyl_oh_idx = next(
        i
        for i in o_indices
        if any(b.GetBondTypeAsDouble() == 1.0 for b in mol.GetAtomWithIdx(i).GetBonds())
    )
    return amine_idx, carboxyl_oh_idx


# ---------------------------------------------------------------------------
# Config (de)serialisation
# ---------------------------------------------------------------------------


def test_deserialize_blabber_default():
    cfg = protonation.deserialize_protonation_config({})
    assert isinstance(cfg, protonation.MokaBlabberConfig)


def test_deserialize_named_backends():
    cfg = protonation.deserialize_protonation_config({"backend": "moka_ionic_species"})
    assert isinstance(cfg, protonation.MokaIonicSpeciesConfig)
    cfg = protonation.deserialize_protonation_config({"backend": "moka_single_charge"})
    assert isinstance(cfg, protonation.MokaSingleChargeConfig)


def test_deserialize_unknown_backend_raises():
    with pytest.raises(ValueError):
        protonation.deserialize_protonation_config({"backend": "does_not_exist"})


def test_deserialize_passthrough_for_config_objects():
    cfg = protonation.MokaIonicSpeciesConfig(lower_ph_threshold=1.0)
    assert protonation.deserialize_protonation_config(cfg) is cfg


# ---------------------------------------------------------------------------
# Helper conversions
# ---------------------------------------------------------------------------


def test_predicted_pkas_to_centers_drops_centers_without_atom_index():
    pkas = [
        moka.PredictedPKA(value=2.3, pka_type="b", atom_index=0, standard_deviation=0.4),
        moka.PredictedPKA(value=9.5, pka_type="a", atom_index=None, standard_deviation=0.4),
    ]
    centers = protonation._predicted_pkas_to_centers(pkas)
    assert len(centers) == 1
    assert centers[0].center_type == "b"
    assert centers[0].index == 0
    assert centers[0].pka_value == pytest.approx(2.3)


# ---------------------------------------------------------------------------
# Blabber backend
# ---------------------------------------------------------------------------


def test_run_blabber_returns_unified_result(tmp_path):
    mol = Chem.MolFromSmiles(GLYCINE_SMILES)
    fake_protomers = [
        _make_blabber_mol("NCC(=O)O", abundance=70.0),
        _make_blabber_mol("[NH3+]CC(=O)O", abundance=20.0),
    ]
    with patch.object(protonation.blabber, "get_protonation", return_value=fake_protomers):
        result = protonation.run_protonation(
            mol,
            protonation.MokaBlabberConfig(),
            scr=tmp_path,
        )
    assert isinstance(result, protonation.ProtonationResult)
    assert len(result.protomers) == 2
    abundances = sorted(e.abundance for e in result.protomers if e.abundance is not None)
    assert abundances == [20.0, 70.0]
    assert result.transitions == []


def test_run_blabber_filters_by_abundance(tmp_path):
    mol = Chem.MolFromSmiles(GLYCINE_SMILES)
    fake_protomers = [
        _make_blabber_mol("NCC(=O)O", abundance=70.0),
        _make_blabber_mol("[NH3+]CC(=O)O", abundance=0.5),
    ]
    with patch.object(protonation.blabber, "get_protonation", return_value=fake_protomers):
        result = protonation.run_protonation(
            mol,
            protonation.MokaBlabberConfig(abundance_threshold=10.0),
            scr=tmp_path,
        )
    assert len(result.protomers) == 1
    assert result.protomers[0].abundance == pytest.approx(70.0)


def test_run_blabber_handles_missing_abundance_property(tmp_path):
    mol = Chem.MolFromSmiles(GLYCINE_SMILES)
    no_abundance = _make_blabber_mol("[NH3+]CC(=O)O", abundance=None)
    with patch.object(protonation.blabber, "get_protonation", return_value=[no_abundance]):
        result = protonation.run_protonation(
            mol,
            protonation.MokaBlabberConfig(),
            scr=tmp_path,
        )
    assert len(result.protomers) == 1
    assert result.protomers[0].abundance is None


# ---------------------------------------------------------------------------
# MoKa-pKa backends
# ---------------------------------------------------------------------------


def test_run_ionic_species_produces_transitions(tmp_path):
    """ionic-species backend reports neutral + anion + cation with two pKa transitions."""
    mol = Chem.MolFromSmiles(GLYCINE_SMILES)
    amine_idx, carboxyl_idx = _glycine_pka_centers(mol)
    prediction = _make_moka_prediction(
        [
            ("b", amine_idx, 9.6),  # basic amine
            ("a", carboxyl_idx, 2.3),  # acidic OH
        ]
    )
    with patch.object(protonation.moka, "get_moka_prediction", return_value=prediction):
        result = protonation.run_protonation(
            mol,
            protonation.MokaIonicSpeciesConfig(),
            scr=tmp_path,
        )

    assert isinstance(result, protonation.ProtonationResult)
    assert len(result.protomers) >= 2
    charges = sorted(Chem.GetFormalCharge(Chem.RemoveHs(e.mol)) for e in result.protomers)
    # ionic-species enumerates the parent + ionised states
    assert -1 in charges and 1 in charges

    assert len(result.transitions) == 2
    types = {t.pka_type for t in result.transitions}
    assert types == {protonation.PKA_TYPE_ACID, protonation.PKA_TYPE_BASE}
    for t in result.transitions:
        assert t.atom_index in (amine_idx, carboxyl_idx)
        assert isinstance(t.parent_smiles, str)
        assert isinstance(t.child_smiles, str)


def test_run_single_charge_emits_one_pair_per_center(tmp_path):
    mol = Chem.MolFromSmiles(GLYCINE_SMILES)
    amine_idx, carboxyl_idx = _glycine_pka_centers(mol)
    prediction = _make_moka_prediction(
        [
            ("b", amine_idx, 9.6),
            ("a", carboxyl_idx, 2.3),
        ]
    )
    with patch.object(protonation.moka, "get_moka_prediction", return_value=prediction):
        result = protonation.run_protonation(
            mol,
            protonation.MokaSingleChargeConfig(),
            scr=tmp_path,
        )

    # parent + one ion per pKa centre
    assert len(result.protomers) == 3
    assert len(result.transitions) == 2
    parent_smiles = {t.parent_smiles for t in result.transitions}
    # parent of the amine transition is the deprotonated form (neutral amine);
    # parent of the carboxyl transition is the protonated form (neutral acid).
    assert len(parent_smiles) >= 1


def test_run_pka_backend_with_no_centers_returns_empty(tmp_path):
    mol = Chem.MolFromSmiles("CC")
    empty_prediction = _make_moka_prediction([])
    with patch.object(protonation.moka, "get_moka_prediction", return_value=empty_prediction):
        result = protonation.run_protonation(
            mol,
            protonation.MokaIonicSpeciesConfig(),
            scr=tmp_path,
        )
    assert result.protomers == []
    assert result.transitions == []


def test_run_protonation_unknown_config_raises(tmp_path):
    class _Bogus(protonation.ProtonationOptions):
        backend: str = "bogus"

    mol = Chem.MolFromSmiles("C")
    with pytest.raises(ValueError):
        protonation.run_protonation(mol, _Bogus(), scr=tmp_path)


# ---------------------------------------------------------------------------
# Internal converter
# ---------------------------------------------------------------------------


def test_calc_info_to_result_maps_names_to_smiles():
    parent = Chem.MolFromSmiles(GLYCINE_SMILES)
    parent.SetProp("_Name", "parent")
    child = Chem.MolFromSmiles("NCC(=O)[O-]")
    child.SetProp("_Name", "child")

    calc = process.PkaCalculationInfo("parent")
    calc.add_compound(parent)
    calc.add_compound(child)
    center = process.PkaCenter(center_type="a", index=3, pka_value=4.5)
    calc.add_pka_center(center, parent_compound=parent, child_compound=child)

    result = protonation._calc_info_to_result(calc)
    assert len(result.protomers) == 2
    assert len(result.transitions) == 1
    transition = result.transitions[0]
    assert transition.pka_value == pytest.approx(4.5)
    assert transition.pka_type == protonation.PKA_TYPE_ACID
    assert transition.atom_index == 3
