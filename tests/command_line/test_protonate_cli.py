"""Tests for the protonate CLI page.

The MoKa wrappers are monkey-patched in every test, so no external software
is required.
"""

import shutil
from unittest.mock import patch

import pytest
from context import create_homedir_tmp_path  # pylint: disable=import-error
from rdkit import Chem

# ===========================================================================
# SMILES Test Strings - Global Constants
# ===========================================================================
SMILES_GLYCINE_NEUTRAL = "NCC(=O)O"
SMILES_GLYCINE_ANION = "NCC(=O)[O-]"
SMILES_GLYCINE_CATION = "[NH3+]CC(=O)O"
SMILES_ACETIC_ACID = "CC(=O)O"
SMILES_ACETATE = "CC(=O)[O-]"
SMILES_ACETATE_SECONDARY = "[CH2-]C(=O)O"
SMILES_ACETIC_CATION = "C[C+](O)O"

from qm_atlas.command_line.base_config import CalculationInput
from qm_atlas.command_line.file_interface.compound_dir import create_cpd_dir
from qm_atlas.command_line.file_interface.input_file import InputConfig
from qm_atlas.command_line.pages import protonate
from qm_atlas.command_line.pages.protonate import (
    ProtonateInterfaceConfig,
    ProtonateOptions,
    ProtonateSubmissionConfig,
    run_local,
)
from qm_atlas.command_line.pages.read_input import ReadInputConfig
from qm_atlas.command_line.pages.read_input import run as read_input_run
from qm_atlas.tasks import protonation


@pytest.fixture
def temp_work_dir():
    tmp_path = create_homedir_tmp_path()
    yield tmp_path
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)


def _make_compound_dir(temp_work_dir, smiles: str = SMILES_GLYCINE_NEUTRAL, name: str = "glycine"):
    results_dir = temp_work_dir / "results"
    csv_file = temp_work_dir / "in.csv"
    csv_file.write_text(f"name,smiles\n{name},{smiles}\n")
    read_input_run(
        ReadInputConfig(
            input_file=[csv_file],
            results_directory=results_dir,
            input_config=InputConfig(),
        )
    )
    return results_dir, results_dir / name


def _build_config(
    cpd_dir_path,
    backend: protonation.ProtonationConfig | None = None,
    **backend_kwargs,
) -> ProtonateInterfaceConfig:
    if backend is None:
        backend = protonation.MokaBlabberConfig(**backend_kwargs)
    return ProtonateInterfaceConfig(
        input=CalculationInput(
            results_directory=cpd_dir_path.parent,
            compound_directories=[cpd_dir_path],
        ),
        submission_config=ProtonateSubmissionConfig(submit=False),
        protonate_options=ProtonateOptions(backend=backend),
    )


def _mol_with_abundance(smiles: str, abundance: float) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    mol.SetProp(protonation.ABUNDANCE_PROP, str(abundance))
    return mol


def _make_result(*entries, transitions=None) -> protonation.ProtonationResult:
    """Build a ProtonationResult from (smiles, abundance) tuples."""
    protomer_entries = [
        protonation.ProtomerEntry(mol=Chem.MolFromSmiles(smi), abundance=abundance)
        for smi, abundance in entries
    ]
    return protonation.ProtonationResult(
        protomers=protomer_entries,
        transitions=transitions or [],
    )


# ===========================================================================
# Tests for the moka_blabber backend (default).
# ===========================================================================


def test_protomers_registered_with_charge_suffix(temp_work_dir):
    """Each charged protomer gets a `_A`/`_BH` suffix; tiebreaker counter applies."""
    _, cpd_dir_path = _make_compound_dir(temp_work_dir, smiles=SMILES_ACETIC_ACID, name="acetic")

    result = _make_result(
        (SMILES_ACETATE, 80.0),
        (SMILES_ACETATE_SECONDARY, 5.0),
        (SMILES_ACETIC_CATION, 1.0),
    )

    with patch.object(
        protonate.protonation,
        "run_protonation",
        return_value=result,
    ) as mock_run:
        run_local(_build_config(cpd_dir_path))

    mock_run.assert_called_once()
    cpd = create_cpd_dir(cpd_dir_path, create_new=False)
    names = set(cpd.registry_handler.get_registered_names())
    assert {"acetic", "acetic_A", "acetic_A2", "acetic_BH"} <= names


def test_duplicate_smiles_are_skipped(temp_work_dir):
    _, cpd_dir_path = _make_compound_dir(temp_work_dir, smiles=SMILES_ACETIC_ACID, name="acetic")

    # one protomer matches existing neutral state -> skip; another is new
    result = _make_result(
        (SMILES_ACETIC_ACID, 50.0),
        (SMILES_ACETATE, 50.0),
    )

    with patch.object(
        protonate.protonation,
        "run_protonation",
        return_value=result,
    ):
        run_local(_build_config(cpd_dir_path))

    cpd = create_cpd_dir(cpd_dir_path, create_new=False)
    names = set(cpd.registry_handler.get_registered_names())
    # one duplicate skipped, one new state registered, no name like 'acetic2' generated
    assert names == {"acetic", "acetic_A"}


def test_empty_backend_output_is_noop(temp_work_dir):
    _, cpd_dir_path = _make_compound_dir(temp_work_dir, smiles=SMILES_ACETIC_ACID, name="acetic")

    with patch.object(
        protonate.protonation, "run_protonation", return_value=protonation.ProtonationResult()
    ):
        run_local(_build_config(cpd_dir_path))

    cpd = create_cpd_dir(cpd_dir_path, create_new=False)
    assert set(cpd.registry_handler.get_registered_names()) == {"acetic"}
    assert cpd.get_all_pka_info() == []


def test_backend_options_are_forwarded(temp_work_dir):
    """Backend options reach the wrapper through the task dispatcher."""
    _, cpd_dir_path = _make_compound_dir(temp_work_dir, smiles=SMILES_ACETIC_ACID, name="acetic")

    captured = {}

    def fake_blabber_get_protonation(mol, scr, remove_zero_charged, ph, threshold, **kwargs):
        captured["ph"] = ph
        captured["threshold"] = threshold
        captured["remove_zero_charged"] = remove_zero_charged
        return [_mol_with_abundance(SMILES_ACETATE, 90.0)]

    with patch.object(
        protonation.blabber, "get_protonation", side_effect=fake_blabber_get_protonation
    ):
        run_local(
            _build_config(
                cpd_dir_path,
                ph="6-8",
                abundance_threshold=1.0,
                remove_zero_charged=True,
            )
        )

    assert captured == {
        "ph": "6-8",
        "threshold": 1.0,
        "remove_zero_charged": True,
    }


def test_abundance_filter_drops_low_abundance(temp_work_dir):
    """The task-level abundance filter drops protomers below the threshold."""
    _, cpd_dir_path = _make_compound_dir(temp_work_dir, smiles=SMILES_ACETIC_ACID, name="acetic")

    high = _mol_with_abundance(SMILES_ACETATE, 80.0)
    low = _mol_with_abundance(SMILES_ACETATE_SECONDARY, 0.1)

    with patch.object(
        protonation.blabber,
        "get_protonation",
        return_value=[high, low],
    ):
        run_local(_build_config(cpd_dir_path, abundance_threshold=10.0))

    cpd = create_cpd_dir(cpd_dir_path, create_new=False)
    names = set(cpd.registry_handler.get_registered_names())
    assert names == {"acetic", "acetic_A"}


def test_blabber_writes_no_pka_info(temp_work_dir):
    """The blabber backend does not produce pKa transitions or atom indices."""
    _, cpd_dir_path = _make_compound_dir(temp_work_dir, smiles=SMILES_ACETIC_ACID, name="acetic")
    result = _make_result((SMILES_ACETATE, 80.0))

    with patch.object(protonate.protonation, "run_protonation", return_value=result):
        run_local(_build_config(cpd_dir_path))

    cpd = create_cpd_dir(cpd_dir_path, create_new=False)
    assert cpd.get_all_pka_info() == []


# ===========================================================================
# Tests for the moka_ionic_species / moka_single_charge backends.
# ===========================================================================


def test_ionic_species_writes_pka_transitions(temp_work_dir):
    """Transitions reported by the backend are written to pka.csv with atom indices."""
    _, cpd_dir_path = _make_compound_dir(
        temp_work_dir, smiles=SMILES_GLYCINE_NEUTRAL, name="glycine"
    )

    transitions = [
        protonation.ProtonationPkaTransition(
            parent_smiles=SMILES_GLYCINE_NEUTRAL,
            child_smiles=SMILES_GLYCINE_CATION,
            pka_value=2.3,
            pka_type=protonation.PKA_TYPE_BASE,
            atom_index=0,
        ),
        protonation.ProtonationPkaTransition(
            parent_smiles=SMILES_GLYCINE_ANION,
            child_smiles=SMILES_GLYCINE_NEUTRAL,
            pka_value=9.6,
            pka_type=protonation.PKA_TYPE_ACID,
            atom_index=3,
        ),
    ]
    result = protonation.ProtonationResult(
        protomers=[
            protonation.ProtomerEntry(mol=Chem.MolFromSmiles(SMILES_GLYCINE_CATION)),
            protonation.ProtomerEntry(mol=Chem.MolFromSmiles(SMILES_GLYCINE_ANION)),
        ],
        transitions=transitions,
    )

    backend = protonation.MokaIonicSpeciesConfig()
    with patch.object(protonate.protonation, "run_protonation", return_value=result):
        run_local(_build_config(cpd_dir_path, backend=backend))

    cpd = create_cpd_dir(cpd_dir_path, create_new=False)
    names = set(cpd.registry_handler.get_registered_names())
    assert {"glycine", "glycine_A", "glycine_BH"} <= names

    pka_entries = cpd.get_all_pka_info()
    assert len(pka_entries) == 2
    methods = {p.method for p in pka_entries}
    assert methods == {"moka_ionic_species"}

    by_type = {p.pka_type: p for p in pka_entries}
    base = by_type[protonation.PKA_TYPE_BASE]
    assert base.pka_value == pytest.approx(2.3)
    assert base.atom_index == 0
    assert base.parent_states == ["glycine"]
    assert base.child_states == ["glycine_BH"]

    acid = by_type[protonation.PKA_TYPE_ACID]
    assert acid.pka_value == pytest.approx(9.6)
    assert acid.atom_index == 3
    assert acid.parent_states == ["glycine_A"]
    assert acid.child_states == ["glycine"]


def test_single_charge_backend_uses_correct_method_label(temp_work_dir):
    _, cpd_dir_path = _make_compound_dir(
        temp_work_dir, smiles=SMILES_GLYCINE_NEUTRAL, name="glycine"
    )
    transitions = [
        protonation.ProtonationPkaTransition(
            parent_smiles=SMILES_GLYCINE_NEUTRAL,
            child_smiles=SMILES_GLYCINE_CATION,
            pka_value=2.3,
            pka_type=protonation.PKA_TYPE_BASE,
            atom_index=0,
        ),
    ]
    result = protonation.ProtonationResult(
        protomers=[
            protonation.ProtomerEntry(mol=Chem.MolFromSmiles(SMILES_GLYCINE_CATION)),
        ],
        transitions=transitions,
    )
    backend = protonation.MokaSingleChargeConfig()
    with patch.object(protonate.protonation, "run_protonation", return_value=result):
        run_local(_build_config(cpd_dir_path, backend=backend))

    cpd = create_cpd_dir(cpd_dir_path, create_new=False)
    pka_entries = cpd.get_all_pka_info()
    assert len(pka_entries) == 1
    assert pka_entries[0].method == "moka_single_charge"
    assert pka_entries[0].atom_index == 0


def test_transitions_for_unregistered_smiles_are_skipped(temp_work_dir):
    """If the backend reports a transition involving an unknown SMILES, it is skipped."""
    _, cpd_dir_path = _make_compound_dir(
        temp_work_dir, smiles=SMILES_GLYCINE_NEUTRAL, name="glycine"
    )
    # transition refers to a SMILES we never register (no protomer for it)
    transitions = [
        protonation.ProtonationPkaTransition(
            parent_smiles=SMILES_GLYCINE_NEUTRAL,
            child_smiles="[CH3+]",  # bogus child smiles
            pka_value=5.5,
            pka_type=protonation.PKA_TYPE_BASE,
            atom_index=0,
        ),
    ]
    result = protonation.ProtonationResult(protomers=[], transitions=transitions)
    backend = protonation.MokaIonicSpeciesConfig()
    with patch.object(protonate.protonation, "run_protonation", return_value=result):
        run_local(_build_config(cpd_dir_path, backend=backend))

    cpd = create_cpd_dir(cpd_dir_path, create_new=False)
    assert cpd.get_all_pka_info() == []


def test_deserialize_backend_via_dict(temp_work_dir):
    """A dict in YAML maps to the right backend via the registry."""
    _, cpd_dir_path = _make_compound_dir(temp_work_dir, smiles=SMILES_ACETIC_ACID, name="acetic")
    config = ProtonateInterfaceConfig(
        input=CalculationInput(
            results_directory=cpd_dir_path.parent,
            compound_directories=[cpd_dir_path],
        ),
        submission_config=ProtonateSubmissionConfig(submit=False),
        protonate_options=ProtonateOptions(
            backend={"backend": "moka_single_charge", "lower_ph_threshold": 1.0},
        ),
    )
    assert isinstance(
        config.protonate_options.backend, protonation.MokaSingleChargeConfig
    )  # pylint: disable=all
    assert config.protonate_options.backend.lower_ph_threshold == 1.0  # pylint: disable=all


# ===========================================================================
# YAML config round-trip
# ===========================================================================


@pytest.mark.parametrize(
    "backend_dict, expected_backend",
    [
        ({"backend": "moka_blabber", "ph": "6-9", "remove_zero_charged": True}, "moka_blabber"),
        ({"backend": "moka_ionic_species", "lower_ph_threshold": 2.0}, "moka_ionic_species"),
        ({"backend": "moka_single_charge", "upper_ph_threshold": 9.0}, "moka_single_charge"),
    ],
)
def test_config_roundtrip(config_roundtrip, backend_dict, expected_backend):
    # Single discriminated-union field: the fragile case for jsonargparse.
    _cfg, dump = config_roundtrip(
        protonate.load_config, {"protonate_options": {"backend": backend_dict}}
    )
    assert dump["protonate_options"]["backend"]["backend"] == expected_backend


def test_config_roundtrip_default_backend(config_roundtrip):
    # No backend given: stays None at the options level, coalesced downstream.
    _cfg, dump = config_roundtrip(protonate.load_config, {})
    assert dump["protonate_options"]["backend"] is None
