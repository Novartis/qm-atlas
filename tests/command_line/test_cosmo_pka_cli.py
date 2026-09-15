"""Tests for the cosmo_pka CLI page.

The COSMOtherm pKa wrapper is monkey-patched in every test, so no external
software is required.
"""

import shutil
from unittest.mock import patch

import pytest
from context import create_homedir_tmp_path  # pylint: disable=import-error
from rdkit import Chem
from rdkit.Chem import AllChem

# ===========================================================================
# SMILES Test Strings - Global Constants
# ===========================================================================
SMILES_ACETIC_ACID = "CC(=O)O"
SMILES_ACETATE = "CC(=O)[O-]"
SMILES_ACETATE_SECONDARY = "[CH2-]C(=O)O"
SMILES_ACETIC_CATION = "C[C+](O)O"
SMILES_METHYLAMINE = "CN"
SMILES_METHYLAMMONIUM = "[NH3+]C"

from qm_atlas.command_line.base_config import CalculationInput
from qm_atlas.command_line.file_interface.compound_dir import create_cpd_dir
from qm_atlas.command_line.file_interface.input_file import InputConfig
from qm_atlas.command_line.pages import cosmo_pka
from qm_atlas.command_line.pages.cosmo_pka import (
    CosmoPkaInterfaceConfig,
    CosmoPkaOptions,
    CosmoPkaSubmissionConfig,
    _get_form_and_roles,
    _method_label,
    run_local,
)
from qm_atlas.command_line.pages.read_input import ReadInputConfig
from qm_atlas.command_line.pages.read_input import run as read_input_run


@pytest.fixture
def temp_work_dir():
    tmp_path = create_homedir_tmp_path()
    yield tmp_path
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)


def _add_state(cpd, smiles: str, state_name: str, *, n_conformers: int = 1):
    """Register a state with *n_conformers* dummy result + cosmo files."""
    mol = Chem.MolFromSmiles(smiles)
    charge = Chem.GetFormalCharge(mol)
    input_path = cpd.input_dir / f"{state_name}.sdf"
    input_path.write_text("")
    cpd.registry_handler.register_state(
        name=state_name,
        smiles=Chem.MolToSmiles(mol),
        charge=charge,
        input_file_path=input_path,
    )
    for i in range(1, n_conformers + 1):
        result_sdf = cpd.results_dir / f"{state_name}_c{i:03d}.sdf"
        mol3d = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(mol3d, randomSeed=42 + i)
        writer = Chem.SDWriter(str(result_sdf))
        writer.write(mol3d)
        writer.close()
        result_sdf.with_suffix(".cosmo").write_text("fake cosmo\n")
        cpd.registry_handler.add_result_files(state_name, [result_sdf])


def _make_compound_with_states(temp_work_dir, parent_smiles=SMILES_ACETIC_ACID, name="acetic"):
    results_dir = temp_work_dir / "results"
    csv_file = temp_work_dir / "in.csv"
    csv_file.write_text(f"name,smiles\n{name},{parent_smiles}\n")
    read_input_run(
        ReadInputConfig(
            input_file=[csv_file],
            results_directory=results_dir,
            input_config=InputConfig(),
        )
    )
    cpd_dir_path = results_dir / name
    cpd = create_cpd_dir(cpd_dir_path, create_new=False)
    # The parent (neutral) state was registered by read_input but has no result files yet.
    # Add a cosmo file for it so it participates in pKa calculations.
    parent_state = cpd.registry_handler.find_name_by_smiles(parent_smiles)
    parent_sdf = cpd.results_dir / f"{parent_state}_c001.sdf"
    parent_mol = Chem.AddHs(Chem.MolFromSmiles(parent_smiles))
    AllChem.EmbedMolecule(parent_mol, randomSeed=1)
    writer = Chem.SDWriter(str(parent_sdf))
    writer.write(parent_mol)
    writer.close()
    parent_sdf.with_suffix(".cosmo").write_text("fake cosmo\n")
    cpd.registry_handler.add_result_files(parent_state, [parent_sdf])
    return results_dir, cpd_dir_path, cpd, parent_state


def _build_config(cpd_dir_path, **option_kwargs) -> CosmoPkaInterfaceConfig:
    return CosmoPkaInterfaceConfig(
        input=CalculationInput(
            results_directory=cpd_dir_path.parent,
            compound_directories=[cpd_dir_path],
        ),
        submission_config=CosmoPkaSubmissionConfig(submit=False),
        cosmo_pka_options=CosmoPkaOptions(**option_kwargs),
    )


def test_macro_one_pka_per_adjacent_charge_group(temp_work_dir):
    """Macro mode pools all states of a given charge into one calculation."""
    _, cpd_dir_path, cpd, parent_state = _make_compound_with_states(temp_work_dir)
    # two anion states (charge -1), one cation (charge +1)
    _add_state(cpd, "CC(=O)[O-]", f"{parent_state}_A", n_conformers=2)
    _add_state(cpd, "[CH2-]C(=O)O", f"{parent_state}_A2")
    _add_state(cpd, "C[C+](O)O", f"{parent_state}_BH")

    with patch.object(cosmo_pka.cosmo_tasks, "calculate_pka", return_value=4.76) as mock_pka:
        run_local(_build_config(cpd_dir_path, mode="macro"))

    # Two transitions exist: (+1 -> 0) and (0 -> -1)
    assert mock_pka.call_count == 2

    cpd = create_cpd_dir(cpd_dir_path, create_new=False)
    pka_entries = cpd.get_all_pka_info()
    assert len(pka_entries) == 2
    methods = {p.method for p in pka_entries}
    assert methods == {"cosmotherm_bp-tzvpd_macro"}
    # The anion group should aggregate both anion state names
    anion_entry = next(p for p in pka_entries if set(p.child_states) >= {f"{parent_state}_A"})
    assert set(anion_entry.child_states) == {f"{parent_state}_A", f"{parent_state}_A2"}
    assert anion_entry.parent_states == [parent_state]


def test_micro_pka_per_state_pair(temp_work_dir):
    """Micro mode enumerates each (parent, child) state pair with |dq|=1."""
    _, cpd_dir_path, cpd, parent_state = _make_compound_with_states(temp_work_dir)
    _add_state(cpd, "CC(=O)[O-]", f"{parent_state}_A")
    _add_state(cpd, "[CH2-]C(=O)O", f"{parent_state}_A2")

    with patch.object(cosmo_pka.cosmo_tasks, "calculate_pka", return_value=4.5) as mock_pka:
        run_local(_build_config(cpd_dir_path, mode="micro"))

    # Two pairs: (parent, A) and (parent, A2)
    assert mock_pka.call_count == 2

    cpd = create_cpd_dir(cpd_dir_path, create_new=False)
    pka_entries = cpd.get_all_pka_info()
    assert len(pka_entries) == 2
    methods = {p.method for p in pka_entries}
    assert methods == {"cosmotherm_bp-tzvpd_micro"}
    child_sets = {tuple(p.child_states) for p in pka_entries}
    assert child_sets == {(f"{parent_state}_A",), (f"{parent_state}_A2",)}


def test_macro_skips_when_no_adjacent_charge(temp_work_dir):
    """No pKa entries are written when there are no adjacent charge pairs."""
    _, cpd_dir_path, _cpd, _parent = _make_compound_with_states(temp_work_dir)
    # Only the parent (neutral) is registered with cosmo files.

    with patch.object(cosmo_pka.cosmo_tasks, "calculate_pka") as mock_pka:
        run_local(_build_config(cpd_dir_path, mode="macro"))

    mock_pka.assert_not_called()
    cpd = create_cpd_dir(cpd_dir_path, create_new=False)
    assert cpd.get_all_pka_info() == []


def test_state_without_cosmo_files_is_skipped(temp_work_dir):
    """States with no cosmo files on disk are skipped from the pools."""
    _, cpd_dir_path, cpd, parent_state = _make_compound_with_states(temp_work_dir)
    # register an anion state but DELETE its cosmo file
    _add_state(cpd, "CC(=O)[O-]", f"{parent_state}_A")
    anion_files = cpd.registry_handler.get_result_files(f"{parent_state}_A")
    for sdf in anion_files:
        sdf.with_suffix(".cosmo").unlink()

    with patch.object(cosmo_pka.cosmo_tasks, "calculate_pka") as mock_pka:
        run_local(_build_config(cpd_dir_path, mode="macro"))

    mock_pka.assert_not_called()


def test_options_forwarded_to_wrapper(temp_work_dir):
    """Solvent / level / temperature options are forwarded to calculate_pka."""
    _, cpd_dir_path, cpd, parent_state = _make_compound_with_states(temp_work_dir)
    _add_state(cpd, "CC(=O)[O-]", f"{parent_state}_A")

    with patch.object(cosmo_pka.cosmo_tasks, "calculate_pka", return_value=4.76) as mock_pka:
        run_local(
            _build_config(
                cpd_dir_path,
                mode="macro",
                level="bp-tzvp",
                solvent_name="h2o",
                temperature_Celsius=37.0,
            )
        )

    mock_pka.assert_called_once()
    _, kwargs = mock_pka.call_args
    assert kwargs["level"] == "bp-tzvp"
    assert kwargs["solvent_name"] == "h2o"
    assert kwargs["temperature_Celsius"] == 37.0
    assert kwargs["form"] == "ACID"


# ===========================================================================
# Tests for _get_form_and_roles (unit tests)
# ===========================================================================


@pytest.mark.parametrize(
    "higher_q, lower_q, reverse_form, expected_form, expected_parent_q, expected_child_q",
    [
        # Natural ACID: lower charge is negative
        (-1, -2, False, "ACID", -1, -2),
        (0, -1, False, "ACID", 0, -1),
        # Natural BASE: higher charge is positive
        (1, 0, False, "BASE", 0, 1),
        (2, 1, False, "BASE", 1, 2),
        # reverse_form flips ACID -> BASE
        (0, -1, True, "BASE", -1, 0),
        (-1, -2, True, "BASE", -2, -1),
        # reverse_form flips BASE -> ACID
        (1, 0, True, "ACID", 1, 0),
        (2, 1, True, "ACID", 2, 1),
    ],
)
def test_get_form_and_roles(
    higher_q, lower_q, reverse_form, expected_form, expected_parent_q, expected_child_q
):
    form, parent_q, child_q = _get_form_and_roles(higher_q, lower_q, reverse_form)
    assert form == expected_form
    assert parent_q == expected_parent_q
    assert child_q == expected_child_q


# ===========================================================================
# Tests for _method_label
# ===========================================================================


def test_method_label_defaults():
    """Default options produce a clean label with no extra suffix."""
    opts = CosmoPkaOptions()
    assert _method_label(opts) == "cosmotherm_bp-tzvpd_macro"


def test_method_label_non_default_solvent():
    """Non-water solvent is appended to the label."""
    opts = CosmoPkaOptions(solvent_name="dmso")
    assert _method_label(opts) == "cosmotherm_bp-tzvpd_macro_dmso"


@pytest.mark.parametrize("solvent", ["h2o", "water", "H2O", "Water"])
def test_method_label_water_aliases_not_appended(solvent):
    """h2o and water (any case) are treated as default and not appended."""
    opts = CosmoPkaOptions(solvent_name=solvent)
    assert _method_label(opts) == "cosmotherm_bp-tzvpd_macro"


def test_method_label_non_default_temperature():
    """Non-default temperature is appended as TC=X."""
    opts = CosmoPkaOptions(temperature_Celsius=37.0)
    assert _method_label(opts) == "cosmotherm_bp-tzvpd_macro_TC=37.0"


def test_method_label_solvent_and_temperature():
    """Both non-default solvent and temperature are appended."""
    opts = CosmoPkaOptions(solvent_name="dmso", temperature_Celsius=37.0)
    assert _method_label(opts) == "cosmotherm_bp-tzvpd_macro_dmso_TC=37.0"


def test_method_label_different_level_and_mode():
    """Level and mode are always included."""
    opts = CosmoPkaOptions(level="bp-tzvp", mode="micro")
    assert _method_label(opts) == "cosmotherm_bp-tzvp_micro"


# ===========================================================================
# Tests for BASE form (positive-charge transitions)
# ===========================================================================


def test_macro_base_form_for_cation_transition(temp_work_dir):
    """Positive-charge transition (0->+1) uses BASE form with lower-charge state as parent."""
    _, cpd_dir_path, cpd, parent_state = _make_compound_with_states(
        temp_work_dir, parent_smiles=SMILES_METHYLAMINE, name="methylamine"
    )
    _add_state(cpd, SMILES_METHYLAMMONIUM, f"{parent_state}_BH")

    with patch.object(cosmo_pka.cosmo_tasks, "calculate_pka", return_value=10.6) as mock_pka:
        run_local(_build_config(cpd_dir_path, mode="macro"))

    mock_pka.assert_called_once()
    _, kwargs = mock_pka.call_args
    assert kwargs["form"] == "BASE"
    # In BASE form, parent = lower-charge (neutral) state
    assert len(kwargs["parent_conformers"]) >= 1
    assert len(kwargs["child_conformers"]) >= 1

    cpd = create_cpd_dir(cpd_dir_path, create_new=False)
    pka_entries = cpd.get_all_pka_info()
    assert len(pka_entries) == 1
    assert pka_entries[0].pka_type == "BASE"
    assert pka_entries[0].parent_states == [parent_state]
    assert pka_entries[0].child_states == [f"{parent_state}_BH"]


def test_micro_base_form_for_cation_transition(temp_work_dir):
    """Micro mode also uses BASE form and correct parent/child for positive-charge pairs."""
    _, cpd_dir_path, cpd, parent_state = _make_compound_with_states(
        temp_work_dir, parent_smiles=SMILES_METHYLAMINE, name="methylamine"
    )
    _add_state(cpd, SMILES_METHYLAMMONIUM, f"{parent_state}_BH")

    with patch.object(cosmo_pka.cosmo_tasks, "calculate_pka", return_value=10.6) as mock_pka:
        run_local(_build_config(cpd_dir_path, mode="micro"))

    mock_pka.assert_called_once()
    _, kwargs = mock_pka.call_args
    assert kwargs["form"] == "BASE"

    cpd = create_cpd_dir(cpd_dir_path, create_new=False)
    pka_entries = cpd.get_all_pka_info()
    assert len(pka_entries) == 1
    assert pka_entries[0].pka_type == "BASE"


# ===========================================================================
# Tests for pka_type stored in entries
# ===========================================================================


def test_macro_pka_type_acid_stored(temp_work_dir):
    """pka_type='ACID' is stored for negative-charge transitions."""
    _, cpd_dir_path, cpd, parent_state = _make_compound_with_states(temp_work_dir)
    _add_state(cpd, SMILES_ACETATE, f"{parent_state}_A")

    with patch.object(cosmo_pka.cosmo_tasks, "calculate_pka", return_value=4.76):
        run_local(_build_config(cpd_dir_path, mode="macro"))

    cpd = create_cpd_dir(cpd_dir_path, create_new=False)
    pka_entries = cpd.get_all_pka_info()
    assert len(pka_entries) == 1
    assert pka_entries[0].pka_type == "ACID"


def test_macro_mixed_charges_produces_both_forms(temp_work_dir):
    """A compound with anion and cation states produces both ACID and BASE entries."""
    _, cpd_dir_path, cpd, parent_state = _make_compound_with_states(temp_work_dir)
    _add_state(cpd, SMILES_ACETATE, f"{parent_state}_A")
    _add_state(cpd, SMILES_ACETIC_CATION, f"{parent_state}_BH")

    with patch.object(cosmo_pka.cosmo_tasks, "calculate_pka", return_value=5.0):
        run_local(_build_config(cpd_dir_path, mode="macro"))

    cpd = create_cpd_dir(cpd_dir_path, create_new=False)
    pka_entries = cpd.get_all_pka_info()
    assert len(pka_entries) == 2
    pka_types = {p.pka_type for p in pka_entries}
    assert pka_types == {"ACID", "BASE"}


# ===========================================================================
# Tests for reverse_form
# ===========================================================================


def test_reverse_form_flips_acid_to_base(temp_work_dir):
    """reverse_form=True causes a negative-charge transition to use BASE form."""
    _, cpd_dir_path, cpd, parent_state = _make_compound_with_states(temp_work_dir)
    _add_state(cpd, SMILES_ACETATE, f"{parent_state}_A")

    with patch.object(cosmo_pka.cosmo_tasks, "calculate_pka", return_value=4.76) as mock_pka:
        run_local(_build_config(cpd_dir_path, mode="macro", reverse_form=True))

    mock_pka.assert_called_once()
    _, kwargs = mock_pka.call_args
    assert kwargs["form"] == "BASE"
    # In BASE form, the lower-charge (anion) state becomes the parent
    cpd = create_cpd_dir(cpd_dir_path, create_new=False)
    pka_entries = cpd.get_all_pka_info()
    assert pka_entries[0].pka_type == "BASE"
    assert pka_entries[0].parent_states == [f"{parent_state}_A"]
    assert pka_entries[0].child_states == [parent_state]


def test_reverse_form_flips_base_to_acid(temp_work_dir):
    """reverse_form=True causes a positive-charge transition to use ACID form."""
    _, cpd_dir_path, cpd, parent_state = _make_compound_with_states(
        temp_work_dir, parent_smiles=SMILES_METHYLAMINE, name="methylamine"
    )
    _add_state(cpd, SMILES_METHYLAMMONIUM, f"{parent_state}_BH")

    with patch.object(cosmo_pka.cosmo_tasks, "calculate_pka", return_value=10.6) as mock_pka:
        run_local(_build_config(cpd_dir_path, mode="macro", reverse_form=True))

    mock_pka.assert_called_once()
    _, kwargs = mock_pka.call_args
    assert kwargs["form"] == "ACID"
    # In ACID form, the higher-charge (cation) state becomes the parent
    cpd = create_cpd_dir(cpd_dir_path, create_new=False)
    pka_entries = cpd.get_all_pka_info()
    assert pka_entries[0].pka_type == "ACID"
    assert pka_entries[0].parent_states == [f"{parent_state}_BH"]
    assert pka_entries[0].child_states == [parent_state]


# ===========================================================================
# Tests for multi-conformer cosmo file pooling
# ===========================================================================


def test_macro_all_conformer_cosmo_files_pooled(temp_work_dir):
    """All cosmo files for all states in a charge group are passed as one pool."""
    _, cpd_dir_path, cpd, parent_state = _make_compound_with_states(temp_work_dir)
    # parent has 1 conformer; two anion states with 2 and 3 conformers respectively
    _add_state(cpd, SMILES_ACETATE, f"{parent_state}_A", n_conformers=2)
    _add_state(cpd, SMILES_ACETATE_SECONDARY, f"{parent_state}_A2", n_conformers=3)

    with patch.object(cosmo_pka.cosmo_tasks, "calculate_pka", return_value=4.76) as mock_pka:
        run_local(_build_config(cpd_dir_path, mode="macro"))

    mock_pka.assert_called_once()
    _, kwargs = mock_pka.call_args
    # parent pool: 1 cosmo file; child pool: 2+3 = 5 cosmo files
    assert len(kwargs["parent_conformers"]) == 1
    assert len(kwargs["child_conformers"]) == 5


# ===========================================================================
# Tests for exception handling
# ===========================================================================


def test_failed_calculation_is_skipped_macro(temp_work_dir):
    """A calculate_pka exception in macro mode is caught; other pairs still run."""
    _, cpd_dir_path, cpd, parent_state = _make_compound_with_states(temp_work_dir)
    _add_state(cpd, SMILES_ACETATE, f"{parent_state}_A")
    _add_state(cpd, SMILES_ACETIC_CATION, f"{parent_state}_BH")

    call_count = 0

    def fail_first_call(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("simulated COSMOtherm failure")
        return 5.0

    with patch.object(cosmo_pka.cosmo_tasks, "calculate_pka", side_effect=fail_first_call):
        run_local(_build_config(cpd_dir_path, mode="macro"))

    # Two pairs (ACID and BASE); one failed, one succeeded
    assert call_count == 2
    cpd = create_cpd_dir(cpd_dir_path, create_new=False)
    pka_entries = cpd.get_all_pka_info()
    assert len(pka_entries) == 1


def test_failed_calculation_is_skipped_micro(temp_work_dir):
    """A calculate_pka exception in micro mode is caught; other pairs still run."""
    _, cpd_dir_path, cpd, parent_state = _make_compound_with_states(temp_work_dir)
    _add_state(cpd, SMILES_ACETATE, f"{parent_state}_A")
    _add_state(cpd, SMILES_ACETATE_SECONDARY, f"{parent_state}_A2")

    call_count = 0

    def fail_first_call(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("simulated COSMOtherm failure")
        return 4.5

    with patch.object(cosmo_pka.cosmo_tasks, "calculate_pka", side_effect=fail_first_call):
        run_local(_build_config(cpd_dir_path, mode="micro"))

    assert call_count == 2
    cpd = create_cpd_dir(cpd_dir_path, create_new=False)
    pka_entries = cpd.get_all_pka_info()
    assert len(pka_entries) == 1


def test_config_roundtrip(config_roundtrip):
    """COSMO pKa options round-trip with non-default mode / solvent / form."""
    _cfg, dump = config_roundtrip(
        cosmo_pka.load_config,
        {
            "cosmo_pka_options": {
                "mode": "micro",
                "level": "bp-tzvp",
                "solvent_name": "dimethylsulfoxide",
                "temperature_Celsius": 37.0,
                "reverse_form": True,
            }
        },
    )
    opts = dump["cosmo_pka_options"]
    assert opts["mode"] == "micro"
    assert opts["reverse_form"] is True
