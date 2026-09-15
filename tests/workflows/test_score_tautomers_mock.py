"""Mocked test for workflows/score_tautomers.generate_tautomers.

The tautomer enumeration, conformer expansion, DFT single points and COSMOtherm
descriptor calculation are all mocked, so the test isolates the scoring and
energy-window selection logic without any external software.
"""

import pandas as pd
import pytest
from rdkit import Chem

from qm_atlas.workflows import score_tautomers


def _mols(*smiles):
    return [Chem.MolFromSmiles(s) for s in smiles]


def _energy_df(min_energy):
    return pd.DataFrame({"E_COSMO+dE+Mu": [min_energy, min_energy + 5.0]})


@pytest.fixture
def patched_pipeline(monkeypatch):
    """Patch every external step so only the selection logic runs."""
    # Conformer expansion is the identity (returns the tautomer unchanged).
    monkeypatch.setattr(
        score_tautomers.fast_conformers, "generate_fast_conformers", lambda mol, **k: mol
    )
    monkeypatch.setattr(
        score_tautomers.calculate_properties,
        "calculate_property_mol",
        lambda *a, **k: ["sp_prop"],
    )
    monkeypatch.setattr(
        score_tautomers.calculate_properties,
        "extract_cosmo_output",
        lambda sp_prop, remove_from_dict=True: "cosmo_out",
    )


def test_generate_tautomers_keeps_within_energy_window(monkeypatch, patched_pipeline):
    t1, t2, t3 = _mols(
        "CCO", "OCC", "CCC"
    )  # ethanol, ethanol, propane (CHEMBL545/.../CHEMBL135416)
    monkeypatch.setattr(score_tautomers.unicon, "get_tautomers", lambda mol: [t1, t2, t3])
    monkeypatch.setattr(
        score_tautomers.tm_utils, "extract_converged", lambda mol, outs: (mol, outs, None)
    )
    # Relative min free energies: 0, 2, 10 kcal/mol.
    monkeypatch.setattr(
        score_tautomers.cosmo_tasks,
        "calculate_descriptors_df",
        MagicSideEffect([_energy_df(0.0), _energy_df(2.0), _energy_df(10.0)]),
    )

    result = score_tautomers.generate_tautomers(Chem.MolFromSmiles("CCO"), energy_threshold=7)

    # Within 7 kcal/mol of the most stable -> t1 and t2 kept, t3 dropped.
    assert result == [t1, t2]


def test_generate_tautomers_drops_unconverged(monkeypatch, patched_pipeline):
    t1, t2, t3 = _mols("CCO", "OCC", "CCC")
    monkeypatch.setattr(score_tautomers.unicon, "get_tautomers", lambda mol: [t1, t2, t3])

    # The second tautomer has no converged conformers -> energy = +inf -> dropped.
    converged = MagicSideEffect(
        [
            (t1, ["cosmo_out"], None),
            (t2, [], None),
            (t3, ["cosmo_out"], None),
        ]
    )
    monkeypatch.setattr(score_tautomers.tm_utils, "extract_converged", converged)
    monkeypatch.setattr(
        score_tautomers.cosmo_tasks,
        "calculate_descriptors_df",
        MagicSideEffect([_energy_df(0.0), _energy_df(1.0)]),  # only t1 and t3 reach here
    )

    result = score_tautomers.generate_tautomers(Chem.MolFromSmiles("CCO"), energy_threshold=7)

    assert result == [t1, t3]


class MagicSideEffect:
    """Callable returning successive values from a list, ignoring its arguments."""

    def __init__(self, values):
        self._values = list(values)
        self._i = 0

    def __call__(self, *args, **kwargs):
        value = self._values[self._i]
        self._i += 1
        return value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
