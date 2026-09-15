from pathlib import Path

import numpy as np
import pytest
from context import require_software  # pylint: disable=import-error
from rdkit import Chem

from qm_atlas.wrappers.moka import moka

GLYCINE_SMI = "NCC(=O)O"  # glycine: CHEMBL773

EXP_CENTERS = np.array([4, 0])
EXP_TYPES = np.array(["a", "b"])
EXP_PKAS = np.array([2.21, 9.95])


@require_software("moka")
def test_moka():

    glycine_mol = Chem.MolFromSmiles(GLYCINE_SMI)

    prediction = moka.get_moka_prediction(glycine_mol, model=None)

    centers = np.array([pka.atom_index for pka in prediction.predicted_pkas])
    types = np.array([pka.pka_type for pka in prediction.predicted_pkas])
    pkas = np.array([pka.value for pka in prediction.predicted_pkas])

    assert (centers == EXP_CENTERS).all()
    assert (types == EXP_TYPES).all()
    assert np.allclose(pkas, EXP_PKAS, atol=1.0e-01)


@require_software("moka")
def test_moka_model():

    glycine_mol = Chem.MolFromSmiles(GLYCINE_SMI)

    # make sure the default model can be loaded without errors
    prediction = moka.get_moka_prediction(glycine_mol)

    centers = np.array([pka.atom_index for pka in prediction.predicted_pkas])
    types = np.array([pka.pka_type for pka in prediction.predicted_pkas])

    assert (centers == EXP_CENTERS).all()
    assert (types == EXP_TYPES).all()


@require_software("moka")
def test_moka_error():
    glycine_mol = Chem.MolFromSmiles(GLYCINE_SMI)

    with pytest.raises(RuntimeError):
        moka.get_moka_prediction(glycine_mol, cmd="moka_cli_fake")

    with pytest.raises(RuntimeError):
        fake_model = Path("/some/ridiculous/path.mkd")
        moka.get_moka_prediction(glycine_mol, model=fake_model)
