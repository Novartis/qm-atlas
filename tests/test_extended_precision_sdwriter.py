import re
import tempfile

import pytest
from rdkit import Chem  # type: ignore

from qm_atlas.command_line.utils.extended_precision_sdwriter import ExtendedPrecisionSDWriter

test_ctab = """
     RDKit          3D

  0  0  0  0  0  0  0  0  0  0999 V3000
M  V30 BEGIN CTAB
M  V30 COUNTS 6 6 0 0 0
M  V30 BEGIN ATOM
M  V30 1 C 9.108151111110001 -9.924341111110000 -2.733731111110000 0
M  V30 2 C 14.251771111109999 4.443791111110000 -1.143111111110000 0
M  V30 3 C 2.284001111110000 13.193901111110000 0.485701111110000 0
M  V30 4 C -10.664941111110000 6.482351111110000 -3.681571111110000 0
M  V30 5 C -13.233911111110000 -6.170071111110000 4.189661111110000 0
M  V30 6 N -0.583911111110000 -11.656241111110001 7.868421111110000 0
M  V30 END ATOM
M  V30 BEGIN BOND
M  V30 1 1 1 2
M  V30 2 1 2 3
M  V30 3 1 3 4
M  V30 4 1 4 5
M  V30 5 1 5 6
M  V30 6 1 6 1
M  V30 END BOND
M  V30 END CTAB
M  END
"""
test_mol = Chem.MolFromMolBlock(test_ctab)
assert test_mol


def test_instantiate_sdwriter_with_none_stream() -> None:
    with pytest.raises(ValueError):
        _ = ExtendedPrecisionSDWriter(None)


def test_instantiate_sdwriter_with_file_path() -> None:
    with tempfile.NamedTemporaryFile("w") as tmp:
        with ExtendedPrecisionSDWriter(tmp.name) as writer:
            writer.write(test_mol)
        with open(tmp.name) as hnd:
            sdtext = hnd.read()
        assert "V3000" in sdtext
        assert "111111" in sdtext


def test_instantiate_sdwriter_with_file_obj() -> None:
    with tempfile.NamedTemporaryFile("w") as tmp:
        with ExtendedPrecisionSDWriter(tmp.file) as writer:
            writer.write(test_mol)
        with open(tmp.name) as hnd:
            sdtext = hnd.read()
        assert "V3000" in sdtext
        assert re.search(r"\.(\d){15}\s", sdtext) is not None
        assert "111111" in sdtext


def test_instantiate_sdwriter_with_5_digits() -> None:
    with tempfile.NamedTemporaryFile("w") as tmp:
        with ExtendedPrecisionSDWriter(tmp.file) as writer:
            writer.SetCoordFormat("{:.5f}")
            writer.write(test_mol)
        with open(tmp.name) as hnd:
            sdtext = hnd.read()
        assert "V3000" in sdtext
        assert re.search(r"\.(\d){15}\s", sdtext) is None
        assert re.search(r"\.(\d){5}\s", sdtext) is not None
        assert "111111" not in sdtext


def test_instantiate_sdwriter_as_v2000() -> None:
    with tempfile.NamedTemporaryFile("w") as tmp:
        with ExtendedPrecisionSDWriter(tmp.file) as writer:
            writer.SetForceV3000(False)
            writer.write(test_mol)
        with open(tmp.name) as hnd:
            sdtext = hnd.read()
        assert "V2000" in sdtext
        assert re.search(r"\.(\d){15}\s", sdtext) is None
        assert re.search(r"\.(\d){4}\s", sdtext) is not None
        assert "111111" not in sdtext
