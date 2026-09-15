import re
import sys
from io import TextIOWrapper
from typing import Any, Self

from rdkit import Chem  # type: ignore


class ExtendedPrecisionSDWriter(Chem.SDWriter):  # type: ignore
    PARSE_ATOM_LINE_REGEX = re.compile(
        r"^(M  V30\s+)(\d+)(\s+\w+\s+)([0123456789\-\.eE]+)(\s+)([0123456789\-\.eE]+)(\s+)([0123456789\-\.eE]+)(\s+.*)$"
    )

    def __init__(self, arg1: str | TextIOWrapper) -> None:
        super().__init__(arg1)
        if isinstance(arg1, str):
            self._hnd: TextIOWrapper | None = open(arg1, "w")
        else:
            self._hnd = arg1
        self._coord_format: str | None = "{:.15f}"
        self.SetForceV3000(True)
        self._in_atom_block = False
        self._mol: Chem.Mol | None = None
        self._conf_id = -1

    def __enter__(self, *args: Any, **kwargs: Any) -> Self:
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._hnd is not None:
            if self._hnd not in (sys.stdout, sys.stderr):
                self._hnd.close()
            self._hnd = None
            super().close()

    def flush(self) -> None:
        if self._hnd is None:
            raise OSError("Attempting to flush to a closed stream")
        self._hnd.flush()

    def SetCoordFormat(self, coord_format: str | None) -> None:
        self._coord_format = coord_format

    def GetCoordFormat(self) -> str | None:
        return self._coord_format

    def process_mol_line(self, mol_line: str) -> str:
        if self._mol is None:
            raise RuntimeError("Mol must be set before processing mol lines")
        if self._coord_format is None:
            raise RuntimeError("Coord format must be set before processing mol lines")
        mol_line_rstripped = mol_line.rstrip()
        if mol_line_rstripped.endswith("END ATOM"):
            self._in_atom_block = False
        if self._in_atom_block:
            try:
                match = self.PARSE_ATOM_LINE_REGEX.match(mol_line)
                if match:
                    atom_idx = int(match.group(2))
                    if atom_idx < 1 or atom_idx > self._mol.GetNumAtoms():
                        match = None
                if match:
                    atom_idx -= 1  # type: ignore
                    pos = self._mol.GetConformer(self._conf_id).GetAtomPosition(atom_idx)
                    x = self._coord_format.format(pos.x)
                    y = self._coord_format.format(pos.y)
                    z = self._coord_format.format(pos.z)
                    mol_line = f"{match.group(1)}{match.group(2)}{match.group(3)}{x}{match.group(5)}{y}{match.group(7)}{z}{match.group(9)}"
            except Exception as e:
                print(f"{e}", file=sys.stderr)
        if mol_line_rstripped.endswith("BEGIN ATOM"):
            self._in_atom_block = True
        return mol_line

    def write(self, mol: Chem.Mol, confId: int = -1) -> None:
        if self._hnd is None:
            raise OSError("Attempting to write to a closed stream")
        if mol is None:
            raise TypeError("mol must not be None")
        self._mol = mol
        self._conf_id = confId
        mol_text = super().GetText(mol, confId, self.GetKekulize(), self.GetForceV3000())
        mol_lines = mol_text.split("\n")
        is_v3000 = mol_lines[3].rstrip().endswith("V3000")

        conformer_exists = True
        try:
            mol.GetConformer(confId)
        except ValueError:
            conformer_exists = False

        if is_v3000 and self._coord_format is not None and conformer_exists:
            mol_text = "\n".join(self.process_mol_line(mol_line) for mol_line in mol_lines)
        self._hnd.write(mol_text)
