from pathlib import Path

from rdkit import Chem

from qm_atlas.command_line.utils.extended_precision_sdwriter import ExtendedPrecisionSDWriter
from qm_atlas.constants import DEFAULT_V3000_COORD_FORMAT

# Internal solvation column
COL_COSMO = "cosmo_output"
COL_GRAPHIDX = "graph_idx"


def write_sdf(
    mol: Chem.Mol,
    sdf_file: Path,
    use_v2000: bool = False,
    conf_ids: list[int] | None = None,
):
    if conf_ids is None:
        conf_ids = [-1]

    with ExtendedPrecisionSDWriter(str(sdf_file.resolve())) as writer:
        if use_v2000:
            writer.SetForceV3000(False)
        else:
            writer.SetForceV3000(True)
            writer.SetCoordFormat(DEFAULT_V3000_COORD_FORMAT)

        for conf_id in conf_ids:
            writer.write(mol, confId=conf_id)
