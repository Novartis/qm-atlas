"""Conformer alignment task with pluggable backends.

Aligns all conformers of a probe molecule onto a reference conformation. The
backend is selected through an options model carrying a ``backend`` discriminator
field, so additional alignment engines can be added without changing callers.

Currently only the RDKit backend is implemented; it reuses the functions in
:mod:`qm_atlas.tasks.utils.conformer_geometry`.

Adding a new backend means:

1. Adding a new ``AlignmentOptions`` subclass with a unique ``backend`` literal.
2. Registering it in :data:`OPTIONS_CLASS_REGISTRY` and its implementation in
   :data:`ALIGNMENT_FUNCTIONS`.
3. Adapting :data:`AlignmentConfig` accordingly.
"""

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from rdkit import Chem

from qm_atlas.tasks.utils import conformer_geometry

_logger = logging.getLogger(__name__)

RDKIT_NAME = "rdkit"


class AlignmentOptions(BaseModel):
    """Base class for conformer-alignment backends.

    Each concrete backend declares a unique ``backend`` literal that acts as the
    discriminator for (de)serialization within :data:`AlignmentConfig`.
    """


class RdkitAlignmentOptions(AlignmentOptions):
    """Options for the RDKit alignment backend.

    Uses the hydrogen-separated best-RMS overlay from
    :func:`qm_atlas.tasks.utils.conformer_geometry.align_mol_to_reference`, which
    aligns on the heavy-atom skeleton and matches hydrogens afterwards.
    """

    # No tunable fields yet; reject unknown keys as typos.
    model_config = ConfigDict(extra="forbid")

    backend: Literal["rdkit"] = Field(
        default="rdkit",
        description="Alignment backend to use",
    )


OPTIONS_CLASS_REGISTRY: dict[str, type[AlignmentOptions]] = {
    RDKIT_NAME: RdkitAlignmentOptions,
}

# The concrete options type used for the ``backend`` field. With a single backend
# this is just ``RdkitAlignmentOptions``; once a second backend is added turn it
# into ``Annotated[RdkitAlignmentOptions | NewOptions, Field(discriminator="backend")]``.
AlignmentConfig = RdkitAlignmentOptions


def deserialize_alignment_config(data: Any) -> AlignmentOptions:
    """Deserialize an alignment config from a dict using its ``backend`` field."""
    if isinstance(data, AlignmentOptions):
        return data
    if not isinstance(data, dict):
        raise TypeError(f"Cannot deserialize alignment config from {type(data)!r}")

    backend_name = data.get("backend", RDKIT_NAME)
    if backend_name not in OPTIONS_CLASS_REGISTRY:
        raise ValueError(f"Unknown alignment backend: {backend_name!r}")
    return OPTIONS_CLASS_REGISTRY[backend_name](**data)


ALIGNMENT_FUNCTIONS = {
    RDKIT_NAME: conformer_geometry.align_mol_to_reference,
}


def align_to_reference(
    prb_mol: Chem.Mol,
    ref_mol: Chem.Mol,
    config: AlignmentConfig,
    prb_cids: list[int] | None = None,
    ref_cid: int = 0,
) -> tuple[Chem.Mol, dict[int, float]]:
    """Align all conformers of *prb_mol* onto a reference conformation.

    Dispatches to the backend named by ``config.backend``. The returned molecule
    keeps the probe's atom ordering and molecule-level properties, with its
    conformer coordinates rewritten to overlay the reference.

    Args:
        prb_mol (Chem.Mol): Molecule whose conformers should be aligned.
        ref_mol (Chem.Mol): Molecule providing the reference conformation.
        config (AlignmentConfig): Backend options (selects the alignment engine).
        prb_cids (list[int] | None): Conformer IDs of *prb_mol* to align. Defaults
            to all conformers.
        ref_cid (int): Conformer ID of the reference conformation in *ref_mol*.
            Defaults to 0.

    Returns:
        tuple[Chem.Mol, dict[int, float]]: A molecule carrying the aligned probe
        conformers (original conformer IDs preserved) and a mapping from conformer
        ID to the RMSD against the reference conformation.
    """
    align_fn = ALIGNMENT_FUNCTIONS[config.backend]
    options_kwargs = config.model_dump(exclude={"backend"})
    return align_fn(prb_mol, ref_mol, prb_cids=prb_cids, ref_cid=ref_cid, **options_kwargs)
