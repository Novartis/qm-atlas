"""Cresset ``align`` backend for conformer alignment.

Overlays all conformers of a probe molecule onto a reference conformation using
Cresset's command-line ``align`` tool (field-point based alignment). The public
:func:`align_mol_to_reference` mirrors the signature and return contract of the
RDKit backend in :mod:`qm_atlas.tasks.utils.conformer_geometry`, so the two are
interchangeable through :data:`qm_atlas.tasks.align.ALIGNMENT_FUNCTIONS`.

The alignment runs through a temporary-SDF round trip: probe conformers are
written to disk (with Cresset field-point properties stripped, as the original
post-processing script did), ``align`` is invoked on the batch, and the returned
coordinates are transferred back onto a copy of the probe molecule so its atom
ordering and molecule-level properties are preserved.
"""

import copy
import logging
from pathlib import Path

import numpy as np
from ppqm.utils import WorkDir
from rdkit import Chem

from qm_atlas import constants
from qm_atlas.software_environment import SOFTWARE_CONFIG
from qm_atlas.wrappers.common import run_command

_logger = logging.getLogger(__name__)

# Field-point property blocks bloat the SDF and confuse ``align`` when a molecule
# is aligned more than once; the original script stripped them before every call.
_STRIP_PROP_MARKER = "cresset_fieldpoint"

# Conformer-level key used to carry the Cresset similarity score back to the align
# page; the leading underscore keeps it out of the written SDF, where it is
# re-emitted under the user-chosen property name instead.
SIMILARITY_CONF_PROP = "_cresset_similarity"

# SD tag under which ``align`` writes the overall similarity score (0-1); verified
# against cresset 12.0.0 output. Warn (never fail) if a future version renames it.
SIMILARITY_TAG = "Sim"

# Mirrors the invocation in collect_and_align.sh, without ``--writeref`` so the
# output holds only the aligned probe conformers, one record each in input order.
DEFAULT_ALIGN_FLAGS = "--alignments 1 -n -o sdf --readmode fixed_confs"


def _strip_cresset_props(mol: Chem.Mol) -> None:
    """Remove Cresset field-point properties in place (see :data:`_STRIP_PROP_MARKER`)."""
    for prop in list(mol.GetPropNames(includePrivate=True)):
        if _STRIP_PROP_MARKER in prop:
            mol.ClearProp(prop)


def _write_conformers_sdf(mol: Chem.Mol, sdf_file: Path, conf_ids: list[int]) -> None:
    """Write *conf_ids* of *mol* as separate V2000 SDF records with field points stripped."""
    clean = copy.deepcopy(mol)
    _strip_cresset_props(clean)
    with Chem.SDWriter(str(sdf_file)) as writer:
        for conf_id in conf_ids:
            writer.write(clean, confId=conf_id)


def _coordinate_rmsd(
    probe_conf: Chem.Conformer, ref_conf: Chem.Conformer, atom_map: list[int]
) -> float:
    """RMSD between an already-aligned probe conformer and the reference (no re-fit)."""
    probe_pos = np.array([list(probe_conf.GetAtomPosition(i)) for i in range(len(atom_map))])
    ref_pos = np.array([list(ref_conf.GetAtomPosition(atom_map[i])) for i in range(len(atom_map))])
    return float(np.sqrt(np.mean(np.sum((probe_pos - ref_pos) ** 2, axis=1))))


def _extract_similarity(record: Chem.Mol) -> float | None:
    """Return the Cresset similarity score (SD tag :data:`SIMILARITY_TAG`).

    Returns None and logs a warning when the tag is missing or unparseable; the
    alignment itself matters more than recovering the score.
    """
    if not record.HasProp(SIMILARITY_TAG):
        _logger.warning(f"Cresset align output has no '{SIMILARITY_TAG}' tag; skipping similarity")
        return None
    try:
        return float(record.GetProp(SIMILARITY_TAG))
    except ValueError:
        _logger.warning(f"Could not parse Cresset similarity from tag '{SIMILARITY_TAG}'")
        return None


def align_mol_to_reference(
    prb_mol: Chem.Mol,
    ref_mol: Chem.Mol,
    prb_cids: list[int] | None = None,
    ref_cid: int = 0,
    align_flags: str = DEFAULT_ALIGN_FLAGS,
    scr: Path = constants.DEFAULT_SCR,
    keep_files: bool = False,
    align_cmd: str | None = None,
) -> tuple[Chem.Mol, dict[int, float]]:
    """Align conformers of *prb_mol* onto a reference using Cresset ``align``.

    Args:
        prb_mol (Chem.Mol):
            The molecule whose conformers should be aligned.
        ref_mol (Chem.Mol):
            The molecule providing the reference conformation.
        prb_cids (list[int] | None):
            Conformer IDs of *prb_mol* to align. Defaults to all conformers.
        ref_cid (int):
            Conformer ID of the reference conformation in *ref_mol*. Defaults to 0.
        align_flags (str):
            Command-line flags passed to ``align`` (before the two SDF paths).
            Defaults to :data:`DEFAULT_ALIGN_FLAGS`.
        scr (Path):
            Scratch directory for the temporary SDF files. Defaults to
            ``constants.DEFAULT_SCR``.
        keep_files (bool):
            Keep the temporary working directory for debugging. Defaults to False.
        align_cmd (str | None):
            The ``align`` command. If None, resolved from the software configuration.

    Raises:
        RuntimeError: If ``align`` fails or returns an unexpected number of records.

    Returns:
        tuple[Chem.Mol, dict[int, float]]: A copy of *prb_mol* carrying the aligned
        conformers (original conformer IDs preserved) and a mapping from conformer
        ID to the RMSD against the reference conformation.
    """
    if prb_cids is None:
        prb_cids = [conf.GetId() for conf in prb_mol.GetConformers()]

    software_env_manager = SOFTWARE_CONFIG.get_environment_manager()
    if align_cmd is None:
        align_cmd = software_env_manager.get_command("cresset", "align")

    temp = WorkDir(dir=scr, prefix="cresset_align_", keep=keep_files)
    work_dir = temp.get_path()

    ref_sdf = work_dir / "_ref.sdf"
    probe_sdf = work_dir / "_probe.sdf"
    out_sdf = work_dir / "_aligned.sdf"

    _write_conformers_sdf(ref_mol, ref_sdf, [ref_cid])
    _write_conformers_sdf(prb_mol, probe_sdf, prb_cids)

    cmd = f"{align_cmd} {align_flags} {ref_sdf.name} {probe_sdf.name}"
    cmd = f"cd {work_dir}; " + cmd
    _logger.debug(cmd)

    env = software_env_manager.get_run_environment("cresset")
    stdout, _ = run_command(cmd, env=env)

    with open(out_sdf, "w", encoding="utf-8") as f:
        f.write(stdout or "")

    supplier = Chem.SDMolSupplier(str(out_sdf), removeHs=False, sanitize=True)
    aligned_records = [m for m in supplier if m is not None]
    if len(aligned_records) != len(prb_cids):
        raise RuntimeError(
            f"Cresset align returned {len(aligned_records)} records for "
            f"{len(prb_cids)} input conformers"
        )

    # Transfer aligned coordinates back onto a copy of the probe so atom ordering
    # and molecule-level properties survive; align preserves atom order per record.
    result = copy.deepcopy(prb_mol)
    for conf in list(result.GetConformers()):
        if conf.GetId() not in set(prb_cids):
            result.RemoveConformer(conf.GetId())

    # Atom map from probe indices to reference indices for the RMSD (identity when
    # probe and reference share the same graph, as in the strain workflow).
    atom_map = list(ref_mol.GetSubstructMatch(prb_mol))
    if not atom_map:
        atom_map = list(range(prb_mol.GetNumAtoms()))

    ref_conf = ref_mol.GetConformer(ref_cid)
    rmsd_dict: dict[int, float] = {}
    for cid, record in zip(prb_cids, aligned_records):
        if record.GetNumAtoms() != result.GetNumAtoms():
            raise RuntimeError("Cresset align changed the atom count of a conformer")
        target_conf = result.GetConformer(cid)
        record_conf = record.GetConformer()
        for idx in range(record.GetNumAtoms()):
            target_conf.SetAtomPosition(idx, record_conf.GetAtomPosition(idx))
        rmsd_dict[cid] = _coordinate_rmsd(target_conf, ref_conf, atom_map)
        similarity = _extract_similarity(record)
        if similarity is not None:
            target_conf.SetDoubleProp(SIMILARITY_CONF_PROP, similarity)

    return result, rmsd_dict
