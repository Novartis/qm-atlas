import logging

from ppqm import chembridge
from rdkit import Chem
from rdkit.Chem import Descriptors

from qm_atlas.utils import conformer_is_3d

_logger = logging.getLogger(__name__)

METALS = [
    "Sn",
    "Pt",
    "Te",
    "Pd",
    "Lu",
    "Ge",
    "Zn",
    "Cu",
    "Co",
    "Ni",
    "Fe",
    "Hg",
    "Zr",
    "Mn",
    "Ag",
    "Bi",
    "Cd",
    "Cr",
    "Ti",
    "Al",
    "Au",
    "Mo",
    "V",
    "Mg",
    "In",
    "Ga",
    "Pb",
    "Ca",
    "W",
]


def ignore_molecule(
    molobj,
    max_heavy_atoms: int = 60,
    max_molecular_weight: float = 900.0,
    max_unassigned_stereo_centers: int = 2,
    metals: list[str] | None = None,
    override_checks: bool = False,
    require_hydrogens: bool = False,
    require_3d: bool = False,
) -> str | None:
    """
    Should we just ignore this molecule?

    Checks if molecules includes too heavy atoms or too many unassigned chiral
    centers

    Arguments:
        molobj: rdkit mol object
        max_heavy_atoms: Maximum number of heavy atoms in the molecule
        max_molecular_weight: Maximum molecular weight of the molecule
        max_unassigned_stereo_centers: Maximum number of unassigned stereocenters
        metals: List of heavy metals to ignore
        override_checks: If True, the checks on the size and number of unassigned stereocenters.
            Note that the checks for fragments, metals, and incomplete molecules
            are still performed.
        require_hydrogens: If True, ignore molecules that do not contain hydrogens
        require_3d: If True, ignore molecules that do not have 3D coordinates

    Returns:
        reason (str | None): None or reason for ignoring molecule
    """
    if metals is None:
        metals = METALS

    # Check sanity of smiles
    smiles = chembridge.molobj_to_smiles(molobj)

    # Vgcif len(Smi) < 2 or '*' in Smi or 'R' in Smi: continue
    if "*" in smiles or "R" in smiles:
        reason = "Not a complete molecule"
        return reason

    if "." in smiles:
        reason = "Too many fragments in molecule"
        return reason

    # Check for metals
    metals_str = "[" + ",".join(metals) + "]"
    metals_query = Chem.MolFromSmarts(metals_str)
    if molobj.HasSubstructMatch(metals_query):
        reason = "Contains heavy metals"
        return reason

    n_heavy_atoms = molobj.GetNumHeavyAtoms()
    _logger.debug(f"Molecule has {n_heavy_atoms} heavy atoms")
    weight = Descriptors.ExactMolWt(molobj)
    _logger.debug(f"Molecule has {weight:.1f} weight")
    n_undefined_centers = chembridge.get_undefined_stereocenters(molobj)
    _logger.debug(f"Molecule has {n_undefined_centers} undefined stereo centers")

    if require_hydrogens and Chem.AddHs(molobj).GetNumAtoms() > molobj.GetNumAtoms():
        reason = "Molecule does not contain hydrogens"
        return reason

    if require_3d and not conformer_is_3d(molobj, 0):
        reason = "Molecule does not have 3D coordinates"
        return reason

    # Num-Heavy-atoms ignore rule
    reason = None
    if n_heavy_atoms > max_heavy_atoms:
        reason = f"Too many heavy atoms: {n_heavy_atoms}"

    elif weight > max_molecular_weight:
        reason = f"Too heavy: {weight:.1f}"

    elif n_undefined_centers > max_unassigned_stereo_centers:
        reason = f"Too many unassigned stereocenteres: {n_undefined_centers}"

    if reason is None:
        return None

    if override_checks:
        _logger.warning(reason)
        return None

    return reason
