import logging
from pathlib import Path

import pandas as pd
from pandas.errors import ParserError
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from qm_atlas.command_line.file_interface import input_check
from qm_atlas.utils import has_macrocycle

_logger = logging.getLogger(__name__)

COL_NAME = "name"
COL_SMILES = "smiles"
DEFAULT_SEP = ","

# RDKit property used to carry the parent-compound grouping key from input
# readers to the compound-directory creation step.  When set, all molecules
# sharing the same value are registered as multiple states in a single
# compound directory whose name equals this key.
PARENT_KEY_PROP = "_qm_atlas_parent_key"


class InputConfig(BaseModel):
    """Configuration for reading input files and creating compound directories.

    Attributes:
        override_input_checks: Skip validation checks on molecules
        csv_col_name: CSV column name for molecule names
        csv_col_smiles: CSV column name for SMILES strings
        csv_sep: CSV separator character
        max_heavy_atoms: Maximum number of heavy atoms in the molecule
        max_molecular_weight: Maximum molecular weight of the molecule
        max_unassigned_stereo_centers: Maximum number of unassigned stereocenters
        metals: List of heavy metals to ignore
        require_hydrogens: If True, ignore molecules that do not contain hydrogens
        require_3d: If True, ignore molecules that do not have 3D coordinates
    """

    override_input_checks: bool = Field(
        default=False,
        description="Skip validation checks on molecules (heavy atoms, molecular weight, stereocenters)",
    )
    csv_col_name: str = Field(default=COL_NAME, description="CSV column name for molecule names")
    csv_col_smiles: str = Field(
        default=COL_SMILES, description="CSV column name for SMILES strings"
    )
    csv_sep: str = Field(default=DEFAULT_SEP, description="CSV separator character")
    parent_col: str | None = Field(
        default=None,
        description=(
            "Optional CSV column whose value groups rows into a single parent compound. "
            "All rows sharing the same value are registered as separate states in the "
            "same compound directory (directory name = parent value). When unset, each "
            "row becomes its own compound."
        ),
    )
    parent_property: str | None = Field(
        default=None,
        description=(
            "Optional SDF property whose value groups records into a single parent "
            "compound. Same semantics as parent_col but for SDF inputs. When unset, "
            "each unique structure becomes its own compound."
        ),
    )
    detect_states: bool = Field(
        default=False,
        description=(
            "Automatically detect tautomers and protonation states belonging to the "
            "same compound and group them into a single compound directory. Each "
            "molecule is neutralised and its canonical tautomer is computed; molecules "
            "that share the same canonical tautomer are grouped under the first "
            "molecule's name. Has no effect when parent_col or parent_property is set."
        ),
    )
    max_heavy_atoms: int = Field(
        default=60,
        description="Maximum number of heavy atoms in the molecule",
    )
    max_molecular_weight: float = Field(
        default=900.0,
        description="Maximum molecular weight of the molecule",
    )
    max_unassigned_stereo_centers: int = Field(
        default=2,
        description="Maximum number of unassigned stereocenters",
    )
    require_hydrogens: bool = Field(
        default=False,
        description="If True, ignore molecules that do not contain hydrogens",
    )
    require_3d: bool = Field(
        default=False,
        description="If True, ignore molecules that do not have 3D coordinates",
    )


def sanitize_mol(mol: Chem.Mol):
    # sanitization hack as long as there is a bug in rdkit about nitrogens in
    # macrocycles
    # TODO: remove once rdkit bug is fixed.
    if has_macrocycle(mol):
        sanitize_flag = (
            Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_CLEANUPCHIRALITY
        )
    else:
        sanitize_flag = Chem.SanitizeFlags.SANITIZE_ALL
    Chem.SanitizeMol(mol, sanitize_flag)

    # make sure stereochemistry of the input molecule is captured
    Chem.AssignStereochemistry(mol)
    Chem.SetDoubleBondNeighborDirections(mol, mol.GetConformer())
    Chem.AssignAtomChiralTagsFromStructure(mol, replaceExistingTags=False)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)


def validate_and_filter_molecules(
    mols: list[Chem.Mol],
    input_config: InputConfig,
) -> list[Chem.Mol]:
    """Filter out molecules that fail validation checks.

    Args:
        mols: List of molecules to validate
        input_config: Configuration for input validation

    Returns:
        List of valid molecules
    """
    valid_mols = []

    for mol in mols:
        if mol is None:
            continue

        reason = input_check.ignore_molecule(
            mol,
            max_heavy_atoms=input_config.max_heavy_atoms,
            max_molecular_weight=input_config.max_molecular_weight,
            max_unassigned_stereo_centers=input_config.max_unassigned_stereo_centers,
            override_checks=input_config.override_input_checks,
            require_hydrogens=input_config.require_hydrogens,
            require_3d=input_config.require_3d,
        )

        if reason is not None:
            mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else "Unknown"
            _logger.warning(f"Ignoring molecule {mol_name}: {reason}")
            continue

        valid_mols.append(mol)

    return valid_mols


def has_name(mol: Chem.Mol) -> bool:
    """checks whether a molecule has a name. Note that molecules read from sdf
    files have an empty name if the respective field is not set,
    but mol.HasProp("_Name") evaluates to 1.

    Args:
        mol (Chem.Mol): The molecule

    Returns:
        (bool)
    """

    if not mol.HasProp("_Name"):
        return False

    if len(mol.GetProp("_Name")) == 0:
        return False

    return True


def assign_name(mols: list[Chem.Mol], name: str) -> list[Chem.Mol]:
    if len(mols) > 1:
        _logger.error("Cannot assign name from filename for multi-molecule sdf files.")
        _logger.error("Continue with sdf header names")
        return mols

    mol = mols[0]
    mol.SetProp("_Name", name)
    return [mol]


def _get_neutral_canonical_tautomer_smiles(mol: Chem.Mol) -> str | None:
    """Return the canonical SMILES of the neutral canonical tautomer.

    The molecule is stripped of explicit hydrogens and conformers, neutralised,
    then tautomer-canonicalised.  Returns *None* if any step fails.

    Args:
        mol: Input molecule (not modified).

    Returns:
        Canonical SMILES string, or *None* on failure.
    """
    try:
        # Work on topology only: strip explicit H and remove 3-D coordinates
        mol_2d = Chem.RemoveAllHs(mol, sanitize=False)
        mol_2d = Chem.RWMol(mol_2d)
        mol_2d.RemoveAllConformers()
        Chem.SanitizeMol(mol_2d)

        # Neutralise charges to treat different protonation states as equivalent
        neutral = rdMolStandardize.Uncharger().uncharge(mol_2d)

        # Canonicalise tautomer
        canonical = rdMolStandardize.TautomerEnumerator().Canonicalize(neutral)

        return Chem.MolToSmiles(canonical, isomericSmiles=True, canonical=True)
    except Exception:
        return None


def detect_and_assign_parent_keys(mols: list[Chem.Mol]) -> list[Chem.Mol]:
    """Assign :data:`PARENT_KEY_PROP` by grouping tautomers and protonation states.

    Each molecule is neutralised and its canonical tautomer is computed.  All
    molecules that share the same canonical tautomer SMILES are grouped under
    the same parent key (the ``_Name`` of the first molecule encountered for
    that tautomer).  Molecules whose canonical tautomer cannot be determined
    are left unchanged and will be treated as standalone compounds.

    Args:
        mols: List of molecules (modified in-place).

    Returns:
        The same list with :data:`PARENT_KEY_PROP` set on grouped molecules.
    """
    canonical_to_parent: dict[str, str] = {}

    for mol in mols:
        mol_name = mol.GetProp("_Name")
        canonical_smiles = _get_neutral_canonical_tautomer_smiles(mol)

        if canonical_smiles is None:
            _logger.warning(
                f"Could not determine canonical tautomer for '{mol_name}'; "
                "treating it as a standalone compound."
            )
            continue

        if canonical_smiles not in canonical_to_parent:
            canonical_to_parent[canonical_smiles] = mol_name

        mol.SetProp(PARENT_KEY_PROP, canonical_to_parent[canonical_smiles])

    return mols


def read_input_sdf(
    sdf_file: Path,
    parent_property: str | None = None,
) -> list[Chem.Mol]:
    """reads in an sdf file with mulitple conformers provided for each molecule.
        Recognizing that a new entry in the sdf file is a different conformation
        of the same molcule is based on the name or the SMILES.

    Args:
        sdf_file (Path):
            The sdf file to read.
        parent_property (str | None):
            Optional SDF property name. When set, each record's value for this
            property is attached to the resulting mol as :data:`PARENT_KEY_PROP`
            so that downstream grouping creates one compound directory per
            parent value.

    Returns:
        List[Chem.Mol]: A list of rdkit molecules with multiple conformations each.
    """

    path_str = str(sdf_file.resolve())
    mol_from_smiles = dict()
    name_from_smiles = dict()

    try:
        with Chem.SDMolSupplier(path_str, removeHs=False, sanitize=False) as supplier:
            for idx, mol in enumerate(supplier):
                if mol is None:
                    _logger.error(f"Could not parse record {idx} in {sdf_file}; skipping it.")
                    continue

                record_name = mol.GetProp("_Name") if has_name(mol) else f"{sdf_file.stem}:{idx}"
                try:
                    sanitize_mol(mol)
                except Chem.rdchem.MolSanitizeException as exc:
                    _logger.error(
                        f"Could not sanitize record {record_name} (index {idx}) in "
                        f"{sdf_file}: {exc}. Skipping this compound."
                    )
                    continue

                smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                if smiles in mol_from_smiles:

                    mol_from_smiles[smiles].AddConformer(mol.GetConformer(), assignId=True)
                    prev_name = name_from_smiles[smiles]
                    if mol.HasProp("_Name") and mol.GetProp("_Name") != prev_name:
                        _logger.warning(
                            f"Structures {mol.GetProp('_Name')} and {prev_name} are identical. Merged with name {prev_name}"
                        )
                    continue

                # set name and make it compatible with the workflow expectations
                if not has_name(mol):
                    filename = sdf_file.stem
                    mol_name = f"{filename}:{idx}"
                else:
                    mol_name = mol.GetProp("_Name")
                new_mol_name = make_name_compatible(mol_name)
                mol.SetProp("_Name", new_mol_name)

                if parent_property is not None:
                    if not mol.HasProp(parent_property):
                        _logger.warning(
                            f"SDF record {new_mol_name} in {sdf_file} is missing the "
                            f"parent property '{parent_property}'; treating it as a "
                            f"standalone compound."
                        )
                    else:
                        parent_key = make_name_compatible(str(mol.GetProp(parent_property)))
                        mol.SetProp(PARENT_KEY_PROP, parent_key)

                mol_from_smiles[smiles] = mol
                name_from_smiles[smiles] = new_mol_name

    except OSError:
        _logger.error(f"Could not read sdf file {sdf_file}. Please check the file and try again.")
        return list()

    return list(mol_from_smiles.values())


def read_input_csv(
    file: Path,
    col_name: str = COL_NAME,
    col_smiles: str = COL_SMILES,
    sep: str = DEFAULT_SEP,
    parent_col: str | None = None,
) -> list[Chem.Mol]:
    """Read CSV file and return a list of RDKit molecule objects.

    When *parent_col* is provided, the value of that column is attached to
    each returned mol as :data:`PARENT_KEY_PROP` so downstream code can group
    rows that share the same parent compound.
    """

    options = dict(
        sep=sep,
        engine="python",
        skipinitialspace=True,  # skip spaces after delimiter
        skip_blank_lines=True,
    )

    usecols = [col_name, col_smiles]
    if parent_col is not None and parent_col not in usecols:
        usecols.append(parent_col)

    try:
        mol_df: pd.DataFrame = pd.read_csv(file, usecols=usecols, **options)  # type: ignore
    except (FileNotFoundError, ValueError, UnicodeDecodeError, ParserError) as e:
        _logger.error(f"Error reading CSV file {file}: {e}")
        return list()

    mols = []
    for _, row in mol_df.iterrows():
        mol = Chem.MolFromSmiles(row[col_smiles])
        if mol is None:
            _logger.warning(f"Could not parse SMILES: {row[col_smiles]}")
            continue

        mol.SetProp("_Name", make_name_compatible(str(row[col_name])))
        if parent_col is not None:
            parent_value = row[parent_col]
            if pd.isna(parent_value) or str(parent_value).strip() == "":
                _logger.warning(
                    f"Row for {row[col_name]} in {file} is missing a value in parent "
                    f"column '{parent_col}'; treating it as a standalone compound."
                )
            else:
                mol.SetProp(PARENT_KEY_PROP, make_name_compatible(str(parent_value)))
        mols.append(mol)

    return mols


def get_molecules_from_directory(
    dir_path: Path,
    config: InputConfig,
) -> list[Chem.Mol]:
    """Read molecules from all SDF and CSV files in a directory.

    Args:
        dir_path: Directory path
        config: InputConfig with CSV column names and separator

    Returns:
        List of RDKit molecule objects
    """
    mols = []

    # Read all SDF files
    for sdf_file in dir_path.glob("*.sdf"):
        mols.extend(read_input_sdf(sdf_file, parent_property=config.parent_property))

    # Read all CSV files
    for csv_file in dir_path.glob("*.csv"):
        mols.extend(
            read_input_csv(
                csv_file,
                col_name=config.csv_col_name,
                col_smiles=config.csv_col_smiles,
                sep=config.csv_sep,
                parent_col=config.parent_col,
            )
        )

    return mols


def read_all_molecules(
    input_paths: list[Path],
    config: InputConfig,
) -> list[Chem.Mol]:
    """Read all molecules from input files and directories.

    Args:
        input_paths: List of input file or directory paths
        config: InputConfig with CSV column names and validation options

    Returns:
        List of validated RDKit molecule objects
    """
    all_mols = []

    for input_path in input_paths:
        input_path = Path(input_path)

        if input_path.is_dir():
            _logger.info(f"Reading molecules from directory: {input_path}")
            mols = get_molecules_from_directory(input_path, config)
        elif input_path.suffix == ".sdf":
            _logger.info(f"Reading molecules from file: {input_path}")
            mols = read_input_sdf(input_path, parent_property=config.parent_property)
        elif input_path.suffix == ".csv":
            _logger.info(f"Reading molecules from file: {input_path}")
            mols = read_input_csv(
                input_path,
                col_name=config.csv_col_name,
                col_smiles=config.csv_col_smiles,
                sep=config.csv_sep,
                parent_col=config.parent_col,
            )
        else:
            _logger.warning(f"Input path could not be read: {input_path}")
            continue

        all_mols.extend(mols)

    filtered_mols = validate_and_filter_molecules(
        all_mols,
        config,
    )

    if config.detect_states and config.parent_col is None and config.parent_property is None:
        _logger.info("Auto-detecting tautomers and protonation states for grouping.")
        filtered_mols = detect_and_assign_parent_keys(filtered_mols)

    return filtered_mols


def make_name_compatible(mol_name: str) -> str:
    """makes a molecule name compatible with the expectations of the workflow.
    There are three rules:
    1. The name should not contain any spaces.
    2. The name must not contain _c, as this is used to indicate the conformations
    which are generated by the workflow.
    3. The name must not contain #, this creates problems in cosmotherm calculations.

    Args:
        mol_name (str): The name of the molecule

    Returns:
        str: The compatible name.
    """

    new_name = mol_name.replace(" ", "_")
    new_name = new_name.replace("_c", "_C")
    new_name = new_name.replace("#", "_number")
    if new_name != mol_name:
        _logger.warning(f"Name {mol_name} contains spaces, _c, or #. Replacing with {new_name}")
    return new_name
