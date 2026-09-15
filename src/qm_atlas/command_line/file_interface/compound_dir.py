import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from filelock import FileLock
from rdkit import Chem

from qm_atlas.command_line.utils import format_utils
from qm_atlas.tasks.common import Property, ScalarProperty

_logger = logging.getLogger(__name__)


REF_NAME_ADDITION = "_ref"
OPT_NAME_ADDITION = "_opt"

# SDF property that preserves the original name of a user-provided reference
# conformation, so its origin remains traceable after files are renamed.
INPUT_NAME_PROPERTY = "Input_Name"

CSV_CONFORMER_FILE = "{mol_name}_conformer_properties.csv"
CSV_MOLECULE_FILE = "{mol_name}_properties.csv"
CSV_PKA_FILE = "{mol_name}_pka.csv"
CSV_SOLUBILITY_SCREENING_FILE = "{mol_name}_solubility_screening.csv"
CSV_COCRYSTAL_SCREENING_FILE = "{mol_name}_cocrystal_screening.csv"

CONFORMER_COLUMN = "Conformation"
STATE_COLUMN = "State_Name"
PKA_TYPE_COLUMN = "PKA_Type"
PKA_VALUE_COLUMN = "PKA_Value"
PKA_METHOD_COLUMN = "Method"
PKA_PARENT_STATES_COLUMN = "Parent_States"
PKA_CHILD_STATES_COLUMN = "Child_States"
PKA_ATOM_INDEX_COLUMN = "Atom_Index"
PKA_STATES_SEPARATOR = "|"  # Separator for list entries in CSV

INPUT_DIR = "input"
REFERENCE_DIR = "reference_conformations"
RESULTS_DIR = "results"
REGISTRY_FILE = "registry.json"
TRACE_DIR = "trace_files"


@dataclass
class PkaInfo:
    pka_type: str
    pka_value: float
    method: str
    parent_states: list[str]
    child_states: list[str]
    atom_index: int | None = None


def _parse_atom_index(row: pd.Series) -> int | None:
    """Read the Atom_Index column from a pKa CSV row, tolerating missing values."""
    if PKA_ATOM_INDEX_COLUMN not in row.index:
        return None
    value = row[PKA_ATOM_INDEX_COLUMN]
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def canonical_smiles(mol: Chem.Mol) -> str:
    """Return the isomeric canonical SMILES used as the registry chemical identifier.

    Hydrogens are removed before generation so the identifier is independent of
    the explicit-H state of the molecule. Stereochemistry and isotopes are
    retained.

    Args:
        mol (Chem.Mol): The molecule to canonicalise.

    Returns:
        str: The isomeric canonical SMILES.
    """
    return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True, canonical=True)


def stereo_insensitive_smiles(mol: Chem.Mol) -> str:
    """Return a stereo-insensitive canonical SMILES for chemical matching fallback.

    Uses ``isomericSmiles=False``, which drops stereochemistry and isotope
    labels while retaining connectivity and formal charge. Retaining charge
    keeps distinct protonation states separate, which matters for a registry
    that tracks protomers and tautomers as individual states.

    Args:
        mol (Chem.Mol): The molecule to canonicalise.

    Returns:
        str: The stereo-insensitive canonical SMILES.
    """
    return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=False, canonical=True)


def stereo_insensitive_from_smiles(smiles: str) -> str | None:
    """Derive a stereo-insensitive canonical SMILES from a stored SMILES string.

    This lets callers compare a query molecule against the registry's stored
    (isomeric) SMILES without persisting a second identifier: the stored SMILES
    is parsed and re-emitted with ``isomericSmiles=False`` using the same code
    path as :func:`stereo_insensitive_smiles`.

    Args:
        smiles (str): A (typically isomeric) SMILES string from the registry.

    Returns:
        str | None: The stereo-insensitive canonical SMILES, or ``None`` if the
            input SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)


@dataclass
class ReferenceRecord:
    """Links a reference input file to its result files.

    Paths are stored as absolute :class:`Path` objects internally.
    ``to_dict`` converts them to POSIX strings relative to *main_dir*
    so the JSON stays portable.  ``from_dict`` resolves the relative
    strings back to absolute paths.

    Attributes:
        input_file: Absolute path to the reference input.
        result_files: Absolute paths to result files produced from this
            reference.
    """

    input_file: Path
    result_files: list[Path] = field(default_factory=list)

    def to_dict(self, main_dir: Path) -> dict:
        """Serialise to a dict with paths relative to *main_dir*."""
        return {
            "input_file": self.input_file.relative_to(main_dir).as_posix(),
            "result_files": [p.relative_to(main_dir).as_posix() for p in self.result_files],
        }

    @classmethod
    def from_dict(cls, ref_dict: dict, main_dir: Path) -> "ReferenceRecord":
        """Deserialise from a dict, resolving paths against *main_dir*."""
        return cls(
            input_file=main_dir / ref_dict["input_file"],
            result_files=[main_dir / p for p in ref_dict.get("result_files", [])],
        )


@dataclass
class RegistryEntry:
    """Dataclass to hold information about a tautomer or protonation state
    registered in the compound directory.

    Paths are stored as absolute :class:`Path` objects internally.
    ``to_dict`` converts them to POSIX strings relative to *main_dir*;
    ``from_dict`` resolves the relative strings back to absolute paths.

    Attributes:
        name: The name of the state (e.g. NVP123, NVP123_A, NVP123_BH)
        smiles: The SMILES representation of the state
        charge: The formal charge of the state
        input_file: Absolute path to the input file
        result_files: Absolute paths to result files for this state
        references: List of :class:`ReferenceRecord` entries, each linking
            a reference input file to its result files.
        description: Free-text description of the state
    """

    name: str
    smiles: str
    charge: int
    input_file: Path
    result_files: list[Path] = field(default_factory=list)
    references: list[ReferenceRecord] = field(default_factory=list)
    description: str = ""

    def to_dict(self, main_dir: Path) -> dict:
        """Serialise to a dict with paths relative to *main_dir*."""
        return {
            "name": self.name,
            "smiles": self.smiles,
            "charge": self.charge,
            "input_file": self.input_file.relative_to(main_dir).as_posix(),
            "result_files": [p.relative_to(main_dir).as_posix() for p in self.result_files],
            "references": [ref.to_dict(main_dir) for ref in self.references],
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, ref_dict: dict, main_dir: Path) -> "RegistryEntry":
        """Deserialise from a dict, resolving paths against *main_dir*."""
        refs_raw = ref_dict.get("references", [])
        references = [ReferenceRecord.from_dict(r, main_dir) for r in refs_raw]
        return cls(
            name=ref_dict["name"],
            smiles=ref_dict["smiles"],
            charge=ref_dict["charge"],
            input_file=main_dir / ref_dict["input_file"],
            result_files=[main_dir / p for p in ref_dict.get("result_files", [])],
            references=references,
            description=ref_dict.get("description", ""),
        )

    def get_result_files(self, include_references: bool = True) -> list[Path]:
        """Returns all result files associated with this registry entry, including those linked through references."""
        if not include_references:
            return list(self.result_files)
        ref_result_files = [file for ref in self.references for file in ref.result_files]
        return self.result_files + ref_result_files


class RegistryHandler:
    """Handles reading and writing of the registry.json file.

    The registry tracks tautomer / protonation states for a compound.  Each
    state is first *registered* (name, SMILES, charge, input file).  Result
    files and reference inputs / results can be added later through dedicated
    methods.

    All path arguments accepted by public methods may be either absolute or
    already relative to ``main_dir``.  They are stored relative to
    ``main_dir`` in JSON and returned as absolute paths.
    """

    def __init__(self, main_dir: Path, registry_file: Path):
        self._main_dir = main_dir
        self._registry_file = registry_file
        self._input_dir = self._main_dir / INPUT_DIR
        self._reference_dir = self._main_dir / REFERENCE_DIR
        self._results_dir = self._main_dir / RESULTS_DIR

    def _registry_file_lock(self):
        lock_file = self._registry_file.with_suffix(".json.lock")
        return FileLock(lock_file)

    def initialize(self):
        """Create an empty registry.json if it does not exist yet."""
        with self._registry_file_lock():
            if not self._registry_file.is_file():
                self._write(dict())

    # -- low-level I/O --------------------------------------------------------

    def get_registry(self) -> dict[str, RegistryEntry]:
        if not self._registry_file.is_file():
            return dict()
        with open(self._registry_file, "r", encoding="utf-8") as rf:
            _registry: dict = json.load(rf)
        return {
            name: RegistryEntry.from_dict(entry_dict, self._main_dir)
            for name, entry_dict in _registry.items()
        }

    def _write(self, registry: dict[str, RegistryEntry]):
        write_dict = {name: entry.to_dict(self._main_dir) for name, entry in registry.items()}
        with open(self._registry_file, "w", encoding="utf-8") as rf:
            json.dump(write_dict, rf, indent=2)

    # -- registration ---------------------------------------------------------

    def register_state(
        self,
        name: str,
        smiles: str,
        charge: int,
        input_file_path: Path,
        description: str = "",
    ):
        """Register a tautomer or protonation state in registry.json.

        Args:
            name: Unique name for the state.
            smiles: SMILES of the state.
            charge: Formal charge.
            input_file_path: Path to the input file (absolute).
            description: Optional free-text description.
        """

        with self._registry_file_lock():
            registry = self.get_registry()
            if name in registry:
                raise ValueError(f"State '{name}' is already registered")
            _logger.info(f"Registering state '{name}' with SMILES: {smiles} (charge: {charge})")
            entry = RegistryEntry(
                name=name,
                smiles=smiles,
                charge=charge,
                input_file=input_file_path,
                description=description,
            )
            registry[name] = entry
            self._write(registry)

    # -- result files ---------------------------------------------------------

    def add_result_files(self, name: str, result_file_paths: list[Path]):
        """Append multiple result file paths to the registered state *name*.

        Args:
            name: Registered state name.
            result_file_paths: Paths to the result files.
        """
        with self._registry_file_lock():
            registry = self.get_registry()
            entry = registry.get(name)
            if entry is None:
                raise ValueError(f"State '{name}' is not registered")
            added_count = 0
            for path in result_file_paths:
                if path not in entry.result_files:
                    entry.result_files.append(path)
                    added_count += 1
            _logger.debug(f"Added {added_count} result file(s) to state '{name}'")
            self._write(registry)

    # -- reference inputs & results -------------------------------------------

    def add_reference_input(self, name: str) -> Path:
        """Add a reference input file to the registered state *name*.

        Creates a new :class:`ReferenceRecord` with no result files yet.

        Args:
            name: Registered state name.

        Returns:
            Path: Absolute path of the newly allocated reference input file.

        Raises:
            ValueError: If *name* is not registered.
        """
        with self._registry_file_lock():
            registry = self.get_registry()
            registry_entry = registry.get(name)
            if registry_entry is None:
                raise ValueError(f"State '{name}' is not registered")

            ref_index = 0
            ref_name = f"{name}{REF_NAME_ADDITION}_{ref_index}"
            while any(ref.input_file.stem == ref_name for ref in registry_entry.references):
                ref_index += 1
                ref_name = f"{name}{REF_NAME_ADDITION}_{ref_index}"

            new_ref_file = self._reference_dir / f"{ref_name}.sdf"
            registry_entry.references.append(ReferenceRecord(input_file=new_ref_file))
            self._write(registry)

        return new_ref_file

    def add_reference_result(
        self,
        name: str,
        reference_input_file: Path,
    ) -> Path:
        """Add a result file produced from a specific reference input.

        The *reference_input_file* must already have been added via
        :meth:`add_reference_input`.  This keeps full traceability from
        every reference result back to the reference input that produced it.

        Args:
            name: Registered state name.
            reference_input_file: The reference input this result came from.

        Raises:
            ValueError: If *name* is not registered or *reference_input_file*
                has not been added.
        """
        with self._registry_file_lock():
            registry = self.get_registry()
            entry = registry.get(name, None)
            if entry is None:
                raise ValueError(f"State '{name}' is not registered")
            for ref in entry.references:
                if ref.input_file.resolve() == reference_input_file.resolve():
                    break
            else:
                available_refs = [str(ref.input_file) for ref in entry.references]
                _logger.error(
                    f"Reference input not found for state '{name}'. "
                    f"Searched for: {reference_input_file}. "
                    f"Available reference inputs: {available_refs}"
                )
                raise ValueError(
                    f"Reference input '{reference_input_file}' not found for state '{name}'"
                )

            opt_idx = 0
            ref_opt_name = f"{reference_input_file.stem}{OPT_NAME_ADDITION}_{opt_idx}"
            while any(Path(f).stem == ref_opt_name for f in ref.result_files):
                opt_idx += 1
                ref_opt_name = f"{reference_input_file.stem}{OPT_NAME_ADDITION}_{opt_idx}"

            result_file_path = self._results_dir / f"{ref_opt_name}.sdf"
            ref.result_files.append(result_file_path)
            self._write(registry)
        return result_file_path

    # -- queries --------------------------------------------------------------

    def get_entry(self, name: str) -> RegistryEntry:
        """Return the entry for *name*, raising ``ValueError`` if not found."""
        registry = self.get_registry()
        if name not in registry:
            raise ValueError(f"State '{name}' is not registered")
        return registry[name]

    def get_registered_names(self, charge: int | None = None) -> list[str]:
        """Returns registered state names, optionally filtered by *charge*."""
        registry = self.get_registry()
        if charge is None:
            return list(registry.keys())
        return [name for name, entry in registry.items() if entry.charge == charge]

    def get_registered_smiles(self, charge: int | None = None) -> list[str]:
        """Returns registered SMILES strings, optionally filtered by *charge*."""
        registry = self.get_registry()
        if charge is None:
            return [entry.smiles for entry in registry.values()]
        return [entry.smiles for entry in registry.values() if entry.charge == charge]

    def find_name_by_smiles(self, smiles: str) -> str:
        """Find the registered state name corresponding to a given SMILES string.

        Args:
            smiles: The SMILES string to search for.

        Returns:
            The name of the registered state with the given SMILES.

        Raises:
            ValueError: If no registered state has the given SMILES.
        """
        registry = self.get_registry()
        for name, entry in registry.items():
            if entry.smiles == smiles:
                return name
        # Log all available SMILES for debugging
        available_smiles = {name: entry.smiles for name, entry in registry.items()}
        _logger.debug(
            f"SMILES search failed. Searched for: '{smiles}'. "
            f"Available registered states: {available_smiles}"
        )
        raise ValueError(f"No registered state found with SMILES '{smiles}'")

    def find_name_by_mol(self, mol: Chem.Mol, allow_stereo_fallback: bool = True) -> str:
        """Find the registered state name for a molecule via two-tier matching.

        First attempts an exact match on the isomeric canonical SMILES (the
        identifier stored at registration). If that fails and
        *allow_stereo_fallback* is set, logs a warning and falls back to a
        stereo-insensitive match (see :func:`stereo_insensitive_smiles`), which
        tolerates stereochemistry that may have been perceived from 3D
        coordinates but was absent from the registered 2D input.

        Args:
            mol (Chem.Mol): The molecule to identify.
            allow_stereo_fallback (bool): When ``False``, only an exact isomeric
                SMILES match is accepted. Callers resolving ownership across
                several directories use this to try exact matches first, so a
                stereo-insensitive collision with a sibling stereoisomer cannot
                mask a perfect match.

        Returns:
            str: The name of the matching registered state.

        Raises:
            ValueError: If no state matches, or if the stereo-insensitive
                fallback matches more than one state (ambiguous).
        """
        query_isomeric = canonical_smiles(mol)
        try:
            return self.find_name_by_smiles(query_isomeric)
        except ValueError:
            pass

        if not allow_stereo_fallback:
            raise ValueError(f"No registered state found with exact SMILES '{query_isomeric}'")

        query_flat = stereo_insensitive_smiles(mol)
        registry = self.get_registry()
        matches = [
            name
            for name, entry in registry.items()
            if stereo_insensitive_from_smiles(entry.smiles) == query_flat
        ]

        if len(matches) == 1:
            _logger.warning(
                f"No exact SMILES match for '{query_isomeric}'; matched state "
                f"'{matches[0]}' using stereo-insensitive SMILES '{query_flat}'."
            )
            return matches[0]

        if len(matches) == 0:
            available_smiles = {name: entry.smiles for name, entry in registry.items()}
            _logger.debug(
                f"Chemical match failed. Searched for isomeric '{query_isomeric}' "
                f"and stereo-insensitive '{query_flat}'. "
                f"Available registered states: {available_smiles}"
            )
            raise ValueError(f"No registered state found matching molecule '{query_isomeric}'")

        raise ValueError(
            f"Ambiguous stereo-insensitive match for '{query_flat}': "
            f"candidates {matches}. Refusing to guess."
        )

    def find_state_by_file(self, file_path: Path) -> str:
        """Find the registered state that owns a given file.

        Searches every registered state's input file, result files, reference
        input files, and reference result files for one whose resolved path
        matches *file_path*. This is the SMILES-free routing primitive: a worker
        path always carries a file that was written (and registered) by the
        pipeline, so its owning state can be recovered exactly.

        Args:
            file_path (Path): Absolute or relative path to a registered file.

        Returns:
            str: The name of the state that owns the file.

        Raises:
            ValueError: If no registered state owns the file.
        """
        resolved = file_path.resolve()
        registry = self.get_registry()
        for name, entry in registry.items():
            candidates = [entry.input_file, *entry.result_files]
            for ref in entry.references:
                candidates.append(ref.input_file)
                candidates.extend(ref.result_files)
            if any(candidate.resolve() == resolved for candidate in candidates):
                return name

        raise ValueError(f"No registered state owns file '{file_path}'")

    def get_result_files(self, name: str, include_references: bool = True) -> list[Path]:
        """Returns all result files for a specific registered state, including those linked through references."""
        registry_entry = self.get_entry(name)
        return registry_entry.get_result_files(include_references=include_references)

    def get_reference_inputs(self, name: str) -> list[Path]:
        """Returns all reference input files for a registered state.

        Args:
            name: Registered state name.

        Returns:
            list[Path]: Absolute paths to all reference input files for this state.

        Raises:
            ValueError: If *name* is not registered.
        """
        entry = self.get_entry(name)
        return [ref.input_file for ref in entry.references]

    def get_reference_results(self, name: str, reference_input_path: Path) -> list[Path]:
        """Returns result files for a specific reference input.

        Args:
            name: Registered state name.
            reference_input_path: The reference input file to query.

        Returns:
            list[Path]: Absolute paths to result files associated with this reference input.

        Raises:
            ValueError: If *name* is not registered or *reference_input_path* is not found.
        """
        entry = self.get_entry(name)
        for ref in entry.references:
            if ref.input_file == reference_input_path:
                return ref.result_files
        raise ValueError(f"Reference input '{reference_input_path}' not found for state '{name}'")

    def has_reference_results(self, name: str, reference_input_path: Path) -> bool:
        """Check if a reference input has any result files.

        Args:
            name: Registered state name.
            reference_input_path: The reference input file to query.

        Returns:
            bool: True if results exist, False otherwise (or if the reference input is not found).
        """
        try:
            results = self.get_reference_results(name, reference_input_path)
            return len(results) > 0
        except ValueError:
            return False


class CpdDir:
    """class to handle reading and writing to a single directory used to hold all
    calculations results for a single compound. This directory has the following
    structure:

    -- cpd_dir
      |
      | registry.json (information about states, inputs, references, and results)
      | {compound}_properties.csv (conformer-averaged properties for all states)
      | {compound}_conformer_properties.csv (conformer-dependent properties)
      | {compound}_pka.csv (information about calculated pKa values for the compound)
      |-- input (input files for each tautomer/protonation state)
      |-- reference_conformations (reference conformations for each input state)
      |-- results (3d sdf and cosmo files (if calculated) for all conformers)
      |-- log_files (log files for calculations)
      |-- trace_files (trace files from calculations)

    The sdf files also contain the results of property calculations set as properties
    on the mols. This is also handled within this class. Filelocks are used so that
    multiple processes can safely write to these files using the methods exposed here.
    """

    def __init__(
        self,
        directory: Path,
        registry_handler: RegistryHandler,
    ):

        self.dir = directory
        self.cpd_name = self.dir.stem
        self.log_dir = self.dir / "log_files"
        self.input_dir = self.dir / INPUT_DIR
        self.reference_dir = self.dir / REFERENCE_DIR
        self.results_dir = self.dir / RESULTS_DIR
        self.registry_handler = registry_handler
        self.conformer_csv = self.dir / CSV_CONFORMER_FILE.format(mol_name=self.cpd_name)
        self.molecule_csv = self.dir / CSV_MOLECULE_FILE.format(mol_name=self.cpd_name)
        self.pka_csv = self.dir / CSV_PKA_FILE.format(mol_name=self.cpd_name)
        self.solubility_screening_csv = self.dir / CSV_SOLUBILITY_SCREENING_FILE.format(
            mol_name=self.cpd_name
        )
        self.cocrystal_screening_csv = self.dir / CSV_COCRYSTAL_SCREENING_FILE.format(
            mol_name=self.cpd_name
        )

    def initialize(self):
        """Initialize the compound directory by creating necessary subdirectories and files."""
        self.dir.mkdir(exist_ok=True, parents=True)
        self.input_dir.mkdir(exist_ok=True, parents=False)
        self.log_dir.mkdir(exist_ok=True, parents=False)
        self.reference_dir.mkdir(exist_ok=True, parents=False)
        self.results_dir.mkdir(exist_ok=True, parents=False)
        self.registry_handler.initialize()

    def __str__(self):
        return str(self.dir.resolve())

    # ----- add methods below ----

    def add_input_structure(
        self,
        mol: Chem.Mol,
        overwrite: bool = False,
        description: str = "",
    ) -> Path:
        """Add an input structure to the input/ subdirectory and register it
        in registry.json.

        Args:
            mol (Chem.Mol): The molecule to add as input.
            overwrite (bool): Whether to overwrite an existing file. Defaults to False.

        Returns:
            Path: The path to the written sdf file.

        Raises:
            ValueError: If the molecule does not have a _Name property set, or if a file
                with the same name already exists and overwrite is False.
        """
        if not self.input_dir.is_dir():
            self.initialize()

        if not mol.HasProp("_Name"):
            raise ValueError("Input molecule must have a _Name property set")
        mol_name = mol.GetProp("_Name")
        charge = Chem.rdmolops.GetFormalCharge(mol)
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True, canonical=True)

        sdf_file = self.input_dir / f"{mol_name}.sdf"

        if not overwrite and sdf_file.is_file():
            return sdf_file

        # Register the state in registry.json
        self.registry_handler.register_state(
            mol_name, smiles, charge, sdf_file, description=description
        )

        conf_ids = get_conf_ids(mol)
        format_utils.write_sdf(mol, sdf_file, conf_ids=conf_ids)

        return sdf_file

    def check_registry(self, mol: Chem.Mol) -> str:
        """Verify that a molecule's ``_Name`` corresponds to a registered state.

        The state name is the authoritative identity. A SMILES mismatch (for
        example, stereochemistry perceived from 3D coordinates that was absent
        from the registered 2D input) is logged as a warning rather than raised,
        so it cannot break the pipeline.

        Args:
            mol (Chem.Mol): The molecule to check. Must have a ``_Name`` property set.

        Returns:
            str: The registered state name (the molecule's ``_Name``).

        Raises:
            ValueError: If the molecule has no ``_Name`` property, or the name is
                not registered.
        """
        if not mol.HasProp("_Name"):
            raise ValueError("Molecule must have a _Name property set")
        mol_name = mol.GetProp("_Name")
        registry_entry = self.registry_handler.get_entry(mol_name)
        compound_smiles = canonical_smiles(mol)
        if registry_entry.smiles != compound_smiles:
            _logger.warning(
                f"SMILES mismatch for state '{mol_name}': "
                f"registered '{registry_entry.smiles}', got '{compound_smiles}'. "
                f"Proceeding on the basis of the registered state name."
            )
        return mol_name

    def add_result_conformers(
        self,
        mol: Chem.Mol,
        overwrite: bool = False,
    ) -> list[Path]:
        """Add a conformer to the results/ subdirectory.

        Args:
            mol (Chem.Mol):
                The molecule to add. All conformations are added, and the _Name property is used
                as the base name for the sdf files and to check the registry for the associated
                state.
            overwrite (bool): Whether to overwrite an existing file. Defaults to False.

        Returns:
            list[Path]: The paths to the written sdf files.
        """
        mol_name = self.check_registry(mol)
        conf_ids = get_conf_ids(mol)

        if not self.results_dir.is_dir():
            self.initialize()

        sdf_files = list()
        for conf_id in conf_ids:

            write_mol = Chem.Mol(mol)
            sdf_file = self.results_dir / f"{mol_name}_c{conf_id}.sdf"
            write_mol.SetProp("_Name", f"{mol_name}_c{conf_id}")

            if not overwrite:
                if sdf_file.is_file():
                    raise ValueError("SDF file already exists")

            format_utils.write_sdf(write_mol, sdf_file, conf_ids=[conf_id])
            sdf_files.append(sdf_file)

        self.registry_handler.add_result_files(mol_name, sdf_files)

        return sdf_files

    def add_reference_input(
        self,
        mol: Chem.Mol,
        conf_id: int,
    ) -> Path:
        """Sets a reference structure for the compound directory.

        Reference conformations are stored in the reference_conformations/
        subdirectory with the naming convention {compound}_{REF_NAME_ADDITION}_{N}.sdf.

        Args:
            mol (Chem.Mol): The molecule to use as a reference
        """
        if not self.reference_dir.is_dir():
            self.initialize()

        conf_ids = get_conf_ids(mol)
        if conf_id not in conf_ids:
            raise ValueError("Provided Molecule does not contain the provided conf_id")

        mol_name = self.find_state_by_mol(mol)
        ref_input_sdf = self.registry_handler.add_reference_input(mol_name)

        # Preserve the original input name as a property so the origin of this
        # reference conformation stays traceable after the file is renamed to
        # the state-based convention. Written on a copy to avoid mutating the
        # caller's molecule.
        write_mol = Chem.Mol(mol)
        if not write_mol.HasProp(INPUT_NAME_PROPERTY) and write_mol.HasProp("_Name"):
            write_mol.SetProp(INPUT_NAME_PROPERTY, write_mol.GetProp("_Name"))

        format_utils.write_sdf(write_mol, ref_input_sdf, conf_ids=[conf_id])
        return ref_input_sdf

    def add_reference_result(
        self,
        mol: Chem.Mol,
        reference_input_file: Path,
    ) -> Path:
        """Add a result conformer for a specific reference input.

        Args:
            mol (Chem.Mol):
                The molecule to add. All conformations are added, and the _Name property is used
                as the base name for the sdf files and to check the registry for the associated state.
            reference_input_file (Path):
                The reference input file associated with the result conformers.
        """
        mol_name = self.find_state_by_file(reference_input_file)
        conf_ids = [conf.GetId() for conf in mol.GetConformers()]
        if len(conf_ids) != 1:
            raise ValueError("Reference result conformer must have exactly one conformer")

        ref_result_file = self.registry_handler.add_reference_result(
            name=mol_name, reference_input_file=reference_input_file
        )
        ref_opt_name = ref_result_file.stem
        write_mol = Chem.Mol(mol)
        write_mol.SetProp("_Name", ref_opt_name)
        format_utils.write_sdf(write_mol, ref_result_file, conf_ids=conf_ids)

        return ref_result_file

    # ---- logging-related methods below ----

    def get_trace_dir(self) -> Path:
        """returns the directory where the trace files are stored. This is a subdirectory
        of the compound directory.

        Returns: Path
            The path to the trace directory
        """
        trace_dir = self.dir / TRACE_DIR
        trace_dir.mkdir(exist_ok=True, parents=False)
        return trace_dir

    def get_associated_log_file(self, sdf_file: Path) -> Path:

        if not sdf_file.suffix == ".sdf":
            raise ValueError("Input file must be sdf")
        filename = sdf_file.stem
        return self.log_dir / f"{filename}.log"

    def get_associated_cosmo_file(self, sdf_file: Path) -> Path:

        if not sdf_file.suffix == ".sdf":
            raise ValueError("Input file must be sdf")
        return sdf_file.with_suffix(".cosmo")

    def get_pka_log_file(self):

        return self.log_dir / "cosmo_pka.log"

    # ----- get methods below ----

    def get_input_sdf_files(self) -> list[Path]:
        """Returns all input sdf files from the input/ subdirectory.

        Returns:
            list[Path]: A list of sdf file paths.
        """
        if not self.input_dir.is_dir():
            return list()
        return [file for file in self.input_dir.iterdir() if file.suffix == ".sdf"]

    def get_registered_smiles(self) -> list[str]:
        """Returns the registered SMILES strings from registry.json.

        Returns:
            list[str]: A list of registered SMILES strings.
        """
        return self.registry_handler.get_registered_smiles()

    def find_state_by_file(self, file_path: Path) -> str:
        """Return the registered state that owns *file_path*.

        SMILES-free routing primitive: resolves a worker-path file (input,
        result, or reference file) back to its owning state via the registry.

        Args:
            file_path (Path): Path to a registered file within this compound directory.

        Returns:
            str: The owning state name.

        Raises:
            ValueError: If no registered state owns the file.
        """
        return self.registry_handler.find_state_by_file(file_path)

    def find_state_by_mol(self, mol: Chem.Mol, allow_stereo_fallback: bool = True) -> str:
        """Return the registered state chemically matching *mol*.

        Uses two-tier matching (exact isomeric SMILES, then a stereo-insensitive
        fallback). Intended for molecules that are not yet backed by a
        registered file; prefer :meth:`find_state_by_file` whenever a file path
        is available.

        Args:
            mol (Chem.Mol): The molecule to identify.
            allow_stereo_fallback (bool): When ``False``, only an exact isomeric
                SMILES match is accepted (see :meth:`find_name_by_mol`).

        Returns:
            str: The matching state name.

        Raises:
            ValueError: If no state matches or the fallback match is ambiguous.
        """
        return self.registry_handler.find_name_by_mol(
            mol, allow_stereo_fallback=allow_stereo_fallback
        )

    def get_result_files(self, name: str, include_references: bool = True) -> list[Path]:
        """Returns all sdf files for conformers matching the given base name from the results/ subdirectory.

        Args:
            name (str): The base name to match conformers against.
            include_references (bool): Whether to also include reference-optimization
                result files linked to this state. Set False to inspect only the
                conformer-expansion results.

        Returns:
            list[Path]: A list of sdf file paths.
        """
        result_files = self.registry_handler.get_result_files(
            name, include_references=include_references
        )

        returned_sdf_files = list()
        for file in result_files:
            if file.suffix != ".sdf":
                raise RuntimeError("Result files must be sdf files")

            if not file.is_file():
                _logger.warning(f"Result file {file} listed in registry but not found on disk")
                continue

            returned_sdf_files.append(file)

        return returned_sdf_files

    def get_reference_inputs(self, state_name: str) -> list[Path]:
        """Returns all reference input files for a registered state.

        Args:
            state_name (str): The registered state name.

        Returns:
            list[Path]: Absolute paths to all reference input files for this state.
        """
        return self.registry_handler.get_reference_inputs(state_name)

    def get_reference_results(self, state_name: str, reference_input_file: Path) -> list[Path]:
        """Returns all results associated with a reference input.

        Args:
            state_name (str): The registered state name.
            reference_input_file (Path): The reference input file to query.

        Returns:
            list[Path]: A list of sdf file paths.

        Raises:
            ValueError: If the reference input is not found for this state.
        """
        result_files = self.registry_handler.get_reference_results(
            state_name, reference_input_file
        )

        sdf_files = list()
        for file in result_files:
            if file.suffix != ".sdf":
                raise RuntimeError("Reference result files must be sdf files")

            if not file.is_file():
                _logger.warning(
                    f"Reference result file {file} listed in registry but not found on disk"
                )
                continue

            sdf_files.append(file)

        return sdf_files

    def has_reference_results(self, state_name: str, reference_input_file: Path) -> bool:
        """Check if a reference input has any result files.

        Args:
            state_name (str): The registered state name.
            reference_input_file (Path): The reference input file to query.

        Returns:
            bool: True if results exist for this reference input, False otherwise.
        """
        return self.registry_handler.has_reference_results(state_name, reference_input_file)

    def extract_mol(self, sdf_file: Path) -> Chem.Mol:

        with Chem.SDMolSupplier(str(sdf_file.resolve()), removeHs=False) as supplier:
            if len(supplier) == 0:
                raise ValueError(f"No molecules found in {sdf_file}")
            mol = supplier[0]
            if mol is None:
                raise ValueError(f"Failed to parse molecule from {sdf_file}")

        return mol  # type: ignore

    # ---- check methods below ----

    def has_results(self) -> bool:
        """Check if any result files exist in the results/ directory.

        Returns:
            bool: True if there are result sdf files, False otherwise.
        """
        registered_names = self.registry_handler.get_registered_names()
        for name in registered_names:
            if self.has_results_for_name(name):
                return True
        return False

    def has_results_for_name(self, name: str) -> bool:
        """Check if result files exist for a specific compound/state name.

        Args:
            name (str): The base name to check for results.

        Returns:
            bool: True if there are matching result sdf files, False otherwise.
        """
        if not self.results_dir.is_dir():
            return False

        result_files = self.get_result_files(name)
        return len(result_files) > 0

    # ----- add property methods below ----

    def add_cosmo_result(self, sdf_file: Path, cosmo_output: str, overwrite: bool = False):
        """

        Args:
            sdf_file (Path):
                The sdf file to work on
            cosmo_output (str):
                The cosmo output to add as a property
            overwrite (bool):
                Whether to overwrite an existing cosmo output file.
        """
        cosmo_file = self.get_associated_cosmo_file(sdf_file)
        if cosmo_file.exists() and not overwrite:
            raise FileExistsError(
                f"Cosmo output file {cosmo_file} already exists. Use overwrite=True to overwrite."
            )

        with open(cosmo_file, "w", encoding="utf-8") as cf:
            cf.write(cosmo_output)

    def add_properties_to_sdf(
        self,
        sdf_file: Path,
        property_dict: dict[str, Property],
    ):
        """add calculated properties to an sdf file. Atom-based properties are set on
        the atoms and turned into list properties using rdkit. This ensures they are
        automatically parsed as atom properties when rdkit reads the sdf file.

        Args:
            sdf_file (Path):
                The sdf file to work on
            property_dict (dict[str, Property]):
                dictionary mapping names of properties to their values. The values are Property
                objects which know how to set themselves on the mol.
        """

        if not self.dir in sdf_file.parents:
            raise ValueError("SDF file must be in compound directory")

        lock_file = sdf_file.with_suffix(".sdf.lock")
        lock = FileLock(lock_file)
        with lock:

            mol = self.extract_mol(sdf_file)
            for property_key, property_val in property_dict.items():
                if mol.HasProp(property_key):
                    old_val = mol.GetProp(property_key)
                    _logger.warning(f"Property {property_key} was {old_val}. Will be overwritten")
                property_val.set_property_on_mol(mol, property_key)

            format_utils.write_sdf(mol, sdf_file, use_v2000=False, conf_ids=get_conf_ids(mol))

    def add_representative_structure(
        self,
        state_name: str,
        mol: Chem.Mol,
        property_dict: dict[str, Property],
        output_suffix: str = "_representative",
    ) -> Path:
        """Write a single representative structure SDF for a state.

        The *mol* provides the representative geometry (e.g. the lowest-energy
        conformer); the values in *property_dict* are stamped onto it, so
        atom-based properties become per-atom SDF properties that RDKit parses
        back onto the atoms. The file is written to the compound directory root
        as ``{state_name}{output_suffix}.sdf`` and is not registered
        as a result conformer.

        Args:
            state_name (str): The registered state the representative belongs to.
            mol (Chem.Mol): The molecule providing the representative geometry.
            property_dict (dict[str, Property]): Properties to stamp onto the mol.
            output_suffix (str): Suffix appended to the state name for the file
                and the molecule ``_Name``.

        Returns:
            Path: The path to the written representative SDF file.
        """
        write_mol = Chem.Mol(mol)
        representative_name = f"{state_name}{output_suffix}"
        write_mol.SetProp("_Name", representative_name)
        for property_key, property_val in property_dict.items():
            property_val.set_property_on_mol(write_mol, property_key)

        sdf_file = self.dir / f"{representative_name}.sdf"
        format_utils.write_sdf(
            write_mol, sdf_file, use_v2000=False, conf_ids=get_conf_ids(write_mol)
        )
        return sdf_file

    def add_properties_to_conformer_csv(
        self,
        sdf_file: Path,
        property_dict: dict[str, Property],
    ):
        """add calculated properties to a csv file.

        Args:
            sdf_file (Path):
                The sdf file corresponding to the conformer.
            property_dict (dict[str, Property]):
                dictionary mapping names of properties to their values. The values are Property
                objects which know how to set themselves on the mol.
        """
        lock_file = self.conformer_csv.with_suffix(".csv.lock")
        lock = FileLock(lock_file)
        conformer_name = sdf_file.stem
        with lock:
            conf_df = read_csv(self.conformer_csv)
            for property_key, property_val in property_dict.items():
                if not isinstance(property_val, ScalarProperty):
                    continue
                conf_df = add_property_to_dataframe(
                    conf_df, conformer_name, property_key, property_val.get_property_value()
                )
            write_csv(conf_df, self.conformer_csv)

    def add_properties_to_molecule_csv(
        self,
        state_name: str,
        property_dict: dict[str, Property],
    ):
        """add calculated properties to a csv file.

        Args:
            state_name (str):
                The name of the state corresponding to the molecule.
            property_dict (dict[str, Property]):
                dictionary mapping names of properties to their values. The values are Property
                objects which know how to set themselves on the mol.
        """
        lock_file = self.molecule_csv.with_suffix(".csv.lock")
        lock = FileLock(lock_file)
        with lock:
            mol_df = read_csv(self.molecule_csv)
            for property_key, property_val in property_dict.items():
                mol_df = add_property_to_dataframe(
                    mol_df,
                    state_name,
                    property_key,
                    property_val.get_property_value(),
                    identifier_col=STATE_COLUMN,
                )
            write_csv(mol_df, self.molecule_csv)

    # ----- pKa CSV handling methods below ----

    def add_pka_info(self, pka_info: PkaInfo):
        """Add or update PkaInfo in the pka.csv file.

        If an entry with matching pka_type, method, parent_states, and child_states
        already exists, its pka_value will be updated. Otherwise, a new entry is added.

        Args:
            pka_info (PkaInfo):
                The pKa information to store.
        """
        lock_file = self.pka_csv.with_suffix(".csv.lock")
        lock = FileLock(lock_file)
        with lock:
            pka_df = self._read_pka_csv()
            # Convert PkaInfo to a dictionary row
            row_dict = self._pka_info_to_row(pka_info)

            # Check if an entry with matching pka_type, method, parent_states, and child_states exists
            matching_row_idx = self._find_matching_pka_row(
                pka_df,
                pka_info.pka_type,
                pka_info.method,
                pka_info.parent_states,
                pka_info.child_states,
            )

            if matching_row_idx is not None:
                # Update existing row
                for key, value in row_dict.items():
                    pka_df.loc[matching_row_idx, key] = value
            else:
                # Add new row
                pka_df = pd.concat([pka_df, pd.DataFrame([row_dict])], ignore_index=True)

            write_csv(pka_df, self.pka_csv)

    def get_all_pka_info(self) -> list[PkaInfo]:
        """Retrieve all PkaInfo entries from the pka.csv file.

        Returns:
            list[PkaInfo]: A list of all PkaInfo objects.
        """
        pka_df = self._read_pka_csv()
        if pka_df.empty:
            return []

        pka_list = []
        for _, row in pka_df.iterrows():
            pka_info = self._row_to_pka_info(row)
            pka_list.append(pka_info)

        return pka_list

    def _read_pka_csv(self) -> pd.DataFrame:
        """Read the pka.csv file into a DataFrame.

        Returns:
            pd.DataFrame: The pKa data, or an empty DataFrame if the file does not exist.
        """
        return read_csv(self.pka_csv)

    def _pka_info_to_row(self, pka_info: PkaInfo) -> dict:
        """Convert a PkaInfo object to a dictionary suitable for a DataFrame row.

        Args:
            pka_info (PkaInfo):
                The pKa information.

        Returns:
            dict: A dictionary with the pKa data.
        """
        return {
            PKA_TYPE_COLUMN: pka_info.pka_type,
            PKA_VALUE_COLUMN: pka_info.pka_value,
            PKA_METHOD_COLUMN: pka_info.method,
            PKA_PARENT_STATES_COLUMN: PKA_STATES_SEPARATOR.join(pka_info.parent_states),
            PKA_CHILD_STATES_COLUMN: PKA_STATES_SEPARATOR.join(pka_info.child_states),
            PKA_ATOM_INDEX_COLUMN: (
                float(pka_info.atom_index) if pka_info.atom_index is not None else float("nan")
            ),
        }

    def _row_to_pka_info(self, row: pd.Series) -> PkaInfo:
        """Convert a DataFrame row to a PkaInfo object.

        Args:
            row (pd.Series):
                A row from the pka.csv DataFrame.

        Returns:
            PkaInfo: The reconstructed pKa information.
        """
        # Split parent and child states, handling empty strings and NaN values
        parent_states_value = row[PKA_PARENT_STATES_COLUMN]
        child_states_value = row[PKA_CHILD_STATES_COLUMN]

        # Check for NaN/None and convert to empty string
        parent_states_str = (
            str(parent_states_value).strip() if not pd.isna(parent_states_value) else ""
        )
        child_states_str = (
            str(child_states_value).strip() if not pd.isna(child_states_value) else ""
        )

        parent_states = (
            [s.strip() for s in parent_states_str.split(PKA_STATES_SEPARATOR)]
            if parent_states_str
            else []
        )
        child_states = (
            [s.strip() for s in child_states_str.split(PKA_STATES_SEPARATOR)]
            if child_states_str
            else []
        )

        return PkaInfo(
            pka_type=str(row[PKA_TYPE_COLUMN]),
            pka_value=float(row[PKA_VALUE_COLUMN]),
            method=str(row[PKA_METHOD_COLUMN]),
            parent_states=parent_states,
            child_states=child_states,
            atom_index=_parse_atom_index(row),
        )

    def _find_matching_pka_row(
        self,
        pka_df: pd.DataFrame,
        pka_type: str,
        method: str,
        parent_states: list[str],
        child_states: list[str],
    ) -> int | None:
        """Find a DataFrame row index that matches the given pKa parameters.

        Args:
            pka_df (pd.DataFrame):
                The pKa DataFrame.
            pka_type (str):
                The pKa type to match.
            method (str):
                The method to match.
            parent_states (list[str]):
                The parent states to match.
            child_states (list[str]):
                The child states to match.

        Returns:
            int | None: The index of the matching row, or None if not found.
        """
        if pka_df.empty:
            return None

        parent_states_str = PKA_STATES_SEPARATOR.join(parent_states)
        child_states_str = PKA_STATES_SEPARATOR.join(child_states)

        for idx, row in pka_df.iterrows():
            if (
                str(row[PKA_TYPE_COLUMN]) == pka_type
                and str(row[PKA_METHOD_COLUMN]) == method
                and str(row[PKA_PARENT_STATES_COLUMN]).strip() == parent_states_str
                and str(row[PKA_CHILD_STATES_COLUMN]).strip() == child_states_str
            ):
                return int(idx)

        return None

    def append_solubility_results(self, df: pd.DataFrame) -> None:
        """Append solubility screening results to the compound's solubility screening CSV.

        If the file already exists its contents are preserved and the new rows are
        appended with an outer join so differing column sets are handled gracefully.

        Args:
            df: DataFrame to append. Bookkeeping and parsed solvent columns are
                expected to have been added by the caller before this method is invoked.
        """
        lock = FileLock(self.solubility_screening_csv.with_suffix(".csv.lock"))
        with lock:
            if self.solubility_screening_csv.exists():
                existing = pd.read_csv(self.solubility_screening_csv)
                combined = pd.concat([existing, df], ignore_index=True, sort=False, join="outer")
            else:
                combined = df
            combined.to_csv(self.solubility_screening_csv, index=False)

    def append_cocrystal_results(self, df: pd.DataFrame) -> None:
        """Append cocrystal screening results to the compound's cocrystal screening CSV.

        If the file already exists its contents are preserved and the new rows are
        appended with an outer join so differing column sets are handled gracefully.

        Args:
            df: DataFrame to append. Bookkeeping columns (e.g. state names) are
                expected to have been added by the caller before this method is invoked.
        """
        lock = FileLock(self.cocrystal_screening_csv.with_suffix(".csv.lock"))
        with lock:
            if self.cocrystal_screening_csv.exists():
                existing = pd.read_csv(self.cocrystal_screening_csv)
                combined = pd.concat([existing, df], ignore_index=True, sort=False, join="outer")
            else:
                combined = df
            combined.to_csv(self.cocrystal_screening_csv, index=False)


def create_cpd_dir(directory: Path, create_new: bool = False) -> CpdDir:
    """Factory function that creates a :class:`CpdDir` with its
    dependency-injected handlers included.

    Args:
        directory (Path): Root directory for the compound.
        create_new (bool):
            If ``True`` (default), create the directory structure and
            initialise the JSON files when they do not exist yet.

    Returns:
        CpdDir: A fully initialised compound directory instance.
    """
    directory = Path(directory)
    registry_file = directory / REGISTRY_FILE

    registry_handler = RegistryHandler(directory, registry_file)

    cpd_dir = CpdDir(directory, registry_handler)

    if create_new:
        cpd_dir.initialize()

    return cpd_dir


def is_cpd_dir(directory: Path) -> bool:
    """Checks if a given directory is a compound directory by looking for the presence of the registry.json file.

    Args:
        directory (Path): The directory to check.

    Returns:
        bool: True if the directory is a compound directory, False otherwise.
    """
    registry_file = directory / REGISTRY_FILE
    return registry_file.is_file()


def find_cpd_dirs(root_dir: Path) -> list[CpdDir]:
    """Searches for compound directories under *root_dir* and returns a list of
    :class:`CpdDir` instances for them.

    Args:
        root_dir (Path): The directory to search under.

    Returns:
        list[CpdDir]: A list of :class:`CpdDir` instances for the found compound directories.
    """
    cpd_dirs = []
    for subdir in root_dir.iterdir():
        if subdir.is_dir():
            if is_cpd_dir(subdir):
                cpd_dirs.append(create_cpd_dir(subdir))
    return cpd_dirs


def add_property_to_dataframe(
    property_df: pd.DataFrame,
    identifier: str,
    property_name: str,
    property_value: float | str,
    identifier_col: str = CONFORMER_COLUMN,
) -> pd.DataFrame:
    """Add a property value to a pandas DataFrame, creating a new row if the identifier is not already present.

    Args:
        property_df (pd.DataFrame): The DataFrame to update.
        identifier (str): The identifier for the row (e.g. conformer name or state name).
        property_name (str): The name of the property to add.
        property_value (float | str): The value of the property to add.
        identifier_col (str): The name of the column used for identifiers. Defaults to CONFORMER_COLUMN.
    """
    if (
        identifier_col not in property_df.columns
        or identifier not in property_df[identifier_col].values
    ):
        new_row = {identifier_col: identifier, property_name: property_value}
        return pd.concat([property_df, pd.DataFrame([new_row])], ignore_index=True)

    r_property_df = property_df.copy()
    r_property_df.loc[r_property_df[identifier_col] == identifier, property_name] = property_value
    return r_property_df


def read_csv(csv_path: Path) -> pd.DataFrame:
    """Read a CSV file into a pandas DataFrame. If the file does not exist or is empty, return an empty DataFrame.

    Args:
        csv_path (Path): The path to the CSV file.
    Returns:
        pd.DataFrame: The contents of the CSV file as a DataFrame, or an empty DataFrame if the file does not exist or is empty.
    """
    if not csv_path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_csv(df: pd.DataFrame, csv_path: Path):
    """Write a pandas DataFrame to a CSV file.

    Args:
        csv_path (Path): The path to the CSV file.
        df (pd.DataFrame): The DataFrame to write.
    """
    df.to_csv(csv_path, index=False)


def get_conf_ids(mol: Chem.Mol) -> list[int]:

    conf_ids = [conformer.GetId() for conformer in mol.GetConformers()]
    if len(conf_ids) == 0:
        return [-1]
    return conf_ids
