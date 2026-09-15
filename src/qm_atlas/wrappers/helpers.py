import gzip
import logging
import re
import time
import traceback
from collections.abc import Iterable
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from qm_atlas.software_environment import SOFTWARE_CONFIG
from qm_atlas.wrappers.common import run_command

_logger = logging.getLogger(__name__)


def check_licenses(
    software: str,
    feature_name: str,
    lic_cmd: str | None = None,
) -> tuple[int, int]:
    """
    Checks how many licenses are available

    Args:
        software (str): the software whose license server to query
        feature_name (str): the feature (module) name to be searched in the output
        lic_cmd (str | None): the command to query the license server. Defaults to None.

    Returns:
        tuple[int, int]: the number of licenses issued and the number in use.
    """

    software_env_manager = SOFTWARE_CONFIG.get_environment_manager()
    env = software_env_manager.get_run_environment(software)
    if lic_cmd is None:

        # if lictool has not been provided, assume no license limitation
        if not software_env_manager.command_available(software, "lictool"):
            return 1000, 0

        lic_cmd = software_env_manager.get_command(software, "lictool")

    stdout, _ = run_command(lic_cmd, env=env)

    if stdout is None:
        raise RuntimeError(f"Invalid Response from license command: {lic_cmd}")

    match_str = (
        rf"Users\s+of\s+{feature_name}:\s*\(Total\s+of\s+(\d+)\s+licenses\s+issued;"
        r"\s*Total\s+of\s+(\d+)\s+licenses\s+in\s+use.*$"
    )
    regex = re.compile(match_str)
    lines = stdout.split("\n")
    for line in lines:
        match = regex.match(line)
        if match:
            licenses_issued, licenses_used = int(match.group(1)), int(match.group(2))
            return licenses_issued, licenses_used

    raise RuntimeError(f"Invalid Response from license command: {lic_cmd}")


def wait_for_license(
    software: str,
    feature_name: str,
    lic_cmd: str | None = None,
    buffer: int = 1,
    timeout: int = 60,
):
    """waits until a license is available

    Args:
        software (str):
            The software whose license server to query.
        buffer (int, optional):
            The number of licenses to leave as a buffer. Defaults to 1.
        timeout (int, optional):
            The number of minutes after which to timeout. Defaults to 60.
        lic_cmd (str):
            The command to query the license server
        feature_name (str):
            The feature (module) name to be searched in the output

    Raises:
        TimeoutError: If a license did not become available in the allocated time.

    returns None
    """
    counter = 0
    while True:
        num_issued, num_used = check_licenses(software, feature_name, lic_cmd=lic_cmd)
        num_available_lic = num_issued - num_used
        if num_available_lic > buffer:
            break
        else:
            time.sleep(30)
            counter += 1
        if counter > 2 * timeout:
            raise TimeoutError

    _logger.debug(f"{num_available_lic} available for {feature_name}. Starting")


def read_conformers_as_mols(file: Path) -> list[Chem.Mol] | None:
    """Reads a maegz or sdf file containing conformers of the same graph and
    returns them as a list of separate rdkit molecules (each with a single
    conformer).

    Args:
        file (Path): The file in .sdf, .sdf.gz, .mae or .maegz format

    Returns:
        list[Chem.Mol]: One rdkit molecule per conformer record in the file.
    """

    try:
        if file.suffix == ".maegz":
            with gzip.open(file) as out_unzipped:
                with Chem.MaeMolSupplier(out_unzipped, removeHs=False) as supplier:
                    confs_as_mols = [mol for mol in supplier]

        elif file.suffix == ".mae":
            with Chem.MaeMolSupplier(str(file), removeHs=False) as supplier:
                confs_as_mols = [mol for mol in supplier]

        elif file.suffix == ".sdf.gz":
            with gzip.open(file) as out_unzipped:
                with Chem.SDMolSupplier(out_unzipped, removeHs=False) as supplier:
                    confs_as_mols = [mol for mol in supplier]

        elif file.suffix == ".sdf":
            with Chem.SDMolSupplier(str(file), removeHs=False) as supplier:
                confs_as_mols = [mol for mol in supplier]
        else:
            _logger.error(
                f"Unknown file format. Expected .maegz, .mae, .sdf.gz or .sdf, got {file.suffix}."
                "Exiting"
            )
            return None

        return confs_as_mols

    except OSError:
        _logger.error(f"{file} empty. Exiting conformer generation")
        return None

    except Exception as exc:
        _logger.error(f"{file} could not be read. Exiting conformer generation")
        _logger.error(f"Got Exception {exc}")
        for line in traceback.format_exc().split("\n"):
            _logger.error(line)
        return None


def store_atom_map_numbers(mol: Chem.Mol) -> Chem.Mol:
    """Store the atom map numbers in the molecule. These can be used to track
        changes of atom indices after conformer generation.

    Args:
        mol (Chem.Mol): The input molecule.

    Returns:
        Chem.Mol: A copy of the molecule with atom map numbers stored.
    """

    mol_with_map = Chem.Mol(mol)
    for atom in mol_with_map.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    return mol_with_map


def map_conformers_onto_template(
    template_mol: Chem.Mol,
    generated_mols: Iterable[Chem.Mol],
) -> Chem.Mol:
    """Transfer the 3D coordinates of generated conformers back onto the original
    molecular graph.

    Conformer generators may re-perceive the molecular graph (e.g. assign or
    remove a stereo center, reorder atoms). To guarantee that the graph is never
    changed, this function keeps the ``template_mol`` graph untouched and only
    copies the generated 3D coordinates onto it as new conformers.

    Atoms in the generated molecules are matched to the template via atom map
    numbers, which must have been stored on the molecule passed to the conformer
    generator using :func:`store_atom_map_numbers`. If a generated atom carries
    no atom map number, its position is matched by atom index instead.

    Args:
        template_mol (Chem.Mol):
            The molecule whose graph should be preserved. Its own conformers are
            ignored.
        generated_mols (Iterable[Chem.Mol]):
            The molecules produced by the conformer generator, each carrying a
            single conformer.

    Returns:
        Chem.Mol: A copy of ``template_mol`` carrying the generated conformers.
    """

    return_mol = Chem.Mol(template_mol)
    return_mol.RemoveAllConformers()

    dummy_mol = Chem.Mol(return_mol)
    AllChem.Compute2DCoords(dummy_mol)

    for mol in generated_mols:
        if mol is None:
            continue
        generated_conf = mol.GetConformer()
        _mol = Chem.Mol(dummy_mol)
        new_conf = _mol.GetConformer()
        for atom in mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            old_idx = atom.GetIdx()
            template_idx = map_num - 1 if map_num > 0 else old_idx
            new_conf.SetAtomPosition(template_idx, generated_conf.GetAtomPosition(old_idx))
        return_mol.AddConformer(new_conf, assignId=True)

    return return_mol
