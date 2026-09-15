import logging
import traceback
from pathlib import Path

from ppqm.utils.files import WorkDir
from rdkit import Chem
from rdkit.Chem import AllChem

from qm_atlas import utils
from qm_atlas.constants import DEFAULT_SCR
from qm_atlas.software_environment import SOFTWARE_CONFIG
from qm_atlas.utils import conformer_is_3d
from qm_atlas.wrappers.common import run_command
from qm_atlas.wrappers.helpers import (
    map_conformers_onto_template,
    read_conformers_as_mols,
    store_atom_map_numbers,
    wait_for_license,
)
from qm_atlas.wrappers.macromodel import utils as mm_utils

_logger = logging.getLogger(__name__)


SOFTWARE_NAME = "macromodel"
FEATURE_NAME = "PSP_PLOP"
NUM_LICENSES_NEEDED = 8  # TODO: check does this depend on n_cores?
COMMAND_NAME = "macrocycle"


def generate_conformers(
    mol: Chem.Mol,
    license_buffer: int = 1,
    license_timeout: int = 60,
    scr: Path = DEFAULT_SCR,
    keep_files: bool = False,
    max_conformers: int = 1000,
    energy_cutoff: float = 30.0,
    rms_threshold: float = 0.3,
    num_cores: int = 1,
    **conf_gen_kwargs,
) -> Chem.Mol:
    """Generate Conformers using Schrodinger macrocycle script.

    Args:
        mol (Chem.Mol):
            The molecule to work on.
        license_buffer (int, optional):
            The number of licenses to leave available. Defaults to 1.
        license_timeout (int, optional):
            The number of minutes to spend waiting for a license. Defaults to 60.
        scr (Path, optional):
            The scratch directory to use. Defaults to DEFAULT_SCR.
        keep_files (bool, optional):
            Whether to keep the intermediate files. Defaults to False.
        max_conformers (int, optional):
            The maximum number of conformers to generate. Defaults to 1000.
        energy_cutoff (float, optional):
            The energy cutoff in kcal/mol. Defaults to 30.0.
        rms_threshold (float, optional):
            The RMS threshold in Angstroms. Defaults to 0.3.
        num_cores (int, optional):
            The number of cores to use. Defaults to 1.
        **conf_gen_kwargs:
            Additional keyword arguments to pass to the macrocycle script.
            Keys and Values will be converted to command line arguments.

    Raises:
        RuntimeError: If the conformer generation fails or no output file is produced.

    Returns:
        Chem.Mol: An rdkit mol with conformers.
    """

    if not utils.has_macrocycle(mol, ring_threshold=8):
        _logger.error("molecule is not a macrocycle, skipping macrocycle conformer generation")
        raise RuntimeError(
            "molecule is not a macrocycle, skipping macrocycle conformer generation"
        )

    temp = WorkDir(dir=scr, prefix="mm_macrocycle_", keep=keep_files)
    scr = temp.get_path()
    scr.mkdir(parents=True, exist_ok=True)

    _mol = Chem.AddHs(mol)
    if not conformer_is_3d(_mol):
        AllChem.EmbedMolecule(_mol, randomSeed=42)  # deterministic result

    # Atom map numbers are stored so the original graph can be restored after
    # conformer generation.
    input_mol = store_atom_map_numbers(_mol)
    input_sdf = scr / "input.sdf"
    with Chem.SDWriter(str(input_sdf)) as writer:
        writer.write(input_mol)

    # generate 3d Input with LigPrep
    input_mae = mm_utils.convert_to_mae(input_sdf)
    jobname = "macrocycle"

    software_env_manager = SOFTWARE_CONFIG.get_environment_manager()
    mc_cmd = software_env_manager.get_command(SOFTWARE_NAME, COMMAND_NAME)
    if mc_cmd is None:
        _logger.error("Could not find macrocycle command in software environment")
        raise RuntimeError("Macrocycle command not found")

    command_list = [
        mc_cmd,
        str(input_mae.resolve()),
        "-conf_only",
        "-target_nconf",
        str(max_conformers),
        "-energy_cutoff",
        str(energy_cutoff),
        "-redundant_cutoff",
        str(rms_threshold),
        "-NJOBS",
        str(num_cores),
        "-HOST",
        f"localhost:{num_cores}",
        "-WAIT",
        "-jobname",
        jobname,
    ]

    command_list.extend([f"{key} {value}" for key, value in conf_gen_kwargs.items()])
    cmd = " ".join(command_list)

    try:
        buffer = license_buffer + NUM_LICENSES_NEEDED - 1
        wait_for_license(SOFTWARE_NAME, FEATURE_NAME, buffer=buffer, timeout=license_timeout)
    except TimeoutError as exc:
        _logger.error("No Macromodel license available. Exiting conformer generation")
        raise RuntimeError(
            "No Macromodel license available. Exiting conformer generation"
        ) from exc
    except Exception as exc:
        _logger.error(f"got exception {exc} while waiting for license")
        _logger.error(traceback.format_exc())
        raise RuntimeError(
            "Macrocycle conformer generation failed while waiting for license"
        ) from exc

    env = software_env_manager.get_run_environment("macromodel", n_cores=num_cores)
    run_command(cmd, env=env, cwd=scr)
    outfile = scr / f"{jobname}-out.maegz"

    if not outfile.is_file():
        _logger.error("No .maegz output found. Exiting conformer generation")
        mm_utils.log_macromodel_errors(scr)
        raise RuntimeError("Macrocycle conformer generation failed: no output file produced")

    generated_mols = read_conformers_as_mols(outfile)
    if not generated_mols:
        _logger.error("Macromodel produced an empty output file. Exiting conformer generation")
        mm_utils.log_macromodel_errors(scr)
        raise RuntimeError("Macrocycle conformer generation failed: output file is empty")

    return map_conformers_onto_template(_mol, generated_mols)
