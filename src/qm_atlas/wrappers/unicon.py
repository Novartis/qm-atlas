# Usage: unicon -i example.sdf
#  Available options:
#
#  General options:
#   -h [ --help ]                    Prints help message
#   -v [ --verbosity ] arg           Set verbosity level
#                                    (0 = Quiet, 1 = Errors, 2 = Warnings, 3 =
#                                    Info)
#   --jobs arg (=1)                  Number of threads to use (0 = auto, 1 =
#                                    single [default]), e.g.: -j 4
#
#  Input options:
#   -i [ --input ] arg               Input file(s) suffix is required or
#                                    directories. Several input files or
#                                    directories can be given separated by
#                                    spaces, (e.g. -i a.mol2 b.sdf dataDir)
#   --inFormat arg                   Input file format (e.g. MOL(*.mol)
#                                    MOL2(*.mol2) PDB(*.pdb) SDF(*.sdf)
#                                    SMILES(*.smi *.smiles) INCHI(*.inchi)
#   --from arg                       The start entry of the input to be used for
#                                    processing
#   --to arg                         The last entry of the input to be used for
#                                    processing
#   --multi                          Convert all components into multiple entries
#                                    (Largest Component is converted by default).
#
#  Output options:
#   -o [ --output ] arg              Output file, suffix is required.
#   --outFormat arg                  Output file format (e.g. MOL2(*.mol2)
#                                    SDF(*.sdf) SMILES(*.smi *.smiles))
#   -s [ --split ] arg               Number of input entries the split files will
#                                    contain
#   -k [ --key ]                     In case of InChI output use 'key' option to
#                                    write InChI keys
#
#  Configuration:
#   -t [ --tautomer ] arg            Generate tautomers:
#                                    topscoring = enumerate only best tautomers
#                                    ensemble = enumerate all tautomers
#                                    single = generate only one normalized tautomer
#   -p [ --protonation ] arg         Generate protonation states:
#                                    topscoring = enumerate only best protonation states
#                                    ensemble = enumerate all protonation states
#                                    single = generate only one normalized protonation state
#   -m [ --maxNofConfs ] arg (=250)  Set the maximum number of generated conformations
#   -c [ --conformer ] arg           Generate conformers with quality level
#                                    (1 = Fast, 2 = Best)
#   -g [ --generate ] arg (=1)       Generate coordinates for output
#                                    (1 = keep original, 2 = 2D-Coordinates, 3 = 3D-Coordinates)
#   --hydrogens                      Consider hydrogen clashes during conformation and coordinate
#                                    generation
#   --extract arg                    Extract ligand with given name
#
#  License:
#   --license arg                    To reactivate the executable, please provide
#                                    a new license key.
#

import logging
from pathlib import Path

from ppqm import chembridge
from ppqm.utils import WorkDir
from rdkit.Chem import Mol

from qm_atlas import constants, utils
from qm_atlas.software_environment import SOFTWARE_CONFIG
from qm_atlas.wrappers.common import run_command

_logger = logging.getLogger(__name__)


def _get_tautomers_from_smiles(
    smiles,
    scr=constants.DEFAULT_SCR,
    return_canonical=True,
    cmd=None,
    keep_files=False,
):
    """Internal function to generate tautomers from SMILES string."""

    # Get command from environment manager
    software_env_manager = SOFTWARE_CONFIG.get_environment_manager()
    if cmd is None:
        cmd = software_env_manager.get_command("unicon", "unicon")

    temp = WorkDir(dir=scr, prefix="unicon_", keep=keep_files)
    work_dir = temp.get_path()

    ext = "smi"
    filename = "_tmp_unicon_.smi"

    filename_in = work_dir / filename
    filename_out = work_dir / (filename + ".o")

    with utils.open_utf8(filename_in, "w") as f:
        f.write(smiles)

    # empty file
    with utils.open_utf8(filename_out, "w") as f:
        pass

    options = [
        cmd,
        f"-i {filename}",
        f"-o {filename}.o",
        "-t=ensemble",
        "--inFormat smiles",
        f"--outFormat {ext}",
    ]

    cmd = " ".join(options)

    cmd = f"cd {work_dir}; " + cmd

    _logger.debug(cmd)

    env = software_env_manager.get_run_environment("unicon")
    stdout, stderr = run_command(cmd, env=env)
    stdout = "" if stdout is None else stdout
    stderr = "" if stderr is None else stderr
    lines = stdout + "\n" + stderr
    lines = lines.split("\n")

    with utils.open_utf8(filename_out) as f:
        tautomers = f.readlines()
        tautomers = [x.strip() for x in tautomers]

    if len(tautomers) == 0:

        for line in lines:
            line = line.lower()
            if "error" in line:
                line = line.replace("error", "").replace(":", "")
                _logger.error(line)

        raise RuntimeError("UNICON tautomer generation failed")

    # Clean duplicates
    tautomers = list(set(tautomers))

    if return_canonical:
        tautomers = [chembridge.get_canonical_smiles(smi) for smi in tautomers]

    return tautomers


def get_tautomers(
    mol: Mol,
    scr: Path = constants.DEFAULT_SCR,
    keep_files: bool = False,
    return_canonical: bool = True,
    cmd: str | None = None,
) -> list[Mol]:
    """Generate tautomers for a molecule using UNICON. See references [1], [2]
    for details on the UNICON tautomer generation algorithm.

    Args:
        mol (Mol):
            The rdkit Mol to generate tautomers for.
        scr (Path):
            The scratch directory to use for temporary files. Defaults to
            constants.DEFAULT_SCR.
        keep_files (bool):
            Whether to keep the temporary files. Defaults to False. Useful for debugging.
        return_canonical (bool):
            Whether to return canonical SMILES for the generated tautomers.
            Defaults to True.
        cmd (str | None):
            The command to use for unicon. If None, the command is resolved from the
            software configuration.

    Raises:
        RuntimeError: If the UNICON tautomer generation fails.

    Returns:
        list[Mol]: A list of rdkit Mol objects representing the generated tautomers.

    References:
        [1] Sommer, K., Friedrich, N.-O., Bietz, S., Hilbig, M., Inhester, T. & Rarey, M.
        UNICON: A Powerful and Easy-to-Use Compound Library Converter.
        *J. Chem. Inf. Model.* **56**, 1105-1111 (2016).
        https://doi.org/10.1021/acs.jcim.6b00069

        [2] Urbaczek, S., Kolodzik, A. & Rarey, M. The Valence State Combination
        Model: A Generic Framework for Handling Tautomers and Protonation States.
        *J. Chem. Inf. Model.* **54**, 756-766 (2014).
        https://doi.org/10.1021/ci400724v
    """

    smiles = chembridge.molobj_to_smiles(mol)
    smiles_list = _get_tautomers_from_smiles(
        smiles,
        scr=scr,
        keep_files=keep_files,
        return_canonical=return_canonical,
        cmd=cmd,
    )

    rtn_list = []

    for tausmi in smiles_list:
        taut_molobj = chembridge.smiles_to_molobj(tausmi)
        rtn_list.append(taut_molobj)

    return rtn_list
