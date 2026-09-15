# usage: ./blabber_sd.bin [options] <filename>
#
#  <filename> can be a SD (.sd or .sdf) or MOL2 (.mol2) file
#
# options are:
#  -h, --help                     display this help
#      --version                  show version info
#      --input-type=<sd|mol2>     input file type (autodetect)
#  -p <pH list>, --pH=<pH list>   report most abundant species
#                                 at given pH (7.4)
#  -n, --neutralize               report only the neutral species
#      --load-model=<database>    use a custom model database for predictions
#  -o, --output=FILE              output file name (SD only)
#      --explicit-h=<mode>        select hydrogens to keep on output (pristine)
#                                 <mode> can be one of:
#                                   * none: no hydrogens
#                                   * pristine: only existing hydrogens
#                                   * all: both existing and added hydrogens
#  -m, --minimize                 perform full 3D minimization
#  -t <abundance>                 report all species above the threshold
#  -e, --equivalent-protomers     generate all equivalent protomers
#
# <pH list> is a comma separated list of the following:
#     * single value
#     * range ( min-max )
# Examples:
#     * single value:    --pH=6
#     * single range:    --pH=6-10
#     * multiple values: --pH=6,7,8
#     * mixed:           --pH=6,7-10

import logging
import pathlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from ppqm import chembridge
from ppqm.utils.files import WorkDir
from rdkit.Chem import Mol

from qm_atlas import constants, utils
from qm_atlas.software_environment import SOFTWARE_CONFIG, USE_DEFAULT_MODEL, get_moka_model
from qm_atlas.wrappers.common import run_command

_logger = logging.getLogger(__name__)

BLABBER_CMD = "blabber_sd"

# pH
BLABBER_PH = 7.4

# abundance threshold
BLABBER_THRESHOLD = None

BLABBER_FILENAME = "_tmp_blabber.sdf"
COLUMN_ABUNDANCE = "ABUNDANCE"


def get_protonation(
    molobj: Mol,
    scr: Path = constants.DEFAULT_SCR,
    filename: str | Path = BLABBER_FILENAME,
    remove_zero_charged: bool = False,
    keep_files: bool = False,
    **kwargs,
) -> list[Mol]:
    """Enumerate protonation states of *molobj* with MoKa Blabber and return them as molecules."""

    filename = scr / filename
    sdfstr = chembridge.molobj_to_sdfstr(molobj)

    if sdfstr is None:
        raise ValueError("Unable to generate SDF")

    sdfs = get_protonation_from_sdf(sdfstr, scr=scr, keep_files=keep_files, **kwargs)
    molobjs = [chembridge.sdfstr_to_molobj(sdf) for sdf in sdfs]

    def _iterate_molobjs():
        def _abs_charge(mol):
            charges = chembridge.get_atom_charges(mol)
            charges = np.array(charges)
            charges = np.abs(charges)
            return charges.sum()

        for molobj in molobjs:

            if molobj is None:
                continue

            value = (
                molobj.GetProp(COLUMN_ABUNDANCE)
                if molobj.HasProp(COLUMN_ABUNDANCE)
                else float("nan")
            )

            if remove_zero_charged and _abs_charge(molobj) == 0:
                continue

            if isinstance(value, str) and "No species found" in value:
                continue

            if isinstance(value, str):
                value = value.split("%")
                value = value[0]
                value = utils.as_float(value)

            molobj.SetProp(COLUMN_ABUNDANCE, str(value))

            yield molobj

    # Filter the protonated molobjs
    molobjs = [mol for mol in _iterate_molobjs()]

    return molobjs


def get_protonation_from_sdf(
    sdfstr: str,
    cmd: str = BLABBER_CMD,
    scr: Path = constants.DEFAULT_SCR,
    threshold: int | None = BLABBER_THRESHOLD,
    model: Path | None = USE_DEFAULT_MODEL,
    ph: float = BLABBER_PH,
    use_tempfile: bool = True,
    keep_files: bool = False,
) -> list[str]:
    """Run MoKa Blabber on an SDF string and return the protonated species as SDF strings."""

    # Resolve model if sentinel value is used
    if model is USE_DEFAULT_MODEL:
        model = get_moka_model()

    if use_tempfile:
        temp = WorkDir(dir=scr, prefix="blabber_", keep=keep_files)
        scr = pathlib.Path(temp.name)

    # NOTE Filenames needs sdf suffix for blabber to read
    filename = "_tmp_blabber.sdf"
    filename_out = "_tmp_blabber.o.sdf"

    # Create inputfile
    with open(scr / filename, "w") as f:
        f.write(sdfstr)

    # Ensure empty output file
    with open(scr / filename_out, "w") as f:
        pass

    args = [f"{cmd}"]

    if model is not None:
        args += [f"--load-model={model}"]

    if ph is not None:
        args += [f"-p {ph}"]

    if threshold is not None:
        args += [f"-t {threshold}"]

    args += [f"-o {filename_out}"]
    args += [f"{filename}"]

    cmd = " ".join(args)

    _logger.debug(f"{scr} {cmd}")

    software_env_manager = SOFTWARE_CONFIG.get_environment_manager()
    env = software_env_manager.get_run_environment("moka")
    stdout, stderr = run_command(cmd, env=env, cwd=scr)

    with open(scr / filename_out, "r") as f:
        lines = f.readlines()

    sdfs = generate_sdfs(lines)
    sdfs = list(sdfs)

    if len(sdfs) == 0:
        stderr = stderr.split("\n")
        for line in stderr:
            line = line.strip()
            if line == "":
                continue
            _logger.error(line)

    return sdfs


def generate_sdfs(lines: list[str]) -> Iterator[str]:
    """Generate SDF strings from lines of strings"""

    # Stream the result in sdfs
    sdf = []

    for line in lines:

        sdf.append(line)

        if constants.SDFSEP in line:
            sdf = "".join(sdf)
            yield sdf
            sdf = []

    # Collect the last one, if no seperator is found
    if len(sdf) > 1:
        sdf = "".join(sdf)
        yield sdf

    return
