import logging
import traceback
from pathlib import Path

from jinja2 import Template
from ppqm.utils.files import WorkDir
from rdkit import Chem

from qm_atlas.constants import DEFAULT_SCR
from qm_atlas.software_environment import SOFTWARE_CONFIG
from qm_atlas.utils import open_utf8
from qm_atlas.wrappers import corina
from qm_atlas.wrappers.common import run_command
from qm_atlas.wrappers.helpers import (
    map_conformers_onto_template,
    read_conformers_as_mols,
    store_atom_map_numbers,
    wait_for_license,
)
from qm_atlas.wrappers.macromodel import utils as mm_utils

_logger = logging.getLogger(__name__)


# TODO: This is a bit messy, e.g. what happens if you insert integers with many digits?
# Does that screw up the formatting of the table. Could pandas help?

MM_CONF_SCRIPT = """mm_input.mae
mm_output.maegz
 MMOD       0      1      0      0     0.0000     0.0000     0.0000     0.0000
 DEBG      55      0      0      0     0.0000     0.0000     0.0000     0.0000
 FFLD      {{ force_field }}      1      0      {{ (force_field == 10) | int }}     1.0000     0.0000     0.0000     0.0000
 SOLV       3      {{ solvent_num }}      0      0     0.0000     0.0000     0.0000     0.0000
 EXNB       0      0      0      0     0.0000     0.0000     0.0000     0.0000
 BDCO       0      0      0      0    89.4427 99999.0000     0.0000     0.0000
 READ       0      0      0      0     0.0000     0.0000     0.0000     0.0000
 CRMS       0      0      0      0     0.0000     {{ '%0.4f' | format(rms_threshold) }}     0.0000     2.0000
 {{ mode }}   {{ steps }}    {{ max_conformers }}      0      0     0.0000     0.0000     {% if mode == 'LMCS' %}3.0000     6.0000{% else  %}0.0000     0.0000{% endif %}
 MCNV       1      5      0      0     0.0000     0.0000     0.0000     0.0000
 MCSS       2      0      0      0    {{ '%0.4f' | format(mm_threshold_e) }}     0.0000     0.0000     0.0000
 MCOP       1      0      0      0     {% if mode == 'LMCS' %}0.5000{% else %}0.0000{% endif %}     0.0000     0.0000     0.0000
 DEMX       0    833      0      0    {{ '%0.4f' | format(mm_threshold_e) }}    {{ '%0.4f' | format(mm_threshold_e * 2) }}     0.0000     0.0000
 MSYM       0      0      0      0     0.0000     0.0000     0.0000     0.0000
 AUOP       0      0      0      0   100.0000     0.0000     0.0000     0.0000
 AUTO       0      2      1      1     0.0000     1.0000     0.0000     3.0000
 CONV       2      0      0      0     0.0500     0.0000     0.0000     0.0000
 MINI       1      0   {{ steps_minimize }}      0     0.0000     0.0000     0.0000     0.0000
"""

MACROMODEL_DEFAULTS = {
    "mode": "MCMM",  # Macromodel conf search type. MCMM or LMCS
    "mm_threshold_e": 30,  # Macromodel E threshold for confgen
    "rms_threshold": 0.3,  # Macromodel RMSD threshold for confgen
    "max_conformers": 1000,  # The maximum number of conformers to generate (set 0 to ignore)
    "steps": 5000,  # Macromodel MonteCarlo or LMCS steps
    "force_field": 14,  # 14=OPLS2005 16=OPLS4  10=MMFF94s,
    "steps_minimize": 1000,  # Macromodel number of MM minimization steps
    "solvent_num": 1,  # 1 - water, 9 - octanol
}


# tested what licenses are used when running bmin on Nov 24 2022
SOFTWARE_NAME = "macromodel"
FEATURE_NAME = "MMOD_MACROMODEL"
NUM_LICENSES_NEEDED = 2


def setup_mm_conf(
    template_str: str = MM_CONF_SCRIPT,
    scr: Path = DEFAULT_SCR,
    keep_files: bool = False,
    **conf_gen_kwargs,
) -> tuple[WorkDir, Path]:
    """sets up a temporary directory to run the Macromodel confsearch in.

    Args:
        template_str (str, optional):
            A template for the Macromodel conformer search script (.com).
            Defaults to MM_CONF_SCRIPT.
        scr (Path, optional):
            The scratch space to use for the temp dir. Defaults to DEFAULT_SCR.
        keep_files (bool, optional):
            whether to keep the files. Needed for debugging. Defaults to False.
        **conf_gen_kwargs:
            Additional MacroModel conformer-search parameters substituted into the
            script template (e.g. mode, force_field, steps, rms_threshold).

    Returns:
        temp (WorkDir):
            The WorkDir instance
        mm_script (Path):
            The path to the mm script.
    """

    # set up temporary directory
    mm_kwargs = dict(MACROMODEL_DEFAULTS)
    mm_kwargs.update(conf_gen_kwargs)
    temp = WorkDir(dir=scr, prefix="mm_conf_", keep=keep_files)
    scr = temp.get_path()
    scr.mkdir(parents=True, exist_ok=True)

    # create conformer generation script
    conf_gen_template = Template(template_str)
    msg = conf_gen_template.render(**mm_kwargs)
    mm_script = scr / "mm_conf.com"
    with open_utf8(mm_script, "w") as outf:
        outf.write(msg)

    return temp, mm_script


def generate_conformers(
    mol: Chem.Mol,
    license_buffer: int = 1,
    license_timeout: int = 60,
    scr: Path = DEFAULT_SCR,
    keep_files: bool = False,
    **conf_gen_kwargs,
) -> Chem.Mol:
    """Generate Conformers using Macromodel.

    Args:
        mol (Chem.Mol):
            The molecule to work on.
        license_buffer (int, optional):
            The number of licenses to leave available. Defaults to 2.
        license_timeout (int, optional):
            The number of minutes to spend waiting for a license. Defaults to 60.
        scr (Path, optional):
            The scratch space to use. Defaults to DEFAULT_SCR.
        keep_files (bool, optional):
            Whether to keep the temporary files. Used for debugging. Defaults to False.
        **conf_gen_kwargs:
            Additional MacroModel conformer-search parameters forwarded to
            :func:`setup_mm_conf`.

    Raises:
        RuntimeError: If the conformer generation fails or no output file is produced.

    Returns:
        Chem.Mol: An rdkit mol with conformers.
    """

    # Set up temporary directory with run script
    temp, mm_script = setup_mm_conf(keep_files=keep_files, scr=scr, **conf_gen_kwargs)
    scr = temp.get_path()

    # create initial molecule and write to sdf file. Atom map numbers are stored
    # so the original graph can be restored after conformer generation.
    init_mol = corina.generate_conformers_molobj(Chem.AddHs(mol))
    input_mol = store_atom_map_numbers(init_mol)
    sdf_file = scr / "mm_input.sdf"
    with Chem.SDWriter(str(sdf_file)) as writer:
        writer.write(input_mol)

    # convert sdf file to mae file
    mm_utils.convert_to_mae(sdf_file)

    # wait until a license becomes available
    try:
        buffer = license_buffer + NUM_LICENSES_NEEDED - 1
        wait_for_license(SOFTWARE_NAME, FEATURE_NAME, buffer=buffer, timeout=license_timeout)
    except TimeoutError:
        _logger.error("No Macromodel license available. Exiting conformer generation")
        return
    except Exception as exc:
        _logger.error(f"got exception {exc} while waiting for license")
        _logger.error(traceback.format_exc())
        raise

    # run Schrodinger
    software_env_manager = SOFTWARE_CONFIG.get_environment_manager()
    bmin_cmd = software_env_manager.get_command("macromodel", "bmin")
    cmd = f"{bmin_cmd} -LOCAL -WAIT {mm_script.stem}"
    env = software_env_manager.get_run_environment("macromodel")
    run_command(cmd, env=env, cwd=scr)

    outfile = scr / "mm_output.maegz"

    if not outfile.is_file():
        _logger.error("No .maegz output found. Exiting conformer generation")
        mm_utils.log_macromodel_errors(scr)
        raise RuntimeError("Macromodel conformer generation failed: no output file produced")

    generated_mols = read_conformers_as_mols(outfile)
    if not generated_mols:
        _logger.error("Macromodel produced an empty output file. Exiting conformer generation")
        mm_utils.log_macromodel_errors(scr)
        raise RuntimeError("Macromodel conformer generation failed: output file is empty")

    return map_conformers_onto_template(init_mol, generated_mols)
