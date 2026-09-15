import logging
from pathlib import Path

from jinja2 import Template
from ppqm.utils.files import WorkDir
from rdkit import Chem

from qm_atlas.constants import DEFAULT_SCR
from qm_atlas.software_environment import SOFTWARE_CONFIG
from qm_atlas.wrappers import corina
from qm_atlas.wrappers.common import run_command
from qm_atlas.wrappers.helpers import (
    map_conformers_onto_template,
    read_conformers_as_mols,
    store_atom_map_numbers,
)

_logger = logging.getLogger(__name__)


MOE_CONF_SCRIPT = """#svl
function db_ImportSD;
function ConfSearch;
function db_ExportSD;

function main []
  local fb = db_Open [ 'fb.mdb', 'create' ];
  db_ImportSD [ 'fb.mdb', 'moe_input.sdf', 'molecule' ];

  ConfSearch [ outfile: 'csq.mdb', infile: 'fb.mdb', infile_data: 1, infile_esel: 0,
               rot_amide: 1, rot_double: 1, invert_sp3: 0, chair_only: 0, pot_charge: 1,
               method: {{ mode }}, maxfail: 100, maxit: 1000,
               maxconf: {{ max_conformers }}, gtest: 0.005, mm_maxit: 100,
               cutoff: 7, cutoff_chi: 1, rmsd: {{ rms_threshold_no_H }},
               rmsd_H: {{ rms_threshold }}, free_shape: 0];

  db_ExportSD [ 'csq.mdb', 'moe_output.sdf', [] ];
endfunction
"""


MOE_DEFAULTS = {
    "mode": "'LowModeMD'",
    "max_conformers": 1000,
    "rms_threshold": 0.3,
}


def setup_moe_conf(
    template_str: str = MOE_CONF_SCRIPT,
    scr: Path = DEFAULT_SCR,
    keep_files: bool = False,
    **conf_gen_kwargs,
) -> tuple[WorkDir, Path]:
    """sets up a temporary directory to run the MOE confsearch in.

    Args:
        template_str (str, optional):
            A template for the MOE conformer search script (svl).
            Defaults to MOE_CONF_SCRIPT.
        scr (Path, optional):
            The scratch space to use for the temp dir. Defaults to DEFAULT_SCR.
        keep_files (bool, optional):
            whether to keep the files. Needed for debugging. Defaults to False.
        **conf_gen_kwargs:
            Additional MOE conformer-search parameters substituted into the script
            template (e.g. mode, max_conformers, rms_threshold).

    Returns:
        temp (WorkDir):
            The WorkDir instance
        moe_script (Path):
            The path to the moe script.
    """

    # combine MOE keyword arguments with defaults
    moe_kwargs = dict(MOE_DEFAULTS)
    moe_kwargs.update(conf_gen_kwargs)

    # default is to include Hs in rmsd comparison
    if "rms_threshold_no_H" not in moe_kwargs:
        moe_kwargs["rms_threshold_no_H"] = moe_kwargs["rms_threshold"]

    # set up temporary directory
    temp = WorkDir(dir=scr, prefix="moe_conf_", keep=keep_files)
    scr = temp.get_path()
    scr.mkdir(parents=True, exist_ok=True)

    # create conformer generation script
    conf_gen_template = Template(template_str)
    msg = conf_gen_template.render(**moe_kwargs)
    moe_script = scr / "moe_conf.svl"
    with open(moe_script, "w") as outf:
        outf.write(msg)

    return temp, moe_script


def generate_conformers(
    mol: Chem.Mol,
    scr: Path = DEFAULT_SCR,
    keep_files: bool = False,
    **conf_gen_kwargs,
) -> Chem.Mol:
    """Generate Conformers using MOE.

    Args:
        mol (Chem.Mol):
            The molecule to work on.
        scr (Path, optional):
            The scratch space to use. Defaults to DEFAULT_SCR.
        keep_files (bool, optional):
            Whether to keep the temporary files. Used for debugging. Defaults to False.
        **conf_gen_kwargs:
            Additional MOE conformer-search parameters forwarded to :func:`setup_moe_conf`.

    Raises:
        RuntimeError: If the conformer generation fails or no output file is produced.

    Returns:
        Chem.Mol: An rdkit mol with conformers. Returns None if an error is raised.
    """

    # Set up temporary directory with run script
    temp, moe_script = setup_moe_conf(keep_files=keep_files, scr=scr, **conf_gen_kwargs)
    scr = temp.get_path()

    # create initial molecule and write to sdf file. Atom map numbers are stored
    # so the original graph can be restored after conformer generation.
    init_mol = corina.generate_conformers_molobj(Chem.AddHs(mol))
    input_mol = store_atom_map_numbers(init_mol)
    sdf_file = scr / "moe_input.sdf"
    with Chem.SDWriter(str(sdf_file.resolve())) as writer:
        writer.write(input_mol)

    # run MOE
    software_env_manager = SOFTWARE_CONFIG.get_environment_manager()
    moebatch_cmd = software_env_manager.get_command("moe", "moebatch")
    moe_cmd = f"{moebatch_cmd} -run {moe_script.name}"
    env = software_env_manager.get_run_environment("moe")
    run_command(moe_cmd, env=env, cwd=scr)

    moe_outfile = scr / "moe_output.sdf"

    if not moe_outfile.is_file():
        _logger.error("No MOE output found. Exiting conformer generation")
        raise RuntimeError("No MOE output found. Exiting conformer generation")

    generated_mols = read_conformers_as_mols(moe_outfile)
    if not generated_mols:
        raise RuntimeError("MOE conformer generation failed: output file is empty")

    return map_conformers_onto_template(init_mol, generated_mols)
