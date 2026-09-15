"""CORINA 3D structure generator command-line reference.

The text below is the verbatim CORINA command-line help, kept here for reference.

::

    NAME
    corina - 3D structure generator
    (All rights Molecular Networks GmbH, www.mn-am.com)

    SYNOPSIS
    corina [options] [infile [outfile]]

    OPTIONS
    Options follow the UNIX command line convention: -o [suboption[=value],...]
    I.e., options start with a `-' and are separated by spaces.
    Suboptions may have values and can be concatenated by commas.
    Example: corina -n f=42,t=43 < in > out

    -i - Input file type and options
        t=<value> - File type (sdf, rdf, smiles, mol2, mol, ctx, mmod, mae, inchi)
        sdfi2n=<value> - Copy data item <value> to name (sdf)
        sdfi2c=<value> - Copy data item <value> to comment (sdf)
        dummies - Allow dummy atom types (smiles/sdf/mol/mol2)
        sdfict - Ignore E/Z configurations from 2D coordinates (sdf)
        expandapo - Expand attachment points "M  APO" (sdf)
        force3d - Force 3D from 2D (sdf)
        csdmol2 - Allow CSD MOL2 extensions (mol2)
        xelement - Allow extra elements (mol2)
        sc#=<value> - Read structure from column number <value> (smiles/InChI)
        scn=<value> - Read structure from column number <value> (smiles/InChI) - same as sc#
        nc#=<value> - Read name from column number <value> (smiles/InChI)
        ncn=<value> - Read name from column number <value> (smiles/InChI) - same as nc#
        cc#=<value> - Read comment from column number <value> (smiles/InChI)
        ccn=<value> - Read comment from column number <value> (smiles/InChI) - same as cc#
        sep=<value> - Provide input file column separator character (smiles/InChI)
    -o - Output file type and options
        t=<value> - File type (sdf, sdf2, sdf3, rdf, rdf2, rdf3, mol, mol2, pdb,
        ctx, mmod, mae, top, par, odb, cif, dic, dyana, mopacxyz)

        a - Append output to output file
        pascom - Pass comment from input to output
        mdldb - Write additional data fields in SDFiles for MDL databases (sdf)
        mdlcompact - Write compact SDFiles (sdf)
        mdl3dparity - Add atom stereo parities (sdf)
        lname - Allow long names >80 char. (sdf)
        mdlbond4 - Write aromatic bonds as bond type "4" (sdf)
        sdfn2i=<value> - Write name into data item <value> (sdf)
        sdfc2i=<value> - Write comment into data item <value> (sdf)
        sdfs2i=<value> - Write structure (smiles/InChI)) into data item <value> (sdf)
        gold - Assign atom and bond types required by GOLD (mol/mol2)
        m2l - Add isotopic mass labels to atom names (mol/mol2)
        noccat - Suppress automatic amidinium conversion to C.cat (mol/mol2)
        fcharges - Write formal charges into the partial charge column (mol2)
        nodummies - Suppress writing of records with dummy atoms (mol/mol2)
        xelement - Allow extra elements (mol2)
        pdbatom - Write ATOM instead of HETATM keyword (pdb)
        pdbnoconect - Skip CONECT statements (pdb)
        pdbludi - Create input file for Ludi fragment database (pdb)
        pdbludilabel - Generate unique labels for Ludi fragments (pdb)
        pdbelement - Add atomic element symbol to HETATM/ATOM lines (PDB V2.3)
        resnam=<value> - Set residue name to <value> (pdb/mmod/mae/top/dic/cif/odb/dyana)
        resno=<value> - Set residue number to <value> (pdb/mmod/mae/dyana)
        keepnames - Keep any given atom names (pdb/dyana)
        typchr=<value> - Set atom type character(s) to <value> (top/par)
        dicid=<value> - Set group ID number to <value> (dic)
        novar - No variable torsions (cif)
        multor - Multiple torsions (cif)
        coplan - Set strict restraints for coplanar methoxy groups at aromatic rings (cif)
        chirvol - Set chiral volume to value "both" at flexible ring systems
        hlabel - Label hydrogen atoms separately (pdb, cif)
        xlabel - Label each atom type separately by applying extended schema [0-9;A-Z] (pdb/cif/mmod/maestro)
        flexrta - Set flexible TAs for aliphatic ring systems (size > 4, cif)
        mopackeys=<value> - Write MOPAC keywords '<value>' (mopacxyz, '<value>' to be quoted)
        mopacaddchg - Calculate and add MOPAC CHARGE keyword automatically (mopacxyz)
        mopacoptflag=<value> - Write MOPAC optimization flag <value> (mopacxyz)
        split - Split output to single structure files, number output file names consecutively
        splitn0=<value> - Split output to single structure files, number output file names consecutively with <value> leading '0's
    -t - Trace
        s - Write trace output to stderr
        n - Suppress trace output
        tracefile=<value> - Set tracefile name to <value>
    -n - Record number
        n=<value> - Process only record <value>
        f=<value> - Process all records from <value>
        t=<value> - Process all records to <value>
    -d - CORINA driver options
        wh - Write added hydrogen atoms
        rs - Remove small fragments (counter ions, solvent molecules, etc.)
        neu - Neutralize charges of [C,S,P]-[O-], [NH+] and S=O(=O)[N-]
        maxat=<value> - Change maximum number of atoms (default 999)
        flexx - Set all parameters to interface to FlexX
        axchir - Process potential stereogenic axes for handling of axial chirality
        stergen - Generate stereo isomers
        msc=<value> - Set maximum number of processed stereo centers to <value> (-d stergen)
        msi=<value> - Set maximum number of output stereoisomers to <value> (-d stergen)
        preserve - Preserve defined stereo centers (-d stergen)
        preserverel - Preserve relative stereochemistry of defined stereo centers (-d stergen)
        chiralflag - Preserve defined stereo centers if chiral flag is set (sdf, -d stergen)
        v3000 - Permute stereo centers and groups according to SD V3000 specifications (-d stergen)
        noflapn - Do not flap pyramidal nitrogens (-d stergen)
        rc - Generate multiple ring conformations
        mc=<value> - Set maximum number of conformations to <value> (-d rc)
        de=<value> - Set energy window delta E for multiple conformations to <value> kJ/mol (-d rc)
        timeout=<value> - Set timeout for multiple conformations to <value> ms (-d rc)
        flapn - Flap pyramidal ring nitrogen in multiple conformations (-d rc)
        sc - Generate multiple ring conformations "simultaneously" (-d rc)
        symoff - Suppress symmetry check for conformations (-d rc)
        ringatom=<value> - Denote ring system for conformation analysis by atom <value> (-d rc)
        ampax - Amplify energy penalty for axial ring substituents
        planil - Force anilinic ring nitrogen atoms to planar geometry
        sanpyr - Force pyramidal geometry of sulfonamide nitrogen atoms (default: planar)
        names - Append conformer/isomer count to compound names (-d rc,stergen)
        r2d - Remove 2D records (failure cases) from the output
        wb - Write also "bad" 3D models having close contacts
        no3d - Suppress 3D generation process (file conversion only)
        ori - Orient 3D structures according to their moments of inertia
        ist - Ignore all stereo information
        i3dst - Ignore stereo information derived from 3D structure given in input file
        ow - Override wedge symbols at stereo centers that have also stereo flag set
        amide - Generate amide bonds with configuration given in input file
        3dst - Force stereo descriptors from 3D structure given in input file
        newtypes - Generate new atom types, ignore any given type and aromaticity
        errorfile=<value> - Name of a file containing all erroneous molecules (sdf/smiles/InChI)
    -v - Version
    -h - This help
    -h <o> | all - Details for option <o> or for all options.
    -m - write man page to stdout
"""


# LDC Defaults
# maxStereo=2
# stereoOptions = 'stergen'
# if enableAxialChirality:
#     stereoOptions += ',axchir'
# driverOptions =
# 'wh,{0},noflapn,preserve,chiralflag,msc={1}'.format(stereoOptions, maxStereo)
# traceOptions = 'tracefile={0}'.format(traceFilePath)

# ResCoSS Defaults
# corina -d wh,r2d %s %s" % (molecule, name3d)

import logging

from ppqm import chembridge
from ppqm.utils import WorkDir
from rdkit import Chem
from rdkit.Chem import AllChem

from qm_atlas import constants, utils
from qm_atlas.software_environment import SOFTWARE_CONFIG
from qm_atlas.wrappers.common import run_command

CORINA_CMD = "corina"
CORINA_DRIVEROPT = "-d wh,stergen,noflapn,preserve,chiralflag,msc=2"
CORINA_FILES = ["corina.trc"]

_logger = logging.getLogger(__name__)


def generate_conformers_molobj(
    mol: Chem.Mol,
    corina_cmd=None,
    corina_driveropt=CORINA_DRIVEROPT,
    scr=constants.DEFAULT_SCR,
    keep_files: bool = False,
) -> Chem.Mol:
    """Generate a 3D conformer for a molecule using CORINA.

    Args:
        mol (Chem.Mol):
            The rdkit Mol to generate a conformer for.
        corina_cmd (str | None):
            The command to use for corina. If None, the command is resolved from the
            software configuration.
        corina_driveropt (str):
            The CORINA driver options string. Defaults to :data:`CORINA_DRIVEROPT`.
        scr (Path):
            The scratch directory to use for temporary files. Defaults to
            constants.DEFAULT_SCR.
        keep_files (bool):
            Whether to keep the temporary files. Defaults to False. Useful for debugging.

    Raises:
        RuntimeError: If the CORINA conformer generation fails.

    Returns:
        Chem.Mol: The molecule with the 3D conformer generated by CORINA.
    """

    molobj = chembridge.copy_molobj(mol)

    # Ensure starting conformer for SDF input
    AllChem.Compute2DCoords(molobj)

    # Generate input sdf string and start generating sdfs
    sdfstr = chembridge.molobj_to_sdfstr(molobj)

    # Generate conformers
    sdfstrs = _generate_conformers_sdf(
        sdfstr,
        corina_cmd=corina_cmd,
        corina_driveropt=corina_driveropt,
        scr=scr,
        keep_files=keep_files,
    )

    for sdf in sdfstrs:

        molobj_prime = chembridge.sdfstr_to_molobj(sdf)

        if molobj_prime is None:
            _logger.error("conformer failed, skipped")
            continue

        # This obj should only contain one conformer
        conf = molobj_prime.GetConformer()
        # coordinates = conf.GetPositions()
        # coordinates = np.array(coordinates)
        # conf = Chem.Conformer(n_atoms)
        # chembridge.conformer_set_coordinates(conf, coordinates)

        # Add corina conformer to original molobj
        molobj.RemoveAllConformers()
        molobj.AddConformer(conf, assignId=True)

    return molobj


def _generate_conformers_sdf(
    sdfstr,
    corina_cmd=None,
    corina_driveropt=CORINA_DRIVEROPT,
    scr=constants.DEFAULT_SCR,
    keep_files: bool = False,
):

    # Get command from environment manager
    software_env_manager = SOFTWARE_CONFIG.get_environment_manager()
    if corina_cmd is None:
        corina_cmd = software_env_manager.get_command("corina", "corina")

    temp = WorkDir(dir=scr, prefix="corina_", keep=keep_files)
    work_dir = temp.get_path()

    sdfseperator = "$$$$"
    filename = "_tmp_corina.sdf"

    with utils.open_utf8(work_dir / filename, "w") as f:
        f.write(sdfstr)

    cmd = " ".join([corina_cmd, corina_driveropt, f"{filename}"])
    cmd = f"cd {work_dir}; " + cmd

    _logger.debug(cmd)

    env = software_env_manager.get_run_environment("corina")
    stdout, _ = run_command(cmd, env=env)
    lines = stdout.split("\n")

    # Stream the result in sdfs
    sdf = []
    for line in lines:

        if sdfseperator in line:
            yield "\n".join(sdf)
            sdf = []
        else:
            sdf.append(line)

    # at least three lines in sdf
    if len(sdf) > 3:
        yield "".join(sdf)
