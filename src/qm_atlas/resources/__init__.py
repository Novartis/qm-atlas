import pathlib

RESOURCES = pathlib.Path(__file__).parent


# Bundled .cosmo solvent files for solvents not shipped with a standard
# cosmotherm installation (e.g. perfluoropyrrole, used by the ReSCoSS workflow).
COSMO_SOLVENTS_DIR = RESOURCES / "cosmo_solvents"


def get_resource(filename):
    """Get absolute path for module resource"""
    return RESOURCES / filename
