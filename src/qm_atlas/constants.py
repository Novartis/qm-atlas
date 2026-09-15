from qm_atlas import utils

DEFAULT_V3000_COORD_FORMAT = "{0:.15f}"

DEFAULT_SCR = utils.get_scratch_dir()
"""Default scratch directory, resolved at import time by
:func:`qm_atlas.utils.get_scratch_dir`."""

COSMO_OUTPUT_KEY = "cosmo_output"

# SDF File seperator
SDFSEP = "$$$$\n"

DEFAULT_XTB_OPTIONS = {"gfn": 2, "alpb": "water"}
