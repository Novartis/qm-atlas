# %% [markdown]
# # Workflow: Score Tautomers (DFT / COSMOtherm)
#
# This notebook documents `qm_atlas.workflows.score_tautomers.generate_tautomers`
# (also exposed as `qm_atlas.workflows.generate_tautomers`). It is the
# high-accuracy sibling of the `task_tautomer_generation` pipeline: it
# enumerates tautomers, expands conformers, and ranks them using Turbomole DFT
# single points with COSMOtherm solvation, keeping those within an energy window.
#
# Use the **task** version (`task_tautomer_generation`) for fast xTB-level
# screening, and this **workflow** version when you need DFT/COSMOtherm-quality
# tautomer free energies.
#
# Run the cells sequentially. Convert this script to a notebook with:
#
# ```bash
# jupytext --to notebook workflow_score_tautomers.py
# ```

# %% [markdown]
# ## Software configuration
#
# `qm_atlas` locates external programs through `QM_ATLAS_SOFTWARE_CONFIG_FILE`.
# Set it via a git-ignored `.env` file in the repository root (copy
# `.env.example` to `.env`); see the README (*Software Configuration*).
#
# **Prerequisites:** this is a heavy workflow. It needs Unicon (tautomer
# enumeration), xTB (conformer optimization), Turbomole (DFT single points) and
# COSMOtherm (solvation free energies).

# %%
import os
from pathlib import Path

_config_file = os.environ.get("QM_ATLAS_SOFTWARE_CONFIG_FILE")
if not _config_file:
    raise EnvironmentError(
        "QM_ATLAS_SOFTWARE_CONFIG_FILE is not set.\n"
        "Copy .env.example to .env and set the path to your software config, "
        "or export the variable before launching VS Code / Jupyter.\n"
        "See the README (Software Configuration) for details."
    )
if not Path(_config_file).is_file():
    raise FileNotFoundError(f"Software config file does not exist: {_config_file}")

print(f"Using software config: {_config_file}")

# %% [markdown]
# ## Imports and input molecule

# %%
from tempfile import TemporaryDirectory

from rdkit import Chem

from qm_atlas.workflows import score_tautomers

_tmpdir = TemporaryDirectory()
SCR = Path(_tmpdir.name)

mol = Chem.MolFromSmiles("Nc1nc2[nH]cnc2c(=O)[nH]1")  # guanine
mol.SetProp("_Name", "guanine")
mol

# %% [markdown]
# ## Run the workflow
#
# `energy_threshold` (kcal/mol) sets how far above the most stable tautomer to
# keep, and `solvent_name` selects the COSMOtherm solvent. The function returns
# a list of the surviving tautomers as `Chem.Mol` objects.

# %%
# This cell runs real DFT + COSMOtherm calculations and can be slow. Uncomment
# to run where Unicon, xTB, Turbomole and COSMOtherm are configured.

stable_tautomers = score_tautomers.generate_tautomers(
    mol,
    energy_threshold=1.5,  # kcal/mol, the threshold is artificially low so that some conformers are filtered out
    solvent_name="h2o",
    n_cores=1,
    scr=SCR,
)
print(f"Kept {len(stable_tautomers)} tautomer(s):")
for taut in stable_tautomers:
    print(Chem.MolToSmiles(taut))

# %% [markdown]
# ## Customising the levels of theory
#
# The DFT single point (`single_point_config`), COSMOtherm settings
# (`cosmo_options`), conformer generation and optimization stages can all be
# overridden. Defaults mirror the other workflows:
#
# * `single_point_config` — `TurbomoleSinglePointOptions` preset for tautomers.
# * `conformer_generation_options` — the fast-conformer default.
# * `optimization_config` / `final_optimization_config` — the fast-conformer
#   xTB configs.
#
# See the `task_calculate_properties` and `workflow_fast_conformers` notebooks
# for how to build those objects.

# %%
_tmpdir.cleanup()
