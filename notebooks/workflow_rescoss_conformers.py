# %% [markdown]
# # Workflow: ReSCoSS Conformers
#
# This notebook documents `qm_atlas.workflows.generate_rescoss_conformers`, the
# **Re**levant **S**olution **Co**nformer **S**election **S**ystem. It produces
# a compact set of conformers meant to represent the geometries that matter
# across several solvents, combining force-field generation, xTB optimization,
# COSMOtherm solvent screening, clustering, and (optionally) DFT re-optimization.
#
# Run the cells sequentially. Convert this script to a notebook with:
#
# ```bash
# jupytext --to notebook workflow_rescoss_conformers.py
# ```

# %% [markdown]
# ## Software configuration
#
# `qm_atlas` locates external programs through `QM_ATLAS_SOFTWARE_CONFIG_FILE`.
# Set it via a git-ignored `.env` file in the repository root (copy
# `.env.example` to `.env`); see the README (*Software Configuration*).
#
# **Prerequisites:** this is a heavy workflow. It needs xTB, Turbomole and
# COSMOtherm, plus a force-field conformer generator (MacroModel, MOE, OMEGA or
# RDKit). Expect it to take a while even for small molecules.

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

from qm_atlas.workflows import rescoss_conformers

_tmpdir = TemporaryDirectory()
SCR = Path(_tmpdir.name)

mol = Chem.MolFromSmiles("C(C(=O)O)N")  # glycine
mol.SetProp("_Name", "glycine")

# The solvents the workflow screens over:
print("ReSCoSS solvents:", rescoss_conformers.RESCOSS_SOLVENTS)

# %% [markdown]
# ## Run the workflow
#
# The key knobs are `num_clusters` (how many conformer clusters to keep) and
# `num_per_cluster` (how many representatives per cluster). The workflow returns
# a single molecule carrying the selected conformers.

# %%
# This cell performs real xTB + Turbomole + COSMOtherm calculations and can be
# slow. Needs to run in an environment where those programs are configured.

rescoss_mol = rescoss_conformers.generate_rescoss_conformers(
    mol,
    num_clusters=3,
    num_per_cluster=3,
    rmsd_threshold=rescoss_conformers.RMSD_THRESHOLD,
    scr=SCR,
    n_cores=4,
)
print(f"ReSCoSS returned {rescoss_mol.GetNumConformers()} conformers")

# %% [markdown]
# ## Adapting the final DFT re-optimization
#
# By default the selected conformers are re-optimized with
# `DEFAULT_FINAL_OPTIMIZATION_CONFIG`. Pass `final_optimization_config=None` to
# skip it, or provide your own list of optimization configs (see the
# `task_optimize` notebook). Custom solvent `.cosmo` directories can be added
# via `solvents_dirs`.

# %%
_tmpdir.cleanup()
