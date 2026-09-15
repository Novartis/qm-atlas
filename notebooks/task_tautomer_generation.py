# %% [markdown]
# # Task: Tautomer Generation and Selection
#
# This notebook documents `qm_atlas.tasks.tautomer_generation`, a three-step
# pipeline for enumerating a molecule's tautomers and keeping the energetically
# relevant ones:
#
# 1. `generate_tautomers` — enumerate tautomers from a graph (uses Unicon).
# 2. `generate_tautomer_conformers` — embed fast 3D conformers for each tautomer.
# 3. `select_stable_tautomers` — screen by xTB energy and keep those within an
#    energy window of the most stable tautomer.
#
# Run the cells sequentially. Convert this script to a notebook with:
#
# ```bash
# jupytext --to notebook task_tautomer_generation.py
# ```

# %% [markdown]
# ## Software configuration
#
# `qm_atlas` locates external programs through `QM_ATLAS_SOFTWARE_CONFIG_FILE`.
# Set it via a git-ignored `.env` file in the repository root (copy
# `.env.example` to `.env`); see the README (*Software Configuration*).
#
# **Prerequisites:** tautomer enumeration needs Unicon; conformer generation and
# energy screening need xTB (plus OMEGA or RDKit for embedding).

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

from qm_atlas.tasks import tautomer_generation

_tmpdir = TemporaryDirectory()
SCR = Path(_tmpdir.name)

mol = Chem.MolFromSmiles("Nc1nc2[nH]cnc2c(=O)[nH]1")  # guanine, CHEMBL219568
mol.SetProp("_Name", "guanine")
mol

# %% [markdown]
# ## Step 1 — Enumerate tautomers
#
# `generate_tautomers` returns a list of graph-level `Chem.Mol` tautomers.

# %%
tautomers = tautomer_generation.generate_tautomers(mol, scr=SCR)
print(f"Generated {len(tautomers)} tautomers")
for taut in tautomers:
    print(Chem.MolToSmiles(taut))

# %% [markdown]
# ## Step 2 — Embed conformers
#
# `generate_tautomer_conformers` embeds a small conformer ensemble for each
# tautomer (defaults to the `TAUTOMER_CONFORMER_OPTIONS` preset: OMEGA → RDKit
# fallback, up to five conformers).

# %%
tautomers_3d = tautomer_generation.generate_tautomer_conformers(
    tautomers,
    options=tautomer_generation.TAUTOMER_CONFORMER_OPTIONS,
    scr=SCR,
    n_cores=1,
    show_progress=True,
)
for taut in tautomers_3d:
    print(f"{Chem.MolToSmiles(Chem.RemoveHs(taut))}: {taut.GetNumConformers()} conformers")

# %% [markdown]
# ## Step 3 — Select stable tautomers
#
# `select_stable_tautomers` runs an xTB single point on every conformer, takes
# the lowest energy per tautomer, and keeps tautomers within
# `threshold_kcal_mol` of the most stable one. The returned molecules are clean
# graph objects (conformers stripped, explicit Hs removed).

# %%
stable = tautomer_generation.select_stable_tautomers(
    tautomers_3d,
    threshold_kcal_mol=4.0,
    n_cores=1,
    scr=SCR,
    show_progress=True,
)
print(f"Kept {len(stable)} stable tautomer(s):")
for taut in stable:
    print(Chem.MolToSmiles(taut))

# %% [markdown]
# The `xtb_sp_config` argument lets you change the screening level (defaults to
# the ALPB/water `XTB_SP_CONFIG` preset). For a higher-accuracy DFT/COSMOtherm
# tautomer ranking, see the `workflow_score_tautomers` notebook.

# %%
_tmpdir.cleanup()
