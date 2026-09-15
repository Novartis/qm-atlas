# %% [markdown]
# # Task: Conformer Generation
#
# This notebook documents `qm_atlas.tasks.conformer_generation`, which turns a
# molecular graph (SMILES / 2D `Chem.Mol`) into a molecule carrying a set of
# deduplicated 3D conformers.
#
# It supports several backends (RDKit, CORINA, OMEGA, MOE, MacroModel). The
# notebook first walks through each single generator and its typical options,
# then shows the two ways of combining them: a *fallback* chain (use the first
# one that succeeds) or a *joint set* (run several, merge, and deduplicate by
# RMSD).
#
# Run the cells sequentially. Convert this script to a notebook with
# [jupytext](https://jupytext.readthedocs.io):
#
# ```bash
# jupytext --to notebook task_conformer_generation.py
# ```

# %% [markdown]
# ## Software configuration
#
# `qm_atlas` wraps external programs and finds them through a software config
# file pointed to by the `QM_ATLAS_SOFTWARE_CONFIG_FILE` environment variable.
# Set it outside the notebook rather than hard-coding a path here — the
# recommended way is a git-ignored `.env` file in the repository root that
# VS Code's Python/Jupyter extension loads automatically at kernel start. Copy
# `.env.example` to `.env` and edit the path; see the README
# (*Software Configuration*) for details.
#
# **Prerequisites:** the RDKit backend needs no external software. The CORINA,
# OMEGA, MOE and MacroModel backends each require that program to be installed
# and enabled in your software config.

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
# ## Imports

# %%
from tempfile import TemporaryDirectory

from rdkit import Chem

from qm_atlas.tasks import conformer_generation

# Scratch directory for intermediate files (replace with a persistent path to
# keep them around).
_tmpdir = TemporaryDirectory()
SCR = Path(_tmpdir.name)

# Input molecule — swap in any SMILES.
retinoic_acid_smi = "CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C(=O)O)C(C)(C)CCC1"  # CHEMBL38
mol = Chem.MolFromSmiles(retinoic_acid_smi)
mol.SetProp("_Name", "example_molecule")
mol

# %% [markdown]
# ## Single conformer generators
#
# Every backend has its own Pydantic options class, whose ``backend`` field selects
# the generator. The cells below build a typical options object for each backend,
# one per cell, and call the matching ``generate_*_conformers`` helper (passing the
# options with ``model_dump(exclude={"backend"})``). Only the RDKit backend needs no
# external software, so we actually run it; the licensed backends require the
# program to be enabled in your software config.

# %% [markdown]
# ### RDKit (`rdkit`) — built-in, no external software
#
# Embeds conformers with a distance-geometry method (`ETKDGv3`, `KDG`, ...) and
# deduplicates them by RMSD.

# %%
rdkit_options = conformer_generation.RdkitOptions(
    method="ETKDGv3",
    max_conformers=50,
    rms_threshold=0.5,
)
print(rdkit_options)

# RDKit needs no external software, so we can run it directly:
mol_rdkit = conformer_generation.generate_rdkit_conformers(
    mol, **rdkit_options.model_dump(exclude={"backend"})
)
print(f"RDKit generated {mol_rdkit.GetNumConformers()} conformers")

# %% [markdown]
# ### CORINA (`corina`)
#
# Fast rule-based generator. `CorinaOptions` exposes no extra options; the wrapper
# drives CORINA with its default driver settings. Requires CORINA.

# %%
corina_options = conformer_generation.CorinaOptions()
print(corina_options)
# With CORINA enabled in your software config:
mol_corina = conformer_generation.generate_corina_conformers(
    mol, **corina_options.model_dump(exclude={"backend"})
)
print(f"CORINA generated {mol_corina.GetNumConformers()} conformers")

# %% [markdown]
# ### OpenEye OMEGA (`omega`)
#
# Knowledge-based conformer ensembles; can optionally start from a CORINA 3D
# structure. Requires an OpenEye OMEGA license. The code will automatically call `oeomega macrocycle` if the compound has a ring with more than 10 atoms

# %%
omega_options = conformer_generation.OmegaOptions(
    rms_threshold=0.5,
    max_conformers=100,
    start_from_corina=False,
    n_cores=1,
)
print(omega_options)
# With OMEGA enabled in your software config:
mol_omega = conformer_generation.generate_omega_conformers(
    mol, **omega_options.model_dump(exclude={"backend"})
)
print(f"OMEGA generated {mol_omega.GetNumConformers()} conformers")

# %% [markdown]
# ### MOE (`moe`)
#
# Molecular Operating Environment search (e.g. `LowModeMD`). The `mode` string
# is wrapped in single quotes as MOE expects. Requires MOE.

# %%
moe_options = conformer_generation.MoeOptions(
    rms_threshold=0.3,
    max_conformers=100,
    mode="'LowModeMD'",
)
print(moe_options)
# With MOE enabled in your software config:
mol_moe = conformer_generation.generate_moe_conformers(
    mol, **moe_options.model_dump(exclude={"backend"})
)
print(f"MOE generated {mol_moe.GetNumConformers()} conformers")

# %% [markdown]
# ### MacroModel (`macromodel`)
#
# Schrödinger MacroModel search (`MCMM`/`LMCS`) with a choice of force field and
# implicit solvent. `max_conformers=0` means unlimited. Requires MacroModel.

# %%
macromodel_options = conformer_generation.MacromodelOptions(
    rms_threshold=0.3,
    max_conformers=0,
    mode="MCMM",
    force_field=16,  # 10=MMFF94s, 14=OPLS2005, 16=OPLS4
    solvent_num=1,  # 1=water, 9=octanol
)
print(macromodel_options)
# With MacroModel enabled in your software config:
mol_macromodel = conformer_generation.generate_macromodel_conformers(
    mol, **macromodel_options.model_dump(exclude={"backend"})
)
print(f"MacroModel generated {mol_macromodel.GetNumConformers()} conformers")

# %% [markdown]
# ### MacroModel macrocycle mode (`macrocycle`)
#
# Dedicated macrocycle sampling built on MacroModel, controlled by an energy
# window (`energy_cutoff`, in kcal/mol). Requires MacroModel.

# %%
macrocycle_options = conformer_generation.MacrocycleOptions(
    max_conformers=100,
    energy_cutoff=30.0,
    rms_threshold=0.3,
    num_cores=1,
)
print(macrocycle_options)

# macrocycle example from ChEMBL, the program expects a macrocycle
CHEMBL4566196 = "C1COCCOCCOCCO1"  # CHEMBL4566196
test_macrocycle = Chem.MolFromSmiles(CHEMBL4566196)
test_macrocycle.SetProp("_Name", "CHEMBL4566196")

# With Macromodel enabled in your software config:
mol_macrocycle = conformer_generation.generate_macrocycle_conformers(
    test_macrocycle, **macrocycle_options.model_dump(exclude={"backend"})
)
print(f"Macrocycle generated {mol_macrocycle.GetNumConformers()} conformers")

# %% [markdown]
# ## Combining conformer generators
#
# Several generators are combined through the high-level `generate_conformers`
# entry point, which takes a `ConformerGenerationOptions` object. The
# `combination_name` field selects the strategy: `fallback` or `joint_set`.

# %% [markdown]
# ### Fallback chain
#
# `combination_name="fallback"` tries each generator in order and returns the
# first that succeeds — useful when a preferred (licensed) backend may be
# unavailable and RDKit should act as a safety net.

# %%
fallback_options = conformer_generation.ConformerGenerationOptions(
    conf_gens=[
        # Try OMEGA first (requires a license); fall back to RDKit.
        conformer_generation.OmegaOptions(
            rms_threshold=0.5,
            max_conformers=100,
            start_from_corina=False,
        ),
        conformer_generation.RdkitOptions(
            method="ETKDGv3",
            max_conformers=50,
            rms_threshold=0.5,
        ),
    ],
    combination_name="fallback",
)

mol_fallback = conformer_generation.generate_conformers(
    mol,
    conformer_generation_options=fallback_options,
    num_cores=1,
    trace_dir=SCR,
)
print(f"Fallback generated {mol_fallback.GetNumConformers()} conformers")

# %% [markdown]
# ### Joint set
#
# `combination_name="joint_set"` runs all generators, merges their conformers,
# and deduplicates the pooled set by RMSD. This gives broader coverage of
# conformational space at the cost of more compute.

# %%
joint_options = conformer_generation.ConformerGenerationOptions(
    conf_gens=[
        conformer_generation.RdkitOptions(method="ETKDGv3", max_conformers=30, rms_threshold=0.3),
        conformer_generation.RdkitOptions(method="ETKDG", max_conformers=30, rms_threshold=0.3),
    ],
    combination_name="joint_set",
    rms_threshold=0.3,
    rmsd_method="tradeoff",
)

mol_joint = conformer_generation.generate_conformers(
    mol,
    conformer_generation_options=joint_options,
    num_cores=1,
    trace_dir=SCR,
)
print(f"Joint set generated {mol_joint.GetNumConformers()} conformers")

# %%
# Clean up scratch directory
_tmpdir.cleanup()
