# %% [markdown]
# # Task: COSMOtherm Properties
#
# This notebook documents `qm_atlas.tasks.cosmo_properties` and related
# COSMOtherm helpers. All of them consume `.cosmo` files — the surface-charge
# output produced by a Turbomole COSMO single point — for each conformer of a
# compound.
#
# One combined notebook covers every COSMOtherm property:
#
# | Property | Config / function |
# | --- | --- |
# | log P (octanol/water) | `CosmoLogPConfig` |
# | ΔG of solvation | `CosmoDeltaGConfig` |
# | COSMO descriptors | `CosmoDescriptorsConfig` |
# | polar surface area (COSMO-PSA) | `CosmoPsaConfig` |
# | solubility | `cosmo_solubility.calculate_solubility` |
# | pKa | `qm-atlas cosmo_pka` CLI / `task_protonation` |
#
# Run the cells sequentially. Convert this script to a notebook with:
#
# ```bash
# jupytext --to notebook task_cosmo_properties.py
# ```

# %% [markdown]
# ## Software configuration
#
# `qm_atlas` locates external programs through `QM_ATLAS_SOFTWARE_CONFIG_FILE`.
# Set it via a git-ignored `.env` file in the repository root (copy
# `.env.example` to `.env`); see the README (*Software Configuration*).
#
# **Prerequisites:** all cells below need COSMOtherm. The `.cosmo` input files
# themselves come from a Turbomole COSMO single point (see the
# `workflow_rescoss_conformers` notebook for a pipeline that produces them). For
# this documentation we reuse pre-computed `.cosmo` files shipped in the test
# resources.

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
# ## Input: conformer `.cosmo` files
#
# Point `COSMO_FILES` at the `.cosmo` files for the conformers of one compound.
# Here we use the seven glycine conformers from the test resources.
#
# from pathlib import Path

# %%
from qm_atlas.tasks import cosmo_properties

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
COSMO_DIR = REPO_ROOT / "tests" / "resources" / "cosmotherm"
GLYCINE_NEUTRAL_COSMO_FILES = sorted(COSMO_DIR.glob("glycine_c*.cosmo"))
GLYCINE_ZWITTERION_COSMO_FILES = sorted(COSMO_DIR.glob("glycine_ZH*.cosmo"))
GLYCINE_ANION_COSMO_FILES = sorted(COSMO_DIR.glob("glycine_A*.cosmo"))
GLYCINE_CATION_COSMO_FILES = sorted(COSMO_DIR.glob("glycine_BH*.cosmo"))

assert GLYCINE_NEUTRAL_COSMO_FILES, f"No .cosmo files found under {COSMO_DIR}"
assert GLYCINE_ZWITTERION_COSMO_FILES, f"No .cosmo files found under {COSMO_DIR}"
assert GLYCINE_ANION_COSMO_FILES, f"No .cosmo files found under {COSMO_DIR}"
assert GLYCINE_CATION_COSMO_FILES, f"No .cosmo files found under {COSMO_DIR}"
print(f"{len(GLYCINE_NEUTRAL_COSMO_FILES)} neutral conformer .cosmo files")
print(f"{len(GLYCINE_ZWITTERION_COSMO_FILES)} zwitterion conformer .cosmo files")
print(f"{len(GLYCINE_ANION_COSMO_FILES)} anion conformer .cosmo files")
print(f"{len(GLYCINE_CATION_COSMO_FILES)} cation conformer .cosmo files")

# %% [markdown]
# ## log P (octanol / water)
#
# `CosmoLogPConfig` computes the partition coefficient. By default it uses wet
# octanol (`is_woctanol=True`) and water/octanol as the two phases. A single
# aggregated result dict is returned for the compound.

# %%
logp_config = cosmo_properties.CosmoLogPConfig(is_woctanol=True)
logp_result = cosmo_properties.run_cosmo_calculation(
    conformer_cosmo_files=GLYCINE_NEUTRAL_COSMO_FILES,
    config=logp_config,
)
for name, prop in logp_result.items():
    print(f"{name}: {prop.get_property_value()}")

# %%
scr = Path(".").resolve()
logp_result = cosmo_properties.run_cosmo_calculation(
    conformer_cosmo_files=GLYCINE_NEUTRAL_COSMO_FILES, config=logp_config, scr=scr, keep_files=True
)

# %% [markdown]
# ## ΔG of solvation
#
# `CosmoDeltaGConfig` computes the free energy of solvation in a chosen solvent
# (`solvent_name`, e.g. `"h2o"`; `"self"` uses the compound as its own solvent).

# %%
delta_g_config = cosmo_properties.CosmoDeltaGConfig(solvent_name="h2o")
delta_g_result = cosmo_properties.run_cosmo_calculation(
    conformer_cosmo_files=GLYCINE_NEUTRAL_COSMO_FILES,
    config=delta_g_config,
)
for name, prop in delta_g_result.items():
    print(f"{name}: {prop.get_property_value()}")

# %% [markdown]
# When troubleshooting a failed calculation, you may want to check the input and output files of the cosmotherm run. This can be done as shown below:

# %%
scr = Path("./_local_scr_").resolve()
scr.mkdir(exist_ok=True)
delta_g_result = cosmo_properties.run_cosmo_calculation(
    conformer_cosmo_files=GLYCINE_NEUTRAL_COSMO_FILES,
    config=delta_g_config,
    scr=scr,
    keep_files=True,
)

# %% [markdown]
# ## COSMO descriptors
#
# `CosmoDescriptorsConfig` returns a panel of sigma-profile descriptors. This is
# a **per-conformer** calculation, so the result is a list of dicts (one per
# input `.cosmo` file).

# %%
descriptors_config = cosmo_properties.CosmoDescriptorsConfig(solvent_name="h2o")
descriptors_result = cosmo_properties.run_cosmo_calculation(
    conformer_cosmo_files=GLYCINE_NEUTRAL_COSMO_FILES,
    config=descriptors_config,
)
for conf_idx, props in enumerate(descriptors_result):
    values = {name: prop.get_property_value() for name, prop in props.items()}
    print(f"Conformer {conf_idx}: {values}")

# %% [markdown]
# ## COSMO polar surface area (COSMO-PSA)
#
# `CosmoPsaConfig` integrates the polar part of the COSMO surface. Also
# per-conformer.

# %%
psa_config = cosmo_properties.CosmoPsaConfig(smoothen=True)
psa_result = cosmo_properties.run_cosmo_calculation(
    conformer_cosmo_files=GLYCINE_NEUTRAL_COSMO_FILES,
    config=psa_config,
)
for conf_idx, props in enumerate(psa_result):
    values = {name: prop.get_property_value() for name, prop in props.items()}
    print(f"Conformer {conf_idx}: {values}")

# %%
psa_config = cosmo_properties.CosmoPsaConfig(smoothen=True)
psa_result = cosmo_properties.run_cosmo_calculation(
    conformer_cosmo_files=GLYCINE_ZWITTERION_COSMO_FILES,
    config=psa_config,
)
for conf_idx, props in enumerate(psa_result):
    values = {name: prop.get_property_value() for name, prop in props.items()}
    print(f"Conformer {conf_idx}: {values}")

# %% [markdown]
# ## pKa
#
# COSMOtherm pKa prediction is exposed in `qm_atlas.wrappers.cosmotherm`.
# `cosmo_tasks.calculate_pka`, but there is no interface for it within tasks, since it expects cosmo files for two protonation states.
#
# For fast empirical pKa (MoKa) and protonation-state enumeration, see the
# `task_protonation` notebook.

# %%
from qm_atlas.wrappers.cosmotherm import cosmo_tasks

predicted_pka_value_1 = cosmo_tasks.calculate_pka(
    parent_conformers=GLYCINE_ZWITTERION_COSMO_FILES,
    child_conformers=GLYCINE_ANION_COSMO_FILES,
    form="ACID",
)

predicted_pka_value_2 = cosmo_tasks.calculate_pka(
    parent_conformers=GLYCINE_ZWITTERION_COSMO_FILES,
    child_conformers=GLYCINE_CATION_COSMO_FILES,
    form="BASE",
)
predicted_pka_value_1, predicted_pka_value_2

# %% [markdown]
# ## Solubility
#
# Solubility lives in `qm_atlas.wrappers.cosmotherm.cosmo_solubility`.
# `calculate_solubility` estimates solubility in one or more pure solvents,
# `calculate_solubility_mixture` handles binary or ternary mixtures of solvents. Without any reference data, the calculation provides results for the relative solubilities in different solvents or mixtures of solvents. Optionally, you can provide experimental melting temperature and enthalpy as well as measured reference solubilities to obtain absolute predictions.

# %%
from qm_atlas.wrappers.cosmotherm import cosmo_solubility

solubility_df = cosmo_solubility.calculate_solubility_mixture(
    compound_conformers=GLYCINE_NEUTRAL_COSMO_FILES,
    solvent_combinations=[["h2o", "ethanol"], ["h2o", "acetone"]],
    num_steps_binary=3,
    temperature=25.0,
)

parsed_solubility_df = cosmo_solubility.parse_solubility_data(solubility_df)
parsed_solubility_df

# %% [markdown]
# ## Co-crystal Screening
#
# An interface for cosmotherm co-crystal screening is exposed in `qm_atlas.wrappers.cosmotherm.cosmo_cocrystal`. cosmotherm calculates an excess enthalpy for the mixture, see the cosmotherm documentation for more details. Most coformers will not be contained in the database of .cosmo files shipped wiith cosmotherm; additional .cosmo files for coformers of interest can be provided in a separate, user-created database. Here, we use solvents as examples for simplicity

# %%
from qm_atlas.wrappers.cosmotherm import cosmo_cocrystal

cocrystal_screening_df = cosmo_cocrystal.screen_cocrystals(
    compound_conformers=GLYCINE_NEUTRAL_COSMO_FILES,
    coformers=["h2o", "ethanol"],
    coformers_dirs=None,  # point this to a directory with .cosmo files for the coformers
    # this will be searched by name in the coformers list, e.g. "ethanol" will look for "ethanol_c0.cosmo"
)
cocrystal_screening_df
