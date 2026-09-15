# %% [markdown]
# # Task: Protonation State Enumeration
#
# This notebook documents `qm_atlas.tasks.protonation`, which enumerates the
# relevant protonation states (protomers) of a molecule at a given pH using
# MoKa. Three backends are available:
#
# | Config | Backend | Notes |
# | --- | --- | --- |
# | `MokaBlabberConfig` | MoKa `blabber_sd` | protomers with abundances at a pH |
# | `MokaIonicSpeciesConfig` | MoKa pKa | ionic species within a pH window |
# | `MokaSingleChargeConfig` | MoKa pKa | singly-charged ions within a pH window |
#
# All backends return a `ProtonationResult` with a list of `ProtomerEntry`
# objects (`.mol`, `.abundance`) and, where available, pKa `transitions`.
#
# Run the cells sequentially. Convert this script to a notebook with:
#
# ```bash
# jupytext --to notebook task_protonation.py
# ```

# %% [markdown]
# ## Software configuration
#
# `qm_atlas` locates external programs through `QM_ATLAS_SOFTWARE_CONFIG_FILE`.
# Set it via a git-ignored `.env` file in the repository root (copy
# `.env.example` to `.env`); see the README (*Software Configuration*).
#
# **Prerequisites:** all backends in this notebook require MoKa.

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

from qm_atlas.tasks import protonation

_tmpdir = TemporaryDirectory()
SCR = Path(_tmpdir.name)

glycine = Chem.MolFromSmiles("C(C(=O)O)N")  # glycine, ChEMBL773
glycine.SetProp("_Name", "glycine")
glycine

# %% [markdown]
# ## Protomers at physiological pH (blabber)
#
# `MokaBlabberConfig` returns protomers with an abundance (population in
# percent) at the requested pH. Use `abundance_threshold` to drop rare states.

# %%
blabber_config = protonation.MokaBlabberConfig(
    ph="7.4",
    abundance_threshold=1.0,  # drop protomers below 1 % abundance
    remove_zero_charged=False,
)

result = protonation.run_protonation(glycine, config=blabber_config, scr=SCR)

print(f"{len(result.protomers)} protomer(s):")
for entry in result.protomers:
    smiles = Chem.MolToSmiles(entry.mol)
    charge = Chem.GetFormalCharge(entry.mol)
    print(f"  {smiles}  charge={charge:+d}  abundance={entry.abundance}")

# %% [markdown]
# ## Scanning a pH range
#
# `ph` accepts a single value, a range (`"6-10"`) or a comma-separated list.

# %%
range_config = protonation.MokaBlabberConfig(
    ph="2-12",
    abundance_threshold=1.0,  # drop protomers below 1 % abundance
    # without an abundance threshold, blabber reports the most populated protonation state only
)
range_result = protonation.run_protonation(glycine, config=range_config, scr=SCR)
print(f"{len(range_result.protomers)} protomers across pH 2-12")
for entry in range_result.protomers:
    print(f"  {Chem.MolToSmiles(entry.mol)}  charge={Chem.GetFormalCharge(entry.mol):+d}")

# %% [markdown]
# ## pKa-driven backends
#
# The pKa-based backends take an optional pH window
# (`lower_ph_threshold` / `upper_ph_threshold`) instead of an explicit pH:
#
# * `MokaIonicSpeciesConfig` — enumerate ionic species. Due to the way MoKa works, you'll obtain a single protonation state per charge
# * `MokaSingleChargeConfig` — enumerate singly-charged ions. Each center will be (de-)protonated separately. This can e.g. be useful for comparing different centers

# %%
ionic_config = protonation.MokaIonicSpeciesConfig(
    lower_ph_threshold=2.0,
    upper_ph_threshold=12.0,
)

# Uncomment to run (needs MoKa):
moka_result = protonation.run_protonation(glycine, config=ionic_config, scr=SCR)
for entry in moka_result.protomers:
    print(Chem.MolToSmiles(Chem.RemoveHs(entry.mol)))

# %% [markdown]
# Running in this way, you don't obtain values for the abundance of the protonation states, but pKa values and the associated protonation states. Where the backend reports them, `result.transitions` links pairs of protonation states by
# their predicted pKa.

# %%
for transition in moka_result.transitions:
    print(
        f"{transition.pka_type} pKa={transition.pka_value:.2f} "
        f"atom={transition.atom_index}: "
        f"{transition.parent_smiles} -> {transition.child_smiles}"
    )

# %% [markdown]
# To compare the single charge approach, we consider salicylic acid:

# %%
salicylic_acid = Chem.MolFromSmiles("O=C(O)c1ccccc1O")  # salicylic acid, ChEMBL424
salicylic_acid.SetProp("_Name", "salicylic acid")
salicylic_acid

# %% [markdown]
# Proceeding as above we get a protonation state with charge -2:

# %%
ionic_config = protonation.MokaIonicSpeciesConfig(
    lower_ph_threshold=-2.0,
    upper_ph_threshold=16.0,
)

# Uncomment to run (needs MoKa):
moka_result = protonation.run_protonation(salicylic_acid, config=ionic_config, scr=SCR)
for entry in moka_result.protomers:
    print(Chem.MolToSmiles(Chem.RemoveHs(entry.mol)))

for transition in moka_result.transitions:
    print(
        f"{transition.pka_type} pKa={transition.pka_value:.2f} "
        f"atom={transition.atom_index}: "
        f"{transition.parent_smiles} -> {transition.child_smiles}"
    )

# %% [markdown]
# The "single charge" approach will instead deprotonate the centers one by one:

# %%
ionic_config = protonation.MokaSingleChargeConfig(
    lower_ph_threshold=-2.0,
    upper_ph_threshold=16.0,
)

moka_result = protonation.run_protonation(salicylic_acid, config=ionic_config, scr=SCR)
for entry in moka_result.protomers:
    print(Chem.MolToSmiles(Chem.RemoveHs(entry.mol)))

for transition in moka_result.transitions:
    print(
        f"{transition.pka_type} pKa={transition.pka_value:.2f} "
        f"atom={transition.atom_index}: "
        f"{transition.parent_smiles} -> {transition.child_smiles}"
    )

# %%
_tmpdir.cleanup()
