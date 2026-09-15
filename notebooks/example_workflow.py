# %% [markdown]
# # Full Graph-Preparation Workflow
#
# This script walks through a pipeline for generating tautomers, protonation states,
# stereoisomers, and conformers for a drug-like molecule. For all of the generated
# conformers, the script calculates XTB single-point energies in vacuum and in solvent.
#
# Each cell corresponds to one discrete task.  Run them sequentially, or
# use [jupytext](https://jupytext.readthedocs.io) to convert to a `.ipynb`:
#
# ```bash
# jupytext --to notebook full_workflow.py
# ```

# %% [markdown]
# ## Software configuration
#
# `qm_atlas` wraps external programs (xTB, Turbomole, COSMOtherm, ...). It finds
# them through a software config file pointed to by the
# `QM_ATLAS_SOFTWARE_CONFIG_FILE` environment variable.
#
# Rather than hard-coding a path here, set the variable outside the notebook so
# this file stays portable. The recommended way is a git-ignored `.env` file in
# the repository root — VS Code's Python/Jupyter extension loads it automatically
# when the kernel starts:
#
# ```properties
# QM_ATLAS_SOFTWARE_CONFIG_FILE=/path/to/your/software_config.yaml
# ```
#
# Copy `.env.example` to `.env` and edit the path (restart the kernel after
# changing `.env`). See the README (*Software Configuration*) for details and
# alternatives.

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
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from ppqm import chembridge
from rdkit import Chem
from rdkit.Chem import rdmolops

from qm_atlas.command_line.file_interface import input_check
from qm_atlas.tasks.calculate_properties import (
    XTB_SP_CONFIG,
    XTB_SP_SOLVENT_CONFIGS,
    get_xtb_energies_and_weights,
)
from qm_atlas.tasks.protonation import MokaBlabberConfig, run_protonation
from qm_atlas.tasks.tautomer_generation import (
    TAUTOMER_CONFORMER_OPTIONS,
    generate_tautomer_conformers,
    generate_tautomers,
    select_stable_tautomers,
)
from qm_atlas.utils import remove_salt
from qm_atlas.workflows.fast_conformers import generate_fast_conformers
from qm_atlas.wrappers.moka import blabber

# %% [markdown]
# ## Configuration

# %%
# Input molecule — swap in any SMILES or pass a Chem.Mol object
SMILES = "C[C@H](O)[C@H](C)[C@H](N)C(=O)O"  # CHEMBL508094

# Energy window for tautomer filtering
THRESHOLD_KCAL_MOL = 7.0

# Parallelism
N_CORES = 1

# Scratch directory (replace with a persistent path if you want to keep files)
_tmpdir = TemporaryDirectory()
SCR = Path(_tmpdir.name)

# %% [markdown]
# ## Step 1 — Parse and validate the molecule

# %%
then = time.time()

smiles = remove_salt(SMILES)
molobj = chembridge.smiles_to_molobj(smiles)
print(f"Parsed SMILES: {smiles}")

ignore_reason = input_check.ignore_molecule(molobj)
if ignore_reason is not None:
    raise ValueError(f"Molecule rejected: {ignore_reason}")

molobj = chembridge.neutralize_molobj(molobj)
print("Molecule passed validation and was neutralized.")

# %% [markdown]
# ## Step 2 — Generate tautomers

# %%
tautomers = generate_tautomers(molobj, scr=SCR)
print(f"Found {len(tautomers)} unique tautomer(s)")

# %% [markdown]
# ## Step 3 — Generate fast conformers for each tautomer

# %%
tautomers_3d = generate_tautomer_conformers(
    tautomers,
    options=TAUTOMER_CONFORMER_OPTIONS,
    scr=SCR,
    n_cores=N_CORES,
    show_progress=True,
)
print(f"Conformers per tautomer: {[t.GetNumConformers() for t in tautomers_3d]}")

# %% [markdown]
# ## Step 4 — Filter stable tautomers by XTB energy

# %%
stable_tautomers = select_stable_tautomers(
    tautomers_3d,
    xtb_sp_config=XTB_SP_CONFIG,
    threshold_kcal_mol=THRESHOLD_KCAL_MOL,
    n_cores=N_CORES,
    scr=SCR,
    show_progress=True,
)
print(f"Stable tautomers after energy filter: {len(stable_tautomers)}")

# %% [markdown]
# ## Step 5 — Generate protonation states

# %%
protonation_config = MokaBlabberConfig()
protonation_states: list[Chem.Mol] = []
for tautomer in stable_tautomers:
    result = run_protonation(tautomer, config=protonation_config, scr=SCR)
    protonation_states += [entry.mol for entry in result.protomers]

charges = [rdmolops.GetFormalCharge(mol) for mol in protonation_states]
unique_charges = np.unique(charges)
print(
    f"Found {len(protonation_states)} protomer(s) "
    f"with charge(s): {', '.join(str(c) for c in unique_charges)}"
)

# %% [markdown]
# ## Step 6 — Enumerate stereocenters

# %%
# Enumerate unassigned stereocenters for both neutral and charged molecules.
# chembridge.enumerate_stereocenters returns None when there are no unassigned
# centers, so we fall back to [mol] in that case.
#
# enumerate_stereocenters strips molecule properties from the returned isomers,
# so restore the protonation abundance (set by run_protonation via
# blabber.COLUMN_ABUNDANCE) onto each enumerated isomer.
def _enumerate(mols: list) -> list:
    result = []
    for mol in mols:
        stereo = chembridge.enumerate_stereocenters(mol)
        if stereo is None:
            result.append(mol)
            continue
        if mol.HasProp(blabber.COLUMN_ABUNDANCE):
            abundance = mol.GetDoubleProp(blabber.COLUMN_ABUNDANCE)
            for isomer in stereo:
                isomer.SetDoubleProp(blabber.COLUMN_ABUNDANCE, abundance)
        result += stereo
    return result


all_neutral = _enumerate(stable_tautomers)
all_charged = _enumerate(protonation_states)

all_graphs = all_charged + all_neutral

# Deduplicate across the combined set
all_graphs = chembridge.unique(all_graphs)
all_charges = [rdmolops.GetFormalCharge(mol) for mol in all_graphs]
all_smis = [
    chembridge.molobj_to_smiles(mol, remove_hs=True, canonical=True, sanitize=False)
    for mol in all_graphs
]
all_props = [chembridge.get_properties_from_molobj(mol) for mol in all_graphs]

print(f"Total unique graphs (after stereo enumeration + dedup): {len(all_graphs)}")

# %% [markdown]
# ## Step 7 — Generate final conformers for each graph

# %%
time_gen_start = time.time()

for i, mol in enumerate(all_graphs):
    print(f"Generating conformers for graph {i + 1}/{len(all_graphs)} ...")
    mol = generate_fast_conformers(
        mol,
        scr=SCR,
        n_cores=N_CORES,
        show_progress=True,
    )
    if mol.GetNumConformers() == 0:
        raise RuntimeError(f"No conformers generated for graph {i}")
    all_graphs[i] = mol

time_gen = time.time()
print(f"Conformer generation done in {time_gen - time_gen_start:.1f}s")

# %% [markdown]
# ## Step 8 — Calculate XTB solvation energies and build results DataFrame

# %%
columns = ["molobj", "smiles", "charge", "protonation_abundance", "graph_idx"]
results_df = pd.DataFrame(columns=columns)

for i, molobj in enumerate(all_graphs):
    smiles_i = all_smis[i]
    props = all_props[i]
    charge = all_charges[i]
    protonation_abundance = (
        props[blabber.COLUMN_ABUNDANCE] if blabber.COLUMN_ABUNDANCE in props else float("nan")
    )

    row_constants = {
        "protonation_abundance": protonation_abundance,
        "smiles": smiles_i,
        "charge": charge,
        "graph_idx": int(i),
    }

    molobjs = chembridge.molobj_to_molobjs(molobj)

    print(f"Solvation energies for graph {i + 1}/{len(all_graphs)} ...")

    rows_xtb = get_xtb_energies_and_weights(
        molobj,
        final_xtb_sp_configs=XTB_SP_SOLVENT_CONFIGS,
        show_progress=True,
        scr=SCR,
    )

    for molobj_prime, row_xtb in zip(molobjs, rows_xtb):
        row = {**row_xtb}
        weights = [row[key] for key in row if "weight" in key]
        row["weights_sum"] = sum(weights)
        row["molobj"] = molobj_prime
        results_df = pd.concat(
            [results_df, pd.DataFrame([{**row, **row_constants}])],
            ignore_index=True,
        )

time_prop = time.time()
print(f"Property calculation done in {time_prop - time_gen:.1f}s")

# %% [markdown]
# ## Results

# %%
print(f"Total conformer rows: {len(results_df)}")
print(f"Unique graphs:        {results_df['graph_idx'].nunique()}")
print(f"Charges present:      {sorted(results_df['charge'].unique())}")
print()
results_df.drop(columns=["molobj"]).head(25)

# %%
# Clean up scratch directory
_tmpdir.cleanup()
