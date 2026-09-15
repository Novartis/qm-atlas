# %%
import logging
import sys

from matplotlib import pyplot as plt
from rdkit import Chem

from qm_atlas.tasks.utils import conformer_geometry

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# %% [markdown]
# # Analysis of duplicate conformers using RMSD
#
# We study different methods to calculate the RMSD between conformations of the same molecule. The RMSD results are used to find duplicate conformations based on an RMSD threshold. One challenge here is that we aim to include the hydrogen atoms in these calculations, which can lead to combinatorial explosions if one considers all possible ways that the atoms in two conformations can be matched.
#
# As an example molecule, we consider retinoic acid, which is flexible and has a few methyl groups to show the speed-up that can be achieved
#
# Below, we compare four methods for calculating RMSD values:
# * "exact": This relies on rdkit's GetBestRMS function which enumerates all possible ways of matching the atoms in the two conformations. Since our example compound is not too large, this is feasible, but it does not scale to larger compounds with many methyl groups. In cases where the number of matches can still be enumerated, this provides the true RMSD value, and we benchmark the other methods against it.
# * "fast": An approach which relies on rdkit's GetConformerRMSMatrix, ignoring the different ways of matching atoms.
# * "tradeoff": A custom method that aims to integrate hydrogen atoms while avoiding the combinatorial explosion. It first compares the conformations without hydrogens, and figures out the best way to match the heavy atoms. The hydrogens are then added, with a match of the heavy atoms already known. This may overestimate RMSD values, but for conformations which are close (the relevant case for deduplication), the heavy atom match established in this way is very likely correct
# * "strip-hydrogens": To have another comparison, we also consider using GetBestRMS on the conformations without hydrogens. This is a fast and simple option, which gives reasonable results. It is the only method that can underestimate the true RMSD.
#

# %%
retinoic_acid_smi = "CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C(=O)O)C(C)(C)CCC1"  # CHEMBL38
retinoic_acid = Chem.MolFromSmiles(retinoic_acid_smi)

retinoic_acid

# %%
from rdkit.Chem import rdDistGeom

mol_3d = Chem.AddHs(retinoic_acid)
mol_3d.RemoveAllConformers()

# set up rdkit conformer generation
etkdg = rdDistGeom.ETKDGv3()
etkdg.verbose = False
etkdg.randomSeed = 1

rdDistGeom.EmbedMultipleConfs(mol_3d, numConfs=50, params=etkdg)
mol_3d.GetNumConformers()

# %% [markdown]
# ## Run RMSD calculations

# %%
# %%time

# find duplicates using the exact method. For this example (50 conformers and a rather small molecule),

rmsd_matrix_exact, idx_map_exact = conformer_geometry.get_rmsd_matrix(
    mol_3d, num_cores=8, method="exact"
)
duplicate_idcs_exact = conformer_geometry.get_duplicate_idcs(
    mol_3d, 0.5, num_cores=8, method="exact"
)

print("Timing for RMSD exact method (8 cores)")

# %%
# %%time

# find duplicates using the tradeoff method

rmsd_matrix_tradeoff, idx_map_tradeoff = conformer_geometry.get_rmsd_matrix(
    mol_3d, num_cores=8, method="tradeoff"
)
duplicate_idcs_tradeoff = conformer_geometry.get_duplicate_idcs(
    mol_3d, 0.5, num_cores=8, method="tradeoff"
)

print("Timing for RMSD tradeoff method (8 cores)")

# %%
# %%time

# find duplicates using the fast method, no multiprocessing implemented

rmsd_matrix_fast, idx_map_fast = conformer_geometry.get_rmsd_matrix(
    mol_3d, num_cores=1, method="fast"
)
duplicate_idcs_fast = conformer_geometry.get_duplicate_idcs(
    mol_3d, 0.5, num_cores=1, method="fast"
)

print("Timing for RMSD fast method (1 core)")

# %%
# %%time

# find duplicates using the strip-hydrogens method

rmsd_matrix_strip, idx_map_strip = conformer_geometry.get_rmsd_matrix(
    mol_3d, num_cores=8, method="strip-hydrogens"
)
duplicate_idcs_strip = conformer_geometry.get_duplicate_idcs(
    mol_3d, 0.5, num_cores=8, method="strip-hydrogens"
)

print("Timing for RMSD strip-hydrogens method (8 cores)")

# %% [markdown]
# ## Analyze Results

# %%
print(f"Number of duplicates found using exact method: {len(duplicate_idcs_exact)}")
print(f"Number of duplicates found using tradeoff method: {len(duplicate_idcs_tradeoff)}")
print(f"Number of duplicates found using fast method: {len(duplicate_idcs_fast)}")
print(f"Number of duplicates found using strip-hydrogens method: {len(duplicate_idcs_strip)}")

# %% [markdown]
# We observe that the tradeoff method finds as many duplicates as the exact approach, ignoring the different ways of matching leads to missing all duplicates, and the strip-hydrogens method finds additional duplicates, due to underestimating the true RMSD
#
# The tradeoff methods actually finds the same duplicates as the exact method:

# %%
# We need to align the RMSD matrices to the same conformer pair ordering
# Each method may use different index mappings, so we must reorganize them

# Get the conformer pair indices for each method using their index maps
exact_pairs = [idx_map_exact(i) for i in range(len(rmsd_matrix_exact))]
tradeoff_pairs = [idx_map_tradeoff(i) for i in range(len(rmsd_matrix_tradeoff))]
fast_pairs = [idx_map_fast(i) for i in range(len(rmsd_matrix_fast))]
strip_pairs = [idx_map_strip(i) for i in range(len(rmsd_matrix_strip))]

# Use exact_pairs as reference ordering
rmsd_matrix_exact_aligned = rmsd_matrix_exact.copy()

# Create aligned RMSD matrices
rmsd_matrix_tradeoff_aligned = [0.0] * len(rmsd_matrix_exact)
for i, pair in enumerate(exact_pairs):
    j = tradeoff_pairs.index(pair)
    rmsd_matrix_tradeoff_aligned[i] = rmsd_matrix_tradeoff[j]

rmsd_matrix_fast_aligned = [0.0] * len(rmsd_matrix_exact)
for i, pair in enumerate(exact_pairs):
    j = fast_pairs.index(pair)
    rmsd_matrix_fast_aligned[i] = rmsd_matrix_fast[j]

rmsd_matrix_strip_aligned = [0.0] * len(rmsd_matrix_exact)
for i, pair in enumerate(exact_pairs):
    j = strip_pairs.index(pair)
    rmsd_matrix_strip_aligned[i] = rmsd_matrix_strip[j]

print("RMSD matrices aligned to exact method's pair ordering")

# %%
# Plot 1: Tradeoff vs Exact
plt.figure(figsize=(6, 5))
plt.scatter(rmsd_matrix_exact_aligned, rmsd_matrix_tradeoff_aligned, alpha=0.6)
max_val = max(max(rmsd_matrix_exact_aligned), max(rmsd_matrix_tradeoff_aligned))
plt.plot([0, max_val], [0, max_val], "r--", label="Perfect agreement")
plt.xlabel("Exact RMSD")
plt.ylabel("Tradeoff RMSD")
plt.title("Tradeoff vs Exact")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# We see a near-perfect agreement here. Considering more conformations and different structures, one can also find cases where the tradeoff method overestimates the true RMSD. These are however limited to pairs of conformations which differ quite strongly, and will hence not be identified as duplicates

# %%
# Plot 2: Fast vs Exact
plt.figure(figsize=(6, 5))
plt.scatter(rmsd_matrix_exact_aligned, rmsd_matrix_fast_aligned, alpha=0.6, color="orange")
max_val = max(max(rmsd_matrix_exact_aligned), max(rmsd_matrix_fast_aligned))
plt.plot([0, max_val], [0, max_val], "r--", label="Perfect agreement")
plt.xlabel("Exact RMSD")
plt.ylabel("Fast RMSD")
plt.title("Fast vs Exact")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# As expected, ignoring the different possibilities to match atoms leads to a significant overestimation of the true RMSD values, obscuring all duplicates

# %%
# Plot 3: Strip-hydrogens vs Exact
plt.figure(figsize=(6, 5))
plt.scatter(rmsd_matrix_exact_aligned, rmsd_matrix_strip_aligned, alpha=0.6, color="green")
max_val = max(max(rmsd_matrix_exact_aligned), max(rmsd_matrix_strip_aligned))
plt.plot([0, max_val], [0, max_val], "r--", label="Perfect agreement")
plt.xlabel("Exact RMSD")
plt.ylabel("Strip-hydrogens RMSD")
plt.title("Strip-hydrogens vs Exact")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# Stripping Hydrogens for the RMSD calculation gives lower RMSD values on average. Thresholds should be adapted correspondingly if relying on this method.
