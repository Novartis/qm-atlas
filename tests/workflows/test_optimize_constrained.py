from conftest import RESOURCES  # pylint: disable=import-error
from context import require_software  # pylint: disable=import-error
from rdkit import Chem

from qm_atlas.tasks import optimize
from qm_atlas.workflows import optimize_constrained

TEST_DIR = RESOURCES / "cpd_dir_example" / "glycine"
TEST_SDF = TEST_DIR / "results" / "glycine_c0.sdf"


@require_software("xtb", "turbomole")
def test_optimization():

    test_mol = Chem.MolFromMolFile(str(TEST_SDF.resolve()), removeHs=False)

    optimization_config = [
        [
            optimize.XtbTurbomoleOptions(
                max_num_steps=1,
                opt_level="normal",
                basis="def2-TZVP",
                functional=["b-p"],
                constrain=True,
                force_constant=0.1,
                torsions_fixed=True,
                angles_fixed=False,
                bonds_fixed=False,
            ),
        ],
    ]

    optimized_mols = optimize_constrained.run_constrained_optimization(
        test_mol, optimization_config=optimization_config
    )

    assert len(optimized_mols) == 1

    mol_opt = optimized_mols[0]

    assert mol_opt.GetNumConformers() == 1
