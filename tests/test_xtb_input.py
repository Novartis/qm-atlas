"""Tests for how optimization options are turned into an xtb xcontrol file."""

import pytest
from jinja2 import Template
from rdkit import Chem
from rdkit.Chem import AllChem

from qm_atlas.tasks import optimize
from qm_atlas.wrappers import xtb
from qm_atlas.wrappers.turbomole import xtb_opt

# The rendering of the global-switch constraint block, kept verbatim so that a change
# to the template has to be a deliberate one.
GLOBAL_CONSTRAIN_BLOCK = """
$constrain
  all torsions=true
  force constant=0.5 """

RENDER_KWARGS = {
    "max_num_steps": 100,
    "opt_level": "normal",
    "torsions_fixed": True,
    "heavy_atom_torsions_fixed": False,
    "angles_fixed": False,
    "bonds_fixed": False,
    "force_constant": 0.5,
}
OPT_TEMPLATES = [xtb.XCONTROL_TEMPLATE_OPT, xtb_opt.XCONTROL_TEMPLATE_CSTR]


def build_mol(smiles: str = "CC(=O)Nc1ccc(OCC(=O)N2CCOCC2)cc1") -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def render(template_str: str, mol: Chem.Mol, **kwargs) -> str:
    context = {"charge": 0, "num_unpaired_electrons": 0, **RENDER_KWARGS, **kwargs}
    context["dihedral_constraints"] = xtb.resolve_dihedral_constraints(
        mol,
        constrain=context.get("constrain", False),
        heavy_atom_torsions_fixed=context.get("heavy_atom_torsions_fixed", False),
    )
    return Template(template_str).render(**context)


# ---------------------------------------------------------------------------
# resolve_dihedral_constraints
# ---------------------------------------------------------------------------


def test_no_dihedral_constraints_without_constrain():
    assert xtb.resolve_dihedral_constraints(None, constrain=False) == []


def test_no_dihedral_constraints_without_the_heavy_atom_flag():
    assert (
        xtb.resolve_dihedral_constraints(
            build_mol(), constrain=True, heavy_atom_torsions_fixed=False
        )
        == []
    )


def test_heavy_atom_flag_warns_when_nothing_can_be_restrained(caplog):
    result = xtb.resolve_dihedral_constraints(
        build_mol("C"), constrain=True, heavy_atom_torsions_fixed=True
    )
    assert result == []
    assert "no heavy-atom torsion" in caplog.text


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("template_str", OPT_TEMPLATES)
def test_unconstrained_render_has_no_constraint_block(template_str):
    assert "$constrain" not in render(template_str, build_mol(), constrain=False)


@pytest.mark.parametrize("template_str", OPT_TEMPLATES)
def test_all_torsions_render_uses_global_switches_without_elements(template_str):
    rendered = render(template_str, build_mol(), constrain=True)
    assert rendered.endswith(GLOBAL_CONSTRAIN_BLOCK)
    assert "elements:" not in rendered


@pytest.mark.parametrize("template_str", OPT_TEMPLATES)
def test_heavy_atom_render_writes_explicit_dihedrals_only(template_str):
    mol = build_mol()
    dihedral_constraints = xtb.resolve_dihedral_constraints(
        mol, constrain=True, heavy_atom_torsions_fixed=True
    )
    # Atom indices handed to xtb must be 1-based.
    assert len(dihedral_constraints) > 0
    assert all(
        1 <= index <= mol.GetNumAtoms() for torsion in dihedral_constraints for index in torsion
    )

    rendered = render(
        template_str,
        mol,
        constrain=True,
        torsions_fixed=False,
        heavy_atom_torsions_fixed=True,
    )
    assert "elements:" not in rendered
    assert "all torsions" not in rendered
    dihedral_lines = [line for line in rendered.splitlines() if "dihedral:" in line]
    assert len(dihedral_lines) == len(dihedral_constraints)
    for line in dihedral_lines:
        indices, reference = line.strip().removeprefix("dihedral: ").rsplit(",", maxsplit=1)
        assert reference == "auto"
        assert len(indices.split(",")) == 4
    assert "force constant=0.5" in rendered


def test_write_xcontrol_file_writes_heavy_atom_dihedrals(tmp_path):
    class _Temp:
        def get_path(self):
            return tmp_path

    path = xtb.write_xcontrol_file(
        build_mol(),
        _Temp(),
        xtb.XCONTROL_TEMPLATE_OPT,
        conf_id=0,
        max_num_steps=100,
        opt_level="normal",
        constrain=True,
        torsions_fixed=False,
        heavy_atom_torsions_fixed=True,
        angles_fixed=False,
        bonds_fixed=False,
        force_constant=0.5,
    )
    content = path.read_text(encoding="utf-8")
    assert "dihedral:" in content
    assert "elements:" not in content
    assert "all torsions" not in content


# ---------------------------------------------------------------------------
# Options plumbing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("options_class", [optimize.XtbOptions, optimize.XtbTurbomoleOptions])
def test_options_default_to_fixing_heavy_atom_torsions(options_class):
    options = options_class()
    assert options.heavy_atom_torsions_fixed is True
    assert options.torsions_fixed is False
    assert options.angles_fixed is False
    assert options.bonds_fixed is False


@pytest.mark.parametrize("options_class", [optimize.XtbOptions, optimize.XtbTurbomoleOptions])
def test_heavy_atom_flag_reaches_the_optimizer_kwargs(options_class):
    options = options_class(constrain=True, torsions_fixed=False, heavy_atom_torsions_fixed=True)
    dumped = options.model_dump()
    assert dumped["heavy_atom_torsions_fixed"] is True
    assert dumped["torsions_fixed"] is False


# ---------------------------------------------------------------------------
# GFN Hamiltonian selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("template_str", [xtb.XCONTROL_TEMPLATE_OPT, xtb.XCONTROL_TEMPLATE_SP])
@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"method": "0"}, "method=0"),
        ({"method": "1"}, "method=1"),
        ({}, "method=2"),  # GFN2 is the default
        ({"gfn_method": 1}, "method=1"),  # legacy variable is still understood
        ({"method": "0", "gfn_method": 1}, "method=0"),  # method wins over the legacy variable
    ],
)
def test_gfn_hamiltonian_selection(template_str, kwargs, expected):
    rendered = Template(template_str).render(charge=0, num_unpaired_electrons=0, **kwargs)
    assert expected in rendered


def test_xtb_options_method_reaches_the_xcontrol_file(tmp_path):
    class _Temp:
        def get_path(self):
            return tmp_path

    path = xtb.write_xcontrol_file(
        build_mol("CCO"),
        _Temp(),
        xtb.XCONTROL_TEMPLATE_OPT,
        conf_id=0,
        **optimize.XtbOptions(method="1").model_dump(exclude={"backend"}),
    )
    assert "method=1" in path.read_text(encoding="utf-8")
