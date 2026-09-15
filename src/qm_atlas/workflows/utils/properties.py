from rdkit import Chem


def read_properties(mol: Chem.Mol) -> dict[str, str]:
    """Read properties from a molecule and return them as a dictionary."""
    properties = mol.GetPropsAsDict()
    if mol.HasProp("_Name"):
        properties["_Name"] = mol.GetProp("_Name")
    return properties


def restore_properties(mol: Chem.Mol, properties: dict[str, str]) -> None:
    """Restore properties to a molecule from a dictionary."""
    for key, value in properties.items():
        if isinstance(value, str):
            mol.SetProp(key, value)
        elif isinstance(value, (int, float)):
            mol.SetDoubleProp(key, float(value))
        else:
            mol.SetProp(key, str(value))
