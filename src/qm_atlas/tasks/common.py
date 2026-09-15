"""Handle writing of calculated molecular properties."""

import json
from typing import Any, Protocol

import numpy as np
from rdkit import Chem


class Property(Protocol):
    """Abstract base class for a calculated molecular property.

    Subclasses must implement:
    - get_property_value: Return the calculated property value
    - set_property_on_mol: Set the calculated property value on a molecule
    - from_mol: Read the calculated property value back from a molecule
    """

    def get_property_value(self) -> Any:
        """Get the calculated property value."""
        raise NotImplementedError

    def set_property_on_mol(self, mol: Chem.Mol, property_name: str):
        """Set the calculated property value on a molecule.
            The molecule is modified in place.

        Args:
            mol: RDKit molecule to set the property on
            property_name: Name to use for the property on the molecule
        Raises:
            NotImplementedError: If the method is not implemented in a subclass
        """
        raise NotImplementedError

    @classmethod
    def from_mol(cls, mol: Chem.Mol, property_name: str) -> "Property":
        """Read the calculated property value back from a molecule.

        Args:
            mol: RDKit molecule to read the property from
            property_name: Name of the property on the molecule

        Returns:
            Property: A property instance holding the value read from the molecule.

        Raises:
            KeyError: If the property is not present on the molecule.
            NotImplementedError: If the method is not implemented in a subclass
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(value={self.get_property_value()})"


class ScalarProperty(Property):
    """Class for scalar calculated molecular properties."""

    def __init__(self, calculated_property: float):
        self.calculated_property = calculated_property

    def get_property_value(self) -> float:
        """Get the scalar calculated property value."""
        return self.calculated_property

    def set_property_on_mol(self, mol: Chem.Mol, property_name: str):
        """Set the scalar calculated property value on a molecule as a property.

        Args:
            mol: RDKit molecule to set the property on
            property_name: Name to use for the property on the molecule
        """
        try:
            float_value = float(self.calculated_property)
            mol.SetDoubleProp(property_name, float_value)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Cannot convert calculated property to float: {self.calculated_property}"
            ) from e

    @classmethod
    def from_mol(cls, mol: Chem.Mol, property_name: str) -> "ScalarProperty":
        """Read a scalar property value from a molecule.

        Args:
            mol: RDKit molecule to read the property from
            property_name: Name of the property on the molecule

        Returns:
            ScalarProperty: The scalar value stored on the molecule.

        Raises:
            KeyError: If the property is not present on the molecule.
        """
        if not mol.HasProp(property_name):
            raise KeyError(property_name)
        return cls(mol.GetDoubleProp(property_name))


class TensorProperty(Property):
    """Class for tensor-valued calculated molecular properties.
    These could be vectors, matrices, or higher-dimensional tensors."""

    def __init__(self, calculated_property: np.ndarray | list):
        self.calculated_property = np.array(calculated_property)

    def get_property_value(self) -> np.ndarray:
        """Get the tensor-valued calculated property value."""
        return self.calculated_property

    def set_property_on_mol(self, mol: Chem.Mol, property_name: str):
        """Set the tensor calculated property value on a molecule as a property.

        Args:
            mol: RDKit molecule to set the property on
            property_name: Name to use for the property on the molecule
        """
        mol.SetProp(property_name, json.dumps(self.calculated_property.tolist()))

    @classmethod
    def from_mol(cls, mol: Chem.Mol, property_name: str) -> "TensorProperty":
        """Read a tensor property value from a molecule.

        The value is expected to have been stored as a JSON-serialized list.

        Args:
            mol: RDKit molecule to read the property from
            property_name: Name of the property on the molecule

        Returns:
            TensorProperty: The tensor value stored on the molecule.

        Raises:
            KeyError: If the property is not present on the molecule.
        """
        if not mol.HasProp(property_name):
            raise KeyError(property_name)
        return cls(np.array(json.loads(mol.GetProp(property_name))))


class AtomBasedProperty(Property):
    """Class for atom-based calculated molecular properties."""

    def __init__(self, calculated_property: list[float] | dict[int, float]):
        self.calculated_property = calculated_property

    def get_property_value(self) -> list[float] | dict[int, float]:
        """Get the atom-based calculated property value."""
        return self.calculated_property

    def set_property_on_mol(self, mol: Chem.Mol, property_name: str):
        """Set the atom-based calculated property value on a molecule as a property.

        Args:
            mol: RDKit molecule to set the property on
            property_name: Name to use for the property on the molecule
        """
        iterator = (
            self.calculated_property.items()
            if isinstance(self.calculated_property, dict)
            else enumerate(self.calculated_property)
        )
        for atom_idx, value in iterator:
            if not isinstance(atom_idx, int):
                raise ValueError("Atom indices must be integers.")
            if atom_idx < 0 or atom_idx >= mol.GetNumAtoms():
                raise ValueError(
                    f"Atom index {atom_idx} is out of bounds for molecule with {mol.GetNumAtoms()} atoms."
                )
            atom = mol.GetAtomWithIdx(atom_idx)
            atom.SetDoubleProp(property_name, float(value))
        Chem.CreateAtomDoublePropertyList(mol, property_name)

    @classmethod
    def from_mol(cls, mol: Chem.Mol, property_name: str) -> "AtomBasedProperty":
        """Read an atom-based property value from a molecule.

        Every atom must carry the property; the values are returned as a list
        ordered by atom index.

        Args:
            mol: RDKit molecule to read the property from
            property_name: Name of the property on the atoms

        Returns:
            AtomBasedProperty: The per-atom values ordered by atom index.

        Raises:
            KeyError: If any atom is missing the property.
        """
        values: list[float] = []
        for atom in mol.GetAtoms():
            if not atom.HasProp(property_name):
                raise KeyError(property_name)
            values.append(atom.GetDoubleProp(property_name))
        return cls(values)


class BondBasedProperty(Property):
    """Class for bond-based calculated molecular properties."""

    def __init__(self, calculated_property: dict[tuple[int, int], float]):
        self.calculated_property = calculated_property

    def get_property_value(self) -> dict[tuple[int, int], float]:
        """Get the bond-based calculated property value."""
        return self.calculated_property

    def set_property_on_mol(self, mol: Chem.Mol, property_name: str):
        """Set the bond-based calculated property value on a molecule as a property.

        Args:
            mol: RDKit molecule to set the property on
            property_name: Name to use for the property on the molecule
        """
        write_dict = {}
        for (atom_idx1, atom_idx2), value in self.calculated_property.items():
            new_key = f"{min(atom_idx1, atom_idx2)}-{max(atom_idx1, atom_idx2)}"
            write_dict[new_key] = value
        mol.SetProp(property_name, json.dumps(write_dict))

    @classmethod
    def from_mol(cls, mol: Chem.Mol, property_name: str) -> "BondBasedProperty":
        """Read a bond-based property value from a molecule.

        Args:
            mol: RDKit molecule to read the property from
            property_name: Name of the property on the molecule

        Returns:
            BondBasedProperty: The per-bond values keyed by atom-index pairs.

        Raises:
            KeyError: If the property is not present on the molecule.
        """
        if not mol.HasProp(property_name):
            raise KeyError(property_name)
        raw: dict[str, float] = json.loads(mol.GetProp(property_name))
        result: dict[tuple[int, int], float] = {}
        for key, value in raw.items():
            atom_idx1_str, atom_idx2_str = key.split("-")
            result[(int(atom_idx1_str), int(atom_idx2_str))] = value
        return cls(result)


class StringProperty(Property):
    """Class for string-valued calculated molecular properties."""

    def __init__(self, calculated_property: str):
        self.calculated_property = calculated_property

    def get_property_value(self) -> str:
        """Get the string-valued calculated property value."""
        return self.calculated_property

    def set_property_on_mol(self, mol: Chem.Mol, property_name: str):
        """Set the string-valued calculated property value on a molecule as a property.

        Args:
            mol: RDKit molecule to set the property on
            property_name: Name to use for the property on the molecule
        """
        mol.SetProp(property_name, self.calculated_property)

    @classmethod
    def from_mol(cls, mol: Chem.Mol, property_name: str) -> "StringProperty":
        """Read a string property value from a molecule.

        Args:
            mol: RDKit molecule to read the property from
            property_name: Name of the property on the molecule

        Returns:
            StringProperty: The string value stored on the molecule.

        Raises:
            KeyError: If the property is not present on the molecule.
        """
        if not mol.HasProp(property_name):
            raise KeyError(property_name)
        return cls(mol.GetProp(property_name))


class BoolProperty(Property):
    """Class for boolean calculated molecular properties."""

    def __init__(self, calculated_property: bool):
        self.calculated_property = calculated_property

    def get_property_value(self) -> bool:
        """Get the boolean calculated property value."""
        return self.calculated_property

    def set_property_on_mol(self, mol: Chem.Mol, property_name: str):
        """Set the boolean calculated property value on a molecule as a property.

        Args:
            mol: RDKit molecule to set the property on
            property_name: Name to use for the property on the molecule
        """
        mol.SetBoolProp(property_name, self.calculated_property)

    @classmethod
    def from_mol(cls, mol: Chem.Mol, property_name: str) -> "BoolProperty":
        """Read a boolean property value from a molecule.

        Args:
            mol: RDKit molecule to read the property from
            property_name: Name of the property on the molecule

        Returns:
            BoolProperty: The boolean value stored on the molecule.

        Raises:
            KeyError: If the property is not present on the molecule.
        """
        if not mol.HasProp(property_name):
            raise KeyError(property_name)
        return cls(mol.GetBoolProp(property_name))
