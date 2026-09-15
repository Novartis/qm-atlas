import json
import logging
import time
import traceback
from collections.abc import Iterable
from io import StringIO
from pathlib import Path

import pandas as pd
from rdkit import Chem

from qm_atlas.constants import DEFAULT_SCR
from qm_atlas.software_environment import SOFTWARE_CONFIG
from qm_atlas.tasks import common as calculated_properties
from qm_atlas.wrappers.common import run_command
from qm_atlas.wrappers.helpers import wait_for_license
from qm_atlas.wrappers.jaguar import common

DEFAULT_OPTIONS = {
    "-WAIT": "",
}

SOFTWARE_NAME = "jaguar"
FEATURE_NAME = "JAGUAR_MAIN"
NUM_LICS_PER_CORE = 2

PROPERTY_NAME = "jaguar_hydrogen_abstraction_energy"


_logger = logging.getLogger(__name__)


def split_dict(input_dict: dict, chunk_size: int) -> Iterable[dict]:
    """Yield successive chunks of *input_dict* with at most *chunk_size* items each."""

    current_dict = dict()
    for key, val in input_dict.items():
        current_dict[key] = val
        if len(current_dict) == chunk_size:
            yield current_dict
            current_dict = dict()
    if current_dict:
        yield (current_dict)


def get_hydrogen_abstraction_energies(
    mol: Chem.Mol,
    conf_id: int = 0,
    atom_idcs: list[int] | str = "C",
    geo_opt_settings: dict[str, str] | None = None,
    single_point_settings: dict[str, str] | None = None,
    jaguar_options: dict[str, str] | None = None,
    n_cores: int = 1,
    scr: Path = DEFAULT_SCR,
    keep_files: bool = False,
    license_buffer: int = 40,
    license_timeout: int = 2880,  # wait for up to 2 days
) -> dict[str, calculated_properties.AtomBasedProperty]:
    """Calculates Hydrogen Abstraction Energies using the Jaguar script h_abstraction.py

    Args:
        mol (Chem.Mol):
            The molecule to work on.
        atom_idcs (Union[List[int], str]):
            The indices of the heavy atoms to abstract hydrogen atoms from. Can be a
            list of integers, "C" (all carbons), "N" (all nitrogens), "C,N" (both) or the
            key of a property set of the molecule containing these indices as a
            json-encoded list.
        conf_id (int, optional):
            The conformer ID of the conformer to consider. Defaults to 0.
        geo_opt_settings (Dict[str, str], optional):
            Settings for the geomtry optimization. For example, to change the functional,
            specify {"dftname": "b3lyp"}, to change the basis {"basis": "lacvp*"}, and
            the scf convergence criterion {"econv": "4.0e-05"}.
            Defaults to None, resulting in jaguar defaults (b3lyp, lacvp*, 5.e-5 scf convergence)
        single_point_settings (Dict[str, str], optional):
            Options for the single point energy calculations.
            Defaults to None, resulting in jaguar defaults.
        jaguar_options (Dict[str, str], optional):
            Additional options to run jaguar calculations. Defaults to None.
        n_cores (int, optional):
            The number of cores to use. Defaults to 1.
        scr (Path, optional):
            The scratch directory to use for temporary files. Defaults to DEFAULT_SCR.
        keep_files (bool, optional):
            whether to keep that scratch directory after the calculation has run.
            Defaults to False.
        license_buffer (int, optional):
            A buffer of licenses to leave. Defaults to 40.
        license_timeout (int, optional):
            How long to wait for a license (in minutes). Defaults to 2880 (2 days).

    Raises:
        RuntimeError: If the calculation fails or the output file cannot be found.

    Returns:
        dict[str, calculated_properties.AtomBasedProperty]:
            A dictionary containing the calculated hydrogen abstraction energies for the specified heavy atoms, as an AtomBasedProperty object.
    """
    if isinstance(atom_idcs, str):

        if atom_idcs == "C":
            atom_idcs = [at.GetIdx() for at in mol.GetAtoms() if at.GetSymbol() == "C"]
        elif atom_idcs == "N":
            atom_idcs = [at.GetIdx() for at in mol.GetAtoms() if at.GetSymbol() == "N"]
        elif atom_idcs == "C,N":
            atom_idcs = [at.GetIdx() for at in mol.GetAtoms() if at.GetSymbol() in ["C", "N"]]
        else:
            atom_idcs = json.loads(mol.GetProp(atom_idcs))

    heavy_to_hydrogen_idx = get_idx_map(mol, atom_idcs)

    if not heavy_to_hydrogen_idx:
        _logger.warning("Found no Hydrogens connected to heavy atoms")
        return dict()

    # Jaguar jobs have a tendency to fail when confronted with many sites. We split them
    # decompose dictionary into chunks
    chunk_size = max(1, n_cores - 1)

    # iterate over chunks:
    abstraction_energies = dict()
    for partial_dict in split_dict(heavy_to_hydrogen_idx, chunk_size):
        new_abs_engs = run_jaguar_hydrogen_abstraction(
            mol,
            partial_dict,
            geo_opt_settings=geo_opt_settings,
            single_point_settings=single_point_settings,
            conf_id=conf_id,
            jaguar_options=jaguar_options,
            n_cores=n_cores,
            scr=scr,
            keep_files=keep_files,
            license_buffer=license_buffer,
            license_timeout=license_timeout,
        )
        abstraction_energies.update(new_abs_engs)

    return abstraction_energies


def run_jaguar_hydrogen_abstraction(
    mol: Chem.Mol,
    heavy_to_hydrogen_idx: dict[int, int],
    conf_id: int = 0,
    geo_opt_settings: dict[str, str] | None = None,
    single_point_settings: dict[str, str] | None = None,
    jaguar_options: dict[str, str] | None = None,
    n_cores: int = 1,
    scr: Path = DEFAULT_SCR,
    keep_files: bool = False,
    license_buffer: int = 40,
    license_timeout: int = 2880,  # wait for up to 2 days
) -> dict[str, calculated_properties.AtomBasedProperty]:
    """Run the Jaguar h_abstraction workflow for the given heavy->hydrogen atom pairs."""

    if geo_opt_settings is None:
        geo_opt_settings = {}

    if single_point_settings is None:
        single_point_settings = {}

    if jaguar_options is None:
        jaguar_options = {}

    temp, mae_file = common.prepare_jaguar_input(
        mol, keep_files=keep_files, scr=scr, conf_id=conf_id
    )
    scr = temp.get_path()

    # Set Up Jaguar Command

    h_idcs_str = ",".join([str(idx) for idx in sorted(heavy_to_hydrogen_idx.values())])

    software_env_manager = SOFTWARE_CONFIG.get_environment_manager()
    jaguar_cmd = software_env_manager.get_command("jaguar", "jaguar")
    cmd = f"{jaguar_cmd} run h_abstraction.py -atoms={h_idcs_str} {mae_file.resolve()} "

    options = dict(DEFAULT_OPTIONS)
    options.update(jaguar_options)
    options["-PARALLEL"] = n_cores
    options_str = " ".join([f"{key} {val}" for key, val in options.items()])
    for key, val in geo_opt_settings.items():
        options_str += f"-geok={key}={val} "
    for key, val in single_point_settings.items():
        options_str += f"-spk={key}={val} "
    cmd += options_str

    # check if license is available
    try:
        buffer = license_buffer + n_cores * NUM_LICS_PER_CORE - 1
        wait_for_license(SOFTWARE_NAME, FEATURE_NAME, buffer=buffer, timeout=license_timeout)
    except TimeoutError as exc:
        _logger.error("No Jaguar license available. Exiting BDE calculation")
        raise RuntimeError("No Jaguar license available") from exc
    except Exception as exc:
        _logger.error(f"got exception {exc} while waiting for license")
        _logger.error(traceback.format_exc())
        raise RuntimeError("Error while waiting for Jaguar license") from exc

    # Run Jaguar hydrogen abstraction script
    env = software_env_manager.get_run_environment("jaguar")
    stdout, stderr = run_command(cmd, env=env, cwd=scr)

    _logger.debug(f"Jaguar Output was: {stdout}")
    if stderr:
        _logger.warning(f"Jaguar run reported error: {stderr}")

    # give a bit of buffer for the out file to be ready
    time.sleep(10)

    # collect output file
    out_files = [file for file in scr.iterdir() if file.suffix == ".out"]
    if len(out_files) == 0:
        _logger.error("Jaguar Run did not produce output file")
        common.extract_log_file(scr)
        raise RuntimeError("Jaguar Run did not produce output file")

    if len(out_files) >= 2:
        _logger.error("Jaguar Run produced more than one output file")
        common.extract_log_file(scr)
        raise RuntimeError("Jaguar Run produced more than one output file")

    try:
        abstraction_energies = parse_h_abs_out(out_files[0], heavy_to_hydrogen_idx)
        prop_dict = {PROPERTY_NAME: calculated_properties.AtomBasedProperty(abstraction_energies)}
        return prop_dict

    except RuntimeError as exc:
        raise exc

    except Exception as exc:
        _logger.error(f"Got unexpected Exception {exc} parsing output file")
        _logger.error(traceback.format_exc())
        raise RuntimeError("Error parsing Jaguar output file") from exc


def get_idx_map(
    mol: Chem.Mol,
    atom_idcs: list[int],
) -> dict[int, int]:
    """Map each heavy-atom index in *atom_idcs* to the index of one attached hydrogen."""

    heavy_to_hydrogen_idx = dict()

    for atom_idx in atom_idcs:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == "H":
                # Schrodinger uses 1-based indices
                heavy_to_hydrogen_idx[atom_idx] = neighbor.GetIdx() + 1
                break
        else:  # excuted only if for loop was not terminated by break
            _logger.warning(f"Found no H connected to atom {atom_idx}, skipping it.")

    return heavy_to_hydrogen_idx


def parse_h_abs_out(out_file: Path, heavy_to_hydrogen_idx: dict[int, int]) -> dict[int, float]:
    """Parse a Jaguar h_abstraction output file into a mapping of heavy-atom index to energy."""

    abstraction_energies = dict()

    with open(out_file, "r", encoding="utf-8") as h_out:
        lines = h_out.readlines()

    for idx, line in enumerate(lines):
        if "---------" in line:
            break
    else:
        _logger.error("Error Parsing Schrodinger Output file. Output file was:")
        _logger.error("\n".join(lines))
        raise RuntimeError("Error Parsing Schrodinger Output file")

    start_line = idx + 1
    stop_line = idx + 1 + len(heavy_to_hydrogen_idx)

    h_abs_df = pd.read_csv(
        StringIO("".join(lines[start_line:stop_line])),
        engine="python",
        sep=" ",
        skipinitialspace=True,
        names=["Structure", "Atom", "Energy"],
    )

    hydrogen_to_heavy_idx = {hyd: hvy for hvy, hyd in heavy_to_hydrogen_idx.items()}
    h_idcs = sorted(heavy_to_hydrogen_idx.values())

    # check for problems
    for idx, row in h_abs_df.iterrows():
        hyd_idx = h_idcs[int(idx)]
        if row["Atom"] != f"H{hyd_idx}":
            _logger.error("Error Parsing Schrodinger Output file. Output file was:")
            with open(out_file, "r", encoding="utf-8") as s_out:
                log_str = s_out.read()
            _logger.error(log_str)
            raise RuntimeError("Error Parsing Schrodinger Output file")
        hvy_idx = hydrogen_to_heavy_idx[hyd_idx]
        abstraction_energies[hvy_idx] = row["Energy"]

    return abstraction_energies
