import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

from ppqm.utils.files import WorkDir
from rdkit import Chem

from qm_atlas import constants
from qm_atlas.software_environment import SOFTWARE_CONFIG, USE_DEFAULT_MODEL, get_moka_model
from qm_atlas.wrappers.common import run_command

_logger = logging.getLogger(__name__)

MOKA_CMD = "moka_cli"


class PkaTypes:
    ACIDIC = "a"
    BASIC = "b"


@dataclass
class PredictedPKA:
    value: float
    pka_type: str
    atom_index: int | None
    standard_deviation: float | None = None

    def __lt__(self, other: "PredictedPKA"):
        return self.value < other.value

    def get_moka_string(self) -> str:
        atom_str = f"{self.atom_index + 1}" if self.atom_index is not None else "0"
        sd_str = f"{self.standard_deviation:.2f}" if self.standard_deviation is not None else "0.0"
        return f"{self.pka_type} {self.value:.2f} {atom_str} {sd_str}"

    def get_moka_std(self) -> float:
        if self.standard_deviation is not None:
            return self.standard_deviation
        return 1.0


@dataclass
class MokaPrediction:
    name: str
    covalent_hydration: str
    unstable_tautomer: bool | None
    num_centers: int
    predicted_pkas: list[PredictedPKA]

    def get_moka_line(self) -> str:
        if self.unstable_tautomer is None:
            unstable_tautomer_str = "-"
        else:
            unstable_tautomer_str = "1" if self.unstable_tautomer else "0"
        header = (
            f"{self.name} {self.covalent_hydration} {unstable_tautomer_str} {self.num_centers}"
        )
        pkas = " ".join(pka.get_moka_string() for pka in self.predicted_pkas)
        return f"{header} {pkas}"


def _find_header_end(line_lst: list[str]) -> int:
    """Return the index of the first header field (CH) after the molecule name.

    The molecule name may contain spaces, so the fixed ``CH UT IC`` header
    cannot be located by absolute position. Instead, anchor on the ionizable
    centers: each is introduced by an ``a``/``b`` marker, the ``IC`` count sits
    just before the first marker, and ``CH UT`` precede it. Candidate markers
    are validated against the ``IC`` count so a name containing a bare ``a``/``b``
    token is not mistaken for a center.
    """
    marker_indices = [
        idx for idx, el in enumerate(line_lst) if el in (PkaTypes.ACIDIC, PkaTypes.BASIC)
    ]

    for pos, marker_idx in enumerate(marker_indices):
        # CH UT IC must precede the first real marker.
        if marker_idx < 3:
            continue
        num_remaining = len(marker_indices) - pos
        try:
            declared_centers = int(line_lst[marker_idx - 1])
        except ValueError:
            continue
        if declared_centers != num_remaining:
            continue
        if line_lst[marker_idx - 2] not in ("0", "1", "-"):  # UT
            continue
        if line_lst[marker_idx - 3] not in ("0", "1"):  # CH
            continue
        return marker_idx - 3

    # No ionizable centers: the line is just "NAME CH UT 0".
    return len(line_lst) - 3


def parse_moka_output(line: str) -> MokaPrediction:
    """
    Parses a line of MoKa output and returns a MokaPrediction object.

    Format is: NAME CH UT IC ( a|b PK ATOM SD [QP] ) * IC
        NAME - the molecule name (may contain spaces)
        CH - covalent hydration flag: 0 (false) or 1 (true)
        UT - unstable tautomer flag: 0 (false), 1 (true) or - (not computed)
        IC - number of ionizable centers

    Then, for each ionizable center:
        a|b - type of ionizable center (acid or basic)
        PK - predicted pKa value
        ATOM - atom number
        SD - standard deviation of prediction
        QP - quality parameter (optional, older versions)

    The molecule name may contain spaces. Recent MoKa versions wrap it in double
    quotes; older versions do not, in which case the header is located by anchoring
    on the ionizable-center markers.
    """
    line = line.strip()

    if line.startswith('"'):
        end_quote = line.index('"', 1)
        name = line[1:end_quote]
        header_lst = line[end_quote + 1 :].split()
    else:
        line_lst = line.split()
        header_start = _find_header_end(line_lst)
        if header_start < 1:
            raise RuntimeError(f"Could not parse MoKa output line: {line!r}")
        name = " ".join(line_lst[:header_start])
        header_lst = line_lst[header_start:]

    if len(header_lst) < 3:
        raise RuntimeError(f"Could not parse MoKa output line: {line!r}")

    covalent_hydration = header_lst[0]
    unstable_tautomer_str = header_lst[1]
    unstable_tautomer = None if unstable_tautomer_str == "-" else unstable_tautomer_str == "1"
    num_centers = int(header_lst[2])

    # Find ionizable centers by locating "a" or "b" markers after the header
    center_indices = [
        idx
        for idx in range(3, len(header_lst))
        if header_lst[idx] in (PkaTypes.ACIDIC, PkaTypes.BASIC)
    ]

    predicted_pkas = []
    for idx in center_indices:
        center_type = header_lst[idx]
        pk = float(header_lst[idx + 1])
        atom_idx = int(header_lst[idx + 2]) - 1
        sd = float(header_lst[idx + 3])
        predicted_pkas.append(
            PredictedPKA(
                value=pk,
                pka_type=center_type,
                atom_index=atom_idx,
                standard_deviation=sd,
            )
        )

    return MokaPrediction(
        name=name,
        covalent_hydration=covalent_hydration,
        unstable_tautomer=unstable_tautomer,
        num_centers=num_centers,
        predicted_pkas=predicted_pkas,
    )


def get_moka_version(cmd: str | None = None) -> str:
    """Get the version string of MoKa."""
    software_env_manager = SOFTWARE_CONFIG.get_environment_manager()
    if cmd is None:
        cmd = software_env_manager.get_command("moka", MOKA_CMD)
    env = software_env_manager.get_run_environment("moka")
    try:
        _, stderr = run_command(f"{cmd} --version", env=env)
        for line in stderr.splitlines():
            if "MoKa" in line:
                return line.replace(" ", "_")
        raise RuntimeError("MoKa version not found in the output.")
    except RuntimeError as e:
        raise RuntimeError(f"Failed to get Moka version: {e}") from e


def moka_version_is_v5(cmd: str | None = None) -> bool:
    """Return True if the configured MoKa executable is version 5.x."""
    version = get_moka_version(cmd=cmd)
    return version.startswith("MoKa_5.")


def run_moka_sdf(
    sdf_file: Path,
    cmd: str | None = None,
    model: Path | None = USE_DEFAULT_MODEL,
    lower_ph_threshold: float | None = None,
    upper_ph_threshold: float | None = None,
) -> dict[str, MokaPrediction]:
    """Run MoKa on an SDF file and return the results."""

    # Resolve model if sentinel value is used
    if model is USE_DEFAULT_MODEL:
        model = get_moka_model()

    software_env_manager = SOFTWARE_CONFIG.get_environment_manager()
    if cmd is None:
        cmd = software_env_manager.get_command("moka", MOKA_CMD)

    options = [cmd, str(sdf_file.resolve())]

    if model is not None:
        if moka_version_is_v5(cmd=cmd):
            options.append(f"--load-pka-model={model.resolve()}")
        else:
            options.append(f"--load-model={model.resolve()}")

    if lower_ph_threshold is not None:
        options.append(f"--basic-lo-limit {lower_ph_threshold}")
    if upper_ph_threshold is not None:
        options.append(f"--acid-hi-limit {upper_ph_threshold}")

    command = " ".join(options)
    _logger.debug(command)

    env = software_env_manager.get_run_environment("moka")
    stdout, stderr = run_command(command, env=env)

    if stderr:
        _logger.warning(f"MoKa prediction noted error: {stderr}")

    if not stdout or not stdout.strip():
        _logger.error("MoKa prediction failed: no output")
        raise RuntimeError("MoKa prediction failed: no output")

    results = {}
    for line in stdout.strip().split("\n"):
        moka_pred = parse_moka_output(line)
        results[moka_pred.name] = moka_pred

    return results


def get_moka_prediction_on_batch(
    molecules: list[Chem.Mol],
    cmd: str | None = None,
    model: Path | None = USE_DEFAULT_MODEL,
    scr: Path = constants.DEFAULT_SCR,
    lower_ph_threshold: float | None = None,
    upper_ph_threshold: float | None = None,
    chunk_size: int = 100,
    keep_files: bool = False,
) -> dict[str, MokaPrediction]:
    """
    Run MoKa pKa prediction on a batch of molecules.

    Writes the molecules to a temporary SDF file, runs MoKa, and returns the
    results as a dict mapping molecule name to MokaPrediction.
    """
    if not scr.is_dir():
        scr.mkdir(parents=True, exist_ok=True)

    if len(molecules) > chunk_size:
        results = {}
        for i in range(0, len(molecules), chunk_size):
            chunk = molecules[i : i + chunk_size]
            chunk_results = get_moka_prediction_on_batch(
                chunk,
                cmd=cmd,
                model=model,
                scr=scr,
                lower_ph_threshold=lower_ph_threshold,
                upper_ph_threshold=upper_ph_threshold,
                chunk_size=chunk_size,
                keep_files=keep_files,
            )
            results.update(chunk_results)
        return results

    unique_id = uuid.uuid4().hex
    temp = WorkDir(dir=scr, prefix="moka_", keep=keep_files)
    sdf_file = temp.get_path() / f"temp_moka_{unique_id}.sdf"
    with Chem.SDWriter(str(sdf_file)) as writer:
        for mol in molecules:
            writer.write(mol)
    return run_moka_sdf(
        sdf_file,
        cmd=cmd,
        model=model,
        lower_ph_threshold=lower_ph_threshold,
        upper_ph_threshold=upper_ph_threshold,
    )


def get_moka_prediction(
    mol: Chem.Mol,
    cmd: str | None = None,
    model: Path | None = USE_DEFAULT_MODEL,
    scr: Path = constants.DEFAULT_SCR,
    lower_ph_threshold: float | None = None,
    upper_ph_threshold: float | None = None,
    keep_files: bool = False,
) -> MokaPrediction:
    """
    Run MoKa pKa prediction on a single molecule.

    Args:
        mol: RDKit molecule to predict pKa values for.
        cmd: MoKa command to use. Defaults to the command from SOFTWARE_CONFIG.
        model: Path to a custom MoKa model. Defaults to the model from SOFTWARE_CONFIG.
               Pass None to use MoKa's built-in model.
        scr: Scratch directory for temporary files.
        lower_ph_threshold: Lower pH limit for basic centers.
        upper_ph_threshold: Upper pH limit for acidic centers.
        keep_files: Whether to keep the temporary files. Used for debugging. Defaults to False.

    Returns:
        MokaPrediction with the predicted pKa values.
    """
    _mol = Chem.Mol(mol)  # create a copy to avoid modifying the original
    if not _mol.HasProp("_Name"):
        _mol.SetProp("_Name", f"mol_{uuid.uuid4().hex[:8]}")
    results = get_moka_prediction_on_batch(
        [_mol],
        cmd=cmd,
        model=model,
        scr=scr,
        lower_ph_threshold=lower_ph_threshold,
        upper_ph_threshold=upper_ph_threshold,
        keep_files=keep_files,
    )
    name = _mol.GetProp("_Name")
    if name in results:
        return results[name]

    raise RuntimeError(f"MoKa prediction failed: no results for molecule {name}")
