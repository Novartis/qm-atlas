"""Preview the SDF property tag names a calculation config will produce.

Both ``conformer_properties`` and ``cosmotherm_properties`` fully describe their
calculation tasks through task-options classes that implement
``get_property_names()`` and ``get_property_types()``. This module turns a list
of such tasks into a human-readable table of the property tags (and their
:class:`~qm_atlas.tasks.common.Property` types) that a run would write to the
result SDF files, without running any calculation.

It backs the ``--print-output-names`` flag on both CLIs.
"""

from pathlib import Path
from typing import Any, Protocol

import yaml

PRINT_OUTPUT_NAMES_FLAGS = ("--print-output-names", "--print_output_names")


class _NamedTask(Protocol):
    def get_property_names(self) -> dict[str, str]:
        ...

    def get_property_types(self) -> dict[str, type]:
        ...


def collect_output_names(tasks: list[_NamedTask]) -> list[tuple[str, str, str]]:
    """Return ``(tag, property_type_name, task_label)`` for every output tag.

    Args:
        tasks: Calculation task-options objects.

    Returns:
        list[tuple[str, str, str]]: One row per SDF property tag.
    """
    rows: list[tuple[str, str, str]] = []
    for task in tasks:
        names = task.get_property_names()
        types = task.get_property_types()
        label = getattr(task, "backend", "") or ""
        for tag in names.values():
            prop_type = types.get(tag)
            type_name = prop_type.__name__ if isinstance(prop_type, type) else ""
            rows.append((tag, type_name, str(label)))
    return rows


def format_output_names(tasks: list[_NamedTask]) -> str:
    """Format the output property tags of *tasks* as an aligned table."""
    rows = collect_output_names(tasks)
    if not rows:
        return "No output properties configured."

    name_width = max(len(row[0]) for row in rows + [("PROPERTY", "", "")])
    type_width = max(len(row[1]) for row in rows + [("", "TYPE", "")])

    header = f"{'PROPERTY'.ljust(name_width)}  {'TYPE'.ljust(type_width)}  TASK"
    lines = [header, "-" * len(header)]
    for tag, type_name, label in rows:
        lines.append(f"{tag.ljust(name_width)}  {type_name.ljust(type_width)}  {label}")
    return "\n".join(lines)


def extract_config_path(argv: list[str]) -> Path | None:
    """Return the value of a ``--config`` argument from *argv*, if present."""
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if arg.startswith("--config="):
            return Path(arg.split("=", 1)[1])
    return None


def load_options_section(config_path: Path, options_key: str) -> dict[str, Any]:
    """Load an options sub-section from a YAML config file.

    Args:
        config_path: Path to the YAML config file.
        options_key: Top-level key holding the calculation options
            (e.g. ``conformer_property_options``).

    Returns:
        dict[str, Any]: The options sub-mapping, or an empty dict if absent.
    """
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    section = data.get(options_key, {})
    return section if isinstance(section, dict) else {}
