"""Describe available configuration options using jsonargparse formatting.

This is an alternative implementation that leverages jsonargparse's built-in
formatting of Pydantic models. It produces output consistent with `--help` style.

Usage:
    qm-atlas describe                                    # list all categories
    qm-atlas describe calculators                        # list available property calculators
    qm-atlas describe calculators xtb_single_point       # show options in jsonargparse format
    qm-atlas describe optimizers jobex                   # show optimizer options
"""

from __future__ import annotations

import sys

from jsonargparse import ArgumentParser
from pydantic import BaseModel

from qm_atlas.tasks import (
    align,
    calculate_properties,
    conformer_generation,
    cosmo_properties,
    optimize,
)

# ---------------------------------------------------------------------------
# Lazy registry loading — avoids heavy imports until actually needed
# ---------------------------------------------------------------------------


CONFORMER_GENRATOR_OPTIONS = {
    name: opts_cls
    for name, (_, opts_cls) in conformer_generation.GENERATOR_FUNCTIONS_AND_OPTIONS.items()
}


CATEGORIES: dict[str, tuple[str, dict]] = {
    "calculators": (
        "Property calculators for conformer_properties (dispatch key: backend)",
        calculate_properties.CONFIG_CLASS_REGISTRY,
    ),
    "optimizers": (
        "Geometry optimization drivers for fast_conformers / rescoss_conformers / optimize_reference_conformers (dispatch key: backend)",
        optimize.OPTIONS_CLASS_REGISTRY,
    ),
    "conformer_generators": (
        "Conformer generators for fast_conformers / rescoss_conformers (dispatch key: backend)",
        CONFORMER_GENRATOR_OPTIONS,
    ),
    "cosmo_tasks": (
        "COSMOtherm task types for cosmotherm_properties (dispatch key: backend)",
        cosmo_properties.OPTIONS_CLASS_REGISTRY,  # or _load_cosmo_tasks for lazy loading
    ),
    "aligners": (
        "Alignment backends for the align command (dispatch key: backend)",
        align.OPTIONS_CLASS_REGISTRY,
    ),
}


# ---------------------------------------------------------------------------
# Command logic
# ---------------------------------------------------------------------------


def _print_categories() -> None:
    """Print all available categories."""
    print("\nAvailable categories:\n")
    for cat_name, (description, _loader) in CATEGORIES.items():
        print(f"  {cat_name:<25} {description}")
    print(
        """\nUsage:
        qm-atlas describe <category>              List entries in a category
        qm-atlas describe <category> <name>        Show detailed options for an entry
        """
    )


def _print_category_entries(category: str) -> None:
    """Print all entries in a category."""
    description, registry = CATEGORIES[category]

    print(f"\n{description}\n")
    print(f"Available entries ({len(registry)}):\n")

    for name, cls in registry.items():
        doc = cls.__doc__
        # Use first line of docstring as short description
        short_desc = doc.strip().split("\n")[0] if doc else ""
        print(f"  {name:<35} {short_desc}")

    print(f"\nRun 'qm-atlas describe {category} <name>' for detailed options.\n")


def _print_entry_detail_jsonargparse(entry_name: str, model_cls: type[BaseModel]) -> None:
    """Print detailed options for a specific entry using jsonargparse formatting."""
    doc = model_cls.__doc__
    short_desc = doc.strip().split("\n")[0] if doc else ""

    print(f"\n{entry_name}")
    print(f"{'=' * len(entry_name)}")
    if short_desc:
        print(f"{short_desc}\n")

    # Create a minimal parser just to leverage jsonargparse's formatting
    parser = ArgumentParser(description=f"Options for {entry_name}", add_help=False)
    parser.add_class_arguments(model_cls, as_positional=False)

    # Capture the help output and extract only the options section
    help_text = parser.format_help()
    lines = help_text.split("\n")

    # Find the first line that starts an option (--) and print from there onwards
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("--"):
            start_idx = i
            break

    if start_idx is not None:
        print("\n".join(lines[start_idx:]))


def _print_entry_detail(category: str, entry_name: str) -> None:
    """Print detailed options for a specific entry."""
    _description, registry = CATEGORIES[category]

    if entry_name not in registry:
        print(f"\nUnknown entry '{entry_name}' in category '{category}'.")
        print(f"Available entries: {', '.join(registry.keys())}\n")
        sys.exit(1)

    cls = registry[entry_name]
    _print_entry_detail_jsonargparse(entry_name, cls)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Entry point for the describe command."""
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in ("--help", "-h"):
        _print_categories()
        return

    category = argv[0]
    if category not in CATEGORIES:
        print(f"\nUnknown category '{category}'.")
        _print_categories()
        sys.exit(1)

    if len(argv) < 2:
        _print_category_entries(category)
    else:
        _print_entry_detail(category, argv[1])


if __name__ == "__main__":
    main()
