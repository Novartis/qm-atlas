"""
Main workflow functions for molecular property calculation and conformer generation.

This module exposes the primary workflow entry points that users should use:
- generate_fast_conformers: Fast conformer generation using force fields
- generate_rescoss_conformers: ReSCoSS workflow for solution conformers
- generate_tautomers: Tautomer generation and scoring workflow
- run_constrained_optimization: Constrained geometry optimization
"""

# Also expose submodules for users who want helper functions
from qm_atlas.workflows import (
    fast_conformers,
    optimize_constrained,
    rescoss_conformers,
    score_tautomers,
)

# Main workflow entry points - exposed at package level
from qm_atlas.workflows.fast_conformers import generate_fast_conformers
from qm_atlas.workflows.optimize_constrained import run_constrained_optimization
from qm_atlas.workflows.rescoss_conformers import generate_rescoss_conformers
from qm_atlas.workflows.score_tautomers import generate_tautomers

__all__ = [
    # Main workflow functions (prioritized at package level)
    "generate_fast_conformers",
    "generate_rescoss_conformers",
    "generate_tautomers",
    "run_constrained_optimization",
    # Submodules for advanced users
    "fast_conformers",
    "rescoss_conformers",
    "score_tautomers",
    "optimize_constrained",
]
