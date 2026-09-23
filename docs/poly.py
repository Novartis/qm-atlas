"""sphinx-polyversion configuration for building multi-version documentation.

Run locally with::

    make -C docs multiversion          # or: sphinx-polyversion docs/poly.py

It checks out every matching git branch/tag into a temporary worktree and builds
its docs into ``docs/_build/html/<version>/``. A root ``index.html`` redirects to
the default version. Builds run in the *current* environment (no per-version
virtualenvs), so the already-installed dependencies are reused.
"""

from pathlib import Path

from sphinx_polyversion import DefaultDriver
from sphinx_polyversion.environment import Environment
from sphinx_polyversion.git import Git
from sphinx_polyversion.sphinx import SphinxBuilder

# Repository root (parent of this docs/ directory)
ROOT = Path(__file__).resolve().parent.parent

# Which refs to build documentation for
BRANCH_REGEX = r"^main$"  # branches
TAG_REGEX = r"^v\d+\.\d+.*$"  # release tags like v1.2.3

# Docs source dir (relative to each checked-out repo) and output dir (absolute)
SOURCE_DIR = "docs"
OUTPUT_DIR = str(ROOT / "docs" / "_build" / "html")

DefaultDriver(
    ROOT,
    OUTPUT_DIR,
    vcs=Git(branch_regex=BRANCH_REGEX, tag_regex=TAG_REGEX),
    builder=SphinxBuilder(SOURCE_DIR, args=["-b", "html"]),
    env=Environment,  # build in the current environment, no isolated venvs
    template_dir=ROOT / "docs" / "_polyversion",
).run()
