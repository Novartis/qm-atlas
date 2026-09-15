"""Sphinx configuration for qm_atlas documentation."""

import os
import sys

# Add the src directory to the path so we can import qm_atlas
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from qm_atlas.version import VERSION

project = "qm_atlas"
copyright = "2026, Jimmy Kromann, Hagen Muenkler"
author = "Jimmy Kromann, Hagen Muenkler"
release = VERSION

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",
    "sphinxcontrib.autodoc_pydantic",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "_polyversion", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]

# sphinx-polyversion: when invoked through `sphinx-polyversion`, the metadata
# about all built versions is passed via the POLYVERSION_DATA env var. load()
# reads it and populates html_context with `revisions` (all versions) and
# `current` (the version being built). During a plain `make html` build the env
# var is absent, so we skip it and the version switcher renders nothing.
try:
    from sphinx_polyversion import load
    from sphinx_polyversion.api import LoadError

    load(globals())
except (ImportError, LoadError):  # not building via sphinx-polyversion
    pass

# Show the version switcher in the furo sidebar
html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "versions.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
        "sidebar/variant-selector.html",
    ]
}

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
# Render default argument values as written in the source (e.g. ``DEFAULT_SCR``)
# instead of their evaluated repr (e.g. ``PosixPath('/scratch/<user>/...')``).
autodoc_preserve_defaults = True
autosummary_generate = True

# Silence noise from sphinx-autodoc-typehints trying to resolve pydantic's own
# internal ``Field`` / validator annotations (e.g. the unresolved ``JsonValue``
# forward reference). These originate in the pydantic library, not in this project.
suppress_warnings = [
    "sphinx_autodoc_typehints.forward_reference",
    "sphinx_autodoc_typehints.guarded_import",
]

# autodoc-pydantic settings: show field options, hide config/validator noise
autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_model_show_validator_members = False
autodoc_pydantic_model_show_field_summary = False
autodoc_pydantic_model_member_order = "bysource"
autodoc_pydantic_field_list_validators = False
autodoc_pydantic_field_show_constraints = True

# Napoleon settings (Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Intersphinx mapping. Resolving these cross-references (e.g. ``bool`` -> python
# docs, ``rdkit...Mol`` -> rdkit docs) requires network access to fetch each
# project's ``objects.inv`` at build time; without it the types render as plain
# text instead of links.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "rdkit": ("https://www.rdkit.org/docs/", None),
}

# MyST parser (allows including .md files)
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}


def _strip_inherited_value_docstring(app, what, name, obj, options, lines):
    """Improve rendering of documented module-level constants.

    - Multi-line string constants (e.g. Jinja templates) are shown as a readable
      literal block reproducing their actual value, instead of an escaped one-line repr.
    - For other constants, autodoc falls back to ``type(value).__doc__`` when the
      constant has no docstring of its own, rendering noise like ``str(object='') -> str``;
      those inherited docstrings are cleared so the shown value speaks for itself.
    """
    import inspect

    if what not in ("data", "attribute"):
        return

    if isinstance(obj, str) and "\n" in obj:
        lines[:] = ["::", ""] + ["    " + line for line in obj.splitlines()] + [""]
        return

    if lines:
        type_doc = inspect.getdoc(type(obj))
        if type_doc and "\n".join(lines).strip() == type_doc.strip():
            lines[:] = []


import re

_DEFAULT_SCR_XREF = ":data:`~qm_atlas.constants.DEFAULT_SCR`"
# Match a plain ``DEFAULT_SCR`` / ``constants.DEFAULT_SCR`` token not already in a role.
_DEFAULT_SCR_RE = re.compile(r"(?<![`\w.])(?:constants\.)?DEFAULT_SCR(?![`\w])")


def _linkify_default_scr(app, what, name, obj, options, lines):
    """Turn plain-text ``DEFAULT_SCR`` docstring mentions into cross-references."""
    if name == "qm_atlas.constants.DEFAULT_SCR":
        return
    for i, line in enumerate(lines):
        if "DEFAULT_SCR" in line:
            lines[i] = _DEFAULT_SCR_RE.sub(_DEFAULT_SCR_XREF, line)


def setup(app):
    app.connect("autodoc-process-docstring", _strip_inherited_value_docstring)
    app.connect("autodoc-process-docstring", _linkify_default_scr)
