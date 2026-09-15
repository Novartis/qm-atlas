# Minimal makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXPOLY    ?= sphinx-polyversion
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help multiversion Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build documentation for all whitelisted versions (branches/tags) into
# _build/html/<version>/, with a root redirect to the default version.
multiversion:
	$(SPHINXPOLY) poly.py

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
