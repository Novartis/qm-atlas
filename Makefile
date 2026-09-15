
# Commands
uv=uv
python=uv run python
pip=uv pip

# Version
version_file=src/qm_atlas/version.py
VERSION=$(shell grep -oP '(?<=VERSION = ")[^"]+' $(version_file))
VERSION_MAJOR=$(shell echo $(VERSION) | cut -d'.' -f1)
VERSION_MINOR=$(shell echo $(VERSION) | cut -d'.' -f2)
VERSION_PATCH=$(shell echo $(VERSION) | cut -d'.' -f3)
GIT_COMMIT=$(shell git rev-parse --short HEAD)

# Test
coverage_threshold=80
module=qm_atlas

tmp_dir := ${TMPDIR}
ifeq ($(tmp_dir), )
	tmp_dir := "."
endif
tmp_dir := $(tmp_dir)/.pytest_scr
current_dir = $(shell pwd)

.PHONY: clean test docs version set-version bump-version-patch bump-version-minor bump-version-major bump-version-dev commit-tag-version build-dist test-dist test-software

all: sync install-pre-commit

# Setup

sync:
	${uv} sync --dev

install: sync

install-pre-commit:
	${python} -m pre_commit install

# CI

build-dist:
	${uv} build

test-dist:
	uvx twine check dist/*

# Admin

freeze-env:
	${uv} export --frozen > requirements_`date +%m-%d-%y`.txt

format:
	${uv} run pre-commit run --all-files

test: test-unit

test-unit:
	${python} -m pytest --basetemp=$(tmp_dir) -rs tests/ --durations=50

# Run the wrapper self-tests against a software config (pass software_config=... to
# point at a specific one; otherwise QM_ATLAS_SOFTWARE_CONFIG_FILE / default is used).
test-software:
	${python} -m qm_atlas test_software $(if $(software_config),--software_config $(software_config),)

# Combined coverage over two sessions (unit tests + wrapper self-tests, which run
# in a separate pytest session) merged via --cov-append.
test-cov:
	${python} -m pytest --basetemp=$(tmp_dir) -rs --cov=${module} --cov-report= tests/
	${python} -m qm_atlas test_software $(if $(software_config),--software_config $(software_config),) --cov=${module} --cov-append --cov-report=term --cov-fail-under ${coverage_threshold}

test-cov-html:
	${python} -m pytest --basetemp=$(tmp_dir) -rs --cov=${module} --cov-report= tests/
	${python} -m qm_atlas test_software $(if $(software_config),--software_config $(software_config),) --cov=${module} --cov-append --cov-report=html

# Smoke-test the notebooks by running their tracked .py sources top-to-bottom.
# They need QM_ATLAS_SOFTWARE_CONFIG_FILE; uv loads it from .env, as VS Code does.
test-ipynb:
	@test -f .env || { echo ".env not found; copy .env.example to .env and set the config path"; exit 1; }
	@for nb in notebooks/*.py; do \
		echo "=== Running $$nb ==="; \
		${uv} run --env-file .env python "$$nb" || exit 1; \
	done

# Notebook conversion
notebooks-to-py:
	@echo "Converting .ipynb files to .py..."
	${uv} run jupytext --to py:percent notebooks/*.ipynb
	@echo "Conversion complete. Updated .py files are ready to commit."

notebooks-from-py:
	@echo "Generating .ipynb files from .py..."
	${uv} run jupytext --to notebook notebooks/*.py
	@echo "Generated .ipynb files locally."

# Docs

# Build the HTML docs (reuses docs/Makefile). Output: docs/_build/html.
docs:
	${uv} run --extra docs $(MAKE) -C docs html

todo:
	grep "# TODO" */*.py | sed -e 's/    //g' | sed -e 's/# TODO//'

## Version

version:
	echo $(VERSION)

set-version:
	test ! -z "$(VERSION)"
	sed -i 's/VERSION = ".*"/VERSION = "$(VERSION)"/' $(version_file)

bump-version-patch:
	$(MAKE) set-version VERSION=$(VERSION_MAJOR).$(VERSION_MINOR).$(shell awk 'BEGIN{print $(VERSION_PATCH)+1}')

bump-version-minor:
	$(MAKE) set-version VERSION=$(VERSION_MAJOR).$(shell awk 'BEGIN{print $(VERSION_MINOR)+1}').0

bump-version-major:
	$(MAKE) set-version VERSION=$(shell awk 'BEGIN{print $(VERSION_MAJOR)+1}').0.0

bump-version-dev:
	test ! -z "$(GIT_COMMIT)"
	$(MAKE) set-version VERSION=$(VERSION_MAJOR).$(VERSION_MINOR).$(VERSION_PATCH).dev+git.$(GIT_COMMIT)

commit-tag-version:
	git commit -m "Version $(VERSION)" --no-verify $(version_file)
	git tag 'v$(VERSION)'

# Clean

clean:
	find . -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf _tmp_* htmlcov nbconvert
	rm -rf dist/ *.egg-info/ src/*.egg-info/
