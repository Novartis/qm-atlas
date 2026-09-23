
# qm_atlas

**QM-Assisted Toolchain for Ligand Assessment and Scoring**<br/>
A Python toolkit for automating quantum chemistry workflows in small-molecule drug discovery and development.

`qm_atlas` provides conformer generation, quantum mechanical property calculations, and COSMO-RS-based thermodynamic predictions through a unified CLI and Python API.

> **Platform support:** `qm_atlas` has been developed and tested exclusively on Linux, and no attempt has been made to make it portable across other platforms. Running on other operating systems is unsupported.

## Installation

All dependencies are available on PyPI, so `qm_atlas` can be installed directly from GitHub. [`uv`](https://docs.astral.sh/uv/) is the recommended installer:

```bash
git clone https://github.com/Novartis/qm-atlas.git
cd qm-atlas
uv sync
```

To install into an existing environment with `pip`:

```bash
pip install "qm_atlas @ git+https://github.com/Novartis/qm-atlas.git"
```

For a development setup, see [Setup for development](#setup-for-development).

## Overview

`qm_atlas` is organized in three layers, from low-level building blocks to ready-to-use pipelines (see the [documentation](https://opensource.nibr.com/qm-atlas/) for details):

- **Wrappers** — thin interfaces around individual external programs (xTB, Turbomole, COSMOtherm, Jaguar, and the conformer generators).
- **Tasks** — individual calculation steps (e.g., conformer generation, optimization, property calculation) that expose a uniform interface with interchangeable backends built on top of the wrappers.
- **Workflows** — higher-level, multi-step pipelines that combine tasks into complete procedures such as the conformer-generation workflows below.

The same functionality is available through the command-line interface and the Python API.

### Conformer Generation

- **ReSCoSS** (Relevant Solution Conformer Sampling and Selection, [[1]](#references)), a multi-step workflow that generates a large conformer pool, optimizes geometries at the xTB level, clusters conformers using COSMO-based descriptors, and selects representative structures per cluster for downstream DFT calculations
- **Fast conformers** — a much faster workflow relying on a simplified clustering based on SASA and dipole moments, with xTB optimization of selected conformations


### Property Calculations

- **COSMO-RS properties** via COSMOtherm: logP, solvation free energies (ΔG_solv), pKa, COSMO polar surface area, COSMO descriptors
- **Solubility & co-crystal screening** — thermodynamic solubility predictions and co-crystal stability assessments for development workflows
- **Quantum mechanical properties** — single-point energy, HOMO-LUMO gaps, dipole moments, Fukui indices, NMR shieldings, VCD spectra (via Turbomole or xTB)
- **Conformational analysis** — Boltzmann-weighted conformer populations in different solvents, strain energy estimation for bioactive poses

### Supported Backends

| Program | Capabilities |
|---------|-------------|
| xTB | Geometry optimization, single-point energy, Fukui indices, partial charges |
| Turbomole | Geometry optimization, single point energy, NMR shieldings, VCD, thermochemistry |
| COSMOtherm | LogP, ΔG_solv, pKa, solubility, co-crystal screening, COSMO descriptors |
| Jaguar | Quantum chemistry calculations supported by Schrödinger |
| CORINA / OMEGA / MOE / Macromodel / RDKit | 3D conformer generation |
| MoKa | pKa prediction |
| UNICON | Tautomer generation |

### CLI

The `qm-atlas` command-line interface provides subcommands for each stage of a calculation workflow:

| Command | Description |
|---------|-------------|
| `read_input` | Parse input molecules (SDF) and create compound directories |
| `add_reference_conformers` | Register reference conformers (e.g., bioactive poses) in compound directories |
| `generate_tautomers` | Generate tautomers and register as new states in compound directories |
| `protonate` | Generate protonation states and register as new states in compound directories |
| `fast_conformers` | Fast conformer generation workflow |
| `rescoss_conformers` | ReSCoSS conformer generation and selection |
| `optimize_reference_conformers` | Constrained optimization of reference conformers |
| `conformer_properties` | Calculate QM descriptors on conformer ensembles |
| `cosmotherm_properties` | Run COSMO-RS property calculations (logP, solvation free energy, etc.) |
| `cosmo_pka` | Calculate pKa values with COSMOtherm across registered protonation states |
| `solvent_screening` | Run COSMOtherm pure-solvent and mixture solubility screening |
| `cocrystal_screening` | Run COSMOtherm co-crystal screening |

There are two additional interfaces:

| Command | Description |
|---------|-------------|
| `pipeline` | Execute multi-step workflows from a YAML configuration, chaining all of the above commands |
| `describe` | List available calculators, optimizers, and generators, and their options to provide in-depth documentation for the above commands |

### Pipeline Mode

Multi-step workflows can be defined in YAML and executed in a single call:

```bash
qm-atlas pipeline --config my_workflow.yaml
```

This enables reproducible, end-to-end calculation pipelines from input structures to final properties.

## Software Configuration

`qm_atlas` can connect to a number of supported external programs (xTB, Turbomole, COSMOtherm, Jaguar, etc.), which users may or may not have access to. All external programs are **disabled by default** (`available: false`), so the package can run out of the box without any external dependencies installed, although its functionality will be extremely limited if none of these are available.

To enable programs available on your system, create a configuration file based on `example_software_config.yaml` and point the `QM_ATLAS_SOFTWARE_CONFIG_FILE` environment variable to it:

```bash
export QM_ATLAS_SOFTWARE_CONFIG_FILE=/path/to/your/software_config.yaml
```

Your file is merged field-by-field into the defaults — you only need to specify what differs (typically `available: true` and a `module` name). Tests that depend on a software package are skipped automatically when that package is marked `available: false`.

You can test a software_config file directly on an installed version of qm_atlas, without checking out the source code, using:

    # Check the tools enabled in a specific configuration
    qm-atlas test_software --software_config /path/to/software_config.yaml

See [docs/software_config.rst](docs/software_config.rst) for the full reference, including the default configuration, all supported fields, and merge semantics.

### Configuring notebooks (VS Code / Jupyter)

The interactive notebooks read the same `QM_ATLAS_SOFTWARE_CONFIG_FILE` variable, but a `jupyter lab` shell `export` does not reach a kernel that VS Code starts for you. You can rely on a git-ignored `.env` file in the repository root. VS Code's Python/Jupyter extension loads it into the kernel environment automatically at startup:

```bash
cp .env.example .env
# then edit .env and set QM_ATLAS_SOFTWARE_CONFIG_FILE to your software config
```

```properties
# .env
QM_ATLAS_SOFTWARE_CONFIG_FILE=/path/to/your/software_config.yaml
```

Restart the Jupyter kernel after editing `.env`. The workflow notebook validates this variable in its first cell and fails early with a clear message if it is missing. `.env` is git-ignored; `.env.example` is committed as a template.


### Interactive Examples

The `notebooks/` directory contains runnable, documented examples. Every notebook
starts with the same software-configuration check described above, then walks
through one task or workflow.

**End-to-end example**

- `example_workflow` — a full graph-preparation pipeline (tautomers →
  protonation → stereo → conformers → xTB energies).

**Tasks** (`src/qm_atlas/tasks`)

- `task_conformer_generation` — generate 3D conformers (RDKit / CORINA / OMEGA / MOE / MacroModel).
- `task_optimize` — geometry optimization with xTB, Turbomole `jobex`, or xTB-driven Turbomole.
- `task_calculate_properties` — single-point properties (xTB, Turbomole, Jaguar).
- `task_cosmo_properties` — all COSMOtherm properties (log P, ΔG solvation, descriptors, PSA, solubility, pKa).
- `task_boltzmann` — Boltzmann weighting and conformer averaging (pure Python, no external software).
- `task_tautomer_generation` — enumerate, embed and xTB-screen tautomers.
- `task_protonation` — enumerate protonation states / protomers with MoKa.

**Workflows** (`src/qm_atlas/workflows`)

- `workflow_fast_conformers` — fast force-field + xTB conformer pipeline.
- `workflow_rescoss_conformers` — ReSCoSS solution-conformer selection (xTB + Turbomole + COSMOtherm).
- `workflow_optimize_constrained` — constrained geometry optimization.
- `workflow_score_tautomers` — DFT / COSMOtherm tautomer ranking.

There is also `analyze_deduplication`, an analysis notebook on RMSD-based
conformer deduplication.

**To use the interactive notebooks:**

```bash
# Generate .ipynb notebooks from source files
make notebooks-from-py

# Launch Jupyter
jupyter notebook notebooks/
```

The notebooks are generated locally from source Python files (kept clean in Git). Run `make notebooks-from-py` after pulling updates to ensure you have the latest documentation.

## Running locally vs. on an HPC cluster

Every `qm_atlas` command can run in two modes, controlled by the `submit` option (default `submit: false`):

- **Locally** (default) — calculations run in the current process/machine. This works everywhere and requires no scheduler.
- **Submitted to a cluster** — for larger workloads, jobs can be dispatched to an HPC scheduler for parallel execution. This is more convenient for big batches, but it is **optional**: everything can be run locally without it.

### Scheduler support

Cluster submission is currently implemented **only for UGE (Univa/Altair Grid Engine)** — the `engine` option defaults to `"uge"` and no other scheduler is wired up yet. The scheduler integration lives in the external [`hpc-funcs`](https://pypi.org/project/hpc-funcs/) package plus a thin layer in this repository.

Support for other schedulers (e.g. **Slurm**) is **not yet available but can be added in the future** by extending `hpc-funcs` and the submission layer here. Until then, run with `submit: false` on non-UGE systems.

> **Note for contributors:** the current maintainers only have access to a UGE cluster and therefore cannot develop or test wrappers for other schedulers (such as Slurm). Contributions adding and validating additional scheduler backends are very welcome and would need to come from someone with access to the corresponding system. See [CONTRIBUTORS.md](CONTRIBUTORS.md).

### How a submission script is built

When a command runs with `submit: true`, `qm_atlas` writes a script (into `qm_atlas_submissions/`) that rebuilds the environment from scratch: it runs the `module purge`/`module load` lines from the config's `submission` block, activates the Python environment, `export`s `QM_ATLAS_SOFTWARE_CONFIG_FILE`, and then runs the payload.

The activation line comes from `submission.command_prepend`. **If you don't set it explicitly, it is auto-detected** — first from the `VIRTUAL_ENV` environment variable, then from `sys.prefix`. This is a potential source of subtle problems with submission scripts, if you have another virtual environment active and load qm_atlas without resetting the  `VIRTUAL_ENV` environment variable. To avoid this, set `submission.command_prepend` explicitly in your config, or `deactivate` any personal venv before submitting. You can check the scripts inside `qm_atlas_submissions/` to make sure everything works as expected.

## Setup for development

This repository uses [uv](https://docs.astral.sh/uv/) for dependency management and environment setup.

- Clone repository
- `make`

### Running tests and tools

All commands should be run through uv to use the correct environment:

    # Run tests
    uv run pytest tests/

    # Run tests with coverage
    uv run pytest --cov=qm_atlas tests/

    # Format code with pre-commit
    uv run pre-commit run --all-files

    # Or use Makefile targets
    make test
    make test-cov
    make format

#### Tests and external software

Many tests exercise wrappers around commercial or otherwise licensed programs (Turbomole, COSMOtherm, Jaguar, MOE, MacroModel, OMEGA, MoKa, …). These tests are **skipped automatically** whenever the corresponding program is marked as `available: false` in the software configuration.

With the **default configuration** (no `QM_ATLAS_SOFTWARE_CONFIG_FILE` set, or a config in which the programs are disabled), only the tests that do not depend on external software will be executed — everything requiring a licensed backend is skipped. This is the mode used for continuous integration on GitHub, where such software is not available:

    # Runs only tests that need no external software (CI-equivalent)
    uv run pytest tests/

To run the **full** test suite, you must be on a system where the required programs are installed, and point `QM_ATLAS_SOFTWARE_CONFIG_FILE` at a config that enables them (see [Software Configuration](#software-configuration)). The green checkmark on a GitHub pull request therefore only certifies the software-independent subset; the complete suite has to be validated separately on an appropriately provisioned machine.

#### Checking a software configuration (`qm-atlas test_software`)

The wrapper tests are available **within the installed package**; therefore, you can validate a software configuration without a source code checkout:

    # Check the tools enabled by a specific configuration
    qm-atlas test_software --software_config /path/to/software_config.yaml

    # Or use QM_ATLAS_SOFTWARE_CONFIG_FILE (e.g. set by an environment module)
    qm-atlas test_software

Tools that are marked as unavailable in the configuration are skipped; the ones that are enabled are exercised against the real programs. A clean test run means every configured tool works, while a failure points at a broken install or an expired license rather than a code change. The exit code is the pass/fail signal (handy for CI); `--report results.xml` writes a JUnit report, and any extra arguments are forwarded to `pytest` (e.g. `-k turbomole` to check a single backend).

This makes it easy to **compare different software configurations** — point `--software_config` at each one in turn — and it is the mechanism the internal deployment uses as its acceptance check. Locally you can also run `make test-software software_config=<path>`.

### Git setup

For commit hooks, set up pre-commit:

    uv run pre-commit install

This ensures code is formatted before committing.

### Editing Notebooks

Notebooks are stored as Python files with cell markers for clean version control. To edit and update them:

1. **Generate interactive notebooks locally:**
   ```bash
   make notebooks-from-py
   ```

2. **Edit in Jupyter:**
   ```bash
   jupyter notebook notebooks/
   ```

3. **Convert back to Python files before committing:**
   ```bash
   make notebooks-to-py
   ```

   Or let the pre-commit hook handle it automatically:
   ```bash
   git add notebooks/*.ipynb
   git commit  # Hook converts to .py and stages for you
   ```

4. **Commit only the `.py` files:**
   ```bash
   git commit -m "Update notebook documentation"
   ```

This workflow keeps your Git history clean (no large `.ipynb` files) while allowing interactive development in Jupyter.

## Matomo Tracking

`qm_atlas` includes optional usage tracking via Matomo to monitor command usage and help improve the toolkit. Tracking is **disabled by default** and only activates when the required environment variables are configured.

### Environment Variables

Matomo tracking requires the following environment variables to be set:

| Variable | Description |
|----------|-------------|
| `MATOMO_URL` | URL of the Matomo server (e.g., `http://matomo.example.com/matomo.php`) |
| `MATOMO_SITE_ID` | Matomo site ID for tracking |
| `MATOMO_TIMEOUT` | Request timeout in seconds (default: `100`) |
| `USER` | Username of the person running the command (typically set automatically) |
| `HOSTNAME` | Hostname of the machine (typically set automatically) |

### Default Behavior

- If `MATOMO_URL` or `MATOMO_SITE_ID` are not set, tracking is automatically disabled. The CLI continues to function normally without any tracking overhead.
- **If all variables are set**: Usage data is collected asynchronously in the background without blocking command execution.

### Disabling Tracking

Even if all environment variables are configured, you can opt out of tracking for a specific command using the `--incognito` flag:

```bash
qm-atlas <command> --incognito <args>
```

This can for example be used for automated testing scripts, which should not count towards usage of the CLI.

### Example Setup

To enable Matomo tracking, set the environment variables:

```bash
export MATOMO_URL="https://your-matomo-instance/matomo.php"
export MATOMO_SITE_ID="12"
export MATOMO_TIMEOUT="100"
# USER and HOSTNAME are typically auto-detected from your system
```

Then, run any `qm_atlas` command normally, and usage data will be tracked.

## Code of Conduct

- Always create a feature branch and open a pull request. Never commit directly
  to `master`
- Always use `pre-commit` for formatting your code
- For every new feature added, a test must be added

# References

[1] Udvarhelyi, A., Rodde, S. & Wilcken, R. ReSCoSS: a flexible quantum chemistry workflow
identifying relevant solution conformers of drug-like molecules.
J Comput Aided Mol Des 35, 399-415 (2021). https://doi.org/10.1007/s10822-020-00337-7
