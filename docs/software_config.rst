Software Configuration
======================

``qm_atlas`` wraps a number of external programs (xTB, Turbomole, COSMOtherm,
Jaguar, etc.).  The software configuration system lets you describe which of
these programs are installed on your machine, which environment modules to load
before running them, and where non-standard binaries live.

How it works
------------

A built-in default configuration is always loaded from
``src/qm_atlas/resources/default_software_config.yaml`` (bundled with the
package).  That file sets ``available: false`` for every supported program and
maps each command name to itself (i.e. the binary is assumed to be on
``$PATH``).

If the environment variable ``QM_ATLAS_SOFTWARE_CONFIG_FILE`` is set, the YAML
file it points to is loaded and **merged field-by-field** into the defaults.
This means:

* You only need to specify what differs from the defaults.
* Unmentioned software entries keep their default values (``available: false``).
* Within a software entry, unmentioned fields are inherited from the default.
* ``commands`` and ``environment_variables`` are merged key-by-key, so you can
  override a single command path without re-specifying the rest.

Activating the configuration
-----------------------------

.. code-block:: bash

   export QM_ATLAS_SOFTWARE_CONFIG_FILE=/path/to/your/software_config.yaml

If the variable is not set, ``qm_atlas`` runs entirely on defaults (all
programs disabled). This is the expected state for CI environments and fresh
checkouts.

If the variable *is* set but points to a file that does not exist, an
:class:`EnvironmentError` is raised immediately so the misconfiguration is
caught early.

Writing your configuration file
--------------------------------

Start from ``example_software_config.yaml`` (in the repository root).  The
minimal entry to enable a program is:

.. code-block:: yaml

   software_environments:
       xtb:
           available: true
           module: "xtb/6.7.1"

The following fields are recognised per software entry:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - ``available``
     - ``true`` / ``false``.  When ``false``, all tests that depend on this
       program are skipped automatically and calling
       :func:`~qm_atlas.software_environment.EnvironmentManager.get_command`
       raises an error.
   * - ``commands``
     - Mapping of logical command names to executable paths or invocation
       strings.  Dict-merged with defaults: you can override a single binary
       path without re-specifying the others.
   * - ``module``
     - Name (or list of names) of the environment module(s) to load before
       running the software.  See `Controlling the runtime environment`_ below.
   * - ``modulepaths``
     - List of paths to prepend to ``$MODULEPATH`` before loading modules.
   * - ``thread_control_env_vars``
     - List of environment variables that should be set to the number of
       requested cores (e.g. ``OMP_NUM_THREADS``).  Replaces the default list.
   * - ``environment_variables``
     - Arbitrary additional environment variables to set.  Dict-merged with
       any variables derived from loaded modules.  See
       `Controlling the runtime environment`_ below.

Controlling the runtime environment
-------------------------------------

Each software entry has its own isolated runtime environment.  This is
intentional: different programs can have conflicting requirements (e.g.
some softwares override the ``xtb`` executable), and running them in the same environment can cause subtle failures.

**Default behaviour — no environment specified**

If neither ``module`` nor ``environment_variables`` is set for a software
entry, ``qm_atlas`` simply inherits the process's current environment when
invoking that program.  This works when the required binary is already on
``$PATH`` and all necessary variables are set in the shell that launched
``qm_atlas`` and you're not worried about conflicts between programs.
It also means that if you have a working environment set up in your shell, you can get started with
a minimal config by just enabling the programs you want to use and leaving the environment fields unset.

If you want *all* programs to share a single common environment, set that
environment up in your shell before launching ``qm_atlas`` and leave the
environment fields unset.

**Per-software environment via** ``environment_variables``

The simplest way to modify the environment for a specific program is to provide
an explicit mapping of variable names to values:

.. code-block:: yaml

   software_environments:
       xtb:
           available: true
           environment_variables:
               XTBPATH: "/data/xtb/share/xtb"

These variables are merged on top of whatever the current process environment
provides before the subprocess is launched.

**Per-software environment via** ``module``

For HPC systems that use `Lmod <https://lmod.readthedocs.io>`_
, you can instruct ``qm_atlas`` to resolve the environment
changes that result from loading one or more modules:

.. code-block:: yaml

   software_environments:
       turbomole:
           available: true
           module: "cosmologic/2026"

       xtb_turbomole:
           available: true
           modulepaths: ["/opt/local/modulefiles"]
           module: ["cosmologic/2026", "xtb/6.7.1"]

When the ``EnvironmentManager`` is initialised, it executes each ``module
load`` command, compares the resulting environment to the baseline, and records
the *diff*.  That diff is what gets applied when the software is later invoked
— not a full shell environment snapshot.  This means ``module``-derived
variables and hand-written ``environment_variables`` can be freely combined; the
latter are merged on top of the former.

**Environment for submitted HPC scripts**

If there is a specific environment you set up before running ``qm_atlas``, you need to
provide instructions for how to set up that same environment in the submission script. By default,
only your virtual python environment and your setup for the software config file are forwarded to the
submission script, but any other environment variables or modules that you set up in your shell will
not be automatically forwarded. If you have all necessary modules specified in the software config file,
no additional setup is needed for HPC submission, as the software config will be loaded and applied in the submission script.
However, if you rely on additional environment variables or modules that are not specified in the software config file,
you need to ensure the right environment is set up in the submission script. This can be done using the ``submission``
block in your config file to specify additional modules to load or commands to run at the start of the script.

.. code-block:: yaml

   submission:
       module_purge: true          # call "module purge" at the start of every script
       module_use:                 # paths to add to MODULEPATH in the script
           - "/opt/local/modulefiles"
       module_load:                # modules to load in every submission script
           - "some/shared/module"
       command_prepend: "source /path/to/venv/bin/activate"
           # Shell command prepended before the payload.  If omitted, the
           # currently active Python virtual environment is detected
           # automatically so qm_atlas itself is importable in the script.

Individual software entries still apply their own ``module`` / ``environment_variables``
settings on top of this shared baseline when they execute within the submitted job.

.. _submission-script-anatomy:

How a submission script is built
---------------------------------

When a command runs with ``submit: true``, ``qm_atlas`` writes a script (into
``qm_atlas_submissions/``) that rebuilds the environment from scratch: it runs
the ``module purge`` / ``module load`` lines from the config's ``submission``
block, activates the Python environment, ``export``\ s
``QM_ATLAS_SOFTWARE_CONFIG_FILE``, and then runs the payload.

The activation line comes from ``submission.command_prepend``. **If you don't
set it explicitly, it is auto-detected** — first from the ``VIRTUAL_ENV``
environment variable, then from ``sys.prefix``.  This is a potential source of
subtle problems with submission scripts, if you have another virtual environment
active and load ``qm_atlas`` without resetting the ``VIRTUAL_ENV`` environment
variable.  To avoid this, set ``submission.command_prepend`` explicitly in your
config, or ``deactivate`` any personal venv before submitting.  You can check
the scripts inside ``qm_atlas_submissions/`` to make sure everything works as
expected.

Default configuration reference
---------------------------------

The file ``src/qm_atlas/resources/default_software_config.yaml`` is the
reference for supported software names and their default command
mappings. There is no need to edit this file; all
customisation should go in your own config file.

.. literalinclude:: ../src/qm_atlas/resources/default_software_config.yaml
   :language: yaml

Global settings
----------------

The configuration file may also contain a ``global_settings`` block for
absolute paths to external data files (MoKa model, COSMO solvent directories,
etc.).  This block has no defaults and is ignored when no config file is
provided.

.. code-block:: yaml

   global_settings:
       moka_model: "/path/to/retrained/moka/model.pt"
       cosmo_solvents_dir_fine:
           - "/path/to/solvents/fine/"
       cosmo_solvents_dir_tzvp:
           - "/path/to/solvents/tzvp/"

Effect on tests
---------------

Tests that require external programs use the
:func:`~qm_atlas.software_environment.EnvironmentManager.software_available`
check.  When ``available: false`` (the default), those tests are skipped with a
descriptive message rather than failing.  This means:

* The test suite always passes on a machine without any external QM software.
* Enabling a program in your config file automatically un-skips the
  corresponding tests.

Checking a configuration with ``test_software``
------------------------------------------------

You do not need a source code checkout to verify a configuration.  The installed
package ships these wrapper tests and exposes them through the standalone
``qm-atlas test_software`` command:

.. code-block:: bash

   # check the programs enabled in a specific configuration
   qm-atlas test_software --software_config /path/to/software_config.yaml

   # or use QM_ATLAS_SOFTWARE_CONFIG_FILE (e.g. set by an environment module)
   qm-atlas test_software

Programs marked ``available: false`` are skipped; the enabled ones are exercised
against the real binaries.  A clean run confirms every configured tool works,
while a failure points at a broken install or an expired license rather than a
code change.  Point ``--software_config`` at different files to compare
configurations.  See :doc:`cli` for the full command description.
