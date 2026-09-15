Command-Line Interface
======================

``qm-atlas`` ships a single executable, ``qm-atlas``, that dispatches to a set
of subcommands. Running it without arguments prints an overview of the
available commands:

.. code-block:: bash

   qm-atlas

The ``pipeline`` command
------------------------

The main entry point for multi-step workflows is the ``pipeline`` command. It chains
several calculation steps into a single workflow that is fully
described in a ``.yaml`` configuration file:

.. code-block:: bash

   qm-atlas pipeline --config my_workflow.yaml \
       --input_file+ ./molecules.sdf \
       --results_directory ./results

The config file lists the steps to run, in order, together with the options for
each step. This lets you specify complex calculations — for example "read
input, generate conformers, optimize them, then compute COSMOtherm properties"
— entirely in one place and reproduce them from a single command. Steps that
submit jobs to a cluster automatically block subsequent steps until they
finish, and the whole pipeline can itself be submitted to the cluster.

Step commands
-------------

Each step that the pipeline runs is also a standalone subcommand, and each
of them can be called directly. This is useful for running or re-running a
single stage of a workflow. The available step commands include:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Command
     - Description
   * - ``read_input``
     - Read input molecules and create compound directories.
   * - ``add_reference_conformers``
     - Register reference conformers in compound directories.
   * - ``fast_conformers``
     - Run the fast conformer-generation workflow.
   * - ``rescoss_conformers``
     - Run the ReSCoSS conformer-generation workflow.
   * - ``optimize_reference_conformers``
     - Constrained optimization of reference conformers.
   * - ``conformer_properties``
     - Calculate conformer-based QM properties (Fukui, NMR, …).
   * - ``cosmotherm_properties``
     - Calculate COSMOtherm properties (logP, ΔG_solv, …).
   * - ``cosmo_pka``
     - Calculate pKa values with COSMOtherm.
   * - ``generate_tautomers``
     - Generate tautomers and register them as new states.
   * - ``protonate``
     - Generate protonation states and register them as new states.
   * - ``solvent_screening``
     - COSMOtherm pure-solvent and mixture solubility screening.
   * - ``cocrystal_screening``
     - COSMOtherm co-crystal screening.
   * - ``calculate_boltzmann_weights``
     - Per-conformer Boltzmann weights from a calculated energy property.
   * - ``aggregate_conformer_properties``
     - Aggregate per-conformer properties into per-state values.
   * - ``collect``
     - Collect results into a single directory.

Getting help
------------

Every command accepts ``--help`` to print its own options:

.. code-block:: bash

   qm-atlas --help                 # global options and command list
   qm-atlas pipeline --help        # options for the pipeline command
   qm-atlas fast_conformers --help # options for an individual step

Discovering options with ``describe``
-------------------------------------

Many step commands dispatch to interchangeable backends — different property
calculators, geometry optimizers, or conformer generators — each with their own
options. The ``describe`` command lists these backends and their configurable
options so you know what to put in a config file:

.. code-block:: bash

   qm-atlas describe                          # list all option categories
   qm-atlas describe calculators              # list available property calculators
   qm-atlas describe calculators xtb_single_point   # options for one calculator
   qm-atlas describe optimizers jobex         # options for one optimizer

Checking the software environment (``test_software``)
-----------------------------------------------------

``test_software`` is a standalone utility command — it is **not** a workflow
step and plays no part in the ``pipeline``. It runs the wrapper self-tests that
ship with the installed package against a software configuration, so you can
verify that the configured external programs actually work:

.. code-block:: bash

   qm-atlas test_software --software_config /path/to/software_config.yaml
   qm-atlas test_software    # uses QM_ATLAS_SOFTWARE_CONFIG_FILE / defaults

Programs disabled in the configuration are skipped; enabled ones are exercised
against the real binaries. The exit code is the pass/fail signal (useful for
CI), ``--report results.xml`` writes a JUnit report, and any extra arguments are
forwarded to pytest (e.g. ``-k turbomole`` to check a single backend). See
:doc:`software_config` for how program availability is determined.

Global options
--------------

The following options apply to any command:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``--debug``
     - Set logging to debug level.
   * - ``--silent``
     - Set logging to error-only level.
   * - ``--version``
     - Print the version and exit.
   * - ``--software_config``
     - Path to a software configuration YAML file (overrides the
       ``QM_ATLAS_SOFTWARE_CONFIG_FILE`` environment variable). See
       :doc:`software_config`.

Example pipelines
-----------------

The ``cli_examples/`` directory in the repository contains ready-to-run
``pipeline`` configurations that illustrate common end-to-end workflows. Each
file documents, in its header comment, what the pipeline does and how to invoke
it — copy one as a starting point and adapt it to your needs.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Example
     - Description
   * - ``cli_examples/mini_strain.yaml``
     - Minimal strain-energy estimate that runs entirely on the local machine:
       A small conformer ensemble based on rdkit's ETKDGv3, and xtb-based energies.
   * - ``cli_examples/fast_strain.yaml``
     - Strain energy using the fast conformer ensemble as the reference, with
       g-xtb energies and cluster submission.
   * - ``cli_examples/rescoss_strain.yaml``
     - Strain energy using the ReSCoSS conformer ensemble and COSMOtherm
       energies as the reference.
   * - ``cli_examples/rescoss_properties.yaml``
     - ReSCoSS conformer generation followed by COSMOtherm property
       calculations (logP, ΔG_solv, descriptors, PSA).
   * - ``cli_examples/complex_properties.yaml``
     - Multi-property pipeline: ReSCoSS + COSMO properties, Turbomole Fukui and
       NMR shieldings, a Jaguar calculation run on only the lowest-energy
       conformer, and Boltzmann-weighted aggregation. See
       :ref:`single-conformer-expensive`.
   * - ``cli_examples/cocrystal_screening.yaml``
     - Standalone co-crystal screening on the ``.cosmo`` files in an existing
       results directory. See :ref:`screening-existing-results`.
   * - ``cli_examples/solvent_screening.yaml``
     - Standalone solvent (solubility) screening on the ``.cosmo`` files in an
       existing results directory. See :ref:`screening-existing-results`.

All examples are run the same way — point ``--config`` at the file and inject
the input and output paths on the command line. The strain pipelines
additionally take a ``--reference_input_file`` with the reference conformers:

.. code-block:: bash

   qm-atlas pipeline --config cli_examples/rescoss_strain.yaml \
       --input_file+ ./molecules.sdf \
       --reference_input_file+ ./reference_conformers.sdf \
       --results_directory ./results

The ``rescoss_strain`` example in full:

.. literalinclude:: ../cli_examples/rescoss_strain.yaml
   :language: yaml
   :caption: cli_examples/rescoss_strain.yaml

.. _single-conformer-expensive:

Running an expensive calculation on a single conformer
------------------------------------------------------

``cli_examples/complex_properties.yaml`` is a larger pipeline that layers
several property calculations on top of the ReSCoSS + COSMO workflow: Turbomole
Fukui indices and NMR shieldings for every conformer, a Jaguar
hydrogen-abstraction calculation, and a final Boltzmann-weighted aggregation.

Its main point is a common cost-saving pattern: run an expensive method on only
the single most relevant conformer instead of on the whole ensemble. Each
``conformer_properties`` step accepts a ``conformer_selection`` block that filters
the result conformers *before* the calculation runs. Using ``keep_lowest: 1`` on a
per-conformer energy keeps just the global-minimum conformer — here ranked by the
COSMO-RS energy written earlier by the ``cosmotherm_properties`` step:

.. code-block:: yaml

   - command: conformer_properties
     conformer_selection:
       property_filters:
         - property_name: "cosmo_water_E_COSMO+dE+Mu"
           keep_lowest: 1
     conformer_property_options:
       calculation_tasks:
         - backend: "jaguar_h_abstraction_energies"

The same ``conformer_selection`` mechanism accepts energy windows, absolute
bounds, and reference-only sources (see
:class:`~qm_atlas.command_line.utils.conformer_selection.ConformerSelection`),
so it can just as easily restrict a calculation to, say, every conformer within
2 kcal/mol of the minimum.

The pipeline finishes by computing per-conformer Boltzmann weights from the same
COSMO-RS energy and taking Boltzmann-weighted averages of the per-atom Fukui and
NMR properties in an ``aggregate_conformer_properties`` step.

The ``complex_properties`` example in full:

.. literalinclude:: ../cli_examples/property_calculation_workflow.yaml
   :language: yaml
   :caption: cli_examples/property_calculation_workflow.yaml

.. _screening-existing-results:

Screening an existing results directory
---------------------------------------

The co-crystal and solvent screenings both operate on ``.cosmo`` files that were
generated earlier — for example by the ``conformer_properties`` step of
``cli_examples/rescoss_properties.yaml``, which writes COSMO-FINE ``.cosmo``
files for every conformer. Rather than re-running the whole pipeline, the two
screening steps can be run as standalone commands against the results directory
that already contains those files.

Because these are standalone commands (not ``pipeline`` steps), the results
directory is injected through ``--input.results_directory`` instead of
``--results_directory``:

.. code-block:: bash

   qm-atlas cocrystal_screening \
       --config cli_examples/cocrystal_screening.yaml \
       --input.results_directory ./results

   qm-atlas solvent_screening \
       --config cli_examples/solvent_screening.yaml \
       --input.results_directory ./results

.. literalinclude:: ../cli_examples/cocrystal_screening.yaml
   :language: yaml
   :caption: cli_examples/cocrystal_screening.yaml

.. literalinclude:: ../cli_examples/solvent_screening.yaml
   :language: yaml
   :caption: cli_examples/solvent_screening.yaml

Combining a base config with reference solubilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reference solubilities anchor the solvent screening to measured values, turning
COSMOtherm's *relative* screening into an *absolute* (iterative) one. Because
they change from compound to compound, it is convenient to keep the base
screening definition fixed and supply the references separately.

The ``--config`` option can be given more than once: each additional file is
deep-merged over the previous ones. This lets you keep a standard
``solvent_screening.yaml`` and layer a small ``references.yaml`` on top without
editing the base file:

.. code-block:: bash

   qm-atlas solvent_screening \
       --config cli_examples/solvent_screening.yaml \
       --config cli_examples/references.yaml \
       --input.results_directory ./results

The ``references`` list from the second file is injected into the
``solvent_screening_options`` of the first, leaving everything else untouched:

.. literalinclude:: ../cli_examples/references.yaml
   :language: yaml
   :caption: cli_examples/references.yaml

.. note::

   Merging additional ``--config`` files works for standalone commands like
   ``solvent_screening`` because ``references`` lives at the top level of the
   command's options. It cannot be used to inject references into a ``pipeline``
   config, where the screening options are nested inside the ``steps`` list.
