qm_atlas
========

``qm_atlas`` is a toolkit for automated quantum-chemistry workflows in
small-molecule drug discovery: conformer generation, quantum-mechanical
property calculations, and COSMO-RS-based thermodynamic predictions.

It can be used in two complementary ways:

* **A flexible command-line interface.** The CLI exposes every calculation
  step as a subcommand and provides a single ``pipeline`` entry point that
  chains these steps into complex, multi-step calculations that are fully
  described in a ``.yaml`` configuration file. See :doc:`cli`.

* **A Python API.** The same functionality is available programmatically. The
  API lets you drive external software through thin *wrappers*, run individual
  calculation steps (*tasks*) with interchangeable backends, and combine those
  steps into higher-level *workflows*. See the :doc:`api/index`.

.. note::

   ``qm_atlas`` orchestrates external programs (xTB, Turbomole, COSMOtherm,
   Jaguar, and various conformer generators). Before running real
   calculations you must tell ``qm_atlas`` which of these programs are
   installed and how to invoke them. See :doc:`software_config`.

Where to go next
----------------

* :doc:`usage` — installation and next steps.
* :doc:`software_config` — connecting ``qm_atlas`` to external software.
* :doc:`cli` — the command-line interface and the ``pipeline`` command.
* :doc:`api/index` — the Python API (highlights + full reference).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   software_config
   cli
   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
