Getting Started
===============

Installation
------------

All dependencies are available on PyPI, so ``qm_atlas`` can be installed directly
from GitHub. `uv <https://docs.astral.sh/uv/>`_ is the recommended installer:

.. code-block:: bash

   git clone https://github.com/Novartis/qm-atlas.git
   cd qm-atlas
   uv sync

To install into an existing environment with ``pip``:

.. code-block:: bash

   pip install "qm_atlas @ git+https://github.com/Novartis/qm-atlas.git"

The documentation dependencies are available through the ``docs`` extra:

.. code-block:: bash

   uv sync --extra docs        # with uv
   pip install -e ".[docs]"    # with pip (editable clone)

.. note::

   ``qm_atlas`` orchestrates external quantum-chemistry and cheminformatics
   programs (xTB, Turbomole, COSMOtherm, Jaguar, and various conformer
   generators). Before running real calculations you must tell ``qm_atlas``
   which of these programs are available and how to invoke them. See
   :doc:`software_config`.

Usage
-----

``qm_atlas`` can be used both from the command line and as a Python library:

* :doc:`cli` — the command-line interface, including the ``pipeline`` command
  and ready-to-run example workflows.
* :doc:`api/index` — the Python API reference (curated highlights plus the full
  module reference).
