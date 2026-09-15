Workflows
=========

Main Entry Points
-----------------

These are the primary workflow functions that most users should use:

.. autofunction:: qm_atlas.workflows.generate_fast_conformers
.. autofunction:: qm_atlas.workflows.generate_rescoss_conformers
.. autofunction:: qm_atlas.workflows.generate_tautomers
.. autofunction:: qm_atlas.workflows.run_constrained_optimization

Detailed Module Documentation
-----------------------------

.. automodule:: qm_atlas.workflows.fast_conformers
   :members:
   :undoc-members:

.. autodata:: qm_atlas.workflows.fast_conformers.FAST_DEFAULT_OPTIONS
.. autodata:: qm_atlas.workflows.fast_conformers.DEFAULT_OPTIMIZATION_CONFIG
.. autodata:: qm_atlas.workflows.fast_conformers.DEFAULT_FINAL_OPTIMIZATION_CONFIG

.. automodule:: qm_atlas.workflows.rescoss_conformers
   :members:
   :undoc-members:

.. autodata:: qm_atlas.workflows.rescoss_conformers.RESCOSS_DEFAULT_CONF_GENS
.. autodata:: qm_atlas.workflows.rescoss_conformers.XTB_OPTIMIZATION_CONFIG
.. autodata:: qm_atlas.workflows.rescoss_conformers.DEFAULT_FINAL_OPTIMIZATION_CONFIG
.. autodata:: qm_atlas.workflows.rescoss_conformers.RMSD_THRESHOLD
.. autodata:: qm_atlas.workflows.rescoss_conformers.DEFAULT_NUM_CLUSTERS
.. autodata:: qm_atlas.workflows.rescoss_conformers.DEFAULT_NUM_PER_CLUSTER

.. automodule:: qm_atlas.workflows.score_tautomers
   :members:
   :undoc-members:

.. autodata:: qm_atlas.workflows.score_tautomers.TAUTOMER_TM_SP_CONFIG

.. automodule:: qm_atlas.workflows.optimize_constrained
   :members:
   :undoc-members:

.. autodata:: qm_atlas.workflows.optimize_constrained.DEFAULT_OPTIMIZATION_CONFIG
