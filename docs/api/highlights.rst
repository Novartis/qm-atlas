Key API
=======

This page collects the most commonly used functions of the Python API,
grouped by module. It is a curated entry point — for the complete, generated
documentation of every module see the :doc:`index` (Full API Reference).

Tasks
-----

Main entry points for individual calculation tasks. Wrappers for external programs are
grouped here to provide a uniform interface. For each task, there are functions operating
on a single conformation, all conformers of a molecule, or a batch of molecules. Only the
functions that operate on a single molecule are listed here. This layer also has the
configuration classes that select the backend and its options, these options classes are
passed through to the workflows and the command-line interface.

Conformer generation
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: qm_atlas.tasks.conformer_generation.generate_conformers
   :no-index:

.. autopydantic_model:: qm_atlas.tasks.conformer_generation.ConformerGenerationOptions
   :no-index:
.. autopydantic_model:: qm_atlas.tasks.conformer_generation.RdkitOptions
   :no-index:
.. autopydantic_model:: qm_atlas.tasks.conformer_generation.MacromodelOptions
   :no-index:
.. autopydantic_model:: qm_atlas.tasks.conformer_generation.OmegaOptions
   :no-index:
.. autopydantic_model:: qm_atlas.tasks.conformer_generation.MoeOptions
   :no-index:
.. autopydantic_model:: qm_atlas.tasks.conformer_generation.CorinaOptions
   :no-index:
.. autopydantic_model:: qm_atlas.tasks.conformer_generation.MacrocycleOptions
   :no-index:

Geometry optimization
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: qm_atlas.tasks.optimize.optimize_mol
   :no-index:

.. autopydantic_model:: qm_atlas.tasks.optimize.XtbOptions
   :no-index:
.. autopydantic_model:: qm_atlas.tasks.optimize.JobexOptions
   :no-index:
.. autopydantic_model:: qm_atlas.tasks.optimize.XtbTurbomoleOptions
   :no-index:

Property calculation
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: qm_atlas.tasks.calculate_properties.calculate_property_mol
   :no-index:

.. autopydantic_model:: qm_atlas.tasks.calculate_properties.XtbSinglePointOptions
   :no-index:
.. autopydantic_model:: qm_atlas.tasks.calculate_properties.TurbomoleSinglePointOptions
   :no-index:
.. autopydantic_model:: qm_atlas.tasks.calculate_properties.TurbomoleFreehOptions
   :no-index:
.. autopydantic_model:: qm_atlas.tasks.calculate_properties.TurbomoleFukuiOptions
   :no-index:
.. autopydantic_model:: qm_atlas.tasks.calculate_properties.TurbomoleNmrShieldingOptions
   :no-index:
.. autopydantic_model:: qm_atlas.tasks.calculate_properties.TurbomoleVcdOptions
   :no-index:
.. autopydantic_model:: qm_atlas.tasks.calculate_properties.JaguarHydrogenAbstractionOptions
   :no-index:
.. autopydantic_model:: qm_atlas.tasks.calculate_properties.JaguarQmDescriptorsOptions
   :no-index:

COSMO properties
~~~~~~~~~~~~~~~~~

.. autofunction:: qm_atlas.tasks.cosmo_properties.run_cosmo_calculation
   :no-index:

.. autopydantic_model:: qm_atlas.tasks.cosmo_properties.CosmoLogPConfig
   :no-index:
   :inherited-members: BaseModel
.. autopydantic_model:: qm_atlas.tasks.cosmo_properties.CosmoDeltaGConfig
   :no-index:
   :inherited-members: BaseModel
.. autopydantic_model:: qm_atlas.tasks.cosmo_properties.CosmoDescriptorsConfig
   :no-index:
   :inherited-members: BaseModel
.. autopydantic_model:: qm_atlas.tasks.cosmo_properties.CosmoPsaConfig
   :no-index:

Protonation
~~~~~~~~~~~

.. autofunction:: qm_atlas.tasks.protonation.run_protonation
   :no-index:

.. autopydantic_model:: qm_atlas.tasks.protonation.MokaBlabberConfig
   :no-index:
.. autopydantic_model:: qm_atlas.tasks.protonation.MokaIonicSpeciesConfig
   :no-index:
.. autopydantic_model:: qm_atlas.tasks.protonation.MokaSingleChargeConfig
   :no-index:

Tautomers
~~~~~~~~~

.. autofunction:: qm_atlas.tasks.tautomer_generation.generate_tautomers
   :no-index:



Workflows
---------

High-level entry points that combine several tasks into a complete calculation.

ReSCoSS conformers
~~~~~~~~~~~~~~~~~~~

.. autofunction:: qm_atlas.workflows.generate_rescoss_conformers
   :no-index:

Fast conformers
~~~~~~~~~~~~~~~

.. autofunction:: qm_atlas.workflows.generate_fast_conformers
   :no-index:

Tautomers
~~~~~~~~~

.. autofunction:: qm_atlas.workflows.generate_tautomers
   :no-index:

Constrained optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: qm_atlas.workflows.run_constrained_optimization
   :no-index:


Wrappers
--------

Thin interfaces to the external quantum-chemistry and cheminformatics programs.
Only the main entry points that the tasks build on are listed here, grouped by
program.

xtb
~~~

.. autofunction:: qm_atlas.wrappers.xtb.optimize_geometry
   :no-index:
.. autofunction:: qm_atlas.wrappers.xtb.calculate_single_point
   :no-index:

Turbomole
~~~~~~~~~

.. autofunction:: qm_atlas.wrappers.turbomole.single_point.run_ridft
   :no-index:
.. autofunction:: qm_atlas.wrappers.turbomole.jobex.run_jobex
   :no-index:
.. autofunction:: qm_atlas.wrappers.turbomole.xtb_opt.run_xtb_tm_optimization
   :no-index:
.. autofunction:: qm_atlas.wrappers.turbomole.freeh.calculate_freeh
   :no-index:
.. autofunction:: qm_atlas.wrappers.turbomole.fukui.calculate_fukui
   :no-index:
.. autofunction:: qm_atlas.wrappers.turbomole.nmr_shielding.calculate_nmr_shieldings
   :no-index:
.. autofunction:: qm_atlas.wrappers.turbomole.vcd.calculate_vcd
   :no-index:

COSMOtherm
~~~~~~~~~~

.. autofunction:: qm_atlas.wrappers.cosmotherm.interface.calculate_cosmotherm
   :no-index:
.. autofunction:: qm_atlas.wrappers.cosmotherm.cosmo_tasks.calculate_logp
   :no-index:
.. autofunction:: qm_atlas.wrappers.cosmotherm.cosmo_tasks.calculate_delta_g
   :no-index:
.. autofunction:: qm_atlas.wrappers.cosmotherm.cosmo_tasks.calculate_descriptors
   :no-index:
.. autofunction:: qm_atlas.wrappers.cosmotherm.cosmo_tasks.calculate_descriptors_df
   :no-index:
.. autofunction:: qm_atlas.wrappers.cosmotherm.cosmo_solubility.calculate_solubility
   :no-index:
.. autofunction:: qm_atlas.wrappers.cosmotherm.cosmo_solubility.calculate_solubility_mixture
   :no-index:
.. autofunction:: qm_atlas.wrappers.cosmotherm.cosmo_solubility.parse_solubility_data
   :no-index:
.. autofunction:: qm_atlas.wrappers.cosmotherm.cosmo_cocrystal.screen_cocrystals
   :no-index:
.. autofunction:: qm_atlas.wrappers.cosmotherm.cpsa.extract_psa_from_cosmo_file
   :no-index:

Jaguar
~~~~~~

.. autofunction:: qm_atlas.wrappers.jaguar.bond_dissociation.get_hydrogen_abstraction_energies
   :no-index:
.. autofunction:: qm_atlas.wrappers.jaguar.qm_descriptors.calculate_descriptors
   :no-index:

MoKa
~~~~

.. autofunction:: qm_atlas.wrappers.moka.blabber.get_protonation
   :no-index:
.. autofunction:: qm_atlas.wrappers.moka.process.generate_ionic_species
   :no-index:
.. autofunction:: qm_atlas.wrappers.moka.process.generate_single_charge_ions
   :no-index:

MacroModel
~~~~~~~~~~

.. autofunction:: qm_atlas.wrappers.macromodel.generate_conformers
   :no-index:
.. autofunction:: qm_atlas.wrappers.macromodel.generate_macrocycle_conformers
   :no-index:

MOE
~~~

.. autofunction:: qm_atlas.wrappers.moe.generate_conformers
   :no-index:

OpenEye Omega
~~~~~~~~~~~~~

.. autofunction:: qm_atlas.wrappers.openeye.generate_conformers_molobj
   :no-index:

CORINA
~~~~~~

.. autofunction:: qm_atlas.wrappers.corina.generate_conformers_molobj
   :no-index:

UNICON
~~~~~~

.. autofunction:: qm_atlas.wrappers.unicon.get_tautomers
   :no-index:
