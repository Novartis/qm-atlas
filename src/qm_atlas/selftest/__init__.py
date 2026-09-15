"""Packaged software self-test for qm_atlas.

Exposes the ``qm-atlas test_software`` command, which runs the wrapper test suite
(shipped under :mod:`qm_atlas.selftest`'s ``suite/`` directory) against a software
configuration. Tools that are unavailable per the configuration are skipped, so
the command doubles as an environment / license health check on any install.
"""
