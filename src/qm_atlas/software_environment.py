import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from hpc_funcs import lmod

DEFAULT_SOFTWARE_CONFIG_FILE = Path(__file__).parent / "resources" / "default_software_config.yaml"

SOFTWARE_CONFIG_ENV_VAR = "QM_ATLAS_SOFTWARE_CONFIG_FILE"


def get_python_activation_command() -> str | None:
    """Auto-detect the activation command for the currently active Python virtual environment.

    Checks ``VIRTUAL_ENV`` (set by any activated venv/virtualenv) and falls back to
    ``sys.prefix``.  Returns ``None`` when no virtual environment is detected (e.g.
    running in the system Python).
    """
    # VIRTUAL_ENV is the most reliable indicator – it is set by `source .../activate`
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        activate_script = Path(virtual_env) / "bin" / "activate"
        if activate_script.exists():
            return f"source {activate_script}"

    # Fall back to sys.prefix (covers uv-managed envs that may not export VIRTUAL_ENV)
    activate_script = Path(sys.prefix) / "bin" / "activate"
    if activate_script.exists():
        return f"source {activate_script}"

    return None


class _UseDefaultModel:
    """Sentinel value indicating to use the default model from config"""


USE_DEFAULT_MODEL = _UseDefaultModel()

MOKA_MODEL_KEY = "moka_model"
COSMO_SOLVENTS_DIR_KEY_PREFIX = "cosmo_solvents_dir"


def get_cosmo_solvents_dir_key(level: str):
    """Get the appropriate global settings key for COSMO solvents directories based on level.

    Args:
        level: The cosmotherm level of theory (e.g. "bp-tzvpd", "bp-tzvp", "bp-svp", "dmol3-pbe")

    Returns:
        The global settings key to use for retrieving the solvents directories for the given level
    """
    return f"{COSMO_SOLVENTS_DIR_KEY_PREFIX}_{level}"


@dataclass(kw_only=True)
class SoftwareEnvironment:
    commands: dict[str, str]
    environment_variables: dict[str, str]
    thread_control_variables: list[str]
    available: bool = False


def _merge_software_entry(default: dict, override: dict) -> dict:
    """Merge an override software entry into a default entry at the field level.

    Dict-valued fields (``commands``, ``environment_variables``) are merged
    key-by-key so individual commands can be overridden without re-specifying
    all of them.  All other fields (``available``, ``module``, ``modulepaths``,
    ``thread_control_env_vars``, etc.) are replaced wholesale by the override
    value.
    """
    result = dict(default)
    for key, value in override.items():
        if key in ("commands", "environment_variables") and isinstance(result.get(key), dict):
            result[key] = {**result[key], **value}
        else:
            result[key] = value
    return result


def _merge_software_configs(default: dict, user: dict) -> dict:
    """Merge user software environment dict into the defaults.

    Software entries present in *user* are merged field-by-field into the
    corresponding default entry (see :func:`_merge_software_entry`).  Entries
    only present in *user* (unknown to the defaults) are included as-is.
    """
    result = dict(default)
    for software, user_entry in user.items():
        if software in result:
            result[software] = _merge_software_entry(result[software], user_entry)
        else:
            result[software] = user_entry
    return result


def get_env_from_modules(
    module_names: str | list[str], module_paths: list[str] | None = None
) -> dict[str, str]:

    if module_paths is None:
        module_paths = []

    modulepath_env = {}
    modulepath = os.environ.get("MODULEPATH", "")
    for path in module_paths:
        modulepath = f"{path}:{modulepath}"
    modulepath_env["MODULEPATH"] = modulepath

    environment_variables = {}
    module_names = [module_names] if isinstance(module_names, str) else module_names
    for module_name in module_names:

        update_environment_variables = lmod.get_load_environment(
            module_name,
            env={**os.environ, **modulepath_env, **environment_variables},
        )
        environment_variables.update(update_environment_variables)

    return environment_variables


class EnvironmentManager:
    # Allowed keys for submission configuration
    ALLOWED_SUBMISSION_KEYS = {
        "module_purge",
        "module_use",
        "module_load",
        "command_prepend",
    }

    ALLOWED_GLOBAL_SETTINGS_KEYS = {
        "moka_model",
        "cosmo_solvents_dir_bp-tzvpd",
        "cosmo_solvents_dir_bp-tzvp",
        "cosmo_solvents_dir_bp-svp",
        "cosmo_solvents_dir_dmol3-pbe",
    }

    def __init__(self, config_file: Path | None = None):

        self.software_config = self._load_software_environments(config_file)
        self.submission_config = self._load_submission_config(config_file)
        self.global_settings = self._load_global_settings(config_file)

    def _load_software_environments(
        self, config_file: Path | None
    ) -> dict[str, SoftwareEnvironment]:

        # Load built-in defaults
        with open(DEFAULT_SOFTWARE_CONFIG_FILE, "r", encoding="utf-8") as f:
            default_dict = yaml.safe_load(f)
        default_software_env_dict = default_dict.get("software_environments", {})

        # Merge user overrides on top of defaults
        if config_file is not None:
            with open(config_file, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
            user_software_env_dict = config_dict.get("software_environments", {})
            software_env_dict = _merge_software_configs(
                default_software_env_dict, user_software_env_dict
            )
        else:
            software_env_dict = default_software_env_dict

        software_config = dict()

        for software, instructions in software_env_dict.items():
            if "commands" not in instructions:
                raise ValueError(f"Software environment '{software}' missing 'commands' in config")

            commands = instructions["commands"]
            environment_variables = {}

            if "module" in instructions:

                environment_variables = get_env_from_modules(
                    module_names=instructions["module"],
                    module_paths=instructions.get("modulepaths", None),
                )

            if "environment_variables" in instructions:
                environment_variables = {
                    **environment_variables,
                    **instructions["environment_variables"],
                }

            if "thread_control_env_vars" in instructions:
                thread_control_env_vars = instructions["thread_control_env_vars"]
            else:
                thread_control_env_vars = []

            software_config[software] = SoftwareEnvironment(
                commands=commands,
                environment_variables=environment_variables,
                thread_control_variables=thread_control_env_vars,
                available=instructions.get("available", False),
            )

        return software_config

    def _load_submission_config(self, config_file: Path | None) -> dict[str, Any]:

        if config_file is None:
            submission_config_data = {}
        else:
            with open(config_file, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
            submission_config_data = config_dict.get("submission", {})

        # Initialize with None values for all allowed keys
        submission_config = {}

        # Update with values from config, but only accept allowed keys
        for key, value in submission_config_data.items():
            if key in self.ALLOWED_SUBMISSION_KEYS:
                submission_config[key] = value

        # If command_prepend is not explicitly configured, auto-detect from the
        # currently active Python virtual environment so users do not need to
        # hard-code their personal venv path in the config file.
        if "command_prepend" not in submission_config:
            auto_cmd = get_python_activation_command()
            if auto_cmd is not None:
                submission_config["command_prepend"] = auto_cmd

        return submission_config

    def _load_global_settings(self, config_file: Path | None) -> dict[str, Any]:

        if config_file is None:
            global_settings_data = {}
        else:
            with open(config_file, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
            global_settings_data = config_dict.get("global_settings", {})

        # Initialize with None values for all allowed keys
        global_settings = {}

        # Update with values from config, but only accept allowed keys
        for key, value in global_settings_data.items():
            if key in self.ALLOWED_GLOBAL_SETTINGS_KEYS:
                global_settings[key] = value

        return global_settings

    def software_available(self, software: str) -> bool:
        if not software in self.software_config:
            return False
        return self.software_config[software].available

    def command_available(self, software: str, command_name: str) -> bool:
        if not self.software_available(software):
            return False

        return command_name in self.software_config[software].commands

    def get_command(self, software: str, command_name: str) -> str:

        if not self.software_available(software):
            raise OSError(f"Software environment '{software}' not available")

        if command_name not in self.software_config[software].commands:
            raise OSError(f"Command '{command_name}' not found for software '{software}'")

        return self.software_config[software].commands[command_name]

    def get_environment_variables(self, software: str, n_cores: int = 1) -> dict[str, str]:

        if not self.software_available(software):
            raise OSError(f"Software environment '{software}' not available")

        specific_environment = self.software_config[software].environment_variables
        thread_control_vars = self.software_config[software].thread_control_variables
        for var in thread_control_vars:
            specific_environment[var] = str(n_cores)
        return specific_environment

    def get_run_environment(
        self,
        software: str,
        n_cores: int = 1,
    ) -> dict[str, str]:
        """
        Get the complete runtime environment for a software package.

        Args:
            software: Name of the software environment

        Returns:
            Complete environment dictionary combining system env + software env + thread control
        """

        # Start with system environment
        env = dict(os.environ)

        # Add software-specific environment variables
        software_env = self.get_environment_variables(software, n_cores=n_cores)
        env.update(software_env)

        return env

    def get_submission_config(self) -> dict[str, Any]:
        return self.submission_config

    def get_global_settings(self) -> dict[str, Any]:
        return self.global_settings


def get_config_file() -> Path | None:
    """Return the user-supplied software config file path, or ``None`` if the
    environment variable :data:`SOFTWARE_CONFIG_ENV_VAR` is not set.

    Raises :class:`EnvironmentError` when the variable *is* set but points to a
    file that does not exist, so typos are caught early.
    """
    if SOFTWARE_CONFIG_ENV_VAR not in os.environ:
        return None

    config_file = Path(os.environ[SOFTWARE_CONFIG_ENV_VAR])
    if config_file.is_file():
        return config_file

    raise OSError(f"Config file specified in '{SOFTWARE_CONFIG_ENV_VAR}' not found: {config_file}")


def load_environment_manager(
    config_file: Path | None = None,
) -> EnvironmentManager:

    if config_file is None:
        config_file = get_config_file()
    return EnvironmentManager(config_file)


class SoftwareConfig:
    def __init__(self):
        self._environment_manager: EnvironmentManager | None = None

    def get_environment_manager(self) -> EnvironmentManager:
        if self._environment_manager is None:
            self._environment_manager = load_environment_manager()
        return self._environment_manager

    def reload_environment_manager(self, config_file: Path | None = None) -> None:
        self._environment_manager = load_environment_manager(config_file)


SOFTWARE_CONFIG = SoftwareConfig()


def get_moka_model() -> Path | None:
    """Get the default MoKa model path from global settings.

    Returns:
        Path to the model file or None if not configured
    """
    global_settings = SOFTWARE_CONFIG.get_environment_manager().get_global_settings()
    model_path = global_settings.get(MOKA_MODEL_KEY, None)
    return Path(model_path) if model_path else None


def get_cosmo_solvents_dirs(level: str) -> list[Path]:
    """Get the appropriate COSMO solvents directories for a given cosmotherm level.

    Args:
        level: The cosmotherm level of theory (e.g. "bp-tzvpd", "bp-tzvp", "bp-svp", "dmol3-pbe")

    Returns:
        List of paths to the solvents directories appropriate for the given level
    """
    key = get_cosmo_solvents_dir_key(level)
    global_settings = SOFTWARE_CONFIG.get_environment_manager().get_global_settings()
    dir_paths = global_settings.get(key, None)
    if dir_paths is None:
        return []
    return [Path(p) for p in dir_paths]
