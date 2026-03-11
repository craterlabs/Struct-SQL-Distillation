#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Loader.

Centralized configuration management for loading settings from INI files.
Handles paths, Azure OpenAI credentials, database settings, and training parameters.

@author: Khushboo Thaker, Yony Bresler, Khalid Eidoo
@license: Crater Labs (C)
"""

import configparser
import os
import logging
from typing import Dict
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Custom exception for configuration related errors."""
    pass


class ConfigLoader:
    """Loads configuration settings from a specified INI file path.

    Prioritizes environment variables, then config file, then default values.
    Set AZURE_OPENAI_API_KEY in the environment to avoid storing credentials in config.ini.
    """

    def __init__(self, full_config_file_path: str):
        """Load configuration from an INI file.

        Args:
            full_config_file_path: Absolute or relative path to the config.ini file.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ConfigurationError: If the file cannot be parsed.
        """
        self.config_file_path = full_config_file_path
        self.config = configparser.ConfigParser()
        self._project_root = self._determine_project_root()
        self._load_config()

    def _get_section_dict(self, section: str) -> Dict[str, str]:
        """Return all key-value pairs in a config section as a plain dict."""
        if section not in self.config:
            raise ConfigurationError(f"Section '{section}' not found in the configuration file.")
        return dict(self.config.items(section))

    def _determine_project_root(self) -> str:
        """Determines the project root directory."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(current_dir)

    def _load_config(self):
        """Loads the configuration from the INI file."""
        if not os.path.exists(self.config_file_path):
            raise FileNotFoundError(
                f"Configuration file '{self.config_file_path}' not found. "
                "Ensure it's in the project root or specified correctly."
            )
        try:
            self.config.read(self.config_file_path)
            logger.info(f"Configuration loaded from {self.config_file_path}")
        except configparser.Error as e:
            raise ConfigurationError(f"Error reading config file {self.config_file_path}: {e}") from e
        except Exception as e:
            raise ConfigurationError(f"Unexpected error loading config file {self.config_file_path}: {e}") from e

    def _get_setting(self, section: str, option: str, default=None, is_secret: bool = False, type_caster=str):
        """Helper to get a setting from the config file, or use a default.
        
        Args:
            section: INI file section name.
            option: Option name within the section.
            default: Default value if not found.
            is_secret: If True, logs will mask the value.
            type_caster: Function to cast the value.

        Returns:
            Any: The retrieved and type-casted configuration value.

        Raises:
            ConfigurationError: If a required setting is not found or has an invalid format.
        """
        if self.config.has_option(section, option):
            try:
                value_from_config = self.config.get(section, option)
                log_message = f"Loaded {'secret' if is_secret else 'setting'} '{option}' from config file."
                logger.debug(log_message)
                return type_caster(value_from_config)
            except ValueError as e:
                raise ConfigurationError(
                    f"Config option '{option}' in section '{section}' has invalid format for type {type_caster.__name__}: {e}"
                ) from e
            except (configparser.NoOptionError, configparser.NoSectionError) as e:
                raise ConfigurationError(
                    f"Missing option or section for '{option}' in '{section}' of config file: {e}"
                ) from e
        elif default is not None:
            logger.debug(f"Using default value for setting '{option}'.")
            return default
        else:
            raise ConfigurationError(
                f"Required setting '{option}' not found in config file, and no default provided."
            )

    def _resolve_and_validate_path(self, relative_path: str, is_directory: bool = False) -> str:
        """Resolves a relative path to an absolute path and validates its existence and type.
        
        Args:
            relative_path: The path from config.ini (relative to project root).
            is_directory: True if the path is expected to be a directory, False for a file.

        Returns:
            str: The absolute, validated path.

        Raises:
            ConfigurationError: If the path is invalid or does not exist.
        """
        absolute_path = os.path.join(self._project_root, relative_path)

        if not os.path.exists(absolute_path):
            raise ConfigurationError(f"Path '{absolute_path}' does not exist. Check config.ini.")
        
        if is_directory and not os.path.isdir(absolute_path):
            raise ConfigurationError(f"Path '{absolute_path}' is not a directory. Check config.ini.")
        elif not is_directory and not os.path.isfile(absolute_path):
            raise ConfigurationError(f"Path '{absolute_path}' is not a file. Check config.ini.")
            
        return absolute_path

    def get_azure_openai_connection_settings(self) -> dict:
        """Retrieve Azure OpenAI connection settings.

        Returns:
            Dict with 'endpoint', 'api_key', 'api_version' keys.

        Raises:
            ConfigurationError: If API key is empty or missing.
        """
        endpoint = self._get_setting('azure_openai', 'azure_endpoint')
        # Env var takes precedence over config file so credentials can be injected
        # in CI/CD and containerised environments without modifying config.ini.
        api_key = (
            os.environ.get("AZURE_OPENAI_API_KEY")
            or self._get_setting('azure_openai', 'api_key', is_secret=True, default="")
        )
        api_version = self._get_setting('azure_openai', 'api_version', default="2024-02-01")

        if not api_key:
            raise ConfigurationError(
                "Azure OpenAI API key not found. Set the AZURE_OPENAI_API_KEY environment variable "
                "or add 'api_key' to the [azure_openai] section in config.ini."
            )

        parsed = urlparse(endpoint)
        if parsed.scheme != "https":
            raise ConfigurationError("azure_endpoint must use HTTPS (e.g. https://your-resource.openai.azure.com/).")
        if not parsed.netloc:
            raise ConfigurationError("azure_endpoint must be a valid URL with a host.")

        return {
            "endpoint": endpoint,
            "api_key": api_key,
            "api_version": api_version
        }

    def get_azure_openai_deployment_names(self) -> dict:
        """Retrieve Azure OpenAI deployment names.

        Returns:
            Dict with 'chat_deployment_name' key.

        Raises:
            ConfigurationError: If deployment name is empty.
        """
        chat_deployment_name = self._get_setting('azure_openai', 'chat_deployment_name')

        if not chat_deployment_name:
            raise ConfigurationError("Chat deployment name cannot be empty.")

        return {
            "chat_deployment_name": chat_deployment_name,
        }

    def get_llm_parameters(self) -> dict:
        """Retrieve optional LLM parameters.

        Returns:
            Dict with 'temperature', 'max_tokens', 'top_p', 'seed'.
        """
        temperature = self._get_setting('azure_openai', 'temperature', default=None, type_caster=float)
        max_tokens = self._get_setting('azure_openai', 'max_tokens', default=500, type_caster=int)
        top_p = self._get_setting('azure_openai', 'top_p', default=1.0, type_caster=float)
        seed = self._get_setting('azure_openai', "seed", default=42, type_caster=int)
        return {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "seed": seed
        }

    def get_database_settings(self) -> dict:
        """Retrieve database configuration settings.

        Returns:
            Dict with 'db_type' and 'database_root_path' keys.

        Raises:
            ConfigurationError: If database section or options are missing.
        """
        try:
            db_type = self._get_setting('database', 'db_type')
            database_root_path_relative = self._get_setting('database', 'database_root_path')
            database_root_path = self._resolve_and_validate_path(database_root_path_relative, is_directory=True)

            return {
                "db_type": db_type,
                "database_root_path": database_root_path
            }
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            raise ConfigurationError(f"Missing database config: {e}. Ensure '[database]' section and options exist.") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error reading database config: {e}") from e

    def get_database_train_settings(self) -> dict:
        """Retrieve training database configuration settings.

        Returns:
            Dict with 'db_type' and 'database_root_path' for training databases.

        Raises:
            ConfigurationError: If database section or options are missing.
        """
        try:
            db_type = self._get_setting('database', 'db_type')
            database_train_root_path_relative = self._get_setting('database', 'database_train_root_path')
            database_train_root_path = self._resolve_and_validate_path(database_train_root_path_relative, is_directory=True)

            return {
                "db_type": db_type,
                "database_root_path": database_train_root_path,
            }
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            raise ConfigurationError(f"Missing database config: {e}. Ensure '[database]' section and options exist.") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error reading database config: {e}") from e

    def get_bird_dataset_paths(self) -> dict:
        """Retrieve paths to BIRD dataset JSON files.

        Returns:
            Dict with 'dev_tables_json_path', 'dev_questions_json_path', 'dev_tied_append_json_path'.

        Raises:
            ConfigurationError: If BIRD dataset paths section is missing.
        """
        try:
            dev_tables_json_path_relative = self._get_setting('bird_dataset_paths', 'dev_tables_json_path')
            dev_questions_json_path_relative = self._get_setting('bird_dataset_paths', 'dev_questions_json_path')
            dev_tied_append_json_path_relative = self._get_setting('bird_dataset_paths', 'dev_tied_append_json_path')

            dev_tables_json_path = self._resolve_and_validate_path(dev_tables_json_path_relative, is_directory=False)
            dev_questions_json_path = self._resolve_and_validate_path(dev_questions_json_path_relative, is_directory=False)
            dev_tied_append_json_path = self._resolve_and_validate_path(dev_tied_append_json_path_relative, is_directory=False)
            
            logger.debug(f"Resolved dev_tables_json_path: {dev_tables_json_path}")

            return {
                "dev_tables_json_path": dev_tables_json_path,
                "dev_questions_json_path": dev_questions_json_path,
                "dev_tied_append_json_path": dev_tied_append_json_path
            }
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            raise ConfigurationError(f"Missing BIRD dataset config: {e}. Ensure '[bird_dataset_paths]' section and options exist.") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error reading BIRD dataset paths config: {e}") from e

    def get_bird_train_dataset_paths(self) -> dict:
        """Retrieve paths to BIRD training dataset JSON files.

        Returns:
            Dict with 'train_tables_json_path', 'train_questions_json_path',
            'bird_train_db_root', and optionally 'train_questions_level_json_path'
            if configured in [bird_training_paths].

        Raises:
            ConfigurationError: If the [bird_training_paths] section is missing
                or any required path does not exist on disk.
        """
        # Get all parameters from the 'bird_training_paths' section
        params = self._get_section_dict('bird_training_paths')

        # Resolve relative paths to absolute using project root
        resolved = {
            "train_tables_json_path": self._resolve_and_validate_path(params.get('train_tables_json_path'), is_directory=False),
            "train_questions_json_path": self._resolve_and_validate_path(params.get('train_questions_json_path'), is_directory=False),
            "bird_train_db_root": self._resolve_and_validate_path(params.get('bird_train_db_root'), is_directory=True)
        }
        level_path = params.get('train_questions_level_json_path')
        if level_path:
            resolved["train_questions_level_json_path"] = os.path.normpath(
                os.path.join(self._project_root, level_path)
            )
        return resolved



    def get_general_paths(self) -> Dict[str, str]:
        """Retrieve general system paths from the [paths] config section.

        Returns:
            Dict with 'base_data_dir', 'base_output_dir', 'model_cache_dir',
            'kd_data_dir' keys.

        Raises:
            ConfigurationError: If the [paths] section is missing or a required
                setting cannot be read.
        """
        return {
            "base_data_dir": self._get_setting("paths", "base_data_dir"),
            "base_output_dir": self._get_setting("paths", "base_output_dir"),
            "model_cache_dir": self._get_setting("paths", "model_cache_dir"),
            "kd_data_dir": self._get_setting("paths", "kd_data_dir", default="./data/kd_data")
        }

    def get_distillation_parameters(self) -> dict:
        """Retrieve and convert knowledge distillation parameters from the config file.

        Returns:
            Dict with 'lora_r', 'lora_alpha', 'learning_rate', 'batch_size', 'epochs',
            'quantization', 'max_length', 'max_new_tokens', 'train_split_percentage',
            'val_split_percentage'.

        Raises:
            ConfigurationError: If the [distillation_parameters] section is missing.
        """
        params = self._get_section_dict('distillation_parameters')

        # Convert numeric parameters to appropriate types
        params['lora_r'] = int(params.get('lora_r', 16))
        params['lora_alpha'] = int(params.get('lora_alpha', 32))
        params['learning_rate'] = float(params.get('learning_rate', 2e-4))
        params['batch_size'] = int(params.get('batch_size', 1))
        params['epochs'] = int(params.get('epochs', 3))
        params['quantization'] = int(params.get('quantization', 4))
        params['max_length'] = int(params.get('max_length', 15000))
        params['max_new_tokens'] = int(params.get('max_new_tokens', 2000))
        params['train_split_percentage'] = float(params.get('train_split_percentage', 0.8))
        params['val_split_percentage'] = float(params.get('val_split_percentage', 0.2))

        return params
    
    def get_training_parameters(self) -> dict:
        """Retrieve training parameters from the [training] section.

        Returns:
            Dict with 'gradient_accumulation_steps', 'logging_steps', 'eval_steps',
            'save_steps', 'warmup_steps', 'early_stopping_patience',
            'early_stopping_threshold', 'seed', 'optim'.

        Raises:
            ConfigurationError: If the [training] section is missing.
        """
        params = self._get_section_dict('training')

        params['gradient_accumulation_steps'] = int(params.get('gradient_accumulation_steps', 6))
        params['logging_steps'] = int(params.get('logging_steps', 10))
        params['eval_steps'] = int(params.get('eval_steps', 20))
        params['save_steps'] = int(params.get('save_steps', 20))
        params['warmup_steps'] = int(params.get('warmup_steps', 10))
        params['early_stopping_patience'] = int(params.get('early_stopping_patience', 8))
        params['early_stopping_threshold'] = float(params.get('early_stopping_threshold', 0.001))
        params['seed'] = int(params.get('seed', 42))
        params['optim'] = params.get('optim', 'paged_adamw_32bit')

        return params

    def get_bird_kd_dataset_paths(self) -> dict:
        """Retrieve paths to BIRD knowledge distillation dataset files.

        Returns:
            Dict with 'train_tables_json_path', 'train_questions_json_path', 'bird_train_db_root'.

        Raises:
            ConfigurationError: If the [bird_kd_paths] section is missing or any
                required path does not exist on disk.
        """
        params = self._get_section_dict('bird_kd_paths')

        # Resolve relative paths to absolute using project root
        return {
            "train_tables_json_path": self._resolve_and_validate_path(params.get('train_tables_json_path'), is_directory=False),
            "train_questions_json_path": self._resolve_and_validate_path(params.get('train_questions_json_path'), is_directory=False),
            "bird_train_db_root": self._resolve_and_validate_path(params.get('bird_train_db_root'), is_directory=True)
        }
