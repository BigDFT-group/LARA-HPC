"""Configuration loading utilities."""

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from pydantic import ValidationError

from laraq.config.models import Config
from laraq.exceptions import (
    ConfigFileNotFoundError,
    ConfigParseError,
    ConfigValidationError,
)


def load_config(config_path: str | Path) -> Config:
    """Load and validate configuration from a TOML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Validated Config object

    Raises:
        ConfigFileNotFoundError: If the config file doesn't exist
        ConfigParseError: If the TOML file is malformed
        ConfigValidationError: If the configuration is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigFileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigParseError(f"Failed to parse TOML file: {e}") from e
    except Exception as e:
        raise ConfigParseError(f"Failed to read config file: {e}") from e

    try:
        config = Config(**data)
        config.config_path = str(config_path)
    except ValidationError as e:
        error_messages = []
        for error in e.errors():
            loc = " -> ".join(str(l) for l in error["loc"])
            msg = error["msg"]
            error_messages.append(f"  {loc}: {msg}")

        raise ConfigValidationError(
            "Configuration validation failed:\n" + "\n".join(error_messages)
        ) from e
    except ValueError as e:
        raise ConfigValidationError(str(e)) from e

    return config
