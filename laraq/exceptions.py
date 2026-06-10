"""Custom exceptions for laraq."""


class LaraqError(Exception):
    """Base exception for laraq."""

    pass


class ConfigError(LaraqError):
    """Exception raised for configuration errors."""

    pass


class ConfigFileNotFoundError(ConfigError):
    """Exception raised when configuration file is not found."""

    pass


class ConfigParseError(ConfigError):
    """Exception raised when configuration file cannot be parsed."""

    pass


class ConfigValidationError(ConfigError):
    """Exception raised when configuration validation fails."""

    pass
