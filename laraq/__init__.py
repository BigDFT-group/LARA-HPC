"""laraq - Agent-driven code generation system.

This package provides a CLI tool and configuration management
for multi-provider AI agent systems.
"""

from laraq.agent import BaseAgent, Message, State
from laraq.agents import (
    NaiveCodeAgent,
    DirectorAgent,
    RemoteExecutorAgent,
    NaiveResearchAgent,
)
from laraq.config.loader import load_config
from laraq.config.models import (
    AnthropicConfig,
    Config,
    OllamaConfig,
    OpenAICompatibleConfig,
)
from laraq.exceptions import (
    ConfigError,
    ConfigFileNotFoundError,
    ConfigParseError,
    ConfigValidationError,
    LaraqError,
)

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "load_config",
    "Config",
    "OllamaConfig",
    "OpenAICompatibleConfig",
    "AnthropicConfig",
    # Exceptions
    "LaraqError",
    "ConfigError",
    "ConfigFileNotFoundError",
    "ConfigParseError",
    "ConfigValidationError",
    # Agent framework
    "BaseAgent",
    "Message",
    "State",
    # Agents
    "NaiveResearchAgent",
    "NaiveCodeAgent",
    "RemoteExecutorAgent",
    "DirectorAgent",
]
