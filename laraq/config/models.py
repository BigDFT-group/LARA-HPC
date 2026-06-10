"""Configuration models for laraq.

This module defines Pydantic models for validating configuration
for different AI provider types.
"""

from typing import Any, Literal
from pydantic import BaseModel, Field


class OllamaConfig(BaseModel):
    """Configuration for Ollama provider."""

    base_url: str = Field(..., description="Base URL for Ollama API")
    model: str = Field(..., description="Model name to use for LLM")
    embedding_model: str | None = Field(None, description="Model name for embeddings")


class OpenAICompatibleConfig(BaseModel):
    """Configuration for OpenAI-compatible provider."""

    base_url: str = Field(..., description="Base URL for OpenAI-compatible API")
    api_key: str = Field(..., description="API key for authentication")
    model: str = Field(..., description="Model name to use for LLM")
    embedding_model: str | None = Field(None, description="Model name for embeddings")


class AnthropicConfig(BaseModel):
    """Configuration for Anthropic provider."""

    api_key: str = Field(..., description="Anthropic API key")
    model: str = Field(..., description="Model name to use for LLM")
    embedding_model: str | None = Field(None, description="Model name for embeddings (if using Anthropic for embeddings)")


class GoogleConfig(BaseModel):
    """Configuration for Google Gemini provider."""

    api_key: str = Field(..., description="Google API key")
    model: str = Field(..., description="Model name to use for LLM (e.g., gemini-1.5-pro)")
    embedding_model: str | None = Field(None, description="Model name for embeddings (e.g., text-embedding-004)")


class AgentConfig(BaseModel):
    """Agent configuration specifying which providers to use."""

    llm_provider: Literal["ollama", "openai-compatible", "anthropic", "google"] = Field(
        ..., description="AI provider to use for LLM operations"
    )
    embedding_provider: Literal["ollama", "openai-compatible", "google"] = Field(
        ..., description="AI provider to use for embeddings"
    )


class AgentsConfig(BaseModel):
    """Configuration for which agent implementations to use."""

    intent_extractor: str = Field("intent_extractor", description="Intent extraction agent implementation")
    physics_params: str = Field("physics_params", description="Physics parameter extraction agent implementation")
    module_mapper: str = Field("module_mapper", description="Module mapper agent implementation")
    research: str = Field("rag_research", description="Research agent implementation")
    code: str = Field("llm_code", description="Code generation agent implementation")
    validator: str = Field("validator", description="Code validation agent implementation")
    dry_run: str = Field("dry_run", description="Dry run execution agent implementation")
    director: str = Field("director", description="Director agent implementation")

    # Agent-specific configurations
    intent_extractor_config: dict[str, Any] = Field(default_factory=dict, description="Config for intent extractor agent")
    physics_params_config: dict[str, Any] = Field(default_factory=dict, description="Config for physics params agent")
    module_mapper_config: dict[str, Any] = Field(default_factory=dict, description="Config for module mapper agent")
    research_config: dict[str, Any] = Field(default_factory=dict, description="Config for research agent")
    code_config: dict[str, Any] = Field(default_factory=dict, description="Config for code agent")
    validator_config: dict[str, Any] = Field(default_factory=dict, description="Config for validator agent")
    dry_run_config: dict[str, Any] = Field(default_factory=dict, description="Config for dry run agent")
    director_config: dict[str, Any] = Field(default_factory=dict, description="Config for director agent")


class RemoteConfig(BaseModel, extra="allow"):
    """Configuration for remote HPC execution via remotemanager.

    Extra fields (e.g. ssh_insert, ssh_prepend) are passed through
    to remotemanager's Computer constructor as **kwargs.
    """

    template: str | None = Field(
        None,
        description="Job script template (optional; inline string or path to a file)",
    )
    host: str = Field(..., description="SSH hostname of the remote machine")
    user: str | None = Field(None, description="SSH username (optional; defaults to the current user)")
    remote_dir: str | None = Field(
        None,
        description="Working directory on the remote machine (optional; laraq defaults to a project-specific directory)",
    )
    submitter: str | None = Field(None, description="Job scheduler command (optional)")


class Config(BaseModel):
    """Main configuration container.

    Validates that the selected provider's configuration is present
    and properly configured.
    """

    agent: AgentConfig
    agents: AgentsConfig = Field(default_factory=AgentsConfig, description="Agent implementations to use")
    template: Literal["bigdft_remote", "bigdft_local"] = Field("bigdft_remote", description="Execution template: 'bigdft_remote' (remote HPC, JSON-serializable returns) or 'bigdft_local' (local execution, any return type)")
    max_retries: int = Field(1, description="Maximum number of research→code→validate→dry-run retry cycles on failure")
    ollama: OllamaConfig | None = None
    openai_compatible: OpenAICompatibleConfig | None = None
    anthropic: AnthropicConfig | None = None
    google: GoogleConfig | None = None
    remote: RemoteConfig | None = None
    config_path: str | None = Field(None, exclude=True, description="Path to the loaded config file")

    def model_post_init(self, __context):
        """Validate that the selected providers have configuration."""
        llm_provider = self.agent.llm_provider
        embedding_provider = self.agent.embedding_provider

        # Validate LLM provider configuration exists
        if llm_provider == "ollama" and self.ollama is None:
            raise ValueError("Ollama LLM provider selected but [ollama] configuration is missing")
        elif llm_provider == "openai-compatible" and self.openai_compatible is None:
            raise ValueError("OpenAI-compatible LLM provider selected but [openai_compatible] configuration is missing")
        elif llm_provider == "anthropic" and self.anthropic is None:
            raise ValueError("Anthropic LLM provider selected but [anthropic] configuration is missing")
        elif llm_provider == "google" and self.google is None:
            raise ValueError("Google LLM provider selected but [google] configuration is missing")

        # Validate embedding provider configuration exists
        if embedding_provider == "ollama" and self.ollama is None:
            raise ValueError("Ollama embedding provider selected but [ollama] configuration is missing")
        elif embedding_provider == "openai-compatible" and self.openai_compatible is None:
            raise ValueError("OpenAI-compatible embedding provider selected but [openai_compatible] configuration is missing")
        elif embedding_provider == "google" and self.google is None:
            raise ValueError("Google embedding provider selected but [google] configuration is missing")

        # Validate embedding model is specified if provider has it
        embedding_config = self._get_provider_config(embedding_provider)
        if embedding_config.embedding_model is None:
            raise ValueError(
                f"Embedding provider '{embedding_provider}' selected but no embedding_model specified in configuration"
            )

    def _get_provider_config(self, provider: str) -> OllamaConfig | OpenAICompatibleConfig | AnthropicConfig | GoogleConfig:
        """Get configuration for a specific provider."""
        if provider == "ollama":
            if self.ollama is None:
                raise ValueError("Ollama provider selected but [ollama] configuration is missing")
            return self.ollama
        elif provider == "openai-compatible":
            if self.openai_compatible is None:
                raise ValueError("OpenAI-compatible provider selected but [openai_compatible] configuration is missing")
            return self.openai_compatible
        elif provider == "anthropic":
            if self.anthropic is None:
                raise ValueError("Anthropic provider selected but [anthropic] configuration is missing")
            return self.anthropic
        elif provider == "google":
            if self.google is None:
                raise ValueError("Google provider selected but [google] configuration is missing")
            return self.google
        raise ValueError(f"Unknown provider: {provider}")

    def get_llm_config(self) -> OllamaConfig | OpenAICompatibleConfig | AnthropicConfig | GoogleConfig:
        """Get the configuration for the LLM provider."""
        return self._get_provider_config(self.agent.llm_provider)

    def get_embedding_config(self) -> OllamaConfig | OpenAICompatibleConfig | AnthropicConfig | GoogleConfig:
        """Get the configuration for the embedding provider."""
        return self._get_provider_config(self.agent.embedding_provider)

    def get_active_config(self) -> OllamaConfig | OpenAICompatibleConfig | AnthropicConfig | GoogleConfig:
        """Get the configuration for the LLM provider (legacy method)."""
        return self.get_llm_config()
