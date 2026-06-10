"""Agent registry and factory for creating agent instances from configuration."""

from pathlib import Path
from typing import Any

from laraq.agent import BaseAgent
from laraq.agents import (
    NaiveCodeAgent,
    DirectorAgent,
    RemoteExecutorAgent,
    LocalExecutorAgent,
    NaiveResearchAgent,
    ValidatorAgent,
    RAGResearchAgent,
    LLMCodeAgent,
    DryRunAgent,
    ModuleMapperAgent,
    IntentExtractorAgent,
    PhysicsParamsAgent,
)
from laraq.config.models import Config
from laraq.embeddings import create_embedding_client
from laraq.llm import create_llm_client
from laraq.logging import DEFAULT_LOGFILE


class AgentRegistry:
    """Registry of available agent implementations."""

    def __init__(self):
        """Initialize the agent registry."""
        self._agents: dict[str, type[BaseAgent]] = {}
        self._register_default_agents()

    def _register_default_agents(self):
        """Register built-in agent implementations."""
        self.register("naive_research", NaiveResearchAgent)
        self.register("naive_code", NaiveCodeAgent)
        self.register("validator", ValidatorAgent)
        self.register("dry_run", DryRunAgent)
        self.register("remote_executor", RemoteExecutorAgent)
        self.register("director", DirectorAgent)
        self.register("rag_research", RAGResearchAgent)
        self.register("llm_code", LLMCodeAgent)
        self.register("module_mapper", ModuleMapperAgent)
        self.register("intent_extractor", IntentExtractorAgent)
        self.register("physics_params", PhysicsParamsAgent)
        self.register("local_executor", LocalExecutorAgent)

    def register(self, name: str, agent_class: type[BaseAgent]):
        """Register an agent implementation.

        Args:
            name: Name to register the agent under
            agent_class: Agent class to register
        """
        self._agents[name] = agent_class

    def create(self, name: str, **kwargs) -> BaseAgent:
        """Create an agent instance.

        Args:
            name: Name of the agent to create
            **kwargs: Arguments to pass to agent constructor

        Returns:
            Agent instance

        Raises:
            ValueError: If agent name is not registered
        """
        if name not in self._agents:
            raise ValueError(f"Unknown agent type: {name}")

        agent_class = self._agents[name]
        return agent_class(**kwargs)

    def list_agents(self) -> list[str]:
        """List all registered agent names.

        Returns:
            List of registered agent names
        """
        return list(self._agents.keys())


# Global registry instance
registry = AgentRegistry()


def _llm_client_for_agent(config: Config, agent_config: dict) -> object:
    """Create an LLM client, using a per-agent model override if configured."""
    return create_llm_client(config, model_override=agent_config.get("model"))


def _agent_kwargs(agent_config: dict) -> dict:
    """Return agent config dict with the model key stripped out."""
    return {k: v for k, v in agent_config.items() if k != "model"}


def create_agents_from_config(config: Config) -> dict[str, BaseAgent]:
    """Create agent instances from configuration.

    Args:
        config: Full configuration object

    Returns:
        Dictionary mapping agent roles to instances
    """
    agents_config = config.agents
    logfile = DEFAULT_LOGFILE
    agents: dict[str, BaseAgent] = {}

    # Phase 1 agents: all need an LLM client
    llm_agent_specs = [
        ("intent_extractor", agents_config.intent_extractor, agents_config.intent_extractor_config),
        ("physics_params", agents_config.physics_params, agents_config.physics_params_config),
        ("module_mapper", agents_config.module_mapper, agents_config.module_mapper_config),
    ]
    for role, agent_name, agent_config in llm_agent_specs:
        kwargs = _agent_kwargs(agent_config)
        kwargs["logfile"] = logfile
        kwargs["llm_client"] = _llm_client_for_agent(config, agent_config)
        agents[role] = registry.create(agent_name, **kwargs)

    # Research agent — needs embedding client and optional LLM for query expansion
    research_kwargs = agents_config.research_config.copy()
    research_kwargs["logfile"] = logfile
    if agents_config.research == "rag_research":
        research_kwargs["embedding_client"] = create_embedding_client(config)
        research_kwargs["llm_client"] = create_llm_client(config)
    agents["research"] = registry.create(agents_config.research, **research_kwargs)

    # Code agent — needs LLM client and serialization flag
    code_kwargs = agents_config.code_config.copy()
    code_kwargs["logfile"] = logfile
    if agents_config.code == "llm_code":
        code_kwargs["llm_client"] = create_llm_client(config)
        code_kwargs["json_serializable"] = config.template != "bigdft_local"
    agents["code"] = registry.create(agents_config.code, **code_kwargs)

    # Simple agents: just config + logfile
    simple_agent_specs = [
        ("validator", agents_config.validator, agents_config.validator_config),
        ("dry_run", agents_config.dry_run, agents_config.dry_run_config),
    ]
    for role, agent_name, agent_config in simple_agent_specs:
        kwargs = agent_config.copy()
        kwargs["logfile"] = logfile
        agents[role] = registry.create(agent_name, **kwargs)

    # Executor agent — depends on template
    if config.template == "bigdft_local":
        agents["executor"] = registry.create("local_executor", logfile=logfile)
    else:
        if config.remote is None:
            raise ValueError("bigdft_remote template requires a [remote] configuration section")
        # Collect extra remote fields (e.g. ssh_insert) for remotemanager
        known_remote_fields = {"template", "host", "user", "remote_dir", "submitter"}
        extra_computer_kwargs = {
            k: v for k, v in config.remote.model_dump().items()
            if k not in known_remote_fields and v is not None
        }
        agents["executor"] = registry.create(
            "remote_executor",
            template=config.remote.template or "#!/bin/bash",
            host=config.remote.host,
            user=config.remote.user,
            remote_dir=config.remote.remote_dir,
            submitter=config.remote.submitter,
            config_filename=Path(config.config_path).name if config.config_path else None,
            computer_kwargs=extra_computer_kwargs,
            logfile=logfile,
        )

    # Director — wraps all Phase 2/3 sub-agents
    director_kwargs = agents_config.director_config.copy()
    director_kwargs["logfile"] = logfile
    director_kwargs["sub_agents"] = [
        agents["research"], agents["code"], agents["validator"],
        agents["dry_run"], agents["executor"],
    ]
    agents["director"] = registry.create(agents_config.director, **director_kwargs)

    return agents
