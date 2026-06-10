"""Abstract base agent for laraq.

This module defines the abstract base class for agents that can be
composed into hierarchical multi-agent systems.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal, TypedDict

from laraq.logging import LoggingMixin

MessageType = Literal["human", "ai", "system"]


class Message:
    """Base message type for agent communication.

    Compatible with LangChain's BaseMessage structure.
    """

    def __init__(
        self,
        content: str,
        type: MessageType = "human",
        additional_kwargs: dict[str, Any] | None = None,
        response_metadata: dict[str, Any] | None = None
    ):
        """Initialize a message.

        Args:
            content: The message content
            type: The message type ("human", "ai", "system")
            additional_kwargs: Optional structured metadata for status, errors, validation, etc.
            response_metadata: Optional model response metadata
        """
        self.content = content
        self.type = type
        self.additional_kwargs = additional_kwargs or {}
        self.response_metadata = response_metadata or {}


class State:
    """Agent state container.

    Holds the conversation history and context for agent execution.
    """

    def __init__(self, messages: list[Message] | None = None):
        """Initialize state.

        Args:
            messages: List of messages in the conversation
        """
        self.messages = messages or []


class AgentResult(TypedDict):
    """Standard result returned by agent __call__ methods."""

    message: Message


def agent_result(message: Message) -> AgentResult:
    """Build the standard single-message agent result."""
    return {"message": message}


class BaseAgent(LoggingMixin, ABC):
    """Abstract base class for agents.

    This class defines the interface for agents in a hierarchical
    multi-agent system. Agents can have:
    - A system prompt that guides their behavior
    - Tools they can use
    - Sub-agents they can delegate to
    - A parent agent in the hierarchy

    Attributes:
        name: Unique identifier for the agent
        system_prompt: Instructions that guide the agent's behavior
        response_format: Optional structured output format
        wants: Configuration for what context the agent needs
        is_director: Whether this agent coordinates other agents
    """

    name: ClassVar[str] = NotImplemented
    system_prompt: ClassVar[str] = NotImplemented
    response_format: Any | None = None
    wants: ClassVar[dict[str, bool]] = {}
    is_director: ClassVar[bool] = False

    def __init__(self, sub_agents: list["BaseAgent"] | None = None, logfile: str | None = None) -> None:
        """Initialize the agent.

        Args:
            sub_agents: List of agents this agent can delegate to
            logfile: Path to log file (optional, defaults to "lara_chat.log")
        """
        self._sub_agents = sub_agents or []
        self._parent: BaseAgent | None = None

        # Set up logging
        if logfile is not None:
            self._logfile = logfile
        self.source_name = self.name

        # Set up parent relationships
        for agent in self._sub_agents:
            agent.parent = self

    @property
    def sub_agents(self) -> list["BaseAgent"]:
        """Get the list of sub-agents."""
        return self._sub_agents

    @property
    def parent(self) -> "BaseAgent | None":
        """Get the parent agent."""
        return self._parent

    @parent.setter
    def parent(self, parent: "BaseAgent") -> None:
        """Set the parent agent."""
        self._parent = parent

    @property
    def parent_name(self) -> str | None:
        """Get the name of the parent agent."""
        if self._parent is None:
            return None
        return self._parent.name

    @abstractmethod
    def __call__(self, state: State | str) -> AgentResult:
        """Execute the agent.

        This is the main entry point for agent execution. It should:
        1. Process the input state
        2. Invoke the underlying model
        3. Return the response

        Args:
            state: Current conversation state or a string query

        Returns:
            Dictionary containing the agent's response
        """
        pass

    @abstractmethod
    def invoke(self, messages: list[Message]) -> list[Message]:
        """Invoke the agent's model with a list of messages.

        Args:
            messages: List of messages to send to the model

        Returns:
            List of response messages from the model
        """
        pass

    def pre_agent(self, state: State) -> list[Message]:
        """Prepare messages before agent invocation.

        This method extracts and prepares the messages from the state
        that will be sent to the model. Subclasses can override to
        customize message preparation.

        Args:
            state: Current conversation state

        Returns:
            List of messages to send to the model
        """
        return state.messages

    def post_agent(self, responses: list[Message]) -> AgentResult:
        """Process responses after agent invocation.

        This method handles post-processing of the model's responses.
        Subclasses can override to customize response handling.

        Args:
            responses: List of response messages from the model

        Returns:
            Dictionary containing processed responses
        """
        if responses:
            return agent_result(responses[-1])
        return agent_result(Message(content="", type="ai"))

    def model_override(self, state: State | None = None) -> AgentResult | None:
        """Override the model invocation.

        If this method returns a non-None value, the model invocation
        will be skipped and this value will be returned instead.
        Useful for implementing shortcuts or special cases.

        Args:
            state: Current conversation state

        Returns:
            Override response or None to proceed normally
        """
        return None

    @property
    def full_system_prompt(self) -> str:
        """Build the complete system prompt.

        Constructs the full system prompt by combining the base prompt
        with information about available tools and sub-agents.

        Returns:
            Complete system prompt string
        """
        parts = [self.system_prompt.strip()]

        if self.sub_agents:
            parts.append("\nSUB-AGENTS:")
            for agent in self.sub_agents:
                doc = agent.__class__.__doc__ or "No description"
                parts.append(f"- {agent.name}: {doc.strip()}")

        return "\n".join(parts)

    def init_agent(self) -> None:
        """Initialize the agent.

        Called once before the agent is added to the graph.
        Subclasses can override to perform setup (load models,
        initialize tools, etc.).
        """
        pass  # Default: no-op

    def decide_next_agent(self, state: State) -> str:
        """Decide which agent should execute next.

        This method is called to determine the next agent in a
        multi-agent workflow. By default, returns the parent agent
        or "END" if there is no parent.

        Args:
            state: Current conversation state

        Returns:
            Name of the next agent or "END" to terminate execution
        """
        return self.parent_name or "END"

    def log_interaction(self, input_data: str, output_data: str) -> None:
        """Log an agent interaction.

        Args:
            input_data: Input to the agent
            output_data: Output from the agent
        """
        log_entry = f"""
{'=' * 80}
AGENT: {self.name}
{'=' * 80}
INPUT:
{input_data}
{'-' * 40}
OUTPUT:
{output_data}
{'=' * 80}

"""
        self.write_log(log_entry, append=True)
