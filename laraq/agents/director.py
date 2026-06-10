"""Director agent that orchestrates other agents."""

from enum import Enum
from typing import ClassVar

from laraq.agent import AgentResult, BaseAgent, Message, State, agent_result


class Phase(Enum):
    """Director routing phases."""
    INITIAL = "initial"
    RETRY = "retry"
    EXECUTE = "execute"
    FAILED = "failed"


class DirectorAgent(BaseAgent):
    """A director agent that coordinates sub-agents using dynamic routing.

    This director routes execution through its sub-agents in sequence,
    using the decide_next_agent() method for dynamic routing in a graph.
    """

    name: ClassVar[str] = "director"
    system_prompt: ClassVar[str] = "Coordinates sub-agents using dynamic routing."
    is_director: ClassVar[bool] = True

    def __init__(self, sub_agents: list[BaseAgent], **kwargs) -> None:
        """Initialize the director agent.

        Args:
            sub_agents: List of agents to route through in sequence (research, code, validator, decision, executor)
            **kwargs: Additional arguments passed to BaseAgent (e.g., logfile)
        """
        super().__init__(sub_agents=sub_agents, **kwargs)

        # Build workflow map
        self._agent_map = {agent.name: agent for agent in sub_agents}

        # Track routing state
        self._retry_count = 0
        self._max_retries = 1
        self._current_phase = Phase.INITIAL

    def __call__(self, state: State | str) -> AgentResult:
        """Execute the director (passes state through without transformation).

        The director doesn't transform the state itself - it just routes
        execution to sub-agents via decide_next_agent().

        Args:
            state: Input state

        Returns:
            Dictionary with the current state messages
        """
        # Reset routing state for each new invocation
        self._retry_count = 0
        self._current_phase = Phase.INITIAL

        # Convert string input to State if needed
        if isinstance(state, str):
            state = State(messages=[Message(content=state, type="human")])

        # Director passes through - routing is handled by decide_next_agent
        if state.messages:
            return agent_result(state.messages[-1])
        else:
            return agent_result(Message(content="", type="ai"))

    def invoke(self, messages: list[Message]) -> list[Message]:
        """Invoke the director.

        Args:
            messages: Input messages

        Returns:
            List containing pass-through messages
        """
        if messages:
            return [messages[-1]]
        else:
            return [Message(content="", type="ai")]

    def decide_next_agent(self, state: State) -> str:
        """Route to the next agent based on current phase and validation results.

        Workflow:
        1. Initial: research → code → validator
        2. If validator fails and retries available: research → code → validator
        3. If validator succeeds: executor
        4. If validator fails and no retries: END

        Args:
            state: Current conversation state

        Returns:
            Name of next agent or "END"
        """
        if not state.messages:
            return "rag_research" if "rag_research" in self._agent_map else "naive_research"

        last_msg = state.messages[-1]
        agent_type = last_msg.additional_kwargs.get("agent", "")

        # Initial phase: research → code → validator
        if self._current_phase == Phase.INITIAL:
            if agent_type == "":
                # Start of workflow
                return "rag_research" if "rag_research" in self._agent_map else "naive_research"
            elif agent_type == "rag_research" or agent_type == "naive_research":
                return "llm_code" if "llm_code" in self._agent_map else "naive_code"
            elif agent_type == "llm_code" or agent_type == "naive_code":
                return "validator"
            elif agent_type == "validator":
                # Check validation result
                status = last_msg.additional_kwargs.get("status", "success")
                if status == "success":
                    # Validation passed, go to execution
                    self._current_phase = Phase.EXECUTE
                    return "remote_executor"
                else:
                    # Validation failed
                    if self._retry_count < self._max_retries:
                        # We can retry, go back to research
                        self._retry_count += 1
                        self._current_phase = Phase.RETRY
                        return "rag_research" if "rag_research" in self._agent_map else "naive_research"
                    else:
                        # No retries left, fail
                        self._current_phase = Phase.FAILED
                        return "END"

        # Retry phase: research → code → validator → execute or fail
        elif self._current_phase == Phase.RETRY:
            if agent_type in ["rag_research", "naive_research"]:
                # After research, generate code
                return "llm_code" if "llm_code" in self._agent_map else "naive_code"
            elif agent_type in ["llm_code", "naive_code"]:
                # After code, validate
                return "validator"
            elif agent_type == "validator":
                # Check validation result again
                status = last_msg.additional_kwargs.get("status", "success")
                if status == "success":
                    # Validation passed, go to execution
                    self._current_phase = Phase.EXECUTE
                    return "remote_executor"
                else:
                    # Validation failed again, no more retries
                    self._current_phase = Phase.FAILED
                    return "END"

        # Execute phase: executor → END
        elif self._current_phase == Phase.EXECUTE:
            if agent_type == "remote_executor":
                return "END"

        # Failed phase or unknown state
        return "END"
