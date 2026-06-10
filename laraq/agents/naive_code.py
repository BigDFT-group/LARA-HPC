"""Naive code agent that generates print statements."""

from typing import Any, ClassVar

from laraq.agent import AgentResult, BaseAgent, Message, State, agent_result


class NaiveCodeAgent(BaseAgent):
    """A simple code generation agent that returns messages as string literals.

    This agent takes any message it receives and generates Python code
    that returns that message as a string, without calling any LLM.
    """

    name: ClassVar[str] = "naive_code"
    system_prompt: ClassVar[str] = "Generates Python code that returns the received message as a string."
    is_director: ClassVar[bool] = False

    def __call__(self, state: State | str) -> AgentResult:
        """Execute the agent by generating print code.

        Args:
            state: Input state containing the message to wrap in code

        Returns:
            Dictionary with generated Python code as a message
        """
        # Convert string input to State if needed
        if isinstance(state, str):
            message_content = state
        else:
            # Extract the last user message from state
            message_content = self._extract_last_message(state)

        # Generate Python code
        code = self._generate_return_code(message_content)

        response_message = Message(
            content=code,
            type="ai",
            additional_kwargs={"agent": "naive_code"}
        )
        return agent_result(response_message)

    def invoke(self, messages: list[Message]) -> list[Message]:
        """Invoke the agent to generate print code.

        Args:
            messages: Input messages

        Returns:
            List containing a single message with generated code
        """
        # Get the last message content
        message_content = messages[-1].content if messages else ""

        # Generate code
        code = self._generate_return_code(message_content)

        return [Message(
            content=code,
            type="ai",
            additional_kwargs={"agent": "naive_code"}
        )]

    def model_override(self, state: State | None = None) -> AgentResult | None:
        """Override model invocation to generate code directly.

        This agent doesn't need to call an LLM - it just wraps
        the message in a print statement.

        Args:
            state: Current state

        Returns:
            Dictionary with generated code message
        """
        if state is None:
            return agent_result(Message(content='print("")', type="ai"))

        message_content = self._extract_last_message(state)
        code = self._generate_return_code(message_content)

        response_message = Message(
            content=code,
            type="ai",
            additional_kwargs={"agent": "naive_code"}
        )
        return agent_result(response_message)

    def _extract_last_message(self, state: State) -> str:
        """Extract the last user message from state.

        Args:
            state: Current conversation state

        Returns:
            Content of the last message
        """
        if not state.messages:
            return ""

        # Get the last message
        return state.messages[-1].content

    def _generate_return_code(self, message: str) -> str:
        """Generate Python code that returns the message.

        Args:
            message: The message to return as a string literal

        Returns:
            Python code as a string
        """
        # Escape special characters for safe Python string
        escaped_message = (
            message.replace("\\", "\\\\")  # Escape backslashes first
            .replace('"', '\\"')            # Escape double quotes
            .replace("\n", "\\n")           # Escape newlines
            .replace("\r", "\\r")           # Escape carriage returns
            .replace("\t", "\\t")           # Escape tabs
        )

        # Generate the code - just the string literal
        code = f'"{escaped_message}"'

        return code
