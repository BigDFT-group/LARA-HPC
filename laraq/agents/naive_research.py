"""Naive research agent that returns file content."""

from pathlib import Path
from typing import Any, ClassVar

from laraq.agent import AgentResult, BaseAgent, Message, State, agent_result


class NaiveResearchAgent(BaseAgent):
    """A simple research agent that returns the content of a predefined file.

    This agent reads a file once during initialization and returns its
    content whenever invoked, without calling any LLM.
    """

    name: ClassVar[str] = "naive_research"
    system_prompt: ClassVar[str] = "Returns the content of a predefined file."
    is_director: ClassVar[bool] = False

    def __init__(
        self,
        file_path: str | Path = "docs/project_info.md",
        sub_agents: list[BaseAgent] | None = None,
        **kwargs
    ) -> None:
        """Initialize the naive research agent.

        Args:
            file_path: Path to the file to read and return (default: docs/project_info.md)
            sub_agents: Optional list of sub-agents (unused for this agent)
            **kwargs: Additional arguments passed to BaseAgent (e.g., logfile)
        """
        super().__init__(sub_agents=sub_agents, **kwargs)

        self.file_path = Path(file_path)

        # Read the file content during initialization
        if not self.file_path.exists():
            raise FileNotFoundError(f"Research file not found: {self.file_path}")

        self.file_content = self.file_path.read_text()

    def __call__(self, state: State | str) -> AgentResult:
        """Execute the agent by searching for lines matching the first word.

        Args:
            state: Input state containing the user's query

        Returns:
            Dictionary with matching lines as a message
        """
        # Convert string input to State if needed
        if isinstance(state, str):
            query = state
        else:
            # Extract the last user message
            if state.messages:
                query = state.messages[-1].content
            else:
                query = ""

        # Get the first word from the query
        first_word = query.split()[0] if query.split() else ""

        # Search for lines containing the first word (case-insensitive)
        matching_lines = self._search_for_word(first_word)

        response_message = Message(
            content=matching_lines,
            type="ai",
            additional_kwargs={"agent": "naive_research"}
        )
        return agent_result(response_message)

    def invoke(self, messages: list[Message]) -> list[Message]:
        """Invoke the agent with messages.

        Args:
            messages: Input messages

        Returns:
            List containing matching lines
        """
        query = messages[-1].content if messages else ""
        first_word = query.split()[0] if query.split() else ""
        matching_lines = self._search_for_word(first_word)
        return [Message(
            content=matching_lines,
            type="ai",
            additional_kwargs={"agent": "naive_research"},
        )]

    def model_override(self, state: State | None = None) -> AgentResult | None:
        """Override model invocation to search directly.

        This agent doesn't need to call an LLM - it just searches
        the file content.

        Args:
            state: Current state

        Returns:
            None to let __call__ handle the logic
        """
        return None

    def _search_for_word(self, word: str) -> str:
        """Search for lines containing the given word, including the next line.

        Args:
            word: Word to search for (case-insensitive)

        Returns:
            Matching lines with their next line, or a message if no matches
        """
        if not word:
            return "No search term provided."

        word_lower = word.lower()
        lines = self.file_content.split('\n')
        matching_sections = []

        for i, line in enumerate(lines):
            if word_lower in line.lower():
                # Add the matching line
                section = [line]
                # Add the next line if it exists
                if i + 1 < len(lines):
                    section.append(lines[i + 1])
                matching_sections.append('\n'.join(section))

        if matching_sections:
            return '\n'.join(matching_sections)
        else:
            return f"No lines found containing '{word}'."
