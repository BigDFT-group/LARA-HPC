"""Intent extraction agent for parsing user requests into structured intent."""

from typing import Any, ClassVar

from laraq.agent import AgentResult, BaseAgent, Message, State, agent_result
from laraq.code_utils import extract_json_from_response
from laraq.llm import LLMClient


INTENT_SYSTEM_PROMPT = """You are an intent extraction assistant. Given a user request, \
extract the core intent and generate search hints for documentation retrieval.

IMPORTANT: You must respond with ONLY a JSON object in this exact format:
{
    "intent": "concise description of what the user wants to accomplish",
    "calculation_type": "short label, e.g. single_point, geometry_optimisation, molecular_dynamics, analysis, io, other",
    "search_hints": ["search term 1", "search term 2", "search term 3"]
}

Rules:
1. "intent" must be a single concise sentence
2. "calculation_type" must be a short label describing the kind of task
3. "search_hints" must be 3-5 documentation-style terms (class names, method names, API concepts)
4. Only output the JSON object, nothing else"""


class IntentExtractorAgent(BaseAgent):
    """Extracts structured intent and search hints from a user request.

    This agent:
    1. Reads the user query
    2. Asks the LLM to identify the core intent and calculation type
    3. Generates documentation-style search hints for the RAG stage
    """

    name: ClassVar[str] = "intent_extractor"
    system_prompt: ClassVar[str] = INTENT_SYSTEM_PROMPT
    is_director: ClassVar[bool] = False

    def __init__(
        self,
        llm_client: LLMClient,
        sub_agents: list["BaseAgent"] | None = None,
        **kwargs
    ) -> None:
        """Initialize the intent extractor agent.

        Args:
            llm_client: Client for LLM generation
            sub_agents: Optional list of sub-agents
            **kwargs: Additional arguments passed to BaseAgent (e.g., logfile)
        """
        super().__init__(sub_agents=sub_agents, **kwargs)
        self.llm_client = llm_client

    def __call__(self, state: State | str) -> AgentResult:
        """Execute the agent to extract intent from the user query.

        Args:
            state: Input state containing the user query

        Returns:
            Dictionary with extracted intent as a message
        """
        if isinstance(state, str):
            query = state
        else:
            query = ""
            for msg in state.messages:
                if msg.type == "human":
                    query = msg.content
                    break

        if not query:
            return agent_result(Message(
                content="No query provided.",
                type="ai",
                additional_kwargs={
                    "agent": "intent_extractor",
                    "intent": "",
                    "calculation_type": "other",
                    "search_hints": [],
                }
            ))

        try:
            llm_response = self.llm_client.generate(
                prompt=query,
                system_prompt=self.system_prompt,
            )
            result = self._parse_response(llm_response)
        except Exception as e:
            result = {
                "intent": query,
                "calculation_type": "other",
                "search_hints": [],
                "error": str(e),
            }

        content = result["intent"] or query
        response_message = Message(
            content=content,
            type="ai",
            additional_kwargs={
                "agent": "intent_extractor",
                "intent": result["intent"],
                "calculation_type": result["calculation_type"],
                "search_hints": result["search_hints"],
            }
        )
        return agent_result(response_message)

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse the LLM response to extract intent fields.

        Args:
            response: Raw LLM response

        Returns:
            Dictionary with intent, calculation_type, and search_hints
        """
        data = extract_json_from_response(response)
        if data is not None:
            return self._extract_fields(data)
        return {"intent": "", "calculation_type": "other", "search_hints": []}

    def _extract_fields(self, data: dict) -> dict[str, Any]:
        """Extract and normalise fields from a parsed JSON response.

        Args:
            data: Parsed JSON dictionary

        Returns:
            Normalised result dictionary
        """
        intent = str(data.get("intent", ""))
        calculation_type = str(data.get("calculation_type", "other"))
        search_hints = data.get("search_hints", [])
        if not isinstance(search_hints, list):
            search_hints = []
        return {
            "intent": intent,
            "calculation_type": calculation_type,
            "search_hints": [str(h) for h in search_hints],
        }

    def invoke(self, messages: list[Message]) -> list[Message]:
        """Invoke the agent with messages.

        Args:
            messages: Input messages

        Returns:
            List containing the intent extraction result
        """
        state = State(messages=messages)
        result = self(state)
        return [result["message"]]

    def model_override(self, state: State | None = None) -> AgentResult | None:
        """No model override — uses LLM client directly."""
        return None
