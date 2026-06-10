"""Module mapper agent that checks request feasibility against a capabilities document."""

import logging
from pathlib import Path
from typing import Any, ClassVar

from laraq.agent import AgentResult, BaseAgent, Message, State, agent_result
from laraq.code_utils import extract_json_from_response
from laraq.llm import LLMClient

logger = logging.getLogger(__name__)


CAPABILITIES_SYSTEM_PROMPT = """You are a capability checker. Given a reference \
document describing what a software package can and cannot do, and a user request, \
determine whether the request is feasible using that package's API.

IMPORTANT: You must respond with ONLY a JSON object in this exact format:
{
    "feasible": true,
    "reason": "one sentence explanation",
    "modules": ["package.module.X", "package.module.Y"]
}

Rules:
1. "feasible" must be true or false
2. "reason" must be a single sentence
3. "modules" must contain exact module paths from the Module Index in the reference document
4. If not feasible, "modules" must be an empty list []
5. Do not guess — if a capability is not described in the document, say not feasible
6. Only output the JSON object, nothing else"""


class ModuleMapperAgent(BaseAgent):
    """Capability checking agent using a static capabilities document.

    This agent:
    1. Loads a capabilities reference document at initialization
    2. On query, asks the LLM whether the request is feasible
    3. Returns a feasibility verdict and suggested modules
    """

    name: ClassVar[str] = "module_mapper"
    system_prompt: ClassVar[str] = CAPABILITIES_SYSTEM_PROMPT
    is_director: ClassVar[bool] = False

    def __init__(
        self,
        llm_client: LLMClient,
        capabilities_path: str | Path = "docs/capabilities.md",
        sub_agents: list["BaseAgent"] | None = None,
        **kwargs
    ) -> None:
        """Initialize the module mapper agent.

        Args:
            llm_client: Client for LLM generation
            capabilities_path: Path to the capabilities reference document
            sub_agents: Optional list of sub-agents
            **kwargs: Additional arguments passed to BaseAgent (e.g., logfile)
        """
        super().__init__(sub_agents=sub_agents, **kwargs)
        self.llm_client = llm_client
        self.capabilities_path = Path(capabilities_path)
        self.capabilities_text = self.capabilities_path.read_text()

    def __call__(self, state: State | str) -> AgentResult:
        """Execute the agent to check feasibility of the user's request.

        Args:
            state: Input state containing the user query

        Returns:
            Dictionary with a feasibility verdict as a message
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
                    "agent": "module_mapper",
                    "status": "uncertain",
                    "feasible": True,
                    "modules": [],
                    "reason": "No query provided.",
                }
            ))

        prompt = self._build_prompt(query)

        try:
            llm_response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
            )
            result = self._parse_response(llm_response)
        except Exception as e:
            logger.warning("Module mapper LLM call failed, assuming feasible: %s", e)
            result = {
                "feasible": True,
                "reason": f"Capability check failed: {e}",
                "modules": [],
                "status": "uncertain",
            }

        feasible = result["feasible"]
        modules = result["modules"]
        reason = result["reason"]
        status = result.get("status", "feasible" if feasible else "not_feasible")

        if feasible:
            module_str = f": {', '.join(modules)}" if modules else ""
            content = f"Feasible{module_str}. {reason}"
        else:
            content = f"Not feasible: {reason}"

        response_message = Message(
            content=content,
            type="ai",
            additional_kwargs={
                "agent": "module_mapper",
                "status": status,
                "feasible": feasible,
                "modules": modules,
                "reason": reason,
            }
        )
        return agent_result(response_message)

    def _build_prompt(self, query: str) -> str:
        """Build the prompt combining the capabilities document and user query.

        Args:
            query: The user's request

        Returns:
            Formatted prompt string
        """
        return (
            f"## Capabilities Reference\n\n"
            f"{self.capabilities_text}\n\n"
            f"---\n\n"
            f"## User Request\n\n"
            f"{query}"
        )

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse the LLM response to extract the feasibility verdict.

        Args:
            response: Raw LLM response

        Returns:
            Dictionary with feasible, reason, modules, and status fields
        """
        data = extract_json_from_response(response)
        if data is not None:
            return self._extract_fields(data)

        # Fallback: pass through as uncertain so the pipeline is not blocked
        return {
            "feasible": True,
            "reason": "Could not parse capability check response.",
            "modules": [],
            "status": "uncertain",
        }

    def _extract_fields(self, data: dict) -> dict[str, Any]:
        """Extract and normalise fields from a parsed JSON response.

        Args:
            data: Parsed JSON dictionary

        Returns:
            Normalised result dictionary
        """
        feasible = bool(data.get("feasible", True))
        reason = str(data.get("reason", ""))
        modules = data.get("modules", [])
        if not isinstance(modules, list):
            modules = []
        status = "feasible" if feasible else "not_feasible"
        return {"feasible": feasible, "reason": reason, "modules": modules, "status": status}

    def invoke(self, messages: list[Message]) -> list[Message]:
        """Invoke the agent with messages.

        Args:
            messages: Input messages

        Returns:
            List containing the feasibility verdict message
        """
        state = State(messages=messages)
        result = self(state)
        return [result["message"]]

    def model_override(self, state: State | None = None) -> AgentResult | None:
        """No model override — uses LLM client directly."""
        return None
