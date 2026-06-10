"""Physics parameter extraction agent for pulling scientific quantities from user requests."""

from typing import Any, ClassVar

from laraq.agent import AgentResult, BaseAgent, Message, State, agent_result
from laraq.code_utils import extract_json_from_response
from laraq.llm import LLMClient


PHYSICS_PARAMS_SYSTEM_PROMPT = """You are a scientific parameter extraction assistant. \
Given a user request, extract any explicit physical or scientific parameters mentioned.

IMPORTANT: You must respond with ONLY a JSON object containing the parameters found.
If no parameters are present, return an empty object: {}

Extract any of the following if explicitly stated or clearly implied:
- elements: list of chemical element symbols, e.g. ["H", "C", "O"]
- positions: list of atomic positions as coordinate triples
- units: coordinate units, e.g. "bohr" or "angstrom"
- functional: DFT exchange-correlation functional, e.g. "LDA", "PBE", "PBE0"
- boundary_conditions: simulation cell type, e.g. "free", "periodic", "surface", "wire"
- spin_polarised: true or false
- grid_spacing: real-space grid spacing value (float)
- charge: total system charge (integer)
- temperature: simulation temperature in Kelvin (float)
- timestep: MD timestep value
- Any other explicit numerical or named physical parameters from the request

Do not invent values. Only include what is present in the request.
Only output the JSON object, nothing else."""


class PhysicsParamsAgent(BaseAgent):
    """Extracts physical and scientific parameters from a user request.

    This agent:
    1. Reads the user query
    2. Asks the LLM to identify any scientific quantities stated in the request
    3. Returns a flat dictionary of parameter names to values (empty if none found)
    """

    name: ClassVar[str] = "physics_params"
    system_prompt: ClassVar[str] = PHYSICS_PARAMS_SYSTEM_PROMPT
    is_director: ClassVar[bool] = False

    def __init__(
        self,
        llm_client: LLMClient,
        sub_agents: list["BaseAgent"] | None = None,
        **kwargs
    ) -> None:
        """Initialize the physics params agent.

        Args:
            llm_client: Client for LLM generation
            sub_agents: Optional list of sub-agents
            **kwargs: Additional arguments passed to BaseAgent (e.g., logfile)
        """
        super().__init__(sub_agents=sub_agents, **kwargs)
        self.llm_client = llm_client

    def __call__(self, state: State | str) -> AgentResult:
        """Execute the agent to extract physics parameters from the user query.

        Args:
            state: Input state containing the user query

        Returns:
            Dictionary with extracted parameters as a message
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
                content="{}",
                type="ai",
                additional_kwargs={
                    "agent": "physics_params",
                    "params": {},
                }
            ))

        try:
            llm_response = self.llm_client.generate(
                prompt=query,
                system_prompt=self.system_prompt,
            )
            params = self._parse_response(llm_response)
        except Exception:
            params = {}

        if params:
            content = ", ".join(f"{k}: {v}" for k, v in params.items())
        else:
            content = "No physics parameters found."

        response_message = Message(
            content=content,
            type="ai",
            additional_kwargs={
                "agent": "physics_params",
                "params": params,
            }
        )
        return agent_result(response_message)

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse the LLM response to extract the physics parameters dict.

        Args:
            response: Raw LLM response

        Returns:
            Dictionary of parameter names to values (may be empty)
        """
        data = extract_json_from_response(response)
        return data if data is not None else {}

    def invoke(self, messages: list[Message]) -> list[Message]:
        """Invoke the agent with messages.

        Args:
            messages: Input messages

        Returns:
            List containing the physics parameter extraction result
        """
        state = State(messages=messages)
        result = self(state)
        return [result["message"]]

    def model_override(self, state: State | None = None) -> AgentResult | None:
        """No model override — uses LLM client directly."""
        return None
