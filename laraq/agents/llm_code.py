"""LLM-powered code generation agent."""

import re
from typing import Any, ClassVar

from laraq.agent import AgentResult, BaseAgent, Message, State, agent_result
from laraq.code_utils import clean_code_output, extract_json_from_response
from laraq.llm import LLMClient


_PROMPT_HEADER = """You are a code generation assistant. Your task is to write Python code based on the user's request and the provided context.

IMPORTANT: You must respond with ONLY a JSON object in this exact format:
{
    "code": "your python code here",
    "explanation": "brief explanation of what the code does"
}

Rules:
1. The "code" field must contain valid Python code as a single string
2. Use \\n for newlines within the code string
3. The code should be complete and runnable
4. Do not include markdown formatting or code blocks
5. Only output the JSON object, nothing else

CRITICAL CODE STRUCTURE:
- Your code MUST define a single function called f() that takes no arguments
- Put ALL code INCLUDING imports INSIDE the f() function
- Do NOT put any imports at the module level (outside f())
- Do NOT include if __name__ == "__main__": blocks
- Do NOT call f() in your code - it will be called automatically
- The function MUST RETURN the result using a return statement
- Do NOT use print() - always use return to provide the result
- ALWAYS use relative file paths for any input or output files
- NEVER hardcode absolute local paths such as /Users/... or /home/...
- When reading or writing files, assume they will be staged and fetched by the execution system relative to the run directory"""

_JSON_SERIALIZABLE_RULE = """- The return value MUST be JSON-serializable: only Python primitives (int, float, str, bool, None) or standard collections of them (list, dict). Do NOT return library object instances — extract the relevant data from them first. For example, instead of returning an Atom object, return atom.get_position() or a dict of its properties"""

_PROMPT_SUFFIX = """
Example structure:
def f():
    from some_module import SomeClass
    # All your code logic here
    obj = SomeClass()
    result = obj.do_something()
    return result  # ALWAYS return, never print

CRITICAL: When the context shows you library code or API documentation:
- ALWAYS import from the existing library/module instead of redefining classes
- Produce code that just works, instead of having fallbacks for things that don't work
- Use "from ModuleName import ClassName" to import classes
- If you see a class definition in the context (e.g., "class Atom:"), that means it's an EXISTING library to import, NOT code to copy
- Only define new classes if absolutely necessary and not available in the context
- Prefer using existing libraries over reimplementing functionality
- If a task involves files, use relative paths like "data/input.xyz" or "results/output.dat", not absolute machine-specific paths"""


def _build_system_prompt(json_serializable: bool) -> str:
    parts = [_PROMPT_HEADER]
    if json_serializable:
        parts.append(_JSON_SERIALIZABLE_RULE)
    parts.append(_PROMPT_SUFFIX)
    return "\n".join(parts)


CODE_SYSTEM_PROMPT = _build_system_prompt(json_serializable=True)


class LLMCodeAgent(BaseAgent):
    """Code generation agent powered by an LLM.

    This agent:
    1. Takes context from previous messages (e.g., RAG research output)
    2. Constructs a prompt asking the LLM to generate code
    3. Parses the structured response to extract the code
    """

    name: ClassVar[str] = "llm_code"
    system_prompt: ClassVar[str] = CODE_SYSTEM_PROMPT
    is_director: ClassVar[bool] = False

    def __init__(
        self,
        llm_client: LLMClient,
        json_serializable: bool = True,
        sub_agents: list[BaseAgent] | None = None,
        **kwargs
    ) -> None:
        """Initialize the LLM code agent.

        Args:
            llm_client: Client for LLM generation
            sub_agents: Optional list of sub-agents
            **kwargs: Additional arguments passed to BaseAgent (e.g., logfile)
        """
        super().__init__(sub_agents=sub_agents, **kwargs)
        self.llm_client = llm_client
        self.last_prompt = None
        self.system_prompt = _build_system_prompt(json_serializable)

    def __call__(self, state: State | str) -> AgentResult:
        """Execute the agent to generate code.

        Args:
            state: Input state containing context and user query

        Returns:
            Dictionary with generated code as a message
        """
        # Extract messages
        if isinstance(state, str):
            messages = [Message(content=state, type="human")]
        else:
            messages = state.messages if state.messages else []

        if not messages:
            response_message = Message(content="No input provided.", type="ai")
            return agent_result(response_message)

        # Build the prompt with context
        prompt = self._build_prompt(messages)

        # Store the full prompt for logging (system + user)
        self.last_prompt = f"SYSTEM PROMPT:\n{self.system_prompt}\n\n{'='*80}\n\nUSER PROMPT:\n{prompt}"

        # Generate code using LLM
        try:
            llm_response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
            )
            code = self._parse_response(llm_response)
        except Exception as e:
            code = f"# Error generating code: {e}"

        response_message = Message(
            content=code,
            type="ai",
            additional_kwargs={"agent": "llm_code"}
        )
        return agent_result(response_message)

    def _build_prompt(self, messages: list[Message]) -> str:
        """Build the prompt for code generation.

        Args:
            messages: List of messages (user query + context)

        Returns:
            Formatted prompt string
        """
        # Find the original user query (first user message)
        user_query = ""
        research_contexts = []  # Store all research results in order
        additional_context = None
        previous_error = None
        previous_code = None
        modules = []
        physics_params = {}
        intent = ""

        for msg in messages:
            if msg.type == "human" and not user_query:
                user_query = msg.content
            elif msg.type == "ai":
                agent_name = msg.additional_kwargs.get("agent", "")

                if agent_name == "intent_extractor":
                    intent = msg.additional_kwargs.get("intent", "")

                elif agent_name == "physics_params":
                    physics_params = msg.additional_kwargs.get("params", {})

                elif agent_name == "module_mapper":
                    modules = msg.additional_kwargs.get("modules", [])

                # Collect all research results (original and retry)
                elif agent_name in ["rag_research", "naive_research"]:
                    research_contexts.append(msg.content)

                # Additional context supplied by the caller (e.g. MCP agent)
                elif agent_name == "additional_context":
                    additional_context = msg.content

                # Track most recent generated code
                elif agent_name in ["llm_code", "naive_code"]:
                    previous_code = msg.content

                # Extract most recent failure (validation or dry run)
                elif agent_name == "validator":
                    status = msg.additional_kwargs.get("status")
                    if status == "error":
                        previous_error = msg.additional_kwargs.get("error_message", msg.content)

                elif agent_name == "dry_run":
                    status = msg.additional_kwargs.get("status")
                    if status == "error":
                        previous_error = msg.additional_kwargs.get("error_message", msg.content)

        # Build clean prompt
        prompt_parts = []

        # Phase 1 analysis: intent, physics params, modules
        phase1_lines = []
        if intent:
            phase1_lines.append(f"Intent: {intent}")
        if modules:
            phase1_lines.append(f"Relevant modules (use these for imports): {', '.join(modules)}")
        if physics_params:
            params_str = ", ".join(f"{k}: {v}" for k, v in physics_params.items())
            phase1_lines.append(f"Extracted parameters: {params_str}")
        if phase1_lines:
            prompt_parts.append("## Analysis\n" + "\n".join(phase1_lines))

        # Add research context(s)
        if len(research_contexts) == 1:
            prompt_parts.append(f"## Context\n{research_contexts[0]}")
        elif len(research_contexts) > 1:
            prompt_parts.append(f"## Original Context\n{research_contexts[0]}")
            prompt_parts.append(f"## Additional Context (After Failure)\n{research_contexts[-1]}")

        if additional_context:
            prompt_parts.append(f"## Additional Context\n{additional_context}")

        prompt_parts.append(f"## Task\n{user_query}")

        # If this is a retry, show what failed and why
        if previous_error or previous_code:
            prompt_parts.append("## Previous Attempt Failed")
            if previous_code:
                prompt_parts.append(f"Code that failed:\n```python\n{previous_code}\n```")
            if previous_error:
                prompt_parts.append(f"Error:\n{previous_error}")
            if len(research_contexts) > 1:
                prompt_parts.append("Additional research context has been gathered above to help fix this issue.")

        prompt_parts.append("\nGenerate Python code to accomplish this task based on the context provided.")

        return "\n\n".join(prompt_parts)

    def _parse_response(self, response: str) -> str:
        """Parse the LLM response to extract code.

        Args:
            response: Raw LLM response

        Returns:
            Extracted code string
        """
        # Check for refusal responses
        if "refusal" in response.lower() or response.strip().startswith('{"error"'):
            raise ValueError(f"LLM refused to generate code: {response}")

        # Try to extract a JSON object from the response
        data = extract_json_from_response(response)
        if data is not None:
            if "code" in data:
                return clean_code_output(data["code"])
            elif "error" in data:
                raise ValueError(f"LLM returned error: {data['error']}")

        # Fallback: try to extract code from markdown code blocks
        code_match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # Last resort: return the raw response
        return response

    def invoke(self, messages: list[Message]) -> list[Message]:
        """Invoke the agent with messages.

        Args:
            messages: Input messages

        Returns:
            List containing generated code
        """
        state = State(messages=messages)
        result = self(state)
        return [result["message"]]

    def model_override(self, state: State | None = None) -> AgentResult | None:
        """No model override - uses LLM client directly."""
        return None
