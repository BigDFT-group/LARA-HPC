"""Local executor agent that runs Python code in-process."""

from typing import ClassVar

from laraq.agent import AgentResult, BaseAgent, Message, State, agent_result
from laraq.code_utils import extract_code_from_messages


class LocalExecutorAgent(BaseAgent):
    """Executes Python code locally by calling f() directly."""

    name: ClassVar[str] = "local_executor"
    system_prompt: ClassVar[str] = "Executes Python code locally and returns the result."
    is_director: ClassVar[bool] = False

    def __call__(self, state: State | str) -> AgentResult:
        if isinstance(state, str):
            code = state
        else:
            code = self._extract_code(state)

        output = self._execute(code)
        is_error = isinstance(output, str) and output.startswith("ERROR:")

        return agent_result(Message(
            content=str(output) if not is_error else output,
            type="ai",
            additional_kwargs={
                "status": "error" if is_error else "success",
                "agent": "local_executor",
                "executed": True,
                "result": output if not is_error else None,
            }
        ))

    def invoke(self, messages: list[Message]) -> list[Message]:
        code = messages[-1].content if messages else ""
        result = self(code)
        return [result["message"]]

    def model_override(self, state: State | None = None) -> AgentResult | None:
        return None

    def _extract_code(self, state: State) -> str:
        code = extract_code_from_messages(state.messages)
        return code if code else "ERROR: No generated code found in state — execute called before generate_code/validate"

    def _execute(self, code: str) -> object:
        import traceback
        try:
            namespace = {}
            exec(code, namespace)
            if "f" not in namespace or not callable(namespace["f"]):
                return "ERROR: Code did not define a callable function f()"
            return namespace["f"]()
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}\n\n{traceback.format_exc()}"
