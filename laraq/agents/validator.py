"""Validator agent that checks Python code using pyflakes."""

import ast
import sys
from io import StringIO
from typing import Any, ClassVar

from laraq.agent import AgentResult, BaseAgent, Message, State, agent_result
from laraq.code_utils import clean_code_output

try:
    from pyflakes.api import check as pyflakes_check
    from pyflakes.reporter import Reporter
    PYFLAKES_AVAILABLE = True
except ImportError:
    PYFLAKES_AVAILABLE = False


class ValidatorAgent(BaseAgent):
    """A validation agent that checks Python code using pyflakes.

    This agent uses pyflakes to validate code, catching:
    - Syntax errors
    - Undefined variables
    - Missing imports
    - Unused imports
    - And other logical errors

    Falls back to AST parsing if pyflakes is not available.
    """

    name: ClassVar[str] = "validator"
    system_prompt: ClassVar[str] = "Validates Python code using pyflakes static analysis."
    is_director: ClassVar[bool] = False

    def __call__(self, state: State | str) -> AgentResult:
        """Execute the agent by validating Python code.

        Args:
            state: Input state containing Python code to validate

        Returns:
            Dictionary with validation result or the original code
        """
        # Convert string input to State if needed
        if isinstance(state, str):
            code = state
        else:
            # Extract the last message from state
            code = self._extract_last_message(state)

        # Validate the code
        validation_result = self._validate_code(code)

        if validation_result["valid"]:
            # Code is valid, pass it through with success metadata
            response_message = Message(
                content=code,
                type="ai",
                additional_kwargs={
                    "status": "success",
                    "agent": "validator",
                    "validated": True
                }
            )
        else:
            # Code has syntax errors - include error metadata
            error_msg = f"VALIDATION ERROR: {validation_result['error']}\n\nOriginal code:\n{code}"
            response_message = Message(
                content=error_msg,
                type="ai",
                additional_kwargs={
                    "status": "error",
                    "agent": "validator",
                    "validated": False,
                    "error_type": "validation_error",
                    "error_message": validation_result['error']
                }
            )

        return agent_result(response_message)

    def invoke(self, messages: list[Message]) -> list[Message]:
        """Invoke the agent to validate code.

        Args:
            messages: Input messages containing code

        Returns:
            List containing a single message with validation result
        """
        # Get the last message content (the code)
        code = messages[-1].content if messages else ""

        # Validate the code
        validation_result = self._validate_code(code)

        if validation_result["valid"]:
            return [Message(
                content=code,
                type="ai",
                additional_kwargs={
                    "status": "success",
                    "agent": "validator",
                    "validated": True
                }
            )]
        else:
            error_msg = f"VALIDATION ERROR: {validation_result['error']}\n\nOriginal code:\n{code}"
            return [Message(
                content=error_msg,
                type="ai",
                additional_kwargs={
                    "status": "error",
                    "agent": "validator",
                    "validated": False,
                    "error_type": "validation_error",
                    "error_message": validation_result['error']
                }
            )]

    def model_override(self, state: State | None = None) -> AgentResult | None:
        """Override model invocation to validate code directly.

        This agent doesn't need to call an LLM - it just validates
        the Python code using ast.parse().

        Args:
            state: Current state

        Returns:
            Dictionary with validation result
        """
        if state is None:
            return agent_result(Message(content="", type="ai"))

        code = self._extract_last_message(state)
        validation_result = self._validate_code(code)

        if validation_result["valid"]:
            response_message = Message(
                content=code,
                type="ai",
                additional_kwargs={
                    "status": "success",
                    "agent": "validator",
                    "validated": True
                }
            )
        else:
            error_msg = f"VALIDATION ERROR: {validation_result['error']}\n\nOriginal code:\n{code}"
            response_message = Message(
                content=error_msg,
                type="ai",
                additional_kwargs={
                    "status": "error",
                    "agent": "validator",
                    "validated": False,
                    "error_type": "validation_error",
                    "error_message": validation_result['error']
                }
            )

        return agent_result(response_message)

    def _extract_last_message(self, state: State) -> str:
        """Extract the last message from state.

        Args:
            state: Current conversation state

        Returns:
            Content of the last message
        """
        if not state.messages:
            return ""

        return state.messages[-1].content

    def _validate_code(self, code: str) -> dict[str, Any]:
        """Validate Python code using pyflakes or AST parsing.

        Args:
            code: Python code to validate

        Returns:
            Dictionary with 'valid' bool and optional 'error' message
        """
        if not code or not code.strip():
            return {"valid": False, "error": "Empty code provided"}

        # Check for error messages or refusals
        code_stripped = code.strip()
        if code_stripped.startswith('{"error"') or "refusal" in code_stripped.lower():
            return {
                "valid": False,
                "error": f"Code generation failed: {code_stripped[:100]}"
            }

        # Check if it's JSON and try to extract code from it
        if code_stripped.startswith('{'):
            from laraq.code_utils import extract_json_from_response

            data = extract_json_from_response(code_stripped)
            if data is not None:
                if "code" in data:
                    return self._validate_code(clean_code_output(code_stripped))
                else:
                    return {
                        "valid": False,
                        "error": "Received JSON object without 'code' field"
                    }

        # Check custom style rules first
        style_check = self._check_style_rules(code)
        if not style_check["valid"]:
            return style_check

        # Check if imports are valid
        import_check = self._check_imports(code)
        if not import_check["valid"]:
            return import_check

        # Use pyflakes if available
        if PYFLAKES_AVAILABLE:
            return self._validate_with_pyflakes(code)
        else:
            # Fall back to AST parsing
            return self._validate_with_ast(code)

    def _validate_with_pyflakes(self, code: str) -> dict[str, Any]:
        """Validate code using pyflakes.

        Args:
            code: Python code to validate

        Returns:
            Dictionary with 'valid' bool and optional 'error' message
        """
        # Capture pyflakes output
        output = StringIO()
        reporter = Reporter(output, sys.stderr)

        # Run pyflakes check
        warning_count = pyflakes_check(code, '<generated>', reporter=reporter)

        if warning_count == 0:
            return {"valid": True}
        else:
            # Get the errors/warnings
            errors = output.getvalue().strip()
            # Parse and format the errors
            error_lines = errors.split('\n')
            # Take first few errors to avoid overwhelming output
            if len(error_lines) > 5:
                error_summary = '\n'.join(error_lines[:5]) + f"\n... and {len(error_lines) - 5} more errors"
            else:
                error_summary = '\n'.join(error_lines)

            return {
                "valid": False,
                "error": f"Pyflakes found {warning_count} issue(s):\n{error_summary}"
            }

    def _validate_with_ast(self, code: str) -> dict[str, Any]:
        """Validate code using AST parsing (fallback).

        Args:
            code: Python code to validate

        Returns:
            Dictionary with 'valid' bool and optional 'error' message
        """
        try:
            # Try to parse the code
            ast.parse(code)
            return {"valid": True}
        except SyntaxError as e:
            return {
                "valid": False,
                "error": f"SyntaxError at line {e.lineno}: {e.msg}"
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"{type(e).__name__}: {str(e)}"
            }

    def _check_style_rules(self, code: str) -> dict[str, Any]:
        """Check custom style rules for the code.

        Args:
            code: Python code to check

        Returns:
            Dictionary with 'valid' bool and optional 'error' message
        """
        # No custom style rules currently defined
        return {"valid": True}

    def _check_imports(self, code: str) -> dict[str, Any]:
        """Check if all imports in the code are valid.

        Args:
            code: Python code to check

        Returns:
            Dictionary with 'valid' bool and optional 'error' message
        """
        import importlib.util

        try:
            tree = ast.parse(code)
        except SyntaxError:
            # If we can't parse, let pyflakes handle it
            return {"valid": True}

        # Extract all import statements
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                # from: import module, module2
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                # from module import name
                if node.module:
                    imports.append(node.module)

        # Check each import
        invalid_imports = []
        for module_name in imports:
            # Try to find the module spec
            try:
                spec = importlib.util.find_spec(module_name)
                if spec is None:
                    invalid_imports.append(module_name)
            except (ModuleNotFoundError, ValueError, ImportError):
                invalid_imports.append(module_name)

        if invalid_imports:
            return {
                "valid": False,
                "error": f"Invalid import(s): {', '.join(invalid_imports)}. These modules are not installed or do not exist."
            }

        return {"valid": True}
