"""Dry run executor agent that tests code execution with BIGDFT_MPIDRYRUN."""

import os
import traceback
from typing import Any, ClassVar

from laraq.agent import AgentResult, BaseAgent, Message, State, agent_result
from laraq.code_utils import extract_code_from_messages


class DryRunAgent(BaseAgent):
    """A dry run execution agent that tests code with BIGDFT_MPIDRYRUN.

    This agent executes code with the BIGDFT_MPIDRYRUN environment variable
    set to a non-zero integer, which causes BigDFT to run in dry run mode without actually
    executing MPI operations. We only care that the code doesn't error.
    """

    name: ClassVar[str] = "dry_run"
    system_prompt: ClassVar[str] = "Executes code in dry run mode with BIGDFT_MPIDRYRUN."
    is_director: ClassVar[bool] = False

    def __call__(self, state: State | str) -> AgentResult:
        """Execute the agent by running code in dry run mode.

        Args:
            state: Input state containing Python code to execute

        Returns:
            Dictionary with execution result (success/error)
        """
        # Convert string input to State if needed
        if isinstance(state, str):
            code = state
        else:
            # Extract the last message from state
            code = extract_code_from_messages(state.messages)

        # Execute the code in dry run mode
        success, message = self.run_dry_run(code)

        response_message = Message(
            content=message,
            type="ai",
            additional_kwargs={
                "status": "success" if success else "error",
                "agent": "dry_run",
                "dry_run": True
            }
        )
        return agent_result(response_message)

    def invoke(self, messages: list[Message]) -> list[Message]:
        """Invoke the agent to execute code in dry run mode.

        Args:
            messages: Input messages containing code

        Returns:
            List containing a single message with dry run result
        """
        # Extract code from messages (looking backwards for actual code)
        code = extract_code_from_messages(messages)

        # Execute in dry run mode
        success, message = self.run_dry_run(code)

        return [Message(
            content=message,
            type="ai",
            additional_kwargs={
                "status": "success" if success else "error",
                "agent": "dry_run",
                "dry_run": True
            }
        )]

    def model_override(self, state: State | None = None) -> AgentResult | None:
        """Override model invocation to execute code in dry run mode.

        Args:
            state: Current state

        Returns:
            Dictionary with dry run result
        """
        if state is None:
            return agent_result(Message(content="", type="ai"))

        code = extract_code_from_messages(state.messages)
        success, message = self.run_dry_run(code)

        response_message = Message(
            content=message,
            type="ai",
            additional_kwargs={
                "status": "success" if success else "error",
                "agent": "dry_run",
                "dry_run": True
            }
        )
        return agent_result(response_message)

    def run_dry_run(self, code: str, mpi_processes: int = 1) -> tuple[bool, str]:
        """Execute code in dry run mode with BIGDFT_MPIDRYRUN set to a rank count.

        Args:
            code: Python code to execute
            mpi_processes: Value to assign to BIGDFT_MPIDRYRUN

        Returns:
            Tuple of (success: bool, message: str)
        """
        import sys
        from io import StringIO

        if mpi_processes <= 0:
            return (False, "Dry run failed: mpi_processes must be a positive integer")

        # Save original environment variable
        original_value = os.environ.get('BIGDFT_MPIDRYRUN')
        old_stdout = sys.stdout
        old_path = sys.path.copy()

        try:
            # Set dry run environment variable
            os.environ['BIGDFT_MPIDRYRUN'] = str(mpi_processes)

            # Capture stdout
            sys.stdout = captured_output = StringIO()

            # Add current working directory to sys.path for imports
            cwd = os.getcwd()
            if cwd not in sys.path:
                sys.path.insert(0, cwd)

            # Create execution namespace
            exec_globals = {}

            # Execute the code (defines f() and imports)
            exec(code, exec_globals)

            # Call f() if it exists
            if 'f' in exec_globals and callable(exec_globals['f']):
                exec_globals['f']()
            else:
                return (False, "ERROR: Code did not define a callable function f()")

            # Dry run succeeded
            return (True, f"Dry run completed successfully with BIGDFT_MPIDRYRUN={mpi_processes}")

        except Exception as e:
            return (False, f"Dry run failed: {type(e).__name__}: {str(e)}\n\n{traceback.format_exc()}")

        finally:
            # Restore environment variable
            if original_value is None:
                os.environ.pop('BIGDFT_MPIDRYRUN', None)
            else:
                os.environ['BIGDFT_MPIDRYRUN'] = original_value

            # Restore stdout and sys.path
            sys.stdout = old_stdout
            sys.path = old_path
