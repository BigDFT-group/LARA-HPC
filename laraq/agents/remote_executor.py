"""Remote executor agent that runs Python code on an HPC system via remotemanager."""

from pathlib import Path
from typing import Any, ClassVar

from laraq.agent import AgentResult, BaseAgent, Message, State, agent_result
from laraq.code_utils import extract_code_from_messages


class RemoteExecutorAgent(BaseAgent):
    """Executes Python code on a remote HPC system using remotemanager.

    Submits f() to a machine managed by remotemanager. All connection
    parameters are fixed at construction time from the config file.
    """

    name: ClassVar[str] = "remote_executor"
    system_prompt: ClassVar[str] = "Executes Python code on a remote HPC system and returns the result."
    is_director: ClassVar[bool] = False

    def __init__(
        self,
        template: str | None = None,
        host: str | None = None,
        user: str | None = None,
        remote_dir: str | None = None,
        submitter: str | None = None,
        config_filename: str | None = None,
        computer_kwargs: dict[str, Any] | None = None,
        sub_agents: list[BaseAgent] | None = None,
        **kwargs,
    ) -> None:
        """Initialize the remote executor agent.

        Args:
            template: Optional job script template string (no #params#) or path to a file.
            host: SSH hostname of the remote machine. None runs locally.
            user: SSH username. None uses the current user.
            remote_dir: Working directory on the remote machine. None uses the
                        laraq project-specific default.
            submitter: Optional job scheduler command.
            config_filename: Basename of the loaded laraq config file.
            computer_kwargs: Extra keyword arguments passed to remotemanager's Computer
                             (e.g. ssh_insert, ssh_prepend).
            sub_agents: Optional list of sub-agents.
            **kwargs: Additional arguments passed to BaseAgent (e.g., logfile).
        """
        super().__init__(sub_agents=sub_agents, **kwargs)
        # If template looks like a file path, read it.
        if template is not None:
            template_path = Path(template)
            if template_path.is_file():
                self.template = template_path.read_text()
            else:
                self.template = template
        else:
            self.template = None
        self.host = host
        self.user = user
        self.remote_dir = remote_dir
        self.submitter = submitter
        self.config_filename = config_filename
        self.computer_kwargs = computer_kwargs or {}

    def __call__(self, state: State | str) -> AgentResult:
        """Execute the agent by running code on the remote machine.

        Args:
            state: Input state containing Python code to execute

        Returns:
            Dictionary with execution output as a message
        """
        if isinstance(state, str):
            code = state
        else:
            code = extract_code_from_messages(state.messages)

        output = self._execute_code(code)
        is_error = output.startswith("ERROR:")

        response_message = Message(
            content=output,
            type="ai",
            additional_kwargs={
                "status": "error" if is_error else "success",
                "agent": "remote_executor",
                "executed": True,
                "has_error": is_error,
            }
        )
        return agent_result(response_message)

    def invoke(self, messages: list[Message]) -> list[Message]:
        """Invoke the agent to execute code remotely.

        Args:
            messages: Input messages containing code

        Returns:
            List containing a single message with execution output
        """
        code = extract_code_from_messages(messages)
        output = self._execute_code(code)
        is_error = output.startswith("ERROR:")

        return [Message(
            content=output,
            type="ai",
            additional_kwargs={
                "status": "error" if is_error else "success",
                "agent": "remote_executor",
                "executed": True,
                "has_error": is_error,
            }
        )]

    def model_override(self, state: State | None = None) -> AgentResult | None:
        """No model override - executes code directly via remotemanager."""
        return None

    def check_connection(self) -> tuple[bool, str]:
        """Validate that the remotemanager execution target is usable.

        Returns:
            Tuple of (success, message)
        """
        try:
            from remotemanager import Computer
        except ImportError:
            return (False, "remotemanager is not installed")

        try:
            ckwargs = self._build_computer_kwargs()
            computer = Computer(**ckwargs)

            test_connection = getattr(computer, "test_connection", None)
            if callable(test_connection):
                result = test_connection()
                if result is False:
                    return (False, "remotemanager connection test failed")
                if isinstance(result, str) and result.strip():
                    return (True, result)
                return (True, "remotemanager connection test passed")

            # Fall back to successful construction when the installed
            # remotemanager version does not expose a test helper.
            return (True, "remotemanager target initialized")
        except Exception as e:
            return (False, f"{type(e).__name__}: {e}")

    def _build_computer_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Build kwargs dict for remotemanager Computer.

        Args:
            **overrides: Runtime overrides for template placeholders
                         (e.g. nodes=4, walltime="02:00:00").
        """
        ckwargs: dict[str, Any] = dict(self.computer_kwargs)
        if self.template is not None:
            ckwargs["template"] = self.template
        if self.submitter is not None:
            ckwargs["submitter"] = self.submitter
        if self.host is not None:
            ckwargs["host"] = self.host
        if self.user is not None:
            ckwargs["user"] = self.user
        ckwargs.update(overrides)
        return ckwargs

    def _get_extra_files_send(self) -> list[str]:
        """Build an explicit list of top-level entries to stage remotely."""
        local_dir = "laraq_local"
        excluded_names = {local_dir}
        if self.config_filename:
            excluded_names.add(self.config_filename)

        entries: list[str] = []
        for entry in Path.cwd().iterdir():
            if entry.name in excluded_names:
                continue
            if entry.name.startswith("."):
                continue
            if entry.name.startswith("dataset-"):
                continue
            entries.append(entry.name)

        return sorted(entries)

    def _execute_code(self, code: str, **overrides: Any) -> str:
        """Submit code to the remote machine and return the result of f().

        Args:
            code: Python code defining a function f() with no arguments.
            **overrides: Runtime overrides for template placeholders
                         (e.g. nodes=4, walltime="02:00:00").

        Returns:
            String result of f(), or an ERROR:-prefixed message on failure.
        """
        try:
            from remotemanager import Computer, Dataset
        except ImportError:
            return "ERROR: remotemanager is not installed. Run: pip install remotemanager"

        import importlib.util
        import tempfile

        tmp_path = None
        try:
            # Write code to a temp file so remotemanager can inspect its source.
            # exec()-defined functions have no source file, causing inspect.getsource() to fail.
            with tempfile.NamedTemporaryFile(
                suffix=".py", mode="w", delete=False, prefix="laraq_"
            ) as tmp:
                tmp.write(code)
                tmp_path = tmp.name

            spec = importlib.util.spec_from_file_location("_laraq_tmp", tmp_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            if not hasattr(mod, "f") or not callable(mod.f):
                return "ERROR: Code did not define a callable function f()"
            f = mod.f

            url = Computer(**self._build_computer_kwargs(**overrides))

            dataset_kwargs: dict[str, Any] = {
                "function": f,
                "url": url,
                "skip": False,
                "local_dir": "laraq_local",
                "remote_dir": self.remote_dir if self.remote_dir is not None else "laraq_remote",
                "extra_files_send": self._get_extra_files_send(),
                "extra_files_recv": "*",
            }
            ds = Dataset(**dataset_kwargs)
            ds.append_run(arguments={})
            ds.run()
            ds.wait(30, None)
            ds.fetch_results()

            result = ds.results[0]
            ds.wipe_runs(confirm=False)
            return str(result) if result is not None else "(No output)"

        except Exception as e:
            return f"ERROR: {type(e).__name__}: {str(e)}"

        finally:
            if tmp_path is not None:
                Path(tmp_path).unlink(missing_ok=True)
