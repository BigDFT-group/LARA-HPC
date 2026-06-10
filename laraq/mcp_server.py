"""FastMCP server for laraq.

Exposes the agent pipeline as composable tools so the calling LLM
can orchestrate the flow:

    check_server → extract_intent → extract_physics → map_modules → research → generate_code → validate → dry_run → execute

extract_intent, extract_physics, and map_modules are optional and only active when
the corresponding agent is configured.

The execute tool's signature is built dynamically at startup from the job
template placeholders so the agent sees concrete parameter names and defaults.
"""

import importlib.util
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional

from fastmcp import FastMCP

from laraq.agent import State, Message
from laraq.setup_checks import run_setup_checks

mcp = FastMCP("laraq")

_agents: dict[str, Any] | None = None
_cfg: Any | None = None
_lock = threading.Lock()


@mcp.tool()
def check_server() -> dict:
    """STEP 1 of 9: Verify that the loaded laraq server is ready to use.

    Call this first in a fresh session before research or code generation.

    This checks:
    - LLM connectivity
    - embedding connectivity
    - BigDFT importability
    - remotemanager execution connectivity

    If ready is false, stop and report the setup issue to the user instead
    of continuing with code generation. If the failure is a BigDFT import
    error, tell the user they need PyBigDFT and BigDFT installed in the local
    environment running the server.
    """
    with _lock:
        return run_setup_checks(_cfg, executor_agent=_agents["executor"])


@mcp.tool()
def extract_intent(query: str) -> dict:
    """STEP 2 of 9: Parse the user request into structured intent and search hints.

    Call this after check_server and before extract_physics.

    The returned search_hints should be used to inform the research query — they are
    documentation-style terms (class names, method names, API concepts) that will
    retrieve more targeted results than the raw user query.

    Returns:
        {"intent": "...", "calculation_type": "...", "search_hints": ["term1", ...]}
    """
    with _lock:
        result = _agents["intent_extractor"](query)
        msg = result["message"]
        return {
            "intent": msg.additional_kwargs.get("intent", ""),
            "calculation_type": msg.additional_kwargs.get("calculation_type", "other"),
            "search_hints": msg.additional_kwargs.get("search_hints", []),
        }


@mcp.tool()
def extract_physics(query: str) -> dict:
    """STEP 3 of 9: Extract physical and scientific parameters from the user request.

    Call this after extract_intent and before map_modules.

    The returned parameters (elements, positions, units, functional, etc.) should be
    passed as part of additional_context to generate_code so the code generator does
    not need to re-extract them from the query.

    Returns:
        A flat dictionary of parameter names to values, e.g.
        {"elements": ["H", "H"], "units": "bohr", "functional": "LDA"}
        Returns {} if no parameters are found.
    """
    with _lock:
        result = _agents["physics_params"](query)
        msg = result["message"]
        return msg.additional_kwargs.get("params", {})


@mcp.tool()
def map_modules(query: str) -> dict:
    """STEP 4 of 9: Check whether the request is feasible and identify relevant modules.

    Call this after extract_physics and before research.

    If feasible is false, stop and report the reason to the user — do not proceed
    to research or code generation.

    If feasible is true, pass the returned modules list to generate_code as part
    of additional_context so the code generator knows which imports to use.

    Returns:
        {"feasible": true/false, "modules": ["pkg.X", ...], "reason": "<explanation>"}
    """
    with _lock:
        result = _agents["module_mapper"](query)
        msg = result["message"]
        return {
            "feasible": msg.additional_kwargs.get("feasible", True),
            "modules": msg.additional_kwargs.get("modules", []),
            "reason": msg.additional_kwargs.get("reason", ""),
        }


@mcp.tool()
def research(query: str) -> str:
    """STEP 5 of 9: Search documentation for context relevant to the query.

    Uses RAG (semantic search over embedded docs) plus LLM query expansion to
    find the most relevant documentation chunks.

    Always call check_server first. Then call research before generate_code.
    The returned context string must be passed as the `context` argument to
    generate_code.

    If generate_code or validate later fails, call research again (optionally
    with a more specific query) to gather additional context before retrying.
    """
    with _lock:
        result = _agents["research"](query)
        return result["message"].content


@mcp.tool()
def generate_code(query: str, context: str, additional_context: str = "") -> str:
    """STEP 6 of 9: Generate Python code for the query using research context.

    Call check_server, then map_modules, then research, and pass research's output
    as `context`.

    Use `additional_context` to supply any extra information beyond the RAG
    results. If map_modules returned a non-empty modules list, include it here
    so the code generator knows which imports to use. The most useful thing to
    put here is example code you have already used or seen in this session.
    Leave empty if not needed.

    The generated code defines a single f() function with all imports inside it
    and returns the result (no print statements). Do not execute this code
    directly — pass it to validate next.

    On retry (after a validate or dry_run failure), call research again with a
    more targeted query, then call generate_code again with the new context.
    """
    with _lock:
        # LLMCodeAgent._build_prompt scans messages by agent tag to find context
        # and the user query, so we build a minimal State with the right tags.
        messages = [
            Message(content=query, type="human"),
            Message(content=context, type="ai", additional_kwargs={"agent": "rag_research"}),
        ]
        if additional_context:
            messages.append(Message(
                content=additional_context,
                type="ai",
                additional_kwargs={"agent": "additional_context"},
            ))
        result = _agents["code"](State(messages=messages))
        return result["message"].content


@mcp.tool()
def validate(code: str) -> dict:
    """STEP 7 of 9: Validate generated code using pyflakes static analysis and import checks.

    Call this after generate_code and before dry_run. Pass the raw output
    of generate_code as `code`.

    Returns:
        {"valid": true, "code": "<cleaned code>"} on success.
        {"valid": false, "error": "<message>", "code": "<original>"} on failure.

    IMPORTANT: Always use the returned "code" field (not your copy of the input)
    for all subsequent dry_run and execute calls. The validator cleans up JSON
    wrapping and escape sequences that the LLM may have added.

    If valid is false, call research again then generate_code again with fresh
    context before retrying validate.
    """
    with _lock:
        result = _agents["validator"](code)
        msg = result["message"]
        if msg.additional_kwargs.get("status") == "success":
            return {"valid": True, "code": msg.content}
        else:
            return {
                "valid": False,
                "error": msg.additional_kwargs.get("error_message", msg.content),
                "code": code,
            }


@mcp.tool()
def dry_run(code: str, mpi_processes: int = 1) -> dict:
    """STEP 8 of 9: Test code execution with BIGDFT_MPIDRYRUN set to an MPI count.

    Call this after validate succeeds. Pass the "code" field from validate's
    response (not the raw generate_code output).

    This catches runtime import errors, missing attributes, and BigDFT
    configuration issues without running a full simulation. The default
    mpi_processes=1 is the standard dry run. Use larger values when you want
    BigDFT to estimate requirements for a larger MPI count.

    Dry run writes its log files into the current working directory. Those
    files are not part of the remote execution artifact sync into laraq_local.
    The reported memory information is per MPI process. Read the dry-run log
    immediately after a successful dry run, because later runs may overwrite
    the same top-level log filename.

    Returns:
        {"success": true} on success — proceed to execute.
        {"success": false, "error": "<message>"} on failure — call research
        and generate_code again with fresh context before retrying.
    """
    with _lock:
        success, message = _agents["dry_run"].run_dry_run(code, mpi_processes=mpi_processes)
        if success:
            return {"success": True, "message": message, "mpi_processes": mpi_processes}
        else:
            return {"success": False, "error": message, "mpi_processes": mpi_processes}


# The execute tool is registered dynamically in start_mcp_server() so its
# signature reflects the job template placeholders from the config.


@mcp.resource("log://lara_chat")
def get_log() -> str:
    """Get the current agent interaction log."""
    log_path = Path("lara_chat.log")
    if log_path.exists():
        return log_path.read_text()
    return "No log available."


def _get_template_params(executor) -> tuple[list[str], dict[str, Any]]:
    """Extract template placeholder names and config defaults from an executor.

    Returns:
        (param_names, config_defaults) where param_names is a list of
        placeholder names from the template and config_defaults maps
        those names to the values set in the [remote] config section.
    """
    from laraq.agents.remote_executor import RemoteExecutorAgent

    if not isinstance(executor, RemoteExecutorAgent):
        return [], {}

    if not executor.template:
        return [], {}

    try:
        from remotemanager import Computer
        computer = Computer(template=executor.template, host=executor.host or "localhost")
        param_names = list(computer.args)
    except Exception:
        return [], {}

    # Config defaults come from computer_kwargs (the extra [remote] fields)
    config_defaults = {
        name: executor.computer_kwargs[name]
        for name in param_names
        if name in executor.computer_kwargs
    }

    return param_names, config_defaults


def _build_execute_function(
    param_names: list[str],
    config_defaults: dict[str, Any],
):
    """Dynamically build an execute function with template params in its signature.

    FastMCP infers the JSON schema from the function signature, so generating
    a real function with typed parameters is the cleanest way to expose
    template placeholders to the agent.
    """
    # Build parameter list: code is always first, then template params
    param_parts = []
    for name in param_names:
        if name in config_defaults:
            default = config_defaults[name]
            if isinstance(default, str):
                param_parts.append(f"{name}: Optional[str] = {default!r}")
            elif isinstance(default, int):
                param_parts.append(f"{name}: Optional[int] = {default!r}")
            else:
                param_parts.append(f"{name}: Optional[str] = {default!r}")
        else:
            param_parts.append(f"{name}: Optional[int] = None")

    params_str = ", ".join(param_parts)
    if params_str:
        params_str = f", {params_str}"

    # Build the override dict construction
    override_lines = "\n".join(
        f"    if {name} is not None: overrides['{name}'] = {name}"
        for name in param_names
    )

    # Build param docs
    param_docs = "\n".join(
        f"        {name}: Job template parameter (default: {config_defaults.get(name, 'None')})"
        for name in param_names
    )
    if param_docs:
        param_docs = f"\n{param_docs}"

    func_code = f'''
def execute(code: str{params_str}) -> str:
    """STEP 9 of 9: Execute validated Python code and return the result.

    Only call this after validate and dry_run have succeeded and you have shown
    the user the generated code and received their explicit approval.
    Pass the "code" field from validate's response.

    Do NOT skip validate or dry_run before calling this — skipping them risks
    running broken code against the real BigDFT runtime.

    Args:
        code: Validated Python code to execute{param_docs}

    Returns the string result of f(), or an ERROR:-prefixed message on failure.
    """
    overrides = {{}}
{override_lines}
    with _lock:
        output = _agents["executor"]._execute_code(code, **overrides)
        return output
'''

    # Write to temp file so inspect.getsource() works (FastMCP needs this)
    temp_dir = Path(tempfile.gettempdir()) / "laraq_mcp_tools"
    temp_dir.mkdir(exist_ok=True)
    temp_file = temp_dir / "execute.py"
    temp_file.write_text(func_code)

    spec = importlib.util.spec_from_file_location("_laraq_execute", str(temp_file))
    module = importlib.util.module_from_spec(spec)
    # Inject globals the function needs
    module._lock = _lock
    module._agents = None  # Set later in start_mcp_server
    module.Optional = Optional
    spec.loader.exec_module(module)

    return module.execute, module


def _build_simple_execute():
    """Build a simple execute function for bigdft_local (no template params)."""
    def execute(code: str) -> str:
        """STEP 9 of 9: Execute validated Python code and return the result.

        Only call this after validate and dry_run have succeeded and you have shown
        the user the generated code and received their explicit approval.
        Pass the "code" field from validate's response.

        Do NOT skip validate or dry_run before calling this — skipping them risks
        running broken code against the real BigDFT runtime.

        Returns the string result of f(), or an ERROR:-prefixed message on failure.
        """
        with _lock:
            result = _agents["executor"](code)
            return result["message"].content

    return execute


def start_mcp_server(cfg) -> None:
    """Initialize agents and start the FastMCP server over stdio.

    Args:
        cfg: Loaded laraq configuration object
    """
    global _agents, _cfg

    from laraq.config.agents import create_agents_from_config

    # Redirect stdout → stderr during init so print() calls don't corrupt
    # the MCP stdio transport (which uses stdout for protocol messages).
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        _cfg = cfg
        _agents = create_agents_from_config(cfg)
        for agent in _agents.values():
            agent.init_agent()

        # Build and register the execute tool dynamically
        param_names, config_defaults = _get_template_params(_agents["executor"])

        if param_names:
            execute_fn, execute_module = _build_execute_function(param_names, config_defaults)
            # Wire the module's _agents reference to the real agents dict
            execute_module._agents = _agents
            mcp.tool(execute_fn)
        else:
            mcp.tool(_build_simple_execute())

    finally:
        sys.stdout = old_stdout

    mcp.run()
