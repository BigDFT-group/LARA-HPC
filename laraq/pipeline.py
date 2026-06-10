"""Core agent pipeline logic shared between CLI run and test commands."""

from dataclasses import dataclass, field
from typing import Any, Callable

from laraq.agent import Message, State
from laraq.code_utils import extract_code_from_messages


@dataclass
class PipelineResult:
    """Result of a pipeline run."""

    success: bool
    state: State
    code: str
    error: str | None
    flow: list[str] = field(default_factory=list)


def run_pipeline(
    agents: dict[str, Any],
    query: str,
    progress: Callable[[str], None] | None = None,
    max_retries: int = 1,
) -> PipelineResult:
    """Run the full agent pipeline for a query.

    Covers Phase 1 (understanding: intent, physics, module mapping),
    Phase 2 (research → code → validate with up to max_retries retries), and
    Phase 3 (dry run, also retried within the same budget).

    Execution against the real runtime is the caller's responsibility.

    Args:
        agents: Dict of agent instances from create_agents_from_config
        query: The user's natural language query
        progress: Optional callable for progress messages (e.g. click.echo)
        max_retries: Maximum number of research→code→validate→dry-run cycles after the first

    Returns:
        PipelineResult with success status, final state, validated code, and flow
    """
    def _p(msg: str) -> None:
        if progress:
            progress(msg)

    state = State(messages=[Message(content=query, type="human")])
    flow = []
    search_query = query  # may be enhanced by intent_extractor hints

    _p("Phase 1: Understanding")

    _p("  Intent extraction...")
    result = agents["intent_extractor"](query)
    msg = result["message"]
    state.messages.append(msg)
    agents["intent_extractor"].log_interaction(query, msg.content)
    hints = msg.additional_kwargs.get("search_hints", [])
    if hints:
        search_query = f"{query}\n\nSearch hints: {', '.join(hints)}"
    flow.append("intent_extractor")

    _p("  Physics parameter extraction...")
    result = agents["physics_params"](query)
    msg = result["message"]
    state.messages.append(msg)
    agents["physics_params"].log_interaction(query, msg.content)
    flow.append("physics_params")

    _p("  Module mapping...")
    result = agents["module_mapper"](query)
    msg = result["message"]
    state.messages.append(msg)
    agents["module_mapper"].log_interaction(query, msg.content)
    if not msg.additional_kwargs.get("feasible", True):
        reason = msg.additional_kwargs.get("reason", "")
        flow.append("module_mapper → not feasible")
        return PipelineResult(
            success=False,
            state=state,
            code="",
            error=f"Not feasible: {reason}",
            flow=flow,
        )
    modules = msg.additional_kwargs.get("modules", [])
    if modules:
        search_query += f"\nRelevant modules: {', '.join(modules)}"
    flow.append("module_mapper")

    _p("Phase 2: Generation")

    validator_result = None
    dry_run_result = None

    for attempt in range(max_retries + 1):
        is_retry = attempt > 0
        is_last = attempt == max_retries

        agents["director"].write_log(
            f"\n{'#' * 80}\n### ATTEMPT {attempt + 1} of {max_retries + 1}\n{'#' * 80}\n"
        )

        _p("  Research → Code...")
        if attempt == 0:
            flow.append("research → code → validate")

        research_input = search_query if not is_retry else state
        research_result = agents["research"](research_input)
        state.messages.append(research_result["message"])
        log_input = search_query if not is_retry else (dry_run_result or validator_result)["message"].content
        agents["research"].log_interaction(log_input, research_result["message"].content)

        code_result = agents["code"](state)
        state.messages.append(code_result["message"])
        prompt_sent = getattr(agents["code"], "last_prompt", research_result["message"].content)
        agents["code"].log_interaction(prompt_sent, code_result["message"].content)

        if not is_retry:
            _p("Phase 3: Validation")
        _p("  Syntax + structure check...")

        validator_result = agents["validator"](state)
        state.messages.append(validator_result["message"])
        agents["validator"].log_interaction(
            code_result["message"].content,
            validator_result["message"].content,
        )

        validation_status = validator_result["message"].additional_kwargs.get("status", "success")

        if validation_status != "success":
            error_detail = validator_result["message"].additional_kwargs.get("error_message", validator_result["message"].content)
            if is_last:
                flow.append("validation failed")
                return PipelineResult(
                    success=False,
                    state=state,
                    code=extract_code_from_messages(state.messages),
                    error=f"Code validation failed after {max_retries + 1} attempt(s): {error_detail}",
                    flow=flow,
                )
            _p("  Validation failed, retrying...")
            flow.append(f"validation failed → retry {attempt + 1}")
            continue

        _p("  Dry run...")
        dry_run_result = agents["dry_run"](state)
        state.messages.append(dry_run_result["message"])
        agents["dry_run"].log_interaction(
            validator_result["message"].content,
            dry_run_result["message"].content,
        )
        dry_run_status = dry_run_result["message"].additional_kwargs.get("status", "success")

        if dry_run_status != "success":
            if is_last:
                flow.append("dry run failed")
                return PipelineResult(
                    success=False,
                    state=state,
                    code=extract_code_from_messages(state.messages),
                    error=f"Dry run failed after {max_retries + 1} attempt(s): {dry_run_result['message'].content}",
                    flow=flow,
                )
            _p("  Dry run failed, retrying...")
            flow.append(f"dry run failed → retry {attempt + 1}")
            continue

        flow.append("done")
        return PipelineResult(
            success=True,
            state=state,
            code=extract_code_from_messages(state.messages),
            error=None,
            flow=flow,
        )

    # Unreachable — loop always returns, but satisfies type checkers
    return PipelineResult(success=False, state=state, code="", error="Pipeline exhausted retries", flow=flow)
