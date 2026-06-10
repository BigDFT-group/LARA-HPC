"""Utilities for handling generated Python code."""

import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from laraq.agent import Message


def extract_json_from_response(response: str) -> dict[str, Any] | None:
    """Extract a JSON object from an LLM response.

    Handles common LLM output patterns:
    - Raw JSON
    - JSON wrapped in markdown code fences
    - JSON embedded in surrounding text

    Returns the parsed dict, or None if no valid JSON object is found.
    """
    response = response.strip()

    # Strip markdown code fences if present
    fence_match = re.match(r'^```(?:json)?\s*\n(.*?)\n```\s*$', response, re.DOTALL)
    if fence_match:
        response = fence_match.group(1).strip()

    # Try to parse the entire response as JSON
    try:
        data = json.loads(response)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object with balanced braces
    start = response.find("{")
    if start != -1:
        brace_count = 0
        end = start
        for i in range(start, len(response)):
            if response[i] == "{":
                brace_count += 1
            elif response[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break
        if end > start:
            try:
                data = json.loads(response[start:end])
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

    return None


def extract_code_from_messages(messages: "list[Message]") -> str:
    """Extract the most recent validated or generated code from a message list."""
    for msg in reversed(messages):
        agent_type = msg.additional_kwargs.get("agent", "")
        if agent_type in ["llm_code", "naive_code", "validator"]:
            if agent_type == "validator":
                if msg.additional_kwargs.get("status") == "success":
                    return clean_code_output(msg.content)
            else:
                return clean_code_output(msg.content)
    return ""


def clean_code_output(code: str) -> str:
    """Remove common LLM wrappers from generated Python code."""
    code = code.strip()

    fence_match = re.match(r"^```(?:json)?\s*\n(.*?)\n```\s*$", code, re.DOTALL)
    if fence_match:
        code = fence_match.group(1).strip()

    if code.startswith("{"):
        try:
            data = json.loads(code)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(data, dict) and "code" in data:
                code = data["code"]

    return code.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
