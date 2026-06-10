"""Shared setup validation for CLI and MCP entrypoints."""

from typing import Any

from laraq.config.agents import create_agents_from_config
from laraq.embeddings import create_embedding_client
from laraq.llm import create_llm_client


def run_setup_checks(cfg, executor_agent=None) -> dict[str, Any]:
    """Run the full laraq setup validation sequence."""
    results: dict[str, Any] = {
        "ready": False,
        "llm_provider": cfg.agent.llm_provider,
        "embedding_provider": cfg.agent.embedding_provider,
        "checks": [],
    }

    try:
        llm_client = create_llm_client(cfg)
        response = llm_client.generate("Say 'test'", system_prompt="You are a helpful assistant.")
        preview = response[:50] + "..." if len(response) > 50 else response
        results["checks"].append({"name": "llm", "ok": True, "message": preview})
    except Exception as e:
        results["checks"].append({"name": "llm", "ok": False, "message": str(e)})
        return results

    try:
        embedding_client = create_embedding_client(cfg)
        embedding = embedding_client.embed_one("test")
        results["checks"].append({
            "name": "embedding",
            "ok": True,
            "message": f"Embedding dimension: {len(embedding)}",
        })
    except Exception as e:
        results["checks"].append({"name": "embedding", "ok": False, "message": str(e)})
        return results

    try:
        import BigDFT  # noqa: F401

        results["checks"].append({
            "name": "bigdft",
            "ok": True,
            "message": "BigDFT import succeeded",
        })
    except Exception as e:
        results["checks"].append({"name": "bigdft", "ok": False, "message": str(e)})
        return results

    try:
        if getattr(cfg, "template", "bigdft_remote") == "bigdft_local":
            results["checks"].append({
                "name": "execution",
                "ok": True,
                "message": "Local execution (no connection required)",
            })
        else:
            if executor_agent is None:
                agents = create_agents_from_config(cfg)
                executor_agent = agents["executor"]

            if hasattr(executor_agent, "check_connection"):
                success, message = executor_agent.check_connection()
                if not success:
                    raise RuntimeError(message)
                results["checks"].append({"name": "execution", "ok": True, "message": message})
            else:
                results["checks"].append({
                    "name": "execution",
                    "ok": True,
                    "message": "Executor does not expose a connection test",
                })
    except Exception as e:
        results["checks"].append({"name": "execution", "ok": False, "message": str(e)})
        return results

    results["ready"] = True
    return results
