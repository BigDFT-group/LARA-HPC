"""CLI for laraq."""

import shutil
import sys
import traceback
from pathlib import Path

import click

from laraq.config.agents import create_agents_from_config
from laraq.config.loader import load_config
from laraq.exceptions import ConfigError
from laraq.cli.test_runner import load_tests, execute_code, compare_outputs
from laraq.pipeline import run_pipeline
from laraq.setup_checks import run_setup_checks


def mask_api_key(api_key: str) -> str:
    """Mask an API key for display.

    Shows first 8 and last 4 characters, masks the rest.
    """
    if len(api_key) <= 12:
        return "***"
    return f"{api_key[:8]}...{api_key[-4:]}"


@click.group(invoke_without_command=True)
@click.option(
    "--config",
    "-c",
    default="laraq.toml",
    type=click.Path(exists=False),
    help="Path to configuration file",
)
@click.pass_context
def cli(ctx, config):
    """laraq - Agent-driven code generation system.

    Validates and displays configuration for AI providers.
    """
    if ctx.invoked_subcommand is None:
        # Default behavior: validate config
        ctx.invoke(validate, config=config)


@cli.command()
@click.option(
    "--config",
    "-c",
    default="laraq.toml",
    type=click.Path(exists=False),
    help="Path to configuration file",
)
def validate(config):
    """Validate the configuration file."""
    config_path = Path(config)

    click.echo(f"Loading configuration from: {config_path}")
    click.echo()

    try:
        cfg = load_config(config_path)
    except ConfigError as e:
        click.echo(click.style("Configuration Error:", fg="red", bold=True))
        click.echo(click.style(str(e), fg="red"))
        sys.exit(1)
    except Exception as e:
        click.echo(click.style("Unexpected Error:", fg="red", bold=True))
        click.echo(click.style(str(e), fg="red"))
        sys.exit(1)

    # Display validation success
    click.echo(click.style("Configuration Valid!", fg="green", bold=True))
    click.echo()

    # Display LLM provider
    llm_provider = cfg.agent.llm_provider
    embedding_provider = cfg.agent.embedding_provider

    click.echo(f"LLM Provider: {click.style(llm_provider, fg='cyan', bold=True)}")
    llm_config = cfg.get_llm_config()

    click.echo("  Configuration:")
    if llm_provider == "ollama":
        click.echo(f"    Base URL: {llm_config.base_url}")
        click.echo(f"    Model: {llm_config.model}")
    elif llm_provider == "openai-compatible":
        click.echo(f"    Base URL: {llm_config.base_url}")
        click.echo(f"    API Key: {mask_api_key(llm_config.api_key)}")
        click.echo(f"    Model: {llm_config.model}")
    elif llm_provider == "anthropic":
        click.echo(f"    API Key: {mask_api_key(llm_config.api_key)}")
        click.echo(f"    Model: {llm_config.model}")
    elif llm_provider == "google":
        click.echo(f"    API Key: {mask_api_key(llm_config.api_key)}")
        click.echo(f"    Model: {llm_config.model}")

    click.echo()

    # Display embedding provider
    click.echo(f"Embedding Provider: {click.style(embedding_provider, fg='magenta', bold=True)}")
    embedding_config = cfg.get_embedding_config()

    click.echo("  Configuration:")
    if embedding_provider == "ollama":
        click.echo(f"    Base URL: {embedding_config.base_url}")
        click.echo(f"    Model: {embedding_config.embedding_model}")
    elif embedding_provider == "openai-compatible":
        click.echo(f"    Base URL: {embedding_config.base_url}")
        click.echo(f"    API Key: {mask_api_key(embedding_config.api_key)}")
        click.echo(f"    Model: {embedding_config.embedding_model}")
    elif embedding_provider == "google":
        click.echo(f"    API Key: {mask_api_key(embedding_config.api_key)}")
        click.echo(f"    Model: {embedding_config.embedding_model}")

    click.echo()

    click.echo(click.style("Testing connections...", fg="cyan", bold=True))
    click.echo()

    results = run_setup_checks(cfg)
    check_labels = {
        "llm": "Testing LLM connection... ",
        "embedding": "Testing embedding connection... ",
        "bigdft": "Testing BigDFT import... ",
        "execution": "Testing execution connection... ",
    }

    for check in results["checks"]:
        click.echo(check_labels.get(check["name"], f"Testing {check['name']}... "), nl=False)
        if check["ok"]:
            click.echo(click.style("✓ Connected", fg="green"))
            if check["name"] == "llm":
                click.echo(f"  Response: {check['message']}")
            else:
                click.echo(f"  Status: {check['message']}")
        else:
            click.echo(click.style("✗ Failed", fg="red"))
            click.echo(click.style(f"  Error: {check['message']}", fg="red"))
            click.echo()
            click.echo(click.style("Connection test failed. Please check your configuration and connection.", fg="red"))
            sys.exit(1)
        click.echo()

    click.echo()
    click.echo(click.style("✓ All connections successful!", fg="green", bold=True))
    click.echo()
    click.echo(click.style("Ready to use!", fg="green"))


@cli.command()
@click.option(
    "--config",
    "-c",
    default="laraq.toml",
    type=click.Path(exists=False),
    help="Path to configuration file",
)
@click.argument("query", required=False)
def run(config, query):
    """Run the agent workflow with a query.

    If no query is provided, uses a default query.
    """
    config_path = Path(config)

    try:
        cfg = load_config(config_path)
    except ConfigError as e:
        click.echo(click.style("Configuration Error:", fg="red", bold=True))
        click.echo(click.style(str(e), fg="red"))
        sys.exit(1)
    except Exception as e:
        click.echo(click.style("Unexpected Error:", fg="red", bold=True))
        click.echo(click.style(str(e), fg="red"))
        sys.exit(1)

    try:
        agents = create_agents_from_config(cfg)
    except Exception as e:
        click.echo(click.style("Error creating agents:", fg="red", bold=True))
        click.echo(click.style(str(e), fg="red"))
        sys.exit(1)

    click.echo("Initializing agents...")
    for agent in agents.values():
        agent.init_agent()

    agents["director"].clear_log()

    if query is None:
        query = "What is this project about?"

    try:
        result = run_pipeline(
            agents, query,
            progress=lambda msg: click.echo(click.style(msg, fg="cyan")),
            max_retries=cfg.max_retries,
        )
    except Exception as e:
        click.echo(click.style("Error executing workflow:", fg="red", bold=True))
        click.echo(click.style(str(e), fg="red"))
        traceback.print_exc()
        sys.exit(1)

    if not result.success:
        click.echo(click.style(f"FAILED: {result.error}", fg="red"))
        if result.code:
            click.echo(click.style("\nFailed code:", fg="yellow"))
            click.echo(result.code)
        sys.exit(1)

    click.echo(click.style("\nGenerated code:", fg="cyan"))
    click.echo(result.code)

    click.echo(click.style("\nRunning...", fg="cyan"))
    executor_result = agents["executor"](result.state)
    agents["executor"].log_interaction(result.code, executor_result["message"].content)

    exec_status = executor_result["message"].additional_kwargs.get("status", "success")
    if exec_status == "error":
        click.echo(click.style(f"FAILED: Execution error: {executor_result['message'].content}", fg="red"))
        sys.exit(1)

    click.echo(click.style(executor_result["message"].content, fg="green"))


@cli.command()
@click.option(
    "--config",
    "-c",
    default="laraq.toml",
    type=click.Path(exists=False),
    help="Path to configuration file",
)
@click.argument("test_file", type=click.Path(exists=True), required=True)
def test(config, test_file):
    """Run test cases from an XML file.

    Each test contains a query, expected code, expected output with explicit types, and tolerance.
    The system will generate code for each query and compare the results.
    """
    config_path = Path(config)
    test_file_path = Path(test_file)

    click.echo(f"Loading tests from: {test_file_path}")
    click.echo()

    # Load tests
    try:
        tests = load_tests(test_file_path)
    except Exception as e:
        click.echo(click.style("Error loading tests:", fg="red", bold=True))
        click.echo(click.style(str(e), fg="red"))
        sys.exit(1)

    click.echo(f"Found {len(tests)} test(s)")
    click.echo()

    # Load configuration
    try:
        cfg = load_config(config_path)
    except ConfigError as e:
        click.echo(click.style("Configuration Error:", fg="red", bold=True))
        click.echo(click.style(str(e), fg="red"))
        sys.exit(1)

    # Create agents from configuration
    try:
        agents = create_agents_from_config(cfg)
    except Exception as e:
        click.echo(click.style("Error creating agents:", fg="red", bold=True))
        click.echo(click.style(str(e), fg="red"))
        sys.exit(1)

    # Initialize all agents
    click.echo("Initializing agents...")
    for agent in agents.values():
        agent.init_agent()
    click.echo()

    # Run tests
    passed = 0
    failed = 0
    results = []

    for i, test_case in enumerate(tests, 1):
        name = test_case.get("name", f"test_{i}")
        description = test_case.get("description", "")
        query = test_case["query"]
        expected_code = test_case["expected_code"]
        expected_output = test_case["expected_output"]
        tolerance = test_case.get("tolerance", 1e-6)

        click.echo(click.style("=" * 80, fg="cyan"))
        click.echo(click.style(f"Test {i}/{len(tests)}: {name}", fg="cyan", bold=True))
        if description:
            click.echo(f"Description: {description}")
        click.echo(f"Query: {query}")
        click.echo()

        agents["director"].clear_log()

        try:
            result = run_pipeline(agents, query, max_retries=cfg.max_retries)
        except Exception as e:
            click.echo(click.style(f"FAILED: Unexpected error: {e}", fg="red", bold=True))
            traceback.print_exc()
            failed += 1
            results.append({"name": name, "status": "FAILED", "reason": str(e), "flow": ["unexpected error"]})
            click.echo()
            continue

        if not result.success:
            click.echo(click.style(f"FAILED: {result.error}", fg="red", bold=True))
            if result.code:
                click.echo(click.style("\nFailed code:", fg="yellow"))
                click.echo(result.code)
            click.echo(f"Flow: {' → '.join(result.flow)}")
            failed += 1
            results.append({"name": name, "status": "FAILED", "reason": result.error, "flow": result.flow})
            click.echo()
            continue

        click.echo("Generated code:")
        click.echo(click.style(result.code, fg="white"))
        click.echo()

        try:
            expected_result = execute_code(expected_code)
        except Exception as e:
            click.echo(click.style(f"BUG IN TEST SUITE: expected code crashed: {e}", fg="red", bold=True))
            click.echo(click.style("Fix the expected_code in the XML test file.", fg="red"))
            failed += 1
            results.append({"name": name, "status": "FAILED", "reason": f"broken expected code: {e}", "flow": result.flow})
            click.echo()
            continue

        try:
            actual_result = execute_code(result.code)
        except Exception as e:
            flow = result.flow + ["execution error"]
            click.echo(click.style(f"FAILED: Execution error: {e}", fg="red", bold=True))
            click.echo(f"Flow: {' → '.join(flow)}")
            failed += 1
            results.append({"name": name, "status": "FAILED", "reason": str(e), "flow": flow})
            click.echo()
            continue

        match, message = compare_outputs(actual_result, expected_output, tolerance)

        if match:
            flow = result.flow + ["PASSED"]
            click.echo(click.style(f"PASSED: {message}", fg="green", bold=True))
            click.echo(f"Actual result: {actual_result}")
            click.echo(f"Expected result: {expected_result}")
            click.echo(f"Flow: {' → '.join(flow)}")
            passed += 1
            results.append({"name": name, "status": "PASSED", "reason": message, "flow": flow})
        else:
            flow = result.flow + ["comparison failed"]
            click.echo(click.style(f"FAILED: {message}", fg="red", bold=True))
            click.echo(f"Actual result: {actual_result}")
            click.echo(f"Expected result: {expected_result}")
            click.echo(f"Flow: {' → '.join(flow)}")
            failed += 1
            results.append({"name": name, "status": "FAILED", "reason": message, "flow": flow})

        click.echo()

    # Summary
    click.echo(click.style("=" * 80, fg="cyan"))
    click.echo(click.style("TEST SUMMARY", fg="cyan", bold=True))
    click.echo(click.style("=" * 80, fg="cyan"))
    click.echo()

    for result in results:
        status_color = "green" if result["status"] == "PASSED" else "red"
        click.echo(f"  {click.style(result['status'], fg=status_color)}: {result['name']}")
        if "flow" in result:
            click.echo(f"    Flow: {' → '.join(result['flow'])}")

    click.echo()
    click.echo(f"Total: {len(tests)} tests")
    click.echo(click.style(f"Passed: {passed}", fg="green"))
    click.echo(click.style(f"Failed: {failed}", fg="red"))

    if failed > 0:
        sys.exit(1)


@cli.command()
@click.option(
    "--config",
    "-c",
    default="laraq.toml",
    type=click.Path(exists=False),
    help="Path to configuration file",
)
@click.option(
    "--docs-path",
    "-d",
    default="docs",
    type=click.Path(exists=True),
    help="Path to documentation directory",
)
def cache_embeddings(config, docs_path):
    """Pre-generate and cache document embeddings for RAG.

    This command loads all documentation, generates embeddings,
    and saves them to a cache file. Subsequent runs will load
    from cache automatically if docs haven't changed.
    """
    config_path = Path(config)

    click.echo(f"Loading configuration from: {config_path}")

    # Load configuration
    try:
        cfg = load_config(config_path)
    except ConfigError as e:
        click.echo(click.style("Configuration Error:", fg="red", bold=True))
        click.echo(click.style(str(e), fg="red"))
        sys.exit(1)

    # Create agents
    try:
        agents = create_agents_from_config(cfg)
    except Exception as e:
        click.echo(click.style("Error creating agents:", fg="red", bold=True))
        click.echo(click.style(str(e), fg="red"))
        sys.exit(1)

    research_agent = agents["research"]

    # Check if it's a RAG agent
    if not hasattr(research_agent, '_load_and_embed_documents'):
        click.echo(click.style("Error: Research agent doesn't support caching", fg="red"))
        sys.exit(1)

    click.echo(f"Documentation path: {docs_path}")
    click.echo()

    # Force regenerate by clearing cache first
    cache_dir = Path(docs_path) / ".cache"
    if cache_dir.exists():
        click.echo("Clearing existing cache...")
        shutil.rmtree(cache_dir)

    # Initialize agent (will generate and cache embeddings)
    click.echo(click.style("Generating embeddings...", fg="cyan", bold=True))
    try:
        research_agent._ensure_initialized()
    except Exception as e:
        click.echo(click.style("Error generating embeddings:", fg="red", bold=True))
        click.echo(click.style(str(e), fg="red"))
        traceback.print_exc()
        sys.exit(1)

    click.echo()
    click.echo(click.style("✓ Embeddings cached successfully!", fg="green", bold=True))
    click.echo(f"Cache location: {cache_dir}")
    click.echo(f"Total chunks: {len(research_agent.chunks)}")
    click.echo()
    click.echo("Subsequent runs will load embeddings from cache automatically.")


@cli.command()
@click.option(
    "--config",
    "-c",
    default="laraq.toml",
    type=click.Path(exists=False),
    help="Path to configuration file",
)
def mcp(config):
    """Start the laraq MCP server (stdio transport).

    Validates the configuration, initializes all agents (including embedding
    warmup), then starts the FastMCP server. All status output goes to stderr
    so it doesn't interfere with the MCP stdio protocol.
    """
    config_path = Path(config)

    click.echo(f"Loading configuration from: {config_path}", err=True)

    try:
        cfg = load_config(config_path)
    except ConfigError as e:
        click.echo(click.style("Configuration Error:", fg="red", bold=True), err=True)
        click.echo(click.style(str(e), fg="red"), err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(click.style("Unexpected Error:", fg="red", bold=True), err=True)
        click.echo(click.style(str(e), fg="red"), err=True)
        sys.exit(1)

    click.echo(click.style("Configuration valid.", fg="green"), err=True)
    click.echo(f"LLM provider:       {cfg.agent.llm_provider}", err=True)
    click.echo(f"Embedding provider: {cfg.agent.embedding_provider}", err=True)
    click.echo(click.style("Initializing agents...", fg="cyan"), err=True)

    from laraq.mcp_server import start_mcp_server
    start_mcp_server(cfg)


if __name__ == "__main__":
    cli()
