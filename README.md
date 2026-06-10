# laraq

Agent-driven code generation using RAG research, LLM code generation, and automatic validation - aiming at runs on supercomputers (arXiv:2604.22571).

## Installation

```bash
uv sync
```

You also need to install PyBigDFT and BigDFT. Since we're working with a dry run feature in devel, you will need to do this manually (https://gitlab.com/l_sim/bigdft-suite).

The `docs/` directory (BigDFT documentation used for RAG retrieval) is included in this repository. It is the source of context for all code generation. Pre-cache the embeddings before first use:

```bash
uv run laraq cache-embeddings --config laraq.toml
```

## Usage

Example configurations are in `config`. Fill one in and use it in the next set of commands.

Test that your setup is good:
```
uv run laraq --config laraq.toml
```

Run a query:
```bash
uv run laraq run --config laraq.toml "Create a helium atom at position [2.0, 0.0, 0.0] in bohr"
```

Run tests from XML:
```bash
uv run laraq test --config laraq.toml tests/bigdft_tests.xml
```

## MCP Server

laraq can be used as an MCP server, exposing the pipeline as composable tools for Claude or other MCP clients.

Start the server:
```bash
uv run --project /absolute/path/to/laraq laraq mcp --config /absolute/path/to/laraq.toml
```

To wire it up, add the following to your `.mcp.json`:
```json
{
  "mcpServers": {
    "laraq": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--project", "/absolute/path/to/laraq", "laraq", "mcp", "--config", "/absolute/path/to/laraq.toml"]
    }
  }
}
```

Use absolute paths are required for both the laraq source tree and the config file.

The server exposes tools the client LLM calls in order: `check_server` → `extract_intent` → `extract_physics` → `map_modules` → `research` → `generate_code` → `validate` → `dry_run` → `execute`. Execution uses the `template` setting from the config: `bigdft_remote` submits via remotemanager (requires `[remote]`), `bigdft_local` runs in-process.

### Agent instructions (SKILL.md)

`SKILL.md` in the project root contains the full usage instructions for the agent. For Claude Code, install it as a project-local skill so it is automatically available in any session:

```bash
mkdir -p .claude/skills/laraq
cp /path/to/laraq/SKILL.md .claude/skills/laraq/SKILL.md
```

The skill will be picked up automatically by Claude Code. You can also invoke it explicitly with `/laraq`.

## How It Works

The pipeline has three phases.

### Phase 1 — Understanding

Intent extraction parses the user query into a structured intent, a calculation type label, and search hints. The search hints are documentation-style terms (class names, method names) that give the RAG stage a better starting point than the raw query. This is implemented through a prompt to an LLM. Designed for a lighter model.

Physics parameter extraction pulls any explicit scientific quantities out of the query: elements, positions, units, DFT functional, boundary conditions, spin polarisation, etc. Returns an empty dict if nothing is found. This is implemented through a prompt to an LLM. Designed for a heavier model since it requires domain knowledge to do well.

Module mapping checks a capabilities document (`docs/capabilities.md`) to decide whether the request is feasible at all. The LLM is given the full document and the query and returns a yes/no verdict plus the relevant modules. This is implemented through a prompt to an LLM. If not feasible, the pipeline stops here.

### Phase 2 — Generation

RAG research does semantic search over the `docs/` directory using embeddings. The query (or the search hints from Phase 1) is embedded, cosine similarity is computed against pre-chunked document embeddings, and the top-k chunks are returned as context. The LLM also expands the query into several documentation-style search terms to improve recall.

Code generation takes the research context and the user query and asks the LLM to generate Python code. All generated code follows a fixed structure: a single `f()` function with no arguments, all imports inside, returning its result. The return type constraint depends on the `template` setting: `bigdft_remote` requires a JSON-serializable return value so remotemanager can transfer the result back over SSH; `bigdft_local` has no such restriction since execution is in-process. The LLM response itself is JSON-wrapped so the explanation and code block can be parsed separately. 

At this point the physics parameters, relevant modules, and intent from Phase 1 have all been incorporated into the generated code, and it is ready to be dispatched to Phase 3.

### Phase 3 — Validation

Syntax checking runs pyflakes plus an AST import check. Pyflakes catches undefined names, unused imports, and syntax errors. The import check tries to resolve every import in the generated code against the local environment.

Dry run executes the code with `BIGDFT_MPIDRYRUN` set, which makes BigDFT go through its startup and input validation without running a real simulation. Catches runtime errors, missing attributes, and misconfigured inputs cheaply.

If either check fails, the pipeline retries up to `max_retries` times (configurable): it runs research again with error context, regenerates the code, and validates again.

### Execution

Execution is determined by the `template` setting. `bigdft_remote` submits the validated code via remotemanager using the `[remote]` section of the config — this handles SSH connection, job templating, SLURM submission, and result retrieval. `bigdft_local` runs the code in-process, which is useful for development and testing without an HPC cluster.

## Configuration

Create a `laraq.toml`. Ready-to-fill examples for all supported providers (Ollama, OpenAI-compatible, Anthropic, Google) are in `config/`.

| Field | Default | Description |
|-------|---------|-------------|
| `template` | `bigdft_remote` | `bigdft_remote` — submit via remotemanager (requires `[remote]`); `bigdft_local` — run in-process |
| `max_retries` | `1` | Retry budget for the research → code → validate → dry run cycle |

Set `llm_provider` and `embedding_provider` in `[agent]` to select a provider, then add the matching provider section. Note that Anthropic does not supply embeddings, so a second provider is needed for those.

Phase 1 agents can be assigned lighter models to save cost via `[agents.intent_extractor_config]`, `[agents.physics_params_config]`, and `[agents.module_mapper_config]`. RAG retrieval depth is controlled via `[agents.research_config]` (`top_k`, `num_search_terms`).

For remote HPC execution, set `template = "bigdft_remote"` and add a `[remote]` section:

```toml
[remote]
host = "login.hpc.example.com"
user = "myuser"                  # optional, defaults to current user
remote_dir = "/scratch/myuser"   # optional
submitter = "sbatch"             # optional, for job schedulers
template = """
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=01:00:00
#SBATCH --account=myproject

module load python/3.11
"""
```

`host` is required. `template` accepts an inline script or a path to a `.sh` file; if omitted, laraq supplies a minimal default. Remote execution requires passwordless SSH access to the target machine.

## Test Format

Tests are defined in XML with expected code and output:

```xml
<test name="helium_atom_unit_conversion">
  <query>Create a helium atom at position [2.0, 0.0, 0.0] in bohr</query>
  <expected_code><![CDATA[
def f():
    from BigDFT.Atoms import Atom
    atom = Atom({'r': [2.0, 0.0, 0.0], 'sym': 'He', 'units': 'bohr'})
    return atom.get_position('angstroem')
  ]]></expected_code>
  <expected_output type="list">
    <item type="float">1.05835442184</item>
    <item type="float">0.0</item>
    <item type="float">0.0</item>
  </expected_output>
</test>
```

## Docker

A pre-built container is available on Docker Hub with Ollama and the required models (`gpt-oss:20b` for LLM, `qwen3-embedding` for embeddings) already baked in. Use `config/ollama.toml` as your config.

```bash
docker pull wddawson/laraq:latest
docker run -d -p 11434:11434 -v $(pwd):/app --name laraq wddawson/laraq:latest
docker exec -it laraq laraq run --config config/ollama.toml "Your query here"
docker exec -it laraq laraq test --config config/ollama.toml tests/bigdft_tests.xml
```

The container starts Ollama automatically on boot. laraq itself is installed from the mounted volume at runtime, so code changes take effect without rebuilding the image.
