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

Note that these tests are not intended to all pass. They test the quality of the docs, need for extra loops during validation, etc. You can play with the parameters and docs to improve agent performance. 

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

Create a `laraq.toml` by copying one of the ready-to-fill examples in `config/` for your provider (Ollama, OpenAI-compatible, Anthropic, or Google). The config has two main sections: the AI provider setup and the execution target.

The `[agent]` section selects an `llm_provider` and an `embedding_provider`. These can be different — for example, Anthropic does not supply embeddings, so you would pair it with OpenAI or Google for that role. Each provider needs its own configuration section with credentials and model names.

The top-level `template` field controls how generated code is executed. Set it to `bigdft_local` for in-process execution during development, or `bigdft_remote` to submit jobs to an HPC cluster via remotemanager. `max_retries` controls how many times the pipeline will retry the research → code → validate → dry run cycle on failure (default: 1).

Phase 1 agents (intent extraction, physics parameters, module mapping) can each be assigned a different model to balance cost and quality. RAG retrieval depth is tunable via `top_k` and `num_search_terms` in the research config. See the example configs for the full set of options.

For remote HPC execution, add a `[remote]` section with at least a `host` (the SSH hostname). You can inline a job script template directly in the config or point to a `.sh` file. Template placeholders use `#name#` syntax and their values can be set in the same section — for example, `nodes = 2` fills `#nodes#` in the template. When using the MCP server, these placeholders become named parameters on the execute tool, so the agent can override them at runtime. Any extra fields (like `ssh_insert`) are passed through to remotemanager. Remote execution requires passwordless SSH access to the target machine.

```toml
[remote]
host = "login.hpc.example.com"
user = "myuser"                  # optional, defaults to current user
remote_dir = "/scratch/myuser"   # optional
submitter = "sbatch"             # optional, for job schedulers
cpus = 8
template = """
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=#CPUS#
#SBATCH --time=01:00:00
#SBATCH --account=myproject

module load python/3.11
"""
```

Remote execution requires passwordless SSH access to the target machine.

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

