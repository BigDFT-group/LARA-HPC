---
name: laraq
description: Use laraq's MCP tools to generate, validate, dry-run, and execute BigDFT Python workflows with server checks and remote execution guidance.
---

# laraq: BigDFT Python Code Generation

laraq generates, validates, and executes BigDFT Python code from natural
language queries via an MCP server. The following tools must be called in order.

## Tool call order

### 1. `check_server()`
Verify that the server is correctly configured before any research or code
generation. This checks LLM connectivity, embedding connectivity, and the
execution target.

Always call this first in a fresh session. If it reports `ready: false`, stop
and report the setup issue to the user instead of continuing.

If `check_server()` fails with an import error involving `BigDFT`, tell the
user they need to have PyBigDFT and BigDFT installed and available in the local
environment where the laraq server is running.

### 2. `extract_intent(query)`
Parse the user request into a structured intent, calculation type, and search
hints. Always call this after `check_server`.

The returned `search_hints` are documentation-style terms (class names, method
names, API concepts) that will retrieve more targeted RAG results than the raw
query. Use them to build a richer research query in step 5.

### 3. `extract_physics(query)`
Extract physical and scientific parameters from the request: elements,
positions, units, DFT functional, boundary conditions, spin, charge, etc.
Always call this after `extract_intent`.

Pass the returned parameters to `generate_code` via `additional_context` so
the code generator has them without re-extracting from the query.

### 4. `map_modules(query)`
Check whether the request is feasible and identify which BigDFT Python modules
are relevant. Always call this after `extract_physics`.

If `feasible` is `false`, stop and report the reason to the user — do not
proceed to research or code generation.

If `feasible` is `true`, the returned `modules` list tells you which imports
to expect. Append them to your research query and pass them to `generate_code`
via `additional_context`.

### 5. `research(query)`
Search the BigDFT documentation using semantic RAG retrieval plus LLM query
expansion. Call this after Phase 1 (steps 2–4).

Build the query by combining the original request with the search hints from
`extract_intent` and module names from `map_modules`. Concretely:

```
ionization potential water
Search hints: delta-SCF, charge, qcharge, cation calculation
Relevant modules: BigDFT.Atoms, BigDFT.Inputfiles, BigDFT.Calculators
```

If the results are not relevant, try again with a more specific query — use
class names, method names, or API terms rather than a description of what you
want to do. Multiple targeted searches are better than one broad one.

### 6. `generate_code(query, context, additional_context="")`
Generate Python code using the query and the context returned by `research`.
Returns raw code (may contain JSON wrapping — do not clean it yourself).

Always pass Phase 1 results in `additional_context` as plain text. For example:

```
Intent: compute ionization potential of water using delta-SCF
Relevant modules: BigDFT.Atoms, BigDFT.Inputfiles, BigDFT.Calculators
Physics parameters: elements=O,H,H; charge=0/+1; functional=PBE; boundary=free
```

Also include any working example code from this session in `additional_context`
so the generator follows the same style and import patterns.

Always generate code that uses relative file paths. Never use absolute local
paths such as `/Users/...` or `/home/...`.

### 7. `validate(code)`
Run pyflakes static analysis and import checks on the generated code.
Returns `{"valid": true/false, "code": "<cleaned code>", "error": "..."}`.

Always use the returned `"code"` field in subsequent calls — the validator
cleans up JSON wrapping and escape sequences that the LLM may have added.

### 8. `dry_run(code, mpi_processes=1)`
Execute the validated code with `BIGDFT_MPIDRYRUN` set to catch runtime errors
without running a real simulation.
Returns `{"success": true/false, "error": "...", "mpi_processes": ...}`.

After a successful dry run, find and read the BigDFT log file (`log-*.yaml`)
written to the current working directory. Check it for memory usage estimates
and any warnings, and report this to the user. Do this immediately after each
successful dry run, because later runs may overwrite the same top-level log
filename.

The dry-run memory report is per MPI process. If the user asks about memory for
larger MPI counts, use `mpi_processes > 1` and make it clear that the reported
value is per process, not total cluster memory.

### 9. `execute(code, ...template params...)`

Execute the code for real and return the result of `f()`. The execute tool's
signature includes any job template parameters from the `[remote]` config
section (e.g. `nodes`, `ntasks`, `walltime`). These default to the config
values but can be overridden per call — for example if the dry run shows the
job needs more resources. For `bigdft_local`, execute only takes `code`.

Only call this after `validate` and `dry_run` have both succeeded AND you have
completed both of the following steps — in this order — before making the tool
call:

1. **Show the code to the user.** Display the full generated code in a code block.
   Do not summarise it — show it in full.
2. **Ask for explicit approval.** Ask "Shall I run this?" and wait for the user's response.

Execution mode depends on the server's `template` setting: `bigdft_local` runs
in-process; `bigdft_remote` submits via remotemanager using the server's
`[remote]` configuration.

## Retry strategy

If `validate` or `dry_run` fails, do NOT retry with the same code. Instead:
- Call `research` again with a more targeted query that includes the error
- Call `generate_code` again with the new context and the failed code + error in `additional_context`
- Run `validate` and `dry_run` again

Each stage allows one retry before giving up.

## Code contract

All generated code defines a single function `f()` with no arguments. All
imports go inside `f()`. The function must return its result — no `print()`.
`f()` is called automatically by `dry_run` and `execute`.
If files are used, their paths must be relative to the run directory.

## Rules

- **Do not call `execute` without showing the full code and receiving explicit user approval.**
  This is mandatory. Skipping or abbreviating either step is not permitted.
- **Do not use the BigDFT API yourself.** Always go through the
  `check_server` → Phase 1 → `research` → `generate_code` pipeline. If unsure
  whether the tools can handle a request, ask the user first.
- **Always run all Phase 1 tools** (`extract_intent`, `extract_physics`,
  `map_modules`) before `research`. Their outputs directly improve retrieval
  quality and code correctness.
