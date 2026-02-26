# Cortex Memory

**One file. Any LLM. Gets smarter the more you use it.**

Cortex gives LLM conversations persistent memory that lives in a single `.memory` file. Store it in a repo, copy it to a new machine, plug it into Claude, GPT, or any LLM — it just works. No embeddings to rebuild, no database to run, no API keys for retrieval.

The file isn't just storage. Every query teaches it what matters: which memories you actually use, which ones come up together, which context you keep reaching for. That learned structure compounds over time and travels with the file wherever it goes.

```bash
pip install llm-cortex-memory
```

```python
from cortex_memory import Memory

mem = Memory.load("project.memory")
results = mem.query("what did we decide about authentication?")
mem.store("Decided to use JWT with 24-hour expiry and Redis-backed refresh tokens.")
mem.save("project.memory")
```

---

## What makes it different

| | Text file | Vector DB / RAG | Hosted memory | **Cortex** |
|---|---|---|---|---|
| Portable as a file | Yes | No | No | **Yes** |
| Works with any LLM | Yes | No (model-locked) | Varies | **Yes** |
| Learns from usage | No | No | Some | **Yes** |
| Zero infrastructure | Yes | No | No | **Yes** |
| Retrieval at scale | No | Yes | Yes | **Yes** |

**A text file** has no retrieval and treats every entry equally forever.

**A vector database** learns structure, but it belongs to the embedding model that created it. Switch models and the index is worthless.

**Hosted memory** (Mem0, Zep) works but requires cloud APIs. You don't own a file — you rent a service.

**Cortex** is the only memory artifact that is simultaneously portable, model-agnostic, zero-infrastructure, and structurally improved by use.

---

## Installation

```bash
pip install llm-cortex-memory              # core
pip install llm-cortex-memory[anthropic]   # + Claude API harness
pip install llm-cortex-memory[openai]      # + OpenAI API harness
pip install llm-cortex-memory[all]         # everything
```

---

## Quick start

### Create a memory store

```python
from cortex_memory import Memory

mem = Memory.create(
    description="payments-service development",
    tags=["python", "auth", "database"],
)

mem.store("Decided to use JWT tokens with 24-hour expiry.")
mem.store("SQL injection in legacy login fixed with parameterized queries.")
mem.store("Composite index on (user_id, created_at) reduced dashboard query from 8s to 200ms.")

mem.save("project.memory")
```

### Query it anywhere

```python
from cortex_memory import Memory

mem = Memory.load("project.memory")

results = mem.query("what security issues did we fix?", top_k=5)
for r in results:
    print(r)
```

### Merge two memory files

```python
from cortex_memory import Memory

mem_a = Memory.load("alice.memory")
mem_b = Memory.load("bob.memory")

merged = Memory.merge(mem_a, mem_b, description="shared project memory")
merged.save("team.memory")
```

---

## Integration with Claude Code (recommended)

One-command setup. Memory injection and storage happen automatically on every turn.

```bash
pip install llm-cortex-memory
python3 -m cortex_memory install           # project-level setup
python3 -m cortex_memory install --global  # global (cross-project) setup
```

This creates hook files, generates `settings.json` with correct absolute paths, and initializes the `.memory` file. Then restart Claude Code — memory is automatic from that point.

**How it works:**
- `UserPromptSubmit` hook queries memory before each prompt → injects top-5 results as context
- `Stop` hook stores Claude's response after each turn → memory grows every session
- `config.json` controls the source: `project`, `global`, `both` (default), or `off`

**Seed initial context (optional):**

```python
from cortex_memory import Memory
mem = Memory.load(".claude/memory/project.memory")
mem.store("uses Python 3.12, FastAPI, PostgreSQL, deployed on AWS ECS")
mem.store("auth uses JWT with 24h expiry, refresh tokens in Redis")
mem.save(".claude/memory/project.memory")
```

See `examples/claude_code_hooks/setup.md` for tuning options, dev team use cases, the `/memory` slash command, and troubleshooting.

---

## Integration with Claude API

```python
from cortex_memory import ClaudeMemoryHarness

harness = ClaudeMemoryHarness(
    "project.memory",
    model="claude-sonnet-4-6",
    system_prompt="You are a technical assistant with context about this project.",
    top_k=5,
)

response = harness.chat("what indexes did we add to fix the slow queries?")
print(response)

harness.save()  # persists to project.memory
```

Every turn:
1. Queries memory with the user message
2. Injects top-K results into the system prompt as `<memory>` context
3. Calls Claude
4. Stores Claude's response asynchronously

### OpenAI / any OpenAI-compatible API

```python
from cortex_memory import OpenAIMemoryHarness

harness = OpenAIMemoryHarness(
    "project.memory",
    model="gpt-4o",
    # base_url="http://localhost:11434/v1"  # Ollama, Together, Fireworks, etc.
)

response = harness.chat("summarize what we know about the auth service")
harness.save()
```

### Any LLM callable

```python
from cortex_memory import MemoryHarness

def my_llm(messages, system, **kwargs):
    # call any LLM here
    ...

harness = MemoryHarness("project.memory", llm_fn=my_llm)
response = harness.chat("what did we decide?")
```

---

## How it works

BM25 handles the text matching. Three layers on top handle everything else:

**Usage weights** — memories that get retrieved often gain weight. Memories that stop being useful decay. After enough queries, the store has a signal about what matters vs. what's noise. This is behavioral data that doesn't exist in the text itself.

**Co-retrieval clustering** — when memories A and B keep appearing in the same result sets, they accumulate a co-retrieval count and eventually cluster. This is learned associative structure: the store discovers that "JWT rotation," "Redis sessions," and "24h token expiry" belong together because *you* keep retrieving them together — not because they share vocabulary. Retrieving one pulls the others along even if a new query only lexically matches one of them.

**Two-pass retrieval** — clusters make retrieval faster as the store grows. Pass 1 scores cluster representatives, picks the best-matching clusters, Pass 2 scores only their members. At 1,000 memories this skips 85-97% of the store. At 10,000 memories, retrieval still takes ~13ms. The efficiency comes from the structure, and the structure comes from your usage.

---

## File format

A `.memory` file is a zip archive containing:

```
project.memory
├── store.pkl        # Cortex state (BM25 index, weights, clusters, co-retrieval)
├── manifest.json    # metadata: description, tags, query count, LLM hint
└── README.md        # auto-generated summary of top memories and clusters
```

- **~15 bytes per memory** at N=10,000 (148 KB total)
- **81ms load time** at N=10,000
- **Lossless** — two independently loaded instances produce identical results
- No external model required to load or query

---

## Team and shared repositories

`.memory` files are binary — git cannot diff or auto-merge them. The recommended approach is to keep them out of feature branch commits and merge them explicitly using `Memory.merge()` at the points where you want to consolidate context.

**Merging two memory files:**

```python
from cortex_memory import Memory

merged = Memory.merge(
    Memory.load("alice.memory"),
    Memory.load("bob.memory"),
    description="shared project memory",
)
merged.save("team.memory")
```

Merge semantics are non-destructive: memories are unioned, weights are max-pooled (whichever side used a memory more wins), and co-retrieval counts are summed.

**Keeping `.memory` out of PR diffs:**

If you do commit `.memory` files, add these lines to `.gitattributes` so they're hidden from code review diffs:

```
*.memory -diff
*.memory linguist-generated=true
```

**CI pipelines:**

For automated consolidation after branch merges, call `Memory.merge()` directly in your pipeline script — it's a straightforward Python call with no external dependencies beyond numpy and scipy.

---

## Benchmarks

Measured on a MacBook Pro (Apple M-series), N=100-10,000 memories, software engineering conversation corpus.

| N | Precision@8 vs Flat BM25 | Memories skipped | Load time |
|---|---|---|---|
| 100 | +0.05 | 67% | 1ms |
| 500 | -0.08 | 88% | 4ms |
| 1,000 | ~0 | 95% | 8ms |
| 2,000 | ~0 | 96% | - |
| 10,000 | ~0 | 17% | 81ms |

Context coherence (mean co-retrieval count in returned set) grows from 5 to 89 over 200 queries without any preprocessing. Token efficiency is ~22% better than flat retrieval in steady state.

See `benchmark.py` to reproduce.

---

## Repository structure

```
cortex-memory/
├── pyproject.toml                         # package metadata (pip install llm-cortex-memory)
├── src/cortex_memory/                     # installable package
│   ├── __init__.py                        # public API exports
│   ├── cortex.py                          # storage engine (VectorizedBM25, Cortex)
│   ├── memory.py                          # portable artifact (Memory class, merge)
│   ├── harness.py                         # LLM integration (MemoryHarness, Claude/OpenAI)
│   └── install.py                         # one-command Claude Code setup
├── cortex.py                              # standalone (no pip install needed)
├── memory.py                              # standalone
├── harness.py                             # standalone
├── benchmark.py                           # reproduce the benchmarks
├── requirements.txt
└── examples/
    ├── demo.py                            # basic usage, no API needed
    ├── claude_api.py                      # interactive Claude conversation loop
    └── claude_code_hooks/                 # Claude Code hook reference
        ├── on_prompt.py                   # UserPromptSubmit hook
        ├── on_stop.py                     # Stop hook
        ├── config.json                    # memory source config
        ├── memory.md                      # /memory slash command
        ├── settings.json                  # settings.json template
        └── setup.md                       # manual setup, tuning, troubleshooting
```

**Two ways to use:**
- `pip install llm-cortex-memory` — recommended. Hooks use the installed package.
- Clone and copy files — standalone, no pip needed. The root `cortex.py`, `memory.py`, `harness.py` work independently.

---

## API reference

### `Memory`

| Method | Description |
|---|---|
| `Memory.create(description, tags)` | Create a new empty store |
| `Memory.load(path)` | Load from `.memory` file |
| `Memory.merge(a, b, description)` | Union two stores |
| `mem.store(text, memory_id, metadata)` | Add a memory |
| `mem.query(text, top_k)` | Retrieve relevant memories (returns list of strings) |
| `mem.query(text, return_scores=True)` | Returns list of dicts with score/weight/cluster |
| `mem.forget(memory_id)` | Remove a memory |
| `mem.save(path)` | Serialize to disk |
| `mem.stats()` | Store statistics |
| `mem.top_memories(n)` | Most-used memories by weight |
| `mem.clusters(n)` | Current cluster summary |

### `MemoryHarness`

| Method | Description |
|---|---|
| `MemoryHarness(path, llm_fn, ...)` | Create harness with any LLM callable |
| `ClaudeMemoryHarness(path, model, ...)` | Anthropic SDK subclass |
| `OpenAIMemoryHarness(path, model, ...)` | OpenAI SDK subclass |
| `harness.chat(message)` | Send message, get response with memory injection |
| `harness.build_system_prompt(query)` | Get system prompt with injected context (for manual use) |
| `harness.store(text)` | Manually store a memory |
| `harness.query(text)` | Query without LLM call |
| `harness.inject_claude_md(query, path)` | Prepend memories to CLAUDE.md |
| `harness.sync_from_transcript(path)` | Store turns from a JSONL transcript |
| `harness.save()` | Flush and save to disk |
| `harness.reset_conversation()` | Clear conversation history (keep memory) |

---

## License

MIT
