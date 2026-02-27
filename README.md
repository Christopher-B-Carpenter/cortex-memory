<p align="center">
  <img src="assets/banner.svg" alt="Cortex Memory" width="800"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/llm-cortex-memory/"><img src="https://img.shields.io/pypi/v/llm-cortex-memory?color=00d4ff&style=flat-square" alt="PyPI"></a>
  <a href="https://github.com/Christopher-B-Carpenter/cortex-memory/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square" alt="Python">
</p>

Cortex gives LLM conversations persistent memory that lives in a single `.memory` file. No embeddings, no vector database, no API keys for retrieval. Just a file that learns what matters to you and gets better at finding it.

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

## Why not just use RAG?

Traditional RAG systems work. But they come with costs that compound over time:

| | RAG / Vector DB | Hosted memory | Text file | **Cortex** |
|---|---|---|---|---|
| Embedding API calls per query | 1+ | 1+ | 0 | **0** |
| Embedding API calls per store | 1 | 1 | 0 | **0** |
| Portable as a file | No | No | Yes | **Yes** |
| Works with any LLM | No (model-locked) | Varies | Yes | **Yes** |
| Learns from usage | No | Some | No | **Yes** |
| Results validated as group | No | No | No | **Yes** |
| Retrieval cost as N grows | Linear | Linear | None | **Flat** |
| Infrastructure | DB + embedding service | Cloud API | None | **None** |

**The token efficiency problem is real.** A RAG system with 10,000 memories burns an embedding API call on every query and every store. Cortex uses zero — retrieval runs locally on learned BM25 structure. At scale, that's thousands of embedding calls saved per day with no degradation in retrieval quality.

**The coherence problem is worse.** RAG scores each document independently against the query vector. If you ask about "auth architecture," you might get the JWT decision, a Redis config note, and an unrelated API doc that happens to mention "auth" — three independently high-scoring fragments that don't form a coherent context window. Cortex's co-retrieval clustering ensures that memories which *belong together* get retrieved together, because it learned that from your actual usage patterns.

---

## Real-world benchmarks

### Tested against a real memory store

80 memories accumulated over real development sessions. 75 prior queries. No synthetic data.

| Metric | Value |
|---|---|
| Memories scored per query | **12%** (88% skipped via two-pass) |
| Query latency (p50) | **0.06ms** |
| Context coherence | **4.1** mean co-retrieval count |
| Result divergence vs flat BM25 | **36%** of results changed by structure |
| File size | **12.3 KB** (157 bytes/memory) |
| Load time | **1.1ms** |
| Save/load | **Lossless** |

The 88% skip rate means Cortex only scores 12% of the store on each query — the rest are pruned by cluster structure before scoring. That's 88% less computation than flat BM25, with equal or better precision.

### Scaling benchmarks (synthetic)

Measured on Apple M-series, software engineering conversation corpus:

| N | Precision vs Flat BM25 | Memories skipped | Query latency | File size |
|---|---|---|---|---|
| 100 | +0.05 | 67% | <1ms | ~2 KB |
| 500 | -0.08 | 88% | <1ms | ~8 KB |
| 1,000 | ~0 | 95% | <1ms | ~15 KB |
| 2,000 | ~0 | 96% | ~1ms | ~30 KB |
| 10,000 | ~0 | 97% | ~13ms | 148 KB |

Context coherence (mean co-retrieval count in returned set) grows from 5 to 89 over 200 queries without any preprocessing.

**What this means:** retrieval cost stays flat as memory grows. At 10,000 memories, Cortex still scores only 3-5% of the store. A vector database would embed and score all 10,000. Every single time.

See `benchmark.py` and `benchmark_real.py` to reproduce.

---

## How it works

BM25 handles text matching. Three learned layers handle everything else:

**Usage weights** — memories that get retrieved often gain weight. Memories that stop being useful decay. The store develops a signal about what matters vs. what's noise — behavioral data that doesn't exist in the text itself.

**Co-retrieval clustering** — when memories A and B keep appearing in the same result sets, they accumulate a co-retrieval count and eventually cluster. The store discovers that "JWT rotation," "Redis sessions," and "24h token expiry" belong together because *you* keep retrieving them together — not because they share vocabulary. Retrieving one pulls the others along even if a new query only lexically matches one.

**Two-pass retrieval** — clusters make retrieval faster as the store grows. Pass 1 scores cluster representatives and picks the best-matching clusters. Pass 2 scores only their members. At 1,000 memories this skips 95% of the store. The efficiency comes from the structure, and the structure comes from your usage.

None of this requires an embedding model. The structure is learned from retrieval patterns and stored in the file. Move the file to a new machine, plug it into a different LLM — all the learned structure comes with it.

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
- `UserPromptSubmit` hook queries memory before each prompt — injects top-5 results as context
- `Stop` hook stores Claude's response after each turn — memory grows every session
- `config.json` controls the source: `project`, `global`, `both` (default), or `off`

**Seed initial context (optional):**

```python
from cortex_memory import Memory
mem = Memory.load(".claude/memory/project.memory")
mem.store("uses Python 3.12, FastAPI, PostgreSQL, deployed on AWS ECS")
mem.store("auth uses JWT with 24h expiry, refresh tokens in Redis")
mem.save(".claude/memory/project.memory")
```

**Optional: memory statusline**

Add a live memory counter to the Claude Code status bar:

```
⬡ cortex  +5 this session  │  185 memories  7 clusters  194 queries
```

Copy `examples/claude_code_hooks/statusline.py` to your memory directory and add to `settings.json`:

```json
"statusLine": {
  "type": "command",
  "command": "python3 /absolute/path/to/.claude/memory/statusline.py"
}
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

## Git-backed memory — one file, any machine, any LLM

Store your `.memory` file in a private GitHub repo and it stays in sync automatically across every machine and LLM you use. Pull when a session starts, push when it ends. One continuous thread of memory everywhere.

**Setup (once):**

```bash
# Create a private repo on GitHub, then:
git clone git@github.com:you/memory.git ~/memory
pip install llm-cortex-memory
```

**Use with any local LLM (Ollama, LM Studio, etc.):**

```bash
python examples/git_memory_chat.py --repo ~/memory --model llama3.2
```

Pulls latest memory on start, pushes after the session. That's it.

**Use with the harness directly:**

```python
from cortex_memory import OpenAIMemoryHarness

harness = OpenAIMemoryHarness(
    "~/memory/global.memory",
    model="llama3.2",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    git_sync={
        "enabled": True,
        "repo_path": "~/memory",
    },
)

response = harness.chat("what did we decide about auth?")
harness.save()  # pushes to git in background
```

**Add git sync to Claude Code hooks** — extend `~/.claude/memory/config.json`:

```json
{
  "source": "global",
  "top_k": 5,
  "git_sync": {
    "enabled": true,
    "repo_path": "~/memory",
    "remote": "origin",
    "branch": "main"
  }
}
```

The hooks pull before each prompt and push after each response (async, non-blocking). Your Claude Code sessions, local LLM sessions, and API sessions all share the same memory file through git.

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
├── benchmark.py                           # synthetic benchmarks
├── benchmark_real.py                      # real-world benchmarks
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
