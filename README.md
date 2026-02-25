# Cortex Memory

A portable, model-agnostic memory layer for LLM conversations.

Cortex stores conversation memories as plain text, retrieves them with BM25, and builds associative structure from usage patterns over time. The entire memory state — index, weights, clusters — serializes to a single compressed file of approximately **15 bytes per memory**. No embedding model. No database. No API key required for retrieval.

```python
from memory import Memory

mem = Memory.load("project.memory")
results = mem.query("what did we decide about authentication?")
mem.store("Decided to use JWT with 24-hour expiry and Redis-backed refresh tokens.")
mem.save("project.memory")
```

---

## Why

Long-lived projects accumulate context that current tools don't manage well:

- **Plain text logs** grow without structure and have no retrieval
- **RAG / vector databases** are tied to a specific embedding model — swap models and the index degrades or must be rebuilt
- **Hosted memory services** (Mem0, Zep) require cloud APIs and don't produce portable files

Cortex targets the gap: a memory file that travels with a project, survives model changes, requires no infrastructure, and improves structurally through use.

---

## Installation

```bash
pip install numpy scipy          # core dependencies
pip install anthropic            # optional: for ClaudeMemoryHarness
pip install openai               # optional: for OpenAIMemoryHarness
```

Clone this repo:
```bash
git clone https://github.com/your-org/cortex-memory
cd cortex-memory
pip install -r requirements.txt
```

---

## Quick start

### Create a memory store

```python
from memory import Memory

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
mem = Memory.load("project.memory")

results = mem.query("what security issues did we fix?", top_k=5)
for r in results:
    print(r)
```

### Merge two memory files

```python
mem_a = Memory.load("alice.memory")
mem_b = Memory.load("bob.memory")

merged = Memory.merge(mem_a, mem_b, description="shared project memory")
merged.save("team.memory")
```

---

## Integration with Claude

### Option 1: Direct API (recommended for applications)

```python
from harness import ClaudeMemoryHarness

harness = ClaudeMemoryHarness(
    "project.memory",
    model="claude-opus-4-5",
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

### Option 2: Claude Code CLI

Use `claude_code_inject.py` instead of calling `claude` directly.

**Inject memory at session start:**
```bash
python examples/claude_code_inject.py inject
```
This queries your `project.memory`, prepends the results to `CLAUDE.md`, then launches Claude Code. Claude reads the injected context at session start.

With a focused topic:
```bash
python examples/claude_code_inject.py inject --topic "database performance"
```

**Sync memory after a session:**
```bash
python examples/claude_code_inject.py sync
```
This finds the most recent Claude Code transcript, extracts assistant turns, and stores them in `project.memory`.

**Shell alias for convenience:**
```bash
# Add to ~/.zshrc or ~/.bashrc
alias claude-mem='python /path/to/examples/claude_code_inject.py inject'
```

Then use `claude-mem` instead of `claude` to start memory-augmented sessions.

**After each session, sync:**
```bash
python examples/claude_code_inject.py sync
```

Over time, the memory file accumulates everything meaningful that came out of your Claude Code sessions. The cluster structure evolves to reflect which topics you actually work on together.

### Option 3: OpenAI / any OpenAI-compatible API

```python
from harness import OpenAIMemoryHarness

harness = OpenAIMemoryHarness(
    "project.memory",
    model="gpt-4o",
    # base_url="http://localhost:11434/v1"  # Ollama, Together, Fireworks, etc.
)

response = harness.chat("summarize what we know about the auth service")
harness.save()
```

### Option 4: Any LLM callable

```python
from harness import MemoryHarness

def my_llm(messages, system, **kwargs):
    # call any LLM here
    ...

harness = MemoryHarness("project.memory", llm_fn=my_llm)
response = harness.chat("what did we decide?")
```

---

## How it works

Three layers on top of BM25 full-text retrieval:

**1. Usage weights** — each memory has a scalar weight that strengthens when the memory is retrieved and decays slowly over time. Decay is computed lazily (no per-query O(N) loop). Frequently-useful memories surface slightly ahead of equally-relevant alternatives.

**2. Co-retrieval clustering** — when memories A and B appear together in top-K results across multiple queries, they accumulate a co-retrieval count. Above a threshold, they join the same cluster. Clusters emerge from actual usage patterns, not from lexical or semantic similarity.

**3. Two-pass retrieval** — at query time, Pass 1 scores only cluster representatives (O(clusters)), selects the top-matching clusters, and Pass 2 scores only their members. At N=1,000 with ~80 clusters, this scores ~60 memories instead of 1,000. At N=500–2,000, the architecture skips 85–97% of the store while matching flat BM25 precision.

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

## Benchmarks

Measured on a MacBook Pro (Apple M-series), N=100–10,000 memories, software engineering conversation corpus.

| N | Precision@8 vs Flat BM25 | Memories skipped | Load time |
|---|---|---|---|
| 100 | +0.05 | 67% | 1ms |
| 500 | −0.08 | 88% | 4ms |
| 1,000 | ≈0 | 95% | 8ms |
| 2,000 | ≈0 | 96% | — |
| 10,000 | ≈0 | 17% | 81ms |

Context coherence (mean co-retrieval count in returned set) grows from 5 to 89 over 200 queries without any preprocessing. Token efficiency is ~22% better than flat retrieval in steady state.

See `benchmark.py` to reproduce. See `paper/` for the full research paper and figures.

---

## Repository structure

```
cortex-memory/
├── cortex.py              # storage engine — VectorizedBM25, Cortex class
├── memory.py              # portable artifact — Memory class, merge, manifest
├── harness.py             # LLM integration — MemoryHarness, ClaudeMemoryHarness
├── benchmark.py           # benchmark suite
├── make_figures.py        # publication figure generation
├── requirements.txt
├── examples/
│   ├── demo.py                # basic usage, no API needed
│   ├── claude_api.py          # interactive Claude conversation with memory
│   └── claude_code_inject.py  # Claude Code CLI integration
└── paper/
    ├── cortex_paper_v2.docx   # full research paper
    └── figures/               # PDF and PNG figures
```

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
