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

### Option 2: Claude Code (hooks — recommended)

The cleanest integration uses Claude Code's native hook system. Memory injection and storage happen automatically on every turn — no wrapper scripts, no manual steps.

**How it works:**
- `UserPromptSubmit` hook queries memory before each prompt → injects top-5 results as context Claude sees automatically
- `Stop` hook stores Claude's response after each turn → memory grows every session

**Setup (one time):**

```bash
# 1. Copy hooks into your project
mkdir -p .claude/memory/hooks
cp cortex.py memory.py harness.py .claude/memory/
cp examples/claude_code_hooks/on_prompt.py .claude/memory/hooks/
cp examples/claude_code_hooks/on_stop.py   .claude/memory/hooks/

# 2. Add hooks config
cp examples/claude_code_hooks/settings.json .claude/settings.json

# 3. Seed initial memory (optional)
python -c "
from memory import Memory
mem = Memory.create(description='my project')
mem.store('uses Python, FastAPI, PostgreSQL')
mem.save('.claude/memory/project.memory')
"
```

**Then just run `claude` normally.** Memory injection and storage are automatic.

See `examples/claude_code_hooks/setup.md` for full setup instructions, tuning options, and verification steps.

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

## Team and shared repositories

When multiple people (or multiple branches) update the same `.memory` file, git would normally mark it as a binary conflict. A custom merge driver resolves this automatically using `Memory.merge()`.

**Setup (one time per machine):**

```bash
python examples/git_merge_driver.py --install
```

This registers the merge driver in `~/.gitconfig` and adds a `*.memory` entry to `.gitattributes`. Commit `.gitattributes` and share the install command with teammates — that's the only setup required.

**What it does on conflict:**

When git would fail on a binary `.memory` conflict, the driver runs instead:
- Memories: union of both sides
- Weights: max-pooled (whichever branch found a memory more useful wins)
- Co-retrieval counts: summed across both sides
- Clusters: rebuilt from the merged structure

Neither side loses context. The result is committed as the resolved file.

**Merge two files directly (without git):**

```bash
python examples/git_merge_driver.py --merge alice.memory bob.memory --output team.memory
```

**Recommended `.gitattributes`:**

```
*.memory merge=cortex-memory
*.memory -diff
*.memory linguist-generated=true
```

The `-diff` flag hides `.memory` from PR diffs in GitHub/Bitbucket — reviewers won't see binary noise in code reviews. The `linguist-generated` flag excludes it from language stats.

**Limitations and considerations:**

*Path is machine-specific.* The `--install` command writes the absolute path to the script in `~/.gitconfig`. This means every person on every machine must run `--install` from their own clone location — the configuration doesn't travel with the repo. If you move or re-clone the repo, run `--install` again.

*Python environment.* Git calls merge drivers in a non-interactive shell that may not inherit your terminal's PATH or virtual environment. If `numpy`/`scipy` aren't available to the Python git finds, the driver fails silently and git falls back to marking the file as a binary conflict. Verify with `git merge --no-ff` on a test branch after installing. If it fails, set the driver command to use the explicit Python binary path: `python3 /absolute/path/to/git_merge_driver.py %O %A %B`.

*Deletions don't propagate.* `Memory.merge()` is additive — union of both sides. If you called `mem.forget(id)` on one branch to remove a memory, the merge will resurrect it from the other side. There is no delete propagation. If deliberate memory pruning matters for your use case, prune after merging rather than before.

*Base version is unused.* Git passes a base (`%O`), ours (`%A`), and theirs (`%B`) but the driver currently ignores the base and unions ours and theirs directly. It cannot distinguish "added on this branch" from "present since the beginning," which means it cannot propagate deletes or resolve genuine semantic conflicts.

*Does not run in CI.* Git merge drivers are local configuration (`~/.gitconfig`) and are not committed to the repo. They will not run in pipeline environments unless explicitly installed there. For CI-based consolidation, call `Memory.merge()` directly in your pipeline script rather than relying on git's merge driver mechanism.

*Silent failure.* If the driver errors for any reason, git marks the file as conflicted rather than aborting. The merge continues but the `.memory` file may be in an inconsistent state. Check `git status` after any merge that touches `.memory` files.

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

See `benchmark.py` to reproduce.

---

## Repository structure

```
cortex-memory/
├── cortex.py              # storage engine (VectorizedBM25, Cortex)
├── memory.py              # portable artifact (Memory class, merge)
├── harness.py             # LLM integration (MemoryHarness, Claude/OpenAI)
├── benchmark.py           # reproduce the benchmarks
├── requirements.txt
└── examples/
    ├── demo.py                      # basic usage, no API needed
    ├── claude_api.py                # interactive Claude conversation loop
    ├── git_merge_driver.py          # git merge driver for team/shared repos
    └── claude_code_hooks/           # Claude Code native hook integration
        ├── on_prompt.py             # UserPromptSubmit — inject memory as context
        ├── on_stop.py               # Stop — auto-store Claude responses
        ├── settings.json            # .claude/settings.json template
        └── setup.md                 # full setup and tuning guide
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
