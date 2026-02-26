# /memory — Memory source management and operations

Manage Cortex memory for this session. Controls which memory sources are
active and lets you query, store, and inspect memory directly.

---

## Source commands

Switch the active memory source by writing to `.claude/memory/config.json`.

**`/memory source project`** — inject from `.claude/memory/project.memory` only  
**`/memory source global`** — inject from `~/.claude/memory/global.memory` only  
**`/memory source both`** — inject from both (default)  
**`/memory source off`** — disable all memory injection  

These take effect immediately — no restart required.

To switch source, run this Python and confirm the change:
```python
import json, os

config_path = ".claude/memory/config.json"
config = json.load(open(config_path)) if os.path.exists(config_path) else {}
config["source"] = "<SOURCE>"  # project | global | both | off
json.dump(config, open(config_path, "w"), indent=2)
print(f"Memory source set to: {config['source']}")
```

---

## Query commands

**`/memory <question>`** — query memory and show results  
**`/memory status`** — show current config and store stats  
**`/memory store <text>`** — explicitly save something to project memory  

### For a query
```python
import sys, json
sys.path.insert(0, ".claude/memory")
from memory import Memory

mem = Memory.load(".claude/memory/project.memory")
results = mem.query("<QUERY>", top_k=8)
mem.save(".claude/memory/project.memory")
for r in results:
    print(f"• {r}")
```

### For status
```python
import sys, json, os
sys.path.insert(0, ".claude/memory")
from memory import Memory

config_path = ".claude/memory/config.json"
config = json.load(open(config_path)) if os.path.exists(config_path) else {"source": "both"}
print(f"Source: {config.get('source', 'both')}  |  top_k: {config.get('top_k', 5)}")

for label, path in [
    ("Project", ".claude/memory/project.memory"),
    ("Global",  os.path.expanduser("~/.claude/memory/global.memory")),
]:
    if os.path.exists(path):
        mem = Memory.load(path)
        s = mem.stats()
        print(f"\n{label}: {s['n_memories']} memories  {s['n_clusters']} clusters  {s['query_count']} queries")
        for m in mem.top_memories(3):
            print(f"  [{m['retrieval_count']}×] {m['text'][:70]}")
    else:
        print(f"\n{label}: not found ({path})")
```

### For store
```python
import sys
sys.path.insert(0, ".claude/memory")
from memory import Memory

mem = Memory.load(".claude/memory/project.memory")
mem.store("<TEXT>")
mem.save(".claude/memory/project.memory")
print(f"Stored. Total: {mem.memory_count} memories")
```

---

## When to use each source

**project** — working on this specific repo, want focused context without noise
from other projects or personal preferences.

**global** — working in a project without its own memory, or want personal
cross-project context (preferred tools, personal workflow patterns).

**both** — default for most development work. Project memory provides repo-specific
context, global memory provides personal preferences and cross-project patterns.
Claude Code merges both automatically.

**off** — temporary, when context injection is causing confusion on a specific task.
