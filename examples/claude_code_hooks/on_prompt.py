#!/usr/bin/env python3
"""
examples/claude_code_hooks/on_prompt.py — UserPromptSubmit hook

Drop this into .claude/memory/hooks/on_prompt.py in your project and
wire it in .claude/settings.json (see setup.md in this directory).

Memory source is controlled by .claude/memory/config.json:
  source: "project"  — inject from .claude/memory/project.memory only
  source: "global"   — inject from ~/.claude/memory/global.memory only
  source: "both"     — inject from both (default)
  top_k: 5           — memories to inject per prompt

Switch source at any time with the /memory command (see memory.md).
No restart required — the hook reads config on every invocation.
"""

import sys, os, json

HOOK_DIR    = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR  = os.path.dirname(HOOK_DIR)
CONFIG_FILE = os.path.join(MEMORY_DIR, "config.json")
sys.path.insert(0, MEMORY_DIR)

PROJECT_MEMORY = os.path.join(MEMORY_DIR, "project.memory")
GLOBAL_MEMORY  = os.path.expanduser("~/.claude/memory/global.memory")


def load_config():
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except Exception:
        return {"source": "both", "top_k": 5}


def maybe_pull(config):
    """Pull latest memory from git repo if git_sync is configured."""
    gs_cfg = config.get("git_sync", {})
    if not gs_cfg.get("enabled"):
        return
    try:
        from cortex_memory.git_sync import GitSync
        gs = GitSync.from_config(gs_cfg)
        if gs:
            gs.pull()
    except Exception as e:
        sys.stderr.write(f"[memory/on_prompt] git pull error: {e}\n")


def query_source(path, label, prompt, top_k):
    """Query a memory file, return formatted lines or []."""
    if not os.path.exists(path):
        return []
    try:
        from memory import Memory
        mem     = Memory.load(path)
        results = mem.query(prompt, top_k=top_k)
        mem.save(path)
        if results:
            return [f"[{label}]"] + [f"• {r.strip()}" for r in results]
    except Exception as e:
        sys.stderr.write(f"[memory/on_prompt] {label}: {e}\n")
    return []


def main():
    config = load_config()
    source = config.get("source", "both")
    top_k  = config.get("top_k", 5)

    if source == "off":
        sys.exit(0)

    # Pull latest from git repo before querying (blocking, ~<1s)
    maybe_pull(config)

    try:
        event = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    prompt = event.get("prompt", "").strip()
    if not prompt:
        sys.exit(0)

    lines = []

    if source in ("project", "both"):
        lines += query_source(PROJECT_MEMORY, "Project Memory", prompt, top_k)

    if source in ("global", "both"):
        lines += query_source(GLOBAL_MEMORY, "Global Memory", prompt, top_k)

    if not lines:
        sys.exit(0)

    print(json.dumps({"additionalContext": "\n".join(lines) + "\n"}))


if __name__ == "__main__":
    main()
