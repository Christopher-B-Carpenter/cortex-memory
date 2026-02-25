#!/usr/bin/env python3
"""
examples/claude_code_hooks/on_prompt.py — UserPromptSubmit hook

Drop this into .claude/memory/hooks/on_prompt.py in your project and
wire it in .claude/settings.json (see setup.md in this directory).

How it works:
  Claude Code fires UserPromptSubmit before every prompt is processed.
  This hook queries project memory with the user's message and returns
  relevant context as additionalContext — injected automatically into
  Claude's context window, invisible to the user.

  Claude sees [Project Memory] on every turn with no manual steps.

Claude Code stdin schema:
  {
    "hook_event_name": "UserPromptSubmit",
    "session_id":      "abc123",
    "transcript_path": "/Users/.../.claude/projects/.../session.jsonl",
    "cwd":             "/Users/.../my-project",
    "prompt":          "why is the dashboard query slow?"
  }

Return schema:
  { "additionalContext": "string injected before Claude processes prompt" }

Exit 0 always — a hook failure must never block the user's prompt.
"""

import sys, os, json

# ── Resolve paths ─────────────────────────────────────────────────────────────
# Expected layout:
#   .claude/memory/hooks/on_prompt.py   ← this file
#   .claude/memory/cortex.py            ← library
#   .claude/memory/memory.py            ← library
#   .claude/memory/project.memory       ← store
HOOK_DIR    = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR  = os.path.dirname(HOOK_DIR)
MEMORY_FILE = os.path.join(MEMORY_DIR, "project.memory")
sys.path.insert(0, MEMORY_DIR)

TOP_K = 5   # memories to inject per prompt


def main():
    # ── Parse event ───────────────────────────────────────────────────────────
    try:
        event = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    prompt = event.get("prompt", "").strip()
    if not prompt or not os.path.exists(MEMORY_FILE):
        sys.exit(0)

    # ── Query memory ──────────────────────────────────────────────────────────
    try:
        from memory import Memory

        mem     = Memory.load(MEMORY_FILE)
        results = mem.query(prompt, top_k=TOP_K)
        mem.save(MEMORY_FILE)           # persist weight update from this query

        if not results:
            sys.exit(0)

        # ── Build context block ───────────────────────────────────────────────
        lines = ["[Project Memory — relevant context]"]
        for r in results:
            lines.append(f"• {r.strip()}")

        print(json.dumps({"additionalContext": "\n".join(lines) + "\n"}))

    except Exception as e:
        # Log to stderr (visible in Claude Code debug output) but never crash
        sys.stderr.write(f"[memory/on_prompt] {e}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
