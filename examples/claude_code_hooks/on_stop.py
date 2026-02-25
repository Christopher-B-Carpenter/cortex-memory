#!/usr/bin/env python3
"""
examples/claude_code_hooks/on_stop.py — Stop hook

Drop this into .claude/memory/hooks/on_stop.py in your project and
wire it in .claude/settings.json (see setup.md in this directory).

How it works:
  Claude Code fires Stop when Claude finishes each response.
  This hook reads the last assistant turn from the session transcript
  and stores it in project memory automatically.

  Memory grows every session with no manual steps.

Claude Code stdin schema:
  {
    "hook_event_name": "Stop",
    "session_id":      "abc123",
    "transcript_path": "/Users/.../.claude/projects/.../session.jsonl",
    "cwd":             "/Users/.../my-project"
  }

No return value needed — exit 0 silently.
"""

import sys, os, json

# ── Resolve paths ─────────────────────────────────────────────────────────────
HOOK_DIR    = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR  = os.path.dirname(HOOK_DIR)
MEMORY_FILE = os.path.join(MEMORY_DIR, "project.memory")
sys.path.insert(0, MEMORY_DIR)

MIN_LENGTH = 80    # skip short acks / one-liners
MAX_LENGTH = 600   # truncate long responses — front usually has the decision


def last_assistant_text(transcript_path: str) -> str | None:
    """Return the most recent assistant message from a JSONL transcript."""
    if not transcript_path or not os.path.exists(transcript_path):
        return None
    try:
        turns = []
        with open(transcript_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                role    = msg.get("role", "")
                content = msg.get("content", "")
                # Handle Anthropic content block format
                if isinstance(content, list):
                    content = " ".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                if role == "assistant" and isinstance(content, str):
                    turns.append(content.strip())
        return turns[-1] if turns else None
    except Exception:
        return None


def main():
    # ── Parse event ───────────────────────────────────────────────────────────
    try:
        event = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    if not os.path.exists(MEMORY_FILE):
        sys.exit(0)

    # ── Extract last response ─────────────────────────────────────────────────
    text = last_assistant_text(event.get("transcript_path", ""))
    if not text or len(text) < MIN_LENGTH:
        sys.exit(0)

    # Truncate — the front of a response contains the summary/decision;
    # the tail is usually code blocks or boilerplate we don't need.
    if len(text) > MAX_LENGTH:
        text = text[:MAX_LENGTH].rsplit(" ", 1)[0] + "…"

    # ── Store ─────────────────────────────────────────────────────────────────
    try:
        from memory import Memory
        mem = Memory.load(MEMORY_FILE)
        mem.store(text)
        mem.save(MEMORY_FILE)
    except Exception as e:
        sys.stderr.write(f"[memory/on_stop] {e}\n")

    sys.exit(0)


if __name__ == "__main__":
    main()
