#!/usr/bin/env python3
"""
examples/claude_code_hooks/on_stop.py — Stop hook

Drop this into .claude/memory/hooks/on_stop.py and wire it in
.claude/settings.json (see setup.md).

How it works:
  Claude Code fires Stop when Claude finishes each response.
  This hook reads the last assistant turn from the session transcript,
  stores it in memory, and writes session_stats.json for the statusline.

  Memory grows every session with no manual steps.

Claude Code stdin schema:
  {
    "hook_event_name": "Stop",
    "session_id":      "abc123",
    "transcript_path": "/Users/.../.claude/projects/.../session.jsonl",
    "cwd":             "/Users/.../my-project"
  }
"""

import sys, os, json, time

HOOK_DIR    = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR  = os.path.dirname(HOOK_DIR)
MEMORY_FILE = os.path.join(MEMORY_DIR, "project.memory")
STATS_FILE  = os.path.join(MEMORY_DIR, "session_stats.json")
CONFIG_FILE = os.path.join(MEMORY_DIR, "config.json")
sys.path.insert(0, MEMORY_DIR)


def load_config():
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

MIN_LENGTH = 40
MAX_LENGTH = 2000


def last_assistant_text(transcript_path):
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
                # Handle both flat {role, content} and nested {type, message} formats
                if msg.get("type") == "assistant":
                    inner = msg.get("message", {})
                    content = inner.get("content", "")
                else:
                    if msg.get("role") != "assistant":
                        continue
                    content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                if isinstance(content, str) and content.strip():
                    turns.append(content.strip())
        return turns[-1] if turns else None
    except Exception:
        return None


def get_session_id(transcript_path):
    if not transcript_path:
        return None
    return os.path.splitext(os.path.basename(transcript_path))[0]


def load_stats():
    try:
        with open(STATS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def maybe_push_async(config, memory_path):
    """Push memory file to git repo in background (non-blocking)."""
    gs_cfg = config.get("git_sync", {})
    if not gs_cfg.get("enabled"):
        return
    try:
        from cortex_memory.git_sync import GitSync
        gs = GitSync.from_config(gs_cfg)
        if gs:
            gs.push_async(memory_path)
    except Exception as e:
        sys.stderr.write(f"[memory/on_stop] git push error: {e}\n")


def main():
    config = load_config()

    try:
        event = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    if not os.path.exists(MEMORY_FILE):
        sys.exit(0)

    transcript_path = event.get("transcript_path", "")
    session_id = get_session_id(transcript_path)
    text = last_assistant_text(transcript_path)

    try:
        from memory import Memory
        mem = Memory.load(MEMORY_FILE)
        before = mem.memory_count

        if text and len(text) >= MIN_LENGTH:
            if len(text) > MAX_LENGTH:
                text = text[:MAX_LENGTH].rsplit(" ", 1)[0] + "…"
            datestamp = time.strftime("[%Y-%m-%d]")
            mem.store(text, metadata=datestamp)
            mem.save(MEMORY_FILE)

        after = mem.memory_count
        s = mem.stats()

        # Track session baseline (resets when session ID changes)
        stats = load_stats()
        if stats.get("session_id") != session_id:
            session_baseline = before
        else:
            session_baseline = stats.get("session_baseline", before)

        with open(STATS_FILE, "w") as f:
            json.dump({
                "session_id":       session_id,
                "session_baseline": session_baseline,
                "session_added":    after - session_baseline,
                "total":            after,
                "clusters":         s.get("n_clusters", 0),
                "queries":          s.get("query_count", 0),
                "coverage":         s.get("coverage", 0),
                "updated_at":       time.time(),
            }, f)

        # Push to git repo async after storing (non-blocking)
        maybe_push_async(config, MEMORY_FILE)

    except Exception as e:
        sys.stderr.write(f"[memory/on_stop] {e}\n")

    sys.exit(0)


if __name__ == "__main__":
    main()
