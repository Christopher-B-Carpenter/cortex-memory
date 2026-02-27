#!/usr/bin/env python3
"""
examples/claude_code_hooks/statusline.py — Cortex Memory statusline

Shows live memory stats in the Claude Code status bar:
  ⬡ +5 this session  │  185 total  7 clusters  194 queries

Reads session_stats.json written by on_stop.py after each turn.
Near-zero latency — no memory file unpickling on each update.

Setup in ~/.claude/settings.json (adjust path to match your install):
  "statusLine": {
    "type": "command",
    "command": "python3 /path/to/.claude/memory/statusline.py"
  }

Or use the one-command installer:
  python3 -m cortex_memory install --statusline
"""

import sys, os, json, time

# Resolve stats file relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.path.dirname(SCRIPT_DIR)
STATS_FILE = os.path.join(MEMORY_DIR, "session_stats.json")

# ANSI colors
CYAN  = "\033[38;5;51m"
DIM   = "\033[38;5;240m"
GREEN = "\033[38;5;78m"
RESET = "\033[0m"
BOLD  = "\033[1m"


def main():
    # Claude Code passes session data via stdin — consume it
    try:
        json.load(sys.stdin)
    except Exception:
        pass

    try:
        with open(STATS_FILE) as f:
            stats = json.load(f)
    except Exception:
        print(f"{DIM}⬡ memory{RESET}")
        return

    session_added = stats.get("session_added", 0)
    total         = stats.get("total", 0)
    clusters      = stats.get("clusters", 0)
    queries       = stats.get("queries", 0)
    stale         = "?" if time.time() - stats.get("updated_at", 0) > 300 else ""

    delta_color = GREEN if session_added > 0 else DIM
    delta_str   = f"+{session_added}"

    print(
        f"{CYAN}{BOLD}⬡ cortex{RESET}  "
        f"{delta_color}{delta_str} this session{RESET}  "
        f"{DIM}│{RESET}  "
        f"{DIM}{total} memories  {clusters} clusters  {queries} queries{stale}{RESET}"
    )


if __name__ == "__main__":
    main()
