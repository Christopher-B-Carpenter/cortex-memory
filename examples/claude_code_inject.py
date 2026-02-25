#!/usr/bin/env python3
"""
examples/claude_code_inject.py — Cortex + Claude Code CLI

Two modes:

  inject   — Query memory and prepend results to CLAUDE.md, then start Claude Code.
             Run this instead of `claude` at the start of each session.

  sync     — After a Claude Code session, store the transcript into memory.
             Run this after `claude` exits.

Usage:
    # Start a memory-augmented Claude Code session
    python claude_code_inject.py inject

    # Or with a specific topic to focus the memory retrieval
    python claude_code_inject.py inject --topic "auth service"

    # After the session, sync the transcript
    python claude_code_inject.py sync

    # Point to a specific memory file or transcript
    python claude_code_inject.py inject --memory /path/to/project.memory
    python claude_code_inject.py sync --transcript ~/.claude/projects/.../transcript.jsonl

Typical workflow in your shell profile or Makefile:
    alias claude-mem='python /path/to/examples/claude_code_inject.py inject'

How it works:
    inject:
      1. Loads (or creates) project.memory in the current directory
      2. Queries with "current project context" (or --topic)
      3. Prepends a ## Active Memory Context section to CLAUDE.md
      4. Execs `claude` so it reads the enriched CLAUDE.md at startup

    sync:
      1. Finds the most recent Claude Code transcript
      2. Extracts assistant turns above a minimum length
      3. Stores them in project.memory
      4. Saves the updated memory file
"""

import sys, os, subprocess, glob, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from harness import MemoryHarness

DEFAULT_MEMORY = "project.memory"


def find_latest_transcript() -> str | None:
    """Find the most recently modified Claude Code transcript."""
    home = os.path.expanduser("~")
    patterns = [
        os.path.join(home, ".claude", "projects", "*", "*.jsonl"),
        os.path.join(home, ".config", "claude", "projects", "*", "*.jsonl"),
        os.path.join(home, "Library", "Application Support", "Claude", "projects", "*", "*.jsonl"),
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def cmd_inject(args):
    """Inject memory into CLAUDE.md and start Claude Code."""
    memory_path = args.memory or DEFAULT_MEMORY

    harness = MemoryHarness(
        memory_path,
        create_if_missing=True,
        description=os.path.basename(os.getcwd()),
        top_k=args.top_k,
    )

    query = args.topic or f"{os.path.basename(os.getcwd())} project context"

    harness.inject_claude_md(
        query=query,
        claude_md_path=args.claude_md,
        top_k=args.top_k,
    )

    n = harness.mem.memory_count
    q = harness.mem.query_count
    print(f"[cortex] {n} memories, {q} queries → injected into {args.claude_md}")

    # Exec claude (replaces this process)
    claude_args = ["claude"] + (args.claude_args or [])
    os.execvp("claude", claude_args)


def cmd_sync(args):
    """Store a Claude Code transcript into memory."""
    memory_path = args.memory or DEFAULT_MEMORY
    transcript = args.transcript or find_latest_transcript()

    if not transcript:
        print("[cortex] No transcript found. Looked in ~/.claude/projects/")
        print("         Pass --transcript /path/to/file.jsonl explicitly.")
        sys.exit(1)

    harness = MemoryHarness(
        memory_path,
        create_if_missing=True,
        description=os.path.basename(os.getcwd()),
    )

    n_before = harness.mem.memory_count
    n_stored = harness.sync_from_transcript(
        transcript,
        role_filter=args.role,
        min_length=args.min_length,
    )
    harness.save()

    print(f"[cortex] Synced {n_stored} turns from {os.path.basename(transcript)}")
    print(f"         Memory: {n_before} → {harness.mem.memory_count} entries")
    print(f"         Saved: {memory_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cortex memory integration for Claude Code"
    )
    sub = parser.add_subparsers(dest="command")

    p_inject = sub.add_parser("inject", help="Inject memory into CLAUDE.md and start claude")
    p_inject.add_argument("--memory",    default=None,       help="Memory file path (default: ./project.memory)")
    p_inject.add_argument("--topic",     default=None,       help="Query topic for memory retrieval")
    p_inject.add_argument("--top-k",     type=int, default=6,help="Number of memories to inject")
    p_inject.add_argument("--claude-md", default="CLAUDE.md",help="CLAUDE.md path")
    p_inject.add_argument("claude_args", nargs="*",          help="Extra args passed to claude")

    p_sync = sub.add_parser("sync", help="Store a Claude Code session transcript into memory")
    p_sync.add_argument("--memory",     default=None,        help="Memory file path (default: ./project.memory)")
    p_sync.add_argument("--transcript", default=None,        help="Transcript .jsonl path (auto-detected if omitted)")
    p_sync.add_argument("--role",       default="assistant", help="Which role to store (default: assistant)")
    p_sync.add_argument("--min-length", type=int, default=60,help="Minimum response length to store")

    args = parser.parse_args()

    if args.command == "inject":
        cmd_inject(args)
    elif args.command == "sync":
        cmd_sync(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
