"""
examples/claude_api.py — Cortex + Claude API

Portable memory with the Anthropic SDK. The memory file accumulates
context across sessions and across tools — the same .memory file works
with Claude Code hooks, this script, or any other harness.

Typical workflow:
  - Run Claude Code with hooks for daily development sessions
    (memory auto-populates from those sessions)
  - Run this script for interactive Q&A against the accumulated memory
  - The same project.memory file is used by both

Requirements:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

Usage:
    python claude_api.py                          # uses ./project.memory
    python claude_api.py --memory /path/to/file   # any .memory file
    python claude_api.py --topic "auth service"   # focus initial retrieval
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from harness import ClaudeMemoryHarness


def main():
    parser = argparse.ArgumentParser(description="Claude API with Cortex memory")
    parser.add_argument("--memory", default="project.memory",
                        help="Path to .memory file (default: ./project.memory)")
    parser.add_argument("--model",  default="claude-opus-4-5",
                        help="Claude model to use")
    parser.add_argument("--top-k",  type=int, default=5,
                        help="Memories to inject per turn (default: 5)")
    parser.add_argument("--topic",  default=None,
                        help="Optional topic hint for first query")
    args = parser.parse_args()

    harness = ClaudeMemoryHarness(
        args.memory,
        model=args.model,
        system_prompt=(
            "You are a technical assistant with deep context about this project. "
            "The <memory> block in your context contains relevant facts accumulated "
            "from prior sessions. Use it to give informed, continuous answers. "
            "When you reach a decision or solve a problem, state it clearly — "
            "your response will be stored for future sessions."
        ),
        top_k=args.top_k,
        store_responses=True,
        min_store_length=80,
        create_if_missing=True,
        description=os.path.basename(os.path.dirname(os.path.abspath(args.memory))),
    )

    s = harness.mem.stats()
    print(f"\nMemory: {harness.mem}")
    print(f"  {s['n_memories']} memories  "
          f"{s['n_clusters']} clusters  "
          f"{s['query_count']} prior queries")
    print("\nType 'quit' to exit  |  'status' for memory stats  |  "
          "'store <text>' to save something explicitly\n")

    # Optional: warm up retrieval with a topic so first response has context
    if args.topic and s['n_memories'] > 0:
        warm = harness.build_system_prompt(args.topic)
        n_injected = warm.count("\n• ")
        print(f"[memory] Warmed up on topic '{args.topic}' — "
              f"{n_injected} memories pre-loaded\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        # ── Built-in meta commands ─────────────────────────────────────────
        if user_input.lower() in ("quit", "exit", "q"):
            break

        if user_input.lower() == "status":
            s = harness.mem.stats()
            diag = harness.mem.diagnostics()
            print(f"\n  Memories:  {s['n_memories']}")
            print(f"  Clusters:  {s['n_clusters']} ({s['coverage']:.0%} coverage)")
            print(f"  Queries:   {s['query_count']}")
            print(f"  Gini:      {s['weight_gini']:.3f}")
            if diag:
                print(f"  Last query: {diag.get('n_scored','?')} scored  "
                      f"{diag.get('savings_pct',0):.0f}% skipped  "
                      f"{diag.get('latency_ms',0):.1f}ms")
            print()
            continue

        if user_input.lower().startswith("store "):
            text = user_input[6:].strip()
            if text:
                harness.store(text)
                print(f"  [stored] {text[:70]}\n")
            continue

        # ── Normal chat turn ───────────────────────────────────────────────
        response = harness.chat(user_input)
        print(f"\nClaude: {response}\n")

        # Show retrieval diagnostics unobtrusively
        diag = harness.mem.diagnostics()
        if diag and diag.get('n_total', 0) > 0:
            print(f"  [memory: {diag.get('n_scored','?')}/{diag.get('n_total','?')} scored  "
                  f"{diag.get('savings_pct',0):.0f}% skipped  "
                  f"{diag.get('latency_ms',0):.1f}ms]\n")

    harness.save()
    s = harness.mem.stats()
    print(f"\nSaved: {harness.mem}")
    print(f"  {s['n_memories']} memories  {s['query_count']} queries")


if __name__ == "__main__":
    main()
