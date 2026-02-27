"""
examples/git_memory_chat.py — Terminal chat with git-backed memory

A ready-to-run chat loop that keeps your memory file synced to a
private git repo. Works with any OpenAI-compatible LLM including
Ollama, LM Studio, llama.cpp, Together, Fireworks, or OpenAI itself.

Setup:
    1. Create a private GitHub repo (e.g. github.com/you/memory)
    2. Clone it locally: git clone github.com/you/memory ~/memory
    3. Install: pip install llm-cortex-memory
    4. Install Ollama (or any local LLM): https://ollama.com
    5. Run: python git_memory_chat.py

The memory file lives in your git repo and stays in sync automatically:
  - git pull  when the script starts (get latest from any machine)
  - git push  when the session ends (save back to repo)

Same memory file works with Claude Code hooks, the API harness, or
this script — one continuous thread of memory across all sessions,
machines, and LLMs.

Usage:
    python git_memory_chat.py                        # Ollama defaults
    python git_memory_chat.py --model llama3.2       # different model
    python git_memory_chat.py --repo ~/my-memory     # custom repo path
    python git_memory_chat.py --api openai           # use OpenAI API
"""

import sys, os, argparse

try:
    from cortex_memory import OpenAIMemoryHarness
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from harness import OpenAIMemoryHarness


def main():
    parser = argparse.ArgumentParser(
        description="Terminal chat with git-backed Cortex memory"
    )
    parser.add_argument(
        "--repo",
        default=os.path.expanduser("~/memory"),
        help="Path to local clone of your memory repo (default: ~/memory)",
    )
    parser.add_argument(
        "--memory-file",
        default="global.memory",
        help="Memory filename inside the repo (default: global.memory)",
    )
    parser.add_argument(
        "--model",
        default="llama3.2",
        help="Model name (default: llama3.2)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434/v1",
        help="LLM API base URL (default: Ollama at localhost:11434)",
    )
    parser.add_argument(
        "--api-key",
        default="ollama",
        help="API key (default: 'ollama' for local, use real key for cloud)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Memories to inject per turn (default: 5)",
    )
    parser.add_argument(
        "--no-git",
        action="store_true",
        help="Disable git sync (use local memory file only)",
    )
    args = parser.parse_args()

    memory_path = os.path.join(args.repo, args.memory_file)

    git_sync_config = None
    if not args.no_git:
        git_sync_config = {
            "enabled": True,
            "repo_path": args.repo,
            "remote": "origin",
            "branch": "main",
            "commit_message": "cortex: session sync",
        }

    print(f"\nCortex Memory Chat")
    print(f"  Model:  {args.model} @ {args.base_url}")
    print(f"  Memory: {memory_path}")
    if git_sync_config:
        print(f"  Repo:   {args.repo} (git sync enabled)")
    print("\nPulling latest memory from repo..." if git_sync_config else "")

    harness = OpenAIMemoryHarness(
        memory_path=memory_path,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        top_k=args.top_k,
        store_responses=True,
        create_if_missing=True,
        git_sync=git_sync_config,
    )

    s = harness.mem.stats()
    print(f"\nMemory loaded: {s['n_memories']} memories  "
          f"{s['n_clusters']} clusters  {s['query_count']} prior queries")
    print("\nType 'quit' to exit  |  'status' for memory stats\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            break

        if user_input.lower() == "status":
            s = harness.mem.stats()
            print(f"\n  Memories:  {s['n_memories']}")
            print(f"  Clusters:  {s['n_clusters']} ({s.get('coverage', 0):.0%} coverage)")
            print(f"  Queries:   {s['query_count']}")
            print()
            continue

        response = harness.chat(user_input)
        print(f"\nAssistant: {response}\n")

    print("\nSaving and syncing memory...")
    harness.save()  # flush stores + push to git async
    s = harness.mem.stats()
    print(f"Done. {s['n_memories']} memories  {s['query_count']} queries")
    if git_sync_config:
        print("Pushed to git repo (background sync).")


if __name__ == "__main__":
    main()
