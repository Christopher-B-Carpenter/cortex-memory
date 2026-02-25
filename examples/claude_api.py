"""
examples/claude_api.py — Cortex + Claude API (Anthropic SDK)

Every conversation turn:
  1. Query memory with the user's message → top-K relevant memories
  2. Inject memories into the system prompt as <memory> context
  3. Call Claude
  4. Store Claude's response asynchronously

The memory file accumulates across sessions. Move it between machines freely.

Requirements:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from harness import ClaudeMemoryHarness

MEMORY_FILE = "project.memory"

harness = ClaudeMemoryHarness(
    MEMORY_FILE,
    model="claude-opus-4-5",
    system_prompt=(
        "You are a technical assistant with deep context about this project. "
        "Use the memories provided to give informed, continuous answers. "
        "When you make a decision or solve a problem, summarize it concisely."
    ),
    top_k=5,
    store_responses=True,
    description="my project memory",
)

print(f"Memory loaded: {harness}")
print("Type 'quit' to exit. Memory saves on exit.\n")

conversation_history = []

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("quit", "exit", "q"):
        break
    if not user_input:
        continue

    response = harness.chat(user_input)
    conversation_history.append({"role": "user",    "content": user_input})
    conversation_history.append({"role": "assistant","content": response})

    print(f"\nClaude: {response}\n")

    # Show what was retrieved (optional, for debugging)
    diag = harness.diagnostics()
    if diag:
        print(f"  [memory: {diag.get('n_scored', '?')} scored, "
              f"{diag.get('savings_pct', 0):.0f}% skipped, "
              f"{diag.get('latency_ms', 0):.1f}ms]\n")

harness.save()
print(f"\nSaved: {harness}")
