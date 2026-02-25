"""
examples/demo.py — Basic Cortex usage

Shows the three core operations: store, query, save/load.
No API key required — uses a mock LLM to demonstrate the harness.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from memory import Memory
from harness import MemoryHarness


# ─── 1. Create a memory store ─────────────────────────────────────────────────

mem = Memory.create(
    description="payments-service development",
    tags=["python", "auth", "database"],
)

# Store memories (typically these come from LLM responses)
mem.store("Decided to use JWT tokens with 24-hour expiry for session management.")
mem.store("SQL injection found in legacy login query — switched to parameterized queries.")
mem.store("Dashboard query reduced from 8s to 200ms with composite index on (user_id, created_at).")
mem.store("Redis cache added in front of expensive DB lookups; 94% hit rate after TTL tuning.")
mem.store("JWT secret rotation implemented with 24-hour grace period for old tokens.")
mem.store("Blue-green deployment to production — zero downtime, 4 minutes total.")
mem.store("Connection pool exhausted under load; increased max connections to 50 via pgbouncer.")

# Simulate some queries to build co-retrieval structure
for q in [
    "auth token decisions",
    "database performance fixes",
    "security issues",
    "deployment process",
    "caching strategy",
] * 4:
    mem.query(q)

mem.save("project.memory")

print(f"Created: {mem}")
print(f"\nTop memories by weight:")
for m in mem.top_memories(3):
    print(f"  [{m['retrieval_count']}×] {m['text'][:70]}")


# ─── 2. Load on any machine ───────────────────────────────────────────────────

loaded = Memory.load("project.memory")
print(f"\nLoaded: {loaded}")

results = loaded.query("what did we do about authentication security?", top_k=3)
print(f"\nQuery: 'what did we do about authentication security?'")
for r in results:
    print(f"  → {r[:70]}")


# ─── 3. Harness with mock LLM ────────────────────────────────────────────────

def mock_llm(messages, system, **kwargs):
    """Replace with your actual LLM call."""
    last = next(m["content"] for m in reversed(messages) if m["role"] == "user")
    has_memory = "<memory>" in system
    ctx = f"(with {system.count(chr(10) + '- ')} memory items)" if has_memory else "(no memory yet)"
    return f"[Mock response {ctx}] Answering: {last[:60]}"

harness = MemoryHarness(
    "project.memory",
    llm_fn=mock_llm,
    top_k=4,
    description="payments-service dev session",
)

print(f"\n─── Harness demo ───")
for question in [
    "what indexes did we add to fix the slow queries?",
    "remind me about the JWT rotation policy",
]:
    response = harness.chat(question)
    print(f"\nQ: {question}")
    print(f"A: {response}")

harness.save()
print(f"\nFinal store: {harness}")

import os; os.unlink("project.memory")
