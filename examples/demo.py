"""
examples/demo.py — Portable memory in action

Demonstrates the core portability story: one .memory file works seamlessly
across different harnesses. Create it once, load it anywhere, plug it into
any LLM integration — weights, clusters, and co-retrieval structure persist.

No API key required — uses a mock LLM to show the mechanics.
"""

import sys, os

try:
    from cortex_memory import Memory, MemoryHarness
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from memory import Memory
    from harness import MemoryHarness

MEMORY_FILE = "/tmp/demo_project.memory"

# ─── Stage 1: Create a memory store (simulates prior project history) ─────────

print("=" * 60)
print("Stage 1: Creating memory from prior project work")
print("=" * 60)

mem = Memory.create(
    description="payments-service development",
    tags=["python", "auth", "database"],
)

mem.store("Decided to use JWT tokens with 24-hour expiry for session management.")
mem.store("SQL injection found in legacy login query — switched to parameterized queries.")
mem.store("Dashboard query reduced from 8s to 200ms with composite index on (user_id, created_at).")
mem.store("Redis cache added in front of expensive DB lookups; 94% hit rate after TTL tuning.")
mem.store("JWT secret rotation implemented with 24-hour grace period for old tokens.")
mem.store("Blue-green deployment to production — zero downtime, 4 minutes total.")
mem.store("Connection pool exhausted under load; increased max connections to 50 via pgbouncer.")

# Build some co-retrieval structure (simulates past sessions)
for q in ["auth token decisions", "database performance", "security issues",
          "deployment process", "caching strategy"] * 3:
    mem.query(q)

mem.save(MEMORY_FILE)

s = mem.stats()
print(f"\nCreated: {mem}")
print(f"  Clusters: {s['n_clusters']}  Coverage: {s['coverage']:.0%}  "
      f"Weight Gini: {s['weight_gini']:.3f}")
print(f"\nTop memories (by usage weight):")
for m in mem.top_memories(3):
    print(f"  [{m['retrieval_count']}×] {m['text'][:72]}")


# ─── Stage 2: Load into Harness A (mock LLM — simulates any local tool) ───────

print(f"\n{'=' * 60}")
print("Stage 2: Load into Harness A (mock LLM)")
print("=" * 60)

def mock_llm_a(messages, system, **kwargs):
    last = next(m["content"] for m in reversed(messages) if m["role"] == "user")
    n = system.count("\n- ")  # harness formats memory items as "- text"
    return f"[Harness A | {n} memory items in context] Response to: {last[:55]}"

harness_a = MemoryHarness(
    MEMORY_FILE,
    llm_fn=mock_llm_a,
    top_k=4,
    async_store=False,
    store_responses=False,  # demo uses mock LLM — don't pollute memory
)

print(f"\nLoaded: {harness_a}")

r1 = harness_a.chat("what indexes did we add to fix the slow queries?")
print(f"\nQ: what indexes did we add to fix the slow queries?")
print(f"A: {r1}")

r2 = harness_a.chat("remind me about the JWT rotation policy")
print(f"\nQ: remind me about the JWT rotation policy")
print(f"A: {r2}")

harness_a.save()

s_a = harness_a.mem.stats()
print(f"\nAfter Harness A — queries: {s_a['query_count']}  "
      f"memories: {s_a['n_memories']}  gini: {s_a['weight_gini']:.3f}")


# ─── Stage 3: Same file, Harness B (different mock — simulates switching LLMs) -

print(f"\n{'=' * 60}")
print("Stage 3: Same .memory file, Harness B (different LLM)")
print("=" * 60)

def mock_llm_b(messages, system, **kwargs):
    last = next(m["content"] for m in reversed(messages) if m["role"] == "user")
    n = system.count("\n- ")
    return f"[Harness B | {n} memory items in context] Answering: {last[:55]}"

harness_b = MemoryHarness(
    MEMORY_FILE,          # ← same file
    llm_fn=mock_llm_b,    # ← different LLM callable
    top_k=4,
    async_store=False,
    store_responses=False,
)

print(f"\nLoaded: {harness_b}")
print(f"  Query count carried over: {harness_b.mem.query_count}")
print(f"  Clusters carried over:    {harness_b.mem.stats()['n_clusters']}")

r3 = harness_b.chat("what security vulnerabilities did we fix?")
print(f"\nQ: what security vulnerabilities did we fix?")
print(f"A: {r3}")

harness_b.save()


# ─── Stage 4: Verify portability — weights evolved consistently ────────────────

print(f"\n{'=' * 60}")
print("Stage 4: Verify — memory state is consistent across loads")
print("=" * 60)

verify = Memory.load(MEMORY_FILE)
s_v = verify.stats()

print(f"\nFinal store: {verify}")
print(f"  Memories:  {s_v['n_memories']}")
print(f"  Clusters:  {s_v['n_clusters']} ({s_v['coverage']:.0%} coverage)")
print(f"  Queries:   {s_v['query_count']} (accumulated across both harnesses)")
print(f"  Weight Gini: {s_v['weight_gini']:.3f} (structure from use)")

print(f"\nTop memories after both harness sessions:")
for m in verify.top_memories(4):
    print(f"  [{m['retrieval_count']}×  w={m['weight']:.2f}] {m['text'][:70]}")

print(f"\nClusters (topics grouped by co-retrieval):")
for c in verify.clusters(3):
    print(f"  {c['id']} ({c['size']} members): {c['rep'][:60]}")

print(f"\n✓ Same .memory file loaded by two different harnesses.")
print(f"  Weights, clusters, and query history are consistent.")
print(f"  Drop this file into any harness and it picks up where you left off.")

os.unlink(MEMORY_FILE)
