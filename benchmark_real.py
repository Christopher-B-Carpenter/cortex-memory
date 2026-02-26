"""
benchmark_real.py — Benchmark Cortex against a real .memory file.

Unlike the paper benchmarks (synthetic corpus, known topic labels),
this tests against actual accumulated memory with real queries.

Measures:
  1. Retrieval quality: Cortex (with structure) vs fresh BM25 (same content, no structure)
  2. Structure stats: clusters, coverage, weight distribution, co-retrieval density
  3. Latency: query time with and without two-pass
  4. Context coherence: do retrieved results hang together?
  5. Portability: save/load lossless, file size

Usage:
    python benchmark_real.py ~/.claude/memory/global.memory
    python benchmark_real.py .claude/memory/project.memory --queries queries.txt
"""

import sys, os, time, argparse, itertools
import numpy as np

try:
    from cortex_memory import Memory
    from cortex_memory.cortex import Cortex, VectorizedBM25
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from memory import Memory
    from cortex import Cortex, VectorizedBM25


# ---------------------------------------------------------------------------
# Fresh BM25 baseline — same content, zero learned structure
# ---------------------------------------------------------------------------

class FreshBM25:
    """Flat BM25 over the same memories, with no weights/clusters/co-retrieval."""
    def __init__(self, memories):
        self._idx = VectorizedBM25()
        self._texts = {}
        for mid, text in memories:
            self._idx.add(mid, text)
            self._texts[mid] = text

    def query(self, q, k=5):
        t0 = time.perf_counter()
        results = self._idx.search_all(q, top_k=k)
        lat = (time.perf_counter() - t0) * 1000
        return [(mid, score) for mid, score in results], lat


# ---------------------------------------------------------------------------
# Test queries — mix of real-world patterns
# ---------------------------------------------------------------------------

DEFAULT_QUERIES = [
    # Broad context retrieval
    "what do we know about authentication and tokens",
    "Salesforce CPQ configuration and pricing",
    "deployment and CI/CD pipeline",
    "database performance and indexing",
    "git branching workflow",
    # Specific recall
    "JWT token expiry decisions",
    "Zuora billing integration",
    "Bitbucket pipeline issues",
    "Claude Code hooks and memory",
    "NAICS code scoring",
    # Cross-domain
    "what architecture decisions did we make",
    "testing strategy and test automation",
    "Lightning Web Components development",
    "team workflow and preferences",
    "Python environment and tooling",
    # Edge cases
    "something completely unrelated to anything stored",
    "recent work and active projects",
    "troubleshooting and debugging",
    "API integration patterns",
    "data migration approach",
]


def load_queries(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]
    return DEFAULT_QUERIES


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def benchmark_structure(mem):
    """Analyze the learned structure in the memory file."""
    print("\n── Structure Analysis ──")
    s = mem.stats()
    cortex = mem._cortex

    print(f"  Memories:          {s['n_memories']}")
    print(f"  Clusters:          {s['n_clusters']}")
    print(f"  Coverage:          {s.get('coverage', 0):.0%}")
    print(f"  Queries to date:   {s['query_count']}")
    print(f"  Weight Gini:       {s.get('weight_gini', 0):.3f}")
    print(f"  Never retrieved:   {s.get('never_retrieved', 0)}")

    # Weight distribution
    weights = [m.effective_weight(cortex.decay, cortex.query_count, cortex.floor)
               for m in cortex._memories.values()]
    if weights:
        print(f"  Weight range:      {min(weights):.3f} – {max(weights):.3f}")
        print(f"  Weight median:     {np.median(weights):.3f}")

    # Co-retrieval density
    total_coret = sum(sum(v.values()) for v in cortex._coret.values())
    n_pairs = sum(len(v) for v in cortex._coret.values())
    print(f"  Co-retrieval pairs: {n_pairs}")
    print(f"  Total co-retrieval: {total_coret}")

    # Top memories by weight
    print(f"\n  Top 5 memories by usage weight:")
    for m in mem.top_memories(5):
        print(f"    [{m['retrieval_count']:3d}x  w={m['weight']:.2f}] {m['text'][:75]}")

    # Clusters
    clusters = mem.clusters(5)
    if clusters:
        print(f"\n  Top clusters:")
        for cl in clusters:
            print(f"    {cl['id']} ({cl['size']} members): {cl['rep'][:65]}")

    return s


def benchmark_retrieval(mem, queries, k=5):
    """Compare Cortex retrieval vs fresh BM25 over the same content."""
    print(f"\n── Retrieval Comparison ({len(queries)} queries, top_k={k}) ──")

    cortex = mem._cortex

    # Build fresh baseline with same content
    mem_pairs = [(mid, m.text) for mid, m in cortex._memories.items()]
    fresh = FreshBM25(mem_pairs)

    cortex_results = []
    fresh_results = []
    cortex_latencies = []
    fresh_latencies = []
    overlap_scores = []
    cortex_coherences = []

    for q in queries:
        # Cortex query (with learned structure)
        t0 = time.perf_counter()
        c_res = cortex.query(q, top_k=k)
        c_lat = (time.perf_counter() - t0) * 1000
        c_ids = [r['id'] for r in c_res]
        cortex_results.append(c_res)
        cortex_latencies.append(c_lat)

        # Fresh BM25 query (no structure)
        f_res, f_lat = fresh.query(q, k)
        f_ids = [mid for mid, _ in f_res]
        fresh_results.append(f_res)
        fresh_latencies.append(f_lat)

        # Overlap between the two
        c_set = set(c_ids[:k])
        f_set = set(f_ids[:k])
        if c_set or f_set:
            overlap = len(c_set & f_set) / len(c_set | f_set)
            overlap_scores.append(overlap)

        # Coherence: mean co-retrieval count among Cortex results
        pairs = list(itertools.combinations(c_ids[:k], 2))
        if pairs:
            coh = np.mean([
                cortex._coret.get(a, {}).get(b, 0) +
                cortex._coret.get(b, {}).get(a, 0)
                for a, b in pairs
            ])
            cortex_coherences.append(coh)

    # Summary stats
    c_lat_p50 = np.percentile(cortex_latencies, 50)
    c_lat_p95 = np.percentile(cortex_latencies, 95)
    f_lat_p50 = np.percentile(fresh_latencies, 50)
    f_lat_p95 = np.percentile(fresh_latencies, 95)
    mean_overlap = np.mean(overlap_scores) if overlap_scores else 0
    mean_coherence = np.mean(cortex_coherences) if cortex_coherences else 0

    print(f"\n  Latency:")
    print(f"    Cortex  p50={c_lat_p50:.2f}ms  p95={c_lat_p95:.2f}ms")
    print(f"    Fresh   p50={f_lat_p50:.2f}ms  p95={f_lat_p95:.2f}ms")

    print(f"\n  Result overlap (Jaccard):")
    print(f"    Mean:    {mean_overlap:.2f}  (1.0 = identical results, 0.0 = completely different)")
    overlaps_binned = [o for o in overlap_scores]
    high = sum(1 for o in overlaps_binned if o > 0.6)
    low = sum(1 for o in overlaps_binned if o < 0.3)
    print(f"    High overlap (>0.6): {high}/{len(overlaps_binned)} queries")
    print(f"    Low overlap  (<0.3): {low}/{len(overlaps_binned)} queries")

    print(f"\n  Context coherence (Cortex only):")
    print(f"    Mean co-retrieval count among results: {mean_coherence:.1f}")

    # Two-pass efficiency
    savings = [cortex.diagnostics.get('savings_pct', 0)]
    print(f"\n  Two-pass savings: {np.mean(savings):.0f}% of memories skipped (last query)")

    # Show per-query breakdown
    print(f"\n  Per-query breakdown:")
    print(f"  {'Query':<50s} {'Overlap':>8s} {'C.lat':>7s} {'F.lat':>7s}")
    print(f"  {'-'*50} {'-'*8} {'-'*7} {'-'*7}")
    for i, q in enumerate(queries):
        ol = overlap_scores[i] if i < len(overlap_scores) else 0
        cl = cortex_latencies[i]
        fl = fresh_latencies[i]
        marker = " ***" if ol < 0.3 else ""
        print(f"  {q[:50]:<50s} {ol:>7.2f}  {cl:>6.2f}  {fl:>6.2f}{marker}")

    return {
        'c_lat_p50': c_lat_p50, 'c_lat_p95': c_lat_p95,
        'f_lat_p50': f_lat_p50, 'f_lat_p95': f_lat_p95,
        'mean_overlap': mean_overlap,
        'mean_coherence': mean_coherence,
    }


def benchmark_divergence(mem, queries, k=5):
    """Show WHERE Cortex and fresh BM25 disagree — the interesting cases."""
    print(f"\n── Divergence Analysis (where structure changes results) ──")

    cortex = mem._cortex
    mem_pairs = [(mid, m.text) for mid, m in cortex._memories.items()]
    fresh = FreshBM25(mem_pairs)

    for q in queries:
        c_res = cortex.query(q, top_k=k)
        f_res, _ = fresh.query(q, k)

        c_ids = set(r['id'] for r in c_res)
        f_ids = set(mid for mid, _ in f_res)

        # Only show queries where results differ
        cortex_only = c_ids - f_ids
        fresh_only = f_ids - c_ids

        if not cortex_only and not fresh_only:
            continue

        print(f"\n  Query: \"{q}\"")

        if cortex_only:
            print(f"    Cortex added (via structure):")
            for mid in cortex_only:
                m = cortex._memories.get(mid)
                if m:
                    w = m.effective_weight(cortex.decay, cortex.query_count, cortex.floor)
                    print(f"      [{m.retrieval_count}x w={w:.2f}] {m.text[:80]}")

        if fresh_only:
            print(f"    Fresh BM25 had (Cortex dropped):")
            for mid in fresh_only:
                m = cortex._memories.get(mid)
                if m:
                    print(f"      [{m.retrieval_count}x] {m.text[:80]}")


def benchmark_portability(mem, path):
    """Save/load cycle — verify lossless and measure overhead."""
    print(f"\n── Portability ──")
    import tempfile

    with tempfile.NamedTemporaryFile(suffix='.memory', delete=False) as tmp:
        tmp_path = tmp.name

    # Save
    t0 = time.perf_counter()
    mem.save(tmp_path)
    save_ms = (time.perf_counter() - t0) * 1000

    file_kb = os.path.getsize(tmp_path) / 1024
    orig_kb = os.path.getsize(path) / 1024 if os.path.exists(path) else 0

    # Load
    t0 = time.perf_counter()
    mem2 = Memory.load(tmp_path)
    load_ms = (time.perf_counter() - t0) * 1000

    # Verify lossless
    test_queries = ["authentication", "deployment", "database", "architecture", "testing"]
    mismatches = 0
    for q in test_queries:
        r1 = mem.query(q, top_k=3)
        r2 = mem2.query(q, top_k=3)
        if r1 != r2:
            mismatches += 1

    n = mem.memory_count
    print(f"  Original file: {orig_kb:.1f} KB")
    print(f"  Resaved file:  {file_kb:.1f} KB  ({file_kb*1024/max(n,1):.0f} bytes/memory)")
    print(f"  Save time:     {save_ms:.1f} ms")
    print(f"  Load time:     {load_ms:.1f} ms")
    print(f"  Lossless:      {'Yes' if mismatches == 0 else f'No ({mismatches} mismatches)'}")

    os.unlink(tmp_path)
    return {'save_ms': save_ms, 'load_ms': load_ms, 'file_kb': file_kb, 'lossless': mismatches == 0}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark Cortex against a real .memory file")
    parser.add_argument("memory_file", help="Path to .memory file")
    parser.add_argument("--queries", default=None, help="Optional file with one query per line")
    parser.add_argument("--top-k", type=int, default=5, help="Results per query (default: 5)")
    args = parser.parse_args()

    if not os.path.exists(args.memory_file):
        print(f"Error: {args.memory_file} not found")
        sys.exit(1)

    print(f"\n=== Real-World Benchmark: {args.memory_file} ===")
    t0 = time.time()

    mem = Memory.load(args.memory_file)
    queries = load_queries(args.queries)

    benchmark_structure(mem)
    benchmark_retrieval(mem, queries, k=args.top_k)
    benchmark_divergence(mem, queries, k=args.top_k)
    benchmark_portability(mem, args.memory_file)

    print(f"\n=== Done ({time.time()-t0:.1f}s) ===\n")


if __name__ == "__main__":
    main()
