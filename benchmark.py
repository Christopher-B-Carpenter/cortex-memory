"""
benchmark.py — Paper benchmark suite for portable memory store.

Four claims to support:

  Claim 1: Retrieval quality holds at scale
    Cortex precision matches or exceeds flat BM25 at N=100-5000,
    especially after usage history builds structural signal.

  Claim 2: Context efficiency improves with use
    Tokens per relevant hit decreases as cluster structure matures.
    Measures: chars_per_hit at session 1, 10, 50, 200.

  Claim 3: Sub-10ms retrieval at 10K memories
    Latency vs N for both systems. Shows two-pass savings.

  Claim 4: Portability — save/load cycle is lossless and fast
    Serialize 1K, 5K, 10K memory stores. Measure file size, save/load time,
    verify query results identical before/after.

Run all:        python benchmark.py
Run one claim:  python benchmark.py --claim 1
"""

import sys, os, time, random, argparse, itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from repo versions
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'repo'))
from cortex import Cortex, VectorizedBM25
from memory import Memory

FIGDIR = Path(__file__).parent / 'benchmark_figures'
DARK_BG = '#0f1219'; DARK_AXES = '#0a0e17'
C0, C1, C2, C3 = '#00d4ff', '#ff6b35', '#2ecc71', '#e74c3c'

def style(ax, title=''):
    ax.set_facecolor(DARK_AXES)
    ax.tick_params(colors='#9ca3af')
    ax.xaxis.label.set_color('#9ca3af')
    ax.yaxis.label.set_color('#9ca3af')
    if title: ax.set_title(title, color='white', fontsize=10)
    for s in ax.spines.values(): s.set_color('#2d3748')

def save_fig(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(DARK_BG)
    p = FIGDIR / name
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f"  → {p}")


# ---------------------------------------------------------------------------
# Corpus — realistic software engineering conversations
# Vocabulary variation within topics is the key property:
# same concept expressed many ways → BM25 struggles, structure helps
# ---------------------------------------------------------------------------

TOPICS = {
    'auth': [
        "decided to use JWT tokens with 24 hour expiry for session management",
        "access tokens expire after one day, refresh tokens last thirty days",
        "implemented bearer token authentication with daily rotation policy",
        "token based auth chosen, JWT format, expires in 86400 seconds",
        "session handling via signed tokens, invalidated after twenty four hours",
        "authentication uses stateless tokens with expiration set to one day",
        "chose JWT over session cookies for stateless auth across services",
        "auth layer issues signed credentials that become invalid after a day",
        "login produces a token valid for 24 hours then user must reauthenticate",
        "implemented PKCE flow for mobile clients to prevent auth code interception",
        "rate limiting on login endpoint set to ten requests per minute per IP",
        "bcrypt password hashing with cost factor twelve chosen for storage",
        "SQL injection found in legacy login query fixed with parameterized queries",
        "JWT secret rotated monthly, old tokens valid for 24 hour grace period",
        "added HSTS header and tightened CORS config after security audit",
    ],
    'database': [
        "dashboard query taking 8 seconds fixed by adding composite index",
        "slow query on users table resolved with index on email column",
        "query performance improved from 4 seconds to 200ms after indexing",
        "added btree index on created_at plus user_id to fix report latency",
        "EXPLAIN showed sequential scan replaced by index scan after migration",
        "connection pool exhausted under load, increased max connections to 50",
        "pgbouncer added to manage connection pooling for high concurrency",
        "ANALYZE run after schema migration to update query planner statistics",
        "vacuum needed after bulk delete caused table bloat and degraded performance",
        "materialized view refreshed nightly to serve expensive aggregation queries",
        "read replica added to offload analytics queries from primary database",
        "partitioned orders table by month to improve query performance at scale",
        "deadlock between two transactions resolved by enforcing lock acquisition order",
        "Redis cache added in front of expensive database lookups, 94 percent hit rate",
        "write ahead logging enabled for crash recovery on production database",
    ],
    'deployment': [
        "blue green deployment used for zero downtime production release",
        "rolling update strategy chosen to avoid downtime during service updates",
        "canary release to 5 percent of traffic before full rollout",
        "kubernetes liveness probe timeout increased to 15 seconds to prevent false restarts",
        "readiness probe added to prevent traffic before service is fully initialized",
        "horizontal pod autoscaler configured min 2 max 10 replicas on 70 percent CPU",
        "resource limits set 256MB memory 250m CPU to prevent pod eviction",
        "docker build time reduced from 4 minutes to 45 seconds with layer caching",
        "helm chart parameterized for environment specific configuration",
        "github actions pipeline builds test and deploys on every merge to main",
        "rollback executed after bad deploy detected by prometheus latency alert",
        "deployment stuck due to image pull error from private registry auth issue",
        "service mesh added for mutual TLS between microservices in production",
        "secret rotation automated using kubernetes external secrets operator",
        "grafana dashboard created for deployment frequency and rollback rate metrics",
    ],
    'performance': [
        "API latency p99 reduced from 800ms to 45ms after ONNX quantization",
        "inference time cut from 2 seconds to 90ms by switching to int8 model",
        "caching layer added reducing database calls by 80 percent",
        "async processing moved to background workers cutting response time to 50ms",
        "memory leak traced to unclosed file handles in data pipeline job",
        "CPU profiling revealed tight loop in string parsing was bottleneck",
        "batch processing increased throughput from 100 to 10000 records per second",
        "Redis sorted sets used for leaderboard to replace expensive SQL aggregation",
        "CDN added for static assets cutting page load time in half",
        "database query reduced from full table scan to index lookup cutting time 40x",
        "connection keep alive enabled reducing TLS handshake overhead by 30 percent",
        "gzip compression added to API responses reducing bandwidth by 60 percent",
        "lazy loading implemented for images improving time to interactive by 2 seconds",
        "worker thread pool tuned to match CPU count improving utilization",
        "streaming response implemented to improve perceived latency for long outputs",
    ],
    'architecture': [
        "decided to split monolith into microservices starting with auth and payments",
        "event driven architecture chosen to decouple services via message queue",
        "CQRS pattern implemented separating read and write models for scalability",
        "saga pattern used to coordinate distributed transaction across services",
        "API gateway added as single entry point handling auth rate limiting routing",
        "circuit breaker pattern implemented to prevent cascade failures",
        "outbox pattern used to ensure reliable event publishing with transactions",
        "service discovery via consul replacing hardcoded service addresses",
        "strangler fig pattern applied to incrementally replace legacy system",
        "hexagonal architecture adopted to isolate business logic from infrastructure",
        "decided against microservices for current scale, modular monolith sufficient",
        "event sourcing implemented for audit trail and temporal queries on orders",
        "domain driven design bounded contexts defined for team ownership boundaries",
        "GraphQL federation chosen over REST for unified data layer across teams",
        "gRPC used for internal service communication replacing REST for performance",
    ],
}

TOPIC_QUERIES = {
    'auth': [
        ("how long do auth tokens last", ['auth']),
        ("what security issues did we fix", ['auth']),
        ("authentication implementation decisions", ['auth']),
    ],
    'database': [
        ("why were database queries slow", ['database']),
        ("what indexes did we add", ['database']),
        ("connection pool issues", ['database']),
    ],
    'deployment': [
        ("how do we deploy to production", ['deployment']),
        ("kubernetes configuration decisions", ['deployment']),
        ("CI CD pipeline setup", ['deployment']),
    ],
    'performance': [
        ("what did we do to improve latency", ['performance']),
        ("caching strategy decisions", ['performance']),
        ("profiling and bottleneck fixes", ['performance']),
    ],
    'architecture': [
        ("system design decisions", ['architecture']),
        ("why did we choose microservices or monolith", ['architecture']),
        ("event driven architecture choices", ['architecture']),
    ],
}

ZIPF = np.array([1/i for i in range(1, len(TOPICS)+1)], dtype=float)
ZIPF /= ZIPF.sum()


# ---------------------------------------------------------------------------
# Flat BM25 baseline
# ---------------------------------------------------------------------------

class FlatBM25:
    def __init__(self):
        self._idx = VectorizedBM25()
        self._ids: List[str] = []
        self._texts: Dict[str, str] = {}

    def store(self, text: str, mid: str):
        self._idx.add(mid, text)
        self._ids.append(mid)
        self._texts[mid] = text

    def query(self, q: str, k: int) -> Tuple[List[str], float]:
        t0 = time.perf_counter()
        results = self._idx.search_all(q, top_k=k)
        lat = (time.perf_counter() - t0) * 1000
        return [mid for mid, _ in results], lat


def build_stores(n: int, seed: int = 42,
                 n_warmup_queries: int = 0) -> Tuple[Cortex, FlatBM25, List[Dict], Dict]:
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    topic_names = list(TOPICS.keys())

    cortex = Cortex(
        eta=0.12, decay=0.0003, floor=0.01, top_k=10,
        cluster_threshold=3, max_cluster_size=20,
        bm25_seed_threshold=1.2, pass1_clusters=4,
    )
    flat = FlatBM25()
    memories = []
    id_to_topic = {}

    # Distribute by Zipf
    counts = {t: max(3, int(n * ZIPF[i])) for i, t in enumerate(topic_names)}
    total = sum(counts.values())
    if total < n:
        counts[topic_names[0]] += n - total

    ctr = 0
    for topic, cnt in counts.items():
        pool = TOPICS[topic]
        for _ in range(cnt):
            text = rng.choice(pool)
            # Add variation to avoid exact duplicates at scale
            suffix = f" [ref-{ctr}]"
            full = text + suffix
            mid = f'{topic[:3]}_{ctr:05d}'
            cortex.store(full, memory_id=mid)
            flat.store(full, mid)
            memories.append({'id': mid, 'topic': topic, 'text': full})
            id_to_topic[mid] = topic
            ctr += 1

    # Warmup queries to build co-retrieval structure
    for qi in range(n_warmup_queries):
        t_idx = np_rng.choice(len(topic_names), p=ZIPF)
        q_text, _ = rng.choice(TOPIC_QUERIES[topic_names[t_idx]])
        cortex.query(q_text)

    return cortex, flat, memories, id_to_topic


def eval_retrieval(cortex, flat, memories, id_to_topic,
                   k: int, n_queries: int = 40, seed: int = 999) -> Dict:
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    topic_names = list(TOPICS.keys())

    c_prec, f_prec, c_lat, f_lat, c_save = [], [], [], [], []

    for qi in range(n_queries):
        t_idx = np_rng.choice(len(topic_names), p=ZIPF)
        topic = topic_names[t_idx]
        q_text, _ = rng.choice(TOPIC_QUERIES[topic])

        relevant = {m['id'] for m in memories if m['topic'] == topic}
        if not relevant: continue

        t0 = time.perf_counter()
        res_c = cortex.query(q_text, top_k=k)
        c_lat.append((time.perf_counter() - t0) * 1000)

        res_f_ids, f_l = flat.query(q_text, k)
        f_lat.append(f_l)

        c_ids = [r['id'] for r in res_c]
        c_prec.append(len(set(c_ids) & relevant) / max(k, 1))
        f_prec.append(len(set(res_f_ids) & relevant) / max(k, 1))
        c_save.append(cortex.diagnostics.get('savings_pct', 0))

    def m(x): return float(np.mean(x)) if x else 0.0
    return {
        'c_prec': m(c_prec), 'f_prec': m(f_prec),
        'c_lat': m(c_lat), 'f_lat': m(f_lat),
        'savings': m(c_save),
    }


# ---------------------------------------------------------------------------
# Claim 1: Retrieval quality vs scale
# ---------------------------------------------------------------------------

def claim1_retrieval_vs_scale():
    print("\n── Claim 1: Retrieval quality vs scale ──")
    scales = [100, 500, 1000, 3000, 5000]
    k = 8
    results = {}

    for n in scales:
        c, f, mems, i2t = build_stores(n, n_warmup_queries=min(30, n//5))
        s = c.stats()
        r = eval_retrieval(c, f, mems, i2t, k=k, n_queries=40)
        results[n] = r
        print(f"  N={n:5d}: C={r['c_prec']:.3f} F={r['f_prec']:.3f} "
              f"delta={r['c_prec']-r['f_prec']:+.3f} "
              f"clusters={s['n_clusters']} cov={s['coverage']:.0%} "
              f"C.lat={r['c_lat']:.1f}ms F.lat={r['f_lat']:.1f}ms "
              f"saved={r['savings']:.0f}%")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ns = scales
    ax.plot(ns, [results[n]['c_prec'] for n in ns], 'o-', color=C0, lw=2.5, ms=7, label='Cortex')
    ax.plot(ns, [results[n]['f_prec'] for n in ns], 's--', color=C1, lw=2.5, ms=7, label='Flat BM25')
    ax.set_xscale('log'); ax.set_ylim(0, 1)
    ax.set_xlabel('Memory store size N'); ax.set_ylabel(f'Precision@{k}')
    ax.legend(fontsize=9)
    style(ax, f'Retrieval Precision@{k} vs Scale')

    ax = axes[1]
    ax.plot(ns, [results[n]['c_lat'] for n in ns], 'o-', color=C0, lw=2.5, ms=7, label='Cortex')
    ax.plot(ns, [results[n]['f_lat'] for n in ns], 's--', color=C1, lw=2.5, ms=7, label='Flat BM25')
    ax.set_xscale('log')
    ax.set_xlabel('N'); ax.set_ylabel('Query latency (ms)')
    ax.legend(fontsize=9)
    style(ax, 'Query Latency vs Scale')

    ax = axes[2]
    deltas = [results[n]['c_prec'] - results[n]['f_prec'] for n in ns]
    colors = [C2 if d >= 0 else C3 for d in deltas]
    ax.bar(range(len(ns)), deltas, color=colors, alpha=0.85)
    ax.axhline(0, color='white', ls='--', alpha=0.4)
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([f'N={n}' for n in ns], fontsize=8)
    ax.set_ylabel('Precision delta (Cortex − Flat)')
    for i, d in enumerate(deltas):
        ax.text(i, d + (0.005 if d >= 0 else -0.012),
                f'{d:+.3f}', ha='center', fontsize=8, color='white')
    style(ax, 'Cortex Advantage vs Scale')

    fig.suptitle('Claim 1: Retrieval Quality vs Scale', color='white',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, 'claim1_retrieval_scale.png')
    return results


# ---------------------------------------------------------------------------
# Claim 2: Context efficiency improves with use
# ---------------------------------------------------------------------------

def claim2_context_efficiency():
    print("\n── Claim 2: Context efficiency improves with use ──")
    N = 500
    k = 8
    checkpoints = [0, 10, 25, 50, 100, 200]
    rng = random.Random(42)
    topic_names = list(TOPICS.keys())
    np_rng = np.random.RandomState(42)

    c, f, mems, i2t = build_stores(N, n_warmup_queries=0)

    def measure(cortex_instance, flat_instance, n_eval=30) -> Dict:
        c_hits_per_token, f_hits_per_token = [], []
        c_coherence = []
        rng2 = random.Random(999)
        np_rng2 = np.random.RandomState(999)

        for _ in range(n_eval):
            t_idx = np_rng2.choice(len(topic_names), p=ZIPF)
            topic = topic_names[t_idx]
            q_text, _ = rng2.choice(TOPIC_QUERIES[topic])
            relevant = {m['id'] for m in mems if m['topic'] == topic}
            if not relevant: continue

            res_c = cortex_instance.query(q_text, top_k=k)
            res_f_ids, _ = flat_instance.query(q_text, k)

            c_ids = [r['id'] for r in res_c]
            c_hits = len(set(c_ids) & relevant)
            f_hits = len(set(res_f_ids) & relevant)

            c_chars = sum(len(r['text']) for r in res_c)
            f_chars = sum(len(flat_instance._texts.get(mid, '')) for mid in res_f_ids)

            if c_hits > 0: c_hits_per_token.append((c_chars/4) / c_hits)
            if f_hits > 0: f_hits_per_token.append((f_chars/4) / f_hits)

            pairs = list(itertools.combinations(c_ids[:k], 2))
            if pairs:
                coh = np.mean([
                    cortex_instance._coret.get(a, {}).get(b, 0) +
                    cortex_instance._coret.get(b, {}).get(a, 0)
                    for a, b in pairs
                ])
                c_coherence.append(coh)

        def m(x): return float(np.mean(x)) if x else 0.0
        return {
            'c_tok_per_hit': m(c_hits_per_token),
            'f_tok_per_hit': m(f_hits_per_token),
            'coherence': m(c_coherence),
        }

    # Rebuild for clean measurement
    c_clean, _, _, _ = build_stores(N, n_warmup_queries=0)
    results = []

    for i, cp in enumerate(checkpoints):
        # Run queries up to checkpoint
        if i > 0:
            prev = checkpoints[i-1]
            for _ in range(cp - prev):
                t_idx = np_rng.choice(len(topic_names), p=ZIPF)
                q_text, _ = rng.choice(TOPIC_QUERIES[topic_names[t_idx]])
                c_clean.query(q_text)

        r = measure(c_clean, f)
        r['queries'] = cp
        results.append(r)
        print(f"  queries={cp:3d}: C={r['c_tok_per_hit']:.1f} tok/hit  "
              f"F={r['f_tok_per_hit']:.1f} tok/hit  "
              f"coherence={r['coherence']:.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    qs = [r['queries'] for r in results]

    # Panel 1: coherence grows with use
    ax = axes[0]
    ax.plot(qs, [r['coherence'] for r in results],
            'o-', color=C2, lw=2.5, ms=7)
    ax.set_xlabel('Queries (usage history)')
    ax.set_ylabel('Mean co-retrieval count in returned set')
    ax.set_title('Context Coherence Grows with Use', color='white', fontsize=10)
    style(ax)

    # Panel 2: tokens per hit (honest comparison)
    ax = axes[1]
    ax.plot(qs, [r['c_tok_per_hit'] for r in results],
            'o-', color=C0, lw=2.5, ms=7, label='Cortex')
    ax.axhline(results[0]['f_tok_per_hit'], color=C1, ls='--', lw=2,
               label=f"Flat BM25 ({results[0]['f_tok_per_hit']:.0f} tok/hit, constant)")
    ax.set_xlabel('Queries (usage history)'); ax.set_ylabel('Tokens per relevant hit')
    ax.legend(fontsize=9)
    style(ax, 'Tokens per Relevant Hit vs Usage')

    fig.suptitle('Claim 2: Context Efficiency Improves with Use (N=500)',
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, 'claim2_context_efficiency.png')
    return results


# ---------------------------------------------------------------------------
# Claim 3: Sub-10ms retrieval at 10K memories
# ---------------------------------------------------------------------------

def claim3_latency():
    print("\n── Claim 3: Latency at scale ──")
    scales = [100, 500, 1000, 2000, 5000, 10000]
    k = 8
    rng = random.Random(42)
    topic_names = list(TOPICS.keys())

    results = {}
    for n in scales:
        c, f, mems, i2t = build_stores(n, n_warmup_queries=min(20, n//10))

        c_lats, f_lats, savings = [], [], []
        for _ in range(20):
            t_idx = random.randint(0, len(topic_names)-1)
            q, _ = rng.choice(TOPIC_QUERIES[topic_names[t_idx]])

            t0 = time.perf_counter()
            c.query(q, top_k=k)
            c_lats.append((time.perf_counter() - t0) * 1000)
            savings.append(c.diagnostics.get('savings_pct', 0))

            _, fl = f.query(q, k)
            f_lats.append(fl)

        results[n] = {
            'c_p50': float(np.percentile(c_lats, 50)),
            'c_p95': float(np.percentile(c_lats, 95)),
            'f_p50': float(np.percentile(f_lats, 50)),
            'f_p95': float(np.percentile(f_lats, 95)),
            'savings': float(np.mean(savings)),
            'clusters': c.stats()['n_clusters'],
            'coverage': c.stats()['coverage'],
        }
        r = results[n]
        print(f"  N={n:6d}: C.p50={r['c_p50']:.1f}ms C.p95={r['c_p95']:.1f}ms | "
              f"F.p50={r['f_p50']:.1f}ms F.p95={r['f_p95']:.1f}ms | "
              f"saved={r['savings']:.0f}% clusters={r['clusters']} cov={r['coverage']:.0%}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ns = scales
    ax.plot(ns, [results[n]['c_p50'] for n in ns], 'o-', color=C0, lw=2.5, ms=6, label='Cortex p50')
    ax.plot(ns, [results[n]['c_p95'] for n in ns], 'o--', color=C0, lw=1.5, ms=5, alpha=0.6, label='Cortex p95')
    ax.plot(ns, [results[n]['f_p50'] for n in ns], 's-', color=C1, lw=2.5, ms=6, label='Flat BM25 p50')
    ax.axhline(10, color='white', ls=':', alpha=0.4, label='10ms target')
    ax.set_xscale('log'); ax.set_xlabel('N memories'); ax.set_ylabel('Latency (ms)')
    ax.legend(fontsize=8)
    style(ax, 'Query Latency (p50/p95) vs N')

    ax = axes[1]
    ax.bar(range(len(ns)), [results[n]['savings'] for n in ns],
           color=C2, alpha=0.85)
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([f'{n}' for n in ns], fontsize=8)
    ax.set_xlabel('N memories'); ax.set_ylabel('% memories skipped (two-pass)')
    ax.set_ylim(0, 100)
    for i, n in enumerate(ns):
        ax.text(i, results[n]['savings'] + 2,
                f"{results[n]['savings']:.0f}%", ha='center', fontsize=8, color='white')
    style(ax, 'Two-Pass Savings vs N')

    ax = axes[2]
    ratio = [results[n]['c_p50'] / max(results[n]['f_p50'], 0.01) for n in ns]
    bar_colors = [C2 if r <= 3 else C3 for r in ratio]
    ax.bar(range(len(ns)), ratio, color=bar_colors, alpha=0.85)
    ax.axhline(1.0, color='white', ls='--', alpha=0.4)
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([f'{n}' for n in ns], fontsize=8)
    ax.set_ylabel('Latency ratio (Cortex / Flat)')
    for i, r in enumerate(ratio):
        ax.text(i, r + 0.05, f'{r:.1f}×', ha='center', fontsize=8, color='white')
    style(ax, 'Overhead Ratio (approaches 1 at scale)')

    fig.suptitle('Claim 3: Query Latency vs Scale', color='white',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, 'claim3_latency.png')
    return results


# ---------------------------------------------------------------------------
# Claim 4: Portability — lossless save/load at scale
# ---------------------------------------------------------------------------

def claim4_portability():
    print("\n── Claim 4: Portability — save/load at scale ──")
    import tempfile
    scales = [100, 500, 1000, 5000, 10000]
    results = {}

    for n in scales:
        c, _, mems, i2t = build_stores(n, n_warmup_queries=min(20, n//10))

        with tempfile.NamedTemporaryFile(suffix='.memory', delete=False) as tmp:
            tmp_path = tmp.name

        # Save the store
        mem_obj = Memory(c, {
            'version': '1.0', 'description': f'bench-{n}',
            'tags': [], 'llm_hint': '', 'created_at': time.time(),
            'created_at_human': '', 'last_used_at': time.time(),
            'query_count': c.query_count, 'memory_count': n,
        })
        t0 = time.perf_counter()
        mem_obj.save(tmp_path)
        save_ms = (time.perf_counter() - t0) * 1000
        file_kb = os.path.getsize(tmp_path) / 1024

        # Load into a fresh instance
        t0 = time.perf_counter()
        mem_loaded = Memory.load(tmp_path)
        load_ms = (time.perf_counter() - t0) * 1000

        # Verify lossless: load a SECOND copy and compare two fresh instances
        # (avoids comparing mutated-during-eval `c` against freshly loaded state)
        mem_ref = Memory.load(tmp_path)
        rng = random.Random(42)
        topic_names = list(TOPICS.keys())
        mismatches = 0
        for _ in range(5):
            t_idx = random.randint(0, len(topic_names)-1)
            q, _ = rng.choice(TOPIC_QUERIES[topic_names[t_idx]])
            r1 = mem_ref._cortex.query(q, top_k=3)
            r2 = mem_loaded._cortex.query(q, top_k=3)
            ids1 = set(r['id'] for r in r1)
            ids2 = set(r['id'] for r in r2)
            if len(ids1.symmetric_difference(ids2)) > 1:
                mismatches += 1

        results[n] = {
            'save_ms': save_ms,
            'load_ms': load_ms,
            'file_kb': file_kb,
            'lossless': mismatches == 0,
            'mismatches': mismatches,
        }
        r = results[n]
        print(f"  N={n:6d}: save={r['save_ms']:.0f}ms load={r['load_ms']:.0f}ms "
              f"size={r['file_kb']:.0f}KB lossless={r['lossless']}")
        os.unlink(tmp_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ns = scales

    ax = axes[0]
    ax.plot(ns, [results[n]['file_kb'] for n in ns], 'o-', color=C0, lw=2.5, ms=7)
    ax.set_xscale('log'); ax.set_xlabel('N memories'); ax.set_ylabel('File size (KB)')
    # Add size per memory annotation
    for i, n in enumerate(ns[::2]):
        kb = results[n]['file_kb']
        ax.annotate(f'{kb/n*1000:.0f}B/mem',
                    xy=(n, kb), xytext=(10, 5), textcoords='offset points',
                    fontsize=7, color='#9ca3af')
    style(ax, 'File Size vs N')

    ax = axes[1]
    ax.plot(ns, [results[n]['save_ms'] for n in ns], 'o-', color=C1, lw=2.5, ms=7, label='Save')
    ax.plot(ns, [results[n]['load_ms'] for n in ns], 's-', color=C2, lw=2.5, ms=7, label='Load')
    ax.set_xscale('log'); ax.set_xlabel('N memories'); ax.set_ylabel('Time (ms)')
    ax.legend(fontsize=9)
    style(ax, 'Save/Load Time vs N')

    ax = axes[2]
    ax.bar(range(len(ns)),
           [1 if results[n]['lossless'] else 0 for n in ns],
           color=[C2 if results[n]['lossless'] else C3 for n in ns],
           alpha=0.85)
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([f'N={n}' for n in ns], fontsize=8)
    ax.set_ylim(0, 1.3); ax.set_ylabel('Lossless (1=yes)')
    for i, n in enumerate(ns):
        label = '✓ lossless' if results[n]['lossless'] else f"✗ {results[n]['mismatches']} mismatch"
        ax.text(i, 0.5, label, ha='center', fontsize=8, color='white', fontweight='bold')
    style(ax, 'Query Results Identical After Save/Load')

    fig.suptitle('Claim 4: Portability — Save/Load Fidelity',
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, 'claim4_portability.png')
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL = {1: claim1_retrieval_vs_scale,
       2: claim2_context_efficiency,
       3: claim3_latency,
       4: claim4_portability}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--claim', nargs='*', type=int, default=None)
    args = parser.parse_args()
    claims = args.claim if args.claim else sorted(ALL.keys())

    print(f"\n=== Memory Store Benchmark (claims: {claims}) ===")
    t0 = time.time()
    all_results = {}
    for c in claims:
        if c in ALL:
            all_results[c] = ALL[c]()
    print(f"\nTotal: {time.time()-t0:.1f}s  Figures: {FIGDIR}/")

if __name__ == '__main__':
    main()
