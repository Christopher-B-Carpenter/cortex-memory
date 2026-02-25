"""
memory.py — Portable LLM memory artifact.

A self-contained .memory file that:
  - Works with any LLM (no embeddings, no model dependency)
  - Loads anywhere Python runs
  - Accumulates associative structure through use
  - Three-method API: store / query / save

Usage:
    from memory import Memory

    # Create
    mem = Memory.create("My project memory", tags=["python", "auth"])

    # Use
    mem.store("decided to use JWT with 24h expiry for auth tokens")
    results = mem.query("what did we decide about authentication")
    mem.save("project.memory")

    # Load anywhere
    mem = Memory.load("project.memory")
    results = mem.query("JWT token decisions")

    # Merge two memory files
    merged = Memory.merge(mem_a, mem_b, "Combined memory")
    merged.save("merged.memory")
"""

import os
import json
import time
import zipfile
import hashlib
import tempfile
import textwrap
from datetime import datetime
from typing import List, Dict, Optional, Union
from pathlib import Path

from cortex import Cortex


class Memory:
    """
    Portable LLM memory artifact.

    Wraps Cortex with:
      - Human-readable manifest (JSON)
      - Auto-generated summary of top memories
      - Merge support (union two stores)
      - Single .memory file format (zip of pkl + manifest + readme)
    """

    FORMAT_VERSION = "1.0"

    def __init__(self, cortex: Cortex, manifest: Dict):
        self._cortex = cortex
        self._manifest = manifest

    # -----------------------------------------------------------------------
    # Factory methods
    # -----------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        description: str = "",
        tags: Optional[List[str]] = None,
        llm_hint: str = "",
        **kwargs,
    ) -> "Memory":
        """Create a new empty memory store."""
        cortex = Cortex(**kwargs)
        manifest = {
            "version": cls.FORMAT_VERSION,
            "description": description,
            "tags": tags or [],
            "llm_hint": llm_hint,
            "created_at": time.time(),
            "created_at_human": datetime.now().isoformat(timespec="seconds"),
            "last_used_at": time.time(),
            "query_count": 0,
            "memory_count": 0,
        }
        return cls(cortex, manifest)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Memory":
        """
        Load a memory file. Accepts:
          - path/to/file.memory  (zip bundle)
          - path/to/file.pkl     (raw cortex pickle, auto-wraps in manifest)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Memory file not found: {path}")

        if path.suffix == ".memory":
            return cls._load_bundle(path)
        elif path.suffix in (".pkl", ".pickle"):
            return cls._load_legacy_pkl(path)
        else:
            raise ValueError(f"Unknown format: {path.suffix}. Use .memory or .pkl")

    @classmethod
    def _load_bundle(cls, path: Path) -> "Memory":
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()

            # Load manifest
            if "manifest.json" in names:
                manifest = json.loads(zf.read("manifest.json").decode())
            else:
                manifest = {"version": "unknown", "description": "", "tags": []}

            # Load cortex store
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
                tmp.write(zf.read("store.pkl"))
                tmp_path = tmp.name

        try:
            cortex = Cortex()
            cortex.load(tmp_path)
        finally:
            os.unlink(tmp_path)

        return cls(cortex, manifest)

    @classmethod
    def _load_legacy_pkl(cls, path: Path) -> "Memory":
        """Wrap a raw Cortex pickle in a Memory with auto-generated manifest."""
        cortex = Cortex()
        cortex.load(str(path))
        stats = cortex.stats()
        manifest = {
            "version": cls.FORMAT_VERSION,
            "description": f"Imported from {path.name}",
            "tags": [],
            "llm_hint": "",
            "created_at": time.time(),
            "created_at_human": datetime.now().isoformat(timespec="seconds"),
            "last_used_at": time.time(),
            "query_count": cortex.query_count,
            "memory_count": stats.get("n_memories", 0),
        }
        return cls(cortex, manifest)

    # -----------------------------------------------------------------------
    # Core API
    # -----------------------------------------------------------------------

    def store(self, text: str, memory_id: Optional[str] = None,
              metadata: Optional[str] = None) -> str:
        """
        Store a memory.

        text:        The content to remember (assistant response, decision, fact).
        memory_id:   Optional stable ID. Auto-generated if not provided.
        metadata:    Optional prefix appended to text for retrieval context.
                     E.g. "[2024-01-15 auth] " to add date/topic context.

        Returns the memory ID.
        """
        if metadata:
            full_text = f"{metadata} {text}"
        else:
            full_text = text

        mid = self._cortex.store(full_text, memory_id=memory_id)
        self._manifest["memory_count"] = len(self._cortex._memories)
        self._manifest["last_used_at"] = time.time()
        return mid

    def query(self, text: str, top_k: Optional[int] = None,
              return_scores: bool = False) -> List[Union[str, Dict]]:
        """
        Retrieve relevant memories.

        Returns a list of text strings by default.
        Set return_scores=True to get dicts with score/weight/cluster info.

        The retrieved memories are ranked by:
          - BM25 similarity to the query
          - Co-retrieval cluster coherence (memories seen together surface together)
          - Usage weight (frequently-accessed memories get a small boost)
        """
        results = self._cortex.query(text, top_k=top_k)
        self._manifest["query_count"] = self._cortex.query_count
        self._manifest["last_used_at"] = time.time()

        if return_scores:
            return results
        return [r["text"] for r in results]

    def forget(self, memory_id: str):
        """Remove a specific memory by ID."""
        self._cortex.forget(memory_id)
        self._manifest["memory_count"] = len(self._cortex._memories)

    def save(self, path: Union[str, Path]):
        """
        Save to a .memory bundle (zip of store.pkl + manifest.json + README.md).
        Creates parent directories if needed.
        """
        path = Path(path)
        if path.suffix != ".memory":
            path = path.with_suffix(".memory")
        path.parent.mkdir(parents=True, exist_ok=True)

        # Update manifest
        stats = self._cortex.stats()
        self._manifest["memory_count"] = stats.get("n_memories", 0)
        self._manifest["query_count"] = self._cortex.query_count
        self._manifest["last_used_at"] = time.time()
        self._manifest["last_used_at_human"] = \
            datetime.now().isoformat(timespec="seconds")
        self._manifest["stats"] = {
            "n_clusters": stats.get("n_clusters", 0),
            "coverage": stats.get("coverage", 0),
            "weight_gini": stats.get("weight_gini", 0),
        }

        # Write bundle
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            self._cortex.save(tmp_path)
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(tmp_path, "store.pkl")
                zf.writestr("manifest.json",
                            json.dumps(self._manifest, indent=2))
                zf.writestr("README.md", self._generate_readme())
        finally:
            os.unlink(tmp_path)

    # -----------------------------------------------------------------------
    # Merge
    # -----------------------------------------------------------------------

    @classmethod
    def merge(cls, a: "Memory", b: "Memory",
              description: str = "",
              prefer: str = "higher_weight") -> "Memory":
        """
        Merge two memory stores into one.

        prefer:
          "higher_weight"  — keep the more actively-used version of shared memories
          "a"              — prefer store A's version on conflict
          "b"              — prefer store B's version on conflict

        Merge strategy:
          Memories:   union. Conflicts resolved by prefer.
          Weights:    max(w_a, w_b) — most-used version wins
          Co-retrieval: sum counts from both stores
          Clusters:   dissolve all, rebuild from merged co-retrieval via BM25 seeding
        """
        merged_cortex = Cortex(
            eta=a._cortex.eta,
            decay=a._cortex.decay,
            floor=a._cortex.floor,
            top_k=a._cortex.top_k,
            cluster_threshold=a._cortex.cluster_threshold,
            max_cluster_size=a._cortex.max_cluster_size,
        )

        # Build merged memory set
        all_ids = set(a._cortex._memories.keys()) | set(b._cortex._memories.keys())
        q_a = a._cortex.query_count
        q_b = b._cortex.query_count

        for mid in all_ids:
            mem_a = a._cortex._memories.get(mid)
            mem_b = b._cortex._memories.get(mid)

            if mem_a and not mem_b:
                ew = mem_a.effective_weight(a._cortex.decay, q_a, a._cortex.floor)
                merged_cortex.store(mem_a.text, memory_id=mid)
                merged_cortex._memories[mid].weight = ew
                merged_cortex._memories[mid].retrieval_count = mem_a.retrieval_count

            elif mem_b and not mem_a:
                ew = mem_b.effective_weight(b._cortex.decay, q_b, b._cortex.floor)
                merged_cortex.store(mem_b.text, memory_id=mid)
                merged_cortex._memories[mid].weight = ew
                merged_cortex._memories[mid].retrieval_count = mem_b.retrieval_count

            else:
                # Both have this memory
                ew_a = mem_a.effective_weight(a._cortex.decay, q_a, a._cortex.floor)
                ew_b = mem_b.effective_weight(b._cortex.decay, q_b, b._cortex.floor)

                if prefer == "higher_weight":
                    winner = mem_a if ew_a >= ew_b else mem_b
                    ew = max(ew_a, ew_b)
                elif prefer == "a":
                    winner, ew = mem_a, ew_a
                else:
                    winner, ew = mem_b, ew_b

                merged_cortex.store(winner.text, memory_id=mid)
                merged_cortex._memories[mid].weight = ew
                merged_cortex._memories[mid].retrieval_count = \
                    max(mem_a.retrieval_count, mem_b.retrieval_count)

        # Merge co-retrieval counts (additive)
        from collections import defaultdict
        for id_a, neighbors in a._cortex._coret.items():
            for id_b, count in neighbors.items():
                if id_a in merged_cortex._memories and \
                        id_b in merged_cortex._memories:
                    merged_cortex._coret[id_a][id_b] += count
                    merged_cortex._coret[id_b][id_a] += count

        for id_a, neighbors in b._cortex._coret.items():
            for id_b, count in neighbors.items():
                if id_a in merged_cortex._memories and \
                        id_b in merged_cortex._memories:
                    merged_cortex._coret[id_a][id_b] += count
                    merged_cortex._coret[id_b][id_a] += count

        # Let clusters rebuild naturally from merged co-retrieval
        # (don't inherit either branch's cluster topology)
        merged_cortex.query_count = max(q_a, q_b)

        manifest = {
            "version": cls.FORMAT_VERSION,
            "description": description or f"Merged: {a.description} + {b.description}",
            "tags": list(set(a._manifest.get("tags", []) +
                             b._manifest.get("tags", []))),
            "llm_hint": a._manifest.get("llm_hint", ""),
            "created_at": time.time(),
            "created_at_human": datetime.now().isoformat(timespec="seconds"),
            "last_used_at": time.time(),
            "merged_from": [
                a._manifest.get("description", "store_a"),
                b._manifest.get("description", "store_b"),
            ],
            "query_count": 0,
            "memory_count": len(merged_cortex._memories),
        }
        return cls(merged_cortex, manifest)

    # -----------------------------------------------------------------------
    # Inspection
    # -----------------------------------------------------------------------

    @property
    def description(self) -> str:
        return self._manifest.get("description", "")

    @property
    def tags(self) -> List[str]:
        return self._manifest.get("tags", [])

    @property
    def memory_count(self) -> int:
        return len(self._cortex._memories)

    @property
    def query_count(self) -> int:
        return self._cortex.query_count

    def stats(self) -> Dict:
        """Full stats including manifest and Cortex internals."""
        s = self._cortex.stats()
        s.update({
            "description": self.description,
            "tags": self.tags,
            "created_at": self._manifest.get("created_at_human", ""),
            "last_used_at": self._manifest.get("last_used_at_human", ""),
            "file_size_estimate_kb": self._estimate_size_kb(),
        })
        return s

    def top_memories(self, n: int = 10) -> List[Dict]:
        """Most-used memories by effective weight."""
        q = self._cortex.query_count
        mems = sorted(
            self._cortex._memories.values(),
            key=lambda m: m.effective_weight(
                self._cortex.decay, q, self._cortex.floor
            ),
            reverse=True,
        )[:n]
        return [
            {
                "id": m.id,
                "text": m.text[:100],
                "weight": round(m.effective_weight(
                    self._cortex.decay, q, self._cortex.floor
                ), 4),
                "retrieval_count": m.retrieval_count,
                "cluster": m.cluster_id,
            }
            for m in mems
        ]

    def clusters(self, n: int = 10) -> List[Dict]:
        """Current cluster summary."""
        return self._cortex.show_clusters(n)

    def diagnostics(self) -> Dict:
        """Last query diagnostics (two-pass efficiency etc.)"""
        return self._cortex.diagnostics

    def __repr__(self) -> str:
        s = self._cortex.stats()
        return (
            f"Memory('{self.description}' | "
            f"{s.get('n_memories', 0)} memories, "
            f"{s.get('n_clusters', 0)} clusters, "
            f"{self.query_count} queries)"
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _generate_readme(self) -> str:
        """Auto-generate a human-readable summary of the memory store."""
        s = self._cortex.stats()
        top = self.top_memories(5)
        clusters = self.clusters(5)

        lines = [
            f"# Memory Store",
            f"",
            f"**{self.description}**" if self.description else "",
            f"",
            f"| Property | Value |",
            f"|----------|-------|",
            f"| Memories | {s.get('n_memories', 0)} |",
            f"| Clusters | {s.get('n_clusters', 0)} |",
            f"| Coverage | {s.get('coverage', 0):.0%} |",
            f"| Queries  | {self.query_count} |",
            f"| Created  | {self._manifest.get('created_at_human', '')} |",
            f"| Tags     | {', '.join(self.tags) or 'none'} |",
            f"",
            f"## Most Active Memories",
            f"",
        ]
        for m in top:
            text = textwrap.shorten(m["text"], width=80, placeholder="...")
            lines.append(
                f"- `{m['id']}` (retrieved {m['retrieval_count']}×): {text}"
            )

        if clusters:
            lines += ["", "## Clusters", ""]
            for cl in clusters:
                lines.append(
                    f"- **{cl['id']}** ({cl['size']} members): "
                    f"{textwrap.shorten(cl['rep'], 60, placeholder='...')}"
                )

        lines += [
            "",
            "---",
            "*Generated by [memory.py](https://github.com/sambasafety/sfdc)*",
            f"*Format version: {self.FORMAT_VERSION}*",
        ]
        return "\n".join(line for line in lines if line is not None)

    def _estimate_size_kb(self) -> float:
        """Rough size estimate without writing to disk."""
        n = len(self._cortex._memories)
        avg_text_bytes = 150
        bytes_per_mem = avg_text_bytes + 100  # text + metadata fields
        coret_bytes = sum(
            len(v) * 16
            for v in self._cortex._coret.values()
        )
        return round((n * bytes_per_mem + coret_bytes) / 1024, 1)
