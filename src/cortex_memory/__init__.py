"""
Cortex Memory — Portable, model-agnostic memory layer for LLM conversations.

Usage:
    from cortex_memory import Memory

    mem = Memory.create("my project")
    mem.store("decided to use JWT with 24h expiry")
    results = mem.query("authentication decisions")
    mem.save("project.memory")

    # Load anywhere
    mem = Memory.load("project.memory")
"""

__version__ = "1.0.0"

from cortex_memory.memory import Memory
from cortex_memory.harness import (
    MemoryHarness,
    ClaudeMemoryHarness,
    OpenAIMemoryHarness,
    inject_memory_to_claude_md,
    store_session,
)
from cortex_memory.cortex import Cortex

__all__ = [
    "Memory",
    "MemoryHarness",
    "ClaudeMemoryHarness",
    "OpenAIMemoryHarness",
    "Cortex",
    "inject_memory_to_claude_md",
    "store_session",
]
