"""
harness.py — Transparent memory layer for any LLM conversation.

Wraps an LLM call with automatic memory retrieval (pre-turn) and
storage (post-turn). The LLM never knows memory exists — it just
receives richer context and its responses are stored automatically.

Works with:
  - Anthropic API (Claude)
  - OpenAI API (drop-in compatible interface)
  - Any callable that takes (messages, system) and returns a string
  - Claude Code via CLAUDE.md injection

Usage:
    from harness import MemoryHarness
    harness = MemoryHarness("project.memory")
    response = harness.chat("why is the dashboard query slow?")
"""

import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Dict, Optional, Union

from cortex_memory.memory import Memory


# ---------------------------------------------------------------------------
# Base harness — works with any LLM callable
# ---------------------------------------------------------------------------

class MemoryHarness:
    """
    Transparent memory layer. Wraps any LLM conversation with:
      - Pre-turn: query memory, inject top-K results into system prompt
      - Post-turn: store assistant response asynchronously (non-blocking)

    The LLM sees memory as part of its system prompt context.
    No tool calls, no round-trips, no model awareness required.
    """

    DEFAULT_SYSTEM = (
        "You are a helpful assistant. Use the context in <memory> tags "
        "to inform your responses when relevant. Do not explicitly mention "
        "that you have a memory system unless asked."
    )

    def __init__(
        self,
        memory_path: Union[str, Path],
        llm_fn: Optional[Callable] = None,
        system_prompt: str = "",
        top_k: int = 5,
        store_responses: bool = True,
        store_user_messages: bool = False,
        min_store_length: int = 40,
        async_store: bool = True,
        memory_tag: str = "memory",
        create_if_missing: bool = True,
        description: str = "",
    ):
        """
        memory_path:         Path to .memory file (created if missing).
        llm_fn:              Callable(messages, system) -> str. If None,
                             use inject_context() manually.
        system_prompt:       Base system prompt before memory injection.
        top_k:               Memories to retrieve per turn.
        store_responses:     Auto-store assistant responses.
        store_user_messages: Also store user messages (default: off).
        min_store_length:    Minimum chars before storing a response.
        async_store:         Store in background thread (non-blocking).
        memory_tag:          XML tag name for injected context block.
        create_if_missing:   Create empty .memory file if path not found.
        description:         Description for new memory file.
        """
        self.memory_path = Path(memory_path)
        self.top_k = top_k
        self.store_responses = store_responses
        self.store_user_messages = store_user_messages
        self.min_store_length = min_store_length
        self.async_store = async_store
        self.memory_tag = memory_tag
        self._base_system = system_prompt or self.DEFAULT_SYSTEM
        self._llm_fn = llm_fn
        self._conversation: List[Dict] = []
        self._pending_stores: List[threading.Thread] = []

        # Load or create memory
        if self.memory_path.exists():
            self.mem = Memory.load(str(self.memory_path))
        elif create_if_missing:
            self.mem = Memory.create(
                description=description or self.memory_path.stem,
                tags=[],
            )
            self.mem.save(str(self.memory_path))
        else:
            raise FileNotFoundError(f"Memory file not found: {self.memory_path}")

    # -----------------------------------------------------------------------
    # Core interface
    # -----------------------------------------------------------------------

    def chat(self, user_message: str, **llm_kwargs) -> str:
        """
        Send a message through the memory-augmented LLM.
        Retrieves context, calls LLM, stores response.
        Returns the assistant's response text.
        """
        if self._llm_fn is None:
            raise RuntimeError(
                "No llm_fn provided. Use inject_context() manually "
                "or provide llm_fn at init."
            )

        # Pre-turn: retrieve context
        system = self.build_system_prompt(user_message)

        # Optionally store user message
        if self.store_user_messages and len(user_message) >= self.min_store_length:
            self._store(f"[user] {user_message}")

        # Add to conversation
        self._conversation.append({"role": "user", "content": user_message})

        # Call LLM
        response_text = self._llm_fn(
            messages=list(self._conversation),
            system=system,
            **llm_kwargs,
        )

        # Add response to conversation
        self._conversation.append({"role": "assistant", "content": response_text})

        # Post-turn: store response
        if self.store_responses and len(response_text) >= self.min_store_length:
            timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M")
            self._store(f"[{timestamp}] {response_text}")

        return response_text

    def build_system_prompt(self, query: str) -> str:
        """
        Build system prompt with injected memory context.
        Call this manually if you're managing the LLM call yourself.
        """
        context = self.mem.query(query, top_k=self.top_k)
        if not context:
            return self._base_system

        memory_block = "\n".join(f"- {c}" for c in context)
        return (
            f"{self._base_system}\n\n"
            f"<{self.memory_tag}>\n"
            f"{memory_block}\n"
            f"</{self.memory_tag}>"
        )

    def store(self, text: str, memory_id: Optional[str] = None,
              metadata: Optional[str] = None):
        """Manually store a memory (synchronous)."""
        self.mem.store(text, memory_id=memory_id, metadata=metadata)
        self._autosave()

    def query(self, text: str, top_k: Optional[int] = None) -> List[str]:
        """Query memory directly."""
        return self.mem.query(text, top_k=top_k or self.top_k)

    def reset_conversation(self):
        """Clear conversation history (keep memory store intact)."""
        self._conversation = []

    def save(self, path: Optional[Union[str, Path]] = None):
        """Flush any pending async stores and save memory to disk."""
        self._flush_pending()
        self.mem.save(str(path or self.memory_path))

    # -----------------------------------------------------------------------
    # CLAUDE.md injection (for Claude Code CLI)
    # -----------------------------------------------------------------------

    def inject_claude_md(
        self,
        query: str = "",
        claude_md_path: Union[str, Path] = "CLAUDE.md",
        section_header: str = "## Active Memory Context",
        top_k: Optional[int] = None,
    ):
        """
        Prepend retrieved memory context to CLAUDE.md for Claude Code.

        Run this before starting a Claude Code session:
            harness.inject_claude_md("current project work")
            os.system("claude")

        The memory context is injected as a section in CLAUDE.md,
        which Claude Code reads at session start.
        """
        k = top_k or self.top_k
        context = self.mem.query(query or "current project context", top_k=k)

        if not context:
            return

        claude_md = Path(claude_md_path)
        existing = claude_md.read_text() if claude_md.exists() else ""

        # Remove previous memory injection if present
        if section_header in existing:
            start = existing.index(section_header)
            end_marker = "\n## "
            end = existing.find(end_marker, start + len(section_header))
            if end == -1:
                existing = existing[:start].rstrip()
            else:
                existing = existing[:start].rstrip() + "\n" + existing[end:]

        # Build memory section
        lines = [section_header, ""]
        lines.append(
            f"*{len(context)} memories retrieved from "
            f"`{self.memory_path.name}` — {self.mem.memory_count} total*"
        )
        lines.append("")
        for c in context:
            lines.append(f"- {c[:120]}")
        lines.append("")

        new_content = "\n".join(lines) + "\n" + existing.lstrip()
        claude_md.write_text(new_content)

    def sync_from_transcript(
        self,
        transcript_path: Union[str, Path],
        role_filter: str = "assistant",
        min_length: int = 60,
    ):
        """
        Store messages from a conversation transcript file.
        Supports JSONL format (one JSON object per line with 'role'/'content').
        Used to ingest Claude Code session transcripts post-session.
        """
        import json
        stored = 0
        with open(transcript_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        # Handle Anthropic content blocks
                        content = " ".join(
                            b.get("text", "") for b in content
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    if role == role_filter and len(content) >= min_length:
                        self.mem.store(content[:500])
                        stored += 1
                except (json.JSONDecodeError, KeyError):
                    continue
        if stored:
            self._autosave()
        return stored

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _store(self, text: str):
        if self.async_store:
            t = threading.Thread(
                target=self._store_and_save, args=(text,), daemon=True
            )
            t.start()
            self._pending_stores.append(t)
            # Clean up completed threads
            self._pending_stores = [t for t in self._pending_stores if t.is_alive()]
        else:
            self._store_and_save(text)

    def _store_and_save(self, text: str):
        self.mem.store(text)
        self._autosave()

    def _autosave(self):
        self.mem.save(str(self.memory_path))

    def _flush_pending(self):
        for t in self._pending_stores:
            t.join(timeout=5.0)
        self._pending_stores = []

    def __repr__(self) -> str:
        return (
            f"MemoryHarness('{self.memory_path.name}' | "
            f"{self.mem.memory_count} memories, "
            f"{self.mem.query_count} queries)"
        )


# ---------------------------------------------------------------------------
# Claude-specific harness (Anthropic SDK)
# ---------------------------------------------------------------------------

class ClaudeMemoryHarness(MemoryHarness):
    """
    Memory harness pre-wired for the Anthropic Claude API.

    Usage:
        harness = ClaudeMemoryHarness("project.memory")
        response = harness.chat("what did we decide about auth?")
    """

    DEFAULT_MODEL = "claude-sonnet-4-5"

    def __init__(
        self,
        memory_path: Union[str, Path],
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        self.model = model
        self.max_tokens = max_tokens

        # Lazy import — don't require anthropic unless this class is used
        try:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
            )
        except ImportError:
            raise ImportError(
                "anthropic package required: pip install anthropic"
            )

        super().__init__(
            memory_path=memory_path,
            llm_fn=self._call_claude,
            **kwargs,
        )

    def _call_claude(self, messages: List[Dict], system: str, **kwargs) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            system=system,
            messages=messages,
        )
        return response.content[0].text


# ---------------------------------------------------------------------------
# OpenAI-compatible harness
# ---------------------------------------------------------------------------

class OpenAIMemoryHarness(MemoryHarness):
    """
    Memory harness pre-wired for OpenAI or any OpenAI-compatible API
    (Together, Fireworks, local Ollama with openai compat, etc.)
    """

    DEFAULT_MODEL = "gpt-4o"

    def __init__(
        self,
        memory_path: Union[str, Path],
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        self.model = model
        self.max_tokens = max_tokens

        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=base_url,
            )
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        super().__init__(
            memory_path=memory_path,
            llm_fn=self._call_openai,
            **kwargs,
        )

    def _call_openai(self, messages: List[Dict], system: str, **kwargs) -> str:
        all_messages = [{"role": "system", "content": system}] + messages
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            messages=all_messages,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# CLI utility functions
# ---------------------------------------------------------------------------

def inject_memory_to_claude_md(
    memory_path: str,
    query: str = "",
    claude_md: str = "CLAUDE.md",
    top_k: int = 5,
):
    """
    Standalone function to inject memory into CLAUDE.md.
    Use as a pre-session hook for Claude Code:

        python -c "from harness import inject_memory_to_claude_md; \
                   inject_memory_to_claude_md('project.memory')"
    """
    h = MemoryHarness(memory_path, create_if_missing=False)
    h.inject_claude_md(query=query, claude_md_path=claude_md, top_k=top_k)
    print(f"Injected {top_k} memories into {claude_md}")
    print(f"Store: {h.mem.memory_count} memories, {h.mem.query_count} queries")


def store_session(memory_path: str, transcript_path: str):
    """
    Store a Claude Code session transcript into memory.
    Use as a post-session hook:

        python -c "from harness import store_session; \
                   store_session('project.memory', '.claude/transcript.jsonl')"
    """
    h = MemoryHarness(memory_path)
    n = h.sync_from_transcript(transcript_path)
    h.save()
    print(f"Stored {n} messages from {transcript_path}")
