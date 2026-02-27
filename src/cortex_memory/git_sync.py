"""
git_sync.py — Git sync utility for portable memory files.

Keeps a .memory file in sync with a git repository so it travels
automatically across machines. Pull before reading, push after writing.
All operations are silent on failure — memory ops must never break
because git is unavailable.

Usage:
    from cortex_memory.git_sync import GitSync

    gs = GitSync(repo_path="~/memory")
    gs.pull()                          # before loading memory
    # ... use memory ...
    gs.push("global.memory")           # after saving memory (sync to remote)

Or async (fire-and-forget):
    gs.push_async("global.memory")     # returns immediately
"""

import os
import subprocess
import threading
import sys
from pathlib import Path
from typing import Optional


class GitSync:
    """
    Lightweight git wrapper for syncing a .memory file.

    Designed to be silent — every operation catches exceptions and
    logs to stderr rather than raising. Memory ops always continue.
    """

    def __init__(
        self,
        repo_path: str,
        remote: str = "origin",
        branch: str = "main",
        commit_message: str = "cortex: session sync",
    ):
        self.repo_path = str(Path(repo_path).expanduser().resolve())
        self.remote = remote
        self.branch = branch
        self.commit_message = commit_message
        self._push_thread: Optional[threading.Thread] = None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def pull(self) -> bool:
        """
        Pull latest changes from remote. Call before loading memory.
        Blocks until complete (typically <1s on good connection).
        Returns True on success, False on any error.
        """
        if not self.is_git_repo():
            return False
        try:
            result = subprocess.run(
                ["git", "-C", self.repo_path, "pull", self.remote, self.branch],
                capture_output=True,
                timeout=15,
            )
            if result.returncode != 0:
                sys.stderr.write(
                    f"[cortex/git_sync] pull failed: "
                    f"{result.stderr.decode().strip()}\n"
                )
                return False
            return True
        except Exception as e:
            sys.stderr.write(f"[cortex/git_sync] pull error: {e}\n")
            return False

    def push(self, memory_file: str) -> bool:
        """
        Stage, commit, and push the memory file to remote.
        Blocks until complete. Use push_async() for non-blocking.
        Returns True on success, False on any error.
        """
        if not self.is_git_repo():
            return False
        try:
            filename = os.path.basename(memory_file)

            # Stage
            r = subprocess.run(
                ["git", "-C", self.repo_path, "add", filename],
                capture_output=True, timeout=10,
            )
            if r.returncode != 0:
                sys.stderr.write(
                    f"[cortex/git_sync] add failed: {r.stderr.decode().strip()}\n"
                )
                return False

            # Commit — skip if nothing changed
            r = subprocess.run(
                ["git", "-C", self.repo_path, "commit", "-m", self.commit_message],
                capture_output=True, timeout=10,
            )
            if r.returncode != 0:
                # Non-zero from commit usually means "nothing to commit" — that's fine
                msg = r.stdout.decode().strip() + r.stderr.decode().strip()
                if "nothing to commit" in msg or "nothing added" in msg:
                    return True  # No-op, not an error
                sys.stderr.write(f"[cortex/git_sync] commit failed: {msg}\n")
                return False

            # Push
            r = subprocess.run(
                ["git", "-C", self.repo_path, "push", self.remote, self.branch],
                capture_output=True, timeout=15,
            )
            if r.returncode != 0:
                sys.stderr.write(
                    f"[cortex/git_sync] push failed: {r.stderr.decode().strip()}\n"
                )
                return False

            return True
        except Exception as e:
            sys.stderr.write(f"[cortex/git_sync] push error: {e}\n")
            return False

    def push_async(self, memory_file: str):
        """
        Push in a background thread — returns immediately.
        Ideal for hooks where you don't want to block the turn.
        """
        t = threading.Thread(
            target=self.push, args=(memory_file,), daemon=True
        )
        t.start()
        self._push_thread = t

    def is_git_repo(self) -> bool:
        """Check whether repo_path is a valid git repository."""
        try:
            r = subprocess.run(
                ["git", "-C", self.repo_path, "rev-parse", "--git-dir"],
                capture_output=True, timeout=5,
            )
            return r.returncode == 0
        except Exception:
            return False

    def status(self) -> dict:
        """Return a dict with sync status info."""
        is_repo = self.is_git_repo()
        dirty = False
        if is_repo:
            try:
                r = subprocess.run(
                    ["git", "-C", self.repo_path, "status", "--porcelain"],
                    capture_output=True, timeout=5,
                )
                dirty = bool(r.stdout.strip())
            except Exception:
                pass
        return {
            "repo_path": self.repo_path,
            "remote": self.remote,
            "branch": self.branch,
            "is_git_repo": is_repo,
            "dirty": dirty,
        }

    @classmethod
    def from_config(cls, config: dict) -> Optional["GitSync"]:
        """
        Construct a GitSync from a config dict (e.g. from config.json).
        Returns None if git_sync is disabled or misconfigured.

        Config shape:
          {
            "enabled": true,
            "repo_path": "~/memory",
            "remote": "origin",       # optional, default "origin"
            "branch": "main",         # optional, default "main"
            "commit_message": "..."   # optional
          }
        """
        if not config or not config.get("enabled"):
            return None
        repo_path = config.get("repo_path", "")
        if not repo_path:
            sys.stderr.write("[cortex/git_sync] git_sync.repo_path not set\n")
            return None
        return cls(
            repo_path=repo_path,
            remote=config.get("remote", "origin"),
            branch=config.get("branch", "main"),
            commit_message=config.get("commit_message", "cortex: session sync"),
        )

    def __repr__(self) -> str:
        return f"GitSync('{self.repo_path}' → {self.remote}/{self.branch})"
