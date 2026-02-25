#!/usr/bin/env python3
"""
examples/git_merge_driver.py — Git merge driver for .memory files

Registers with git to automatically merge .memory files using
Cortex's built-in Memory.merge() instead of failing on binary conflict.

─── Setup (run once per machine) ──────────────────────────────────────────────

  python examples/git_merge_driver.py --install

This adds the merge driver to your global ~/.gitconfig and creates a
.gitattributes entry in the current repo.

─── What it does ───────────────────────────────────────────────────────────────

When two branches have diverged memory files, git normally marks the file
as a binary conflict and requires manual resolution. With this driver:

  1. Git calls this script with three file paths: %O (base), %A (ours), %B (theirs)
  2. The script loads all three .memory files
  3. Merges ours ← merge(base→ours, base→theirs) using Memory.merge()
  4. Writes the result to %A (the working tree file)
  5. Exits 0 (success) — git treats the conflict as resolved

Merge semantics (via Memory.merge()):
  - Memories: union of both sides
  - Weights:  max-pooled (more-used memory wins)
  - Co-retrieval counts: summed across both sides
  - Clusters: rebuilt from merged co-retrieval structure

─── Manual usage ───────────────────────────────────────────────────────────────

  # Merge theirs into ours
  python examples/git_merge_driver.py base.memory ours.memory theirs.memory

  # Merge two .memory files directly
  python examples/git_merge_driver.py merge a.memory b.memory --output merged.memory
"""

import sys
import os
import argparse
import subprocess
import tempfile
import shutil

# Resolve library path relative to this file
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
LIBRARY_DIR = os.path.dirname(SCRIPT_DIR)   # repo root
sys.path.insert(0, LIBRARY_DIR)


def run_git_merge(base_path: str, ours_path: str, theirs_path: str) -> int:
    """
    Called by git with three temp file paths. Merges into ours_path.
    Returns 0 on success, 1 on unresolvable conflict.
    """
    from memory import Memory

    # Load all three versions — handle missing base (new file on one side)
    try:
        ours = Memory.load(ours_path) if os.path.exists(ours_path) else None
        theirs = Memory.load(theirs_path) if os.path.exists(theirs_path) else None
        base = Memory.load(base_path) if os.path.exists(base_path) and \
               os.path.getsize(base_path) > 0 else None
    except Exception as e:
        sys.stderr.write(f"[cortex-merge] Failed to load memory files: {e}\n")
        return 1

    # Edge cases
    if ours is None and theirs is None:
        return 0
    if ours is None:
        shutil.copy2(theirs_path, ours_path)
        return 0
    if theirs is None:
        return 0   # ours already in place

    # Merge
    try:
        merged = Memory.merge(
            ours, theirs,
            description=ours.description or theirs.description,
        )
        merged.save(ours_path)

        n_ours   = ours.memory_count
        n_theirs = theirs.memory_count
        n_merged = merged.memory_count
        sys.stderr.write(
            f"[cortex-merge] Merged: {n_ours} (ours) + {n_theirs} (theirs) "
            f"→ {n_merged} memories\n"
        )
        return 0

    except Exception as e:
        sys.stderr.write(f"[cortex-merge] Merge failed: {e}\n")
        return 1


def install(repo_path: str = "."):
    """
    Configure git to use this script as the merge driver for *.memory files.

    Adds to ~/.gitconfig (global):
      [merge "cortex-memory"]
        name  = Cortex memory merge driver
        driver = python /abs/path/to/git_merge_driver.py %O %A %B

    Adds to .gitattributes in repo_path:
      *.memory merge=cortex-memory
    """
    driver_path = os.path.abspath(__file__)
    python_bin  = sys.executable

    driver_cmd = f"{python_bin} {driver_path} %O %A %B"

    # Write global gitconfig entry
    subprocess.run([
        "git", "config", "--global",
        "merge.cortex-memory.name",
        "Cortex memory merge driver"
    ], check=True)

    subprocess.run([
        "git", "config", "--global",
        "merge.cortex-memory.driver",
        driver_cmd
    ], check=True)

    print(f"[cortex-merge] Registered merge driver in ~/.gitconfig")
    print(f"               driver = {driver_cmd}")

    # Write .gitattributes
    gitattributes = os.path.join(repo_path, ".gitattributes")
    entry = "*.memory merge=cortex-memory\n"

    existing = ""
    if os.path.exists(gitattributes):
        existing = open(gitattributes).read()

    if "merge=cortex-memory" in existing:
        print(f"[cortex-merge] .gitattributes already has *.memory entry — skipped")
    else:
        with open(gitattributes, "a") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            f.write(entry)
        print(f"[cortex-merge] Added to {gitattributes}: {entry.strip()}")

    print()
    print("Done. Commit .gitattributes, then share this command with teammates:")
    print()
    print(f"  python {os.path.relpath(driver_path)} --install")


def direct_merge(a_path: str, b_path: str, output_path: str):
    """Merge two .memory files directly (not via git)."""
    from memory import Memory

    a = Memory.load(a_path)
    b = Memory.load(b_path)
    merged = Memory.merge(a, b)
    merged.save(output_path)
    print(f"Merged: {a.memory_count} + {b.memory_count} → {merged.memory_count} memories")
    print(f"Saved:  {output_path}")


def main():
    # When called by git: python driver.py %O %A %B (three positional args, no flags)
    # When called directly: python driver.py [--install | merge a b --output c]
    if len(sys.argv) == 4 and not sys.argv[1].startswith("-"):
        # Git merge driver mode: base ours theirs
        sys.exit(run_git_merge(sys.argv[1], sys.argv[2], sys.argv[3]))

    parser = argparse.ArgumentParser(
        description="Git merge driver for .memory files"
    )
    parser.add_argument("--install", action="store_true",
                        help="Configure git merge driver globally")
    parser.add_argument("--repo", default=".", help="Repo path for --install")
    parser.add_argument("--merge", nargs=2, metavar=("A", "B"),
                        help="Merge two .memory files directly")
    parser.add_argument("--output", "-o", help="Output path for --merge")

    args = parser.parse_args()

    if args.install:
        install(args.repo)
    elif args.merge:
        if not args.output:
            parser.error("--output required with --merge")
        direct_merge(args.merge[0], args.merge[1], args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
