#!/usr/bin/env python3
"""
cortex-install — One-command setup for Cortex Memory with Claude Code.

Usage:
    pip install cortex-memory
    cortex-install                  # project-level setup (run from repo root)
    cortex-install --global         # global setup (~/.claude/memory/)
    cortex-install --global --project  # both

What it does:
    1. Creates .claude/memory/hooks/ directory
    2. Writes on_prompt.py and on_stop.py hooks (using installed package)
    3. Generates config.json with source toggle
    4. Creates an empty project.memory if needed
    5. Prints the settings.json snippet with correct absolute paths
"""

import os
import sys
import json
import shutil
import textwrap
from pathlib import Path


HOOK_ON_PROMPT = '''\
#!/usr/bin/env python3
"""UserPromptSubmit hook — queries Cortex memory and injects context."""

import sys, os, json

CONFIG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

PROJECT_MEMORY = os.path.join(CONFIG_DIR, "project.memory")
GLOBAL_MEMORY = os.path.expanduser("~/.claude/memory/global.memory")


def load_config():
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except Exception:
        return {"source": "both", "top_k": 5}


def query_source(path, label, prompt, top_k):
    if not os.path.exists(path):
        return []
    try:
        from cortex_memory import Memory
        mem = Memory.load(path)
        results = mem.query(prompt, top_k=top_k)
        mem.save(path)
        if results:
            return [f"[{label}]"] + [f"\\u2022 {r.strip()}" for r in results]
    except Exception as e:
        sys.stderr.write(f"[memory/on_prompt] {label}: {e}\\n")
    return []


def main():
    config = load_config()
    source = config.get("source", "both")
    top_k = config.get("top_k", 5)

    if source == "off":
        sys.exit(0)

    try:
        event = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    prompt = event.get("prompt", "").strip()
    if not prompt:
        sys.exit(0)

    lines = []

    if source in ("project", "both"):
        lines += query_source(PROJECT_MEMORY, "Project Memory", prompt, top_k)

    if source in ("global", "both"):
        lines += query_source(GLOBAL_MEMORY, "Global Memory", prompt, top_k)

    if not lines:
        sys.exit(0)

    print(json.dumps({"additionalContext": "\\n".join(lines) + "\\n"}))


if __name__ == "__main__":
    main()
'''

HOOK_ON_STOP = '''\
#!/usr/bin/env python3
"""Stop hook — stores Claude responses into active memory source(s)."""

import sys, os, json

CONFIG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

PROJECT_MEMORY = os.path.join(CONFIG_DIR, "project.memory")
GLOBAL_MEMORY = os.path.expanduser("~/.claude/memory/global.memory")

MIN_LENGTH = 80
MAX_LENGTH = 600


def load_config():
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except Exception:
        return {"source": "both", "top_k": 5}


def last_assistant_text(transcript_path):
    if not transcript_path or not os.path.exists(transcript_path):
        return None
    try:
        turns = []
        with open(transcript_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                role = msg.get("role", "")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                if role == "assistant" and isinstance(content, str):
                    turns.append(content.strip())
        return turns[-1] if turns else None
    except Exception:
        return None


def store_to(path, text):
    try:
        from cortex_memory import Memory
        if os.path.exists(path):
            mem = Memory.load(path)
        else:
            mem = Memory.create()
            os.makedirs(os.path.dirname(path), exist_ok=True)
        mem.store(text)
        mem.save(path)
    except Exception as e:
        sys.stderr.write(f"[memory/on_stop] {path}: {e}\\n")


def main():
    config = load_config()
    source = config.get("source", "both")

    if source == "off":
        sys.exit(0)

    try:
        event = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    text = last_assistant_text(event.get("transcript_path", ""))
    if not text or len(text) < MIN_LENGTH:
        sys.exit(0)

    if len(text) > MAX_LENGTH:
        text = text[:MAX_LENGTH].rsplit(" ", 1)[0] + "\\u2026"

    if source in ("project", "both"):
        store_to(PROJECT_MEMORY, text)

    if source in ("global", "both"):
        store_to(GLOBAL_MEMORY, text)

    sys.exit(0)


if __name__ == "__main__":
    main()
'''

CONFIG_JSON = '{\n  "source": "both",\n  "top_k": 5\n}\n'


def find_python():
    """Find the Python executable path."""
    # Prefer the currently running Python
    python = sys.executable
    if python and os.path.exists(python):
        return python
    # Fall back to common locations
    for p in ["/usr/bin/python3", "/opt/homebrew/bin/python3"]:
        if os.path.exists(p):
            return p
    return "python3"


def setup_location(base_dir, label):
    """Set up hooks, config, and memory in a given base directory."""
    memory_dir = Path(base_dir) / "memory"
    hooks_dir = memory_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)

    # Write hooks
    on_prompt = hooks_dir / "on_prompt.py"
    on_stop = hooks_dir / "on_stop.py"
    on_prompt.write_text(HOOK_ON_PROMPT)
    on_stop.write_text(HOOK_ON_STOP)
    os.chmod(on_prompt, 0o755)
    os.chmod(on_stop, 0o755)

    # Write config.json if not exists
    config_path = memory_dir / "config.json"
    if not config_path.exists():
        config_path.write_text(CONFIG_JSON)

    # Create empty memory file if not exists
    if label == "project":
        mem_file = memory_dir / "project.memory"
    else:
        mem_file = memory_dir / "global.memory"

    if not mem_file.exists():
        from cortex_memory import Memory
        mem = Memory.create(description=f"{label} memory")
        mem.save(str(mem_file))
        print(f"  Created {mem_file}")

    python = find_python()
    prompt_path = str(on_prompt.resolve())
    stop_path = str(on_stop.resolve())

    print(f"\n  [{label}] Hooks written to {hooks_dir}/")
    print(f"  Python: {python}")

    return python, prompt_path, stop_path


def generate_settings_snippet(python, prompt_path, stop_path):
    """Return the hooks config dict to merge into settings.json."""
    return {
        "UserPromptSubmit": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f"{python} {prompt_path}",
                    }
                ]
            }
        ],
        "Stop": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f"{python} {stop_path}",
                    }
                ]
            }
        ],
    }


def update_settings_json(settings_path, hooks_config):
    """Merge hooks config into an existing settings.json, or create one."""
    settings_path = Path(settings_path)
    if settings_path.exists():
        with open(settings_path) as f:
            settings = json.load(f)
    else:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings = {}

    if "hooks" not in settings:
        settings["hooks"] = {}

    settings["hooks"].update(hooks_config)

    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)
        f.write("\n")

    print(f"  Updated {settings_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="cortex-install",
        description="Set up Cortex Memory hooks for Claude Code",
    )
    parser.add_argument(
        "--global", dest="setup_global", action="store_true",
        help="Set up global memory at ~/.claude/memory/",
    )
    parser.add_argument(
        "--project", dest="setup_project", action="store_true",
        help="Set up project memory at .claude/memory/ (default if no flags)",
    )
    parser.add_argument(
        "--no-settings", action="store_true",
        help="Don't modify settings.json — just write hooks and print the snippet",
    )

    args = parser.parse_args()

    # Default: project only if no flags
    if not args.setup_global and not args.setup_project:
        args.setup_project = True

    print("Cortex Memory — Claude Code Setup\n")

    if args.setup_project:
        print("[Project Setup]")
        cwd = Path.cwd()
        claude_dir = cwd / ".claude"
        python, prompt_path, stop_path = setup_location(claude_dir, "project")
        hooks_config = generate_settings_snippet(python, prompt_path, stop_path)

        if not args.no_settings:
            update_settings_json(claude_dir / "settings.json", hooks_config)
        else:
            print("\n  Add this to .claude/settings.json:")
            print(json.dumps({"hooks": hooks_config}, indent=2))

    if args.setup_global:
        print("\n[Global Setup]")
        global_claude = Path.home() / ".claude"
        python, prompt_path, stop_path = setup_location(global_claude, "global")
        hooks_config = generate_settings_snippet(python, prompt_path, stop_path)

        if not args.no_settings:
            update_settings_json(global_claude / "settings.json", hooks_config)
        else:
            print("\n  Add this to ~/.claude/settings.json:")
            print(json.dumps({"hooks": hooks_config}, indent=2))

    print("\nDone. Restart Claude Code to activate hooks.")
    print("Verify with: /hooks (inside Claude Code)")


if __name__ == "__main__":
    main()
