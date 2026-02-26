# Claude Code — Hooks Integration

Cortex integrates with Claude Code via native lifecycle hooks. Memory injection
and storage happen automatically on every turn — no wrapper scripts, no manual steps.

---

## How it works

```
You type a prompt
    ↓
UserPromptSubmit hook → queries memory → injects context Claude sees automatically
    ↓
Claude responds with memory-informed answers
    ↓
Stop hook → stores Claude's response into memory
    ↓
Memory grows. Next prompt gets better context.
```

**`UserPromptSubmit`** — fires before every prompt. Queries the active memory
source(s) and returns `additionalContext` injected silently into Claude's context.

**`Stop`** — fires after every response. Reads the last assistant turn from the
session transcript and stores it automatically.

---

## Memory sources

Two independent memory stores can run simultaneously:

| Source | File | Scope | Committed to git? |
|---|---|---|---|
| **Project** | `.claude/memory/project.memory` | This repo only | Yes — shared with team |
| **Global** | `~/.claude/memory/global.memory` | Every session on this machine | No — personal only |

The `source` field in `.claude/memory/config.json` controls which are active:

```
"project"  — project memory only
"global"   — global memory only
"both"     — both (default)
"off"      — disabled
```

Switch any time with the `/memory` command. No restart required.

---

## Use cases

### Solo developer
Use `"both"`. Global memory accumulates personal preferences and cross-project
patterns (preferred tools, personal workflow). Project memory accumulates
repo-specific decisions and context. Both inject on every prompt.

### Dev team
Commit `project.memory` to the repo. Every developer gets the accumulated
project context from all previous sessions by all team members. Each developer
also has their own `global.memory` for personal context that never pollutes
the shared store.

Pipeline integration (e.g. after a PR merge) can consolidate feature branch
memory into the main branch automatically using `Memory.merge()` — see the
Team and shared repositories section in the main README.

### Focused work
Switch to `"project"` when you want tight repo-specific context without global
noise. Switch to `"off"` temporarily when working on something where injected
context is causing confusion.

---

## Setup

**1. Copy library and hooks into your project:**

```
.claude/
  memory/
    cortex.py          ← copy from repo root
    memory.py          ← copy from repo root
    harness.py         ← copy from repo root
    config.json        ← copy from examples/claude_code_hooks/
    project.memory     ← created on first run or seeded manually
    hooks/
      on_prompt.py     ← copy from examples/claude_code_hooks/
      on_stop.py       ← copy from examples/claude_code_hooks/
  commands/
    memory.md          ← copy from examples/claude_code_hooks/memory.md
```

**2. Add hooks to `.claude/settings.json`:**

> **Use absolute paths.** Claude Code resolves hook commands from its own
> working directory, which may not be your repo root. Relative paths
> silently fail. Replace the placeholder paths below with your actual paths.

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/usr/bin/python3 /absolute/path/to/.claude/memory/hooks/on_prompt.py"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/usr/bin/python3 /absolute/path/to/.claude/memory/hooks/on_stop.py"
          }
        ]
      }
    ]
  }
}
```

Find your Python: `which python3`. Typical values: `/usr/bin/python3` (macOS system)
or `/opt/homebrew/bin/python3` (Homebrew).

**3. Pre-approve hooks in `.claude/settings.local.json`:**

```json
{
  "permissions": {
    "allow": [
      "Bash(/usr/bin/python3 /absolute/path/to/.claude/memory/hooks/on_prompt.py:*)",
      "Bash(/usr/bin/python3 /absolute/path/to/.claude/memory/hooks/on_stop.py:*)"
    ]
  }
}
```

**4. (Optional) Set up global memory:**

For personal cross-project context that follows you across all repos:

```bash
mkdir -p ~/.claude/memory/hooks
cp cortex.py memory.py ~/.claude/memory/
cp examples/claude_code_hooks/on_prompt.py ~/.claude/memory/hooks/
cp examples/claude_code_hooks/on_stop.py   ~/.claude/memory/hooks/
```

Wire into `~/.claude/settings.json` (same structure as above, pointing at
`~/.claude/memory/hooks/`). This fires for every Claude Code session on your
machine regardless of which project you're in.

**5. Seed memory (optional but recommended):**

```python
from memory import Memory
mem = Memory.create(description="my project")
mem.store("uses Python 3.12, FastAPI, PostgreSQL, deployed on AWS ECS")
mem.store("auth uses JWT with 24h expiry, refresh tokens in Redis")
mem.save(".claude/memory/project.memory")
```

**6. Run `claude` normally — restart required after first adding settings.json:**

```bash
claude
```

---

## /memory command

Copy `memory.md` to `.claude/commands/memory.md`. This adds a `/memory`
slash command inside Claude Code sessions:

```
/memory status                        — show active source and store stats
/memory source both                   — switch to both sources
/memory source project                — project memory only
/memory source global                 — global memory only  
/memory source off                    — disable injection
/memory what did we decide about X?  — query memory directly
/memory store <text>                  — explicitly save something
```

Source changes take effect immediately on the next prompt — no restart needed.

---

## Verify it's working

Ask Claude something project-specific that isn't in CLAUDE.md:

```
what project memory context do you have about authentication?
```

If the answer contains specific details from your seeded memories or past
sessions, injection is working. The `additionalContext` is injected silently
into Claude's context — it doesn't appear in your chat view.

---

## Troubleshooting

**Hooks not firing**
Claude Code must restart after `settings.json` is added — no hot-reload.
Exit with `/exit` and relaunch. Run `/hooks` inside Claude Code to confirm
hooks are loaded.

Test a hook directly:
```bash
echo '{"prompt": "test"}' | /usr/bin/python3 /path/to/on_prompt.py
```
Should return JSON with `additionalContext`.

**Python 3.9 compatibility**
The `str | None` union hint syntax requires Python 3.10+. On Python 3.9 you'll see:
```
TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
```
Remove return type annotations from hook function signatures — they're optional.

**Memory file not found**
If the hook fires but injects nothing, `project.memory` doesn't exist yet.
Seed it first, or the file will be created automatically after the first
session that generates a response long enough to store.

---

## What gets stored automatically

`on_stop.py` stores the first 600 characters of every assistant response
above 80 characters. Over time this accumulates decisions, solutions,
and context established across sessions.

Use `/memory store` to explicitly save something important:

```
/memory store decided to use platform events instead of scheduled Apex for real-time sync
```

---

## Notes

- `settings.json` (hook wiring) — commit to repo, shared with team
- `settings.local.json` (permissions) — personal, gitignore it
- `project.memory` — commit to repo, shared with team; grows automatically
- `~/.claude/memory/global.memory` — never commit, personal only
- `config.json` — commit to repo (contains `source` default for the team)
