# Claude Code — Hooks Integration

The hooks approach is how Cortex integrates seamlessly with Claude Code.
No wrapper scripts. No manual steps. Just run `claude` normally.

---

## How it works

**`UserPromptSubmit`** fires before every prompt Claude processes.
`on_prompt.py` queries memory with the user's message and returns
`additionalContext` — injected automatically into Claude's context window.

**`Stop`** fires after every Claude response.
`on_stop.py` reads the last assistant turn from the session transcript
and stores it in memory automatically.

```
You type a prompt
    ↓
on_prompt.py fires → queries memory → injects top-5 results as context
    ↓
Claude responds with memory-informed context
    ↓
on_stop.py fires → stores Claude's response in memory
    ↓
Memory grows. Next prompt gets better context.
```

---

## Setup

**1. Copy the library and hooks into your project:**

```
.claude/
  memory/
    cortex.py          ← copy from repo root
    memory.py          ← copy from repo root
    harness.py         ← copy from repo root
    project.memory     ← created on first run (or seed manually)
    hooks/
      on_prompt.py     ← copy from examples/claude_code_hooks/
      on_stop.py       ← copy from examples/claude_code_hooks/
```

**2. Add hooks to `.claude/settings.json`:**

Copy `settings.json` from this directory into `.claude/settings.json`
in your project root, replacing the placeholder paths with the actual
absolute paths on your machine.

> **Important:** Use absolute paths, not relative ones. Claude Code fires
> hooks from its own working directory which may not be your repo root,
> causing relative paths to silently fail.

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/usr/bin/python3 /Users/you/src/myproject/.claude/memory/hooks/on_prompt.py"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/usr/bin/python3 /Users/you/src/myproject/.claude/memory/hooks/on_stop.py"
          }
        ]
      }
    ]
  }
}
```

Find the right Python path with `which python3`. On macOS this is typically
`/usr/bin/python3` or `/opt/homebrew/bin/python3`.

**3. Allow the hooks in `.claude/settings.local.json`:**

Claude Code will prompt you to approve hooks on first run. To pre-approve
(use the same absolute paths as above):

```json
{
  "permissions": {
    "allow": [
      "Bash(/usr/bin/python3 /Users/you/src/myproject/.claude/memory/hooks/on_prompt.py:*)",
      "Bash(/usr/bin/python3 /Users/you/src/myproject/.claude/memory/hooks/on_stop.py:*)"
    ]
  }
}
```

**4. Seed memory (optional but recommended):**

Create a seed script that stores known project facts before the first session:

```python
from memory import Memory

mem = Memory.create(description="my project")
mem.store("uses Python 3.12, FastAPI, PostgreSQL, deployed on AWS ECS")
mem.store("auth uses JWT with 24h expiry, refresh tokens in Redis")
mem.store("feature branches from main, PRs reviewed by @alice")
# ... add whatever Claude should always know
mem.save(".claude/memory/project.memory")
```

**5. Run `claude` normally:**

```bash
claude
```

That's it. Memory injection and storage happen automatically on every turn.

---

## Verify it's working

Inside a Claude Code session:

```
/memory status
```

You should see query count incrementing each turn as the hooks fire.

To see what context was injected on your last prompt, the `additionalContext`
block appears in Claude's view but not in yours — you can check by asking:

```
what project memory context do you have about this session?
```

---

## Tuning

**`on_prompt.py`** — adjust `TOP_K` (default 5) to inject more or fewer memories per prompt. More context = better answers but slightly more tokens per turn.

**`on_stop.py`** — adjust `MIN_LENGTH` (default 80) and `MAX_LENGTH` (default 600). Raise `MIN_LENGTH` to skip more short responses. Lower `MAX_LENGTH` if you find stored entries are too verbose.

Both hooks fail silently (exit 0) on any error — they will never break a Claude Code session.

---

## Troubleshooting

**Hooks not firing at all**

Claude Code must be restarted after adding `settings.json` — it doesn't hot-reload hook config. Exit with `/exit` and relaunch.

Verify hooks are loaded by running `/hooks` inside Claude Code. If the hooks appear but show an error, check the command path.

To confirm a hook is firing, add a quick test:
```bash
echo '{"prompt": "test"}' | /usr/bin/python3 /path/to/on_prompt.py
```
You should see JSON output with `additionalContext`.

**Python version compatibility**

The hooks require Python 3.9+. The `str | None` union type hint syntax requires Python 3.10+. If you see:
```
TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
```
Remove return type annotations from function signatures in the hook files (they're optional and only for readability).

**Memory file not found**

If the hook fires but injects nothing, check that `project.memory` exists at `.claude/memory/project.memory`. Run the seed script first:
```bash
python3 .claude/memory/seed_memory.py
```

**Context not appearing in responses**

The `additionalContext` is injected into Claude's context silently — it doesn't appear in your chat. You can verify it's being received by asking Claude directly: *"what project memory context do you have?"* If the answer contains project-specific details that aren't in CLAUDE.md, injection is working.

---

## What gets stored

`on_stop.py` stores the first 600 characters of every assistant response above 80 characters. Over time this accumulates:

- Decisions made ("decided to use JWT with 24h expiry")  
- Problems solved ("fixed slow query by adding composite index")  
- Context established ("the payments service uses blue-green deployment")

For explicit important facts, use the `/memory store` command:

```
/memory store decided to defer the Zuora subscription amendment to Phase 2
```

---

## Notes

- The memory file (`.memory`) should be committed to your repo so it persists across machines and is shared with the team.
- The hooks fire on every prompt and every stop — keep them fast. On typical hardware, a memory query takes < 5ms for stores up to 2,000 entries.
- `settings.json` (with hooks config) should be committed. `settings.local.json` (with permissions) is personal and typically gitignored.
