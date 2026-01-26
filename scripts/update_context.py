#!/usr/bin/env python3
import datetime as dt
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOW_FILE = ROOT / "notes" / "now.md"
STATUS_FILE = ROOT / "notes" / "status.md"
TASKS_FILE = ROOT / "notes" / "tasks.md"


def _read_head(path: Path, max_lines: int) -> str:
    if not path.exists():
        return f"(missing: {path})"
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[:max_lines])


def _run_git(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "(git unavailable)"


def main() -> None:
    timestamp = dt.datetime.now().isoformat(timespec="seconds")
    git_status = _run_git(["status", "-sb"])
    git_log = _run_git(["log", "-5", "--oneline"])
    status_head = _read_head(STATUS_FILE, 40)
    tasks_head = _read_head(TASKS_FILE, 40)

    content = f"""# Now — Anki CSV Builder

Updated: {timestamp}

## Quick pointers
- notes/status.md (project status)
- notes/tasks.md (task tracker)
- notes/vision_v2.md (product direction)

## Git status
```
{git_status}
```

## Recent commits
```
{git_log}
```

## Status (head)
```
{status_head}
```

## Tasks (head)
```
{tasks_head}
```

## Session scratchpad
- What I changed:
- Why:
- Next steps:
- Open questions:
"""

    NOW_FILE.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
