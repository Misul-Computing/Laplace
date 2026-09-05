#!/usr/bin/env python3
"""Reject private workspace instructions and state in the tracked repository."""
import subprocess
import sys
from pathlib import PurePosixPath

PRIVATE_FILES = {
    "agents.md", "claude.md", "gemini.md", "skill.md", "memory.md",
    "workspace.md", "plan.md", "gates.md", "copilot-instructions.md",
    ".cursorrules", ".windsurfrules", ".clinerules",
}
PRIVATE_DIRECTORIES = {
    ".agents", ".agent", ".claude", ".codex", ".cursor", ".jspace",
    ".speed-work", ".superpowers", ".performance", "superpowers",
}

paths = subprocess.check_output(["git", "ls-files", "-z"]).decode("utf-8", "surrogateescape").split("\0")
violations = []
for path in filter(None, paths):
    parts = PurePosixPath(path.lower()).parts
    if (parts[-1] in PRIVATE_FILES or parts[-1].startswith("agents.md.") or
            any(part in PRIVATE_DIRECTORIES or part.startswith(".superpowers-")
                for part in parts[:-1]) or
            any(parts[i:i + 2] in ((".github", "instructions"), (".github", "prompts"))
                for i in range(len(parts) - 1))):
        violations.append(path)

if violations:
    print("Private workspace files must not be tracked:", file=sys.stderr)
    for path in violations:
        print(f"  {path!r}", file=sys.stderr)
    sys.exit(1)
print("Repository hygiene: no private workspace files tracked.")
