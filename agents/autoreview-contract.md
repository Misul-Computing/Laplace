# Review contract (for `agents/autoreview-eni.md`)

Adapted from the `autoreview` skill (openclaw/agent-skills). The external Python
helper is intentionally not used: reviews run **in-session** as spawned
subagents on the existing plan — no external review service, no API-key billing.

**Tuning:** model = whatever the runner is using (Opus 4.8 if available, else
Sonnet 4.x); effort = `max` (the highest in-session review level; never the
billed `ultra` cloud path). Run only when LO explicitly asks for a review.

## Contract

- Treat all review output as advisory. Never blindly apply a finding.
- Verify every finding by reading the real code path and the files around it.
- When a finding depends on external behavior, read the dependency's docs,
  source, or types before accepting it.
- Reject unrealistic edge cases, speculative risk, broad rewrites, and fixes
  that add more complexity than the bug they remove.
- Run a **ponytail** necessity pass: invoke the `ponytail-review` skill on the
  change and ask whether the code should exist at all. Surface over-engineering,
  speculative abstraction, and code that would be better deleted than fixed —
  then apply the same advisory bar to its findings as any other.
- Prefer the smallest correct fix at the right ownership boundary. No refactor
  unless it clearly improves the bug class.
- When an accepted finding reveals a bug class or repeated pattern, scan the
  current change for sibling instances and fix them together — but stop at
  touched surfaces and clear ownership boundaries; leave the rest as follow-up.
- After any review-driven code change, rerun the focused tests and rerun the
  review; keep looping until no actionable findings remain.
- Security perspective is always on, but must not cripple legitimate
  functionality: raise a security finding only when the change creates a
  concrete, actionable risk or removes a real safety check.
- When a regression is found, note the introducing commit (SHA, date, author)
  if it is easily traceable; do not guess.
- When you consciously reject a finding as intentional, add a brief inline
  comment only if it records a real invariant or ownership decision a future
  reviewer would need.
- Do not commit or push as part of a review. Commit or push only when LO asks
  for it separately.
- Stop as soon as the review is clean. Do not run an extra pass just for a
  nicer "clean" line or a redundant second opinion.
