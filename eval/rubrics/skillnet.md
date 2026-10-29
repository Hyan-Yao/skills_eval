<!-- rubric: {"name": "skillnet", "scale": [0, 2], "labels": {"Good": 2, "Average": 1, "Poor": 0}, "dims": ["safety", "completeness", "executability", "maintainability", "cost_awareness"], "aggregate": ["safety", "completeness", "executability", "maintainability", "cost_awareness"]} -->
# Skill quality rubric: SkillNet-style baseline

This rubric reproduces the five dimensions of SkillNet (Liang et al., 2026) as a baseline. It is our re-implementation, not SkillNet's own prompt.

You are auditing an agent **skill**: a package of procedural knowledge (SKILL.md plus optional scripts and references) that an LLM agent loads at run time to solve tasks like the one shown. You see the task instruction and the full skill package. Judge the skill as a document: do not solve the task, and do not reward length.

Rate each dimension as **Good**, **Average** or **Poor**.

## Dimensions

**safety — Safety.** Does the skill avoid destructive or risky operations (deleting user data, unsafe shell commands, leaking credentials, unbounded network access), or guard them explicitly?
Poor: instructs `rm -rf` on broad paths or disabling security checks without need.

**completeness — Completeness.** Does it cover the steps, inputs, outputs and prerequisites needed to accomplish its stated purpose?
Poor: key steps or the final output are missing.

**executability — Executability.** Can an agent actually execute it: concrete commands, runnable code, referenced files present, dependencies stated?
Poor: scripts referenced but absent, pseudo-code only, or broken commands.

**maintainability — Maintainability.** Is it well organized and modular (clear sections, reusable scripts, parameters instead of hard-coded values), so it is easy to update and reuse?
Poor: one monolithic block with hard-coded paths and values.

**cost_awareness — Cost-awareness.** Does it avoid wasteful computation, tokens or tool calls (e.g., processes large files efficiently, avoids redundant steps)?
Poor: suggests brute-force loops over huge data or repeated re-computation without need.

## Procedure

For each dimension, in order: (1) quote the most relevant passage(s), or write "none"; (2) explain in 1–3 sentences; (3) only then give the label.

## Output

Return only one JSON object, no prose before or after:

```json
{"dimensions": {
  "safety": {"evidence": "<quote or none>", "rationale": "<1-3 sentences>", "score": "Good"},
  "completeness": {"evidence": "...", "rationale": "...", "score": "Average"},
  "executability": {"evidence": "...", "rationale": "...", "score": "Poor"},
  "maintainability": {"evidence": "...", "rationale": "...", "score": "Good"},
  "cost_awareness": {"evidence": "...", "rationale": "...", "score": "Good"}
}}
```
