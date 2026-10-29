<!-- rubric: {"name": "v2", "scale": [0, 2], "dims": ["A_chain_closure", "B_risk_flagging", "C_verification", "D_scope_complexity", "E_handoff", "F_executability", "G_generality"], "aggregate": ["A_chain_closure", "B_risk_flagging", "C_verification", "D_scope_complexity", "E_handoff"]} -->
# Skill quality rubric V2: gap-based

You are auditing an agent **skill**: a package of procedural knowledge (SKILL.md plus optional scripts and references) that an LLM agent loads at run time to solve tasks like the one shown. You see the task instruction and the full skill package. Judge the skill as a document: do not solve the task, and do not reward length.

Every dimension asks the same question about one kind of information: **if an agent followed this skill on this task, is any execution-critical information missing?** Score each dimension independently on 0–2:

- **2 — Self-contained:** no execution-critical information of this kind is missing.
- **1 — Recoverable gap:** something is missing or vague, but a competent agent can infer it without making a substantive implementation decision.
- **0 — Critical gap:** filling the gap requires an independent decision that may change task correctness, or the skill gives wrong information.

A single critical gap is enough for 0, however good the rest of the skill is. Do not average within a dimension.

## Dimensions

**A_chain_closure — Chain Closure.** Does the workflow connect the task input all the way to the required final output (format, location, fields)?
Critical gap: the workflow stops at an intermediate result (e.g., computes a volume but never says how to turn it into the reported mass or where to write it).

**B_risk_flagging — Risk Flagging.** Are non-obvious, failure-prone details (unit or coordinate conventions, off-by-one, encoding, edge cases, silent library defaults) flagged explicitly where they matter?
Critical gap: a model reports height from the lake bottom while the task needs depth from the surface, and the skill never mentions the conversion.

**C_verification — Verification.** Is verification an explicit step with concrete, actionable checks and expected results (sanity ranges, invariants, schema checks, a command to run)?
Critical gap: no way to tell a correct output from a plausible wrong one; "check your result" without saying how.

**D_scope_complexity — Scope–Complexity Match.** Does the level of detail match the real complexity of the task? Hard sub-steps need detail; trivial ones need little.
Critical gap: a multi-stage numerical procedure is compressed into one sentence ("parse the file and compute the answer").

**E_handoff — Output-to-Next-Step Handoff.** Is it clear how each intermediate output is consumed by the next step (field names, variables, formulas, file paths, function signatures, script invocations)?
Critical gap: a script is mentioned but its arguments or output format are not, so the next step cannot use its result.

**F_executability — Executability.** Can the provided code, commands and scripts run as written in a standard environment: correct syntax, referenced files exist in the package, dependencies stated, invocations complete?
Critical gap: SKILL.md calls `scripts/convert.py` but the package has no such file, or the code uses an undefined function.

**G_generality — Generality.** Is the skill reusable for other instances of this task type, rather than tuned to this one instance?
- 2: no instance-specific constants, file names, or expected answers; parameters are taken from the input.
- 1: some instance-specific values appear but only as clearly labeled examples.
- 0: the skill hard-codes this instance's answer, magic constants, or exact file contents, so it would fail or mislead on a sibling instance.

## Procedure

For each dimension, in order: (1) quote the most relevant passage(s) from the skill, or write "none" if the information is absent; (2) explain in 1–3 sentences whether a gap exists and whether it is recoverable or critical; (3) only then give the score. Judge only what is written in the skill package.

## Output

Return only one JSON object, no prose before or after:

```json
{"dimensions": {
  "A_chain_closure": {"evidence": "<quote or none>", "rationale": "<1-3 sentences>", "score": 0},
  "B_risk_flagging": {"evidence": "...", "rationale": "...", "score": 0},
  "C_verification": {"evidence": "...", "rationale": "...", "score": 0},
  "D_scope_complexity": {"evidence": "...", "rationale": "...", "score": 0},
  "E_handoff": {"evidence": "...", "rationale": "...", "score": 0},
  "F_executability": {"evidence": "...", "rationale": "...", "score": 0},
  "G_generality": {"evidence": "...", "rationale": "...", "score": 0}
}}
```
