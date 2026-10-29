<!-- rubric: {"name": "v1", "scale": [1, 5], "dims": ["completeness", "clarity", "code_contextualization", "validation", "error_handling", "dependency_tooling"], "aggregate": ["completeness", "clarity", "code_contextualization", "validation", "error_handling"]} -->
# Skill quality rubric V1: holistic

You are auditing an agent **skill**: a package of procedural knowledge (SKILL.md plus optional scripts and references) that an LLM agent loads at run time to solve tasks like the one shown. You see the task instruction and the full skill package. Judge the skill as a document: do not solve the task, and do not reward length.

Rate each dimension on a 1–5 scale:

- **5** — excellent: nothing relevant is missing; an agent could follow it without guessing.
- **4** — good: minor omissions that do not affect the result.
- **3** — adequate: covers the essentials but with noticeable gaps or vagueness.
- **2** — weak: important parts missing or unclear.
- **1** — poor: largely absent, wrong, or unusable.

## Dimensions

**completeness — Completeness.** Are all steps needed to complete the task covered, from input to final output?
Example of a low score: the workflow omits the final conversion and output-writing step.

**clarity — Clarity.** Are the instructions clear, specific and unambiguous?
Example of a low score: "process the data appropriately" with no indication of what processing.

**code_contextualization — Code Contextualization.** Is code integrated into the workflow, with its purpose, inputs and outputs explained?
Example of a low score: a code block appears with no statement of when to use it or what it returns.

**validation — Validation & Expected Results.** Are expected outputs and concrete ways to verify correctness provided?
Example of a low score: no expected ranges, invariants or checks.

**error_handling — Error Handling & Recovery.** Are common failures and recovery procedures addressed?
Example of a low score: no mention of what to do when parsing fails or a dependency is missing.

**dependency_tooling — Dependency & Tooling Usability.** Are external tools, scripts and dependencies documented and usable as described? (Reported separately; not part of the overall score.)
Example of a low score: a script is referenced but not included, or required packages are not named.

## Procedure

For each dimension, in order: (1) quote the most relevant passage(s) from the skill, or write "none"; (2) explain your judgement in 1–3 sentences; (3) only then give the score. Judge only what is written in the skill package.

## Output

Return only one JSON object, no prose before or after:

```json
{"dimensions": {
  "completeness": {"evidence": "<quote or none>", "rationale": "<1-3 sentences>", "score": 1},
  "clarity": {"evidence": "...", "rationale": "...", "score": 1},
  "code_contextualization": {"evidence": "...", "rationale": "...", "score": 1},
  "validation": {"evidence": "...", "rationale": "...", "score": 1},
  "error_handling": {"evidence": "...", "rationale": "...", "score": 1},
  "dependency_tooling": {"evidence": "...", "rationale": "...", "score": 1}
}}
```
