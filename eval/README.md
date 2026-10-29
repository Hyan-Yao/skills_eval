# eval: LLM-as-a-judge for skills

Static skill-quality scoring from the skill package alone. The judge sees the task instruction and the skill package (every SKILL.md first, then scripts and references; per-file cap 15k chars, total cap 60k chars, binaries listed but not shown). It never sees success rates or version ids.

## Rubrics (`rubrics/`)

| Rubric | Dimensions | Scale | Aggregate |
|---|---|---|---|
| `v2` (`v2_gap.md`, ours) | A Chain Closure, B Risk Flagging, C Verification, D Scope–Complexity Match, E Output-to-Next-Step Handoff, F Executability, G Generality | 0–2 (2 self-contained / 1 recoverable gap / 0 critical gap) | mean of A–E; F and G reported separately |
| `v1` (`v1_holistic.md`) | Completeness, Clarity, Code Contextualization, Validation & Expected Results, Error Handling & Recovery, Dependency & Tooling | 1–5 | mean of first five |
| `skillnet` (`skillnet.md`, our re-implementation) | Safety, Completeness, Executability, Maintainability, Cost-awareness | Good/Average/Poor → 2/1/0 | mean of all five |

Each judge call quotes evidence and gives a rationale before each score, and returns strict JSON. Each (skill, rubric) is sampled `--n` times (default 3); the reported dimension score is the median.

## Usage

```bash
# credentials: OPENAI_BASE_URL / OPENAI_API_KEY in the environment, or --env-file path/to/.env
python eval/judge.py --rubric all --model gpt-oss-120b \
  --skill-dir skillsbench/tasks/3d-scan-calc/environment/skills \
  --task-dir skillsbench/tasks/3d-scan-calc --task-id 3d-scan-calc --version-id human

# batch: manifest.csv has task_id, version_id, skill_dir, task_dir
python eval/judge.py --rubric all --manifest runs/manifest.csv --out runs/judge.jsonl --workers 8

# analysis: success.csv has task_id, version_id, n_runs, n_success
python eval/analyze.py --judge runs/judge.jsonl --success runs/success.csv --out runs/analysis
```

Responses are cached in `runs/judge_cache/`, keyed by skill text, instruction, rubric, model, temperature and sample index; a re-run only calls the API for missing samples.

`analyze.py` writes:
- `summary.csv`: per metric, the within-task pooled Somers' D (score vs success across versions); pairwise accuracy on version pairs whose success differs by Fisher p < alpha; levels Somers' D across tasks per version; 95% task-bootstrap CIs.
- `within_task.csv`: per-task D.
- `trajectories.csv`: mean score per dimension and success per version (s0…s4, human).
- `trajectory_<rubric>.png`, if matplotlib is installed.

## Tests

```bash
python -m unittest discover -s eval/tests -v
```
