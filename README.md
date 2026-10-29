# skills_eval

Code for checking whether static LLM-judge scores of agent skills track their measured success on SkillsBench.

For each task we build several versions of a skill: the human-written skill shipped with the task, an initial skill `s0` written by the Anthropic skill-creator, and later versions `s1..s4` produced by a skill optimizer (SkillOpt for now). Each version is run 5 times on the task to get a success label. Each version is then scored by an LLM judge that sees only the skill package and the task instruction. The analysis asks whether the judge scores rank versions the same way success does.

The agent and the optimizer are `qwen36-35b-a3b`; the judge is `gpt-oss-120b`. Both are served by the ASU OpenAI-compatible endpoint.

## Layout

| Path | Contents |
|---|---|
| `pipeline/run_task.sh` | Runs one SkillsBench task once with BenchFlow (`qwen-code` agent, Docker sandbox) and a given skills dir, or with no skill. Caps concurrent containers machine-wide (`MAX_CONTAINERS`, default 10). |
| `pipeline/gen_s0.py` | Generates `s0` per task with BenchFlow self-gen mode and the Anthropic skill-creator skill. |
| `pipeline/eval_group.py` | Runs one skill group (`human`, `s0`, ...) N times per task and writes `success.csv` (runs, successes, errors, mean reward). |
| `pipeline/tasks.txt` | Task list (pilot: `earthquake-plate-calculation`, `weighted-gdp-calc`). |
| `skillopt_bf/` | Adapter that runs SkillOpt with BenchFlow rollouts, plus `run.py`, which runs one task's optimization chain and exports every accepted and rejected version. SkillOpt itself is not modified. |
| `eval/skillnet_eval.py` | Wrapper around SkillNet's original evaluator (5 dimensions, Good/Average/Poor mapped to 2/1/0), with repeated sampling and per-task averaging. |
| `eval/judge.py`, `eval/rubrics/` | Our own judge with three rubrics (V1 holistic, V2 gap-based A–G, a SkillNet re-implementation). See `eval/README.md`. |
| `eval/analyze.py` | Within-task Somers' D, pairwise ranking accuracy, cross-task levels and per-dimension trajectories. |
| `reports/` | Result write-ups (`pilot-2026-10-29.html`). |
| `REPOS.md` | Upstream repositories and pinned commits. |

Upstream clones (`SkillOpt/`, `CoEvoSkills/`, `skillsbench/`, `SkillNet/`, `anthropic-skills/`) and all outputs (`runs/`) are git-ignored.

## Setup

1. Clone the upstream repositories into the repo root at the commits listed in `REPOS.md`.
2. Install Docker and make sure the current user can reach the Docker socket.
3. Install BenchFlow: `uv tool install benchflow` (tested with `bench` 0.7.11).
4. Create the Python environment (Python 3.11):
   ```bash
   uv venv --python 3.11 .venv
   uv pip install -p .venv -e SkillNet/skillnet-ai -e SkillOpt matplotlib scipy
   ```
5. Put `OPENAI_API_KEY` and `OPENAI_BASE_URL` in an env file. The scripts currently read `~/workspace/jarvis/.secrets/asu_llm.env`; `eval/skillnet_eval.py` and `eval/judge.py` accept `--env-file`, while `pipeline/run_task.sh` and `pipeline/gen_s0.py` have the path hard-coded.

BenchFlow must get the model as `vllm/<model>` together with `--agent-env BENCHFLOW_PROVIDER_BASE_URL=$OPENAI_BASE_URL`. With the `openai/` prefix it calls api.openai.com and fails with 401.

## Usage

```bash
# one run
pipeline/run_task.sh earthquake-plate-calculation skillsbench/tasks/earthquake-plate-calculation/environment/skills runs/tmp/job1

# s0 from skill-creator
python3 pipeline/gen_s0.py --concurrency 2

# success labels for a group (5 runs per task)
python3 pipeline/eval_group.py --group human --n 5 --concurrency 4
python3 pipeline/eval_group.py --group s0 --skills-root runs/skills/s0 --n 5

# SkillOpt chain for one task: 5 training rollouts and 5 validation runs per step,
# up to 4 accepted versions, stop after 3 rejections in a row
.venv/bin/python skillopt_bf/run.py --task earthquake-plate-calculation \
  --k 5 --sel 5 --target 4 --max-attempts 3 --workers 5 --gate-metric soft

# SkillNet judge; manifest columns: task_id, version_id, skill_root
.venv/bin/python eval/skillnet_eval.py --manifest runs/manifest.csv --out runs/judge_skillnet.jsonl --n 3
```

These runs are long (one SkillsBench run takes up to 15 minutes; one SkillOpt step takes 25–40 minutes). On a shared machine, launch them as separate memory-capped units, for example `systemd-run --user --slice=exp.slice -p MemoryMax=6G ...`.

Outputs:
- `runs/eval/<group>/success.csv`: success per task for a group.
- `runs/skillopt/<task>/`: `versions.json` (every attempt and its gate decision), `val_s<k>.csv`, `val_s<k>_rej<j>.csv`, `train_s<k>_att<j>.csv`.
- `runs/skills/<version>/<task>/<skill>/`: exported skill packs (`s0`, `s1`, ..., `s1_rej1`, ...).

## Status (2026-10-29)

Pilot on 2 tasks; details in `reports/pilot-2026-10-29.html`.

- Human skill: 2/5 successes on each task. `s0`: 2/20 (earthquake) and 0/20 (weighted-gdp).
- SkillOpt with `qwen36-35b-a3b` as optimizer: all 6 candidates were rejected and both chains stopped. Rewards on these tasks are binary, so the soft gate behaves like the hard one, and the candidates pass about the same share of tests as `s0`.
- SkillNet judge: human 2.0 vs `s0` 1.8 (earthquake) and 1.8 vs 1.6 (weighted-gdp). The order matches success, but the gap is small and most scores are at the ceiling.

Known issues:
- The verifier summary given to the SkillOpt optimizer includes failed test names and assertion headlines.
- Concurrent BenchFlow runs occasionally fail with a Docker "network not found" error; such runs have no reward and are rerun by `eval_group.py`.
