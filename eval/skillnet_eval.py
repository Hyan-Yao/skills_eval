"""Score skill packages with SkillNet's original evaluator (zjunlp/SkillNet, skillnet_ai.evaluator).

Five dimensions (safety, completeness, executability, maintainability, cost_awareness), each
Good/Average/Poor, mapped to 2/1/0. The prompt and parsing are SkillNet's own; this wrapper only
adds repeated sampling (median per dimension), task-level averaging over multiple skills, and
JSONL output in the format eval/analyze.py reads.

  .venv/bin/python eval/skillnet_eval.py --manifest runs/manifest.csv --out runs/judge_skillnet.jsonl
  .venv/bin/python eval/skillnet_eval.py --skill-root <dir> --task-id t --version-id human

Manifest columns: task_id, version_id, skill_root (a skill dir, or a dir holding several skill dirs).
Scripts are never executed (run_scripts=False, SkillNet's default).
"""
import argparse
import csv
import json
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from skillnet_ai.evaluator import EvaluatorConfig, Skill, SkillEvaluator

DIMS = ["safety", "completeness", "executability", "maintainability", "cost_awareness"]
LEVEL = {"good": 2, "average": 1, "poor": 0}
DEFAULT_ENV = Path.home() / "workspace/jarvis/.secrets/asu_llm.env"


def load_env(path):
    for line in Path(path).read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))


def skill_dirs(root):
    root = Path(root)
    if (root / "SKILL.md").exists():
        return [root]
    return sorted(p.parent for p in root.glob("*/SKILL.md"))


def levels(result):
    """SkillNet result -> {dim: 0/1/2}; None if any dimension is missing."""
    out = {}
    for d in DIMS:
        lv = str((result.get(d) or {}).get("level", "")).strip().lower()
        if lv not in LEVEL:
            return None
        out[d] = LEVEL[lv]
    return out


def score_skill(evaluator, path, n):
    skill, err = Skill.from_path(str(path))
    if err:
        return {"skill": path.name, "error": err}
    samples, raw = [], []
    for _ in range(n):
        res = evaluator.evaluate(skill)
        raw.append({d: res.get(d) for d in DIMS})
        if (lv := levels(res)) is not None:
            samples.append(lv)
    if not samples:
        return {"skill": path.name, "error": "no parsable sample", "raw": raw}
    return {"skill": path.name, "n_ok": len(samples),
            "scores": {d: statistics.median(s[d] for s in samples) for d in DIMS}, "raw": raw}


def score_root(evaluator, task_id, version_id, root, n):
    per_skill = [score_skill(evaluator, p, n) for p in skill_dirs(root)]
    ok = [s for s in per_skill if "scores" in s]
    row = {"rubric": "skillnet", "model": evaluator.config.model, "task_id": task_id,
           "version_id": version_id, "skill_root": str(root), "n": n, "per_skill": per_skill}
    if not ok:
        row["error"] = "no skill scored"
        return row
    row["scores"] = {d: statistics.mean(s["scores"][d] for s in ok) for d in DIMS}
    row["aggregate"] = statistics.mean(row["scores"].values())
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest")
    ap.add_argument("--skill-root")
    ap.add_argument("--task-id", default="task")
    ap.add_argument("--version-id", default="v")
    ap.add_argument("--out", default="-")
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--temperature", type=float, default=0.3)  # SkillNet default
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--env-file", default=str(DEFAULT_ENV))
    a = ap.parse_args(argv)

    if a.env_file and Path(a.env_file).exists():
        load_env(a.env_file)
    cfg = EvaluatorConfig(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"],
                          model=a.model, temperature=a.temperature, run_scripts=False)
    evaluator = SkillEvaluator(cfg)

    if a.manifest:
        with open(a.manifest) as f:
            jobs = [(r["task_id"], r["version_id"], r["skill_root"]) for r in csv.DictReader(f)]
    else:
        jobs = [(a.task_id, a.version_id, a.skill_root)]

    out = sys.stdout if a.out == "-" else open(a.out, "a")
    with ThreadPoolExecutor(a.workers) as pool:
        for row in pool.map(lambda j: score_root(evaluator, *j, a.n), jobs):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[skillnet] {row['task_id']}/{row['version_id']}: "
                  f"{row.get('scores', row.get('error'))}", file=sys.stderr)


if __name__ == "__main__":
    main()
