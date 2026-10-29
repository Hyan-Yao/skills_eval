"""Run one skill group (e.g. human, s0..s4) on a task list, N fresh runs each, and tally success.

  python3 pipeline/eval_group.py --group human --n 10 --concurrency 4
  python3 pipeline/eval_group.py --group s0 --skills-root runs/skills/s0 --n 10

Skills for task T come from <skills-root>/<T>/ (a dir of skill dirs); for --group human they come
from the task's own environment/skills. Each run gets its own jobs dir under runs/eval/<group>/<T>/rep<k>,
so reruns resume instead of duplicating. Results: runs/eval/<group>/success.csv (task_id, version_id,
n_runs, n_success, n_error) -- the format eval/analyze.py reads.
"""
import argparse
import csv
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "skillsbench/tasks"
DEFAULT_TASKS = ROOT / "pipeline/tasks.txt"


def skills_dir(group, task, skills_root):
    if group == "human":
        return TASKS / task / "environment/skills"
    return Path(skills_root) / task


def run_one(group, task, rep, skills_root):
    jobs = ROOT / f"runs/eval/{group}/{task}/rep{rep}"
    if not list(jobs.glob("*/summary.json")):
        sd = skills_dir(group, task, skills_root)
        cmd = [str(ROOT / "pipeline/run_task.sh"), task, str(sd) if sd else "none", str(jobs)]
        log = jobs.with_suffix(".log")
        jobs.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return task, reward(jobs)


def reward(jobs):
    """1/0 for a finished run, None for an errored or missing one."""
    for res in jobs.glob("*/*/result.json"):
        r = json.loads(res.read_text())
        if r.get("error"):
            return None
        rw = (r.get("rewards") or {}).get("reward", r.get("reward"))
        return None if rw is None else int(float(rw) >= 1.0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--skills-root")
    ap.add_argument("--tasks", default=str(DEFAULT_TASKS))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=4)
    a = ap.parse_args()
    tasks = [t.strip() for t in open(a.tasks) if t.strip() and not t.startswith("#")]
    jobs = [(t, k) for k in range(1, a.n + 1) for t in tasks]
    tally = {t: [0, 0, 0] for t in tasks}  # runs, success, error
    with ThreadPoolExecutor(a.concurrency) as pool:
        for task, r in pool.map(lambda j: run_one(a.group, j[0], j[1], a.skills_root), jobs):
            if r is None:
                tally[task][2] += 1
            else:
                tally[task][0] += 1
                tally[task][1] += r
            print(f"[{a.group}] {task}: {tally[task]}", flush=True)
    out = ROOT / f"runs/eval/{a.group}/success.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "version_id", "n_runs", "n_success", "n_error"])
        for t, (n, s, e) in tally.items():
            w.writerow([t, a.group, n, s, e])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
