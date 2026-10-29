"""Generate s0 for each task with BenchFlow self-gen + Anthropic skill-creator (Qwen as creator).

  python3 pipeline/gen_s0.py --concurrency 2

The creator scene sees only skill-creator and the task prompt (no tests/solution); BenchFlow exports
the generated skill packs, which are copied to runs/skills/s0/<task>/. The solver scene of the same
trial is a free extra run; s0's 10-run success rate is measured separately with eval_group.py.
"""
import argparse
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CREATOR = ROOT / "anthropic-skills/skills/skill-creator"


def env():
    e = dict(os.environ)
    for line in (Path.home() / "workspace/jarvis/.secrets/asu_llm.env").read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            e[k.strip()] = v.strip()
    return e


def gen(task, model):
    jobs = ROOT / f"runs/gen/s0/{task}"
    out = ROOT / f"runs/skills/s0/{task}"
    if not list(out.glob("*/SKILL.md")):
        e = env()
        cmd = ["bench", "eval", "run", "--tasks-dir", str(ROOT / f"skillsbench/tasks/{task}"),
               "--jobs-dir", str(jobs), "--agent", "qwen-code", "--model", f"vllm/{model}",
               "--agent-env", f"BENCHFLOW_PROVIDER_BASE_URL={e['OPENAI_BASE_URL']}",
               "--skill-mode", "self-gen", "--skill-creator-dir", str(CREATOR), "--sandbox", "docker"]
        jobs.mkdir(parents=True, exist_ok=True)
        with open(jobs / "gen.log", "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=e)
        packs = sorted(jobs.glob("**/generated-skills/*/SKILL.md"))
        if packs:
            out.mkdir(parents=True, exist_ok=True)
            for p in packs:
                shutil.copytree(p.parent, out / p.parent.name, dirs_exist_ok=True)
    n = len(list(out.glob("*/SKILL.md")))
    print(f"[s0] {task}: {n} skill(s)", flush=True)
    return task, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(ROOT / "pipeline/tasks.txt"))
    ap.add_argument("--model", default="qwen36-35b-a3b")
    ap.add_argument("--concurrency", type=int, default=2)
    a = ap.parse_args()
    tasks = [t.strip() for t in open(a.tasks) if t.strip() and not t.startswith("#")]
    with ThreadPoolExecutor(a.concurrency) as pool:
        res = list(pool.map(lambda t: gen(t, a.model), tasks))
    missing = [t for t, n in res if n == 0]
    print("missing s0:", missing or "none")


if __name__ == "__main__":
    main()
