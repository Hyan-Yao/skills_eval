"""Run SkillOpt on one SkillsBench task (BenchFlow rollouts, Qwen optimizer) and export s1..sE.

  .venv/bin/python skillopt_bf/run.py --task earthquake-plate-calculation [--k 5] [--epochs 4]

Outputs: runs/skillopt/<task>/ (SkillOpt out_root; resumable), runs/skills/s<k>/<task>/<skill>/
(full skill pack per version), runs/skillopt/<task>/rollouts_step<k>.csv (the K rollouts of s_{k-1}
used for step k), runs/skillopt/<task>/versions.json (provenance).
"""
import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILLOPT = ROOT / "SkillOpt"
MODEL = "qwen36-35b-a3b"

CONFIG = """_base_: {base}
model:
  optimizer_backend: openai_compatible
  target_backend: openai_compatible
  optimizer: {model}
  target: {model}
  reasoning_effort: ""
  rewrite_max_completion_tokens: 16384
train:
  num_epochs: {epochs}
  train_size: {k}
  batch_size: {k}
  accumulation: 1
  seed: 42
gradient:
  minibatch_size: {k}
  merge_batch_size: 8
  analyst_workers: 4
optimizer:
  use_slow_update: false   # slow update adds a second skill edit per epoch -> breaks the s1..sE chain
  slow_update_samples: {k} # meta-skill comparison batch = the same K items (served from rollout cache)
  use_meta_skill: true
evaluation:
  use_gate: false          # every step's candidate becomes the next version
  sel_env_num: 0
  test_env_num: 0
  eval_test: false
env:
  name: skillsbench_bf
  skill_init: {skill_init}
  out_root: {out_root}
  task: {task}
  k: {k}
  skill_pack: {skill_pack}
  workers: {workers}
"""


def load_env():
    for line in (Path.home() / "workspace/jarvis/.secrets/asu_llm.env").read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    os.environ["OPENAI_COMPATIBLE_BASE_URL"] = os.environ["OPENAI_BASE_URL"]
    os.environ["OPENAI_COMPATIBLE_API_KEY"] = os.environ["OPENAI_API_KEY"]
    os.environ["OPENAI_COMPATIBLE_MODEL"] = MODEL
    os.environ.setdefault("OPENAI_COMPATIBLE_MAX_TOKENS", "16384")
    os.environ.setdefault("OPENAI_COMPATIBLE_TIMEOUT_SECONDS", "900")


def export(task, out_root, pack):
    """Copy every saved skill version into runs/skills/s<k>/<task>/<skill>/ and record provenance."""
    history = json.loads((out_root / "history.json").read_text()) if (out_root / "history.json").exists() else []
    by_step = {h.get("step"): h for h in history}
    versions = []
    for f in sorted((out_root / "skills").glob("skill_v*.md")):
        k = int(f.stem.split("_v")[1])
        if k == 0:
            continue
        dest = ROOT / f"runs/skills/s{k}/{task}/{pack.name}"
        shutil.copytree(pack, dest, dirs_exist_ok=True)
        (dest / "SKILL.md").write_text(f.read_text())
        h = by_step.get(k, {})
        versions.append({"version": f"s{k}", "step": k, "epoch": h.get("epoch"),
                         "action": h.get("action") or h.get("gate_action"), "source": str(f), "dest": str(dest)})
    (out_root / "versions.json").write_text(json.dumps(versions, indent=2))
    # step k trains on rollouts of s_{k-1}; collect their rewards
    for k in range(1, len(versions) + 2):
        rows = []
        for p in sorted(out_root.glob(f"**/step_{k:04d}/**/rewards.csv")) + sorted(out_root.glob(f"**/step{k}/**/rewards.csv")):
            rows += list(csv.DictReader(open(p)))
        if rows:
            with open(out_root / f"rollouts_step{k}.csv", "w", newline="") as fo:
                w = csv.DictWriter(fo, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
    return versions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0, help="parallel rollouts (default K)")
    ap.add_argument("--out-root")
    ap.add_argument("--export-only", action="store_true")
    a = ap.parse_args()

    packs = sorted((ROOT / f"runs/skills/s0/{a.task}").glob("*/SKILL.md"))
    if len(packs) != 1:
        sys.exit(f"expected exactly one s0 skill pack for {a.task}, found {len(packs)}")
    pack = packs[0].parent
    out_root = Path(a.out_root or ROOT / f"runs/skillopt/{a.task}").resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not a.export_only:
        load_env()
        cfg = out_root / "config.yaml"
        cfg.write_text(CONFIG.format(base=SKILLOPT / "configs/_base_/default.yaml", model=MODEL,
                                     epochs=a.epochs, k=a.k, skill_init=packs[0], out_root=out_root,
                                     task=a.task, skill_pack=pack, workers=a.workers or a.k))
        sys.path[:0] = [str(SKILLOPT / "scripts"), str(ROOT)]
        import train  # SkillOpt/scripts/train.py
        from skillopt_bf.adapter import BenchFlowAdapter

        train._register_builtins()
        train._register_builtins = lambda: None
        train._ENV_REGISTRY["skillsbench_bf"] = BenchFlowAdapter
        sys.argv = ["train.py", "--config", str(cfg)]
        train.main()

    for v in export(a.task, out_root, pack):
        print(f"[export] {v['version']} <- {v['source']}")


if __name__ == "__main__":
    main()
