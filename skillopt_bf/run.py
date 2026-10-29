"""Run SkillOpt on one SkillsBench task (BenchFlow rollouts, Qwen optimizer) and export the chain.

  .venv/bin/python skillopt_bf/run.py --task earthquake-plate-calculation [--k 5] [--sel 5] [--target 4]

SkillOpt's own gate decides each step (hard success on S fresh selection runs, strict improvement
over the current version). One step per epoch; a rejected step is retried with fresh rollouts, at
most --max-attempts times in a row, after which the chain stops (no force-accept).

Outputs under runs/skillopt/<task>/ (SkillOpt out_root; resumable):
  versions.json            provenance of every attempt (accepted / rejected / skipped)
  val_s<k>.csv             the S selection runs of accepted version s_k (s0 = baseline) = its eval runs
  val_s<k>_rej<j>.csv      selection runs of the j-th rejected candidate for s_k
  train_s<k>_att<j>.csv    the K training rollouts of s_k used by attempt j (extra runs of s_k)
Skill packs: runs/skills/s<k>/<task>/<skill>/ and runs/skills/s<k>_rej<j>/<task>/<skill>/.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILLOPT = ROOT / "SkillOpt"
sys.path.insert(0, str(ROOT))
from skillopt_bf.adapter import ACCEPT  # noqa: E402
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
  num_epochs: {epochs}     # = target x max_attempts; one step per epoch, StopChain ends it early
  train_size: {k}
  batch_size: {k}
  accumulation: 1
  seed: 42
gradient:
  minibatch_size: {k}
  merge_batch_size: 8
  analyst_workers: 4
optimizer:
  use_slow_update: false   # slow update adds a second, force-accepted edit per epoch -> breaks the chain
  slow_update_samples: {k} # meta-skill comparison batch = K items (served from finished runs of those skills)
  use_meta_skill: true
evaluation:
  use_gate: true           # SkillOpt gate: candidate hard score must strictly beat the current one
  gate_metric: hard
  sel_env_num: {sel}
  test_env_num: 0
  eval_test: false
env:
  name: skillsbench_bf
  skill_init: {skill_init}
  out_root: {out_root}
  task: {task}
  k: {k}
  s: {sel}
  target: {target}
  max_attempts: {max_attempts}
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


def _copy_csv(src, dest):
    if src.exists():
        shutil.copy(src, dest)
        return str(dest)
    return None


def export(task, out_root, pack, skills_out):
    """Export every attempt of the chain (accepted, rejected, skipped) with its rewards."""
    history = json.loads((out_root / "history.json").read_text()) if (out_root / "history.json").exists() else []
    _copy_csv(out_root / "selection_eval_baseline/rewards.csv", out_root / "val_s0.csv")
    records, v, attempt = [], 0, 0
    for h in history:
        step = h["step"]
        sdir = out_root / f"steps/step_{step:04d}"
        attempt += 1
        _copy_csv(sdir / "rollout/rewards.csv", out_root / f"train_s{v}_att{attempt}.csv")
        rec = {"step": step, "epoch": h.get("epoch"), "parent": f"s{v}", "attempt": attempt,
               "action": h.get("action"), "selection_hard": h.get("selection_hard"),
               "selection_soft": h.get("selection_soft"), "gate_score_after": h.get("current_score")}
        cand = sdir / "candidate_skill.md"
        if h.get("action") in ACCEPT:
            v, attempt, name = v + 1, 0, f"s{v + 1}"
            rec["val_csv"] = _copy_csv(sdir / "selection_eval/rewards.csv", out_root / f"val_{name}.csv")
        elif cand.exists() and h.get("action") == "reject":
            name = f"s{v + 1}_rej{attempt}"
            rec["val_csv"] = _copy_csv(sdir / "selection_eval/rewards.csv", out_root / f"val_{name}.csv")
        else:
            name = None  # skip_no_patches / skip_no_rewrite: no candidate produced
        if name:
            dest = skills_out / name / task / pack.name
            shutil.copytree(pack, dest, dirs_exist_ok=True)
            (dest / "SKILL.md").write_text(cand.read_text())
            rec.update({"version": name, "source": str(cand), "dest": str(dest)})
        records.append(rec)
    (out_root / "versions.json").write_text(json.dumps(records, indent=2))
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--k", type=int, default=5, help="training rollouts per step")
    ap.add_argument("--sel", type=int, default=5, help="selection (gate) runs per candidate")
    ap.add_argument("--target", type=int, default=4, help="accepted versions to produce (s1..s<target>)")
    ap.add_argument("--max-attempts", type=int, default=3, help="consecutive rejected attempts before stopping")
    ap.add_argument("--workers", type=int, default=0, help="parallel rollouts (default max(K, S))")
    ap.add_argument("--out-root")
    ap.add_argument("--skills-out", default=str(ROOT / "runs/skills"), help="where exported skill packs go")
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
                                     epochs=a.target * a.max_attempts, k=a.k, sel=a.sel, target=a.target,
                                     max_attempts=a.max_attempts, skill_init=packs[0], out_root=out_root,
                                     task=a.task, skill_pack=pack, workers=a.workers or max(a.k, a.sel)))
        sys.path[:0] = [str(SKILLOPT / "scripts"), str(ROOT)]
        import train  # SkillOpt/scripts/train.py
        from skillopt_bf.adapter import BenchFlowAdapter, StopChain

        train._register_builtins()
        train._register_builtins = lambda: None
        train._ENV_REGISTRY["skillsbench_bf"] = BenchFlowAdapter
        sys.argv = ["train.py", "--config", str(cfg)]
        try:
            train.main()
        except StopChain as e:
            print(f"[chain] stopped: {e}")

    for r in export(a.task, out_root, pack, Path(a.skills_out).resolve()):
        print(f"[export] step {r['step']} {r['action']}: {r.get('version', '-')} "
              f"(sel hard={r['selection_hard']})")


if __name__ == "__main__":
    main()
