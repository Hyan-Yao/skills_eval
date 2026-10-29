"""SkillOpt environment adapter that rolls out SkillsBench tasks through BenchFlow.

One SkillOpt run optimizes the skill of a single task. Its "dataset" is K copies of that
task (``<task>__r<i>``); every rollout is a fresh BenchFlow trial (qwen-code agent, Docker)
with the candidate SKILL.md mounted alongside the rest of the s0 skill pack.

Rollouts are cached by (sha256(skill), item id), so the meta-skill comparison at an epoch
boundary reuses the step rollouts of the same skill instead of re-running Docker trials.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from skillopt.datasets.base import BaseDataLoader, BatchSpec
from skillopt.envs.base import EnvAdapter

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "skillsbench/tasks"
OBS_CHARS, ARGS_CHARS, TEXT_CHARS, CONV_CHARS = 2000, 1500, 1500, 30000


def task_body(task):
    """task.md without its YAML frontmatter."""
    text = (TASKS / task / "task.md").read_text()
    return re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S).strip()


def _clip(s, n):
    s = "" if s is None else (s if isinstance(s, str) else json.dumps(s, ensure_ascii=False))
    return s if len(s) <= n else s[: n // 2] + f"\n…[{len(s) - n} chars truncated]…\n" + s[-n // 2:]


def _text(content):
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
    return content or ""


def trajectory_records(rollout_dir: Path) -> list[dict]:
    """BenchFlow llm_trajectory.jsonl -> SkillOpt tool_call / message records (clipped)."""
    path = rollout_dir / "trajectory/llm_trajectory.jsonl"
    if not path.exists():
        return []
    last = None
    for line in path.open():
        if line.strip():
            last = json.loads(line)
    if last is None:
        return []
    msgs = list(last["request"]["body"].get("messages", []))
    choice = ((last.get("response") or {}).get("body") or {}).get("choices") or [{}]
    if choice[0].get("message"):
        msgs.append(choice[0]["message"])
    obs = {m.get("tool_call_id"): _text(m.get("content")) for m in msgs if m.get("role") == "tool"}
    recs = []
    for m in msgs[2:]:  # skip system prompt and the first user turn (harness reminders + task)
        if m.get("role") != "assistant":
            continue
        if _text(m.get("content")).strip():
            recs.append({"role": "assistant", "content": _clip(_text(m["content"]), TEXT_CHARS)})
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            recs.append({"type": "tool_call",
                         "cmd": f"{fn.get('name')}({_clip(fn.get('arguments'), ARGS_CHARS)})",
                         "obs": _clip(obs.get(tc.get("id"), ""), OBS_CHARS)})
    total = sum(len(json.dumps(r)) for r in recs)
    if total > CONV_CHARS:  # keep the opening and the ending of long runs
        head, tail, size = [], [], 0
        for r in recs:
            if size > CONV_CHARS * 0.3:
                break
            head.append(r)
            size += len(json.dumps(r))
        for r in reversed(recs[len(head):]):
            if size > CONV_CHARS:
                break
            tail.insert(0, r)
            size += len(json.dumps(r))
        skipped = len(recs) - len(head) - len(tail)
        recs = head + [{"role": "system", "content": f"[{skipped} middle steps omitted]"}] + tail
    return recs


def verifier_summary(rollout_dir: Path) -> str:
    """Failed test names and assertion headlines from the verifier (no test source)."""
    out = []
    ctrf = rollout_dir / "verifier/ctrf.json"
    if ctrf.exists():
        res = json.loads(ctrf.read_text())["results"]
        s = res["summary"]
        out.append(f"{s['passed']}/{s['tests']} verifier tests passed")
        stdout = rollout_dir / "verifier/test-stdout.txt"
        if stdout.exists():
            out += [l.strip() for l in stdout.read_text().splitlines() if l.startswith("FAILED ")][:10]
    return "\n".join(out)


class TaskCopiesLoader(BaseDataLoader):
    """K identical items for one task; every batch is the full item list."""

    def __init__(self, task: str, k: int):
        self.train_items = [{"id": f"{task}__r{i}", "task": task, "task_type": task} for i in range(k)]
        self.val_items, self.test_items = [], []

    def get_train_size(self):
        return len(self.train_items)

    def build_train_batch(self, batch_size, seed, **kwargs):
        return BatchSpec(phase="train", split="train", seed=seed,
                         batch_size=len(self.train_items), payload=list(self.train_items))

    def build_eval_batch(self, env_num, split, seed, **kwargs):
        return BatchSpec(phase="eval", split=split, seed=seed, batch_size=0, payload=[])


class BenchFlowAdapter(EnvAdapter):
    def __init__(self, task: str = "", k: int = 4, skill_pack: str = "", workers: int = 4,
                 analyst_workers: int = 4, failure_only: bool = False, minibatch_size: int = 4,
                 edit_budget: int = 4) -> None:
        self.task, self.k, self.workers = task, int(k), int(workers)
        self.skill_pack = Path(skill_pack)
        self.analyst_workers, self.failure_only = analyst_workers, failure_only
        self.minibatch_size, self.edit_budget = minibatch_size, edit_budget
        self.dataloader = TaskCopiesLoader(task, self.k)
        self.body = task_body(task)

    def setup(self, cfg):
        super().setup(cfg)
        self.out_root = Path(cfg["out_root"])
        self.cache = self.out_root / "rollout_cache"

    def get_dataloader(self):
        return self.dataloader

    def build_train_env(self, batch_size, seed, **kwargs):
        return list(self.dataloader.build_train_batch(batch_size, seed).payload)

    def build_eval_env(self, env_num, split, seed, **kwargs):
        return []

    def get_task_types(self):
        return [self.task]

    # ── rollout ────────────────────────────────────────────────────────────

    def rollout(self, env_manager, skill_content, out_dir, **kwargs):
        items = list(env_manager or [])
        out = Path(out_dir)
        (out / "predictions").mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256(skill_content.encode()).hexdigest()[:16]
        with ThreadPoolExecutor(self.workers) as pool:
            results = list(pool.map(lambda it: self._one(it, skill_content, h, out), items))
        (out / "rollouts.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
        with open(out / "rewards.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "skill_hash", "reward", "hard", "error", "rollout_dir"])
            for r in results:
                w.writerow([r["id"], h, r["soft"], r["hard"], r.get("error", ""), r.get("rollout_dir", "")])
        return results

    def _one(self, item, skill_content, h, out):
        iid = item["id"]
        cdir = self.cache / h / iid
        if not (cdir / "result.json").exists():
            self._run(skill_content, h, cdir)
        res = json.loads((cdir / "result.json").read_text())
        pdir = out / "predictions" / iid
        pdir.mkdir(parents=True, exist_ok=True)
        shutil.copy(cdir / "conversation.json", pdir / "conversation.json")
        return {**res, "id": iid}

    def _run(self, skill_content, h, cdir):
        cdir.mkdir(parents=True, exist_ok=True)
        pack_root = self.cache / h / "skills"
        if not pack_root.exists():
            tmp = self.cache / h / f"skills.tmp{cdir.name}"
            shutil.copytree(self.skill_pack, tmp / self.skill_pack.name, dirs_exist_ok=True)
            (tmp / self.skill_pack.name / "SKILL.md").write_text(skill_content)
            try:
                tmp.rename(pack_root)
            except OSError:  # another thread won the race
                shutil.rmtree(tmp, ignore_errors=True)
        jobs = cdir / "jobs"
        with open(cdir / "bench.log", "w") as f:
            subprocess.run([str(ROOT / "pipeline/run_task.sh"), self.task, str(pack_root), str(jobs)],
                           stdout=f, stderr=subprocess.STDOUT)
        reward, rollout_dir, err = None, None, ""
        rows = [json.loads(l) for p in sorted(jobs.glob("*/results.jsonl")) for l in p.open() if l.strip()]
        if rows:
            r = rows[-1]
            reward = r.get("reward")
            rollout_dir = Path((r.get("info") or {}).get("rollout_dir") or "")
            err = r.get("error") or ""
        if reward is None:  # infra error: no verifier reward
            res = {"hard": 0, "soft": 0.0, "error": err or "no reward", "fail_reason": f"infra error: {err}"}
            conv = []
        else:
            reward = float(reward)
            summary = verifier_summary(rollout_dir) if rollout_dir else ""
            conv = trajectory_records(rollout_dir) if rollout_dir else []
            conv.append({"role": "system", "content": f"Verifier reward {reward:.3f}. {summary}"})
            res = {"hard": int(reward >= 1.0), "soft": reward, "error": "" if not err else str(err)[:300],
                   "fail_reason": "" if reward >= 1.0 else summary or f"reward {reward:.3f}"}
        res.update({"task_description": self.body, "task_type": self.task, "skill_hash": h,
                    "rollout_dir": str(rollout_dir or ""), "n_turns": sum(1 for c in conv if c.get("type") == "tool_call")})
        (cdir / "conversation.json").write_text(json.dumps(conv, ensure_ascii=False, indent=1))
        (cdir / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
