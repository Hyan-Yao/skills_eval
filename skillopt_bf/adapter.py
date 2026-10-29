"""SkillOpt environment adapter that rolls out SkillsBench tasks through BenchFlow.

One SkillOpt run optimizes the skill of a single task. Its "dataset" is K copies of that
task for training (``<task>__r<i>``) and S copies for the selection gate (``<task>__v<i>``);
every rollout is a fresh BenchFlow trial (qwen-code agent, Docker) with the candidate SKILL.md
mounted alongside the rest of the s0 skill pack.

Rollouts are cached under (sha256(skill), tag). Step training rollouts and selection
evaluations get their own tag (always fresh runs, so a retried step sees new trajectories);
any other rollout of an already-run skill (the meta-skill comparison at an epoch boundary)
reuses existing runs of that skill instead of starting new Docker trials.

Chain control: the trainer runs one step per epoch. Before a step's training rollout the
adapter reads history.json and raises StopChain once ``target`` versions were accepted or
``max_attempts`` consecutive attempts were rejected.
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


class StopChain(Exception):
    """Raised to end the SkillOpt run once the version chain is complete or stuck."""


ACCEPT = {"accept", "accept_new_best", "force_accept"}


def chain_state(history):
    """(accepted versions, consecutive failed attempts since the last acceptance)."""
    accepted = streak = 0
    for h in history:
        if h.get("action") in ACCEPT:
            accepted, streak = accepted + 1, 0
        else:
            streak += 1
    return accepted, streak


class TaskCopiesLoader(BaseDataLoader):
    """K training and S selection copies of one task; every batch is the full item list."""

    def __init__(self, task: str, k: int, s: int = 0):
        self.train_items = [{"id": f"{task}__r{i}", "task": task, "task_type": task} for i in range(k)]
        self.val_items = [{"id": f"{task}__v{i}", "task": task, "task_type": task} for i in range(s)]
        self.test_items = []

    def get_train_size(self):
        return len(self.train_items)

    def build_train_batch(self, batch_size, seed, **kwargs):
        return BatchSpec(phase="train", split="train", seed=seed,
                         batch_size=len(self.train_items), payload=list(self.train_items))

    def build_eval_batch(self, env_num, split, seed, **kwargs):
        items = list(self.val_items) if split in ("valid_seen", "selection", "val") else []
        return BatchSpec(phase="eval", split=split, seed=seed, batch_size=len(items), payload=items)


class BenchFlowAdapter(EnvAdapter):
    def __init__(self, task: str = "", k: int = 5, s: int = 5, target: int = 4, max_attempts: int = 3,
                 skill_pack: str = "", workers: int = 5, analyst_workers: int = 4,
                 failure_only: bool = False, minibatch_size: int = 5, edit_budget: int = 4) -> None:
        self.task, self.k, self.workers = task, int(k), int(workers)
        self.target, self.max_attempts = int(target), int(max_attempts)
        self.skill_pack = Path(skill_pack)
        self.analyst_workers, self.failure_only = analyst_workers, failure_only
        self.minibatch_size, self.edit_budget = minibatch_size, edit_budget
        self.dataloader = TaskCopiesLoader(task, self.k, int(s))
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
        return list(self.dataloader.build_eval_batch(env_num, split, seed).payload)

    def get_task_types(self):
        return [self.task]

    # ── rollout ────────────────────────────────────────────────────────────

    def _tag(self, out: Path):
        """Fresh-run tag for step training / selection / baseline rollouts, None for reusable ones."""
        parts = out.parts
        if "steps" in parts:
            step = parts[parts.index("steps") + 1]
            if out.name == "rollout":
                return f"{step}_train"
            if out.name == "selection_eval":
                return f"{step}_sel"
        if out.name == "selection_eval_baseline":
            return "baseline_sel"
        return None

    def _check_chain(self):
        hist = self.out_root / "history.json"
        accepted, streak = chain_state(json.loads(hist.read_text()) if hist.exists() else [])
        if accepted >= self.target or streak >= self.max_attempts:
            raise StopChain(f"accepted={accepted} consecutive_failed={streak}")

    def rollout(self, env_manager, skill_content, out_dir, **kwargs):
        items = list(env_manager or [])
        out = Path(out_dir)
        tag = self._tag(out)
        if tag and tag.endswith("_train"):
            self._check_chain()
        (out / "predictions").mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256(skill_content.encode()).hexdigest()[:16]
        if tag:
            dirs = [self.cache / h / tag / it["id"] for it in items]
        else:  # reuse any finished runs of this skill, run the shortfall fresh
            done = sorted(p.parent for p in (self.cache / h).glob("*/*/result.json"))
            dirs = (done + [self.cache / h / "extra" / it["id"] for it in items])[: len(items)]
        with ThreadPoolExecutor(self.workers) as pool:
            results = list(pool.map(lambda p: self._one(p[0], p[1], skill_content, h, out), zip(items, dirs)))
        (out / "rollouts.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
        with open(out / "rewards.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "skill_hash", "reward", "hard", "error", "rollout_dir", "cache_dir"])
            for r in results:
                w.writerow([r["id"], h, r["soft"], r["hard"], r.get("error", ""), r.get("rollout_dir", ""), r["cache_dir"]])
        return results

    def _one(self, item, cdir, skill_content, h, out):
        iid = item["id"]
        if not (cdir / "result.json").exists():
            self._run(skill_content, h, cdir)
        res = json.loads((cdir / "result.json").read_text())
        res["cache_dir"] = str(cdir)
        pdir = out / "predictions" / iid
        pdir.mkdir(parents=True, exist_ok=True)
        shutil.copy(cdir / "conversation.json", pdir / "conversation.json")
        return {**res, "id": iid}

    def _run(self, skill_content, h, cdir):
        cdir.mkdir(parents=True, exist_ok=True)
        pack_root = self.cache / h / "skills"
        if not pack_root.exists():
            tmp = self.cache / h / f"skills.tmp-{cdir.parent.name}-{cdir.name}"
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
