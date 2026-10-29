"""LLM-as-a-judge for agent skills (rubrics in eval/rubrics/).

The judge sees the task instruction and the full skill package, never the success rate or version id.
Each (skill, rubric) is judged n times; the per-dimension score is the median over samples.

  python eval/judge.py --rubric v2 --skill-dir S --task-dir T --task-id t --version-id s0
  python eval/judge.py --rubric all --manifest manifest.csv --out runs/judge.jsonl --workers 8
Manifest columns: task_id, version_id, skill_dir, task_dir.
Credentials: OPENAI_BASE_URL / OPENAI_API_KEY from the environment or --env-file.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUBRICS = {"v1": "v1_holistic.md", "v2": "v2_gap.md", "skillnet": "skillnet.md"}
TEXT_EXT = {".md", ".txt", ".py", ".sh", ".json", ".yaml", ".yml", ".toml", ".js", ".ts", ".r", ".sql",
            ".csv", ".xml", ".html", ".css", ".ini", ".cfg", ".tex", ".java", ".c", ".cpp", ".h", ".go", ".rs"}
MAX_FILE_CHARS, MAX_TOTAL_CHARS = 15_000, 60_000


def load_rubric(name):
    text = (ROOT / "rubrics" / RUBRICS[name]).read_text()
    meta = json.loads(re.match(r"<!-- rubric: (\{.*?\}) -->", text).group(1))
    return meta, text.split("-->", 1)[1].strip()


def read_instruction(task_dir):
    task_dir = Path(task_dir)
    if (task_dir / "instruction.md").exists():
        return (task_dir / "instruction.md").read_text().strip()
    text = (task_dir / "task.md").read_text()
    m = re.match(r"---\n.*?\n---[ \t]*\n", text, re.S)  # drop YAML frontmatter (metadata, verifier config)
    return text[m.end() if m else 0:].strip()


def render_skill(skill_dir):
    """Concatenate the skill package: every SKILL.md first, then other text files, with truncation notes."""
    skill_dir = Path(skill_dir)
    files = [p for p in skill_dir.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    files.sort(key=lambda p: (p.name != "SKILL.md", str(p.relative_to(skill_dir))))
    parts, total = [], 0
    for p in files:
        rel = p.relative_to(skill_dir)
        if p.suffix.lower() not in TEXT_EXT and p.name != "SKILL.md":
            parts.append(f"=== {rel} === [binary or non-text file, {p.stat().st_size} bytes, not shown]")
            continue
        body = p.read_text(errors="replace")
        if len(body) > MAX_FILE_CHARS:
            body = body[:MAX_FILE_CHARS] + f"\n[... truncated: file has {len(body)} chars ...]"
        if total + len(body) > MAX_TOTAL_CHARS:
            parts.append(f"=== {rel} === [omitted: package exceeds {MAX_TOTAL_CHARS} chars]")
            continue
        total += len(body)
        parts.append(f"=== {rel} ===\n{body}")
    return "\n\n".join(parts)


def build_messages(rubric_text, instruction, skill_text):
    user = (f"## Task instruction\n\n{instruction}\n\n## Skill package\n\n{skill_text}\n\n"
            "Audit the skill package with the rubric. Return only the JSON object.")
    return [{"role": "system", "content": rubric_text}, {"role": "user", "content": user}]


def extract_json(text):
    """Return the last top-level JSON object in text (handles code fences and leading reasoning)."""
    objs, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"' and depth:
            in_str = True
        elif ch == "{":
            start = i if depth == 0 else start
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                objs.append(text[start:i + 1])
    for s in reversed(objs):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON object found")


def parse_scores(obj, meta):
    """Validate one judge response; return {dim: number} and {dim: {evidence, rationale}}."""
    dims = obj.get("dimensions", obj)
    lo, hi = meta["scale"]
    labels = meta.get("labels", {})
    scores, notes = {}, {}
    for d in meta["dims"]:
        entry = dims[d]
        s = entry["score"] if isinstance(entry, dict) else entry
        s = labels.get(s.strip().capitalize(), s) if isinstance(s, str) else s
        s = float(s)
        if not lo <= s <= hi:
            raise ValueError(f"{d}: score {s} outside {lo}-{hi}")
        scores[d] = s
        notes[d] = {k: entry.get(k, "") for k in ("evidence", "rationale")} if isinstance(entry, dict) else {}
    return scores, notes


class Client:
    """Minimal OpenAI-compatible chat client; `transport(url, headers, body) -> dict` is injectable for tests."""

    def __init__(self, model, base_url=None, api_key=None, temperature=0.7, reasoning_effort=None,
                 max_tokens=8192, transport=None):
        self.model, self.temperature, self.reasoning_effort, self.max_tokens = model, temperature, reasoning_effort, max_tokens
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.transport = transport or self._http

    @staticmethod
    def _http(url, headers, body):
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)

    def chat(self, messages, seed=None):
        body = {"model": self.model, "messages": messages, "temperature": self.temperature,
                "max_tokens": self.max_tokens}
        if seed is not None:
            body["seed"] = seed
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        for attempt in range(4):
            try:
                resp = self.transport(f"{self.base_url}/chat/completions", headers, body)
                return resp["choices"][0]["message"].get("content") or ""
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)


def judge_one(client, rubric, skill_dir, task_dir, n=3, cache_dir=None, parse_retries=3):
    meta, rubric_text = load_rubric(rubric)
    instruction, skill_text = read_instruction(task_dir), render_skill(skill_dir)
    messages = build_messages(rubric_text, instruction, skill_text)
    skill_hash = hashlib.sha256(skill_text.encode()).hexdigest()[:16]
    samples, notes = [], []
    for i in range(n):
        key = hashlib.sha256(json.dumps([skill_text, instruction, rubric_text, client.model,
                                         client.temperature, client.reasoning_effort, i]).encode()).hexdigest()
        path = Path(cache_dir) / f"{key}.json" if cache_dir else None
        if path and path.exists():
            raw = json.loads(path.read_text())["raw"]
        else:
            for attempt in range(parse_retries):
                raw = client.chat(messages, seed=i * 1000 + attempt)
                try:
                    parse_scores(extract_json(raw), meta)
                    break
                except (ValueError, KeyError, TypeError):
                    if attempt == parse_retries - 1:
                        raise RuntimeError(f"unparseable judge output for {skill_dir} ({rubric}): {raw[:300]}")
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"raw": raw}))
        s, nt = parse_scores(extract_json(raw), meta)
        samples.append(s)
        notes.append(nt)
    scores = {d: statistics.median(s[d] for s in samples) for d in meta["dims"]}
    return {"rubric": rubric, "model": client.model, "n": n, "skill_hash": skill_hash, "scores": scores,
            "aggregate": statistics.mean(scores[d] for d in meta["aggregate"]),
            "samples": samples, "notes": notes}


def load_env_file(path):
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip().removeprefix("export "), v.strip().strip("'\""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rubric", choices=[*RUBRICS, "all"], default="all")
    ap.add_argument("--skill-dir"), ap.add_argument("--task-dir")
    ap.add_argument("--task-id", default=""), ap.add_argument("--version-id", default="")
    ap.add_argument("--manifest", help="CSV with task_id, version_id, skill_dir, task_dir")
    ap.add_argument("--out", help="JSONL output (appended); stdout if omitted")
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--env-file")
    ap.add_argument("--cache-dir", default=str(ROOT.parent / "runs" / "judge_cache"))
    args = ap.parse_args(argv)
    if args.env_file:
        load_env_file(args.env_file)

    if args.manifest:
        with open(args.manifest) as f:
            jobs = list(csv.DictReader(f))
    elif args.skill_dir and args.task_dir:
        jobs = [{"task_id": args.task_id, "version_id": args.version_id,
                 "skill_dir": args.skill_dir, "task_dir": args.task_dir}]
    else:
        ap.error("give --manifest or --skill-dir and --task-dir")
    rubrics = list(RUBRICS) if args.rubric == "all" else [args.rubric]
    client = Client(args.model, temperature=args.temperature, reasoning_effort=args.reasoning_effort)
    out = open(args.out, "a") if args.out else sys.stdout

    def run(job_rubric):
        job, rubric = job_rubric
        try:
            row = judge_one(client, rubric, job["skill_dir"], job["task_dir"], n=args.n, cache_dir=args.cache_dir)
        except Exception as e:  # keep the batch going; failures are reported in the output
            row = {"rubric": rubric, "model": args.model, "error": str(e)}
        return {"task_id": job["task_id"], "version_id": job["version_id"], **row}

    with ThreadPoolExecutor(args.workers) as pool:
        for row in pool.map(run, [(j, r) for j in jobs for r in rubrics]):
            out.write(json.dumps(row) + "\n")
            out.flush()
            if "error" in row:
                print(f"[judge] {row['task_id']}/{row['version_id']} {row['rubric']}: {row['error']}", file=sys.stderr)
    if out is not sys.stdout:
        out.close()


if __name__ == "__main__":
    main()
