import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze  # noqa: E402
import judge  # noqa: E402


def make_task(root):
    task = root / "task"
    (task / "environment" / "skills" / "demo" / "scripts").mkdir(parents=True)
    (task / "task.md").write_text("---\nschema_version: '1.3'\nverifier:\n  type: test-script\n---\n\nCompute the mass.\n")
    skill = task / "environment" / "skills"
    (skill / "demo" / "SKILL.md").write_text("---\nname: demo\n---\n# Demo\nRun scripts/run.py.\n")
    (skill / "demo" / "scripts" / "run.py").write_text("print('hi')\n")
    (skill / "demo" / "blob.bin").write_bytes(b"\x00\x01")
    return task, skill


def fake_response(meta, score, wrap=True):
    dims = {d: {"evidence": "x {braces} \"quoted\"", "rationale": "r", "score": score} for d in meta["dims"]}
    text = json.dumps({"dimensions": dims})
    return f"Reasoning first.\n```json\n{text}\n```" if wrap else text


class JudgeTest(unittest.TestCase):
    def test_instruction_and_render(self):
        with tempfile.TemporaryDirectory() as d:
            task, skill = make_task(Path(d))
            self.assertEqual(judge.read_instruction(task), "Compute the mass.")
            text = judge.render_skill(skill)
            self.assertLess(text.index("SKILL.md"), text.index("run.py"))
            self.assertIn("not shown", text)

    def test_extract_json_last_object(self):
        self.assertEqual(judge.extract_json('a {"x": 1} b {"y": "}{"} c'), {"y": "}{"})
        with self.assertRaises(ValueError):
            judge.extract_json("no json")

    def test_skillnet_labels(self):
        meta, _ = judge.load_rubric("skillnet")
        scores, _ = judge.parse_scores(json.loads(fake_response(meta, "Average", wrap=False)), meta)
        self.assertEqual(set(scores.values()), {1.0})

    def test_out_of_range_rejected(self):
        meta, _ = judge.load_rubric("v2")
        with self.assertRaises(ValueError):
            judge.parse_scores(json.loads(fake_response(meta, 3, wrap=False)), meta)

    def test_judge_median_retry_and_cache(self):
        meta, _ = judge.load_rubric("v2")
        replies = ["garbage", fake_response(meta, 0), fake_response(meta, 2), fake_response(meta, 2)]
        calls = []

        def transport(url, headers, body):
            calls.append(body)
            self.assertNotIn("success", json.dumps(body["messages"]).lower())  # judge is blind to outcomes
            return {"choices": [{"message": {"content": replies[len(calls) - 1]}}]}

        with tempfile.TemporaryDirectory() as d:
            task, skill = make_task(Path(d))
            client = judge.Client("m", base_url="http://x/v1", api_key="k", transport=transport)
            row = judge.judge_one(client, "v2", skill, task, n=3, cache_dir=Path(d) / "cache")
            self.assertEqual(len(calls), 4)  # one parse retry
            self.assertEqual(row["scores"]["A_chain_closure"], 2)
            self.assertAlmostEqual(row["aggregate"], 2.0)
            again = judge.judge_one(client, "v2", skill, task, n=3, cache_dir=Path(d) / "cache")
            self.assertEqual(len(calls), 4)  # served from cache
            self.assertEqual(again["scores"], row["scores"])

    def test_cli_manifest(self):
        meta2, _ = judge.load_rubric("v2")
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            task, skill = make_task(d)
            with open(d / "m.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["task_id", "version_id", "skill_dir", "task_dir"])
                w.writerow(["t1", "s0", skill, task])
            orig = judge.Client._http
            judge.Client._http = staticmethod(lambda u, h, b: {"choices": [{"message": {"content": fake_response(meta2, 1)}}]})
            try:
                judge.main(["--rubric", "v2", "--manifest", str(d / "m.csv"), "--out", str(d / "o.jsonl"),
                            "--cache-dir", str(d / "c"), "--n", "2"])
            finally:
                judge.Client._http = orig
            rows = [json.loads(l) for l in open(d / "o.jsonl")]
            self.assertEqual(rows[0]["task_id"], "t1")
            self.assertEqual(rows[0]["scores"]["G_generality"], 1)


class AnalyzeTest(unittest.TestCase):
    def test_somers_and_fisher(self):
        self.assertEqual(analyze.somers_d([1, 2, 3], [0.1, 0.2, 0.3]), 1.0)
        self.assertEqual(analyze.somers_d([1, 2, 3], [0.3, 0.2, 0.1]), -1.0)
        self.assertAlmostEqual(analyze.fisher_p(10, 10, 0, 10), 1.0825088224469026e-05)
        self.assertAlmostEqual(analyze.fisher_p(5, 10, 5, 10), 1.0)

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            versions = ["s0", "s1", "s2", "s3", "s4", "human"]
            with open(d / "succ.csv", "w", newline="") as f, open(d / "j.jsonl", "w") as j:
                w = csv.writer(f)
                w.writerow(["task_id", "version_id", "n_runs", "n_success"])
                for t in range(4):
                    for i, v in enumerate(versions):
                        k = min(10, 2 * i + t % 2)
                        w.writerow([f"t{t}", v, 10, k])
                        good = {"A_chain_closure": min(2, i // 2), "G_generality": 2 - min(2, i // 2)}
                        j.write(json.dumps({"task_id": f"t{t}", "version_id": v, "rubric": "v2",
                                            "scores": good, "aggregate": sum(good.values()) / 2}) + "\n")
            analyze.main(["--judge", str(d / "j.jsonl"), "--success", str(d / "succ.csv"),
                          "--out", str(d / "out"), "--n-boot", "50"])
            rows = list(csv.DictReader(open(d / "out" / "summary.csv")))
            get = lambda m, a, g: float(next(r["value"] for r in rows if r["metric"] == m and r["analysis"] == a and r["group"] == g))
            self.assertEqual(get("v2.A_chain_closure", "within_task_D", "pooled"), 1.0)
            self.assertEqual(get("v2.G_generality", "within_task_D", "pooled"), -1.0)
            self.assertGreater(get("v2.A_chain_closure", "pairwise_accuracy", "p<0.05"), 0.5)
            traj = list(csv.DictReader(open(d / "out" / "trajectories.csv")))
            self.assertEqual(traj[0]["version_id"], "s0")


if __name__ == "__main__":
    unittest.main()
