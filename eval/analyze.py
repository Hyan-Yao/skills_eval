"""Relate judge scores to measured success across skill versions.

  python eval/analyze.py --judge runs/judge.jsonl --success runs/success.csv --out runs/analysis
Success CSV columns: task_id, version_id, n_runs, n_success.
Outputs: summary.csv (within-task / pairwise / levels per metric, with bootstrap CIs), within_task.csv,
trajectories.csv, and figures if matplotlib is installed.

Somers' D is D(success | score): (concordant - discordant) / pairs not tied on score.
"""
import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

VERSION_ORDER = ["s0", "s1", "s2", "s3", "s4", "human"]


def version_key(v):
    return (VERSION_ORDER.index(v), v) if v in VERSION_ORDER else (len(VERSION_ORDER), v)


def load(judge_path, success_path):
    """Return {(task, version): {"success": rate, "n_runs", "n_success", metric: score}}."""
    data = {}
    with open(success_path) as f:
        for r in csv.DictReader(f):
            n, k = int(r["n_runs"]), int(r["n_success"])
            data[(r["task_id"], r["version_id"])] = {"success": k / n, "n_runs": n, "n_success": k}
    with open(judge_path) as f:
        for line in f:
            r = json.loads(line)
            key = (r["task_id"], r["version_id"])
            if "error" in r or key not in data:
                continue
            for d, s in r["scores"].items():
                data[key][f"{r['rubric']}.{d}"] = s
            data[key][f"{r['rubric']}.aggregate"] = r["aggregate"]
    return data


def metrics_of(data):
    ms = set()
    for v in data.values():
        ms |= {k for k in v if "." in k}
    return sorted(ms)


def sign(x):
    return (x > 0) - (x < 0)


def cd_counts(xs, ys):
    """Concordant minus discordant pairs, and pairs not tied on x."""
    num = den = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            sx = sign(xs[i] - xs[j])
            if sx:
                den += 1
                num += sx * sign(ys[i] - ys[j])
    return num, den


def somers_d(xs, ys):
    num, den = cd_counts(xs, ys)
    return num / den if den else float("nan")


def fisher_p(k1, n1, k2, n2):
    """Two-sided Fisher exact test on [[k1, n1-k1], [k2, n2-k2]]."""
    K, N = k1 + k2, n1 + n2
    pmf = lambda a: math.comb(n1, a) * math.comb(n2, K - a) / math.comb(N, K)
    p0 = pmf(k1)
    return min(1.0, sum(pmf(a) for a in range(max(0, K - n2), min(K, n1) + 1) if pmf(a) <= p0 * (1 + 1e-9)))


def by_task(data):
    tasks = defaultdict(list)
    for (t, v), row in data.items():
        tasks[t].append((v, row))
    return {t: sorted(rows, key=lambda r: version_key(r[0])) for t, rows in tasks.items()}


def within_task(tasks, metric):
    """Per-task D and pooled D (pairs only within tasks)."""
    per, num, den = {}, 0, 0
    for t, rows in tasks.items():
        rows = [r for _, r in rows if metric in r]
        n, d = cd_counts([r[metric] for r in rows], [r["success"] for r in rows])
        per[t] = n / d if d else float("nan")
        num, den = num + n, den + d
    return per, (num / den if den else float("nan"))


def pairwise_accuracy(tasks, metric, alpha):
    """Among within-task version pairs whose success differs (Fisher p < alpha), share ranked correctly by the
    score (score ties count 0.5)."""
    hits = total = 0
    for rows in tasks.values():
        rows = [r for _, r in rows if metric in r]
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                if a["success"] == b["success"] or fisher_p(a["n_success"], a["n_runs"], b["n_success"], b["n_runs"]) >= alpha:
                    continue
                total += 1
                s = sign(a[metric] - b[metric]) * sign(a["success"] - b["success"])
                hits += 1 if s > 0 else 0.5 if s == 0 else 0
    return (hits / total if total else float("nan")), total


def levels(data, metric, version):
    rows = [r for (t, v), r in data.items() if v == version and metric in r]
    return somers_d([r[metric] for r in rows], [r["success"] for r in rows]), len(rows)


def bootstrap(stat, tasks, n_boot, seed=0):
    """Percentile CI from resampling tasks with replacement."""
    rng, names, vals = random.Random(seed), list(tasks), []
    for _ in range(n_boot):
        sample = {f"{i}:{t}": tasks[t] for i, t in enumerate(rng.choice(names) for _ in names)}
        v = stat(sample)
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    vals.sort()
    return vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]


def trajectories(data, metrics):
    agg = defaultdict(lambda: defaultdict(list))
    for (t, v), r in data.items():
        for m in ["success", *metrics]:
            if m in r:
                agg[v][m].append(r[m])
    return [{"version_id": v, "metric": m, "mean": sum(xs) / len(xs), "n_tasks": len(xs)}
            for v in sorted(agg, key=version_key) for m, xs in agg[v].items()]


def plot(traj, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyze] matplotlib not installed; skipping figures")
        return
    versions = sorted({r["version_id"] for r in traj}, key=version_key)
    by_rubric = defaultdict(lambda: defaultdict(dict))
    for r in traj:
        rubric = r["metric"].split(".")[0] if "." in r["metric"] else "success"
        by_rubric[rubric][r["metric"]][r["version_id"]] = r["mean"]
    succ = by_rubric.pop("success", {}).get("success", {})
    for rubric, series in by_rubric.items():
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for m, pts in sorted(series.items()):
            ax.plot(versions, [pts.get(v, float("nan")) for v in versions], marker="o", label=m.split(".", 1)[1])
        ax2 = ax.twinx()
        ax2.plot(versions, [succ.get(v, float("nan")) for v in versions], "k--", marker="s", label="success")
        ax2.set_ylabel("success rate")
        ax.set_ylabel(f"{rubric} score")
        ax.legend(fontsize=7, loc="upper left", ncol=2)
        fig.tight_layout()
        fig.savefig(out / f"trajectory_{rubric}.png", dpi=150)
        plt.close(fig)


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--success", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.05, help="Fisher threshold for pairwise accuracy")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = load(args.judge, args.success)
    tasks, metrics = by_task(data), metrics_of(data)
    versions = sorted({v for _, v in data}, key=version_key)
    summary, per_task = [], []
    for m in metrics:
        per, pooled = within_task(tasks, m)
        lo, hi = bootstrap(lambda ts: within_task(ts, m)[1], tasks, args.n_boot)
        summary.append({"metric": m, "analysis": "within_task_D", "group": "pooled", "value": pooled,
                        "ci_lo": lo, "ci_hi": hi, "n": len(tasks)})
        per_task += [{"metric": m, "task_id": t, "D": d} for t, d in per.items()]
        acc, npairs = pairwise_accuracy(tasks, m, args.alpha)
        lo, hi = bootstrap(lambda ts: pairwise_accuracy(ts, m, args.alpha)[0], tasks, args.n_boot)
        summary.append({"metric": m, "analysis": "pairwise_accuracy", "group": f"p<{args.alpha}", "value": acc,
                        "ci_lo": lo, "ci_hi": hi, "n": npairs})
        for v in versions:
            d, n = levels(data, m, v)
            boot = lambda ts: levels({(t, vv): r for t, rows in ts.items() for vv, r in rows}, m, v)[0]
            lo, hi = bootstrap(boot, tasks, args.n_boot)
            summary.append({"metric": m, "analysis": "levels_D", "group": v, "value": d,
                            "ci_lo": lo, "ci_hi": hi, "n": n})
    traj = trajectories(data, metrics)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "within_task.csv", per_task)
    write_csv(out / "trajectories.csv", traj)
    plot(traj, out)
    print(f"[analyze] {len(tasks)} tasks, {len(versions)} versions, {len(metrics)} metrics -> {out}")


if __name__ == "__main__":
    main()
