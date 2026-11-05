"""End-to-end NVFLARE SimEnv runs (real training, real aux-channel requests).

Slow (~3-4 min). Run with:  pytest -q tests/integration
"""
import csv
import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable


def _run(algo, budget, tag):
    cmd = [PY, os.path.join(ROOT, "scripts/run_experiment.py"), "--profile", "smoke", f"--tag={tag}",
           "--only", f"_{algo}_s0$", "--set", f"budget_updates={budget}", "checkpoint_every=12"]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)


def _clean(rid):
    for d in ("results/metrics", "checkpoints", "workspaces/runs"):
        shutil.rmtree(os.path.join(ROOT, d, rid), ignore_errors=True)


def _rows(rid, name):
    with open(os.path.join(ROOT, "results/metrics", rid, name)) as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("algo", ["fedasmu", "fedbuff"])
def test_async_run_semantics(algo):
    rid = f"smoke-itest_cnn_cifar10_{algo}_s0"
    _clean(rid)
    r = _run(algo, 24, "-itest")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    rows = _rows(rid, "updates.csv")
    assert len(rows) == 24 and all(x["accepted"] == "True" for x in rows)
    vt = [float(x["vtime"]) for x in rows]
    assert vt == sorted(vt), "updates must be processed in virtual completion order"
    ev = _rows(rid, "eval.csv")
    assert float(ev[-1]["test_acc"]) > float(ev[0]["test_acc"])  # the global model really learned
    if algo == "fedasmu":
        merged = [x for x in rows if x["merged"] == "True"]
        assert merged, "no fresh global model was merged during local training"
        assert all(int(x["fresh_version"]) > int(x["o"]) for x in merged)
        assert {x["slot_src"] for x in rows} >= {"meta", "q"}
        assert max(int(x["staleness"]) for x in rows) > 1  # stale (slow) updates exist
        assert len({x["global_version"] for x in rows}) == 24  # one global update per arrival
    else:
        assert int(rows[-1]["aggregations"]) == 24 // 10  # K = 10
    s = json.load(open(os.path.join(ROOT, "results/metrics", rid, "server_summary.json")))
    assert s["status"] == "COMPLETED" and s["received"] == 24


def test_checkpoint_resume_no_duplicate_aggregation():
    rid = "smoke-itest-resume_cnn_cifar10_fedasmu_s0"
    _clean(rid)
    assert _run("fedasmu", 12, "-itest-resume").returncode == 0
    os.remove(os.path.join(ROOT, "results/metrics", rid, "status.json"))
    r = _run("fedasmu", 24, "-itest-resume")
    assert r.returncode == 0, r.stdout[-2000:]
    rows = _rows(rid, "updates.csv")
    rec = [int(x["received"]) for x in rows]
    assert rec == list(range(1, 25)), rec  # no duplicated / lost update ids
    assert os.path.exists(os.path.join(ROOT, "results/metrics", rid, "resume_log.txt"))
    s = json.load(open(os.path.join(ROOT, "results/metrics", rid, "server_summary.json")))
    assert s["received"] == 24 and s["global_version"] == 24
