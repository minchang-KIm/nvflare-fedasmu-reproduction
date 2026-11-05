"""Run one or more experiments of a profile inside the NVFLARE simulator (SimEnv).

python scripts/run_experiment.py --profile smoke|pilot|paper|ablation [--only REGEX] [--seeds 0 1]

For each run the script
  1. builds the shared artefacts (Dirichlet partition, device profiles, initial
     weights) -- identical for every algorithm of the same (model, dataset, seed),
  2. writes the run config JSON (consumed by the NVFLARE controller/executor),
  3. builds an NVFLARE ``FedJob`` (custom Controller on the server, custom Executor
     on 'max_concurrency' sites) and executes it with ``SimEnv``,
  4. writes a manifest (config, git hash, versions, digests) and a status file.
Runs whose status is COMPLETED are skipped (no re-execution).
"""
import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["PYTHONPATH"] = ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from src.datasets.cifar import load_tensor_dataset  # noqa: E402
from src.datasets.partition import dirichlet_partition, save_partition  # noqa: E402
from src.models import build_model  # noqa: E402
from src.nvflare_integration.common import sd_digest  # noqa: E402
from src.scheduling.device_profiles import make_profiles, profiles_to_json  # noqa: E402


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def git_hash():
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", "src", "scripts", "configs"], cwd=ROOT) != 0
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def build_run_config(profile, run, seed, exp_over):
    base = yaml.safe_load(open(os.path.join(ROOT, "configs/common/base.yaml")))
    settings = yaml.safe_load(open(os.path.join(ROOT, "configs/common/paper_settings.yaml")))
    key = f"{run['model']}-{run['dataset']}"
    ps = settings[key]
    cfg = copy.deepcopy(base)
    cfg["seed"] = seed
    cfg["local"]["lr"] = ps["lr"]
    for k in ("eta_lambda", "eta_sigma", "eta_iota", "eta_gamma", "eta_upsilon", "eta_rl"):
        cfg["fedasmu"][k] = ps[k]
    cfg["base_epoch_time"] = ps["base_epoch_time"]
    cfg["target_acc"] = ps["target"]
    cfg["paper"] = ps["paper"]
    deep_update(cfg, copy.deepcopy(exp_over))
    deep_update(cfg, {k: v for k, v in run.items() if k not in ("model", "dataset", "algorithm")})
    cfg.update(model=run["model"], dataset=run["dataset"], algorithm=run["algorithm"], profile=profile)
    cfg["run_id"] = f"{profile}_{run['model']}_{run['dataset']}_{run['algorithm']}_s{seed}"
    return cfg


def shared_artifacts(cfg):
    """Partition, profiles, initial weights shared by all algorithms (same model/dataset/seed/#devices)."""
    tag = f"{cfg['dataset']}_m{cfg['num_devices']}_a{cfg['dirichlet_alpha']}_s{cfg['seed']}"
    if cfg.get("max_samples_per_device"):
        tag += f"_max{cfg['max_samples_per_device']}"
    pfile = os.path.join(ROOT, "results/manifests", f"partition_{tag}.npz")
    if not os.path.exists(pfile):
        _, y, _, _, _ = load_tensor_dataset(cfg["dataset"])
        parts = dirichlet_partition(y.numpy(), cfg["num_devices"], cfg["dirichlet_alpha"], cfg["seed"])
        if cfg.get("max_samples_per_device"):
            rng = np.random.RandomState(cfg["seed"])
            parts = [np.sort(rng.permutation(p)[: cfg["max_samples_per_device"]]) for p in parts]
        save_partition(pfile, parts, {"dataset": cfg["dataset"], "num_devices": cfg["num_devices"],
                                      "alpha": cfg["dirichlet_alpha"], "seed": cfg["seed"],
                                      "max_samples_per_device": cfg.get("max_samples_per_device")})
    ifile = os.path.join(ROOT, "checkpoints/init", f"{cfg['model']}_{cfg['dataset']}_s{cfg['seed']}.pt")
    if not os.path.exists(ifile):
        os.makedirs(os.path.dirname(ifile), exist_ok=True)
        torch.manual_seed(cfg["seed"])
        ncls = 100 if cfg["dataset"] == "cifar100" else 10
        sd = build_model(cfg["model"], ncls).state_dict()
        torch.save(sd, ifile)
        with open(os.path.join(ROOT, "results/manifests", f"init_{cfg['model']}_{cfg['dataset']}_s{cfg['seed']}.json"), "w") as f:
            json.dump({"sha256": sd_digest(sd), "num_params": int(sum(v.numel() for v in sd.values()))}, f)
    profiles = make_profiles(cfg["num_devices"], cfg["seed"], cfg["base_epoch_time"], cfg["hetero"], cfg["bw_min"], cfg["bw_max"])
    return pfile, ifile, profiles_to_json(profiles)


def run_one(cfg, dry=False):
    import nvflare
    from nvflare.job_config.api import FedJob
    from nvflare.recipe.sim_env import SimEnv

    from src.nvflare_integration.controller import AsyncSimController
    from src.nvflare_integration.executor import DeviceSimExecutor

    rid = cfg["run_id"]
    mdir = os.path.join(ROOT, "results/metrics", rid)
    status_file = os.path.join(mdir, "status.json")
    if os.path.exists(status_file) and json.load(open(status_file)).get("status") == "COMPLETED":
        print(f"[skip] {rid} already COMPLETED")
        return "SKIPPED"
    os.makedirs(mdir, exist_ok=True)
    pfile, ifile, profiles = shared_artifacts(cfg)
    run_dir = os.path.join(ROOT, "workspaces/runs", rid)
    os.makedirs(run_dir, exist_ok=True)
    cfg.update(partition_file=pfile, init_weights=ifile, profiles=profiles, metrics_dir=mdir, run_dir=run_dir,
               checkpoint_dir=os.path.join(ROOT, "checkpoints", rid))
    if cfg["algorithm"].startswith("fedasmu"):
        hist = os.path.join(ROOT, "results/metrics", f"metahist_{cfg['model']}_{cfg['dataset']}.json")
        cfg["fedasmu"]["meta_pretrain_file"] = hist
    cfg_path = os.path.join(run_dir, "run_config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=1)
    manifest = {"run_id": rid, "git": git_hash(), "nvflare": nvflare.__version__, "torch": torch.__version__,
                "started": time.strftime("%Y-%m-%d %H:%M:%S"), "config": {k: v for k, v in cfg.items() if k != "profiles"},
                "partition_meta": json.load(open(pfile.replace(".npz", ".json"))),
                "init_meta_file": ifile, "profiles_summary": {
                    "epoch_time_min": min(p["epoch_time"] for p in profiles),
                    "epoch_time_max": max(p["epoch_time"] for p in profiles)}}
    with open(os.path.join(mdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    with open(status_file, "w") as f:
        json.dump({"status": "RUNNING", "started": manifest["started"]}, f)
    if dry:
        return "DRY"
    n_sites = cfg["max_concurrency"]
    job = FedJob(name=rid, min_clients=n_sites)
    job.to_server(AsyncSimController(run_config=cfg_path))
    job.to_clients(DeviceSimExecutor(run_config=cfg_path), tasks=["train"])
    ws_root = os.path.join(ROOT, "workspaces/sim")
    env = SimEnv(clients=[f"site-{i + 1}" for i in range(n_sites)], num_threads=n_sites, workspace_root=ws_root)
    t0 = time.time()
    status, err = "FAILED", None
    try:
        env.deploy(job)
        summ = json.load(open(os.path.join(mdir, "server_summary.json")))
        status = summ["status"]
        err = summ.get("error")
    except Exception:
        err = traceback.format_exc()
    with open(status_file, "w") as f:
        json.dump({"status": status, "started": manifest["started"], "wall_seconds": time.time() - t0,
                   "error": err, "git": manifest["git"]}, f, indent=1)
    # keep LSTM meta-model history as 'historical data' for later pre-training (paper)
    mh = os.path.join(mdir, "meta_history.json")
    if status == "COMPLETED" and os.path.exists(mh):
        dst = os.path.join(ROOT, "results/metrics", f"metahist_{cfg['model']}_{cfg['dataset']}.json")
        if not os.path.exists(dst) and cfg["algorithm"] == "fedasmu" and cfg["profile"] == "pilot":
            with open(mh) as fi, open(dst, "w") as fo:
                fo.write(fi.read())
    print(f"[{status}] {rid} wall={time.time() - t0:.0f}s")
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--only", default=None, help="regex on run_id")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the profile part of run_id (e.g. timing tests)")
    ap.add_argument("--set", nargs="*", default=[], help="override key=value (json values), e.g. budget_updates=100")
    args = ap.parse_args()
    exp = yaml.safe_load(open(os.path.join(ROOT, "configs/experiments", f"{args.profile}.yaml")))
    exp["runs"] = exp["runs"]
    seeds = args.seeds if args.seeds is not None else exp.get("seeds", [0])
    over = exp.get("overrides", {}) or {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        deep_update(over, {k: json.loads(v)})
    results = {}
    for seed in seeds:
        for run in exp["runs"]:
            cfg = build_run_config(args.profile + args.tag, run, seed, over)
            if args.only and not re.search(args.only, cfg["run_id"]):
                continue
            results[cfg["run_id"]] = run_one(cfg, dry=args.dry)
    print(json.dumps(results, indent=1))
    return 0 if all(v in ("COMPLETED", "SKIPPED", "DRY") for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
