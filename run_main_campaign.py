"""Main-results campaign: 18 combos (3 datasets x 3 backbones x 2 code lengths).

P_mismatch=10, lambda=0.5, beta=0.1, seed 2. Per-dataset convention: CIFAR-10 and NUS-WIDE run
30 epochs evaluating every epoch, ImageNet runs 100 epochs evaluating every 5. Everything else
-- codeset, alpha, lr, batch size -- comes from config/campaign/{combo}.yaml.

One GPU per job from a device pool, ImageNet enqueued first since it is the long pole.
Only the best checkpoint of each run is kept.

Set PHM_DETERMINISTIC=1 and CUBLAS_WORKSPACE_CONFIG=:4096:8 in the environment for
deterministic training. Evaluate the finished runs with eval_main_campaign.py.

    export PHM_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
    python run_main_campaign.py --devices 0 1 2 3 4 5 6 7
    python eval_main_campaign.py --device cuda:0
"""
import argparse
import glob
import os
import queue
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sconf import Config

REPO = Path(__file__).resolve().parent
RESULTS = REPO / "results" / "main"
CFG_DIR = REPO / "sweep_main" / "configs"
LOG_DIR = REPO / "sweep_main" / "logs"
PY = os.environ.get("PHM_PY", sys.executable)

P = 10.0
SEED = 2
_MULTI = {"nuswide", "nuswide_eva", "nuswide_swin"}
_IMAGENET = {"imagenet", "imagenet_swin", "imagenet_eva"}
# imagenet first (long pole) -> nuswide -> cifar
COMBOS = ["in_vit16", "in_vit32", "in_swin16", "in_swin32", "in_eva16", "in_eva32",
          "nus_vit16", "nus_vit32", "nus_swin16", "nus_swin32", "nus_eva16", "nus_eva32",
          "cifar_vit16", "cifar_vit32", "cifar_swin16", "cifar_swin32", "cifar_eva16", "cifar_eva32"]

_CONFIG_LOCK = threading.Lock()
_PRINT_LOCK = threading.Lock()


def log(msg):
    with _PRINT_LOCK:
        print(msg, flush=True)


def template(combo):
    return str(REPO / "config" / "campaign" / f"{combo}.yaml")


def keep_best_only(job):
    pts = glob.glob(str(RESULTS / job / "**" / "*.pt"), recursive=True)
    if len(pts) <= 1:
        return
    def mp(p):
        m = re.search(r"map([0-9.]+)\.pt", p)
        return float(m.group(1)) if m else -1.0
    best = max(pts, key=mp)
    for p in pts:
        if p != best:
            os.remove(p)


def write_cfg(combo, device):
    job = f"{combo}_P10_s{SEED}"
    with _CONFIG_LOCK:
        cfg = Config(template(combo))
    cfg.seed = SEED
    cfg.mismatch_penalty = P
    cfg.save_every_epoch = False
    cfg.eval_every = 5 if cfg.dataset in _IMAGENET else 1
    cfg.epochs = 100 if cfg.dataset in _IMAGENET else 30
    cfg.encoder_save_path = str(RESULTS)
    cfg.device = device
    script = "main_multi.py" if cfg.dataset in _MULTI else "main.py"
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    p = CFG_DIR / f"{job}.yaml"
    p.write_text(cfg.dumps(modified_color=None, quote_str=True))
    return p, job, script


def run_train(cfg_path, log_path, script):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        rc = subprocess.run([PY, script, "--config", str(cfg_path), "--mode", "train"],
                            cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT).returncode
    return rc == 0


def run_one(combo, device):
    p, job, script = write_cfg(combo, device)
    if (RESULTS / job / "DONE").exists():
        log(f"[skip] {job}")
        return job
    log(f"[start] {job} ({script}) on {device}")
    ok = run_train(p, LOG_DIR / f"{job}.log", script)
    if ok:
        keep_best_only(job)
        (RESULTS / job).mkdir(parents=True, exist_ok=True)
        (RESULTS / job / "DONE").write_text("done")
    log(f"[{'done' if ok else 'err'}] {job}")
    return job if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="+", default=[str(i) for i in range(8)])
    ap.add_argument("--only", nargs="*", default=None, help="combo substring filter (for re-runs)")
    args = ap.parse_args()
    devices = [d if str(d).startswith("cuda") else f"cuda:{d}" for d in args.devices]
    combos = [c for c in COMBOS if (not args.only or any(o in c for o in args.only))]
    RESULTS.mkdir(parents=True, exist_ok=True)
    log(f"[main-campaign] {len(combos)} combos / {len(devices)} GPUs, P={P:g} seed={SEED}")
    dq = queue.Queue()
    for d in devices:
        dq.put(d)

    def worker(c):
        dev = dq.get()
        try:
            return run_one(c, dev)
        finally:
            dq.put(dev)

    with ThreadPoolExecutor(max_workers=len(devices)) as ex:
        futs = [ex.submit(worker, c) for c in combos]
        for f in as_completed(futs):
            f.result()
    log("[main-campaign] training complete")


if __name__ == "__main__":
    main()
