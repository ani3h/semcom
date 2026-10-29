"""Resumable job scheduler for the Stage 2 head training. Safe to kill and re-run: a job is done iff its train_*.json exists, and
partially trained checkpoints are never trusted (they are removed before a job is re-run).

    python experiments/week5b_s2_scheduler.py [--workers 5]

Order: ESC-50 regularisation comparison (seed 42) first; the ESC-50 selection rule is applied automatically once all 12 comparison runs
exist, then the ESC-50 seeds 42-44 with the winning setting are queued. Long jobs (caption text, Europarl text) are scheduled first.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HEADS = Path("outputs/week5b/stage2/heads")
LOGS = Path("outputs/week5b/logs/s2")
SETTING = Path("outputs/week5b/stage2/esc50_setting.json")
PY = "./.venv/bin/python"
SEEDS = (42, 43, 44)
ORDER = ["caption_text", "text", "audio", "cifar10", "mnist"]


def free_memory_pct() -> int:
    """macOS system-wide free memory percentage (reclaimable included). Heads hold 4-5 GB each on an 18 GB machine, so jobs are admitted only
    when memory is available: five concurrent heads plus an embedding job pushed the machine 15 GB into swap and slowed every step ~100x."""
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage:\s*(\d+)%", out)
    return int(m.group(1)) if m else 100


def job(task, ch, var, seed, tag="", overrides=()):
    stem = f"{var}_s{seed}{tag}"
    return {"key": f"{task}/{ch}/{stem}", "done": HEADS / task / ch / f"train_{stem}.json", "ckpt": HEADS / task / ch / f"{stem}.pt",
            "cmd": [PY, "experiments/week5b_s2_train_heads.py", f"common_s2={task}", f"channel={ch}", f"variant={var}", f"seed={seed}",
                    *([f"tag={tag}"] if tag else []), *overrides], "log": LOGS / f"{task}_{ch}_{stem}.log"}


def esc_compare_jobs():
    cfgs = {"_base": ("0.1", "1e-4"), "_R1": ("0.3", "1e-3"), "_R2": ("0.5", "1e-2")}
    return [job("esc50_audio", ch, var, 42, tag, (f"head.dropout={d}", f"train.weight_decay={wd}"))
            for var in ("cos5", "cen") for ch in ("awgn", "rayleigh") for tag, (d, wd) in cfgs.items()]


def main_jobs():
    return [job(t, ch, var, s) for t in ORDER for s in SEEDS for var in ("cos5", "cen") for ch in ("awgn", "rayleigh")]


def esc_final_jobs():
    st = json.loads(SETTING.read_text())["chosen_setting"]
    ov = (f"head.dropout={st['dropout']}", f"train.weight_decay={st['weight_decay']}")
    return [job("esc50_audio", ch, var, s, "", ov) for s in (43, 44) for var in ("cos5", "cen") for ch in ("awgn", "rayleigh")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-free-pct", type=int, default=25)
    args = ap.parse_args()
    workers, min_free = args.workers, args.min_free_pct
    last_start = 0.0
    LOGS.mkdir(parents=True, exist_ok=True)
    running: dict[str, tuple] = {}
    failed: set[str] = set()
    t0 = time.time()
    while True:
        for k, (p, j, fh) in list(running.items()):
            if p.poll() is not None:
                fh.close()
                del running[k]
                if not j["done"].exists():
                    failed.add(k)
                    print(f"[sched] FAILED {k} (exit {p.returncode}); see {j['log']}", flush=True)
                else:
                    print(f"[sched] done {k} [{(time.time() - t0) / 60:.0f} min]", flush=True)
        queue = esc_compare_jobs() + main_jobs()
        cmp_done = all(j["done"].exists() for j in esc_compare_jobs())
        if cmp_done and not SETTING.exists():
            subprocess.run([PY, "experiments/week5b_s2_esc50_select.py"], check=True)
        if SETTING.exists():
            queue += esc_final_jobs()
        pending = [j for j in queue if not j["done"].exists() and j["key"] not in running and j["key"] not in failed]
        # long jobs first, ESC-50 comparison first of all so the selection rule can fire early
        pending.sort(key=lambda j: (0 if "_base" in j["key"] or "_R1" in j["key"] or "_R2" in j["key"] else 1))
        while pending and len(running) < workers and (not running or free_memory_pct() >= min_free) and time.time() - last_start > 60:
            last_start = time.time()  # a new process needs ~a minute to reach its steady memory before the next admission decision
            j = pending.pop(0)
            if not j["ckpt"].with_suffix(".resume.pt").exists():
                j["ckpt"].unlink(missing_ok=True)  # a partial best-checkpoint is only trusted together with its resume state
            fh = open(j["log"], "w")
            running[j["key"]] = (subprocess.Popen(j["cmd"], stdout=fh, stderr=subprocess.STDOUT), j, fh)
            print(f"[sched] start {j['key']}  ({len(running)} running, {len(pending)} queued)", flush=True)
        if not running and not pending:
            break
        time.sleep(5)
    print(f"[sched] all done in {(time.time() - t0) / 60:.0f} min; failed: {sorted(failed) or 'none'}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
