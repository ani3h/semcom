"""Keep a bounded amount of GPU-heavy Stage 2 work active at once by pausing (SIGSTOP) the rest and resuming (SIGCONT) them as slots free up.

    python experiments/week5b_s2_governor.py

Why: the work is compute-bound on one GPU and each process holds 4-7 GB on an 18 GB machine, so more than two concurrent processes add no
throughput and push the machine into swap (observed: ~100x slower steps). Pausing loses no progress, unlike killing. Policy: one slot for
an embedding job when one exists, the other(s) for head training; oldest process first within each group.
"""
import os
import re
import signal
import subprocess
import time

BUDGET = 4  # heavy jobs (text, caption, speech heads; embedding jobs) cost 2, light jobs (image and ESC-50 heads) cost 1
HEAVY = re.compile(r"common_s2=(caption_text|text|audio)\b|week5b_s2_embed\.py")
PAT = {"head": re.compile(r"week5b_s2_train_heads\.py"), "embed": re.compile(r"week5b_s2_embed\.py")}


def processes() -> dict[str, list[tuple[int, int, str, str]]]:
    out = subprocess.run(["ps", "-eo", "pid=,etime=,stat=,command="], capture_output=True, text=True).stdout.splitlines()
    groups = {"head": [], "embed": []}
    for line in out:
        parts = line.split(None, 3)
        if len(parts) < 4 or "governor" in parts[3] or "scheduler" in parts[3] or "run_s2" in parts[3] or "tracking.mode=disabled" in parts[3]:
            continue  # scratch / test runs (tracking.mode=disabled) are never governed
        for g, pat in PAT.items():
            if pat.search(parts[3]) and parts[3].lstrip().startswith(("./.venv/bin/python", "python")):
                et = parts[1]
                d, _, t = et.rpartition("-")
                h = t.split(":")
                secs = sum(int(x) * m for x, m in zip(reversed(h), (1, 60, 3600))) + (int(d) * 86400 if d else 0)
                groups[g].append((int(parts[0]), secs, parts[2], parts[3]))
    for g in groups:
        groups[g].sort(key=lambda x: -x[1])  # oldest first
    return groups


def main() -> None:
    idle = 0
    while idle < 120:  # exit after ~10 minutes with nothing to govern
        g = processes()
        embeds, heads = g["embed"], g["head"]
        cmd = {pid: c for g_ in g.values() for pid, _, _, c in g_}
        order = embeds[:1] + heads + embeds[1:]  # one embedding job first (it is slow and CPU-bound), then heads oldest first
        want_pids, used = set(), 0
        for pid, _, _, _ in order:
            w = 2 if HEAVY.search(cmd[pid]) else 1
            if used + w <= BUDGET:
                want_pids.add(pid)
                used += w
        for pid, _, stat, _ in embeds + heads:
            stopped = stat.startswith("T")
            if pid in want_pids and stopped:
                os.kill(pid, signal.SIGCONT)
                print(f"[gov] resume {pid}", flush=True)
            elif pid not in want_pids and not stopped:
                os.kill(pid, signal.SIGSTOP)
                print(f"[gov] pause {pid}", flush=True)
        idle = idle + 1 if not (embeds or heads) else 0
        time.sleep(5)


if __name__ == "__main__":
    main()
