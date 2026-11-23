#!/usr/bin/env python
"""RobustMerge on a LLaVA pool, re-implemented from the conventions of its released code. Stages: build | eval."""
from src.paths import ROOT as _OM_ROOT
import os, sys, json, time, argparse, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True, choices=["build", "eval"]); args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = _OM_ROOT
from src.llava import llava_pool as V# noqa: E402
LP, log = V.LP, V.log; KEEP, ATT = 0.2, 0.2
def mask_top(M, keep):
    k = max(1, int(round(keep * M.size))); thr = np.partition(np.abs(M).ravel(), M.size - k)[M.size - k]; return (np.abs(M) >= thr).astype(np.float64)
def robustmerge(pool, clients, lam):
    cells = sorted(pool[clients[0]]); out = {}
    for cell in cells:
        At, Bt, Ss = [], [], []
        for n in clients:
            A = np.asarray(pool[n][cell]["A"], dtype=np.float64); B = np.asarray(pool[n][cell]["B"], dtype=np.float64)
            MA, MB = mask_top(A, KEEP), mask_top(B, KEEP); Atil, Btil = A * MA, B * MB
            S = np.abs(A).sum(1) / np.maximum(np.abs(Atil).sum(1), 1e-12); S = np.maximum(S, np.percentile(S, 100 * (1 - ATT) ))
            At.append(Atil); Bt.append(Btil); Ss.append(S)
        Ssum = np.maximum(sum(Ss), 1e-12); St = [S / Ssum for S in Ss]
        out[cell] = {"B": lam * sum(B * S[None, :] for B, S in zip(Bt, St)), "A": lam * sum(At)}
    return out
P = V.SUB or "FULL"
if args.stage == "build":
    pool, projs = V.load_pool(); proj = LP.mean_projector(projs); t0 = time.time()
    for name, lam in (("robustmerge", 2.0), ("robustmerge_p", 1.0)):
        if V.done(name): log(f"[{P}] {name} exists"); continue
        f = robustmerge(pool, list(V.CLIENTS), lam); fro = float(np.sqrt(sum(np.linalg.norm(v["B"] @ v["A"]) ** 2 for v in f.values())))
        V.write(name, f, f"rule={name} RobustMerge (keep {KEEP}, clamp {ATT}, lam {lam} per factor in our scaled-B units; )", proj); log(f"[{P}] {name} built (fro {fro:.2f}, {time.time() - t0:.0f}s)")
else:
    for name in ("robustmerge", "robustmerge_p"):
        f = os.path.join(V.POOL, f"merged_{name}__metrics.json")
        if os.path.exists(f) and all(t in json.load(open(f)) and json.load(open(f))[t].get("acc") is not None for t in V.EVAL_TASKS): log(f"[{P}] metrics of {name} exist"); continue
        V.stage_eval(f"merged:{name}")
