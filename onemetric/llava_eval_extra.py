#!/usr/bin/env python
"""Evaluate one stored LLaVA merge by name on the benchmark's test draw (n = 500)."""
from onemetric.paths import ROOT as _OM_ROOT
import os, sys, json, time, argparse
ROOT = _OM_ROOT
from onemetric import llava_pool as V# noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); args = ap.parse_args(); name = args.name
OUT = os.path.join(ROOT, "results", "f25nm", "vlm"); os.makedirs(OUT, exist_ok=True); out_p = os.path.join(OUT, f"eval_pub_{name}.json")
assert V.done(name), f"merge {name} missing"
t0 = time.time(); V.log(f"== eval {name}"); V.stage_eval(f"merged:{name}")
home = {V.task_of(c): V.acc_of(f"client_{c}", V.task_of(c)) for c in V.CLIENTS}; R, Rt = V.R_of(f"merged_{name}", home); U = V.U_of(f"merged_{name}")
acc = {t: V.acc_of(f"merged_{name}", t) for t in V.EVAL_TASKS}
json.dump({"name": name, "R": R, "R_t": Rt, "U": U, "acc": acc, "n": V.EVAL_N, "seed": V.SEED, "seconds": time.time() - t0}, open(out_p, "w"), indent=1)
V.log(f"eval {name} written ({time.time() - t0:.0f}s)")
