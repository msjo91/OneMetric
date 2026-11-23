#!/usr/bin/env python
"""data (CPU): the eight new tasks of for SigLIP 2, from the same sources and the same seed-0 draws (test 2,000 from the test split, calibration 200
and training up to 10,000 from the train split), preprocessed as siglip_backend.prep does (squash-resize to RES x RES, bilinear) at both SigLIP 2 resolutions in one pass:
224 (S2B16, S2SO) and 256 (S2L16). Writes results/siglip/cache<RES>/<t>_test_{X,y}.npy, <t>_cal_{X,y}.npy, <t>_train10000_{X,y}.npy (deleted by the chain once the task's
adapters are saved), <t>_names.json and sizes.json[t]. Sources and draws come from data_newtasks.py (imported with no task argument, so it builds nothing).
Usage: data_newtasks_siglip.py <task> [<task> ...] [--out <dir>] [--n <train,test,cal>] (the last two for the smoke test only). Prints sizes and class counts only."""
from onemetric.paths import ROOT as _OM_ROOT
import os, sys, json, time, numpy as np
from concurrent.futures import ThreadPoolExecutor
ROOT = _OM_ROOT
args = sys.argv[1:]; OUTDIR = None; NS = (10000, 2000, 200)
if "--out" in args: i = args.index("--out"); OUTDIR = args[i + 1]; del args[i:i + 2]
if "--n" in args: i = args.index("--n"); NS = tuple(int(x) for x in args[i + 1].split(",")); del args[i:i + 2]
_argv = sys.argv; sys.argv = [_argv[0]]; from onemetric import data_newtasks as ND; sys.argv = _argv   # source() and draw() of ; no task argument, so its loop builds nothing
from PIL import Image
RESS = (224, 256); N_TRAIN, N_TEST, N_CAL = NS
CACHES = {r: (os.path.join(OUTDIR, f"cache{r}") if OUTDIR else os.path.join(ROOT, "results", "siglip", f"cache{r}")) for r in RESS}
for d in CACHES.values(): os.makedirs(d, exist_ok=True)
def prep_both(img):   # identical to siglip_backend.prep at each resolution
    img = img.convert("RGB"); return tuple(np.array(img.resize((r, r), Image.BILINEAR), dtype=np.uint8) for r in RESS)
def log(*a): print(time.strftime("%H:%M:%S", time.gmtime(time.time() + 9 * 3600)), *a, flush=True)
for t in args:
    t0 = time.time(); n_tr, n_te, get, names = ND.source(t); assert len(set(names)) == len(names), t
    for sp, n, tag in (("test", N_TEST, "test"), ("train", N_CAL, "cal"), ("train", N_TRAIN, "train10000")):
        if all(os.path.exists(os.path.join(CACHES[r], f"{t}_{tag}_X.npy")) for r in RESS): continue
        idx = ND.draw(n_tr if sp == "train" else n_te, n); im, lab = get(sp, idx)
        with ThreadPoolExecutor(8) as ex: P = list(ex.map(prep_both, im, chunksize=16))
        y = np.asarray(lab, dtype=np.int64); assert y.min() >= 0 and y.max() < len(names), (t, sp)
        for j, r in enumerate(RESS):
            X = np.stack([p[j] for p in P]); np.save(os.path.join(CACHES[r], f"{t}_{tag}_X.npy"), X); np.save(os.path.join(CACHES[r], f"{t}_{tag}_y.npy"), y)
        del P; log(f"{t} {tag} draw {len(idx)} of {n_tr if sp == 'train' else n_te}, labels seen {len(set(lab))} of {len(names)} ({time.time() - t0:.0f}s)")
    for r in RESS:
        json.dump(names, open(os.path.join(CACHES[r], f"{t}_names.json"), "w")); sf = os.path.join(CACHES[r], "sizes.json")
        sz = json.load(open(sf)) if os.path.exists(sf) else {}; sz[t] = min(n_tr, N_TRAIN); json.dump(sz, open(sf, "w"))
    log(f"{t} done: {n_tr} train, {n_te} test, {len(names)} classes ({time.time() - t0:.0f}s)")
