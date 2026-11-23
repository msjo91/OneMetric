#!/usr/bin/env python
"""a new calibration draw for the CLIP pools. For every task of the harness (ARCH=B32 CLIPK_SUB=all16 -> the sixteen DC-Merge
tasks; the cache results/clipbreadth/cache is shared by the three backbones), 200 training images drawn with random.Random(CAL_SEED) from the indices OUTSIDE the draw-0 calibration
set (random.Random(0).sample(range(n), 200) as stage_data draws it), preprocessed as the harness does; saved as <task>_cal<seed>_{X,y}.npy. Idempotent. CAL_TASKS=<csv> limits the tasks."""
from onemetric.paths import ROOT as _OM_ROOT
import os, sys, time, random, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = _OM_ROOT; from onemetric import harness_clip as H
from datasets import load_dataset; from concurrent.futures import ThreadPoolExecutor
SEED = int(os.environ["CAL_SEED"]); assert SEED > 0, "seed 0 is the stored draw"; TAG = f"cal{SEED}"; TASKS = os.environ["CAL_TASKS"].split(",") if os.environ.get("CAL_TASKS") else H.TASKS
for t in TASKS:
    if os.path.exists(H.xfile(t, TAG, "X")): H.log(f"{t} {TAG} exists"); continue
    t0 = time.time(); repo, cfg = H.DS[t]; ds = load_dataset(repo, cfg, split="train") if cfg else load_dataset(repo, split="train")
    wds = repo.startswith("clip-benchmark/"); labcol = "cls" if wds else "label"; imgcol = ("png" if "png" in ds.column_names else "jpg") if wds else "image"
    n = len(ds); draw0 = set(random.Random(0).sample(range(n), min(H.N_CAL, n))); rest = [i for i in range(n) if i not in draw0]
    idx = sorted(random.Random(SEED).sample(rest, min(H.N_CAL, len(rest)))); sub = ds.select(idx)
    with ThreadPoolExecutor(16) as ex: X = np.stack(list(ex.map(H.prep, sub[imgcol], chunksize=16)))
    y = np.asarray(sub[labcol], dtype=np.int64); np.save(H.xfile(t, TAG, "X"), X); np.save(H.xfile(t, TAG, "y"), y)
    H.log(f"{t} {TAG}: {len(idx)} of {n} (outside the {len(draw0)} draw-0 images) ({time.time() - t0:.0f}s)")
H.log(f"calibration draw {SEED} done for {len(TASKS)} tasks")
