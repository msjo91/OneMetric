#!/usr/bin/env python
"""calibration sets of other sizes for the CLIP benchmark tasks (ARCH=B32 CLIPK_SUB=all16; the cache
results/clipbreadth/cache is shared by the three backbones). Nested in the stored draw 0 (200 training images, random.Random(0).sample, stored in sorted index order):
- N < 200: the first N positions of one fixed permutation of the 200 stored images (random.Random(136)); n10 in n50 in n100 in the stored 200.
- N > 200: the stored 200 plus the first N - 200 of a fixed random order (random.Random(136)) of the training images outside draw 0 and outside the public images of when those came from the training split; n300 in n500.
Saved as <task>_caln<N>_{X,y}.npy (read by MR_CALSEED=n<N> in run_method.py and COM_VAR=sn<N> in run_chain_of_merges.py). Idempotent. NS=<csv> sizes (default 10,50,100,300,500)."""
from onemetric.paths import ROOT as _OM_ROOT
import os, sys, time, json, random, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = _OM_ROOT; from onemetric import harness_clip as H
from datasets import load_dataset; from concurrent.futures import ThreadPoolExecutor
NS = [int(x) for x in os.environ.get("NS", "10,50,100,300,500").split(",")]; assert H.N_CAL == 200 and 200 not in NS
PUBSRC = json.load(open(os.path.join(ROOT, "results", "clipclose", "zeroup", "cache", "src.json")))
meta_p = os.path.join(H.CACHE, "ncal_meta.json"); meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
for t in H.TASKS:
    todo = [n for n in NS if not os.path.exists(H.xfile(t, f"caln{n}", "X"))]
    if not todo: H.log(f"{t}: all sizes exist"); continue
    t0 = time.time(); X0, y0 = H.load_x(t, "cal"); assert len(X0) == 200
    perm = random.Random(136).sample(range(200), 200); m = meta.setdefault(t, {})
    for n in [n for n in todo if n < 200]:
        pos = perm[:n]; np.save(H.xfile(t, f"caln{n}", "X"), np.asarray(X0)[pos]); np.save(H.xfile(t, f"caln{n}", "y"), y0[pos]); m[f"n{n}"] = {"positions_in_draw0": pos}
    big = [n for n in todo if n > 200]
    if big:
        repo, cfg = H.DS[t]; ds = load_dataset(repo, cfg, split="train") if cfg else load_dataset(repo, split="train")
        wds = repo.startswith("clip-benchmark/"); labcol = "cls" if wds else "label"; imgcol = ("png" if "png" in ds.column_names else "jpg") if wds else "image"
        n_tr = len(ds); draw0 = sorted(random.Random(0).sample(range(n_tr), min(200, n_tr)))
        assert np.array_equal(np.asarray(ds.select(draw0)[labcol], dtype=np.int64), y0), f"{t}: draw 0 does not reproduce"
        excl = set(draw0) | (set(PUBSRC[t]["idx"]) if PUBSRC.get(t, {}).get("source") == "train" else set())
        rest = [i for i in range(n_tr) if i not in excl]; order = random.Random(136).sample(rest, min(max(big) - 200, len(rest)))
        assert len(order) >= max(big) - 200, f"{t}: only {len(rest)} training images outside draw 0 and the public set"
        sub = ds.select(order)
        with ThreadPoolExecutor(16) as ex: Xe = np.stack(list(ex.map(H.prep, sub[imgcol], chunksize=16)))
        ye = np.asarray(sub[labcol], dtype=np.int64)
        for n in big:
            k = n - 200; np.save(H.xfile(t, f"caln{n}", "X"), np.concatenate([np.asarray(X0), Xe[:k]])); np.save(H.xfile(t, f"caln{n}", "y"), np.concatenate([y0, ye[:k]]))
            m[f"n{n}"] = {"draw0": 200, "extra_train_idx": order[:k], "train_size": n_tr, "excluded_public": PUBSRC.get(t, {}).get("source") == "train"}
    json.dump(meta, open(meta_p, "w")); H.log(f"{t}: sizes {todo} written ({time.time() - t0:.0f}s)")
H.log(f"calibration sets done for {len(H.TASKS)} tasks, sizes {NS}")
