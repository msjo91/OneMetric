#!/usr/bin/env python
"""data, CPU: other public image sets for the fifteen CLIP benchmark tasks.
Draws: two further sets of 200 public images per task by rule (data_public_clip.py --stage pub: the test split outside the stored 2,000-image test draw; for the
tasks whose stored public set came from the training split, the training split outside the calibration draw), drawn with seeds 2 and 3 from the images outside the stored public set and
outside each other, so the three sets share no image. Written to results/clipclose/protect/cache_pubd1 and cache_pubd2.
Counts: the first 50 and the first 100 positions of one fixed permutation (random.Random(153)) of the stored 200 public images, so 50 is inside 100 is inside the stored set.
Written to results/clipclose/protect/cache_pubc50 and cache_pubc100.
Every directory holds <task>_pub_{X,y}.npy as the launcher reads them and a src.json. Idempotent. PD_OUT overrides the parent directory, PD_TASKS=<csv> limits the tasks."""
from src.paths import ROOT as _OM_ROOT
import os, sys, time, json, random, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = _OM_ROOT; os.environ.setdefault("POOL", "dc16"); os.environ.setdefault("ARCH", "B32"); os.environ.setdefault("CLIPK_SUB", "all16")
from src.backbones import harness_clip as H
from datasets import load_dataset; from concurrent.futures import ThreadPoolExecutor
P = os.environ.get("PD_OUT") or os.path.join(ROOT, "results", "clipclose", "protect"); PUBC = os.path.join(ROOT, "results", "clipclose", "zeroup", "cache"); SRC = json.load(open(os.path.join(PUBC, "src.json")))
want = os.environ["PD_TASKS"].split(",") if os.environ.get("PD_TASKS") else [t for t in H.TASKS if t not in H.RET_EXCL]
DIRS = {k: os.path.join(P, f"cache_{k}") for k in ("pubd1", "pubd2", "pubc50", "pubc100")}
for d in DIRS.values(): os.makedirs(d, exist_ok=True)
src = {k: (json.load(open(os.path.join(DIRS[k], "src.json"))) if os.path.exists(os.path.join(DIRS[k], "src.json")) else {}) for k in DIRS}
def put(k, t, X, y, info):
    np.save(os.path.join(DIRS[k], f"{t}_pub_X.tmp.npy"), X); os.replace(os.path.join(DIRS[k], f"{t}_pub_X.tmp.npy"), os.path.join(DIRS[k], f"{t}_pub_X.npy")); np.save(os.path.join(DIRS[k], f"{t}_pub_y.npy"), y)
    src[k][t] = info; json.dump(src[k], open(os.path.join(DIRS[k], "src.json.tmp"), "w")); os.replace(os.path.join(DIRS[k], "src.json.tmp"), os.path.join(DIRS[k], "src.json"))
for t in want:
    if all(os.path.exists(os.path.join(DIRS[k], f"{t}_pub_X.npy")) and t in src[k] for k in DIRS): H.log(f"{t}: exists"); continue
    t0 = time.time(); X0 = np.load(os.path.join(PUBC, f"{t}_pub_X.npy"), mmap_mode="r"); y0 = np.load(os.path.join(PUBC, f"{t}_pub_y.npy")); assert len(X0) == 200
    perm = random.Random(153).sample(range(200), 200)
    for n in (50, 100): put(f"pubc{n}", t, np.asarray(X0)[perm[:n]], y0[perm[:n]], {"positions_in_stored_public_set": perm[:n]})
    where = SRC[t]["source"]; repo, cfg = H.DS[t]; ds = load_dataset(repo, cfg, split=where) if cfg else load_dataset(repo, split=where); wds = repo.startswith("clip-benchmark/")
    labcol = "cls" if wds else "label"; imgcol = ("png" if "png" in ds.column_names else "jpg") if wds else "image"; n = len(ds); lab = np.asarray(ds[labcol], dtype=np.int64)
    assert np.array_equal(lab[SRC[t]["idx"]], y0), f"{t}: the stored public set does not reproduce from its indices"
    excl = set(random.Random(0).sample(range(n), min(2000 if where == "test" else 200, n))) | set(SRC[t]["idx"]); rest = [i for i in range(n) if i not in excl]; used = set()
    for j, seed in ((1, 2), (2, 3)):
        pool = [i for i in rest if i not in used]; assert len(pool) >= 200, f"{t}: only {len(pool)} images left for draw {j}"
        idx = sorted(random.Random(seed).sample(pool, 200)); used |= set(idx); sub = ds.select(idx)
        with ThreadPoolExecutor(int(os.environ.get("PD_THREADS", "8"))) as ex: X = np.stack(list(ex.map(H.prep, sub[imgcol], chunksize=8)))
        put(f"pubd{j}", t, X, lab[idx], {"source": where, "seed": seed, "n_split": n, "n_available": len(pool), "idx": idx})
    H.log(f"{t}: two draws from the {where} split and two counts written ({time.time() - t0:.0f}s)")
H.log(f"public image sets done for {len(want)} tasks")
