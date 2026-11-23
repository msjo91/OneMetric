#!/usr/bin/env python
"""data, CPU: more training images per client for the Gram-size audit, for the fifteen audited CLIP benchmark tasks.
Nested in the stored sets: draw 0 (the 200 calibration images) and 500-image set (draw 0 plus 300 further training images, results/clipbreadth/cache/ncal_meta.json).
From the training images outside those 500, and outside the public images where those came from the training split, one fixed random order (random.Random(152)) of up to 4,800 images:
  rows 0..299     non-members: never enter a Gram
  rows 300..      further Gram images: a Gram of n images is built on the 500-image set plus rows 300 .. 300 + n - 500 (n = 2,000 uses 1,500 of them, n = 5,000 uses 4,500)
A task with fewer training images gets what there is; the levels it cannot reach are left out for it. Preprocessing as the harness (harness_clip.prep), stored as uint8.
Writes results/privaudit3/sets/<task>_extra_{X,y}.npy and meta.json (indices, sizes). Idempotent. GS_OUT overrides the directory, GS_TASKS=<csv> limits the tasks."""
from src.paths import ROOT as _OM_ROOT
import os, sys, time, json, random, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = _OM_ROOT; os.environ.setdefault("POOL", "dc16"); os.environ.setdefault("ARCH", "B32"); os.environ.setdefault("CLIPK_SUB", "all16")
from src.backbones import harness_clip as H
from datasets import load_dataset; from concurrent.futures import ThreadPoolExecutor
OUT = os.environ.get("GS_OUT") or os.path.join(ROOT, "results", "privaudit3", "sets"); os.makedirs(OUT, exist_ok=True); NNON, NADD, CH = 300, 4500, 400
PUBSRC = json.load(open(os.path.join(ROOT, "results", "clipclose", "zeroup", "cache", "src.json"))); NC = json.load(open(os.path.join(H.CACHE, "ncal_meta.json")))
mp = os.path.join(OUT, "meta.json"); meta = json.load(open(mp)) if os.path.exists(mp) else {}
want = os.environ["GS_TASKS"].split(",") if os.environ.get("GS_TASKS") else [t for t in H.TASKS if t not in H.RET_EXCL]
for t in want:
    fx, fy = os.path.join(OUT, f"{t}_extra_X.npy"), os.path.join(OUT, f"{t}_extra_y.npy")
    if os.path.exists(fx) and os.path.exists(fy) and t in meta: H.log(f"{t}: exists"); continue
    t0 = time.time(); repo, cfg = H.DS[t]; ds = load_dataset(repo, cfg, split="train") if cfg else load_dataset(repo, split="train")
    wds = repo.startswith("clip-benchmark/"); labcol = "cls" if wds else "label"; imgcol = ("png" if "png" in ds.column_names else "jpg") if wds else "image"
    n_tr = len(ds); draw0 = sorted(random.Random(0).sample(range(n_tr), min(200, n_tr))); ex500 = NC[t]["n500"]["extra_train_idx"]; y500 = H.load_x(t, "caln500")[1]
    lab = np.asarray(ds[labcol], dtype=np.int64)
    assert NC[t]["n500"]["train_size"] == n_tr and np.array_equal(lab[draw0], y500[:200]) and np.array_equal(lab[ex500], y500[200:]), f"{t}: the stored 500-image set does not reproduce"
    excl = set(draw0) | set(ex500) | (set(PUBSRC[t]["idx"]) if PUBSRC.get(t, {}).get("source") == "train" else set())
    rest = [i for i in range(n_tr) if i not in excl]; k = min(NNON + NADD, len(rest)); order = random.Random(152).sample(rest, k); assert k >= NNON, t
    X = np.empty((k, 224, 224, 3), dtype=np.uint8)
    with ThreadPoolExecutor(int(os.environ.get("GS_THREADS", "8"))) as ex:
        for i in range(0, k, CH):
            sub = ds.select(order[i:i + CH]); X[i:i + CH] = np.stack(list(ex.map(H.prep, sub[imgcol], chunksize=8)))
    np.save(fx + ".tmp.npy", X); os.replace(fx + ".tmp.npy", fx); np.save(fy + ".tmp.npy", lab[order]); os.replace(fy + ".tmp.npy", fy)
    meta[t] = {"train_size": n_tr, "rows": k, "non_member_rows": [0, NNON], "gram_rows": [NNON, k], "largest_gram": 500 + k - NNON, "train_idx": order, "public_from_training_split_excluded": PUBSRC.get(t, {}).get("source") == "train"}
    json.dump(meta, open(mp + ".tmp", "w")); os.replace(mp + ".tmp", mp); H.log(f"{t}: {k} rows written, largest Gram {meta[t]['largest_gram']} images ({time.time() - t0:.0f}s)")
H.log(f"image sets done for {len(want)} tasks")
