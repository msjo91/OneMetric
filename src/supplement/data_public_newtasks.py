#!/usr/bin/env python
"""data, CPU: public images of the same task for the eight new CLIP tasks, by the rule of (data_public_clip.py --stage pub):
200 images from the test split outside the stored 2,000-image test draw (seed 0), drawn with seed 1; else from the train split outside the client's training and calibration draws.
Sources and preprocessing as data_newtasks.py. Check per task: the calibration draw re-made from the source equals the stored one.
Writes results/clipclose/protect/cache_pubnew/<task>_pub_{X,y}.npy and src_new.json; the eleven benchmark tasks of the new-task pools are linked from cache. Stops if a target exists. PN_DRY=1: sizes only."""
from src.paths import ROOT as _OM_ROOT
import os, sys, json, random, tarfile, io, time, numpy as np
from concurrent.futures import ThreadPoolExecutor
os.environ.setdefault("DINO_ROOT", "dino2"); ROOT = _OM_ROOT
from src.backbones import dino_data as DD
DD.DATA = os.path.join(ROOT, "results", "dino2", "data")
from PIL import Image
def prep(img):
    img = img.convert("RGB"); w, h = img.size; s = 224 / min(w, h); img = img.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC); w, h = img.size; l, t = (w - 224) // 2, (h - 224) // 2
    return np.asarray(img.crop((l, t, l + 224, t + 224)), dtype=np.uint8)
CB = os.path.join(ROOT, "results", "clipbreadth", "cache"); OUT = os.path.join(ROOT, "results", "clipclose", "protect", "cache_pubnew"); ZU = os.path.join(ROOT, "results", "clipclose", "zeroup", "cache"); DRY = os.environ.get("PN_DRY") == "1"
VTAB = {"pcam", "clevr_count", "dsprites_orientation", "smallnorb_azimuth", "dmlab"}; NEW = ["caltech101", "aircraft", "tinyimagenet", "pcam", "clevr_count", "dsprites_orientation", "smallnorb_azimuth", "dmlab"]
OLD = ["cifar100", "eurosat", "gtsrb", "mnist", "oxfordpets", "resisc45", "sst2", "stanford_cars", "stl10", "sun397", "svhn"]
def draw(n, k): return sorted(random.Random(0).sample(range(n), min(k, n)))
def source(task):
    if task in VTAB:
        tr, te = DD.split(task, "train"), DD.split(task, "test"); lab, img, _ = DD.cols(task, tr); S = {"train": tr, "test": te}
        return len(tr), len(te), (lambda sp, idx: (lambda d: (d[img], d[lab]))(S[sp].select(idx)))
    from datasets import load_dataset
    if task == "caltech101": tr, te = load_dataset("dpdl-benchmark/caltech101", split="train"), load_dataset("dpdl-benchmark/caltech101", split="test")
    elif task == "tinyimagenet": tr, te = load_dataset("Maysee/tiny-imagenet", split="train"), load_dataset("Maysee/tiny-imagenet", split="valid")
    if task in ("caltech101", "tinyimagenet"): S = {"train": tr, "test": te}; return len(tr), len(te), (lambda sp, idx: (lambda d: (d["image"], d["label"]))(S[sp].select(idx)))
    A = os.path.join(ROOT, "results", "clipnew", "raw", "aircraft", "fgvc-aircraft-2013b.tar.gz"); tf = tarfile.open(A, "r:gz"); files = {}; want = {"images_variant_trainval.txt", "images_variant_test.txt", "variants.txt"}; imgs = {}
    for m in tf:
        base = os.path.basename(m.name)
        if base in want: files[base] = tf.extractfile(m).read().decode().strip().split("\n")
        elif m.name.endswith(".jpg") and "/data/images/" in m.name: imgs[base[:-4]] = tf.extractfile(m).read()
    tf.close(); ix = {n: i for i, n in enumerate(files["variants.txt"])}; rows = {sp: [(l.split(" ", 1)[0], ix[l.split(" ", 1)[1]]) for l in files[f"images_variant_{tag}.txt"]] for sp, tag in (("train", "trainval"), ("test", "test"))}
    def get(sp, idx): r = [rows[sp][i] for i in idx]; return [Image.open(io.BytesIO(imgs[a])) for a, _ in r], [b for _, b in r]
    return len(rows["train"]), len(rows["test"]), get
if not DRY:
    os.makedirs(OUT, exist_ok=True); assert not any(os.path.lexists(os.path.join(OUT, f"{t}_pub_{s}.npy")) for t in NEW + OLD for s in "Xy"), "a target exists"
src = {}
for t in NEW:
    t0 = time.time(); n_tr, n_te, get = source(t)
    _, yc = get("train", draw(n_tr, 200)); assert np.array_equal(np.asarray(yc, dtype=np.int64), np.load(os.path.join(CB, f"{t}_cal_y.npy"))), f"{t}: the calibration draw does not reproduce"
    rest = sorted(set(range(n_te)) - set(draw(n_te, 2000))); where = "test"
    if len(rest) < 200: rest = sorted(set(range(n_tr)) - set(draw(n_tr, 10000)) - set(draw(n_tr, 200))); where = "train"
    assert len(rest) >= 200, (t, n_tr, n_te, len(rest))
    idx = sorted(random.Random(1).sample(rest, 200)); print(time.strftime("%H:%M:%S"), t, "train", n_tr, "test", n_te, "public from", where, "available", len(rest), flush=True)
    src[t] = {"source": where, "n_train_split": n_tr, "n_test_split": n_te, "n_outside_draw": len(rest), "idx": idx}
    if DRY: continue
    im, lab = get(where, idx)
    with ThreadPoolExecutor(8) as ex: X = np.stack(list(ex.map(prep, im, chunksize=16)))
    y = np.asarray(lab, dtype=np.int64); assert X.shape == (200, 224, 224, 3) and X.dtype == np.uint8
    Xt = np.load(os.path.join(ROOT, "results", "clipdom", "cache", f"{t}_test2000_X.npy"), mmap_mode="r"); Xc = np.load(os.path.join(CB, f"{t}_cal_X.npy")); h = lambda A: {a.tobytes()[:4096] + a.tobytes()[-4096:] for a in A}
    dup = len(h(X) & (h(Xt) | h(Xc))); np.save(os.path.join(OUT, f"{t}_pub_X.npy"), X); np.save(os.path.join(OUT, f"{t}_pub_y.npy"), y)
    src[t]["classes_seen"] = len(set(y.tolist())); src[t]["same_bytes_as_test_or_calibration_image"] = dup; print("   saved", X.shape, "classes", len(set(y.tolist())), "images equal to a test or calibration image:", dup, f"({time.time() - t0:.0f}s)", flush=True)
if not DRY:
    for t in OLD:
        for s in "Xy": os.symlink(os.path.join(ZU, f"{t}_pub_{s}.npy"), os.path.join(OUT, f"{t}_pub_{s}.npy"))
    json.dump(src, open(os.path.join(OUT, "src_new.json"), "w")); print("written")
