#!/usr/bin/env python
"""data (CPU): the eight new tasks for CLIP, preprocessed as the CLIP harness does (shorter side 224, centre crop). Writes, for each task:
results/clipdom/cache/<t>_train10000_{X,y}.npy and <t>_test2000_{X,y}.npy and <t>_names.json (what train_clip_client.py reads) and results/clipbreadth/cache/<t>_test_{X,y}.npy (symlinks to the same draw), <t>_cal_{X,y}.npy,
<t>_names.json, sizes.json[t] = images the client trained on. Seed 0 throughout, as the DC-Merge tasks. Sources: VTAB tasks from results/dino2/data (no new download); Caltech-101 and Tiny-ImageNet from the datasets cache;
FGVC Aircraft from the archive in results/clipnew/raw/aircraft (never unpacked to disk; images read from the tar). Usage: data_newtasks.py <task> [<task> ...]. Prints sizes and class counts only."""
from src.paths import ROOT as _OM_ROOT
import os, sys, json, random, tarfile, io, time, numpy as np
from concurrent.futures import ThreadPoolExecutor
os.environ.setdefault("DINO_ROOT", "dino2"); os.environ.setdefault("POOL", "dc16"); os.environ.setdefault("CLIPK_SUB", "k8")
ROOT = _OM_ROOT
from src.backbones import dino_data as DD
from PIL import Image
def prep(img):
    img = img.convert("RGB"); w, h = img.size; s = 224 / min(w, h); img = img.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC); w, h = img.size; l, t = (w - 224) // 2, (h - 224) // 2
    return np.asarray(img.crop((l, t, l + 224, t + 224)), dtype=np.uint8)
DOM = os.path.join(ROOT, "results", "clipdom", "cache"); CB = os.path.join(ROOT, "results", "clipbreadth", "cache"); os.makedirs(DOM, exist_ok=True); N_TRAIN, N_TEST, N_CAL = 10000, 2000, 200
def log(*a): print(time.strftime("%H:%M:%S", time.gmtime(time.time() + 9 * 3600)), *a, flush=True)
def draw(n, k): return sorted(random.Random(0).sample(range(n), min(k, n)))
VTAB = {"pcam", "clevr_count", "dsprites_orientation", "smallnorb_azimuth", "dmlab"}
def source(task):
    """returns (n_train, n_test, get(split, idx) -> (list of PIL images, labels), names)"""
    if task in VTAB:
        tr, te = DD.split(task, "train"), DD.split(task, "test"); lab, img, _ = DD.cols(task, tr); names = open(os.path.join(DD.DATA, task, "classnames.txt")).read().strip().split("\n")
        S = {"train": tr, "test": te}
        return len(tr), len(te), (lambda sp, idx: (lambda d: (d[img], d[lab]))(S[sp].select(idx))), names
    from datasets import load_dataset
    if task == "caltech101":
        tr, te = load_dataset("dpdl-benchmark/caltech101", split="train"), load_dataset("dpdl-benchmark/caltech101", split="test"); names = [n.replace("_", " ") for n in tr.features["label"].names]
    elif task == "tinyimagenet":
        tr, te = load_dataset("Maysee/tiny-imagenet", split="train"), load_dataset("Maysee/tiny-imagenet", split="valid"); M = {v[0]: v[1] for v in json.load(open(os.path.join(ROOT, "external", "clip_benchmark", "imagenet_class_index.json"))).values()}; M.update(json.load(open(os.path.join(ROOT, "external", "clip_benchmark", "tiny_imagenet_names_extra.json"))))   # 18 of the 200 ids are outside ImageNet-1K's index; their WordNet 3.0 lemmas
        names = [M[w].replace("_", " ") for w in tr.features["label"].names]
    if task in ("caltech101", "tinyimagenet"):
        S = {"train": tr, "test": te}; return len(tr), len(te), (lambda sp, idx: (lambda d: (d["image"], d["label"]))(S[sp].select(idx))), names
    if task == "aircraft":
        A = os.path.join(ROOT, "results", "clipnew", "raw", "aircraft", "fgvc-aircraft-2013b.tar.gz"); tf = tarfile.open(A, "r:gz"); files = {}; want = {"images_variant_trainval.txt", "images_variant_test.txt", "variants.txt"}; imgs = {}
        for m in tf:   # one sequential pass: the three lists, then every image's bytes
            base = os.path.basename(m.name)
            if base in want: files[base] = tf.extractfile(m).read().decode().strip().split("\n")
            elif m.name.endswith(".jpg") and "/data/images/" in m.name: imgs[base[:-4]] = tf.extractfile(m).read()
        tf.close(); names = files["variants.txt"]; ix = {n: i for i, n in enumerate(names)}; rows = {sp: [(l.split(" ", 1)[0], ix[l.split(" ", 1)[1]]) for l in files[f"images_variant_{tag}.txt"]] for sp, tag in (("train", "trainval"), ("test", "test"))}
        def get(sp, idx): r = [rows[sp][i] for i in idx]; return [Image.open(io.BytesIO(imgs[a])) for a, _ in r], [b for _, b in r]
        return len(rows["train"]), len(rows["test"]), get, names
    raise ValueError(task)
for t in sys.argv[1:]:
    t0 = time.time(); n_tr, n_te, get, names = source(t); assert len(set(names)) == len(names), t
    for sp, n, f_dom, tag_cb in (("test", N_TEST, f"{t}_test2000", "test"), ("train", N_CAL, None, "cal"), ("train", N_TRAIN, f"{t}_train10000", None)):
        if f_dom and os.path.exists(os.path.join(DOM, f_dom + "_X.npy")) and (tag_cb is None or os.path.lexists(os.path.join(CB, f"{t}_{tag_cb}_X.npy"))): continue
        if f_dom is None and os.path.exists(os.path.join(CB, f"{t}_{tag_cb}_X.npy")): continue
        idx = draw(n_tr if sp == "train" else n_te, n); im, lab = get(sp, idx)
        with ThreadPoolExecutor(8) as ex: X = np.stack(list(ex.map(prep, im, chunksize=16)))
        y = np.asarray(lab, dtype=np.int64); assert y.min() >= 0 and y.max() < len(names), (t, sp)
        if f_dom:
            np.save(os.path.join(DOM, f_dom + "_X.npy"), X); np.save(os.path.join(DOM, f_dom + "_y.npy"), y)
            if tag_cb:
                for s_ in ("X", "y"): os.symlink(os.path.join(DOM, f"{f_dom}_{s_}.npy"), os.path.join(CB, f"{t}_{tag_cb}_{s_}.npy"))
        else: np.save(os.path.join(CB, f"{t}_{tag_cb}_X.npy"), X); np.save(os.path.join(CB, f"{t}_{tag_cb}_y.npy"), y)
        log(f"{t} {sp} draw {len(idx)} of {n_tr if sp == 'train' else n_te}, labels seen {len(set(lab))} of {len(names)} ({time.time() - t0:.0f}s)")
    json.dump(names, open(os.path.join(DOM, f"{t}_names.json"), "w")); json.dump(names, open(os.path.join(CB, f"{t}_names.json"), "w"))
    sz = json.load(open(os.path.join(CB, "sizes.json"))); sz[t] = min(n_tr, N_TRAIN); json.dump(sz, open(os.path.join(CB, "sizes.json"), "w")); log(f"{t} done: {n_tr} train, {n_te} test, {len(names)} classes ({time.time() - t0:.0f}s)")
