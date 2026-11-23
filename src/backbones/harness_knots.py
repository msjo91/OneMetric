"""CLIP pools of the eight KnOTS tasks with the public KnOTS adapters (ViT-B/32 and L/14). Same functions and stages as harness_clip. Env ARCH = B32 | L14."""
import os, sys, json, time, random, argparse, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
from src.backbones.clip_backend import ROOT, ARCH, NLAYERS, TASKS as TASKS8, CELLS, KNOTS_T, KNOTS_FT, KNOTS_FT_L14, load_adapter, Clip
from src.baselines import merges as ops
SMOKE = bool(os.environ.get("CLIPK_SMOKE")); TASKS = TASKS8[:3] if SMOKE else list(TASKS8); K = len(TASKS); R_CLIENT = 16; RANK = K * R_CLIENT
N_TEST = 64 if SMOKE else 2000; N_CAL = 16 if SMOKE else 200; LAM = 0.9; F2_THRESH = 5.0; BS = 128
BASE = os.path.join(ROOT, "results", "clipknots_smoke" if SMOKE else "clipknots"); CACHE = os.path.join(BASE, "cache"); RA = os.path.join(BASE, ARCH)
MERGES, SEL, EVAL = os.path.join(RA, "merges"), os.path.join(RA, "select"), os.path.join(RA, "eval")
for d in (CACHE, MERGES, SEL, EVAL): os.makedirs(d, exist_ok=True)
MIXA = [(0.25, "25"), (0.5, "50"), (0.625, "625"), (0.75, "75"), (0.875, "875")]
PUB = ["ta", "fedavg_w", "fedit", "ties", "knots_ties", "tsvm", "dcmerge", "isoc", "isocts"]
DS = {"stanford_cars": ("tanganke/stanford_cars", None), "dtd": ("tanganke/dtd", None), "eurosat": ("tanganke/eurosat", None), "gtsrb": ("tanganke/gtsrb", None),
      "mnist": ("ylecun/mnist", None), "resisc45": ("tanganke/resisc45", None), "sun397": ("tanganke/sun397", None), "svhn": ("ufldl-stanford/svhn", "cropped_digits")}
def log(*a): print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)
def jload(p): return json.load(open(p))
def jdump(p, o): json.dump(o, open(p, "w"), indent=1)
def cname(c): return f"{c[0]}.{c[1]}"
def mpath(m): return os.path.join(MERGES, m + ".npz")
def done(m): return os.path.exists(mpath(m))
def save_merge(m, d, meta=None):
    np.savez_compressed(mpath(m), **{cname(c): np.asarray(d[c], dtype=np.float16) for c in CELLS}); jdump(mpath(m) + ".json", meta or {})
def load_merge(m): z = np.load(mpath(m)); return {c: z[cname(c)].astype(np.float32) for c in CELLS}

# ---------------- data ----------------
def prep(img):
    from PIL import Image
    img = img.convert("RGB"); w, h = img.size; s = 224 / min(w, h)
    img = img.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC); w, h = img.size; l, t = (w - 224) // 2, (h - 224) // 2
    return np.asarray(img.crop((l, t, l + 224, t + 224)), dtype=np.uint8)
def xfile(t, tag, s): return os.path.join(CACHE, f"{t}_{tag}_{s}.npy")
def stage_data():
    from datasets import load_dataset; from concurrent.futures import ThreadPoolExecutor
    sizes = jload(os.path.join(CACHE, "sizes.json")) if os.path.exists(os.path.join(CACHE, "sizes.json")) else {}
    for t in TASKS:
        repo, cfg = DS[t]
        for split, n, tag in (("test", N_TEST, "test"), ("train", N_CAL, "cal")):
            if os.path.exists(xfile(t, tag, "X")) and (split == "test" or t in sizes): continue
            t0 = time.time(); ds = load_dataset(repo, cfg, split=split) if cfg else load_dataset(repo, split=split)
            labcol = "label" if "label" in ds.column_names else [c for c in ds.column_names if "label" in c][0]
            imgcol = "image" if "image" in ds.column_names else [c for c in ds.column_names if c != labcol][0]
            if split == "train": sizes[t] = len(ds); jdump(os.path.join(CACHE, "sizes.json"), sizes)
            if os.path.exists(xfile(t, tag, "X")): continue
            idx = sorted(random.Random(0).sample(range(len(ds)), min(n, len(ds)))); sub = ds.select(idx)
            with ThreadPoolExecutor(16) as ex: X = np.stack(list(ex.map(prep, sub[imgcol], chunksize=16)))
            y = np.asarray(sub[labcol], dtype=np.int64)
            if tag == "test":
                feat = ds.features[labcol]; names = [nm.replace("_", " ") for nm in (feat.names if hasattr(feat, "names") else [str(i) for i in range(int(y.max()) + 1)])]
                jdump(os.path.join(CACHE, f"{t}_names.json"), names)
            np.save(xfile(t, tag, "X"), X); np.save(xfile(t, tag, "y"), y); log(f"{t} {tag}: {len(idx)} of {len(ds)} ({time.time() - t0:.0f}s)")
    log(f"data done; train sizes {sizes}")
def load_x(t, tag): return np.load(xfile(t, tag, "X"), mmap_mode="r"), np.load(xfile(t, tag, "y"))
def names_of(t): return jload(os.path.join(CACHE, f"{t}_names.json"))

# ---------------- pool / predictor / build ----------------
def load_deltas():
    pool = {t: load_adapter(t) for t in TASKS}
    D = {t: {c: (pool[t][c]["B"] @ pool[t][c]["A"]).astype(np.float32) for c in CELLS} for t in TASKS}; return pool, D
def features(D):
    N = np.array([[float(np.linalg.norm(D[t][c])) for c in CELLS] for t in TASKS])  # K x cells
    F1 = float(np.median(N.max(0) / np.median(N, 0))); l0 = [i for i, c in enumerate(CELLS) if c[0] == 0]; F2 = float(np.max(np.median(N[:, l0], 0)))
    return {"F1": F1, "F2": F2, "target": "median" if F2 < F2_THRESH else "min", "pred_room": bool(F1 >= 10 and F2 < F2_THRESH), "pred_collapse": bool(F2 >= F2_THRESH),
            "clients": TASKS, "norm_sums": {t: float(np.sqrt((N[i] ** 2).sum())) for i, t in enumerate(TASKS)}, "rank": RANK}
def target(): return jload(os.path.join(RA, "predictor.json"))["target"]
def end(): return "isoc_white" if target() == "median" else "isoc_wmin"
def pre(): return "os" if target() == "median" else "om"
def grid(): return [end()] + [pre() + s for _, s in MIXA] + ["regmean128"]
def a_of(m): return 0.0 if m in ("isoc_white", "isoc_wmin") else (1.0 if m == "regmean128" else dict((pre() + s, a) for a, s in MIXA).get(m, -1.0))
def whiten(deltas, how):
    out = [dict() for _ in deltas]
    for c in CELLS:
        v = np.array([float(np.linalg.norm(d[c])) for d in deltas]); tgt = float(np.median(v)) if how == "median" else float(np.min(v))
        for i, d in enumerate(deltas): out[i][c] = d[c] * (tgt / max(v[i], 1e-12))
    return out

# ---------------- GPU helpers ----------------
_C = None
def clip():
    global _C
    if _C is None: _C = Clip("cuda:0")
    return _C
def heads(C): return {t: C.head(names_of(t), KNOTS_T[t]) for t in TASKS}
@torch.no_grad()
def predict(C, X, H): f = C.features(X, bs=BS); return (f @ H.T).argmax(-1).cpu().numpy()
def collect_grams(C, delta, X, layers):
    acc, cnt, hooks = {}, {}, []
    def mk(key):
        def hook(mod, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).double(); acc[key] = acc.get(key, 0) + x.T @ x; cnt[key] = cnt.get(key, 0) + x.shape[0]
        return hook
    for l in layers:
        sa = C.model.vision_model.encoder.layers[l].self_attn
        hooks.append(sa.q_proj.register_forward_pre_hook(mk((l, "qkv")))); hooks.append(sa.out_proj.register_forward_pre_hook(mk((l, "o"))))
    with torch.no_grad(): C.set_delta(delta); C.features(X, bs=BS)
    for h in hooks: h.remove()
    return {k: (acc[k] / cnt[k]).cpu().numpy() for k in acc}
def shrink(G): return LAM * G + (1 - LAM) * np.diag(np.diag(G))
def regmean_cell(c, D, grams):
    key = (c[0], "o" if c[1] == "out_proj" else "qkv"); gs = [shrink(grams[t][key]) for t in TASKS]; den = sum(gs); d = den.shape[0]
    M = den + 1e-6 * np.trace(den) / d * np.eye(d); N = sum(D[t][c].astype(np.float64) @ g for t, g in zip(TASKS, gs))
    return np.linalg.solve(M.T, N.T).T.astype(np.float32)
def truncate(m, k):
    out, kept = {}, []
    for c in CELLS:
        U, s, Vt = np.linalg.svd(m[c].astype(np.float64), full_matrices=False); kept.append(float((s[:k] ** 2).sum() / max((s ** 2).sum(), 1e-30)))
        out[c] = ((U[:, :k] * s[:k]) @ Vt[:k]).astype(np.float32)
    return out, kept


STAGES = {"data": stage_data}
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="build the test and calibration draws of the tasks"); ap.add_argument("stage", choices=list(STAGES)); a = ap.parse_args(); STAGES[a.stage]()
