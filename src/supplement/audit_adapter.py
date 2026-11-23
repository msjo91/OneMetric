#!/usr/bin/env python
"""membership audit of the adapter alone, the upload that every merging method shares. GPU, no merge.
Env ARCH, CLIPK_SUB (a pool; every task of it not yet audited is done), POOL=dc16, AM_OUT (output root, default results/adaptermia), AM_TASKS=<csv>.
For a client (task t, its adapter) the attacker holds a candidate image with its label, the adapter and the base model. Statistics per image, all read as "lower means member":
  loss  cross-entropy of the client's model (base + adapter, the task's zero-shot head, CLIP's logit scale) at the true label   (primary; the loss-threshold attack)
  conf  minus the largest class probability (needs no label)
  cal   loss of the client's model minus loss of the base model on the same image (the loss calibrated by a reference model)
AUC = P(a non-member scores higher than a member). Placebo: loss and conf of the base model alone, which never saw the task; it measures how far the two image sets differ by themselves.
Members. Fifteen benchmark tasks (public DC-Merge adapters): the 500 training-split images of set (caln500; the first 200 are the calibration images). Their training set is not
published with them; these images are presumed members. Eight new tasks (our own adapters, data_newtasks.py): the adapter was trained on the seed-0 draw of 10,000 training
images, the calibration images are the seed-0 draw of 200; both draws are recomputed here, so membership of each calibration image is exact (all 200 are training images except on dmlab).
Non-members: the task's test draw (2,000 test-split images, the evaluation set); also the task's 200 public images (test split), and, where a new task has calibration images outside the
training draw (dmlab), those same-split images.
Writes <root>/dc/<arch>/<task>.json."""
from src.paths import ROOT as _OM_ROOT
import os, sys, json, time, random, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = _OM_ROOT; os.environ.setdefault("POOL", "dc16")
from src.backbones import harness_clip as H
ARCH, TASKS = H.ARCH, H.TASKS; OUTR = os.path.join(os.environ.get("AM_OUT") or os.path.join(ROOT, "results", "adaptermia"), "dc", ARCH); os.makedirs(OUTR, exist_ok=True); BS = 128
P = os.path.join(ROOT, "results", "clipclose", "protect")
NTR = {"caltech101": 3060, "tinyimagenet": 100000, "smallnorb_azimuth": 24300, "dsprites_orientation": 589824, "dmlab": 65550, "clevr_count": 63000, "pcam": 262144, "aircraft": 6667}   # training-split sizes of the eight new tasks (the log of data_newtasks.py)
def own_flags(t):   # which of the 200 calibration images are in the adapter's training draw: both draws as data_newtasks.draw (seed 0, sorted)
    n = NTR[t]; tr = set(random.Random(0).sample(range(n), min(10000, n))); cal = sorted(random.Random(0).sample(range(n), min(200, n))); return np.array([i in tr for i in cal])
def auc(mem, non): mem, non = np.asarray(mem)[:, None], np.asarray(non)[None, :]; return float(((non > mem).sum() + 0.5 * (non == mem).sum()) / (mem.size * non.size))
def tpr_at(mem, non, fpr): thr = np.quantile(non, fpr, method="lower"); return float(np.mean(np.asarray(mem) < thr))   # members called at this false-positive rate (lower score = member)
def first(*ps): return next((p for p in ps if os.path.exists(p)), None)
@torch.no_grad()
def stats(C, delta, X, y, W, scale):
    C.set_delta(delta); f = C.features(X, bs=BS).float(); lp = torch.log_softmax((scale * (f @ W.float().T)).double(), -1); yy = torch.as_tensor(np.asarray(y), device=lp.device).long()
    return {"loss": (-lp.gather(1, yy[:, None])[:, 0]).cpu().numpy(), "conf": (-lp.max(-1).values.exp()).cpu().numpy(), "acc": float((lp.argmax(-1) == yy).double().mean())}
pool, D = H.load_deltas(); C = H.clip(); HD = H.heads(C); scale = float(C.model.logit_scale.exp()); want = os.environ["AM_TASKS"].split(",") if os.environ.get("AM_TASKS") else TASKS
for t in [t for t in TASKS if t in want]:
    oj = os.path.join(OUTR, f"{t}.json")
    try: os.mkdir(oj + ".claim")
    except FileExistsError: continue
    t0 = time.time(); own = t in NTR
    if own: Xtr, ytr = H.load_x(t, "cal"); flags = own_flags(t); assert len(Xtr) == 200 == len(flags)
    else: Xtr, ytr = H.load_x(t, "caln500"); flags = np.ones(len(Xtr), dtype=bool); assert len(Xtr) == 500 and np.array_equal(np.asarray(Xtr[:200]), np.asarray(H.load_x(t, "cal")[0])), t
    Xte, yte = H.load_x(t, "test"); fpub = first(os.path.join(P, "cache_pubnew", f"{t}_pub_X.npy"), os.path.join(ROOT, "results", "clipclose", "zeroup", "cache", f"{t}_pub_X.npy")); Xp = np.load(fpub, mmap_mode="r"); yp = np.load(fpub[:-5] + "y.npy")
    S = {m: {nm: stats(C, (D[t] if m == "adapter" else None), X, y, HD[t], scale) for nm, (X, y) in (("train", (Xtr, ytr)), ("test", (Xte, yte)), ("public", (Xp, yp)))} for m in ("adapter", "base")}
    for nm in ("train", "test", "public"): S["adapter"][nm]["cal"] = S["adapter"][nm]["loss"] - S["base"][nm]["loss"]
    res = {"task": t, "arch": ARCH, "kind": "own adapter, training draw recomputed" if own else "public adapter, training split presumed", "logit_scale": scale, "n_train_set": int(len(Xtr)), "n_members": int(flags.sum()),
           "n_same_split_nonmembers": int((~flags).sum()), "n_test": int(len(Xte)), "n_public": int(len(Xp)), "public_source": os.path.relpath(fpub, ROOT),
           "acc": {m: {nm: S[m][nm]["acc"] for nm in S[m]} for m in S}, "auc": {}, "tpr01": {}, "auc_public": {}, "same_split": None}
    for m in S:
        st = [k for k in S[m]["train"] if k != "acc"]
        res["auc"][m] = {k: auc(S[m]["train"][k][flags], S[m]["test"][k]) for k in st}; res["tpr01"][m] = {k: tpr_at(S[m]["train"][k][flags], S[m]["test"][k], 0.01) for k in st}
        res["auc_public"][m] = {k: auc(S[m]["train"][k][flags], S[m]["public"][k]) for k in st}
    if flags.sum() >= 20 and (~flags).sum() >= 20:
        res["same_split"] = {"n_members": int(flags.sum()), "n_nonmembers": int((~flags).sum()), "auc": {m: {k: auc(S[m]["train"][k][flags], S[m]["train"][k][~flags]) for k in S[m]["train"] if k != "acc"} for m in S}}
    q = lambda a: [float(x) for x in np.quantile(a, [0, .25, .5, .75, 1])]
    res["loss_quartiles"] = {"members": q(S["adapter"]["train"]["loss"][flags]), "test": q(S["adapter"]["test"]["loss"]), "base_members": q(S["base"]["train"]["loss"][flags]), "base_test": q(S["base"]["test"]["loss"])}
    res["seconds"] = round(time.time() - t0, 1); res["done"] = True; H.jdump(oj + ".tmp", res); os.replace(oj + ".tmp", oj)
    H.log(f"adapter audit {ARCH} {t}: members {res['n_members']}, test {res['n_test']}, same-split non-members {res['n_same_split_nonmembers']} ({res['seconds']}s)")
H.log("adapter audit done")
