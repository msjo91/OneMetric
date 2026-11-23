#!/usr/bin/env python
"""membership audit of what a CLIP client uploads. GPU, no merge. Env ARCH, CLIPK_SUB (a pool; every task of it not yet audited is done), POOL=dc16.
For a client (task t, its adapter): the uploaded Grams are H.collect_grams on its 200 calibration images (24 matrices: 12 layers x the input of q/k/v and of out_proj), exactly what the method reads.
Attack (the server, who knows the adapter): for a candidate image z with token inputs X_z (T x d) at a matrix, S = X_z (n G + ridge)^-1 X_z^T with n = 200 T. If z is one of the 200, n G - X_z^T X_z is positive
semi-definite, so the largest eigenvalue of S is at most 1 and typically small; for another image it is larger. Statistics per image: mean over the 24 matrices of the largest eigenvalue (primary), mean over
matrices of trace(S) / T, largest over matrices of the largest eigenvalue. Lower means member. AUC = P(a non-member scores higher than a member).
Non-members: (train) the 300 further training images of 500-image set, same split as the members, sixteen benchmark tasks; (public) the task's 200 public images (/ ), test split.
Variants of the upload: clean; noise at rho in .01 .03 .1 .3 (gram_noise.py; the attacker adds the noise's spectral scale as ridge); placebo (the Gram of the public images, in which neither the
calibration images nor the train non-members are members: the attack should read .5 on members against train non-members).
Thumbnails: top-1 re-identification of each thumbnail's original among the 200 calibration originals and the 200 public images, by cosine of the base model's image embedding, at 28, 56, 112, 160 pixels and for
the scrambled set of where the sets exist. Writes results/privaudit/dc/<arch>/<task>.json. PA_OUT overrides the output root (code check), PA_TASKS=<csv> limits the tasks.
PA_RHOS=<csv> replaces the noise levels (1,3,10,30 into PA_OUT=results/privaudit2); without it the levels are . Backup audit_gram_noise.py.bak_g148."""
from src.paths import ROOT as _OM_ROOT
import os, sys, json, time, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = _OM_ROOT; os.environ.setdefault("POOL", "dc16")
from src.backbones import harness_clip as H; from src.supplement import gram_noise as gnoise
ARCH, TASKS = H.ARCH, H.TASKS; OUTR = os.path.join(os.environ.get("PA_OUT") or os.path.join(ROOT, "results", "privaudit"), "dc", ARCH); os.makedirs(OUTR, exist_ok=True)
RHOS = [float(x) for x in os.environ["PA_RHOS"].split(",")] if os.environ.get("PA_RHOS") else [.01, .03, .1, .3]; P = os.path.join(ROOT, "results", "clipclose", "protect"); dev = torch.device("cuda:0"); BS = 32
def auc(mem, non): mem, non = np.asarray(mem)[:, None], np.asarray(non)[None, :]; return float(((non > mem).sum() + 0.5 * (non == mem).sum()) / (mem.size * non.size))
def tpr_at(mem, non, fpr): thr = np.quantile(non, fpr, method="lower") if fpr > 0 else np.min(non); return float(np.mean(np.asarray(mem) < thr))   # members called at this false-positive rate (lower score = member)
def first(*ps): return next((p for p in ps if os.path.exists(p)), None)
def score(C, delta, Ls, X):
    """Ls: {variant: {key: Cholesky factor (d x d, GPU, float64) of n G + ridge}}. Returns {variant: {stat: array over images}}."""
    acc = {v: {k: [] for k in Ls[v]} for v in Ls}; hooks = []
    def mk(key):
        def hook(mod, inp):
            x = inp[0].double(); xt = x.transpose(1, 2)
            for v in Ls:
                Y = torch.linalg.solve_triangular(Ls[v][key], xt, upper=False); S = Y.transpose(1, 2) @ Y
                acc[v][key].append(torch.stack([torch.linalg.eigvalsh(S)[:, -1], torch.diagonal(S, dim1=1, dim2=2).mean(1)], 1).cpu())
        return hook
    for l in range(H.NLAYERS):
        sa = C.model.vision_model.encoder.layers[l].self_attn; hooks.append(sa.q_proj.register_forward_pre_hook(mk((l, "qkv")))); hooks.append(sa.out_proj.register_forward_pre_hook(mk((l, "o"))))
    with torch.no_grad(): C.set_delta(delta); C.features(X, bs=BS)
    for h in hooks: h.remove()
    out = {}
    for v in Ls:
        A = torch.stack([torch.cat(acc[v][k]) for k in acc[v]]).numpy()   # keys x images x 2
        out[v] = {"lmax_mean": A[:, :, 0].mean(0), "trace_mean": A[:, :, 1].mean(0), "lmax_max": A[:, :, 0].max(0)}
    return out
def chol(M): return torch.linalg.cholesky(torch.as_tensor(M, device=dev, dtype=torch.float64))
pool, D = H.load_deltas(); C = H.clip(); want = os.environ["PA_TASKS"].split(",") if os.environ.get("PA_TASKS") else TASKS
for t in [t for t in TASKS if t in want]:
    oj = os.path.join(OUTR, f"{t}.json")
    try: os.mkdir(oj + ".claim")
    except FileExistsError: continue
    t0 = time.time(); Xc = H.load_x(t, "cal")[0]; assert len(Xc) == 200
    fpub = first(os.path.join(P, "cache_pubnew", f"{t}_pub_X.npy"), os.path.join(ROOT, "results", "clipclose", "zeroup", "cache", f"{t}_pub_X.npy")); Xp = np.load(fpub, mmap_mode="r")
    ftr = H.xfile(t, "caln500", "X"); Xn = np.load(ftr, mmap_mode="r")[200:] if os.path.exists(ftr) else None
    if Xn is not None: assert np.array_equal(np.load(ftr, mmap_mode="r")[:200], np.asarray(Xc)) and len(Xn) == 300, t
    G = H.collect_grams(C, D[t], Xc, range(H.NLAYERS)); Gp = H.collect_grams(C, D[t], Xp, range(H.NLAYERS)); k0 = next(iter(G)); d = G[k0].shape[0]
    # number of tokens per image: one forward of one image with a counting hook
    cnt = []; h = C.model.vision_model.encoder.layers[0].self_attn.q_proj.register_forward_pre_hook(lambda m, i: cnt.append(i[0].shape[1]))
    with torch.no_grad(): C.features(Xc[:1], bs=1)
    h.remove(); T = cnt[0]; n = 200 * T
    Ls = {"clean": {k: chol(n * G[k] + 1e-9 * np.trace(n * G[k]) / d * np.eye(d)) for k in G}, "placebo": {k: chol(n * Gp[k] + 1e-9 * np.trace(n * Gp[k]) / d * np.eye(d)) for k in Gp}}
    for r in RHOS: Ls[f"rho{r}"] = {k: chol(n * gnoise.noisy(G[k], r, t, k) + n * gnoise.spectral_scale(G[k], r) * np.eye(d)) for k in G}
    sm = score(C, D[t], Ls, Xc); sp = score(C, D[t], Ls, Xp); sn = score(C, D[t], Ls, Xn) if Xn is not None else None
    res = {"task": t, "arch": ARCH, "tokens": T, "d": d, "n_members": 200, "n_train_nonmembers": 0 if Xn is None else len(Xn), "n_public_nonmembers": len(Xp), "public_source": os.path.relpath(fpub, ROOT), "auc": {}, "tpr": {}, "scores_clean": {}}
    for v in Ls:
        res["auc"][v] = {}; res["tpr"][v] = {}
        for nm, s in (("train", sn), ("public", sp)):
            if s is None or (v == "placebo" and nm == "public"): continue
            res["auc"][v][nm] = {st: auc(sm[v][st], s[v][st]) for st in sm[v]}
            res["tpr"][v][nm] = {"at_fpr_0": tpr_at(sm[v]["lmax_mean"], s[v]["lmax_mean"], 0.0), "at_fpr_.01": tpr_at(sm[v]["lmax_mean"], s[v]["lmax_mean"], 0.01)}
    q = lambda a: [float(x) for x in np.quantile(a, [0, .25, .5, .75, 1])]
    res["scores_clean"] = {"members": q(sm["clean"]["lmax_mean"]), "public": q(sp["clean"]["lmax_mean"]), "train": q(sn["clean"]["lmax_mean"]) if sn else None, "members_lmax_max_largest": float(sm["clean"]["lmax_max"].max())}
    # thumbnails: re-identification by the base model's embedding
    C.set_delta(None); cand = torch.cat([C.features(Xc, bs=BS).float(), C.features(Xp, bs=BS).float()]); cand = cand / cand.norm(dim=-1, keepdim=True); res["link"] = {}
    for tag, cdir in (("28", "cache_low"), ("56", "cache_low56"), ("112", "cache_low112"), ("160", "cache_low160"), ("scrambled", "cache_scr")):
        f = os.path.join(P, cdir, f"{t}_pub_X.npy")
        if not os.path.exists(f): continue
        q_ = C.features(np.load(f, mmap_mode="r"), bs=BS).float(); q_ = q_ / q_.norm(dim=-1, keepdim=True); res["link"][tag] = float(((q_ @ cand.T).argmax(1).cpu().numpy() == np.arange(200)).mean())
    res["seconds"] = round(time.time() - t0, 1); res["done"] = True; H.jdump(oj + ".tmp", res); os.replace(oj + ".tmp", oj)
    H.log(f"audit {ARCH} {t}: {T} tokens, d {d}, train non-members {res['n_train_nonmembers']}, thumbnail sets {len(res['link'])} ({res['seconds']}s)")
H.log("audit done")
