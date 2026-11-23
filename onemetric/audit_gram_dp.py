#!/usr/bin/env python
"""Gram uploads with a formal guarantee (gram_dp.py), attacked by membership attack. GPU, no merge.
Env ARCH, CLIPK_SUB (a pool; every benchmark task of it not yet done is done), POOL=dc16, DG_OUT (output root, default results/privaudit4), DG_SETS (image sets), DG_TASKS=<csv>,
DG_EPS (default 8,3,1), DG_SAVE (default 1: store the ViT-B releases that the merges read).
For a client (task t, its adapter): the bounds C_k are the median of ||X_z^T X_z||_F over the task's 200 public images. Clipped sums over the client's 200 calibration images and over
5,000 training images (nested set; a client with fewer uses all it has). Releases, named as the merge variants read them:
  n200c, n5000c            the clipped sum, no noise (no guarantee; shows what clipping alone does)
  n200e<eps>, n5000e<eps>  the Gaussian mechanism at (eps, 1e-5) over the 24 matrices together
each divided by n T and projected to positive semi-definite. Attack as members the 200 calibration images, non-members the 300 held-out training images; statistics
largest eigenvalue (mean over matrices; primary), trace, largest over matrices; the attacker's ridge is the noise's spectral scale.
Writes <root>/dc/<arch>/<task>.json and, for ViT-B, <root>/grams/<arch>/<task>_<name>.npz (float32)."""
from onemetric.paths import ROOT as _OM_ROOT
import os, sys, json, time, math, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = _OM_ROOT; os.environ.setdefault("POOL", "dc16")
from onemetric import harness_clip as H; from onemetric import gram_dp as dpgram
ARCH, TASKS = H.ARCH, H.TASKS; BASE = os.environ.get("DG_OUT") or os.path.join(ROOT, "results", "privaudit4"); OUTR = os.path.join(BASE, "dc", ARCH); os.makedirs(OUTR, exist_ok=True)
SETS = os.environ.get("DG_SETS") or os.path.join(ROOT, "results", "privaudit3", "sets"); EPS = [float(x) for x in os.environ.get("DG_EPS", "8,3,1").split(",")]; DELTA = 1e-5; dev = torch.device("cuda:0"); BS = 32; NNON = 300
SAVE = os.environ.get("DG_SAVE", "1") == "1" and (ARCH in ("B32", "B16") or os.environ.get("DG_SAVEL14") == "1"); MERGED = os.environ["DG_MERGED"].split(",") if os.environ.get("DG_MERGED") else ["n200e8", "n5000c"] + [f"n5000e{e:g}" for e in EPS]; DSUF = f"d{dpgram._DRAW}" if dpgram._DRAW else ""; GRD = os.environ.get("DG_GRAMS") or os.path.join(BASE, "grams")   # DG_SAVEL14, DG_MERGED (which releases are stored), DG_DRAW (suffix d<i> on the stored name), DG_GRAMS (where); defaults unchanged
LAY = range(H.NLAYERS); SENS = math.sqrt(int(os.environ.get("DG_NKEYS", dpgram.NKEYS)))
def auc(mem, non): mem, non = np.asarray(mem)[:, None], np.asarray(non)[None, :]; return float(((non > mem).sum() + 0.5 * (non == mem).sum()) / (mem.size * non.size))
def tpr_at(mem, non, fpr): thr = np.quantile(non, fpr, method="lower") if fpr > 0 else np.min(non); return float(np.mean(np.asarray(mem) < thr))
def score(C, delta, Ls, X):
    """As audit_gram_noise.py: {variant: {stat: array over images}} for Cholesky factors Ls[variant][key] of N G + ridge."""
    acc = {v: {k: [] for k in Ls[v]} for v in Ls}; hooks = []
    def mk(key):
        def hook(mod, inp):
            x = inp[0].double(); xt = x.transpose(1, 2)
            for v in Ls:
                Y = torch.linalg.solve_triangular(Ls[v][key], xt, upper=False); S = Y.transpose(1, 2) @ Y
                acc[v][key].append(torch.stack([torch.linalg.eigvalsh(S)[:, -1], torch.diagonal(S, dim1=1, dim2=2).mean(1)], 1).cpu())
        return hook
    for l in LAY:
        sa = C.model.vision_model.encoder.layers[l].self_attn; hooks.append(sa.q_proj.register_forward_pre_hook(mk((l, "qkv")))); hooks.append(sa.out_proj.register_forward_pre_hook(mk((l, "o"))))
    with torch.no_grad(): C.set_delta(delta); C.features(X, bs=BS)
    for h in hooks: h.remove()
    out = {}
    for v in Ls:
        A = torch.stack([torch.cat(acc[v][k]) for k in acc[v]]).numpy()
        out[v] = {"lmax_mean": A[:, :, 0].mean(0), "trace_mean": A[:, :, 1].mean(0), "lmax_max": A[:, :, 0].max(0)}
    return out
def chol(M): return torch.linalg.cholesky(torch.as_tensor(M, device=dev, dtype=torch.float64))
assert os.environ.get("DG_NKEYS") is None or int(os.environ["DG_NKEYS"]) == 2 * H.NLAYERS, "DG_NKEYS must be the number of released matrices"
pool, D = H.load_deltas(); C = H.clip(); want = os.environ["DG_TASKS"].split(",") if os.environ.get("DG_TASKS") else TASKS; meta = json.load(open(os.path.join(SETS, "meta.json")))
SIG = {e: dpgram.sigma_analytic(e, DELTA, SENS) for e in EPS + ([8.0] if 8.0 not in EPS else [])}
for t in [t for t in TASKS if t in want and t in meta]:
    oj = os.path.join(OUTR, f"{t}.json")
    try: os.mkdir(oj + ".claim")
    except FileExistsError: continue
    t0 = time.time(); Xc = H.load_x(t, "cal")[0]; X5 = H.load_x(t, "caln500")[0]; E = np.load(os.path.join(SETS, f"{t}_extra_X.npy"), mmap_mode="r"); Xp = np.load(os.path.join(ROOT, "results", "clipclose", "zeroup", "cache", f"{t}_pub_X.npy"), mmap_mode="r")
    assert len(Xc) == 200 and len(X5) == 500 and np.array_equal(np.asarray(X5[:200]), np.asarray(Xc)) and len(Xp) == 200, t
    Xn = E[:NNON]; add = E[NNON:]; nbig = min(5000, 500 + len(add))
    nz = dpgram.image_norms(C, D[t], Xp, LAY); B = {k: float(np.median(v)) for k, v in nz.items()}; Bt = {k: torch.tensor(B[k], device=dev, dtype=torch.float64) for k in B}
    S200, cl200, T = dpgram.clipped_sums(C, D[t], Xc, Bt, LAY); Sb, clb, _ = dpgram.clipped_sums(C, D[t], X5[200:500], Bt, LAY); Sc, clc, _ = dpgram.clipped_sums(C, D[t], add[:nbig - 500], Bt, LAY)
    Sbig = {k: S200[k] + Sb[k] + Sc[k] for k in S200}; clbig = {k: (cl200[k] * 200 + clb[k] * 300 + clc[k] * (nbig - 500)) / nbig for k in S200}; d = next(iter(S200.values())).shape[0]
    REL = {"n200c": (S200, 200, 0.0, 0.0), "n5000c": (Sbig, nbig, 0.0, 0.0), "n200e8": (S200, 200, SIG[8.0], 8.0)}
    for e in EPS: REL[f"n5000e{e:g}"] = (Sbig, nbig, SIG[e], e)
    Ls = {}; info = {}
    for name, (S, n, sg, e) in REL.items():
        rel = {k: dpgram.psd(dpgram.release(S[k], B[k], sg, t, k, e, n)) for k in S}
        Ls[name] = {k: chol(rel[k] + (dpgram.spectral_scale(B[k], sg, d) if sg > 0 else 1e-9 * np.trace(rel[k]) / d) * np.eye(d)) for k in S}
        info[name] = {"images": int(n), "epsilon": e if sg > 0 else None, "delta": DELTA if sg > 0 else None, "sigma_in_units_of_the_bound": sg,
                      "noise_over_signal": float(np.mean([sg * B[k] * d / np.linalg.norm(S[k]) for k in S])) if sg > 0 else 0.0, "noise_over_signal_max": float(np.max([sg * B[k] * d / np.linalg.norm(S[k]) for k in S])) if sg > 0 else 0.0}
        if SAVE and name in MERGED:
            GR = os.path.join(GRD, ARCH); os.makedirs(GR, exist_ok=True); gf = os.path.join(GR, f"{t}_{name}{DSUF}.npz"); np.savez(gf + ".tmp.npz", **{f"{k[0]}_{k[1]}": (rel[k] / (n * T)).astype(np.float32) for k in rel}); os.replace(gf + ".tmp.npz", gf)
        del rel
    sm = score(C, D[t], Ls, Xc); sn = score(C, D[t], Ls, Xn)
    res = {"task": t, "arch": ARCH, "tokens": T, "d": d, "n_big": int(nbig), "eps": EPS, "delta": DELTA, "sensitivity": SENS, "n_members": 200, "n_nonmembers": int(len(Xn)), "releases": info,
           "share_clipped": {"n200": float(np.mean(list(cl200.values()))), "n5000": float(np.mean(list(clbig.values())))}, "bound_median_public": {f"{k[0]}_{k[1]}": B[k] for k in B},
           "clipped_over_unclipped_norm_public": float(np.mean([np.minimum(nz[k], B[k]).sum() / nz[k].sum() for k in nz])), "auc": {}, "tpr": {}}
    for v in Ls:
        res["auc"][v] = {st: auc(sm[v][st], sn[v][st]) for st in sm[v]}; res["tpr"][v] = {"at_fpr_0": tpr_at(sm[v]["lmax_mean"], sn[v]["lmax_mean"], 0.0), "at_fpr_.01": tpr_at(sm[v]["lmax_mean"], sn[v]["lmax_mean"], 0.01)}
    res["seconds"] = round(time.time() - t0, 1); res["done"] = True; H.jdump(oj + ".tmp", res); os.replace(oj + ".tmp", oj)
    H.log(f"guarantee audit {ARCH} {t}: {T} tokens, d {d}, {nbig} images in the large Gram ({res['seconds']}s)"); del Ls, S200, Sb, Sc, Sbig; torch.cuda.empty_cache()
H.log("guarantee audit done")
