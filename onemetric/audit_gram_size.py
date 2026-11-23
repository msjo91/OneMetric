#!/usr/bin/env python
"""the membership attack of against Grams built on more of the client's training images. GPU, no merge.
Env ARCH, CLIPK_SUB (a pool; every benchmark task of it not yet audited is done), POOL=dc16, GA_OUT (output root, default results/privaudit3), GA_SETS (image sets, default results/privaudit3/sets), GA_TASKS=<csv>.
For a client (task t, its adapter) and a size n in 200, 500, 2,000, 5,000: the Gram is H.collect_grams over the first n images of the nested set (the 200 calibration images, 300
further images, then rows 300.. of data_gram_sizes.py), 24 matrices. The attack is unchanged: for a candidate image z with token inputs X_z, S = X_z (N G + ridge)^-1 X_z^T
with N = n T tokens; statistics per image: mean over the 24 matrices of the largest eigenvalue (primary), mean of trace(S) / T, largest over matrices of the largest eigenvalue; lower means member.
Members: the 200 calibration images, which are in every Gram. Non-members: rows 0..299 of the new image set, training images that are in no Gram.
Noise: at every size also the uploads with noise of 10 % and 30 % of the Gram's norm (gram_noise.py, the levels found free); the attacker adds the noise's spectral scale as ridge.
Reproduction: at n = 200 without noise, the members against own non-members (the 300 further images of the 500-image set) must give stored AUC.
A size a task cannot reach (fewer training images) is left out for that task; such a task gets its own largest Gram as its last level. Writes <root>/dc/<arch>/<task>.json."""
from onemetric.paths import ROOT as _OM_ROOT
import os, sys, json, time, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = _OM_ROOT; os.environ.setdefault("POOL", "dc16")
from onemetric import harness_clip as H; from onemetric import gram_noise as gnoise
ARCH, TASKS = H.ARCH, H.TASKS; OUTR = os.path.join(os.environ.get("GA_OUT") or os.path.join(ROOT, "results", "privaudit3"), "dc", ARCH); os.makedirs(OUTR, exist_ok=True)
SETS = os.environ.get("GA_SETS") or os.path.join(ROOT, "results", "privaudit3", "sets"); SIZES = [200, 500, 2000, 5000]; RHOS = [0.1, 0.3]; dev = torch.device("cuda:0"); BS = 32; NNON = 300
def auc(mem, non): mem, non = np.asarray(mem)[:, None], np.asarray(non)[None, :]; return float(((non > mem).sum() + 0.5 * (non == mem).sum()) / (mem.size * non.size))
def tpr_at(mem, non, fpr): thr = np.quantile(non, fpr, method="lower") if fpr > 0 else np.min(non); return float(np.mean(np.asarray(mem) < thr))   # members called at this false-positive rate (lower score = member)
def score(C, delta, Ls, X):
    """Ls: {variant: {key: Cholesky factor (d x d, GPU, float64) of N G + ridge}}. Returns {variant: {stat: array over images}}. As audit_gram_noise.py."""
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
pool, D = H.load_deltas(); C = H.clip(); want = os.environ["GA_TASKS"].split(",") if os.environ.get("GA_TASKS") else TASKS; meta = json.load(open(os.path.join(SETS, "meta.json")))
for t in [t for t in TASKS if t in want and t in meta]:
    oj = os.path.join(OUTR, f"{t}.json")
    try: os.mkdir(oj + ".claim")
    except FileExistsError: continue
    t0 = time.time(); Xc = H.load_x(t, "cal")[0]; X5 = H.load_x(t, "caln500")[0]; E = np.load(os.path.join(SETS, f"{t}_extra_X.npy"), mmap_mode="r"); assert len(Xc) == 200 and len(X5) == 500 and np.array_equal(np.asarray(X5[:200]), np.asarray(Xc)) and len(E) == meta[t]["rows"], t
    Xn = E[:NNON]; add = E[NNON:]; cap = 500 + len(add); sizes = [n for n in SIZES if n <= cap] + ([cap] if cap < SIZES[-1] and cap not in SIZES else [])   # a client with fewer than 5,000 images: its own largest Gram is the last level (what a merge at 5,000 would read from it)
    cnt = []; h = C.model.vision_model.encoder.layers[0].self_attn.q_proj.register_forward_pre_hook(lambda m, i: cnt.append(i[0].shape[1]))
    with torch.no_grad(): C.features(Xc[:1], bs=1)
    h.remove(); T = cnt[0]
    # Gram sums over nested prefixes: chunks (0,200], (200,500], (500,2000], (2000,5000]
    bounds = [0] + sizes; SUM = {}; run = None
    for a_, b_ in zip(bounds[:-1], bounds[1:]):
        chunk = X5[a_:b_] if b_ <= 500 else add[a_ - 500:b_ - 500]; assert len(chunk) == b_ - a_, (t, a_, b_, len(chunk))
        g = H.collect_grams(C, D[t], chunk, range(H.NLAYERS)); run = {k: (g[k] * (len(chunk) * T) + (run[k] if run else 0.0)) for k in g}; SUM[b_] = {k: v.copy() for k, v in run.items()}
    k0 = next(iter(run)); d = run[k0].shape[0]; Ls = {}
    if ARCH in ("B32", "B16") and os.environ.get("GA_SAVE", "1") == "1":
        GR = os.path.join(os.path.dirname(os.path.dirname(OUTR)), "grams", ARCH); os.makedirs(GR, exist_ok=True)
        for n in sorted({min(2000, cap), min(5000, cap)}):
            gf = os.path.join(GR, f"{t}_n{n}.npz"); np.savez(gf + ".tmp.npz", **{f"{k[0]}_{k[1]}": SUM[n][k] / (n * T) for k in SUM[n]}); os.replace(gf + ".tmp.npz", gf)
    for n in sizes:
        N = n * T; Ls[f"n{n}"] = {k: chol(SUM[n][k] + 1e-9 * np.trace(SUM[n][k]) / d * np.eye(d)) for k in SUM[n]}
        for r in RHOS: Ls[f"n{n}_rho{r}"] = {k: chol(N * gnoise.noisy(SUM[n][k] / N, r, t, k) + N * gnoise.spectral_scale(SUM[n][k] / N, r) * np.eye(d)) for k in SUM[n]}
    sm = score(C, D[t], Ls, Xc); sn = score(C, D[t], Ls, Xn); s146 = score(C, D[t], {"n200": Ls["n200"]}, X5[200:500])
    res = {"task": t, "arch": ARCH, "tokens": T, "d": d, "sizes": sizes, "rhos": RHOS, "n_members": 200, "n_nonmembers": int(len(Xn)), "largest_gram_possible": int(500 + len(add)), "auc": {}, "tpr": {}, "scores": {}}
    q = lambda a: [float(x) for x in np.quantile(a, [0, .25, .5, .75, 1])]
    for v in Ls:
        res["auc"][v] = {st: auc(sm[v][st], sn[v][st]) for st in sm[v]}; res["tpr"][v] = {"at_fpr_0": tpr_at(sm[v]["lmax_mean"], sn[v]["lmax_mean"], 0.0), "at_fpr_.01": tpr_at(sm[v]["lmax_mean"], sn[v]["lmax_mean"], 0.01)}
        res["scores"][v] = {"members": q(sm[v]["lmax_mean"]), "nonmembers": q(sn[v]["lmax_mean"])}
    res["members_lmax_max_largest"] = {f"n{n}": float(sm[f"n{n}"]["lmax_max"].max()) for n in sizes}
    res["repro_first_audit"] = {"auc": {st: auc(sm["n200"][st], s146["n200"][st]) for st in sm["n200"]}, "tpr": {"at_fpr_0": tpr_at(sm["n200"]["lmax_mean"], s146["n200"]["lmax_mean"], 0.0), "at_fpr_.01": tpr_at(sm["n200"]["lmax_mean"], s146["n200"]["lmax_mean"], 0.01)}}
    res["seconds"] = round(time.time() - t0, 1); res["done"] = True; H.jdump(oj + ".tmp", res); os.replace(oj + ".tmp", oj)
    H.log(f"size audit {ARCH} {t}: {T} tokens, d {d}, sizes {sizes} ({res['seconds']}s)"); del Ls, SUM, run; torch.cuda.empty_cache()
H.log("size audit done")
