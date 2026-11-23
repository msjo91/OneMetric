#!/usr/bin/env python
"""CT-Merging, PACT-Iso-C and ACE-Merging on the public LLaVA pool. Stages build and build_ace write the three merges; llava_eval_extra evaluates them."""
from src.paths import ROOT as _OM_ROOT
import os, sys, json, time, glob, argparse, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="build"); ap.add_argument("--only", default=None); ap.add_argument("--dry", action="store_true"); args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = _OM_ROOT
from src.llava import llava_pool as V# noqa: E402
from src.baselines import merges_ext as ops_new# noqa: E402
M = V.M; LP = V.LP; log = V.log; PREREG = ""
OUT = os.path.join(ROOT, "results", "f25nm", "vlm"); os.makedirs(OUT, exist_ok=True); PK = os.path.join(OUT, "picks_f25nm.json"); BT = os.path.join(OUT, "build_times.json")
UA = os.path.join(ROOT, "results", "unionadopt", "vlm", "FULL"); W0DIR = os.path.join(V.ROOT, "_pact_w0"); V0P = os.path.join(W0DIR, "V0.npz")
PUB12 = ["concat", "fedavg_w", "fedit", "ties64", "knots_ties", "tsvm", "dcmerge", "isoc", "isocts_white", "pub_ct", "pub_pact_isoc", "pub_ace"]
GAMMAS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.3); SS = (0.8, 0.9, 1.1)
def pct(x): return int(round(x * 100))
NEW_CT = [f"ct_x{pct(g)}" for g in GAMMAS]; NEW_PD = [f"pact_x{pct(s)}" for s in SS] + [f"dc_x{pct(s)}" for s in SS]; NEW15 = NEW_CT + NEW_PD
K_CT, R_CT, PACT_K = 8, 15, 4; ACE_CFG = {"eps": 1e-5, "k_frac": 0.5, "tau": 0.03}; ACE_SCALE = 1.0
def ours_name(): return json.load(open(os.path.join(UA, "picks.json")))["final"]
def desc(rule): return f"rule={rule} (F25-NM on the VLM pool)"
def free_gpu():
    import gc; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception as e: log(f"free_gpu: {e}")
def times_update(d):
    T = json.load(open(BT)) if os.path.exists(BT) else {}; T.update(d); json.dump(T, open(BT, "w"), indent=1)
_pool = {}
def pool_projs():
    if not _pool: f, p = V.load_pool(); _pool.update(f=f, p=p)
    return _pool["f"], _pool["p"]
def cells_of(pool):
    cs = sorted(next(iter(pool.values())).keys()); return cs[:int(os.environ["F25N_DRY_CELLS"])] if args.dry and os.environ.get("F25N_DRY_CELLS") else cs
# ---------------- the three 2026 merges, LoRA form ----------------
def whiten(X):
    Lm, _, Rt = np.linalg.svd(X, full_matrices=False); return Lm @ Rt
def ct_cell(fs, k=K_CT, r=R_CT):
    T = len(fs); Us, Ss, Vs = [], [], []
    for f in fs:
        U, S, Vv = M._lowrank_svd(f["B"], f["A"]); o = np.argsort(-S); Us.append(U[:, o]); Ss.append(S[o]); Vs.append(Vv[:, o])
    P = sum(U @ U.T for U in Us) / T; w, E = np.linalg.eigh(P); Uc = E[:, -k:]
    Cm = sum(Uc.T @ f["B"] @ f["A"] for f in fs) / T; Pc, Sc, VcT = np.linalg.svd(Cm, full_matrices=False); Uc = Uc @ Pc; Vc = VcT.T
    rho_c = np.sqrt(float((Sc ** 2).sum()) / k); Ub, Vr, rhos = [], [], []
    for U, S, Vv in zip(Us, Ss, Vs):
        rr = min(r, len(S)); u, s, v = U[:, :rr], S[:rr], Vv[:, :rr]
        z = u - Uc @ (Uc.T @ u); a = np.maximum(np.linalg.norm(z, axis=0), 1e-12); ub = z / a; sb = a * s
        rhos.append((np.sqrt(float((sb ** 2).sum()) / rr), rr)); Ub.append(ub); Vr.append(v)
    Ut = whiten(np.concatenate([Uc] + Ub, axis=1)); Vt = whiten(np.concatenate([Vc] + Vr, axis=1))
    D = np.concatenate([[rho_c] * k] + [[rt] * rr for rt, rr in rhos]); return {"A": Vt.T, "B": Ut * D}
def w0_index():
    snaps = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/*/model.safetensors.index.json")); assert snaps, "LLaVA snapshot missing"
    snap = os.path.dirname(snaps[0]); idx = json.load(open(snaps[0]))["weight_map"]; return snap, idx
def v0_of(cells):
    """top-K right singular vectors of the frozen language-model weight of each cell (randomized SVD, torch); cached when computed for every cell"""
    if os.path.exists(V0P) and not args.dry:
        z = dict(np.load(V0P)); 
        if all(f"{l}.{m}" in z for l, m in cells): return {c: z[f"{c[0]}.{c[1]}"].astype(np.float64) for c in cells}
    import torch; from safetensors import safe_open
    snap, idx = w0_index(); out = {}; t0 = time.time()
    for l, m in cells:
        keys = [k for k in idx if k.endswith(f"layers.{l}.{m}.weight") and "language_model" in k]; assert len(keys) == 1, (l, m, keys)
        with safe_open(os.path.join(snap, idx[keys[0]]), framework="pt") as f: Wt = f.get_tensor(keys[0]).float()
        K = max(1, int(0.01 * Wt.shape[1])); torch.manual_seed(0); _, _, Vv = torch.svd_lowrank(Wt, q=min(K + 16, min(Wt.shape)), niter=4); out[(l, m)] = Vv[:, :K].double().numpy()
    log(f"  V0 for {len(cells)} cells ({time.time() - t0:.0f}s)")
    if not args.dry and len(cells) == len(cells_of(pool_projs()[0])): os.makedirs(W0DIR, exist_ok=True); np.savez(V0P, **{f"{l}.{m}": v.astype(np.float32) for (l, m), v in out.items()})
    return out
def fit_rank(f, rank):
    """factors of rank > rank are SVD-truncated (as subpool_rob.fit64); CT-Merging has k + K r = 113 directions on this seven-client pool, the adapter budget is 112"""
    out = {}
    for c, fs in f.items():
        if fs["A"].shape[0] <= rank: out[c] = fs; continue
        Qb, Rb = np.linalg.qr(fs["B"]); Qa, Ra = np.linalg.qr(fs["A"].T); U, S, Vt = np.linalg.svd(Rb @ Ra.T)
        out[c] = {"B": Qb @ (U[:, :rank] * np.sqrt(S[:rank])), "A": (np.sqrt(S[:rank])[:, None] * Vt[:rank]) @ Qa.T}
    return out
def build_ct(pool, cells): return fit_rank({c: ct_cell([pool[t][c] for t in V.CLIENTS]) for c in cells}, V.RANK)
def build_pact(pool, cells):
    V0 = v0_of(cells); fpool = {t: {} for t in V.CLIENTS}
    for c in cells:
        v0 = V0[c]; LBW = []
        for t in V.CLIENTS:
            A = np.asarray(pool[t][c]["A"], dtype=np.float64); _, _, Vat = np.linalg.svd(A, full_matrices=False); Vt = Vat[:PACT_K].T
            Q, _ = np.linalg.qr(v0 - Vt @ (Vt.T @ v0)); LBW.append(Q)
        for i, t in enumerate(V.CLIENTS):
            P, _ = np.linalg.qr(np.hstack([LBW[s] for s in range(len(V.CLIENTS)) if s != i])); A = np.asarray(pool[t][c]["A"], dtype=np.float64)
            fpool[t][c] = {"A": A - (A @ P) @ P.T, "B": np.asarray(pool[t][c]["B"], dtype=np.float64)}
    return M.merge_isoc(fpool, rank=V.RANK)
def build_ace(pool, cells):
    f = {}; t0 = time.time()
    for i, c in enumerate(cells):
        ds = [{c: (np.asarray(pool[t][c]["B"], dtype=np.float64) @ np.asarray(pool[t][c]["A"], dtype=np.float64))} for t in V.CLIENTS]
        mv = ACE_SCALE * ops_new.ace_merging(ds, **ACE_CFG)[c]; A, B = M._svd_factorize(np.asarray(mv, dtype=np.float32), V.RANK); f[c] = {"A": A, "B": B}
        if (i + 1) % 14 == 0: log(f"  ace {i + 1}/{len(cells)} cells ({time.time() - t0:.0f}s)")
    return f
# ---------------- stages ----------------
def stage_build_ace():
    if V.done("pub_ace") and not args.dry: log("pub_ace exists"); return
    pool, projs = pool_projs(); cells = cells_of(pool); t0 = time.time(); f = build_ace(pool, cells); dt = time.time() - t0; log(f"pub_ace built ({dt:.0f}s, {len(cells)} cells)")
    if not args.dry: V.write("pub_ace", f, desc(f"ace_merging {ACE_CFG} scale {ACE_SCALE} (released ViT-B configuration), rank-{V.RANK} randomized SVD"), LP.mean_projector(projs)); times_update({"pub_ace": dt})
def stage_build():
    pool, projs = pool_projs(); cells = cells_of(pool); pj = LP.mean_projector(projs); T = {}
    if args.dry or not V.done("pub_ct"):
        t0 = time.time(); f = build_ct(pool, cells); T["pub_ct"] = time.time() - t0; log(f"pub_ct built ({T['pub_ct']:.0f}s, rank {next(iter(f.values()))['A'].shape[0]})")
        if not args.dry: V.write("pub_ct", f, desc(f"ct_merging k={K_CT} r={R_CT} gamma=1 (factor space)"), pj)
    if args.dry or not V.done("pub_pact_isoc"):
        t0 = time.time(); f = build_pact(pool, cells); T["pub_pact_isoc"] = time.time() - t0; log(f"pub_pact_isoc built ({T['pub_pact_isoc']:.0f}s)")
        if not args.dry: V.write("pub_pact_isoc", f, desc(f"pact_isoc (PACT protection filter in factor space, K=int(.01*d_in), k={PACT_K}; rank-{V.RANK} Iso-C)"), pj)
    if args.dry:
        t0 = time.time(); build_ace(pool, cells); T["pub_ace"] = time.time() - t0; log("dry run ok: " + " ".join(f"{k}={v:.0f}s" for k, v in T.items())); return
    for base, names, scales in (("pub_ct", NEW_CT, GAMMAS), ("pub_pact_isoc", NEW_PD[:3], SS), ("dcmerge", NEW_PD[3:], SS)):
        if all(V.done(n) for n in names): continue
        if not V.done(base): rebuild_pub(base, pool, pj)
        f = V.read(base)
        for s, n in zip(scales, names):
            if V.done(n): continue
            t0 = time.time(); V.write(n, {c: {"A": v["A"], "B": s * v["B"]} for c, v in f.items()}, desc(f"scale {s} * {base}"), pj); T[n] = time.time() - t0
    times_update(T); log(f"build done ({sum(T.values()):.0f}s)")
def rebuild_pub(name, pool, pj):
    """Re-create a published merge whose adapter was pruned after (its rule-P scores in V.SEL were kept): the recipe of llava_pool.stage_build (mm_merges.py RULES, rank V.RANK)."""
    t0 = time.time(); fn, kw = M.RULES[name]; kw = dict(kw)
    if "rank" in kw: kw["rank"] = V.RANK
    f = fn(pool, **kw)
    for cell, fs in f.items(): assert np.isfinite(fs["A"]).all() and np.isfinite(fs["B"]).all(), f"{name}: non-finite at {cell}"
    V.write(name, f, f"rule={name} kwargs={kw} (mm_merges.py; rebuilt after pruning)", pj); times_update({f"rebuild_{name}": time.time() - t0}); log(f"{name} rebuilt ({time.time() - t0:.0f}s)")
def base_scale(n):
    s = scl(n) / 100; return (("pub_ct" if n.startswith("ct_x") else "pub_pact_isoc" if n.startswith("pact_x") else "dcmerge"), s)
def rebuild_new(n):
    """Re-create a rescaled copy pruned after scoring (deterministic: scale * base)."""
    pool, projs = pool_projs(); pj = LP.mean_projector(projs); base, s = base_scale(n)
    if not V.done(base): rebuild_pub(base, pool, pj)
    f = V.read(base); V.write(n, {c: {"A": v["A"], "B": s * v["B"]} for c, v in f.items()}, desc(f"scale {s} * {base}"), pj); log(f"{n} rebuilt from {base}")
def ensure_merge(n):
    """Make sure the adapter of candidate n exists on disk: a rescaled copy pruned after scoring, a published merge pruned after (the RULES recipe), or the two factor averages of llava_pool.stage_build."""
    if V.done(n): return
    if n in NEW15: rebuild_new(n); return
    pool, projs = pool_projs(); pj = LP.mean_projector(projs); cells = cells_of(pool)
    if n in M.RULES: rebuild_pub(n, pool, pj); return
    if n == "fedavg_w":
        tot = sum(V.SIZES[c] for c in V.CLIENTS); f = {cell: {"A": np.vstack([pool[c][cell]["A"] for c in V.CLIENTS]), "B": np.hstack([pool[c][cell]["B"] * (V.SIZES[c] / tot) for c in V.CLIENTS])} for cell in cells}
        V.write(n, f, f"rule=fedavg_w size-weighted average (rebuilt after pruning)", pj); log(f"{n} rebuilt"); return
    if n == "fedit":
        f = {cell: {"A": np.mean([pool[c][cell]["A"] for c in V.CLIENTS], axis=0), "B": np.mean([pool[c][cell]["B"] for c in V.CLIENTS], axis=0)} for cell in cells}
        V.write(n, f, f"rule=fedit separate averaging of A and B (rebuilt after pruning)", pj); log(f"{n} rebuilt"); return
    raise RuntimeError(f"no recipe to rebuild {n}")
def prune_dir(n):
    import shutil; d = V.mdir(n); sz = sum(os.path.getsize(os.path.join(d, x)) for x in os.listdir(d)) if os.path.isdir(d) else 0
    if os.path.isdir(d): shutil.rmtree(d)
    log(f"pruned {n} after scoring ({sz / 2 ** 30:.2f} GB)")
def score(names):
    todo = [n for n in names if not os.path.exists(os.path.join(V.SEL, f"fr_{n}.json"))]
    if args.only: todo = [n for n in todo if n in args.only.split(",")]
    if not todo: log(f"scores exist for {names}"); return
    for n in todo: assert V.done(n), f"merge missing for {n}"
    ensure_merge(V.TEMPLATE)   # the scorer's rank-112 template adapter (concat) was pruned after
    _, projs = pool_projs(); model, processor, LE = V.gpu_model(); cm = V.cell_modules(model); LP.apply_projector(model, LP.mean_projector(projs))
    pseudo = {c: json.load(open(os.path.join(V.SEL, f"pseudo_{c}.json")))["rows"] for c in V.CLIENTS}; ST = json.load(open(os.path.join(OUT, "score_s.json"))) if os.path.exists(os.path.join(OUT, "score_s.json")) else {}
    for n in todo:
        V.set_factors(model, cm, V.read(n)); res = {}; t0 = time.time()
        for c in V.CLIENTS:
            rows = pseudo[c]; outs = LE.generate(model, processor, rows, {"pad": LE.TASKS[V.task_of(c)]["pad"], "max_new_tokens": V.MAX_NEW}, V.MM, V.BS)
            pf = [V.f1(o, r["text"]) for o, r in zip(outs, rows)]; ex = [float(V.norm_tokens(o) == V.norm_tokens(r["text"])) for o, r in zip(outs, rows)]
            res[c] = {"pf1": float(np.mean(pf)), "exact": float(np.mean(ex)), "n": len(rows), "outs": outs, "refs": [r["text"] for r in rows]}
        json.dump(res, open(os.path.join(V.SEL, f"fr_{n}.json"), "w")); ST[n] = time.time() - t0; json.dump(ST, open(os.path.join(OUT, "score_s.json"), "w"), indent=1); log(f"scored {n} in {ST[n]:.0f}s")
        if n in NEW15: prune_dir(n)
    del model, cm; free_gpu()
def stage_score(): score([m for m in ["pub_ct", "pub_pact_isoc"] + NEW15])
def stage_score_ace(): score(["pub_ace"])
def cls(m, OURS): return 3 if m == OURS else 1
def scl(m): return int(m.rsplit("_x", 1)[1]) if "_x" in m and m.rsplit("_x", 1)[1].isdigit() else 100
def stage_pick():
    OURS = ours_name(); cands = list(dict.fromkeys([OURS] + PUB12 + NEW15)); S = V.scores(cands); missing = [m for m in cands if m not in S]; assert not missing, f"scores missing: {missing}"
    TOP = max(PUB12, key=lambda m: (round(S[m], 12), m)); pick = max(cands, key=lambda m: (round(S[m], 12), cls(m, OURS), scl(m)))
    ST = json.load(open(os.path.join(OUT, "score_s.json"))) if os.path.exists(os.path.join(OUT, "score_s.json")) else {}
    P = {"pool": "vlm/FULL", "ours": OURS, "top": TOP, "pick": pick, "changed": pick != OURS, "s_P": S, "s_P_ours": S[OURS], "s_P_top": S[TOP], "s_P_pick": S[pick], "score_s": list(ST.values()), "prereg": PREREG, "time": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(P, open(PK, "w"), indent=1); log(f"F25-NM pick {pick} ({'CHANGED' if P['changed'] else 'unchanged'}); ours {OURS} s_P {S[OURS]:.4f}; top published {TOP} s_P {S[TOP]:.4f}")
def stage_eval():
    P = json.load(open(PK)); A = json.load(open(os.path.join(UA, "analysis.json"))); P["R_ours"] = A["R"]["final"]; P["R_pp"] = A["R"]["pp"]; P["R_best_pub"] = A["R"]["best_pub"]; P["n"] = A["n"]
    if not P["changed"]: P["R_pick"] = P["R_ours"]; P["d"] = 0.0; P["evaluated"] = False; log("pick unchanged; no evaluation")
    else:
        free_gpu()
        ensure_merge(P["pick"]); ensure_merge(V.TEMPLATE)
        V.stage_eval(f"merged:{P['pick']}"); home = {V.task_of(c): V.acc_of(f"client_{c}", V.task_of(c)) for c in V.CLIENTS}; R, _ = V.R_of(f"merged_{P['pick']}", home); P["R_pick"] = R; P["evaluated"] = True
        P["d"] = (R - P["R_ours"]) if R is not None else None; P["U_pick"] = V.U_of(f"merged_{P['pick']}"); log(f"pick {P['pick']} R {R} vs ours {P['R_ours']:.4f}")
    P["noharm"] = (P["d"] is not None and P["d"] >= -.01); json.dump(P, open(PK, "w"), indent=1)
def stage_analysis():
    P = json.load(open(PK)); S = P["s_P"]; T = json.load(open(BT)) if os.path.exists(BT) else {}
    L = [f", VLM FULL ({time.strftime('%Y-%m-%d %H:%M')}); ours {P['ours']} (s_P {P['s_P_ours']:.4f}); top published {P['top']} (s_P {P['s_P_top']:.4f}); F25-NM pick {P['pick']} ({'CHANGED' if P['changed'] else 'unchanged'})",
         f"R500: ours {P['R_ours']:.4f} | pick {P['R_pick'] if P['R_pick'] is None else round(P['R_pick'], 4)} (evaluated {P['evaluated']}) | d {P['d']} | RegMean++ {P['R_pp']:.4f} | best published (stored) {P['R_best_pub']:.4f} | C1 noharm {int(P['noharm'])}",
         "s_P by candidate:"] + [f"  {m:<16}{S[m]:.4f}{'  <- ours' if m == P['ours'] else ('  <- pick' if m == P['pick'] else '')}" for m in sorted(S, key=lambda m: -S[m])]
    L.append("build seconds: " + " ".join(f"{k}={v:.0f}" for k, v in T.items())); L.append(f"scoring seconds: {[int(x) for x in P['score_s']]}")
    open(os.path.join(OUT, "analysis.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
def stage_prune():
    P = json.load(open(PK)); keep = {P["pick"], P["ours"]}; freed = 0; n = 0
    for m in NEW15:
        if m in keep or not V.done(m): continue
        d = V.mdir(m); freed += sum(os.path.getsize(os.path.join(d, x)) for x in os.listdir(d)); import shutil; shutil.rmtree(d); n += 1
    log(f"pruned {n} NEW15 merges ({freed / 2 ** 30:.2f} GB); kept {sorted(keep)}")
ST = {"build_ace": stage_build_ace, "build": stage_build, "score": stage_score, "score_ace": stage_score_ace, "pick": stage_pick, "eval": stage_eval, "analysis": stage_analysis, "prune": stage_prune}
for s in (["build"] if args.dry else args.stage.split(",")): log(f"== F25-NM VLM stage {s}"); ST[s]()
