#!/usr/bin/env python
"""The sequential and half-sequential Gram ends and the anchored points on a LLaVA pool. Stages: basescore | ppend | halfend | energy | anchorwrite | family --end pp|half | anchorscore | eval."""
from onemetric.paths import ROOT as _OM_ROOT
import os, sys, json, time, gc, glob, argparse, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True, choices=["ppend", "halfend", "basescore", "energy", "anchorwrite", "family", "anchorscore", "eval"])
ap.add_argument("--group", type=int, default=-1); ap.add_argument("--ends", default="pp"); ap.add_argument("--end", default=""); ap.add_argument("--what", default=""); args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = _OM_ROOT
from onemetric import llava_pool as V# noqa: E402
LP, log, MIXA = V.LP, V.log, V.MIXA
SMOKE = os.environ.get("OPT_SMOKE") == "1"   # code test: 8-bit model, 4 rows per client, 2 sequential layers, layer 0 only, a few points; every write under OPT_SCRATCH
OUT = os.path.join(os.environ["OPT_SCRATCH"], "vlm_opt") if SMOKE else os.path.join(ROOT, "results", "mfam", "vlm_opt"); AN = os.path.join(OUT, "anchor"); TMP = os.path.join(AN, "tmp"); LEN = os.path.join(ROOT, "results", "mfam", "vlm_len")
for d in (OUT, AN, TMP): os.makedirs(d, exist_ok=True)
POOLS = {"FULL": list(V.ALL_CLIENTS), **{k: list(v) for k, v in V.SUBS.items() if k.startswith("V")}}; CGRID = [0.03, 0.1, 0.3]
ENDM = {"pp": "regmeanpp_long", "half": "regmeanpp_half"}; CHECK = os.environ.get("OPT_CHECK") == "1" or SMOKE
if SMOKE:
    POOLS = {os.environ.get("OPT_SMOKE_POOL", "FULL"): POOLS[os.environ.get("OPT_SMOKE_POOL", "FULL")]}; NEWM = os.path.join(OUT, "merges"); os.makedirs(NEWM, exist_ok=True); _mdir = V.mdir
    def _new(m): return m == "regmeanpp_half" or m.endswith("pick_ss") or m.startswith("anc_N")
    V.mdir = lambda m: os.path.join(NEWM, m) if _new(m) else _mdir(m)
    _calib = V.calib_rows; V.calib_rows = lambda c, LE=None: _calib(c, LE)[:4]; _lmm = V.lmm_eval
    def _lmm8():
        LE = _lmm(); _lm = LE.load_model; LE.load_model = lambda spec, root, dtype="fp16", device="cuda": _lm(spec, root, dtype="8bit", device=device); return LE
    V.lmm_eval = _lmm8
def pool_root(P): return V.PARENT if P == "FULL" else os.path.join(V.PARENT, "sub", P)
def rank_of(P): return 16 * len(POOLS[P])
def cands(e): return ["mw", "mr"] + [f"{e}{s}{t}" for s in "wr" for _, t in MIXA] + [f"{e}_end"]
def a_of(m, e): return 0.0 if m in ("mw", "mr") else (1.0 if m == f"{e}_end" else dict((t, a) for a, t in MIXA)[m[len(e) + 1:]])
def side_of(m, e): return "r" if (m == "mr" or (m.startswith(e) and m[len(e):len(e) + 1] == "r")) else "w"
def tr_fac(B, A, G): return float(np.sum((B.T @ B) * (A @ G @ A.T)))   # tr(B A G A^T B^T)
def readf(P, name): return LP.read_llava_merged(os.path.join(NEWM, name) if (SMOKE and _new(name)) else os.path.join(pool_root(P), "merges", name))[0]
def mix_rank(pp, iso, a, k):
    """vlm_pool.mix with the pool's own rank k (the module constant RANK is FULL's when no VLM_SUB is set)."""
    f = {}
    for c in sorted(pp):
        Bc = np.hstack([a * np.asarray(pp[c]["B"], dtype=np.float64), (1 - a) * np.asarray(iso[c]["B"], dtype=np.float64)]); Ac = np.vstack([np.asarray(pp[c]["A"], dtype=np.float64), np.asarray(iso[c]["A"], dtype=np.float64)])
        Qb, Rb = np.linalg.qr(Bc); Qa, Ra = np.linalg.qr(Ac.T); U, S, Vt = np.linalg.svd(Rb @ Ra.T); kk = min(k, len(S))
        f[c] = {"B": Qb @ (U[:, :kk] * np.sqrt(S[:kk])), "A": (np.sqrt(S[:kk])[:, None] * Vt[:kk]) @ Qa.T}
    return f
def ends_of(P, e, cells=None):
    sub = (lambda f: {c: f[c] for c in cells}) if cells is not None else (lambda f: f)
    return {"end": sub(readf(P, ENDM[e])), "mw": sub(readf(P, "mw")), "mr": sub(readf(P, "mr"))}
def point(e, m, F, k):
    """F = ends_of(...): the family's Gram end and the two equalised ends (full adapters or the group's cells); k the pool's rank."""
    if m in ("mw", "mr"): return F[m]
    if m == f"{e}_end": return F["end"]
    side = "w" if m[len(e)] == "w" else "r"; return mix_rank(F["end"], F["m" + side], a_of(m, e), k)
def gpu_model_():
    if not V.done(V.TEMPLATE): V.TEMPLATE = "mw"; log(f"template absent: using mw as the rank-{V.RANK} skeleton (as )")
    return V.gpu_model()
def pseudo_rows(c):
    p = os.path.join(V.PARENT, "select", f"pseudo_{c}.json"); return json.load(open(p))["rows"][:4 if SMOKE else None]
def agree(LE, model, processor, clients, rows_of):
    out = {}
    for c in clients:
        rows = rows_of(c); outs = LE.generate(model, processor, rows, {"pad": LE.TASKS[V.task_of(c)]["pad"], "max_new_tokens": V.MAX_NEW}, V.MM, V.BS)
        out[c] = {"pf1": float(np.mean([V.f1(o, r["text"]) for o, r in zip(outs, rows)])), "n": len(rows)}
    return out

def stage_ppend():   # the six compositions' RegMean++ merges were pruned on Oct 26 (results/vlm/sub/*/merges/*/pruned.json): rebuilt by vlm_pool's own stage, the rebuilt mw as template
    if V.done(V.BASE): log(f"{V.BASE} exists"); return
    if not V.done(V.TEMPLATE): V.TEMPLATE = "mw"; log(f"template absent: using mw as the rank-{V.RANK} skeleton")
    t0 = time.time(); V.stage_regmeanpp(); log(f"{V.BASE} rebuilt ({time.time() - t0:.0f}s)")
def stage_halfend():
    name = ENDM["half"]
    if V.done(name): log(f"{name} exists"); return
    t0 = time.time(); pool, projs = V.load_pool(); proj = LP.mean_projector(projs); plain = V.plain_factors(); model, processor, LE = gpu_model_(); cm, lm = V.cell_modules(model), V.layer_modules(model)
    rows = {c: V.calib_rows(c, LE) for c in V.CLIENTS}; merged = {}; half = V.N_LAYERS // 2
    for l in range(2 if SMOKE else half):   # RegMean++ (llava_pool.stage_regmeanpp) on the first half of the layers
        gs = {}
        for c in V.CLIENTS:
            state = dict(merged); state.update({cell: pool[c][cell] for cell in pool[c] if cell not in merged})
            V.set_factors(model, cm, state); LP.apply_projector(model, projs[c])
            gs[c] = V.collect_grams(model, processor, LE, cm, lm, rows[c], [l], LE.TASKS[V.task_of(c)]["pad"], stop_after=l)
        merged.update(V.regmean_cells(pool, gs, l)); log(f"  half end layer {l} ({time.time() - t0:.0f}s)")
    for cell in plain:
        if cell not in merged: merged[cell] = plain[cell]   # one-shot RegMean cells above the cut
    V.write(name, merged, f"rule=regmeanpp layer-sequential in layers 0-{half - 1} of {V.N_LAYERS}, one-shot regmean above lam={V.LAM}", proj)
    json.dump({"seconds": time.time() - t0}, open(os.path.join(OUT, f"halfend_{V.SUB or 'FULL'}.json"), "w")); log(f"half end done ({time.time() - t0:.0f}s)")

def stage_basescore():
    fp = os.path.join(AN, "base_scores.json")
    if os.path.exists(fp): log("base scores exist"); return
    t0 = time.time(); LE = V.lmm_eval(); model, processor = LE.load_model("base", V.MM)
    res = agree(LE, model, processor, list(V.ALL_CLIENTS)[:3] if SMOKE else list(V.ALL_CLIENTS), pseudo_rows)
    if SMOKE: res.update({c: res[list(res)[0]] for c in V.ALL_CLIENTS if c not in res})
    json.dump(res, open(fp, "w"), indent=1)
    log("base agreement " + " ".join(f"{c} {v['pf1']:.4f}" for c, v in res.items()) + f" ({time.time() - t0:.0f}s)")

def rho_of(P, c_):
    p = json.load(open(os.path.join(AN, "base_scores.json"))); los = {t: 1.0 - p[t]["pf1"] for t in POOLS[P]}; m = max(float(np.mean(list(los.values()))), 1e-9)
    return {t: c_ * los[t] / m for t in POOLS[P]}
def stage_energy():
    assert V.SUB is None and args.group >= 0; ends = [e for e in args.ends.split(",") if e]; g0 = args.group * V.GROUP
    layers = list(range(g0, min(g0 + V.GROUP, V.N_LAYERS)))[:1 if CHECK else None]; fp = os.path.join(OUT, f"energy_g{args.group}{'_check' if CHECK else ''}.json")
    out = json.load(open(fp)) if os.path.exists(fp) else {"layers": layers, "num": {}, "den": {}}
    need_den = [(P, e) for P in POOLS for e in ends if f"{P}|{e}" not in out["den"]]
    need_anc = [(P, c_) for P in POOLS for c_ in CGRID if not os.path.exists(os.path.join(TMP, f"{P}_N{c_}_g{args.group}.npz"))] if os.path.exists(os.path.join(AN, "base_scores.json")) else []
    if os.environ.get("OPT_NOANC") == "1": need_anc = []   # the half end's denominators only; wrote the N@c merges and deleted their temporaries
    if out["num"] and not need_den and not need_anc: log(f"energy group {args.group}: nothing to do"); return
    t0 = time.time(); pool, projs = V.load_pool(); clients = list(V.ALL_CLIENTS); model, processor, LE = gpu_model_(); cm, lm = V.cell_modules(model), V.layer_modules(model); G = {}
    for c in clients:
        rows = V.calib_rows(c, LE); V.set_factors(model, cm, pool[c]); LP.apply_projector(model, projs[c])
        g = V.collect_grams(model, processor, LE, cm, lm, rows, layers, LE.TASKS[V.task_of(c)]["pad"], stop_after=layers[-1]); G[c] = {k: V.shrink(v).astype(np.float64) for k, v in g.items()}; del g
    log(f"grams layers {layers} ({time.time() - t0:.0f}s)"); del model; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception: pass
    cells = [(l, mod) for l in layers for mod in V.MODS]
    if not out["num"]: out["num"] = {c: sum(tr_fac(np.asarray(pool[c][cell]["B"], dtype=np.float64), np.asarray(pool[c][cell]["A"], dtype=np.float64), G[c][(cell[0], V.GRAM_OF[cell[1]])]) for cell in cells) for c in clients}
    for P, e in need_den:
        Gs = {(l, kind): sum(G[c][(l, kind)] for c in POOLS[P]) for l in layers for kind in V.HOOK_MOD}; d = {}; F = ends_of(P, e, cells)
        for m in cands(e):
            f = point(e, m, F, rank_of(P)); d[m] = sum(tr_fac(np.asarray(f[cell]["B"], dtype=np.float64), np.asarray(f[cell]["A"], dtype=np.float64), Gs[(cell[0], V.GRAM_OF[cell[1]])]) for cell in cells); del f
        out["den"][f"{P}|{e}"] = d; json.dump(out, open(fp, "w"), indent=1); log(f"  denominators {P} {e} ({time.time() - t0:.0f}s)"); del Gs, F; gc.collect()
    json.dump(out, open(fp, "w"), indent=1)
    from scipy.linalg import cho_factor, cho_solve
    for P, c_ in need_anc:
        t1 = time.time(); rho = rho_of(P, c_); fac = {}
        for l in layers:
            for kind in V.HOOK_MOD:
                Gsum = sum(G[t][(l, kind)] for t in POOLS[P]); M = Gsum + sum(rho[t] * G[t][(l, kind)] for t in POOLS[P]); M = M + 1e-6 * float(np.trace(M)) / M.shape[0] * np.eye(M.shape[0]); cf = cho_factor(M)
                for mod in [m for m in V.MODS if V.GRAM_OF[m] == kind]:
                    A = np.vstack([rho[t] * cho_solve(cf, (np.asarray(pool[t][(l, mod)]["A"], dtype=np.float64) @ G[t][(l, kind)]).T).T for t in POOLS[P]])   # rho_t A_t G~_t M^-1
                    B = np.hstack([np.asarray(pool[t][(l, mod)]["B"], dtype=np.float64) for t in POOLS[P]])
                    assert np.isfinite(A).all() and np.isfinite(B).all(), (P, c_, l, mod); fac[f"{l}.{mod}.A"] = A.astype(np.float32); fac[f"{l}.{mod}.B"] = B.astype(np.float32)
        if CHECK and not SMOKE: log(f"  check N@{c_} {P}: {len(fac) // 2} cells, |A| max {max(float(np.abs(v).max()) for k, v in fac.items() if k.endswith('.A')):.3e} ({time.time() - t1:.0f}s)"); continue
        tmp = os.path.join(TMP, f"{P}_N{c_}_g{args.group}.tmp.npz"); np.savez(tmp, **{k: v.astype(np.float16) for k, v in fac.items()}); os.replace(tmp, os.path.join(TMP, f"{P}_N{c_}_g{args.group}.npz")); log(f"  N@{c_} {P} group {args.group} ({time.time() - t1:.0f}s)")
    log(f"energy group {args.group} done ({time.time() - t0:.0f}s)")

def stage_anchorwrite():
    P = V.SUB or "FULL"; ng = (V.N_LAYERS + V.GROUP - 1) // V.GROUP; _, projs = V.load_pool(); proj = LP.mean_projector(projs)
    for c_ in CGRID:
        name = f"anc_N{c_}"
        if V.done(name): log(f"{name} exists"); continue
        fs = [os.path.join(TMP, f"{P}_N{c_}_g{g}.npz") for g in range(1 if SMOKE else ng)]; assert all(os.path.exists(f) for f in fs), (P, c_, [f for f in fs if not os.path.exists(f)])
        f = {}
        for p in fs:
            z = np.load(p)
            for k in z.files:
                l, mod, ab = k.split(".", 1)[0], k.split(".", 1)[1].rsplit(".", 1)[0], k.rsplit(".", 1)[1]; f.setdefault((int(l), mod), {})[ab] = z[k].astype(np.float64)
        if SMOKE:   # code test only: the cells outside layer 0 are the zero update
            for l in range(V.N_LAYERS):
                for mod in V.MODS:
                    if (l, mod) not in f: x = f[(0, mod)]; f[(l, mod)] = {"A": np.zeros_like(x["A"]), "B": np.zeros_like(x["B"])}
        assert len(f) == V.N_LAYERS * len(V.MODS), (P, c_, len(f))
        V.write(name, f, f"rule=base-anchored point N@{c_}: W = sum_t rho_t B_t A_t G~_t (sum_k G~_k + sum_t rho_t G~_t)^-1, rho_t = c (1 - p_t) / mean (1 - p), p = base agreement", proj)
        for p in fs: os.remove(p)

def stage_family():
    e = args.end; assert e in ENDM; P = V.SUB or "FULL"; D = os.path.join(OUT, e, P); os.makedirs(D, exist_ok=True); t0 = time.time()
    fs = sorted(glob.glob(os.path.join(OUT, "energy_g[0-9]*.json" if SMOKE else "energy_g[0-9].json"))); assert SMOKE or len(fs) == (V.N_LAYERS + V.GROUP - 1) // V.GROUP, f"energy groups {len(fs)}"
    num, den = 0.0, {m: 0.0 for m in cands(e)}
    for fp in fs:
        j = json.load(open(fp)); num += sum(j["num"][c] for c in POOLS[P])
        for m in cands(e): den[m] += j["den"][f"{P}|{e}"][m]
    ss = {m: float(np.sqrt(num / max(den[m], 1e-30))) for m in cands(e)}; _, projs = V.load_pool(); proj = LP.mean_projector(projs)
    if not V.done(V.TEMPLATE): V.TEMPLATE = "mw"; log(f"[{P}] template absent: using mw as the rank-{V.RANK} skeleton")
    model, processor, LE = V.gpu_model(); cm = V.cell_modules(model); LP.apply_projector(model, proj); sc = {}; F = ends_of(P, e)
    for m in (["mw", f"{e}w50", f"{e}_end"] if SMOKE else cands(e)):
        fr = os.path.join(D, f"fr_{m}.json"); fa = os.path.join(LEN, P, f"fr_{m}.json")
        if not os.path.exists(fr) and m in ("mw", "mr") and os.path.exists(fa) and not SMOKE:   # the equalised ends are A's own points at the same s*: reuse A's scores
            sA = json.load(open(os.path.join(LEN, P, "pick.json")))["s_star"][m]
            if abs(sA - ss[m]) < 1e-3 * sA: json.dump(json.load(open(fa)), open(fr, "w")); log(f"[{P}] {m}: A's score reused (s* {ss[m]:.5f} vs A's {sA:.5f})")
            else: log(f"[{P}] {m}: s* {ss[m]:.5f} differs from A's {sA:.5f} by more than 1e-3: scored again")
        if not os.path.exists(fr):
            t1 = time.time(); f = point(e, m, F, rank_of(P)); V.set_factors(model, cm, {c: {"A": v["A"], "B": ss[m] * v["B"]} for c, v in f.items()}); del f
            res = agree(LE, model, processor, V.CLIENTS, lambda c: json.load(open(os.path.join(V.SEL, f"pseudo_{c}.json")))["rows"][:4 if SMOKE else None])
            json.dump(res, open(fr, "w")); log(f"[{P}] {e} {m}: s* {ss[m]:.3f} s_P {np.mean([res[c]['pf1'] for c in V.CLIENTS]):.4f} ({time.time() - t1:.0f}s)")
        r_ = json.load(open(fr)); sc[m] = float(np.mean([r_[c]["pf1"] for c in V.CLIENTS]))
    p = max(sc, key=lambda m: (round(sc[m], 12), a_of(m, e), 1 if side_of(m, e) == "w" else 0)); name = f"{e}pick_ss"
    if not V.done(name): f = point(e, p, F, rank_of(P)); V.write(name, {c: {"A": v["A"], "B": ss[p] * v["B"]} for c, v in f.items()}, f"rule={p} of the {e} family at the closed-form length s*={ss[p]:.4f}", proj)
    json.dump({"pick": p, "name": name, "a": a_of(p, e), "side": side_of(p, e), "s": ss[p], "s_star": ss, "s_P": sc}, open(os.path.join(D, "pick.json"), "w"), indent=1)
    log(f"[{P}] {e} pick {p} s* {ss[p]:.3f} (s_P {sc[p]:.4f}); family done ({time.time() - t0:.0f}s)")

def stage_anchorscore():
    P = V.SUB or "FULL"; D = os.path.join(AN, P); os.makedirs(D, exist_ok=True); fp = os.path.join(D, "scores.json"); res = json.load(open(fp)) if os.path.exists(fp) else {}
    todo = [c_ for c_ in CGRID if str(c_) not in res]
    if todo:
        _, projs = V.load_pool(); proj = LP.mean_projector(projs); model, processor, LE = gpu_model_(); cm = V.cell_modules(model); LP.apply_projector(model, proj)
        for c_ in (todo[:1] if SMOKE else todo):
            t1 = time.time(); V.set_factors(model, cm, V.read(f"anc_N{c_}")); r_ = agree(LE, model, processor, V.CLIENTS, lambda c: json.load(open(os.path.join(V.SEL, f"pseudo_{c}.json")))["rows"][:4 if SMOKE else None])
            res[str(c_)] = {"s_P": float(np.mean([r_[c]["pf1"] for c in V.CLIENTS])), "clients": r_}; json.dump(res, open(fp, "w"), indent=1); log(f"[{P}] N@{c_}: s_P {res[str(c_)]['s_P']:.4f} ({time.time() - t1:.0f}s)")
    b = json.load(open(os.path.join(AN, "base_scores.json"))); res["0.0"] = {"s_P": float(np.mean([b[c]["pf1"] for c in V.CLIENTS])), "clients": {c: b[c] for c in V.CLIENTS}}; json.dump(res, open(fp, "w"), indent=1)

def stage_eval():
    P = V.SUB or "FULL"; kind, x = args.what.split(":", 1)
    name = json.load(open(os.path.join(OUT, x, P, "pick.json")))["name"] if kind == "endpick" else f"anc_N{float(x)}"
    f = os.path.join(V.POOL, f"merged_{name}__metrics.json")
    if os.path.exists(f) and all(t in json.load(open(f)) and json.load(open(f))[t].get("acc") is not None for t in V.EVAL_TASKS): log(f"[{P}] metrics of {name} exist"); return
    if SMOKE: assert V.done(name), name; log(f"[{P}] smoke: would evaluate merged:{V.mdir(name)}"); return
    V.stage_eval(f"merged:{name}")

{"ppend": stage_ppend, "halfend": stage_halfend, "basescore": stage_basescore, "energy": stage_energy, "anchorwrite": stage_anchorwrite, "family": stage_family, "anchorscore": stage_anchorscore, "eval": stage_eval}[args.stage]()
