#!/usr/bin/env python
"""The spectral ends of the method on a LLaVA pool, kept in factor form, the segment points between the ends and the Gram end, their scores, and their evaluation. Stages: ends | family | eval | analysis."""
from src.paths import ROOT as _OM_ROOT
import os, sys, json, time, gc, argparse, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True, choices=["ends", "family", "eval", "analysis"]); args = ap.parse_args()   # family = mixes + rule-P scoring + pick (in-process model); eval and analysis run as separate processes so the evaluator can load its own model
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = _OM_ROOT
from src.llava import llava_pool as V; from src.method import spectral_end as mend# noqa: E402
LP, log = V.LP, V.log; MIXA = V.MIXA; LEVEL, RIDGE, KR = "block", 1e-3, (8, 15)
POOLS = {"FULL": list(V.ALL_CLIENTS), **{k: list(v) for k, v in V.SUBS.items() if k.startswith("V")}}
if os.environ.get("MFAM_POOLS"): POOLS = {k: POOLS[k] for k in os.environ["MFAM_POOLS"].split(",")}   # restrict the ends pass to the new compositions
TAGP = ("_" + "-".join(POOLS)) if os.environ.get("MFAM_POOLS") else ""
def pool_root(P): return V.PARENT if P == "FULL" else os.path.join(V.PARENT, "sub", P)
def pool_target(P):
    p = os.path.join(pool_root(P), "predictor.json"); return json.load(open(p))["target"] if os.path.exists(p) else "median"
def truncate(f, k):
    out = {}
    for c, fs in f.items():
        Qb, Rb = np.linalg.qr(fs["B"]); Qa, Ra = np.linalg.qr(fs["A"].T); U, S, Vt = np.linalg.svd(Rb @ Ra.T); kk = min(k, len(S))
        out[c] = {"B": Qb @ (U[:, :kk] * np.sqrt(S[:kk])), "A": (np.sqrt(S[:kk])[:, None] * Vt[:kk]) @ Qa.T}
    return out
def stage_ends():
    assert V.SUB is None, "run the ends stage without VLM_SUB"
    OUT = os.path.join(ROOT, "results", "mfam", "vlm"); os.makedirs(OUT, exist_ok=True); t00 = time.time()
    pool, projs = V.load_pool(); clients = list(V.ALL_CLIENTS); cells_all = sorted(pool[clients[0]])
    norms = {c: {cell: float(np.linalg.norm(pool[c][cell]["B"] @ pool[c][cell]["A"])) for cell in cells_all} for c in clients}
    targets = {P: pool_target(P) for P in POOLS}; info = {"pools": POOLS, "targets": targets, "level": LEVEL, "ridge": RIDGE, "kr": KR, "seconds": {}}
    model, processor, LE = V.gpu_model(); cm, lm = V.cell_modules(model), V.layer_modules(model); rows = {c: V.calib_rows(c, LE) for c in clients}
    ends = {P: {"w": {}, "r": {}} for P in POOLS}
    for g0 in range(0, V.N_LAYERS, V.GROUP):
        layers = list(range(g0, min(g0 + V.GROUP, V.N_LAYERS))); t0 = time.time(); G = {}
        for c in clients:
            V.set_factors(model, cm, pool[c]); LP.apply_projector(model, projs[c])
            g = V.collect_grams(model, processor, LE, cm, lm, rows[c], layers, LE.TASKS[V.task_of(c)]["pad"], stop_after=layers[-1])
            G[c] = {k: V.shrink(v).astype(np.float32) for k, v in g.items()}; del g
        log(f"grams layers {layers}: {len(clients)} clients ({time.time() - t0:.0f}s)")
        t0 = time.time()
        for P, pcl in POOLS.items():
            for l in layers:
                met = {}
                for kind in V.HOOK_MOD:
                    Gl = [G[c][(l, kind)].astype(np.float64) for c in pcl]; Gs = sum(Gl); met[kind] = (Gl, Gs, mend.gram_metric(Gs, RIDGE))
                for mod in V.MODS:
                    cell = (l, mod); kind = V.GRAM_OF[mod]; Gl, Gs, mt = met[kind]; v = np.array([norms[c][cell] for c in pcl]); tg = float(np.median(v)) if targets[P] == "median" else float(np.min(v))
                    for side, wh in (("w", True), ("r", False)):
                        F = [mend.compact_svd(pool[c][cell]["B"], pool[c][cell]["A"], (tg / max(norms[c][cell], 1e-12)) if wh else 1.0) for c in pcl]
                        B, A = mend.m_cell(F, Gl, Gs, mt, KR[0], KR[1], LEVEL, factors=True); ends[P][side][cell] = {"B": np.asarray(B, dtype=np.float64), "A": np.asarray(A, dtype=np.float64)}
        del G; gc.collect(); log(f"  endpoints of layers {layers} for {list(POOLS)} ({time.time() - t0:.0f}s)")
    del model; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception: pass
    for P, pcl in POOLS.items():
        rank = V.R_CLIENT * len(pcl); proj = LP.mean_projector({c: projs[c] for c in pcl}); mdir = os.path.join(pool_root(P), "merges"); os.makedirs(mdir, exist_ok=True); fro = {}
        for side in ("w", "r"):
            f = truncate(ends[P][side], rank); fro[side] = float(np.sqrt(sum(np.linalg.norm(v["B"] @ v["A"]) ** 2 for v in f.values())))
            fp = {c: (v if v["A"].shape[0] == rank else V.embed(v, rank)) for c, v in f.items()}
            LP.write_llava_merged(fp, os.path.join(mdir, f"m{side}"), rank, projector=proj, description=f"rule=m{side} (M endpoint, {'whitened ' + targets[P] if side == 'w' else 'raw'} pool, block levels, ridge {RIDGE}, k/r {KR}; family) pool={P} clients={pcl} rank={rank}")
        info["seconds"][P] = time.time() - t00; info.setdefault("fro", {})[P] = fro; log(f"{P}: mw/mr written (rank {rank}; fro w {fro['w']:.2f} r {fro['r']:.2f})")
    json.dump(info, open(os.path.join(OUT, f"ends{TAGP}.json"), "w"), indent=1); log(f"ends stage done ({time.time() - t00:.0f}s)")
def stage_family():
    P = V.SUB or "FULL"; OUT = os.path.join(ROOT, "results", "mfam", "vlm", P); os.makedirs(OUT, exist_ok=True); t0 = time.time()
    if V.SUB: V.stage_link()
    for side in ("w", "r"): assert V.done(f"m{side}"), f"{P}: m{side} missing (run --stage ends first)"
    _, projs = V.load_pool(); proj = LP.mean_projector(projs); plain = V.plain_factors(); energy = {}
    for side in ("w", "r"):
        ef = V.read(f"m{side}")
        for a, tag in MIXA:
            n = f"m{side}{tag}"
            if V.done(n): continue
            f, kept = V.mix(plain, ef, a); energy[n] = float(np.mean(kept)); V.write(n, f, f"rule=mix a={a} * regmean_long + (1-a) * m{side}, SVD -> rank {V.RANK}, energy kept mean={np.mean(kept):.4f}", proj)
    names = lambda s: [f"m{s}"] + [f"m{s}{t}" for _, t in MIXA]; CANDS = names("w") + names("r") + ["regmean_long"]
    def a_of(m): return 1.0 if m == "regmean_long" else (0.0 if m in ("mw", "mr") else dict((t, a) for a, t in MIXA)[m[2:]])
    def side_of(m): return "w" if m.startswith("mw") else ("r" if m.startswith("mr") else "w")
    if not V.done(V.TEMPLATE): V.TEMPLATE = "mw"; log(f"[{P}] template concat has no weights here (pruned composition tree): using mw as the rank-{V.RANK} skeleton")   # the template only supplies the module layout; its weights are overwritten per candidate
    V.grid = lambda: CANDS; V.stage_score("grid"); s = V.scores(CANDS); assert all(m in s for m in CANDS), f"{P}: scores missing {[m for m in CANDS if m not in s]}"
    p = max(CANDS, key=lambda m: (round(s[m], 12), a_of(m), 1 if side_of(m) == "w" else 0)); json.dump({"pick": p, "a": a_of(p), "side": side_of(p), "s_P": s, "mix_energy": energy}, open(os.path.join(OUT, "pick.json"), "w"), indent=1)
    log(f"[{P}] pick {p} (s_P {s[p]:.4f}); family stage done ({time.time() - t0:.0f}s)")
def names_all():
    names = lambda s: [f"m{s}"] + [f"m{s}{t}" for _, t in MIXA]; return names("w") + names("r") + ["regmean_long"]
def a_of(m): return 1.0 if m == "regmean_long" else (0.0 if m in ("mw", "mr") else dict((t, a) for a, t in MIXA)[m[2:]])
def side_of(m): return "w" if m.startswith("mw") else ("r" if m.startswith("mr") else "w")
def stage_eval():
    P = V.SUB or "FULL"; OUT = os.path.join(ROOT, "results", "mfam", "vlm", P); p = json.load(open(os.path.join(OUT, "pick.json")))["pick"]
    f = os.path.join(V.POOL, f"merged_{p}__metrics.json")
    if os.path.exists(f) and all(t in json.load(open(f)) and json.load(open(f))[t].get("acc") is not None for t in V.EVAL_TASKS): log(f"[{P}] metrics of {p} exist"); return
    V.stage_eval(f"merged:{p}")
def stage_analysis():
    P = V.SUB or "FULL"; OUT = os.path.join(ROOT, "results", "mfam", "vlm", P); pk = json.load(open(os.path.join(OUT, "pick.json"))); p, s = pk["pick"], pk["s_P"]; t0 = time.time()
    def acc_of(name):
        f = os.path.join(V.POOL, f"{name}__metrics.json"); return json.load(open(f)) if os.path.exists(f) else None
    home = {V.task_of(c): (acc_of(f"client_{c}") or {}).get(V.task_of(c), {}).get("acc") for c in V.CLIENTS}
    def R(name):
        v = acc_of(f"merged_{name}")
        if v is None or any(home[t] is None or v.get(t, {}).get("acc") is None for t in V.SEEN): return None, None
        rt = {t: v[t]["acc"] / home[t] for t in V.SEEN}; return float(np.mean(list(rt.values()))), rt
    oldp = json.load(open(os.path.join(V.ROOT, "pick.json")))["pick"] if os.path.exists(os.path.join(V.ROOT, "pick.json")) else None
    Rp, Rtp = R(p); Ro, Rto = R(oldp) if oldp else (None, None); Rb, _ = R(V.BASE)
    res = {"pool": P, "clients": V.CLIENTS, "seen": V.SEEN, "pick": p, "a": a_of(p), "side": side_of(p), "s_P": s, "R_pick": Rp, "Rt_pick": Rtp, "old_pick": oldp, "R_old": Ro, "Rt_old": Rto, "R_base_regmeanpp": Rb, "seconds": time.time() - t0}
    json.dump(res, open(os.path.join(OUT, "analysis.json"), "w"), indent=1)
    line = f"LLaVA pool {P}: clients {V.CLIENTS}; pick {p} (a {a_of(p)}, side {side_of(p)}, s_P {s[p]:.4f}) R {Rp} | old pick {oldp} R {Ro} | RegMean++ {Rb}"
    if Rtp: line += "\n  per task (new | old): " + " ".join(f"{t} {Rtp[t]:.3f}|{(Rto or {}).get(t, float('nan')):.3f}" for t in Rtp)
    open(os.path.join(OUT, "analysis.txt"), "w").write(line + "\n"); print(line)
{"ends": stage_ends, "family": stage_family, "eval": stage_eval, "analysis": stage_analysis}[args.stage]()
