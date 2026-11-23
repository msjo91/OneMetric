#!/usr/bin/env python
"""The closed-form length on a LLaVA pool: response energies from the factors and the Grams, the points at their length, their scores and evaluation. Stages: mixes | energy | family | eval | analysis."""
from onemetric.paths import ROOT as _OM_ROOT
import os, sys, json, time, gc, glob, argparse, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True, choices=["mixes", "energy", "family", "eval", "analysis"]); ap.add_argument("--group", type=int, default=-1); args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = _OM_ROOT
from onemetric import llava_pool as V# noqa: E402
LP, log = V.LP, V.log; MIXA = V.MIXA; OUT = os.path.join(ROOT, "results", "mfam", "vlm_len"); os.makedirs(OUT, exist_ok=True)
POOLS = {"FULL": list(V.ALL_CLIENTS), **{k: list(v) for k, v in V.SUBS.items() if k.startswith("V")}}; BAND = .01
def pool_root(P): return V.PARENT if P == "FULL" else os.path.join(V.PARENT, "sub", P)
def names(s): return [f"m{s}"] + [f"m{s}{t}" for _, t in MIXA]
CANDS = names("w") + names("r") + ["regmean_long"]
def a_of(m): return 1.0 if m == "regmean_long" else (0.0 if m in ("mw", "mr") else dict((t, a) for a, t in MIXA)[m[2:]])
def side_of(m): return "r" if m.startswith("mr") else "w"
def tr_fac(B, A, G): return float(np.sum((B.T @ B) * (A @ G @ A.T)))   # tr(B A G A^T B^T)
def stage_mixes():
    P = V.SUB or "FULL"; _, projs = V.load_pool(); proj = LP.mean_projector(projs); plain = V.plain_factors()
    for side in ("w", "r"):
        assert V.done(f"m{side}"), f"{P}: m{side} missing"; ef = V.read(f"m{side}")
        for a, tag in MIXA:
            n = f"m{side}{tag}"
            if not V.done(n): f, kept = V.mix(plain, ef, a); V.write(n, f, f"rule=mix a={a} * regmean_long + (1-a) * m{side}, SVD -> rank {V.RANK}, energy kept mean={np.mean(kept):.4f}", proj)
    if not V.done("regmean_long"): V.write("regmean_long", plain, "rule=regmean plain one-shot, the Gram end, adapter rebuilt from regmean_long_factors.npz after the Oct 26 disk prune", proj); log(f"[{P}] regmean_long rebuilt from its stored factors")
    assert all(V.done(m) for m in CANDS), f"{P}: candidates missing"; log(f"[{P}] thirteen points present")
def stage_energy():
    assert V.SUB is None and args.group >= 0; g0 = args.group * V.GROUP; layers = list(range(g0, min(g0 + V.GROUP, V.N_LAYERS))); fp = os.path.join(OUT, f"energy_g{args.group}.json")
    if os.path.exists(fp): log(f"energy group {args.group} exists"); return
    t0 = time.time(); pool, projs = V.load_pool(); clients = list(V.ALL_CLIENTS); model, processor, LE = V.gpu_model(); cm, lm = V.cell_modules(model), V.layer_modules(model); rows = {c: V.calib_rows(c, LE) for c in clients}; G = {}
    for c in clients:
        V.set_factors(model, cm, pool[c]); LP.apply_projector(model, projs[c]); g = V.collect_grams(model, processor, LE, cm, lm, rows[c], layers, LE.TASKS[V.task_of(c)]["pad"], stop_after=layers[-1])
        G[c] = {k: V.shrink(v).astype(np.float64) for k, v in g.items()}; del g
    log(f"grams layers {layers} ({time.time() - t0:.0f}s)"); del model; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception: pass
    cells = [(l, mod) for l in layers for mod in V.MODS]; out = {"layers": layers, "num": {}, "den": {}}
    for c in clients: out["num"][c] = sum(tr_fac(np.asarray(pool[c][cell]["B"], dtype=np.float64), np.asarray(pool[c][cell]["A"], dtype=np.float64), G[c][(cell[0], V.GRAM_OF[cell[1]])]) for cell in cells)
    for P, pcl in POOLS.items():
        Gs = {(l, kind): sum(G[c][(l, kind)] for c in pcl) for l in layers for kind in V.HOOK_MOD}; out["den"][P] = {}
        for m in CANDS:
            f = LP.read_llava_merged(os.path.join(pool_root(P), "merges", m))[0]
            out["den"][P][m] = sum(tr_fac(np.asarray(f[cell]["B"], dtype=np.float64), np.asarray(f[cell]["A"], dtype=np.float64), Gs[(cell[0], V.GRAM_OF[cell[1]])]) for cell in cells); del f
        log(f"  energies of layers {layers} for {P} ({time.time() - t0:.0f}s)"); del Gs; gc.collect()
    json.dump(out, open(fp, "w"), indent=1); log(f"energy group {args.group} done ({time.time() - t0:.0f}s)")
def s_stars(P):
    fs = sorted(glob.glob(os.path.join(OUT, "energy_g*.json"))); assert len(fs) == (V.N_LAYERS + V.GROUP - 1) // V.GROUP, f"energy groups {len(fs)}"; num, den = 0.0, {m: 0.0 for m in CANDS}
    for fp in fs:
        j = json.load(open(fp)); num += sum(j["num"][c] for c in POOLS[P])
        for m in CANDS: den[m] += j["den"][P][m]
    return {m: float(np.sqrt(num / max(den[m], 1e-30))) for m in CANDS}
def stage_family():
    P = V.SUB or "FULL"; D = os.path.join(OUT, P); os.makedirs(D, exist_ok=True); t0 = time.time(); ss = s_stars(P); _, projs = V.load_pool(); proj = LP.mean_projector(projs)
    if not V.done(V.TEMPLATE): V.TEMPLATE = "mw"; log(f"[{P}] template absent: using mw as the rank-{V.RANK} skeleton")
    model, processor, LE = V.gpu_model(); cm = V.cell_modules(model); LP.apply_projector(model, proj); pseudo = {c: json.load(open(os.path.join(V.SEL, f"pseudo_{c}.json")))["rows"] for c in V.CLIENTS}; sc = {}
    for m in CANDS:
        fr = os.path.join(D, f"fr_{m}.json")
        if not os.path.exists(fr):
            t1 = time.time(); f = V.read(m); V.set_factors(model, cm, {c: {"A": v["A"], "B": ss[m] * v["B"]} for c, v in f.items()}); res = {}; del f
            for c in V.CLIENTS:
                rows = pseudo[c]; outs = LE.generate(model, processor, rows, {"pad": LE.TASKS[V.task_of(c)]["pad"], "max_new_tokens": V.MAX_NEW}, V.MM, V.BS); res[c] = {"pf1": float(np.mean([V.f1(o, r["text"]) for o, r in zip(outs, rows)])), "n": len(rows)}
            json.dump(res, open(fr, "w")); log(f"[{P}] {m}: s* {ss[m]:.3f} s_P {np.mean([res[c]['pf1'] for c in V.CLIENTS]):.4f} ({time.time() - t1:.0f}s)")
        r_ = json.load(open(fr)); sc[m] = float(np.mean([r_[c]["pf1"] for c in V.CLIENTS]))
    p = max(CANDS, key=lambda m: (round(sc[m], 12), a_of(m), 1 if side_of(m) == "w" else 0)); name = f"{p}_ss"
    if not V.done(name): f = V.read(p); V.write(name, {c: {"A": v["A"], "B": ss[p] * v["B"]} for c, v in f.items()}, f"rule={p} at the closed-form length s*={ss[p]:.4f}", proj)
    json.dump({"pick": p, "name": name, "a": a_of(p), "side": side_of(p), "s": ss[p], "s_star": ss, "s_P": sc}, open(os.path.join(D, "pick.json"), "w"), indent=1); log(f"[{P}] pick {p} s* {ss[p]:.3f} (s_P {sc[p]:.4f}); family stage done ({time.time() - t0:.0f}s)")
def stage_eval():
    P = V.SUB or "FULL"; name = json.load(open(os.path.join(OUT, P, "pick.json")))["name"]; f = os.path.join(V.POOL, f"merged_{name}__metrics.json")
    if os.path.exists(f) and all(t in json.load(open(f)) and json.load(open(f))[t].get("acc") is not None for t in V.EVAL_TASKS): log(f"[{P}] metrics of {name} exist"); return
    V.stage_eval(f"merged:{name}")
def stage_analysis():
    F = json.load(open(os.path.join(ROOT, "results", "paper3", "family_vlm_length1_2026-11-05.json")))["pools"]; lines = [f"analysis, {time.strftime('%Y-%m-%d %H:%M')}"]; out = {"pools": {}, "missing": []}; UNS = ["AOKVQA", "ImageNet-R", "Screen2words", "TabMWP"]
    for P in POOLS:
        pk = os.path.join(OUT, P, "pick.json"); d = os.path.join(pool_root(P), "pool500")
        if not os.path.exists(pk): out["missing"].append(P); continue
        j = json.load(open(pk)); mf = os.path.join(d, f"merged_{j['name']}__metrics.json")
        if not os.path.exists(mf): out["missing"].append(P); continue
        v = json.load(open(mf)); r = F[P]; seen = r["seen"]; home = {}
        for c in r["clients"]: t = {"flickr30k": "Flickr30k"}.get(c, c); home[t] = json.load(open(os.path.join(d, f"client_{c}__metrics.json")))[t]["acc"]
        Rt = {t: v[t]["acc"] / home[t] for t in seen}; R = float(np.mean(list(Rt.values())))
        out["pools"][P] = {"R": R, "Rt": Rt, "min": float(min(Rt.values())), "pick": j["pick"], "a": j["a"], "side": j["side"], "s": j["s"], "s_r75": j["s_star"]["mr75"], "stored": r["R_family"], "stored_pick": r["pick"], "pp": r["R_regmeanpp"], "best_published": r["best_published"], "R_best_published": r["R_best_published"], "unseen": {t: v[t]["acc"] for t in UNS if t in v}}
        q = out["pools"][P]; lines.append(f"{P}: stored {q['stored']:.4f} ({q['stored_pick']}) | at s* {q['R']:.4f} ({q['pick']} a={q['a']} {q['side']} s*={q['s']:.3f}) d {q['R'] - q['stored']:+.4f} | min {q['min']:.3f} | RegMean++ {q['pp']:.4f} | best published {q['best_published']} {q['R_best_published']:.4f} | s* at mr75 {q['s_r75']:.3f}")
    wtl = lambda ds: [int(sum(x > BAND for x in ds)), int(sum(abs(x) <= BAND for x in ds)), int(sum(x < -BAND for x in ds))]
    if len(out["pools"]) == 7:
        Q = list(out["pools"].values()); d = [q["R"] - q["stored"] for q in Q]; w = wtl(d); holds = float(np.mean(d)) >= -.005 and w[2] <= 1 and sum(1 for q in Q if q["R"] - q["pp"] < -BAND) == 0; better = float(np.mean(d)) > .005 and w[0] > w[2]
        out["calls"] = {"d_mean": float(np.mean(d)), "wtl_vs_stored": w, "wtl_vs_pp": wtl([q["R"] - q["pp"] for q in Q]), "wtl_vs_best_published": wtl([q["R"] - q["R_best_published"] for q in Q]), "verdict": "LLAVA-BETTER" if better else ("LLAVA-HOLDS" if holds else "LLAVA-FAILS")}
    else: out["calls"] = {"verdict": "INCOMPLETE"}
    lines.append("CALLS: " + json.dumps(out["calls"]))
    if out["missing"]: lines.append("MISSING: " + " ".join(out["missing"]))
    open(os.path.join(OUT, "analysis_vlm_len.txt"), "w").write("\n".join(lines) + "\n"); json.dump(out, open(os.path.join(OUT, "analysis_vlm_len.json"), "w"), indent=1); print("\n".join(lines))
{"mixes": stage_mixes, "energy": stage_energy, "family": stage_family, "eval": stage_eval, "analysis": stage_analysis}[args.stage]()
