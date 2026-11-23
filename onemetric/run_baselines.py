#!/usr/bin/env python
"""Every baseline row on one vision pool, on the stored test draw: the twelve one-shot merges at their fixed settings, one-shot RegMean (truncated to the merged rank and dense), RegMean++, LiNeS on DC-Merge, PACT-Iso-C and CT-Merging (beta chosen by the labeled calibration inputs and by the score), and CT-Merging rescaled over its grid (chosen the same two ways).
Env ARCH and CLIPK_SUB for CLIP; RF_KIND=siglip with SIG_ARCH, or RF_KIND=dino with DINO_ARCH, for the other families. Writes results/clipnew/refs/<backbone>/<pool>/refs.json and resumes from it; RF_OUT overrides the folder."""
from onemetric.paths import ROOT as _OM_ROOT
import os, sys, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = _OM_ROOT; os.environ.setdefault("POOL", "dc16")
KIND = os.environ.get("RF_KIND", "dc")   # RF_KIND=siglip runs the same rows on SigLIP 2 (SIG_ARCH)
if KIND == "siglip": from onemetric import harness_siglip as H
elif KIND == "dino": from onemetric import harness_dino as H
else: from onemetric import harness_clip as H
from onemetric import merges_ext as ops_new; from onemetric import roomlib; from onemetric import merges as ops
L = roomlib.Lib(H, ROOT, KIND, ops, ops_new); ARCH, SUB, CELLS, TASKS = L.ARCH, L.SUB, H.CELLS, H.TASKS; NL = H.NLAYERS; N = len(TASKS)
PUB12 = ["ta", "fedavg_w", "fedit", "ties", "knots_ties", "tsvm", "dcmerge", "isoc", "isocts", "ct", "pact_isoc", "ace"]; LINES = ["dcmerge", "pact_isoc", "ct"]; BETAS = [round(.1 * i, 1) for i in range(1, 11)]; GAMMAS = [.3, .4, .5, .6, .7, .8, .9, 1.0, 1.1, 1.3]
OUT = os.environ.get("RF_OUT") or os.path.join(ROOT, "results", "clipnew", "refs", ARCH, SUB); os.makedirs(OUT, exist_ok=True); RJ = os.path.join(OUT, "refs.json")
res = H.jload(RJ) if os.path.exists(RJ) else {"arch": ARCH, "sub": SUB, "tasks": TASKS, "ret": L.RET, "R": {}, "acc": {}, "s_P": {}, "s_L": {}, "picks": {}, "seconds": {}}
def save(): H.jdump(RJ, res)
t0 = time.time(); pool, D = L.pool_deltas(); deltas = [D[t] for t in TASKS]; C = H.clip(); HD = H.heads(C); W0 = {c: C.W0[c].float().cpu().numpy() for c in CELLS}
pseudo = H.jload(os.path.join(H.SEL, "pseudo.json")); Xc = {t: H.load_x(t, "cal") for t in TASKS}; Xt = {t: H.load_x(t, "test") for t in TASKS}
def fro(d): return float(np.sqrt(sum(float(np.sum(np.square(d[c], dtype=np.float64))) for c in CELLS)))
TSUM = {c: sum(D[t][c] for t in TASKS).astype(np.float32) for c in CELLS}; NSUM = fro(TSUM)
def cal(name, delta):
    if name in res["s_P"]: return
    C.set_delta(delta); sp, sl = [], []
    for t in TASKS: pc = H.predict(C, Xc[t][0], HD[t]); sp.append(float((pc == np.asarray(pseudo[t]["pseudo"])).mean())); sl.append(float((pc == np.asarray(Xc[t][1])).mean()))
    res["s_P"][name], res["s_L"][name] = float(np.mean(sp)), float(np.mean(sl)); save()
def test(name, delta):
    if name in res["R"]: return
    t1 = time.time(); C.set_delta(delta); acc = {t: float((H.predict(C, Xt[t][0], HD[t]) == np.asarray(Xt[t][1])).mean()) for t in TASKS}; res["acc"][name] = acc; res["R"][name] = L.R_of(acc); save(); H.log(f"  {name}: test done ({time.time() - t1:.0f}s)")
def build(m):
    if m == "ace": A = ops_new.ACE_VIT[ARCH]; ac = ops_new.ace_merging(deltas, **A["cfg"]); return {c: (A["scale"] * ac[c]).astype(np.float32) for c in CELLS}
    return L.rebuild_pub(m, W0)
def scaled_block(tau, alpha, beta): return {c: ((alpha + beta * c[0] / (NL - 1)) * tau[c]).astype(np.float32) for c in CELLS}
H.log(f"published rows {ARCH} {SUB}: {N} clients ({time.time() - t0:.0f}s setup)")
test("base", {c: np.zeros_like(W0[c], dtype=np.float32) for c in CELLS})
for m in PUB12:
    need = m not in res["R"] or (m in LINES and f"LiNeS({m})" not in res["picks"]) or (m == "ct" and "ct_P" not in res["picks"])
    if not need: continue
    t1 = time.time(); d = build(m); res["seconds"]["build_" + m] = time.time() - t1; H.log(f"  {m} built ({time.time() - t1:.0f}s, fro {fro(d):.2f})"); cal(m, d); test(m, d)
    if m in LINES:
        alpha = NSUM / (N * max(fro(d), 1e-30))
        for b in BETAS: cal(f"LiNeS({m})@{b}", scaled_block(d, alpha, b))
        bL = max(BETAS, key=lambda b: (round(res["s_L"][f"LiNeS({m})@{b}"], 12), -b)); bP = max(BETAS, key=lambda b: (round(res["s_P"][f"LiNeS({m})@{b}"], 12), b))
        res["picks"][f"LiNeS({m})"] = {"labelled": bL, "P": bP, "alpha": alpha}; save(); test(f"LiNeS({m})", scaled_block(d, alpha, bL)); test(f"LiNeS({m})_P", scaled_block(d, alpha, bP))
    if m == "ct":
        for g in GAMMAS: cal(f"ct@{g}", {c: (g * d[c]).astype(np.float32) for c in CELLS})
        gP = max(GAMMAS, key=lambda g: (round(res["s_P"][f"ct@{g}"], 12), g)); gL = max(GAMMAS, key=lambda g: (round(res["s_L"][f"ct@{g}"], 12), g)); res["picks"]["ct_P"] = gP; res["picks"]["ct_L"] = gL; save()
        test("ct_rescaled_P", {c: (gP * d[c]).astype(np.float32) for c in CELLS}); test("ct_rescaled_L", {c: (gL * d[c]).astype(np.float32) for c in CELLS})
    del d; res["seconds"][m] = time.time() - t1; save()
del TSUM
if "regmean128" not in res["R"] or "regmean_dense" not in res["R"]:
    t1 = time.time(); grams = {t: H.collect_grams(C, D[t], Xc[t][0], range(NL)) for t in TASKS}; dense = {c: H.regmean_cell(c, D, grams) for c in CELLS}; del grams; cal("regmean_dense", dense); test("regmean_dense", dense); rm, _ = H.truncate(dense, H.RANK); del dense; cal("regmean128", rm); test("regmean128", rm); del rm; res["seconds"]["regmean"] = time.time() - t1; save()
if "regmeanpp" not in res["R"]:
    t1 = time.time(); merged = {}
    for l in range(NL):
        gs = {}
        for t in TASKS: state = {c: (merged[c] if c in merged else D[t][c]) for c in CELLS}; gs[t] = H.collect_grams(C, state, Xc[t][0], [l])
        for p in (*getattr(H, "_QKV", ("q_proj", "k_proj", "v_proj")), getattr(H, "_OUT", "out_proj")): merged[(l, p)] = H.regmean_cell((l, p), D, gs)
    cal("regmeanpp", merged); test("regmeanpp", merged); res["seconds"]["regmeanpp"] = time.time() - t1; save(); H.log(f"  regmeanpp done ({time.time() - t1:.0f}s)")
res["done"] = True; res["seconds"]["total"] = time.time() - t0; save(); H.log(f"published rows done ({(time.time() - t0) / 60:.0f} min)")
