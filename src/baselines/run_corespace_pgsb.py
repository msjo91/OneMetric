#!/usr/bin/env python
"""the two newest weight-only shared-basis LoRA merges as baseline rows on ONE CLIP pool.
Variants (released settings in SET below): corespace_ties_isoc, pgsb_ties_isoc (the basis-dependent operator), corespace_tsvm_isoc (= TSV-M + Iso-C, basis-equivariant, one row for both methods),
corespace_isoc (= plain Iso-C at alpha .9, a check). Readout: for each variant the alpha chosen on the 200 LABELLED calibration images per client over {.5, .7, .9, 1.1, 1.3, 1.5} (a labelled reference, as LiNeS)
and its test retention. Env as mrun_room (MR_KIND=dc|knots, ARCH, CLIPK_SUB). Writes results/clipclose/newbl/<kind>/<arch>/<sub>/newbl.json (resumable; no merge saved)."""
from src.paths import ROOT as _OM_ROOT
import os, sys, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = _OM_ROOT
KIND = os.environ["MR_KIND"]; assert KIND in ("dc", "knots"), KIND

if KIND == "dc": from src.backbones import harness_clip as H
else: from src.backbones import harness_knots as H
if KIND == "knots" and not hasattr(H, "SUBTAG"): H.SUBTAG = "8task"
from src.baselines import merges as ops
from src.baselines import merges_ext as ops_new
ARCH, SUB, TASKS, CELLS = H.ARCH, H.SUBTAG, H.TASKS, H.CELLS; log = H.log; RET = getattr(H, "RET", TASKS)
OUT = os.environ.get("MR_OUT") or os.path.join(ROOT, "results", "clipclose", "newbl", KIND, ARCH, SUB); os.makedirs(OUT, exist_ok=True); MJ = os.path.join(OUT, "newbl.json")
# Variants and released settings (Core Space, Table 7: ViT-B/32 TIES+Iso-C alpha 2.0 / keep 30 %, Iso-C and TSV+Iso-C alpha 0.9; ViT-L/14 TIES+Iso-C alpha 2.4 / keep 20 %, Iso-C 0.9; B/16 takes B/32's;
# PGSB's selected values are unreported and take the same). Iso-C and TSV-M are basis-invariant, so the shared-basis Iso-C rows equal plain Iso-C and the TSV+Iso-C row is one row for both methods.
SET = {"ties": {"B32": (2.0, 0.3), "B16": (2.0, 0.3), "L14": (2.4, 0.2)}, "other": {"B32": (0.9, None), "B16": (0.9, None), "L14": (0.9, None)}}
VARS = [("corespace", "ties_isoc"), ("pgsb", "ties_isoc"), ("corespace", "tsvm_isoc"), ("corespace", "isoc")]
def setting(op): return SET["ties" if op == "ties_isoc" else "other"][ARCH]
def grid(op): return [1.0, 1.5, 2.0, 2.5, 3.0] if op == "ties_isoc" else [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
res = H.jload(MJ) if os.path.exists(MJ) else {"kind": KIND, "arch": ARCH, "sub": SUB, "tasks": TASKS, "ret": RET, "settings": {f"{m}_{o}": setting(o) for m, o in VARS}, "grids": {o: grid(o) for _, o in VARS}, "cal": {}, "acc": {}, "R": {}, "alpha_lab": {}, "fro": {}, "seconds": {}}
def save(): H.jdump(MJ, res)
t0 = time.time(); pool, D = H.load_deltas(); del D; fac = [{c: (pool[t][c]["B"], pool[t][c]["A"]) for c in CELLS} for t in TASKS]; facd = [{c: {"B": f[c][0], "A": f[c][1]} for c in CELLS} for f in fac]; del pool
C = H.clip(); W0 = {c: C.W0[c].float().cpu().numpy() for c in CELLS}; HD = H.heads(C); Xc = {t: H.load_x(t, "cal") for t in TASKS}
home = {t: H.jload(os.path.join(H.EVAL, f"client_{t}.json"))[t] for t in TASKS}
def build(method, op, alpha):
    t1 = time.time(); keep = setting(op)[1] or 0.3; m = ops_new.corespace(facd, op, alpha, keep) if method == "corespace" else ops_new.pgsb(facd, W0, op, alpha, keep); res["seconds"][f"build_{method}_{op}"] = time.time() - t1; return m
def cal_acc(delta):
    C.set_delta(delta); return float(np.mean([(H.predict(C, Xc[t][0], HD[t]) == np.asarray(Xc[t][1])).mean() for t in TASKS]))
def test_R(delta):
    C.set_delta(delta); acc = {}
    for t in TASKS: X, y = H.load_x(t, "test"); acc[t] = float((H.predict(C, X, HD[t]) == np.asarray(y)).mean())
    return acc, float(np.mean([acc[t] / max(home[t], 1e-9) for t in RET]))
for method, op in VARS:
    name = f"{method}_{op}"; built = {}; A0, _ = setting(op); ALPHAS = sorted(set(grid(op) + [A0]))
    for a in ALPHAS:
        k = f"{name}@{a}"
        if k in res["cal"]: continue
        m = build(method, op, a); built[a] = m; res["cal"][k] = cal_acc(m); res["fro"][k] = ops.fro(m); save(); log(f"  {k}: labelled cal acc {res['cal'][k]:.4f} fro {res['fro'][k]:.1f}")
    a_lab = max(ALPHAS, key=lambda a: (round(res["cal"][f"{name}@{a}"], 12), -abs(a - A0))); res["alpha_lab"][name] = a_lab
    for tag, a in ((name, A0), (name + "_lab", a_lab)):
        if tag in res["R"]: continue
        if tag.endswith("_lab") and a == A0 and name in res["R"]: res["R"][tag] = res["R"][name]; res["acc"][tag] = res["acc"][name]; save(); continue
        m = built.get(a) or build(method, op, a); t1 = time.time(); acc, R = test_R(m); res["acc"][tag] = acc; res["R"][tag] = R; save(); log(f"  evaluated {tag} (alpha {a}) ({time.time() - t1:.0f}s)")
    built.clear()
res["done"] = True; res["seconds"]["total"] = time.time() - t0; save(); log(f"newbl {KIND} {ARCH} {SUB} done ({(time.time() - t0) / 60:.0f} min)")
