#!/usr/bin/env python
"""OneMetric on one vision pool, for one Gram end.

The script builds the two spectral ends and one Gram end, the segment points between them at their closed-form length, scores every point on the clients' unlabeled
calibration inputs, and evaluates the highest-scoring point on the test draw. scripts/run_pool.sh makes the five calls a pool needs, and src.method.summarize applies the
server's choice across them.

Environment
  MR_KIND = dc | knots | siglip | dino, with ARCH / SIG_ARCH / DINO_ARCH and CLIPK_SUB (the pool).
  Gram end: nothing (one-shot) | MR_SEQ=1 (sequential) | MR_SEQ=1 MR_SEQCUT=0.5 (half-sequential) | MR_COMEND=1 (Chain of Merges; run run_chain_of_merges first).
  MR_ADDON=1 adds the anchored points (MR_CGRID, default 0,0.03,0.1,0.3) after the one-shot run's pick.
  Variants of the supplement, each in its own results tree:
    MR_ABL=euclid | ctlevels     the spectral end with a Euclidean right stack, or with CT-Merging's weight-energy levels
    MR_CALSEED=1 | 2 | n<N>      another calibration draw, or a calibration set of N inputs; MR_LOO=1 adds the choice without each client's votes
    MR_PUB=1                     the server scores on public images with pseudo-answers it makes itself; MR_PUBTAG and MR_PUBC select another image set:
        pubsg*                   no Gram upload: the server builds every Gram on the public images
        pubg<m>                  Gaussian noise of relative size m/1000 on every uploaded Gram
        pubgn<N>[r<m>][d<i>]     Grams over N of the client's images, optionally with noise (draw i)
        pubst<name>              Grams read from a stored release (audit_gram_dp.py)
    MR_B3NVAR=1 / MR_B4NVAR=1 / MR_B4NABL=1   the same variant on the sequential ends and anchored points / on the Chain of Merges end
Output: results/clipclose/closedlen/[<tree>/]<kind>/<backbone>/<pool>/mrun.json, resumed if present. MR_OUT overrides the folder. No merge is saved.
MR_PREDOUT=<file.npz> also stores the returned point's test predictions and each client's own."""
import os, re, sys, time
import numpy as np
from src.paths import ROOT

KIND = os.environ["MR_KIND"]; assert KIND in ("dc", "knots", "siglip", "dino"), KIND
if KIND == "dc": from src.backbones import harness_clip as H
elif KIND == "knots": from src.backbones import harness_knots as H
elif KIND == "siglip": from src.backbones import harness_siglip as H
else: from src.backbones import harness_dino as H
if KIND == "knots" and not hasattr(H, "SUBTAG"): H.SUBTAG = "8task"
from src.baselines import merges as ops
from src.method import spectral_end

ARCH, SUB, TASKS, CELLS, RANK = H.ARCH, H.SUBTAG, H.TASKS, H.CELLS, H.RANK
log = H.log; K = len(TASKS); RET = getattr(H, "RET", TASKS)   # the tasks that count in retention (the KnOTS harness counts every task)

# ---- the method's constants ----
A_GRID = [0.0, 0.25, 0.5, 0.625, 0.75, 0.875, 1.0]            # weight of the Gram end on a segment; a = 0 is the spectral end alone
KR = (8, 15)                                                   # consensus directions k_c and residual directions per client r_c
RIDGE = 1e-3                                                   # ridge of the input metric in the orthonormalization of the right stack
C_GRID = [float(x) for x in os.environ.get("MR_CGRID", "0,0.03,0.1,0.3").split(",")]   # anchored points; c = 0 is the base model

# ---- switches ----
flag = lambda name: os.environ.get(name) == "1"
SEQ, COMEND, ADDON, LOO, PUBMODE = flag("MR_SEQ"), flag("MR_COMEND"), flag("MR_ADDON"), flag("MR_LOO"), flag("MR_PUB")
SEQCUT = float(os.environ.get("MR_SEQCUT", "1")); assert 0 < SEQCUT <= 1 and (SEQ or SEQCUT == 1), SEQCUT
ABL = os.environ.get("MR_ABL", ""); assert ABL in ("", "ctlevels", "euclid"), ABL
LEVEL = "ct" if ABL == "ctlevels" else "block"; METRIC_MODE = "euclid" if ABL == "euclid" else "gram"
CALSEED = os.environ.get("MR_CALSEED", ""); CALTAG = f"cal{CALSEED}" if CALSEED else "cal"
PUBTAG = os.environ.get("MR_PUBTAG", "pub"); PUBC_ENV = os.environ.get("MR_PUBC", "")
assert (PUBTAG == "pub") == (not PUBC_ENV) and (PUBMODE or PUBTAG == "pub") and PUBTAG.isalnum(), "MR_PUBTAG other than pub needs MR_PUBC and MR_PUB=1"
B3NVAR = flag("MR_B3NVAR")

# ---- the results tree of this run ----
VARIANT = ABL or (PUBTAG if PUBMODE else (f"s{CALSEED or 0}" if (CALSEED or LOO) else ""))
SEQNAME = "seq" if SEQCUT == 1 else f"seqcut{SEQCUT:g}"
if COMEND:
    COMABL = ABL if flag("MR_B4NABL") else ""
    COMVAR = (PUBTAG if PUBMODE else (f"s{CALSEED}" if CALSEED else "")) if flag("MR_B4NVAR") else ""
    assert not (SEQ or ADDON or B3NVAR or LOO) and not (PUBMODE and CALSEED) and not (COMABL and COMVAR) and ABL == COMABL and bool(PUBMODE or CALSEED) == bool(COMVAR), \
        "MR_COMEND runs alone; a variant needs MR_B4NABL=1 (ablations) or MR_B4NVAR=1 (public images, calibration draws)"
    TREE = f"v_com_{COMABL or COMVAR}" if (COMABL or COMVAR) else "v_com"
elif B3NVAR:
    VTAG = ABL or (PUBTAG if PUBMODE else (f"s{CALSEED}" if CALSEED else ""))
    assert VTAG and not LOO and (SEQ != ADDON), "MR_B3NVAR needs one variant and either MR_SEQ or MR_ADDON"
    TREE = f"v_anchor_{VTAG}" if ADDON else f"v_{SEQNAME}_{VTAG}"
elif SEQ or ADDON:
    assert not VARIANT and not (SEQ and ADDON), "a variant of the sequential ends or the anchored points needs MR_B3NVAR=1"
    TREE = f"v_{SEQNAME}" if SEQ else "v_anchor"
else:
    COMVAR = ""; TREE = f"v_{VARIANT}" if VARIANT else ""
OUT = os.environ.get("MR_OUT") or os.path.join(ROOT, "results", "clipclose", "closedlen", *([TREE] if TREE else []), KIND, ARCH, SUB)
os.makedirs(OUT, exist_ok=True); MJ = os.path.join(OUT, "mrun.json")
res = H.jload(MJ) if os.path.exists(MJ) else {"kind": KIND, "arch": ARCH, "sub": SUB, "tasks": TASKS, "ret": RET, "rank": RANK, "cfg": "block-3-ct", "level": LEVEL, "ridge": RIDGE,
                                             "kr": list(KR), "a_grid": A_GRID, "cal": {}, "acc": {}, "R": {}, "picks": {}, "fro": {}, "seconds": {}}
if ABL: res["ablation"] = {"variant": ABL, "level": LEVEL, "metric": METRIC_MODE}
def save(): H.jdump(MJ, res)

# DINOv2 names its attention output projection "dense"; every family's own name selects the Gram of that cell's input
OUTCELL = getattr(H, "_OUT", "out_proj")
def gkey(c): return (c[0], "o" if c[1] == OUTCELL else "qkv")

# ================= the uploads =================
t0 = time.time(); pool, D = H.load_deltas(); tgt = H.target(); assert tgt in ("median", "min"), tgt
C = H.clip()
if PUBMODE:   # public images of each task at the server; the pseudo-answers come from each client's adapter on them
    PUBC = PUBC_ENV or (os.path.join(ROOT, "results", "clipclose", "zeroup", "cache") if KIND in ("dc", "knots") else os.path.join(ROOT, "results", "clipclose", "zeroup2", f"cache_{KIND}{getattr(H, 'RES', 224)}"))
    Xp = {t: np.load(os.path.join(PUBC, f"{t}_pub_X.npy"), mmap_mode="r") for t in TASKS}; yp = {t: np.load(os.path.join(PUBC, f"{t}_pub_y.npy")) for t in TASKS}
    HD0 = H.heads(C); pub_pseudo, pub_acc = {}, {}
    for t in TASKS: C.set_delta(D[t]); pr = H.predict(C, Xp[t], HD0[t]); pub_pseudo[t] = {"pseudo": pr.tolist()}; pub_acc[t] = float((pr == yp[t]).mean())
    res["public"] = {"client_pub_acc": pub_acc, "source": os.path.relpath(PUBC, ROOT)}; log("  public pseudo-labels made: adapter accuracy on public images " + " ".join(f"{t} {pub_acc[t]:.3f}" for t in TASKS))
if CALSEED:   # another calibration draw: Grams, pseudo-answers and scores on it
    HD0 = H.heads(C); cs_pseudo = {}
    for t in TASKS: C.set_delta(D[t]); cs_pseudo[t] = {"pseudo": H.predict(C, H.load_x(t, CALTAG)[0], HD0[t]).tolist()}
    res["calseed"] = CALSEED; log(f"  calibration draw {CALSEED}: pseudo-labels made for {K} clients")

SRVG = PUBMODE and PUBTAG.startswith("pubsg")   # no Gram upload: a read of a client's calibration images raises
if SRVG:
    _load_x = H.load_x
    def _no_client_images(t, tag):
        assert not str(tag).startswith("cal"), f"no-Gram-upload variant: a client's calibration images were asked for ({t}, {tag})"
        return _load_x(t, tag)
    H.load_x = _no_client_images; res["server_grams"] = {"rule": "no Gram upload: every Gram is built at the server on the variant's public images with the client's adapter"}; log("  no Gram upload: Grams built at the server on the public images")
GSN = re.fullmatch(r"pubgn(\d+)(?:r(\d+))?(?:d(\d+))?", PUBTAG) if PUBMODE else None   # Grams over the first N images of the client's nested training set (data_gram_sizes.py); read from the audited file where audit_gram_size.py stored it
GST = re.fullmatch(r"pubst([a-z0-9]+)", PUBTAG) if PUBMODE else None                    # Grams read from the release that audit_gram_dp.py wrote and attacked
def read_grams(path): z = np.load(path); return {(int(k.split("_", 1)[0]), k.split("_", 1)[1]): z[k] for k in z.files}
t1 = time.time(); grams = {}; gsz = {}; gsrc = {}
for t in TASKS:
    if SRVG: Xg = Xp[t]
    elif GSN:
        n = int(GSN.group(1)); n = min(n, H.jload(os.path.join(ROOT, "results", "privaudit3", "sets", "meta.json"))[t]["largest_gram"]) if n > 500 else n; gf = os.path.join(ROOT, "results", "privaudit3", "grams", ARCH, f"{t}_n{n}.npz")
        if os.path.exists(gf): grams[t] = read_grams(gf); gsz[t] = n; gsrc[t] = "stored"; continue
        X5 = H.load_x(t, "caln500")[0]; Xg = X5[:n] if n <= 500 else np.concatenate([np.asarray(X5), np.asarray(np.load(os.path.join(ROOT, "results", "privaudit3", "sets", f"{t}_extra_X.npy"), mmap_mode="r")[300:300 + n - 500])])
    elif GST: grams[t] = read_grams(os.path.join(ROOT, "results", "privaudit4", "grams", ARCH, f"{t}_{GST.group(1)}.npz")); continue
    else: Xg = H.load_x(t, CALTAG)[0]
    gsz[t] = int(len(Xg)); gsrc[t] = "computed"; grams[t] = H.collect_grams(C, D[t], Xg, range(H.NLAYERS)); del Xg
if GSN: res["gram_size"] = {"asked": int(GSN.group(1)), "images": gsz, "source": gsrc}; log(f"  Gram size {GSN.group(1)}: images per client " + " ".join(f"{t} {gsz[t]} ({gsrc[t]})" for t in TASKS))
if GST: res["stored_grams"] = {"name": GST.group(1), "keys": len(grams[TASKS[0]])}; log(f"  stored release {GST.group(1)}: Grams read for {len(grams)} clients")
GNM = (re.fullmatch(r"pubg(\d+)", PUBTAG) or re.fullmatch(r"pubgn\d+r(\d+)(?:d(\d+))?", PUBTAG)) if PUBMODE else None   # noise of relative Frobenius size m/1000 on every uploaded Gram; pubg0 adds none
if GNM:
    from src.supplement import gram_noise
    RHO = int(GNM.group(1)) / 1000.0; GDRAW = int(GNM.group(2) or 0) if GNM.re.groups > 1 else 0
    grams = {t: {k: gram_noise.noisy(v, RHO, t, k, GDRAW) for k, v in grams[t].items()} for t in TASKS}; res["gnoise"] = {"rho": RHO, "draw": GDRAW, "rule": "G + symmetric Gaussian noise with Frobenius norm rho ||G||_F, projected to positive semi-definite"}; log(f"  Gram noise rho {RHO}")

# ================= the Gram end =================
RM, kept = H.truncate({c: H.regmean_cell(c, D, grams) for c in CELLS}, RANK); res["seconds"]["regmean"] = time.time() - t1; res["fro"]["regmean128"] = ops.fro(RM)   # one-shot: RegMean on the uploaded Grams, truncated to the merged rank
if SEQ:   # sequential: layer by layer, each layer's Grams taken through the merged layers below it; above the cut the one-shot cells
    t1 = time.time(); XS = {t: (Xp[t] if (B3NVAR and PUBMODE) else H.load_x(t, CALTAG)[0]) for t in TASKS}; SQ = {}
    NSEQ = int(round(SEQCUT * H.NLAYERS))
    for l in range(H.NLAYERS):
        if l >= NSEQ:
            for c in CELLS:
                if c[0] == l: SQ[c] = H.regmean_cell(c, D, grams)
            continue
        gl = {t: H.collect_grams(C, {c: (SQ[c] if c in SQ else D[t][c]) for c in CELLS}, XS[t], [l]) for t in TASKS}
        for c in CELLS:
            if c[0] == l: SQ[c] = H.regmean_cell(c, D, gl)
    res["fro"]["regmean128_oneshot"] = res["fro"]["regmean128"]; RM, kept = H.truncate(SQ, RANK); del SQ, XS; res["fro"]["regmean128"] = ops.fro(RM); res["seconds"]["regmean_seq"] = time.time() - t1
    res["seq"] = {"rule": "RegMean++ (layer-sequential Grams on the uploaded calibration inputs, lam .9), truncated to rank 16K", "kept_mean": float(np.mean(kept)), "cut": SEQCUT, "n_seq_layers": NSEQ}; log(f"  sequential Gram end built ({res['seconds']['regmean_seq']:.0f}s)")
if COMEND:   # the stored dense Chain of Merges merge of this pool, truncated to the merged rank
    CMP = os.path.join(ROOT, "results", "clipclose", *(["com_var", COMVAR] if COMVAR else ["com"]), KIND, ARCH, SUB, "com_merge.npz"); zc = np.load(CMP); CM = {c: zc[H.cname(c)].astype(np.float32) for c in CELLS}; del zc
    res["fro"]["regmean128_oneshot"] = res["fro"]["regmean128"]; RM, kept = H.truncate(CM, RANK); del CM; res["fro"]["regmean128"] = ops.fro(RM)
    res["com"] = {"rule": "Chain of Merges, truncated to rank 16K" + (f"; built on {COMVAR}" if COMVAR else ""), "kept_mean": float(np.mean(kept)), "source": os.path.relpath(CMP, ROOT)}; log("  Chain of Merges Gram end loaded")

# ================= energies for the closed-form length =================
# s*(W)^2 = E_own / e(W): E_own is the response energy of the clients' own updates on their own shrunk Grams, e(W) that of W on the pooled shrunk Gram
GSUM, E_OWN = {}, 0.0
for t in TASKS:
    shr = {k: H.shrink(g).astype(np.float64) for k, g in grams[t].items()}
    for k, g in shr.items(): GSUM[k] = GSUM.get(k, 0) + g
    for c in CELLS: d_ = D[t][c].astype(np.float64); E_OWN += float(np.sum((d_ @ shr[gkey(c)]) * d_))
res.setdefault("closed", {"E_own": E_OWN, "points": {}})

# ================= the two spectral ends =================
norms = {t: {c: float(np.linalg.norm(D[t][c])) for c in CELLS} for t in TASKS}; DF0 = D[TASKS[0]][CELLS[0]].copy() if ADDON else None
del D; Gs = {t: {k: H.shrink(g).astype(np.float32) for k, g in grams[t].items()} for t in TASKS}; del grams; GSF = Gs if ADDON else None
fac = {t: {c: (pool[t][c]["B"], pool[t][c]["A"]) for c in CELLS} for t in TASKS}; del pool; FAC = fac if ADDON else None
t1 = time.time()
END = {"w": spectral_end.m_endpoint(CELLS, fac, norms, Gs, gkey, tgt, LEVEL, RIDGE, KR, whiten=True, wk=None, metric_mode=METRIC_MODE),     # the norm-equalized side
       "r": spectral_end.m_endpoint(CELLS, fac, norms, Gs, gkey, tgt, LEVEL, RIDGE, KR, whiten=False, wk=None, metric_mode=METRIC_MODE)}    # the raw side
res["fro"]["end_w"] = ops.fro(END["w"]); res["fro"]["end_r"] = ops.fro(END["r"]); res["seconds"]["ends"] = time.time() - t1; res["target"] = tgt; del fac, Gs; save()
log(f"{KIND} {ARCH} {SUB}: {K} tasks, target {tgt}; ends built ({res['seconds']['ends']:.0f}s); fro " + " ".join(f"{k} {v:.1f}" for k, v in res["fro"].items()))

def point(side, a):
    """The merge at weight a of the Gram end on the segment to the spectral end of this side, truncated to the merged rank."""
    if a == 1.0: return RM
    if a == 0.0: return H.truncate(END[side], RANK)[0]
    return H.truncate({c: a * RM[c] + (1 - a) * END[side][c] for c in CELLS}, RANK)[0]

# ================= the score =================
HD = H.heads(C)
pseudo = pub_pseudo if PUBMODE else (cs_pseudo if CALSEED else H.jload(os.path.join(H.SEL, "pseudo.json")))
Xc = {t: (Xp[t], yp[t]) for t in TASKS} if PUBMODE else {t: H.load_x(t, CALTAG) for t in TASKS}
LAST_CLIENT = {}   # per-client agreement of the last scored candidate
def score(delta):
    """[mean agreement with the clients' pseudo-answers (the score), mean agreement with the labels (reference only)] on the calibration inputs."""
    C.set_delta(delta); sp, sl = [], []
    for t in TASKS: pc = H.predict(C, Xc[t][0], HD[t]); sp.append(float((pc == np.asarray(pseudo[t]["pseudo"])).mean())); sl.append(float((pc == np.asarray(Xc[t][1])).mean()))
    LAST_CLIENT.clear(); LAST_CLIENT.update({t: [sp[i], sl[i]] for i, t in enumerate(TASKS)})
    return [float(np.mean(sp)), float(np.mean(sl))]
def scaled(d, s): return d if s == 1.0 else {c: (s * d[c]).astype(np.float32) for c in CELLS}
def s_star(d): return float(np.sqrt(res["closed"]["E_own"] / max(sum(float(np.sum((d[c].astype(np.float64) @ GSUM[gkey(c)]) * d[c].astype(np.float64))) for c in CELLS), 1e-30)))
def pk_key(side, a): return "rm" if a == 1.0 else f"{side}@{a}"

# every point of the two segments at its closed-form length: thirteen points (the Gram end is shared)
cands = [("w", a) for a in A_GRID] + [("r", a) for a in A_GRID if a < 1.0]
for side, a in cands:
    k = pk_key(side, a)
    if k in res["closed"]["points"]: continue
    t1 = time.time(); d = point(side, a); st = s_star(d); sc_ = score(scaled(d, st))
    res["closed"]["points"][k] = {"side": side, "a": a, "s": st, "s_P": sc_[0], "s_L": sc_[1], "clients": dict(LAST_CLIENT)}; save(); log(f"  closed {k}: s* {st:.3f} s_P {sc_[0]:.4f} ({time.time() - t1:.0f}s)")
PT = res["closed"]["points"]
pick_by = lambda m: max(PT.values(), key=lambda v: (round(v[m], 12), v["a"], 1 if v["side"] == "w" else 0))   # ties go to the larger a, then to the norm-equalized side

# ================= test retention of the pick =================
homef = {t: H.jload(os.path.join(H.EVAL, f"client_{t}.json"))[t] for t in TASKS}   # each client's own test accuracy
def retention(acc): return float(np.mean([acc[t] / max(homef[t], 1e-9) for t in RET]))
def eval_at(name, side, a):
    if name in res["R"]: return
    d = point(side, a); st = PT[pk_key(side, a)]["s"]; C.set_delta(scaled(d, st)); acc_ = {}; PRED_ = {}
    for t in TASKS: X_, y_ = H.load_x(t, "test"); p_ = H.predict(C, X_, HD[t]); PRED_[t] = p_; acc_[t] = float((p_ == np.asarray(y_)).mean())
    if os.environ.get("MR_PREDOUT") and name == "L1":
        D_ = H.load_deltas()[1]
        for t in TASKS: C.set_delta(D_[t]); PRED_["own_" + t] = H.predict(C, H.load_x(t, "test")[0], HD[t])
        del D_
        np.savez(os.environ["MR_PREDOUT"], **PRED_)
    res["acc"][name] = acc_; res["R"][name] = retention(acc_); save(); log(f"  evaluated {name} = {side} a={a} s*={st:.3f}")
P_, L_ = pick_by("s_P"), pick_by("s_L"); res["picks"] = {"P": dict(P_), "L": dict(L_)}; save()
log(f"  pick by the score {P_['side']} a={P_['a']} s*={P_['s']:.3f}; by the labels (reference) {L_['side']} a={L_['a']} s*={L_['s']:.3f}")
eval_at("L1", P_["side"], P_["a"]); eval_at("L1_lab", L_["side"], L_["a"])   # L1: the point this run returns; L1_lab: what the calibration labels would pick

# ================= the anchored points =================
if ADDON:
    # W = argmin sum_k |W G_k^.5|^2 + w_k |(W - dW_k) G_k^.5|^2 per cell, with w_k = c u_k / mean(u) and u_k one minus the base model's agreement with client k
    t1 = time.time(); AO = res.setdefault("addon", {"cgrid": C_GRID, "points": {}}); AP = AO["points"]
    assert np.allclose((FAC[TASKS[0]][CELLS[0]][0] @ FAC[TASKS[0]][CELLS[0]][1]), DF0, atol=1e-4), "factors do not reproduce the update"
    def eval_delta(name, delta):
        if name in res["R"]: return
        C.set_delta(delta); acc_ = {}
        for t in TASKS: X_, y_ = H.load_x(t, "test"); acc_[t] = float((H.predict(C, X_, HD[t]) == np.asarray(y_)).mean())
        res["acc"][name] = acc_; res["R"][name] = retention(acc_); save(); log(f"  evaluated {name}")
    zero = {c: np.zeros((FAC[TASKS[0]][c][0].shape[0], FAC[TASKS[0]][c][1].shape[1]), dtype=np.float32) for c in CELLS}
    if "base" not in AO: sb_ = score(zero); AO["base"] = {"s_P": sb_[0], "s_L": sb_[1], "clients": dict(LAST_CLIENT)}; save(); log(f"  base model: s_P {sb_[0]:.4f}")
    def refit(ag, c_):
        los = {t: 1.0 - ag[t] for t in TASKS}; mean_ = max(float(np.mean(list(los.values()))), 1e-9); rho = {t: c_ * los[t] / mean_ for t in TASKS}; out = {}; Mg = {}
        for c in CELLS:
            gk = gkey(c)
            if gk not in Mg: A_ = GSUM[gk] + sum(rho[t] * GSF[t][gk].astype(np.float64) for t in TASKS); Mg[gk] = A_ + 1e-6 * float(np.trace(A_)) / A_.shape[0] * np.eye(A_.shape[0])
            rhs = sum(rho[t] * (FAC[t][c][0].astype(np.float64) @ (FAC[t][c][1].astype(np.float64) @ GSF[t][gk].astype(np.float64))) for t in TASKS)
            out[c] = np.linalg.solve(Mg[gk].T, rhs.T).T.astype(np.float32)
        return out, rho
    base_agreement = {t: AO["base"]["clients"][t][0] for t in TASKS}
    for c_ in C_GRID:
        k = f"N@{c_}"
        if k not in AP:
            t2 = time.time(); d_, rho = refit(base_agreement, c_); sc_ = score(d_); AP[k] = {"anchor": "N", "c": c_, "s_P": sc_[0], "s_L": sc_[1], "clients": dict(LAST_CLIENT), "rho": rho}; save(); log(f"  anchored {k}: s_P {sc_[0]:.4f} ({time.time() - t2:.0f}s)")
            eval_delta(f"A:{k}", d_)
        elif f"A:{k}" not in res["R"]: eval_delta(f"A:{k}", refit(base_agreement, c_)[0])
    def choose(m):   # the pick stays unless an anchored point agrees strictly better; ties among anchored points go to the smaller c
        best = max(AP, key=lambda k: (round(AP[k][m], 12), -AP[k]["c"])); return best if round(AP[best][m], 12) > round(P_[m], 12) else None
    kP, kL = choose("s_P"), choose("s_L"); AO["pick_P"], AO["pick_L"] = kP, kL; save(); log(f"  anchored choice: by the score {kP}, by the labels {kL} (pick s_P {P_['s_P']:.4f})")
    for nm_, k_ in (("AO", kP), ("AO_lab", kL)):
        src = "L1" if k_ is None else f"A:{k_}"; res["R"][nm_] = res["R"][src]; res["acc"][nm_] = res["acc"][src]; save()
    AO["seconds"] = time.time() - t1; save(); log(f"  anchored points done ({(time.time() - t1) / 60:.1f} min)")

# ================= the choice without each client's votes =================
if LOO:
    res.setdefault("loo", {}); base_ = pk_key(P_["side"], P_["a"])
    for k_ in TASKS:
        if k_ in res["loo"]: continue
        v_ = max(PT.values(), key=lambda v: (round(float(np.mean([v["clients"][t][0] for t in TASKS if t != k_])), 12), v["a"], 1 if v["side"] == "w" else 0)); kk_ = pk_key(v_["side"], v_["a"]); nm_ = "L1" if kk_ == base_ else f"loo:{kk_}"
        eval_at(nm_, v_["side"], v_["a"]); res["loo"][k_] = {"side": v_["side"], "a": v_["a"], "s": v_["s"], "key": kk_, "R": res["R"][nm_], "changed": kk_ != base_}; save()

res["done"] = True; res["seconds"]["total"] = time.time() - t0; save(); log(f"pool done ({(time.time() - t0) / 60:.0f} min)")
