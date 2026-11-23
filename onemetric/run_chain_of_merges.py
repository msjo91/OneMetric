#!/usr/bin/env python
"""Chain of Merges on one vision pool, re-implemented from the paper and its released vision defaults, on the same 200 unlabeled calibration inputs per client as the method. Writes results/clipclose/com/<kind>/<backbone>/<pool>/{com.json, com_merge.npz}; the merge is a baseline row and the fourth Gram end of the method.
Env: MR_KIND, ARCH / SIG_ARCH / DINO_ARCH and CLIPK_SUB as in run_method. COM_VAR and COM_PUBC build the merge on other inputs (public images, other calibration draws) for the variants of the supplement; COM_OUT overrides the folder."""
from onemetric.paths import ROOT as _OM_ROOT
import os, sys, time, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = _OM_ROOT
KIND = os.environ["MR_KIND"]; assert KIND in ("dc", "knots", "siglip", "dino"), KIND

if KIND == "dc": from onemetric import harness_clip as H
elif KIND == "knots": from onemetric import harness_knots as H
elif KIND == "siglip": from onemetric import harness_siglip as H
else: from onemetric import harness_dino as H
if KIND == "knots" and not hasattr(H, "SUBTAG"): H.SUBTAG = "8task"
ARCH, SUB, TASKS, CELLS, RANK = H.ARCH, H.SUBTAG, H.TASKS, H.CELLS, H.RANK; log = H.log; RET = getattr(H, "RET", TASKS)
SMOKE = os.environ.get("COM_SMOKE") == "1"; NL = 2 if SMOKE else H.NLAYERS; NCAL = 8 if SMOKE else None
VAR = os.environ.get("COM_VAR", "")   # "" (the uploaded calibration inputs; nothing changes) | "pub" (public calibration images, as MR_PUB in run_method.py) | "s1", "s2" (calibration draws 1 and 2, as MR_CALSEED); own tree results/clipclose/com_var/<VAR>
PUBC_ENV = os.environ.get("COM_PUBC", "")   # another image set for the build (a directory of <task>_pub_X.npy), under its own COM_VAR tag
import re; NVAR = bool(re.fullmatch(r"sn\d+", VAR))   # calibration sets of another size (MR_CALSEED=n<N> in run_method.py, cache <task>_caln<N>_{X,y}.npy), tree com_var/sn<N>
assert (VAR in ("", "pub", "s1", "s2") or NVAR) != bool(PUBC_ENV) and (not PUBC_ENV or VAR.isalnum()), (VAR, PUBC_ENV); assert not VAR or KIND == "dc" or os.environ.get("COM_VARFAM") == "1", "run on the CLIP benchmark pools"   # COM_VARFAM=1 opens the variants to the other families; nothing changes without it. Backup run_chain_of_merges.py.bak_g156
OUT = os.environ["COM_OUT"] if os.environ.get("COM_OUT") else os.path.join(ROOT, "results", "clipclose", *(["com_var", VAR] if VAR else ["com"]), *(["smoke"] if SMOKE else []), KIND, ARCH, SUB); os.makedirs(OUT, exist_ok=True); MJ = os.path.join(OUT, "com.json")   # COM_OUT: a code check written elsewhere
res = H.jload(MJ) if os.path.exists(MJ) else {"kind": KIND, "arch": ARCH, "sub": SUB, "tasks": TASKS, "rank": RANK, "smoke": SMOKE, "R": {}, "acc": {}, "seconds": {}, "weights": {},
                                             "config": {"alpha": 1.0, "scale": 1.0, "weight_offset": 0.1, "normalize_features": True, "normalize_by_tokens": True, "pinv_rtol": 1e-6,
                                                        "start": "base + mean task vector", "calibration": {"": "the 200 uploaded calibration inputs per client (unlabeled)", "pub": "200 public images per task, nothing private", "s1": "calibration draw 1 (200 per client, unlabeled)", "s2": "calibration draw 2 (200 per client, unlabeled)"}.get(VAR, f"{VAR[2:]} calibration inputs per client" if NVAR else f"200 images per client from {PUBC_ENV}")}}
def save(): H.jdump(MJ, res)
class _Stop(Exception): pass
OUTN = H._OUT if KIND == "dino" else "out_proj"   # the attention output projection; DINOv2 names it differently
def gkey(c): return (c[0], "o" if c[1] == OUTN else "qkv")
def stage_module(C, l, stage):
    if KIND == "dino": return H._dino_linear(C.model, (l, H._QKV[0] if stage == "qkv" else H._OUT))
    sa = C.model.vision_model.encoder.layers[l].self_attn; return sa.q_proj if stage == "qkv" else sa.out_proj
def normgram(C, X, l, stage):
    """Sum over all tokens of f f^T, f = token feature / ||token feature||, at the input of layer l's q_proj (stage qkv) or out_proj (stage o); the forward stops there."""
    mod = stage_module(C, l, stage); acc = {"g": None, "n": 0}
    def hook(_, inp):
        x = inp[0].reshape(-1, inp[0].shape[-1]).double(); x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        acc["g"] = (x.T @ x) if acc["g"] is None else acc["g"] + x.T @ x; acc["n"] += x.shape[0]; raise _Stop
    h = mod.register_forward_pre_hook(hook)
    try:
        for i in range(0, len(X), H.BS):
            try:
                with torch.no_grad(): C.features(np.asarray(X[i:i + H.BS]), bs=H.BS)
            except _Stop: pass
    finally: h.remove()
    return acc["g"].cpu().numpy(), acc["n"]
def solve(cells, D, G, N):
    d = next(iter(G.values())).shape[0]; off = ~np.eye(d, dtype=bool)
    r = np.array([np.abs(G[t][off]).mean() for t in TASKS]); w = (r - r.min() + 0.1) / (r.max() - r.min() + 0.1)
    Gn = {t: G[t] / N[t] for t in TASKS}; gsum = sum(wi * Gn[t] for wi, t in zip(w, TASKS)); P = np.linalg.pinv(gsum, rcond=1e-6, hermitian=True)
    return {c: (sum(wi * (D[t][c].astype(np.float64) @ Gn[t]) for wi, t in zip(w, TASKS)) @ P).astype(np.float32) for c in cells}, w
t0 = time.time(); pool, D = H.load_deltas(); C = H.clip()
if VAR == "pub" or PUBC_ENV: PUBC = PUBC_ENV or os.path.join(ROOT, "results", "clipclose", "zeroup", "cache"); X = {t: np.load(os.path.join(PUBC, f"{t}_pub_X.npy"), mmap_mode="r") for t in TASKS}
else: X = {t: H.load_x(t, "cal" + VAR[1:])[0] for t in TASKS}
if NCAL: X = {t: X[t][:NCAL] for t in TASKS}
log(f"CoM {KIND} {ARCH} {SUB}: {len(TASKS)} clients, {len(X[TASKS[0]])} calibration inputs each, {NL} of {H.NLAYERS} layers" + (" (SMOKE)" if SMOKE else ""))
if "com" not in res["R"] and not res.get("built"):
    M = {c: np.mean([D[t][c] for t in TASKS], axis=0).astype(np.float32) for c in CELLS}; t1 = time.time()
    for l in range(NL):
        for stage in ("qkv", "o"):
            C.set_delta(M); G, N = {}, {}
            for t in TASKS: G[t], N[t] = normgram(C, X[t], l, stage)
            cells = [c for c in CELLS if c[0] == l and gkey(c)[1] == stage]; new, w = solve(cells, D, G, N); M.update(new)
            res["weights"][f"{l}/{stage}"] = [float(x) for x in w]
        log(f"  layer {l} merged ({time.time() - t1:.0f}s)")
    res["seconds"]["merge"] = time.time() - t1; res["built"] = True; save()
    if SMOKE:
        for c in CELLS:
            if c[0] < NL: assert np.isfinite(M[c]).all() and M[c].shape == D[TASKS[0]][c].shape, c
        log(f"SMOKE OK: {sum(1 for c in CELLS if c[0] < NL)} cells merged, weights {res['weights']}"); sys.exit(0)
    np.savez_compressed(os.path.join(OUT, "com_merge.npz"), **{H.cname(c): M[c].astype(np.float16) for c in CELLS})
else:
    z = np.load(os.path.join(OUT, "com_merge.npz")); M = {c: z[H.cname(c)].astype(np.float32) for c in CELLS}
HD = H.heads(C); homef = {t: H.jload(os.path.join(H.EVAL, f"client_{t}.json"))[t] for t in TASKS}
def evaluate(name, delta):
    if name in res["R"]: return
    t1 = time.time(); C.set_delta(delta); acc = {}
    for t in TASKS: Xt, yt = H.load_x(t, "test"); acc[t] = float((H.predict(C, Xt, HD[t]) == np.asarray(yt)).mean())
    res["acc"][name] = acc; res["R"][name] = float(np.mean([acc[t] / max(homef[t], 1e-9) for t in RET])); res["seconds"][f"eval_{name}"] = time.time() - t1; save(); log(f"  evaluated {name} ({time.time() - t1:.0f}s)")
evaluate("com", M)
evaluate("com_trunc", H.truncate(M, RANK)[0])
res["done"] = True; res["seconds"]["total"] = time.time() - t0; save(); log(f"CoM pool done ({(time.time() - t0) / 60:.1f} min)")
