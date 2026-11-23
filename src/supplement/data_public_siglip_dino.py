#!/usr/bin/env python
"""zero-upload rule P (protocol) on the SigLIP 2 and DINOv2 lines. env ZU_LINE = siglip (SIG_ARCH, CLIPK_SUB)
| dino (DINO_ARCH, CLIPK_SUB; DINO_ROOT=dino2, DINO_TASKS=h16 set here). --stage pub builds the line's public image cache at the harness resolution (200 images per task from the test
split outside the stored seed-0 2,000-image draw, else from the train split outside the seed-0 200-image calibration draw; seed 1); --stage pool runs the selector with server-side
pseudo-labels over the union candidates, the shortening stage, eleven published merges (ACE left out) and the fifteen rescaled copies, compares with the private F25-NM pick
and evaluates the zero-upload pick on the stored test draw when its retention is not stored. Z2_SMOKE=1: cheap published merges only, output to $Z2_OUT. Logs carry no retention number."""
from src.paths import ROOT as _OM_ROOT
import os, sys, time, random, argparse, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = _OM_ROOT
ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["pub", "pool"], required=True); args = ap.parse_args()
LINE = os.environ["ZU_LINE"]; assert LINE in ("siglip", "dino"), LINE
if LINE == "siglip": from src.backbones import harness_siglip as H; KIND = "siglip"
else:
    os.environ.setdefault("DINO_ROOT", "dino2"); os.environ.setdefault("DINO_TASKS", "h16"); from src.backbones import harness_dino as H; KIND = "dino"; from src.backbones import dino_data as DD
from src.baselines import merges_ext as ops_new; from src.method import roomlib; from src.baselines import merges as ops
PUBC = os.path.join(ROOT, "results", "clipclose", "zeroup2", f"cache_{LINE}{H.RES}"); os.makedirs(PUBC, exist_ok=True)
def pct(s): return int(round(s * 100))
def load_split(t, split):
    if LINE == "dino": ds = DD.split(t, split); labcol, imgcol, wds = DD.cols(t, ds); return ds, labcol, imgcol
    from datasets import load_dataset; repo, cfg = H.DS[t]; ds = load_dataset(repo, cfg, split=split) if cfg else load_dataset(repo, split=split); wds = repo.startswith("clip-benchmark/")
    return ds, ("cls" if wds else "label"), (("png" if "png" in ds.column_names else "jpg") if wds else "image")
if args.stage == "pub":
    from concurrent.futures import ThreadPoolExecutor
    tasks = os.environ["ZU_TASKS"].split(",") if os.environ.get("ZU_TASKS") else list(H.TASKS16)
    srcp = os.path.join(PUBC, "src.json"); src = H.jload(srcp) if os.path.exists(srcp) else {}
    for t in tasks:
        if os.path.exists(os.path.join(PUBC, f"{t}_pub_X.npy")): continue
        t0 = time.time(); ds, labcol, imgcol = load_split(t, "test"); n = len(ds); draw = set(random.Random(0).sample(range(n), min(H.N_TEST, n))); rest = [i for i in range(n) if i not in draw]; where = "test"
        if len(rest) < 200:
            ds, labcol, imgcol = load_split(t, "train"); cal = set(random.Random(0).sample(range(len(ds)), min(H.N_CAL, len(ds)))); rest = [i for i in range(len(ds)) if i not in cal]; where = "train"
        idx = sorted(random.Random(1).sample(rest, 200)); sub = ds.select(idx)
        with ThreadPoolExecutor(16) as ex: X = np.stack(list(ex.map(H.prep, sub[imgcol], chunksize=16)))
        y = np.asarray(sub[labcol], dtype=np.int64); np.save(os.path.join(PUBC, f"{t}_pub_X.npy"), X); np.save(os.path.join(PUBC, f"{t}_pub_y.npy"), y)
        src[t] = {"source": where, "n_test_split": n, "n_outside_draw": len(rest), "idx": idx}; H.jdump(srcp, src); H.log(f"pub {t}: 200 from {where} ({len(rest)} available; {time.time() - t0:.0f}s)")
    H.log("pub cache done"); sys.exit(0)
import torch
L = roomlib.Lib(H, ROOT, KIND, ops, ops_new); ARCH, SUB = L.ARCH, L.SUB; SMOKE = os.environ.get("Z2_SMOKE") == "1"
ROOMP = os.path.join(ROOT, "results", "clipclose", "lwin", "siglip", ARCH, SUB, "room.json") if LINE == "siglip" else os.path.join(ROOT, "results", "dino2", "lwin", "dino", ARCH, SUB, "room.json")
ROOM = H.jload(ROOMP); OURS = ROOM["ours"]; P = H.jload(os.path.join(L.UA, "picks.json"))
PH = H.jload(os.path.join(ROOT, "results", "f25llm", "f25nm_posthoc.json")); PRIV_ROW = next(r for r in PH[LINE]["rows"] if r["arch"] == ARCH and r["sub"] == SUB); PRIV = PRIV_ROW["nm"]
OUT = os.environ["Z2_OUT"] if SMOKE else os.path.join(ROOT, "results", "clipclose", "zeroup2", LINE, ARCH, SUB); os.makedirs(OUT, exist_ok=True); ZJ = os.path.join(OUT, "zeroup.json")
Xp = {t: np.load(os.path.join(PUBC, f"{t}_pub_X.npy"), mmap_mode="r") for t in H.TASKS}; yp = {t: np.load(os.path.join(PUBC, f"{t}_pub_y.npy")) for t in H.TASKS}; SRC = H.jload(os.path.join(PUBC, "src.json"))
t0 = time.time(); C = H.clip(); HD = H.heads(C); pool, D = L.pool_deltas(); W0 = {c: C.W0[c].float().cpu().numpy() for c in H.CELLS}
pub_pseudo, client_pub_acc = {}, {}
for t in H.TASKS: C.set_delta(D[t]); pub_pseudo[t] = H.predict(C, Xp[t], HD[t]); client_pub_acc[t] = float((pub_pseudo[t] == yp[t]).mean())
H.log(f"zero-upload {LINE} {ARCH} {SUB}: server-side pseudo-labels made for {len(H.TASKS)} clients ({time.time() - t0:.0f}s)")
def score(delta):
    C.set_delta(delta); sp, sl = [], []
    for t in H.TASKS: pr = H.predict(C, Xp[t], HD[t]); sp.append(float((pr == pub_pseudo[t]).mean())); sl.append(float((pr == yp[t]).mean()))
    return float(np.mean(sp)), float(np.mean(sl))
S_pub, S_Lpub = {}, {}
def sc(m, delta): S_pub[m], S_Lpub[m] = score(delta)
# ---- stage 1: union pick over the thirteen union candidates
union = list(dict.fromkeys(L.W + L.U))
for m in union: sc(m, L.load(m))
u = max(union, key=lambda m: (round(S_pub[m], 12), L.a_of(m))); H.log(f"union stage scored ({len(union)} candidates)")
# ---- stage 2: shortening of the union pick (classification pools)
ud = L.load(u); SS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
for s in SS:
    m = u if s == 1.0 else f"{u}_x{pct(s)}"
    if m not in S_pub: sc(m, {c: s * ud[c] for c in H.CELLS})
s_star = max(SS, key=lambda s: (round(S_pub[u if s == 1.0 else f"{u}_x{pct(s)}"], 12), s)); OURS_ZU = u if s_star == 1.0 else f"{u}_x{pct(s_star)}"
# ---- stage 3: published merges and rescaled copies (copies scored on the fly, not kept)
PUB11 = ["ta", "fedavg_w", "fedit", "ties", "knots_ties", "tsvm", "dcmerge", "isoc", "isocts", "ct", "pact_isoc"]
if SMOKE: PUB11 = ["ta", "fedavg_w", "fedit", "isoc"]
built = {}
for m in PUB11:
    t1 = time.time(); built[m] = L.rebuild_pub(m, W0) if m != "isoc" else L.load("isoc"); sc(m, built[m]); H.log(f"  published {m} built + scored ({time.time() - t1:.0f}s)")
NEWSPEC = {}
if not SMOKE:
    for g in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.3): NEWSPEC[f"ct_x{pct(g)}"] = ("ct", g)
    for s in (0.8, 0.9, 1.1): NEWSPEC[f"pact_x{pct(s)}"] = ("pact_isoc", s); NEWSPEC[f"dc_x{pct(s)}"] = ("dcmerge", s)
    for m, (b, g) in NEWSPEC.items(): sc(m, {c: g * built[b][c] for c in H.CELLS})
    H.log(f"rescaled copies scored ({len(NEWSPEC)})")
def cls(m): return 3 if m == OURS_ZU else 1
def scl(m): return int(m.rsplit("_x", 1)[1]) if "_x" in m and m.rsplit("_x", 1)[1].isdigit() else 100
final_set = [OURS_ZU] + PUB11 + list(NEWSPEC); ZU = max(final_set, key=lambda m: (round(S_pub[m], 12), cls(m), scl(m)))
# ---- retention of the zero-upload pick (stored where available, else evaluated on the stored test draw)
def R_stored(m):
    if m == OURS: return ROOM["R_ours"]
    if m in ROOM["R_pub"]: return ROOM["R_pub"][m]
    if m in ROOM["R"]: return ROOM["R"][m]
    return None
def delta_any(m):
    if m == OURS_ZU and m not in union: return {c: s_star * ud[c] for c in H.CELLS}
    if m in NEWSPEC: b, g = NEWSPEC[m]; return {c: g * built[b][c] for c in H.CELLS}
    if m in built: return built[m]
    return L.delta_of(m)
evaluated = {}
def R_any(m):
    r = R_stored(m)
    if r is not None: return r
    if m not in evaluated:
        t1 = time.time(); C.set_delta(delta_any(m)); acc = {}
        for t in H.TASKS: X, y = H.load_x(t, "test"); acc[t] = float((H.predict(C, X, HD[t]) == np.asarray(y)).mean())
        evaluated[m] = {"acc": acc, "R": L.R_of(acc)}; H.log(f"  evaluated one extra candidate on the stored draw ({time.time() - t1:.0f}s)")
    return evaluated[m]["R"]
R_zu = R_any(ZU); R_priv = PRIV_ROW["R_nm"]
# ---- private-signal comparison over common candidates
S_priv = dict(ROOM["pub_sP"]); S_priv.update(ROOM["s_P"]); S_priv[OURS] = ROOM["sP_ours"]
common = [m for m in S_pub if m in S_priv]
def rank(x): o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r
rho = float(np.corrcoef(rank([S_pub[m] for m in common]), rank([S_priv[m] for m in common]))[0, 1]) if len(common) >= 3 else None
out = {"line": LINE, "arch": ARCH, "sub": SUB, "tasks": H.TASKS, "public_source": {t: SRC[t]["source"] for t in H.TASKS}, "client_pub_acc": client_pub_acc,
       "private": {"pick_U": P.get("pick_U"), "s_star_U": P.get("s_star_U"), "final_U": P.get("final_U"), "ours": OURS, "pick": PRIV, "R_pick": R_priv, "R_ours": ROOM["R_ours"], "best12": ROOM["best12"], "R_best12": ROOM["R_pub"][ROOM["best12"]]},
       "zeroup": {"union_pick": u, "s_star": s_star, "ours_zu": OURS_ZU, "pick": ZU, "R_pick": R_zu, "same_pick": ZU == PRIV, "same_union": OURS_ZU == OURS, "d_R": R_zu - R_priv},
       "S_pub": S_pub, "S_Lpub": S_Lpub, "S_priv_common": {m: S_priv[m] for m in common}, "spearman_pub_vs_priv": rho, "evaluated": evaluated, "smoke": SMOKE, "seconds": time.time() - t0}
H.jdump(ZJ, out); H.log(f"zero-upload room done ({time.time() - t0:.0f}s)")
