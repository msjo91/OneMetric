#!/usr/bin/env python
"""zero-upload rule P. --stage pub builds the public image cache (200 images per task from the test split outside the stored
2,000-image draw, else from the train split outside the 200-image calibration draw); --stage pool (env ARCH, CLIPK_SUB) runs the method's selector with server-side pseudo-labels
(each client's own adapter on its task's public images) over the union candidates, the shortening stage, eleven published merges (ACE left out) and the fifteen rescaled copies,
compares with the private pick, and evaluates the zero-upload pick on the stored test draw when needed. ZU_TASKS=<csv> limits the pub stage (smoke); ZU_SMOKE=1 limits the pool stage to the cheap merges.
Writes results/clipclose/zeroup/dc/<arch>/<sub>/zeroup.json; logs carry no retention number."""
from src.paths import ROOT as _OM_ROOT
import os, sys, json, time, random, argparse, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = _OM_ROOT
ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["pub", "pool"], required=True); args = ap.parse_args()
from src.backbones import harness_clip as H
from src.baselines import merges_ext as ops_new; from src.method import roomlib; from src.baselines import merges as ops
PUBC = os.path.join(ROOT, "results", "clipclose", "zeroup", "cache"); os.makedirs(PUBC, exist_ok=True)
def pct(s): return int(round(s * 100))
if args.stage == "pub":
    from datasets import load_dataset; from concurrent.futures import ThreadPoolExecutor
    tasks = os.environ["ZU_TASKS"].split(",") if os.environ.get("ZU_TASKS") else [t for t in H.TASKS16 if t != "fer2013"]
    srcp = os.path.join(PUBC, "src.json"); src = H.jload(srcp) if os.path.exists(srcp) else {}
    for t in tasks:
        if os.path.exists(os.path.join(PUBC, f"{t}_pub_X.npy")): continue
        t0 = time.time(); repo, cfg = H.DS[t]; ds = load_dataset(repo, cfg, split="test") if cfg else load_dataset(repo, split="test"); wds = repo.startswith("clip-benchmark/")
        labcol = "cls" if wds else "label"; imgcol = ("png" if "png" in ds.column_names else "jpg") if wds else "image"
        n = len(ds); draw = set(random.Random(0).sample(range(n), min(2000, n))); rest = [i for i in range(n) if i not in draw]; where = "test"
        if len(rest) < 200:
            tr = load_dataset(repo, cfg, split="train") if cfg else load_dataset(repo, split="train"); cal = set(random.Random(0).sample(range(len(tr)), min(200, len(tr)))); rest = [i for i in range(len(tr)) if i not in cal]; ds = tr; where = "train"
        idx = sorted(random.Random(1).sample(rest, 200)); sub = ds.select(idx)
        with ThreadPoolExecutor(16) as ex: X = np.stack(list(ex.map(H.prep, sub[imgcol], chunksize=16)))
        y = np.asarray(sub[labcol], dtype=np.int64); np.save(os.path.join(PUBC, f"{t}_pub_X.npy"), X); np.save(os.path.join(PUBC, f"{t}_pub_y.npy"), y)
        src[t] = {"source": where, "n_test_split": n, "n_outside_draw": len(rest), "idx": idx}; H.jdump(srcp, src); H.log(f"pub {t}: 200 from {where} ({len(rest)} available; {time.time() - t0:.0f}s)")
    H.log("pub cache done"); sys.exit(0)
import torch
L = roomlib.Lib(H, ROOT, "dc", ops, ops_new); ARCH, SUB = L.ARCH, L.SUB; SMOKE = os.environ.get("ZU_SMOKE") == "1"
ROOM = H.jload(os.path.join(ROOT, "results", "clipclose", "lwin", "dc", ARCH, SUB, "room.json")); OURS = ROOM["ours"]; P = H.jload(os.path.join(L.UA, "picks.json"))
PH = H.jload(os.path.join(ROOT, "results", "f25llm", "f25nm_posthoc.json")); PRIV = next((r["nm"] for r in PH["clip"]["rows"] if r["arch"] == ARCH and r["sub"] == SUB), None)
OUT = os.path.join(ROOT, "results", "clipclose", "zeroup", "dc", ARCH, SUB); os.makedirs(OUT, exist_ok=True); ZJ = os.path.join(OUT, "zeroup.json")
Xp = {t: np.load(os.path.join(PUBC, f"{t}_pub_X.npy"), mmap_mode="r") for t in H.TASKS}; yp = {t: np.load(os.path.join(PUBC, f"{t}_pub_y.npy")) for t in H.TASKS}; SRC = H.jload(os.path.join(PUBC, "src.json"))
t0 = time.time(); C = H.clip(); HD = H.heads(C); pool, D = L.pool_deltas(); deltas = [D[t] for t in H.TASKS]; W0 = {c: C.W0[c].float().cpu().numpy() for c in H.CELLS}
pub_pseudo, client_pub_acc = {}, {}
for t in H.TASKS: C.set_delta(D[t]); pub_pseudo[t] = H.predict(C, Xp[t], HD[t]); client_pub_acc[t] = float((pub_pseudo[t] == yp[t]).mean())
H.log(f"server-side pseudo-labels made for {len(H.TASKS)} clients ({time.time() - t0:.0f}s)")
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
# ---- stage 3: published merges and rescaled copies
PUB11 = ["ta", "fedavg_w", "fedit", "ties", "knots_ties", "tsvm", "dcmerge", "isoc", "isocts", "ct", "pact_isoc"]
if SMOKE: PUB11 = ["ta", "fedavg_w", "fedit", "isoc"]
built = {}
for m in PUB11:
    t1 = time.time(); built[m] = L.rebuild_pub(m, W0) if m != "isoc" else L.load("isoc"); sc(m, built[m]); H.log(f"  published {m} built + scored ({time.time() - t1:.0f}s)")
NEW = {}
if not SMOKE:
    for g in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.3): NEW[f"ct_x{pct(g)}"] = {c: g * built["ct"][c] for c in H.CELLS}
    for s in (0.8, 0.9, 1.1): NEW[f"pact_x{pct(s)}"] = {c: s * built["pact_isoc"][c] for c in H.CELLS}; NEW[f"dc_x{pct(s)}"] = {c: s * built["dcmerge"][c] for c in H.CELLS}
    for m, d in NEW.items(): sc(m, d)
    H.log(f"rescaled copies scored ({len(NEW)})")
def cls(m): return 3 if m == OURS_ZU else 1
def scl(m): return int(m.rsplit("_x", 1)[1]) if "_x" in m and m.rsplit("_x", 1)[1].isdigit() else 100
final_set = [OURS_ZU] + PUB11 + list(NEW); ZU = max(final_set, key=lambda m: (round(S_pub[m], 12), cls(m), scl(m)))
# ---- retention of the zero-upload pick on the stored 2,000-image draw
def R_stored(m):
    if m == OURS: return ROOM["R_ours"]
    if m in ROOM["R_pub"]: return ROOM["R_pub"][m]
    if m in ROOM["R"]: return ROOM["R"][m]
    return None
def delta_any(m):
    if m == OURS_ZU and m not in union: return {c: s_star * ud[c] for c in H.CELLS}
    if m in NEW: return NEW[m]
    if m in built: return built[m]
    return L.delta_of(m)
Xt = {t: H.load_x(t, "test") for t in H.TASKS}; evaluated = {}
def R_any(m):
    r = R_stored(m)
    if r is not None: return r
    if m not in evaluated:
        t1 = time.time(); C.set_delta(delta_any(m)); acc = {t: float((H.predict(C, Xt[t][0], HD[t]) == np.asarray(Xt[t][1])).mean()) for t in H.TASKS}; evaluated[m] = {"acc": acc, "R": L.R_of(acc)}; H.log(f"  evaluated one extra candidate on the stored draw ({time.time() - t1:.0f}s)")
    return evaluated[m]["R"]
R_zu = R_any(ZU); R_priv = R_any(PRIV) if PRIV else None
# ---- private-signal comparison over common candidates
S_priv = dict(ROOM["pub_sP"]); S_priv.update(ROOM["s_P"]); S_priv[OURS] = ROOM["sP_ours"]
common = [m for m in S_pub if m in S_priv]
def rank(x): o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r
rho = float(np.corrcoef(rank([S_pub[m] for m in common]), rank([S_priv[m] for m in common]))[0, 1]) if len(common) >= 3 else None
out = {"arch": ARCH, "sub": SUB, "tasks": H.TASKS, "public_source": {t: SRC[t]["source"] for t in H.TASKS}, "client_pub_acc": client_pub_acc,
       "private": {"pick_U": P["pick_U"], "s_star_U": P["s_star_U"], "final_U": P["final_U"], "ours": OURS, "pick": PRIV, "R_pick": R_priv, "R_ours": ROOM["R_ours"], "best12": ROOM["best12"], "R_best12": ROOM["R_pub"][ROOM["best12"]]},
       "zeroup": {"union_pick": u, "s_star": s_star, "ours_zu": OURS_ZU, "pick": ZU, "R_pick": R_zu, "same_pick": ZU == PRIV, "same_union": OURS_ZU == OURS, "d_R": (R_zu - R_priv) if R_priv is not None else None},
       "S_pub": S_pub, "S_Lpub": S_Lpub, "S_priv_common": {m: S_priv[m] for m in common}, "spearman_pub_vs_priv": rho, "evaluated": evaluated, "smoke": SMOKE, "seconds": time.time() - t0}
H.jdump(ZJ, out); H.log(f"zero-upload room done ({time.time() - t0:.0f}s)")
