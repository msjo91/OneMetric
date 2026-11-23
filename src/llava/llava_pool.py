"""The LLaVA-1.5-7B pools of MM-MergeBench clients (seven clients of rank 16, seven projections in each of 32 blocks) and their compositions V1 to V6: calibration prompts, weight-space baseline merges, Grams, one-shot RegMean and RegMean++, pseudo-answers, the score (word-level F1 of 48-token continuations), and evaluation through mm_eval.
Env VLM_SUB = V1 ... V6 for a composition (unset: the public seven-client pool). Stages: calib | build | grams | regmeanpp | pseudo | score | eval | evalpick | analysis | link. The benchmark checkout (adapters, data, images) is expected under external/mm_mergebench."""
from src.paths import ROOT as _OM_ROOT
import argparse, importlib.util, json, os, random, re, subprocess, sys, time
from collections import Counter
import numpy as np
HOME = _OM_ROOT
from src.llava import mm_pool as LP
from src.llava import mm_merges as M
PREREG = ""
SMOKE = os.environ.get("LLMROOM_SMOKE") == "1"
PY = sys.executable
MM = os.path.join(HOME, "external", "mm_mergebench")
PARENT = os.path.join(HOME, "results", "vlm_smoke" if SMOKE else "vlm")
# VLM_SUB=<name> runs a client composition of the full pool under PARENT/sub/<name>; calibration rows,
# pseudo-answers, expert and base evaluations are per client and are linked from PARENT (stage link). Compositions fixed from results/vlm/sub_predictor.json:
# V1 = 4-client argmax F1, V2 = 5-client argmax F2 (ties -> higher F1), V3 = 5-client argmin F1. Smoke: S1 = two of the smoke pool's three clients.
SUB = os.environ.get("VLM_SUB") or None
SUBS = {"S1": ["OCRVQA", "VizWiz"]} if SMOKE else {"V1": ["IconQA", "OCRVQA", "REC", "ScienceQA"], "V2": ["IconQA", "OCRVQA", "ScienceQA", "VQAv2", "VizWiz"], "V3": ["REC", "ScienceQA", "VQAv2", "VizWiz", "flickr30k"],
        # V4 = 6-client argmin F1, V5 = 6-client argmax F1, V6 = 4-client argmin F1 (results/vlm/sub_predictor_k46.json)
        "V4": ["OCRVQA", "REC", "ScienceQA", "VQAv2", "VizWiz", "flickr30k"], "V5": ["IconQA", "OCRVQA", "REC", "ScienceQA", "VQAv2", "flickr30k"], "V6": ["REC", "VQAv2", "VizWiz", "flickr30k"]}
if SUB: assert SUB in SUBS, SUB; PREREG = "" if SUB in ("V4", "V5", "V6") else ""
GATE = "57th" if SUB else "56th"
ROOT = os.path.join(PARENT, "sub", SUB) if SUB else PARENT
MERGES, POOL, SEL, CAL = [os.path.join(ROOT, d) for d in ("merges", "pool500", "select", "calib")]
for _d in (MERGES, POOL, SEL, CAL): os.makedirs(_d, exist_ok=True)
ALL_CLIENTS = ["ScienceQA", "VQAv2", "REC", "OCRVQA", "VizWiz", "flickr30k", "IconQA"]  # ImageNet excluded: no images on disk -> no client data, no scorable task
CLIENTS = SUBS[SUB] if SUB else (["OCRVQA", "VizWiz", "IconQA"] if SMOKE else ALL_CLIENTS)
def task_of(c): return {"flickr30k": "Flickr30k"}.get(c, c)
SEEN = [task_of(c) for c in CLIENTS]; UNSEEN = [] if (SMOKE or SUB) else ["AOKVQA", "ImageNet-R", "Screen2words", "TabMWP"]; EVAL_TASKS = SEEN + UNSEEN
CONF_N = 8 if SMOKE else 2000; CONF_SEEDS = [42, 7] if SMOKE else [42, 7, 13]  # full-pool confirmation draws (calibration rows excluded)
SIZES = {"ScienceQA": 12726, "VQAv2": 82783, "REC": 55885, "OCRVQA": 165348, "VizWiz": 117155, "flickr30k": 127130, "IconQA": 29859}  # train.json rows
K = len(CLIENTS); R_CLIENT = 16; RANK = R_CLIENT * K
N_CALIB = 8 if SMOKE else 200; EVAL_N = 8 if SMOKE else 500; SEED = 42; N_LAYERS = 32; PP_LAYERS = 2 if SMOKE else 32; GROUP = 4; BS = 4; LAM = 0.9; MAX_NEW = 48
F2_THRESH = 5.0
MODS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
GRAM_OF = {"self_attn.q_proj": "qkv", "self_attn.k_proj": "qkv", "self_attn.v_proj": "qkv", "self_attn.o_proj": "o", "mlp.gate_proj": "gu", "mlp.up_proj": "gu", "mlp.down_proj": "d"}
HOOK_MOD = {"qkv": "self_attn.q_proj", "o": "self_attn.o_proj", "gu": "mlp.gate_proj", "d": "mlp.down_proj"}  # one Gram per shared input
CELL_RE = re.compile(r"language_model\.layers\.(\d+)\.((?:self_attn|mlp)\.(?:[qkvo]|gate|up|down)_proj)$"); LAYER_RE = re.compile(r"language_model\.layers\.(\d+)$")
MIXA = [(.25, "25"), (.5, "50"), (.625, "625"), (.75, "75"), (.875, "875")]
BASE = "regmeanpp_long"; TEMPLATE = "concat"  # rank-112 adapter whose weights are overwritten in place
PUB_FREE = ["concat", "ties64", "dare64", "tsvm", "dcmerge", "knots_ties", "isocts_white", "fedavg_u", "fedavg_w", "fedit"]; PUB_TUNED = ["trim_g50", "dcmerge_g50"]
PUB = ["concat", "fedit"] if SMOKE else (PUB_FREE if SUB else PUB_FREE + PUB_TUNED)  # compositions: label-free published rows only

def log(*a): print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)
def mdir(m): return os.path.join(MERGES, m)
def done(m): return os.path.exists(os.path.join(mdir(m), "adapter_model.safetensors"))
def metrics(name):
    p = os.path.join(POOL, f"{name}__metrics.json"); return json.load(open(p)) if os.path.exists(p) else None
def target():
    p = os.path.join(ROOT, "predictor.json"); return json.load(open(p))["target"] if os.path.exists(p) else "median"
def end(): return "isoc_white" if target() == "median" else "isoc_wmin"
def pre(): return "os" if target() == "median" else "om"
def grid(): return [end()] + [pre() + t for _, t in MIXA] + ["regmean_long"]
def a_of(m): return 0.0 if m in ("isoc_white", "isoc_wmin") else (1.0 if m == "regmean_long" else dict((pre() + t, a) for a, t in MIXA).get(m, 1.0))
def load_pool(): return LP.load_llava_pool(MM, CLIENTS)  # factors[c][(layer, module)] = {A (16, in), B (out, 16)} (alpha folded), projs[c]
def lmm_eval():
    from src.llava import mm_eval; return mm_eval
def embed(fs, rank):
    A = np.zeros((rank, fs["A"].shape[1])); B = np.zeros((fs["B"].shape[0], rank)); r = fs["A"].shape[0]; A[:r] = fs["A"]; B[:, :r] = fs["B"]; return {"A": A, "B": B}
def pad(f): return {c: (v if v["A"].shape[0] == RANK else embed(v, RANK)) for c, v in f.items()}
def write(name, f, desc, projector):
    LP.write_llava_merged(pad(f), mdir(name), RANK, projector=projector, description=f"{desc} | pool=mm_mergebench clients={CLIENTS} rank={RANK} prereg={PREREG}"); log(f"wrote {name}")
def read(name): return LP.read_llava_merged(mdir(name))[0]
# rule P scorer, verbatim from select_fr.py (word-F1 of the continuation vs the pseudo-answer)
def norm_tokens(s): return [w.strip(".:;") for w in s.lower().replace(",", " ").split() if w.strip(".:;")]
def f1(a, b):
    ca, cb = Counter(norm_tokens(a)), Counter(norm_tokens(b))
    if not ca and not cb: return 1.0
    common = sum((ca & cb).values())
    if common == 0: return 0.0
    p = common / sum(ca.values()); r = common / sum(cb.values()); return 2 * p * r / (p + r)


def features(pool):
    cells = list(M._cells(pool)); norms = [M.client_norms(pool, c) for c in cells]; N = np.array([[norms[i][t] for i in range(len(cells))] for t in sorted(pool)])  # one norm pass per cell
    F1 = float(np.median(N.max(0) / np.median(N, 0))); l0 = [i for i, c in enumerate(cells) if c[0] == 0]; F2 = float(np.max(np.median(N[:, l0], 0)))
    return {"F1": F1, "F2": F2, "target": "median" if F2 < F2_THRESH else "min", "pred_room": bool(F1 >= 10 and F2 < F2_THRESH), "pred_collapse": bool(F2 >= F2_THRESH), "clients": sorted(pool)}
def whiten_min(pool):
    out = {t: {} for t in pool}
    for cell in M._cells(pool):
        tgt = float(np.min(np.array(list(M.client_norms(pool, cell).values()))))
        for t in pool:
            B, A = pool[t][cell]["B"], pool[t][cell]["A"]; r = A.shape[0]; U, S, V = M._lowrank_svd(B, A); k = len(S)
            A_out = np.zeros((r, A.shape[1])); B_out = np.zeros((B.shape[0], r))
            if k: A_out[:k] = V.T; B_out[:, :k] = U * (tgt / np.sqrt(k))
            out[t][cell] = {"A": A_out, "B": B_out}
    return out

# ---------------- calibration prompts: unused test rows, disjoint from the n=500 eval draw ----------------
def calib_rows(c, LE=None):
    p = os.path.join(CAL, f"{c}.json")
    if os.path.exists(p): return json.load(open(p))
    LE = LE or lmm_eval(); t = task_of(c)
    full = LE.load_task(t, MM, n=None); drawn = LE.load_task(t, MM, n=EVAL_N, seed=SEED); used = {r["_idx"] for r in drawn}
    rest = [r for r in full if r["_idx"] not in used and (r["image"] is None or os.path.exists(LE.image_path(MM, r["image"])))]
    pick = random.Random(0).sample(rest, min(N_CALIB, len(rest)))
    rows = [{"idx": r["_idx"], "id": r["id"], "image": r["image"], "prompt": r["prompt"]} for r in pick]
    json.dump(rows, open(p, "w")); log(f"calib {c}: {len(rows)} rows drawn (Random(0)) from {len(rest)} unused rows ({len(full)} total, {len(used)} in the eval draw)"); return rows
def stage_link():
    """Composition trees reuse the per-client artefacts of the full pool: calibration rows, pseudo-answers, expert and base evaluations (all client-only, composition-independent)."""
    assert SUB, "link is for compositions"
    pairs = [(os.path.join(PARENT, "calib", f"{c}.json"), os.path.join(CAL, f"{c}.json")) for c in CLIENTS] + [(os.path.join(PARENT, "select", f"pseudo_{c}.json"), os.path.join(SEL, f"pseudo_{c}.json")) for c in CLIENTS]
    pairs += [(os.path.join(PARENT, "pool500", f"client_{c}__metrics.json"), os.path.join(POOL, f"client_{c}__metrics.json")) for c in CLIENTS] + [(os.path.join(PARENT, "pool500", "base__metrics.json"), os.path.join(POOL, "base__metrics.json"))]
    for src, dst in pairs:
        if os.path.lexists(dst): continue
        if os.path.exists(src): os.symlink(src, dst); log(f"linked {os.path.relpath(dst, HOME)} -> {os.path.relpath(src, HOME)}")
        else: log(f"link: source missing (yet) {os.path.relpath(src, HOME)}")
def stage_calib():
    if SUB: stage_link()
    LE = lmm_eval()
    for c in CLIENTS: calib_rows(c, LE)  # identical draw whether linked or recomputed (Random(0) on the same unused rows)

# ---------------- CPU build: predictor, published baselines, one-shot FL rows, whitened endpoint ----------------
def stage_build():
    pool, projs = load_pool(); proj = LP.mean_projector(projs); cells = list(M._cells(pool)); t00 = time.time()
    feat = features(pool); json.dump(feat, open(os.path.join(ROOT, "predictor.json"), "w"), indent=1); log(f"features {feat}")
    names = PUB + [end()]
    for r in names:
        if done(r): continue
        t0 = time.time()
        if r == "fedavg_u": f = {cell: {"A": np.vstack([pool[c][cell]["A"] for c in CLIENTS]), "B": np.hstack([pool[c][cell]["B"] / K for c in CLIENTS])} for cell in cells}; desc = f"rule=fedavg_u uniform average of client updates (1/K sum B_k A_k), K={K}"
        elif r == "fedavg_w":
            tot = sum(SIZES[c] for c in CLIENTS); f = {cell: {"A": np.vstack([pool[c][cell]["A"] for c in CLIENTS]), "B": np.hstack([pool[c][cell]["B"] * (SIZES[c] / tot) for c in CLIENTS])} for cell in cells}
            desc = f"rule=fedavg_w size-weighted average, weights {dict((c, round(SIZES[c] / tot, 4)) for c in CLIENTS)}"
        elif r == "fedit": f = {cell: {"A": np.mean([pool[c][cell]["A"] for c in CLIENTS], axis=0), "B": np.mean([pool[c][cell]["B"] for c in CLIENTS], axis=0)} for cell in cells}; desc = f"rule=fedit separate averaging of A and B, rank {R_CLIENT}, K={K}"
        elif r == "isoc_wmin": f = M.merge_isoc(whiten_min(pool), rank=RANK); desc = "rule=isoc_wmin (whiten to MIN client norm, Iso-C)"
        else:
            fn, kw = M.RULES[r]; kw = dict(kw)
            if "rank" in kw: kw["rank"] = RANK
            f = fn(pool, **kw); desc = f"rule={r} kwargs={kw} (mm_merges.py)"
        for cell, fs in f.items(): assert np.isfinite(fs["A"]).all() and np.isfinite(fs["B"]).all(), f"{r}: non-finite at {cell}"
        write(r, f, desc, proj); log(f"  {r} built in {time.time() - t0:.0f}s")
    nrm = {n: {str(cell): float(np.linalg.norm(fs["B"] @ fs["A"])) for cell, fs in read(n).items() if cell[0] == 0} for n in (end(), "concat")}
    nrm["median_client"] = {str(cell): float(np.median(list(M.client_norms(pool, cell).values()))) for cell in cells if cell[0] == 0}
    json.dump(nrm, open(os.path.join(ROOT, "endpoint_norms.json"), "w"), indent=1); log(f"build done ({time.time() - t00:.0f}s)")

# ---------------- GPU model helpers ----------------
def gpu_model():
    LE = lmm_eval(); assert done(TEMPLATE), f"template {TEMPLATE} missing (run build)"
    model, processor = LE.load_model(f"merged:{mdir(TEMPLATE)}", MM); return model, processor, LE
def cell_modules(model):
    out = {}
    for name, mod in model.named_modules():
        m = CELL_RE.search(name)
        if m and hasattr(mod, "lora_A"): out[(int(m.group(1)), m.group(2))] = mod
    assert len(out) == N_LAYERS * len(MODS), f"found {len(out)} LoRA cells, expected {N_LAYERS * len(MODS)}"
    return out
def layer_modules(model):
    out = {int(LAYER_RE.search(n).group(1)): mod for n, mod in model.named_modules() if LAYER_RE.search(n)}
    assert len(out) == N_LAYERS; return out
def set_factors(model, cm, factors):
    import torch
    with torch.no_grad():
        for cell, fs in factors.items():
            fs = fs if fs["A"].shape[0] == RANK else embed(fs, RANK); wa = cm[cell].lora_A["default"].weight; wb = cm[cell].lora_B["default"].weight
            assert tuple(wa.shape) == fs["A"].shape and tuple(wb.shape) == fs["B"].shape, f"{cell}: template {tuple(wa.shape)}/{tuple(wb.shape)} vs {fs['A'].shape}/{fs['B'].shape}"
            wa.copy_(torch.as_tensor(np.ascontiguousarray(fs["A"]), dtype=wa.dtype)); wb.copy_(torch.as_tensor(np.ascontiguousarray(fs["B"]), dtype=wb.dtype))
class StopForward(Exception): pass
_warned = {"mask": False}
def collect_grams(model, processor, LE, cm, lm, rows, layers, pad_img, stop_after=None):
    """fp32 uncentred second moments of the inputs of q_proj (shared by q/k/v), o_proj, gate_proj (shared with up), down_proj at `layers`; pad positions excluded;
    image tokens included. Returns {(layer, kind): float64 (d, d)} normalised by the row count."""
    import torch
    G, n, cur, hs = {}, {}, {"mask": None}, []
    for l in layers:
        for kind, modname in HOOK_MOD.items():
            def hook(mod, inp, key=(l, kind)):
                x = inp[0].detach(); mask = cur["mask"]
                if mask is not None and x.dim() == 3 and tuple(mask.shape) == tuple(x.shape[:2]): x = x[mask.bool()]
                elif not _warned["mask"]: _warned["mask"] = True; log(f"WARNING: mask shape {None if mask is None else tuple(mask.shape)} vs x {tuple(x.shape)}: no pad exclusion")
                x = x.reshape(-1, x.shape[-1]).float()
                if key not in G: G[key] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=x.device); n[key] = 0
                G[key].addmm_(x.T, x); n[key] += x.shape[0]
            hs.append(cm[(l, modname)].register_forward_pre_hook(hook))
    if stop_after is not None:
        def stop(mod, inp, out): raise StopForward()
        hs.append(lm[stop_after].register_forward_hook(stop))
    image_mean = processor.image_processor.image_mean; model.eval()
    with torch.no_grad():
        for with_image in (True, False):
            idx = [i for i, r in enumerate(rows) if (r["image"] is not None) == with_image]
            for s in range(0, len(idx), BS):
                chunk = idx[s:s + BS]; prompts = [LE.build_prompt(rows[i]["prompt"], with_image) for i in chunk]
                images = [LE.load_image(LE.image_path(MM, rows[i]["image"]), pad_img, image_mean) for i in chunk] if with_image else None
                inputs = processor(images=images, text=prompts, padding=True, return_tensors="pt"); inputs = {k: v.to(model.device) for k, v in inputs.items()}
                if "pixel_values" in inputs: inputs["pixel_values"] = inputs["pixel_values"].to(LE._model_dtype(model))
                cur["mask"] = inputs.get("attention_mask")
                try: model(**inputs, use_cache=False)
                except StopForward: pass
    for h in hs: h.remove()
    out = {k: (G[k] / max(n[k], 1)).double().cpu().numpy() for k in G}; del G; torch.cuda.empty_cache(); return out
def shrink(G, lam=LAM): return lam * G + (1 - lam) * np.diag(np.diag(G))
def inv_ridge(den): return np.linalg.inv(den + 1e-6 * np.trace(den) / den.shape[0] * np.eye(den.shape[0]))
def regmean_cells(pool, grams, layer):
    """grams[c][(layer, kind)] raw Grams -> RegMean factors (rank K*16) of the 7 cells of `layer`: B = [B_c], A = [A_c G~_c] (sum G~)^-1, G~ = shrink(G)."""
    out = {}
    for kind, hook in HOOK_MOD.items():
        Gs = {c: shrink(grams[c][(layer, kind)]) for c in CLIENTS}; inv = inv_ridge(sum(Gs.values()))
        for mod in [m for m in MODS if GRAM_OF[m] == kind]:
            out[(layer, mod)] = {"A": np.vstack([pool[c][(layer, mod)]["A"].astype(np.float64) @ Gs[c] for c in CLIENTS]) @ inv, "B": np.hstack([pool[c][(layer, mod)]["B"].astype(np.float64) for c in CLIENTS])}
    return out
def mix(pp, iso, a):
    f = {}; kept = []
    for c in sorted(pp):
        Bc = np.hstack([a * pp[c]["B"], (1 - a) * iso[c]["B"]]); Ac = np.vstack([pp[c]["A"], iso[c]["A"]])
        Qb, Rb = np.linalg.qr(Bc); Qa, Ra = np.linalg.qr(Ac.T); U, S, Vt = np.linalg.svd(Rb @ Ra.T); k = RANK
        kept.append(float((S[:k] ** 2).sum() / max((S ** 2).sum(), 1e-30))); f[c] = {"B": Qb @ (U[:, :k] * np.sqrt(S[:k])), "A": (np.sqrt(S[:k])[:, None] * Vt[:k]) @ Qa.T}
    return f, kept
FACT = os.path.join(ROOT, "regmean_long_factors.npz")
def plain_factors():
    z = dict(np.load(FACT)); return {(l, mod): {"A": z[f"{l}.{mod}.A"].astype(np.float64), "B": z[f"{l}.{mod}.B"].astype(np.float64)} for l in range(N_LAYERS) for mod in MODS}

# ---------------- GPU stages ----------------
def stage_grams():
    need = [m for m in ["regmean_long"] + [pre() + t for _, t in MIXA] if not done(m)]
    if not need and os.path.exists(FACT): log("grams: all built"); return
    pool, projs = load_pool(); proj = LP.mean_projector(projs); model, processor, LE = gpu_model(); cm, lm = cell_modules(model), layer_modules(model)
    sumG, AG, t00 = {}, {c: {} for c in CLIENTS}, time.time()
    for c in CLIENTS:
        set_factors(model, cm, pool[c]); LP.apply_projector(model, projs[c]); rows = calib_rows(c, LE); pad_img = LE.TASKS[task_of(c)]["pad"]; t0 = time.time(); tr0 = None
        for g0 in range(0, N_LAYERS, GROUP):
            layers = list(range(g0, min(g0 + GROUP, N_LAYERS))); g = collect_grams(model, processor, LE, cm, lm, rows, layers, pad_img, stop_after=layers[-1])
            for (l, kind), G in g.items():
                Gs = shrink(G); sumG[(l, kind)] = sumG[(l, kind)] + Gs if (l, kind) in sumG else Gs
                if l == 0 and kind == "qkv": tr0 = float(np.trace(G))
                for mod in [m for m in MODS if GRAM_OF[m] == kind]: AG[c][(l, mod)] = pool[c][(l, mod)]["A"].astype(np.float64) @ Gs
            del g
        log(f"grams {c}: {len(rows)} prompts, {time.time() - t0:.0f}s, tr(G0 qkv) {tr0:.4g}")
    f = {}
    for l in range(N_LAYERS):
        for kind in HOOK_MOD:
            inv = inv_ridge(sumG[(l, kind)])
            for mod in [m for m in MODS if GRAM_OF[m] == kind]: f[(l, mod)] = {"A": np.vstack([AG[c][(l, mod)] for c in CLIENTS]) @ inv, "B": np.hstack([pool[c][(l, mod)]["B"].astype(np.float64) for c in CLIENTS])}
    del sumG
    np.savez(FACT, **{f"{l}.{mod}.{k}": f[(l, mod)][k].astype(np.float32) for l in range(N_LAYERS) for mod in MODS for k in ("A", "B")})
    if not done("regmean_long"): write("regmean_long", f, f"rule=regmean plain one-shot: one Gram per client on {N_CALIB} unlabelled calibration prompts (own adapter + own projector, image tokens included), lam={LAM}", proj)
    iso = read(end()); energy = {}
    for a, t in MIXA:
        name = pre() + t
        if done(name): continue
        mf, kept = mix(f, iso, a); energy[name] = {"mean": float(np.mean(kept)), "min": float(np.min(kept))}
        write(name, mf, f"rule=mix a={a} * regmean_long + (1-a) * {end()}, SVD -> rank {RANK}, energy kept mean={np.mean(kept):.4f}", proj)
    json.dump(energy, open(os.path.join(ROOT, "mix_energy.json"), "w"), indent=1); log(f"grams stage done ({time.time() - t00:.0f}s)")
def stage_regmeanpp():
    if done(BASE): log(f"{BASE} exists"); return
    pool, projs = load_pool(); proj = LP.mean_projector(projs); plain = plain_factors(); model, processor, LE = gpu_model(); cm, lm = cell_modules(model), layer_modules(model)
    rows = {c: calib_rows(c, LE) for c in CLIENTS}; merged = {}; t0 = time.time()
    for l in range(PP_LAYERS):
        gs = {}
        for c in CLIENTS:
            state = dict(merged); state.update({cell: pool[c][cell] for cell in pool[c] if cell not in merged})
            set_factors(model, cm, state); LP.apply_projector(model, projs[c])
            gs[c] = collect_grams(model, processor, LE, cm, lm, rows[c], [l], LE.TASKS[task_of(c)]["pad"], stop_after=l)
        merged.update(regmean_cells(pool, gs, l)); log(f"  regmeanpp layer {l} ({time.time() - t0:.0f}s)")
    for cell in plain:
        if cell not in merged: merged[cell] = plain[cell]  # smoke only: layers beyond PP_LAYERS take the plain result
    write(BASE, merged, f"rule=regmeanpp layer-sequential ({PP_LAYERS} layers = {PP_LAYERS} dependent Gram rounds; client passes with own projector) lam={LAM}", proj)
def stage_pseudo():
    todo = [c for c in CLIENTS if not os.path.exists(os.path.join(SEL, f"pseudo_{c}.json"))]
    if not todo: log("pseudo: all exist"); return
    pool, projs = load_pool(); model, processor, LE = gpu_model(); cm = cell_modules(model)
    for c in todo:
        set_factors(model, cm, pool[c]); LP.apply_projector(model, projs[c]); rows = calib_rows(c, LE); t0 = time.time()
        outs = LE.generate(model, processor, rows, {"pad": LE.TASKS[task_of(c)]["pad"], "max_new_tokens": MAX_NEW}, MM, BS)
        keep = [dict(r, text=o) for r, o in zip(rows, outs) if o.strip()]
        json.dump({"client": c, "n": len(keep), "rows": keep, "source": f"own adapter + own projector, {MAX_NEW} tokens greedy, calibration draw 0"}, open(os.path.join(SEL, f"pseudo_{c}.json"), "w"))
        log(f"pseudo {c}: {len(keep)}/{len(rows)} rows in {time.time() - t0:.0f}s, e.g. {keep[0]['text'][:60] if keep else ''!r}")
def cands_of(cset):
    if cset == "grid": return grid()
    if cset == "pub": return PUB
    if cset == "pp": return [BASE]
    raise SystemExit(cset)
def stage_score(cset):
    todo = [m for m in cands_of(cset) if not os.path.exists(os.path.join(SEL, f"fr_{m}.json"))]
    if not todo: log(f"score {cset}: all exist"); return
    _, projs = load_pool(); model, processor, LE = gpu_model(); cm = cell_modules(model); LP.apply_projector(model, LP.mean_projector(projs))
    pseudo = {c: json.load(open(os.path.join(SEL, f"pseudo_{c}.json")))["rows"] for c in CLIENTS}
    for m in todo:
        assert done(m), f"candidate {m} not built"; set_factors(model, cm, read(m)); res = {}; t0 = time.time()
        for c in CLIENTS:
            rows = pseudo[c]; outs = LE.generate(model, processor, rows, {"pad": LE.TASKS[task_of(c)]["pad"], "max_new_tokens": MAX_NEW}, MM, BS)
            pf = [f1(o, r["text"]) for o, r in zip(outs, rows)]; ex = [float(norm_tokens(o) == norm_tokens(r["text"])) for o, r in zip(outs, rows)]
            res[c] = {"pf1": float(np.mean(pf)), "exact": float(np.mean(ex)), "n": len(rows), "outs": outs, "refs": [r["text"] for r in rows]}
        json.dump(res, open(os.path.join(SEL, f"fr_{m}.json"), "w")); log(f"scored {m} ({cset}): s_P {np.mean([res[c]['pf1'] for c in CLIENTS]):.4f} in {time.time() - t0:.0f}s")
def stage_eval(spec):
    if spec.startswith("merged:"):
        name = spec.split(":", 1)[1]; assert done(name), f"{name} not built"; spec = f"merged:{mdir(name)}"
    cmd = [PY, "-m", "src.llava.mm_eval", "--model", spec, "--tasks", ",".join(EVAL_TASKS), "--n", str(EVAL_N), "--seed", str(SEED), "--out", POOL, "--root", MM, "--batch_size", str(BS)]
    log(" ".join(cmd)); rc = subprocess.call(cmd, cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # the repository folder, so that `-m src...` resolves
    if rc != 0: raise SystemExit(f"eval {spec} rc={rc}")
def scores(cands):
    s = {}
    for m in cands:
        fp = os.path.join(SEL, f"fr_{m}.json")
        if os.path.exists(fp): fr = json.load(open(fp)); s[m] = float(np.mean([fr[c]["pf1"] for c in CLIENTS]))
    return s
def pick(cands, s): return max(cands, key=lambda m: (round(s[m], 12), -a_of(m)))
def stage_evalpick():
    s = scores(grid()); missing = [m for m in grid() if m not in s]; assert not missing, f"grid scores missing: {missing}"
    p = pick(grid(), s); json.dump({"pick": p, "a": a_of(p), "scores": s}, open(os.path.join(ROOT, "pick.json"), "w"), indent=1); log(f"pick {p} (a={a_of(p)}) scores {s}")
    stage_eval(f"merged:{p}")

# ---------------- full-pool confirmation at n=CONF_N, calibration rows excluded ----------------
def calib_idx(t):
    c = [c for c in CLIENTS if task_of(c) == t]; return {r["idx"] for r in calib_rows(c[0])} if c else set()
def conf_dir(seed): return os.path.join(ROOT, f"pool{CONF_N}_s{seed}")
def resolve_spec(spec):
    if spec == "pick": spec = "merged:" + json.load(open(os.path.join(ROOT, "pick.json")))["pick"]
    if spec.startswith("merged:"):
        name = spec.split(":", 1)[1]; assert done(name), f"{name} not built"; return f"merged:{mdir(name)}"
    return spec
def stage_evalconf(spec, seed):
    """lmm_eval.evaluate re-implemented with one change: the n=CONF_N draw is Random(seed).sample over the test rows minus this task's 200 calibration rows
    (which fed the Grams and the pseudo-answers). Experts (client:<c>) are evaluated on their own task only; merges on every seen task."""
    # stage 2a: the V pools confirm too; their experts are read from the FULL tree (same rows: calibration rows are per task)
    LE = lmm_eval(); spec = resolve_spec(spec); name = LE.model_name(spec); out_dir = conf_dir(seed); os.makedirs(out_dir, exist_ok=True)
    tasks = [task_of(spec.split(":", 1)[1])] if spec.startswith("client:") else SEEN
    stamp = LE.adapter_stamp(spec, MM); results, todo = LE._pending(name, tasks, out_dir, CONF_N, False, stamp, seed=seed)
    if not todo: log(f"evalconf {name} s{seed}: nothing to do"); return
    model, processor = LE.load_model(spec, MM)
    for t in todo:
        t0 = time.time(); cfg = LE.TASKS[t]; ex = calib_idx(t); rows = [r for r in LE.load_task(t, MM, n=None) if r["_idx"] not in ex]
        rows = random.Random(seed).sample(rows, CONF_N) if CONF_N < len(rows) else rows; rows, skipped = LE.drop_missing_images(rows, MM)
        outs = LE.generate(model, processor, rows, cfg, MM, BS)
        with open(os.path.join(out_dir, f"{name}__{t}.jsonl"), "w") as f:
            for r, o in zip(rows, outs): f.write(json.dumps({"id": r["id"], "idx": r["_idx"], "target": r["target"], "output": o, "has_image": r["image"] is not None}) + "\n")
        rec = {"model": name, "task": t, "metric": cfg["metric"], "n_requested": CONF_N, "n_skipped": skipped, "seed": seed, "excluded_calib": len(ex), "seconds": round(time.time() - t0, 1), "adapter_mtime": stamp, "gen_cfg": LE.gen_cfg(t, "fp16")}
        sc = LE.score_task(t, outs, [r["target"] for r in rows]); sc.pop("correct", None); rec.update(sc); results[t] = rec; LE._write_metrics(out_dir, name, results, t)
        log(f"evalconf {name} s{seed} {t}: acc={rec['acc']} n={rec['n']} excl {len(ex)} {rec['seconds']}s")
def stage_confanalysis():
    assert not SUB; p = json.load(open(os.path.join(ROOT, "pick.json")))["pick"]; models = [p, BASE]; out = [f"confirmation: full LLaVA pool, pick {p} vs {BASE}, n={CONF_N} seeds {CONF_SEEDS}, calibration rows excluded from the draws ({time.strftime('%Y-%m-%d %H:%M')})"]
    D, Rs = [], {m: [] for m in models}; res = {"pick": p, "seeds": {}}
    for sd in CONF_SEEDS:
        d = conf_dir(sd); home, R = {}, {}
        for c in CLIENTS:
            f = os.path.join(d, f"client_{c}__metrics.json"); a = json.load(open(f)).get(task_of(c), {}).get("acc") if os.path.exists(f) else None; home[task_of(c)] = a
        for m in models:
            f = os.path.join(d, f"merged_{m}__metrics.json"); v = json.load(open(f)) if os.path.exists(f) else {}
            acc = {t: v.get(t, {}).get("acc") for t in SEEN}; R[m] = None if any(acc[t] is None or home[t] is None for t in SEEN) else {t: acc[t] / home[t] for t in SEEN}
        out.append(f"seed {sd}: home " + " ".join(f"{t}={f3(home[t])}" for t in SEEN))
        for m in models:
            if R[m] is None: out.append(f"  R_{sd}({m}) incomplete"); continue
            r = float(np.mean(list(R[m].values()))); Rs[m].append(r); mt = min(SEEN, key=lambda t: R[m][t])
            out.append(f"  R_{sd}({m:16s}) = {r:.3f}  min {R[m][mt]:.3f} ({mt}) | " + " ".join(f"{t}={R[m][t]:.2f}" for t in SEEN))
        if R[p] is not None and R[BASE] is not None: D.append(float(np.mean(list(R[p].values())) - np.mean(list(R[BASE].values()))))
        res["seeds"][sd] = {"home": home, "R": {m: R[m] for m in models}}
    if len(D) == len(CONF_SEEDS):
        mD, sD = float(np.mean(D)), float(np.std(D)); call = "CONF-GAIN" if (mD >= .01 and all(x > 0 for x in D)) else ("CONF-LOSS" if mD <= -.01 else "CONF-NULL")
        out.append(f"D_s: " + " ".join(f"s{sd}={x:+.3f}" for sd, x in zip(CONF_SEEDS, D)) + f"  mean {mD:+.3f} std {sD:.3f} positive {sum(x > 0 for x in D)}/{len(D)} | R(pick) {np.mean(Rs[p]):.3f} R(base) {np.mean(Rs[BASE]):.3f} | call {call}")
        res.update({"D": D, "mean_D": mD, "std_D": sD, "R_pick": float(np.mean(Rs[p])), "R_base": float(np.mean(Rs[BASE])), "call": call})
    else: out.append("INCOMPLETE confirmation"); res["call"] = "INCOMPLETE"
    txt = "\n".join(out); open(os.path.join(ROOT, "analysis_vlmconf.txt"), "w").write(txt + "\n"); json.dump(res, open(os.path.join(ROOT, "analysis_vlmconf.json"), "w"), indent=1); print(txt)
def stage_subsummary():
    """aggregate over the compositions: COMP-SAFE / COMP-LOSS, PRED-OK / PRED-MISS."""
    assert not SUB; out = [f"compositions summary ({time.strftime('%Y-%m-%d %H:%M')})"]; rows = {}
    for name in SUBS:
        f = os.path.join(PARENT, "sub", name, "analysis_vlm.json")
        if not os.path.exists(f): out.append(f"{name}: no analysis"); continue
        a = json.load(open(f)); c = a["calls"]; rows[name] = a
        out.append(f"{name} {'+'.join(SUBS[name])}: F1 {a['predictor']['F1']:.2f} F2 {a['predictor']['F2']:.2f} target {a['predictor']['target']} | " + (f"pick {a['pick']} R {a['R'][a['pick']]:.3f} vs regmeanpp_long {a['R'][BASE]:.3f} grid max {a['R'][a['grid_max']]:.3f} | " if "pick" in a else "") + " ".join(f"{k}={v}" for k, v in c.items()))
    calls = {}
    if len(rows) == len(SUBS) and all("pick" in a for a in rows.values()):
        pk = [a["calls"]["pick"] for a in rows.values()]; pb = [a["calls"].get("pub") for a in rows.values()]
        calls["comp"] = "COMP-SAFE" if all(x in ("PICK-SAFE", "PICK-GAIN") for x in pk) and all(x in ("TIES-PUB", "BEATS-PUB") for x in pb) else "COMP-LOSS"
        calls["pred"] = "PRED-OK" if all((a["calls"]["room"] == "VLM-ROOM") == bool(a["predictor"]["pred_room"]) for a in rows.values()) else "PRED-MISS"
    else: calls["status"] = "INCOMPLETE"
    out.append("CALLS: " + " / ".join(f"{k}={v}" for k, v in calls.items())); txt = "\n".join(out)
    open(os.path.join(PARENT, "sub", "analysis_vlmsub.txt"), "w").write(txt + "\n"); json.dump({"calls": calls, "subs": {k: v.get("calls") for k, v in rows.items()}}, open(os.path.join(PARENT, "sub", "analysis_vlmsub.json"), "w"), indent=1); print(txt)

# ---------------- analysis ----------------
def acc_of(name, t):
    d = metrics(name); r = d.get(t) if d else None; return r.get("acc") if r else None
def R_of(name, home):
    v = {t: acc_of(name, t) for t in SEEN}
    if any(x is None for x in v.values()): return None, None
    return float(np.mean([v[t] / home[t] for t in SEEN])), {t: v[t] / home[t] for t in SEEN}
def U_of(name):
    v = [acc_of(name, t) for t in UNSEEN]; return None if (not UNSEEN or any(x is None for x in v)) else float(np.mean(v))
def rankdata(x):
    x = np.asarray(x, float); r = np.empty(len(x)); order = np.argsort(x); i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]: j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r
def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y); return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else float("nan")
def f3(x): return "  n/a" if x is None else f"{x:.3f}"
def stage_analysis():
    LE = lmm_eval(); home = {}
    for c in CLIENTS:
        a = acc_of(f"client_{c}", task_of(c)); assert a is not None, f"expert eval missing for {c}"; home[task_of(c)] = a
    feat = json.load(open(os.path.join(ROOT, "predictor.json"))); G = grid(); P_FREE = [m for m in PUB if m in PUB_FREE]; P_TUNED = [m for m in PUB if m in PUB_TUNED]
    names = G + [BASE] + P_FREE + P_TUNED + ["base"]; s = scores(G + [BASE] + PUB)
    R, Rt, U = {}, {}, {}
    for m in names: R[m], Rt[m] = R_of("base" if m == "base" else f"merged_{m}", home); U[m] = U_of("base" if m == "base" else f"merged_{m}")
    out = [f": OneMetric on the LLaVA-1.5-7B MM-MergeBench pool{' composition ' + SUB if SUB else ''} ({time.strftime('%Y-%m-%d %H:%M')})",
           f"clients {CLIENTS}; rank {RANK}; n={EVAL_N} seed {SEED}; R = mean over {len(SEEN)} seen tasks of acc/own-expert acc; U = mean raw acc over unseen {UNSEEN}",
           f"predictor: F1 {feat['F1']:.2f} F2 {feat['F2']:.3f} target {feat['target']} pred_room {feat['pred_room']} pred_collapse {feat['pred_collapse']}",
           "home (own expert, this harness): " + " ".join(f"{t}={home[t]:.3f}" for t in SEEN), ""]
    out.append(f"{'candidate':16s} {'R':>6s} {'U':>6s} {'s_P':>6s}  " + " ".join(f"{t[:9]:>9s}" for t in SEEN))
    for m in names:
        out.append(f"{m:16s} {f3(R[m]):>6s} {f3(U[m]):>6s} {f3(s.get(m)):>6s}  " + (" ".join(f"{Rt[m][t]:9.3f}" for t in SEEN) if Rt[m] else "incomplete"))
    out.append("")
    calls = {}; res = {"R": R, "U": U, "s_P": s, "home": home, "predictor": feat, "clients": CLIENTS, "rank": RANK, "grid": G}
    complete = all(R[m] is not None for m in G + [BASE]) and all(m in s for m in G)
    if not complete: calls["status"] = "INCOMPLETE"; out.append("INCOMPLETE: " + ", ".join([m for m in G + [BASE] if R[m] is None] + [f"score {m}" for m in G if m not in s]))
    else:
        p = pick(G, s); gmax = max(G, key=lambda m: R[m]); R_pp = R[BASE]; res.update({"pick": p, "a_pick": a_of(p), "grid_max": gmax})
        calls["room"] = "VLM-ROOM" if R[gmax] - R_pp >= .02 else "VLM-FLAT"
        d = R[p] - R_pp; calls["pick"] = "PICK-GAIN" if d >= .02 else ("PICK-SAFE" if d >= -.01 else "PICK-LOSS"); calls["sel"] = "SEL-OK" if R[gmax] - R[p] <= .01 else "SEL-LOSS"
        out.append(f"pick {p} (a={a_of(p)}) R {R[p]:.3f}; grid max {gmax} R {R[gmax]:.3f}; regmeanpp_long R {R_pp:.3f}; D(pick-pp) {d:+.3f}; selection loss {R[gmax] - R[p]:.3f}")
        pf = [m for m in P_FREE if R[m] is not None]
        if len(pf) == len(P_FREE):
            bp = max(pf, key=lambda m: R[m]); dp = R[p] - R[bp]; calls["pub"] = "BEATS-PUB" if dp >= .01 else ("TIES-PUB" if dp >= -.01 else "LOSES-PUB")
            out.append(f"best published label-free {bp} R {R[bp]:.3f}; D(pick-pub) {dp:+.3f}" + (f"; tuned rows " + " ".join(f"{m} {f3(R[m])}" for m in P_TUNED) if P_TUNED else ""))
            union = [m for m in G + pf + [BASE] if m in s]
            if len(union) == len(G) + len(pf) + 1:
                u = pick(union, s); calls["union"] = "SEL-ROBUST" if (u in G or R[u] >= R[p] - .01) else "SEL-FRAGILE"; out.append(f"union argmax s_P: {u} (R {R[u]:.3f}, s_P {s[u]:.3f}) -> {calls['union']}")
                pubset = pf + [end(), "regmean_long"]; pp_ = pick(pubset, s); best = max(pubset, key=lambda m: R[m])
                calls["pubpick"] = "PUB-PICK-OK" if R[pp_] >= R[best] - .02 else "PUB-PICK-MISS"
                out.append(f"published-set pick: {pp_} R {R[pp_]:.3f} vs best {best} R {R[best]:.3f} -> {calls['pubpick']}; Spearman(s_P, R) over {len(pubset)} = {spearman([s[m] for m in pubset], [R[m] for m in pubset]):+.2f}")
            else: out.append("union/pub-pick: scores incomplete")
        else: out.append("published rows incomplete: " + ", ".join(m for m in P_FREE if R[m] is None))
        if U[p] is not None and U[BASE] is not None: out.append(f"unseen mean: pick {U[p]:.3f} vs regmeanpp_long {U[BASE]:.3f} vs base {f3(U['base'])}")
    out.append("CALLS: " + " / ".join(f"{k}={v}" for k, v in calls.items())); res["calls"] = calls

    oc = os.path.join(HOME, "results", "mm_reference_eval"); chk = []
    for c in CLIENTS:
        f = os.path.join(oc, f"client_{c}__metrics.json")
        if os.path.exists(f): d = json.load(open(f)); t = task_of(c); a = d.get(t, {}).get("acc"); chk.append(f"{t} {home[t]:.3f} vs reference {f3(a)} (d {home[t] - a:+.3f})" if a is not None else f"{t} reference n/a")
    if chk: out.append("expert consistency vs the stored reference: " + "; ".join(chk)); res["expert_vs_reference"] = chk
    if not SMOKE and not SUB and os.path.exists(os.path.join(oc, "base__metrics.json")):
        ref = {}; oh = {task_of(c): json.load(open(os.path.join(oc, f"client_{c}__metrics.json")))[task_of(c)]["acc"] for c in CLIENTS}
        for m in ("concat", "isoc_white", "isocts_white", "dcmerge", "ties64"):
            f = os.path.join(oc, f"merged_{m}__metrics.json")
            if os.path.exists(f): d = json.load(open(f)); ref[m] = float(np.mean([d[t]["acc"] / oh[t] for t in SEEN]))
        out.append("8-client reference R (same 7 tasks): " + " ".join(f"{m} {v:.3f}" for m, v in ref.items())); res["reference_8clients"] = ref
    txt = "\n".join(out); open(os.path.join(ROOT, "analysis_vlm.txt"), "w").write(txt + "\n"); json.dump(res, open(os.path.join(ROOT, "analysis_vlm.json"), "w"), indent=1); print(txt)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--set", default="grid"); ap.add_argument("--model"); ap.add_argument("--seed", type=int, default=SEED); ap.add_argument("what", nargs="?")
    a = ap.parse_args()
    if a.stage == "calib": stage_calib()
    elif a.stage == "build": stage_build()
    elif a.stage == "grams": stage_grams()
    elif a.stage == "regmeanpp": stage_regmeanpp()
    elif a.stage == "pseudo": stage_pseudo()
    elif a.stage == "score": stage_score(a.set)
    elif a.stage == "eval": stage_eval(a.model)
    elif a.stage == "evalpick": stage_evalpick()
    elif a.stage == "analysis": stage_analysis()
    elif a.stage == "link": stage_link()
    elif a.stage == "evalconf": stage_evalconf(a.model, a.seed)
    elif a.stage == "confanalysis": stage_confanalysis()
    elif a.stage == "subsummary": stage_subsummary()
    elif a.stage == "jobs": print(" ".join({"clients": CLIENTS, "pub": PUB, "grid": grid(), "subs": list(SUBS), "seeds": [str(x) for x in CONF_SEEDS]}[a.what]))
    else: raise SystemExit(f"unknown stage {a.stage}")
