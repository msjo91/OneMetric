#!/usr/bin/env python
"""the one-shot federated-learning baselines FedFisher (Jhunjhunwala, Wang, Joshi, AISTATS 2024;
diagonal and K-FAC) and FedLPA (Liu et al., NeurIPS 2024) on ONE pool, on the stored test draw; no merge saved.
Env FL_KIND = dc | siglip | dino plus the harness variables (dc: ARCH, CLIPK_SUB, POOL=dc16; siglip: SIG_ARCH, CLIPK_SUB; dino: DINO_ARCH, CLIPK_SUB,
DINO_ROOT=dino2, DINO_TASKS=h16).
Client statistics: each client on its own model (base + its update), on its 200 calibration images WITH their labels (the clients own them), fp32, at the
clients' training temperature (100 on CLIP and SigLIP 2, 1 on DINOv2); K-FAC 'expand' over tokens (the input factor is a token mean, the gradient factor a
token sum averaged over images).
  A_k   token-mean input second moment of every adapted layer (RegMean's Gram before its shrinkage)
  Gt_k  output-gradient second moment of log p(y|x), y drawn once per image from the client's own predictive distribution (true Fisher; FedFisher)
  Fd_k  the exact per-image diagonal Fisher of each adapted weight, same draw (FedFisher-Diag)
  Ge_k  as Gt_k with the true labels (empirical Fisher; FedLPA)
p_k = the client's data share, as FedAvg-w. Rows, on the adapted layers only, each truncated to the merged rank like every published row:
  fedfisher_diag        Adam (lr .01, betas .9/.99, eps .01), 2,000 steps from FedAvg on sum_k p_k 1/2 Fd_k (W - W_k)^2, final iterate (the paper keeps the
                        best of every 100 steps by 500 labelled server images; this server has no labels)
  fisher_diag_exact     its minimiser: Fisher merging with a floor of 1e-6 x the cell's mean Fisher (Matena and Raffel 2022)
  fedfisher_kfac        the same Adam on sum_k p_k 1/2 tr[(W - W_k)^T Gt_k (W - W_k) A_k]
  fedfisher_kfac_exact  its minimiser by conjugate gradients from FedAvg, with a proximal weight of 1e-6 x the mean curvature toward FedAvg (minimum distance)
  fedlpa                the released code's closed form on the full weights: factors + sqrt(1 - 1e-4) I (pi = 1),
                        M = (sum_k p_k G~_k)^-1 (sum_k p_k G~_k M_k A~_k) (sum_k p_k A~_k)^-1; the released 10,000 descent steps start here
  fedlpa_exact          the solution of sum_k p_k G~_k M A~_k = sum_k p_k G~_k M_k A~_k (FedLPA's least-squares target), conjugate gradients from the closed form
Writes results/flbase/<kind>/<arch>/<sub>/fl.json (resumable). Logs carry no retention number.
FL_SMOKE=1: 16 images per client, 50 steps, 20 CG iterations, 64 test images, output under FL_SCRATCH. FL_CHECK=1: FedAvg-w through this path only (code check).
FL_LOOPCHECK=1: full statistics, then FedLPA's released 10,000-step loop on two layers; prints its reach; no evaluation."""
from onemetric.paths import ROOT as _OM_ROOT
import os, sys, time, zlib, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = _OM_ROOT; os.environ.setdefault("POOL", "dc16")
KIND = os.environ["FL_KIND"]; assert KIND in ("dc", "siglip", "dino"), KIND

if KIND == "dc": from onemetric import harness_clip as H; from onemetric import clip_backend as WM
elif KIND == "siglip": from onemetric import harness_siglip as H; from onemetric import siglip_backend as WM
else: from onemetric import harness_dino as H; from onemetric import dino_backend as WM
ARCH, SUB, TASKS, CELLS, RANK = H.ARCH, H.SUBTAG, H.TASKS, H.CELLS, H.RANK; K = len(TASKS); RET = getattr(H, "RET", TASKS); log = H.log
SMOKE = os.environ.get("FL_SMOKE") == "1"; CHECK = os.environ.get("FL_CHECK") == "1"; LOOPCHECK = os.environ.get("FL_LOOPCHECK") == "1"; RELEASED = os.environ.get("FL_RELEASED") == "1"
OUTN = WM.OUT if KIND == "dino" else "out_proj"; QN = WM.QKV[0] if KIND == "dino" else "q_proj"
def gkey(c): return (c[0], "o" if c[1] == OUTN else "qkv")
def lin(model, c): return WM.linear(model, c) if KIND == "dino" else getattr(model.vision_model.encoder.layers[c[0]].self_attn, c[1])
TEMP = 1.0 if KIND == "dino" else 100.0
NCAL = int(os.environ.get("FL_NCAL", "200")); assert NCAL in (200, 300, 500) and not (NCAL != 200 and (SMOKE or CHECK or LOOPCHECK or KIND != "dc"))   # FL_NCAL=500 reads calibration set <task>_caln500 (the stored 200 plus 300 further training images, with labels) and writes to results/flbase_n500; FL_OUTROOT overrides the output root (code check). Without them nothing changes. Backup run_fedfisher_fedlpa.py.bak_g144
CALTAG = "cal" if NCAL == 200 else f"caln{NCAL}"
NIMG = 16 if SMOKE else NCAL; STEPS = 50 if SMOKE else 2000; CGMAX = 20 if SMOKE else 500; NTEST = 64 if SMOKE else None
SCR = SMOKE or CHECK or LOOPCHECK
OUT = os.path.join(os.environ["FL_SCRATCH"], KIND, ARCH, SUB) if SCR else os.path.join(os.environ.get("FL_OUTROOT") or os.path.join(ROOT, "results", "flbase" if NCAL == 200 else f"flbase_n{NCAL}"), KIND, ARCH, SUB)
os.makedirs(OUT, exist_ok=True); RJ = os.path.join(OUT, "fl_check.json" if CHECK else "fl.json")
ROWS1 = ["fedfisher_diag", "fisher_diag_exact", "fedfisher_kfac", "fedfisher_kfac_exact"]; ROWS2 = ["fedlpa", "fedlpa_exact"] + (["fedlpa_released"] if RELEASED else [])
LOOPN = 100 if SMOKE else 10000   # FedLPA's released solver: 10,000 steps of size 1.0 on the element-mean squared residual (FL_RELEASED=1 runs it on every layer)
res = H.jload(RJ) if os.path.exists(RJ) else {"kind": KIND, "arch": ARCH, "sub": SUB, "tasks": TASKS, "ret": RET, "rank": RANK, "nimg": NIMG, "steps": STEPS,
                                               "temp": TEMP, "R": {}, "acc": {}, "diag": {}, "seconds": {}}
def save(): tmp = RJ + ".tmp"; H.jdump(tmp, res); os.replace(tmp, RJ)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; dev = torch.device("cuda:0")
t0 = time.time(); pool, D = H.load_deltas(); C = H.clip(); HD = H.heads(C)
home = {t: H.jload(os.path.join(H.EVAL, f"client_{t}.json"))[t] for t in TASKS}
sizes = H.jload(os.path.join(H.CACHE, "sizes.json")); tot = float(sum(sizes[t] for t in TASKS)); P = {t: sizes[t] / tot for t in TASKS}
W0 = {c: C.W0[c].float() for c in CELLS}   # the base weights as the evaluated half-precision model holds them
Pk = torch.tensor([P[t] for t in TASKS], device=dev, dtype=torch.float32).view(K, 1, 1)
def dstack(c): return torch.stack([torch.from_numpy(np.ascontiguousarray(D[t][c])) for t in TASKS]).to(dev).float()
def wavg(c): return (Pk * dstack(c)).sum(0)
def test(name, delta):
    if name in res["R"]: return
    t1 = time.time(); C.set_delta(delta); acc = {}
    for t in TASKS:
        X, y = H.load_x(t, "test")
        if NTEST: X, y = X[:NTEST], y[:NTEST]
        acc[t] = float((H.predict(C, X, HD[t]) == np.asarray(y)).mean())
    res["acc"][name] = acc; res["R"][name] = float(np.mean([acc[t] / max(home[t], 1e-9) for t in RET])); res["seconds"][name] = time.time() - t1; save()
    log(f"  {name}: test done ({time.time() - t1:.0f}s)")
def finish(name, dense):   # dense: {cell: float32 ndarray}; tested truncated to the merged rank (the setting) and untruncated (as RegMean++ is in the tables)
    for c in CELLS: assert np.isfinite(dense[c]).all(), (name, c)
    if name not in res["R"]: tr, kept = H.truncate(dense, RANK); res["diag"].setdefault(name, {})["kept_min"] = float(min(kept)); test(name, tr); del tr
    test(name + "_dense", dense)
H.log(f"FL rows {KIND} {ARCH} {SUB}: {K} clients, {NIMG} images each{' SMOKE' if SMOKE else ''}{' CHECK' if CHECK else ''}{' LOOPCHECK' if LOOPCHECK else ''} ({time.time() - t0:.0f}s setup)")
if CHECK:   # code check: FedAvg-w through this path; the stored fedavg_w row of results/paper3/final_eval.json must be reproduced
    finish("fedavg_check", {c: wavg(c).cpu().numpy() for c in CELLS}); print("CHECK fedavg_w R", res["R"]["fedavg_check"], "dense", res["R"]["fedavg_check_dense"]); sys.exit(0)
_CS = None
def stats_model():
    global _CS
    if _CS is None:
        if KIND == "dc": _CS = WM.Clip("cuda:0"); _CS.model.float()
        else: _CS = WM.Clip("cuda:0", dtype=torch.float32)
        if KIND != "dino": del _CS.model.text_model; torch.cuda.empty_cache()   # the statistics pass reads image features only; the heads come from the evaluated model
        for q in _CS.model.parameters(): q.requires_grad_(False)
    return _CS
BSS = 4 if W0[CELLS[0]].shape[1] > 1024 else (8 if W0[CELLS[0]].shape[1] >= 1024 else 16)   # So400m (1152) at 4 to fit two lanes per GPU
def logits_of(CS, xb, Hf):
    x = torch.from_numpy(np.ascontiguousarray(xb)).to(dev).permute(0, 3, 1, 2).float().div_(255); x = ((x - CS.MEAN.float()) / CS.STD.float()).requires_grad_(True)
    if KIND == "dino": h = CS.model(pixel_values=x).last_hidden_state; f = torch.cat([h[:, 0], h[:, 1:].mean(1), torch.ones(h.shape[0], 1, device=dev)], 1)
    else: f = CS.model.get_image_features(pixel_values=x); f = f / f.norm(dim=-1, keepdim=True)
    return TEMP * (f.float() @ Hf.T)
def client_stats(t, sampled, want_A, want_F):
    """One pass over the client's calibration images. sampled: y drawn from the client's predictive distribution (else the true labels)."""
    CS = stats_model(); CS.set_delta(D[t]); Hf = HD[t].float(); X, y = H.load_x(t, CALTAG); X, y = X[:NIMG], np.asarray(y)[:NIMG]; n = len(y); assert SMOKE or n == NIMG, (t, n)
    A, cnt, G, Fd, hooks = {}, {}, {}, {}, []
    def mk(c):
        def fh(mod, inp, out):
            x = inp[0].detach().float()
            if want_A and c[1] in (QN, OUTN):
                k = gkey(c); x2 = x.reshape(-1, x.shape[-1]); A[k] = A.get(k, 0) + x2.T @ x2; cnt[k] = cnt.get(k, 0) + x2.shape[0]
            def gh(g):
                g = g.detach().float(); g2 = g.reshape(-1, g.shape[-1]); G[c] = G.get(c, 0) + g2.T @ g2
                if want_F: Pn = torch.einsum("bto,bti->boi", g, x); Fd[c] = Fd.get(c, 0) + (Pn * Pn).sum(0)
            out.register_hook(gh)
        return fh
    for c in CELLS: hooks.append(lin(CS.model, c).register_forward_hook(mk(c)))
    gen = torch.Generator(device=dev); gen.manual_seed(zlib.crc32(t.encode()) % (2 ** 31))
    try:
        for i in range(0, n, BSS):
            lp = torch.log_softmax(logits_of(CS, X[i:i + BSS], Hf), -1)
            yy = torch.multinomial(lp.detach().exp(), 1, generator=gen).squeeze(1) if sampled else torch.as_tensor(y[i:i + BSS], device=dev).long()
            (-lp.gather(1, yy[:, None]).sum()).backward()
    finally:
        for h in hooks: h.remove()
    out = {"G": {c: G[c] / n for c in CELLS}}
    if want_A: out["A"] = {k: A[k] / cnt[k] for k in A}
    if want_F: out["Fd"] = {c: Fd[c] / n for c in CELLS}
    return out
def kop(c, Gs, As, shift=0.0):
    Gk = torch.stack([Gs[t][c] for t in TASKS]).to(dev); Ak = torch.stack([As[t][gkey(c)] for t in TASKS]).to(dev)
    if shift: Gk = Gk + shift * torch.eye(Gk.shape[-1], device=dev); Ak = Ak + shift * torch.eye(Ak.shape[-1], device=dev)
    return Gk, Ak, (lambda M: (Pk * torch.matmul(torch.matmul(Gk, M), Ak)).sum(0))
def adam(gradf, W):
    W = W.clone(); m = torch.zeros_like(W); v = torch.zeros_like(W); b1, b2, lr, eps = .9, .99, .01, .01; g0 = None; g = None
    for s in range(1, STEPS + 1):
        g = gradf(W); g0 = float(g.norm()) if g0 is None else g0
        m.mul_(b1).add_(g, alpha=1 - b1); v.mul_(b2).addcmul_(g, g, value=1 - b2)
        W.addcdiv_(m / (1 - b1 ** s), (v / (1 - b2 ** s)).sqrt_().add_(eps), value=-lr)
    return W, float(gradf(W).norm()) / max(g0, 1e-30)
def cg(Aop, b, x0, tol=1e-6):
    x = x0.clone(); r = b - Aop(x); p = r.clone(); rs = (r * r).sum(); bn = float(b.norm()) or 1.0; it = 0
    for it in range(1, CGMAX + 1):
        if float(rs.sqrt()) <= tol * bn: break
        Ap = Aop(p); al = rs / (p * Ap).sum(); x.add_(al * p); r.sub_(al * Ap); rn = (r * r).sum(); p = r + (rn / rs) * p; rs = rn
    return x, it, float(rs.sqrt()) / bn
def mdiag(M): return float(torch.diagonal(M).mean())
COOR = (1 - 1e-4) ** 0.5   # FedLPA released: coor = sqrt(1 - coordinator), coordinator = 1e-4, pi = 1, added to both factors
Ast = None
need1 = any(r not in res["R"] or r + "_dense" not in res["R"] for r in ROWS1) and not LOOPCHECK
if need1:
    t1 = time.time(); Ast, Gst = {}, {}; S1 = {c: torch.zeros_like(W0[c]) for c in CELLS}; S2 = {c: torch.zeros_like(W0[c]) for c in CELLS}; sc = {}
    for t in TASKS:
        st = client_stats(t, True, True, True)
        for c in CELLS: dk = torch.from_numpy(np.ascontiguousarray(D[t][c])).to(dev).float(); S1[c] += P[t] * st["Fd"][c]; S2[c] += P[t] * st["Fd"][c] * dk
        sc[t] = {"A": float(np.mean([mdiag(v) for v in st["A"].values()])), "Gt": float(np.mean([mdiag(v) for v in st["G"].values()])), "Fd": float(np.mean([float(v.mean()) for v in st["Fd"].values()]))}
        Ast[t] = {k: v.cpu() for k, v in st["A"].items()}; Gst[t] = {c: v.cpu() for c, v in st["G"].items()}; del st; torch.cuda.empty_cache()
    res["diag"]["scale_sampled"] = sc; res["seconds"]["stats_sampled"] = time.time() - t1; save(); log(f"  statistics, sampled labels: {K} clients ({time.time() - t1:.0f}s)")
    if "fisher_diag_exact_dense" not in res["R"]:
        out = {}
        for c in CELLS: fl = 1e-6 * float(S1[c].mean()); out[c] = ((S2[c] + fl * wavg(c)) / (S1[c] + fl)).cpu().numpy()
        finish("fisher_diag_exact", out)
    if "fedfisher_diag_dense" not in res["R"]:
        t1 = time.time(); out, rg = {}, []
        for c in CELLS: W, r_ = adam(lambda W, c=c: S1[c] * W - S2[c], wavg(c)); out[c] = W.cpu().numpy(); rg.append(r_)
        res["diag"]["fedfisher_diag"] = {"grad_ratio_max": max(rg), "grad_ratio_med": float(np.median(rg)), "solve_s": time.time() - t1}; finish("fedfisher_diag", out)
    del S1, S2; torch.cuda.empty_cache()
    if any(r + "_dense" not in res["R"] for r in ("fedfisher_kfac", "fedfisher_kfac_exact")):
        t1 = time.time(); oa, oe, rg, its, rr = {}, {}, [], [], []
        for c in CELLS:
            Gk, Ak, L = kop(c, Gst, Ast); Dk = dstack(c); Z = (Pk * torch.matmul(torch.matmul(Gk, Dk), Ak)).sum(0); Wa = (Pk * Dk).sum(0)
            W, r_ = adam(lambda W: L(W) - Z, Wa); oa[c] = W.cpu().numpy(); rg.append(r_)
            lam = 1e-6 * float((Pk.view(-1) * torch.diagonal(Gk, dim1=1, dim2=2).mean(1) * torch.diagonal(Ak, dim1=1, dim2=2).mean(1)).sum())
            We, it, rres = cg(lambda M: L(M) + lam * M, Z + lam * Wa, Wa); oe[c] = We.cpu().numpy(); its.append(it); rr.append(rres)
            del Gk, Ak, Dk, Z, Wa, W, We
        res["diag"]["fedfisher_kfac"] = {"grad_ratio_max": max(rg), "grad_ratio_med": float(np.median(rg))}
        res["diag"]["fedfisher_kfac_exact"] = {"cg_iters_max": max(its), "cg_iters_med": float(np.median(its)), "cg_resid_max": max(rr)}; res["seconds"]["solve_kfac"] = time.time() - t1; save()
        log(f"  K-FAC solves done ({time.time() - t1:.0f}s)"); finish("fedfisher_kfac", oa); finish("fedfisher_kfac_exact", oe); del oa, oe
    del Gst; torch.cuda.empty_cache()
need2 = any(r not in res["R"] or r + "_dense" not in res["R"] for r in ROWS2) or LOOPCHECK
if need2:
    t1 = time.time(); Gem = {}
    if Ast is None: Ast = {}
    sc = {}
    for t in TASKS:
        st = client_stats(t, False, t not in Ast, False); Gem[t] = {c: v.cpu() for c, v in st["G"].items()}
        if "A" in st: Ast[t] = {k: v.cpu() for k, v in st["A"].items()}
        sc[t] = {"Ge": float(np.mean([mdiag(v) for v in st["G"].values()]))}; del st; torch.cuda.empty_cache()
    res["diag"]["scale_true"] = sc; res["seconds"]["stats_true"] = time.time() - t1; save(); log(f"  statistics, true labels: {K} clients ({time.time() - t1:.0f}s)")
    t1 = time.time(); oc, oe, orl, its, rr, reach = {}, {}, {}, [], [], {}
    LC = [CELLS[0], [c for c in CELLS if c[1] == OUTN][-1]] if LOOPCHECK else []
    for c in (LC if LOOPCHECK else CELLS):
        Gk, Ak, L = kop(c, Gem, Ast, shift=COOR); Mk = dstack(c) + W0[c]; Zf = (Pk * torch.matmul(torch.matmul(Gk, Mk), Ak)).sum(0)
        SG = (Pk * Gk).sum(0); SA = (Pk * Ak).sum(0); Mh = torch.linalg.inv(SG) @ Zf @ torch.linalg.inv(SA); oc[c] = (Mh - W0[c]).cpu().numpy()
        Me, it, rres = cg(L, Zf, Mh); oe[c] = (Me - W0[c]).cpu().numpy(); its.append(it); rr.append(rres)
        if LOOPCHECK or RELEASED:   # the released solver verbatim: LOOPN steps, lr 1.0, autograd gradient of the element-mean squared residual, from the closed form
            Ht = Mh.clone().requires_grad_(True); tl = time.time()
            for i in range(LOOPN):
                loss = torch.nn.functional.mse_loss(L(Ht), Zf); loss.backward()
                with torch.no_grad(): Ht -= 1.0 * Ht.grad
                Ht.grad = None
            Ht = Ht.detach(); dE = (Me - Mh); dL = (Ht - Mh)
            reach[str(c)] = {"reach": float(dL.norm() / max(float(dE.norm()), 1e-30)), "cos": float((dL * dE).sum() / max(float(dL.norm() * dE.norm()), 1e-30)),
                             "closed_vs_delta": float(dE.norm() / max(float((Mh - W0[c]).norm()), 1e-30)), "loop_s": time.time() - tl}
            if LOOPCHECK: print("LOOPCHECK", c, reach[str(c)], flush=True)
            else: orl[c] = (Ht - W0[c]).cpu().numpy()
            del Ht, dE, dL
        del Gk, Ak, Mk, Zf, SG, SA, Mh, Me
    if LOOPCHECK: res["diag"]["loopcheck"] = reach; save(); sys.exit(0)
    res["diag"]["fedlpa_exact"] = {"cg_iters_max": max(its), "cg_iters_med": float(np.median(its)), "cg_resid_max": max(rr)}; res["seconds"]["solve_lpa"] = time.time() - t1; save()
    if RELEASED: res["diag"]["fedlpa_released"] = {"reach_med": float(np.median([v["reach"] for v in reach.values()])), "reach_max": max(v["reach"] for v in reach.values()), "loop_s": sum(v["loop_s"] for v in reach.values())}
    log(f"  FedLPA solves done ({time.time() - t1:.0f}s)"); finish("fedlpa", oc); finish("fedlpa_exact", oe)
    if RELEASED: finish("fedlpa_released", orl)
res["diag"]["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
res["done"] = all(r in res["R"] and r + "_dense" in res["R"] for r in ROWS1 + ROWS2); res["seconds"]["total"] = time.time() - t0; save(); log(f"FL rows done ({(time.time() - t0) / 60:.0f} min)")
