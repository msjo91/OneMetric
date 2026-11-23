#!/usr/bin/env python
"""the one-shot FL methods that run in this setting, beyond FedFisher / FedLPA. Stored test draws, stored pools, no merge saved.
  GH-OFL (Turazza, Picone, Mamei, ICLR 2026; arXiv 2602.01186; no released code): each client uploads per-class counts, first and second moments of the
    FROZEN backbone's embedding of its training images; the server builds heads. Closed form: NB-diag (per-class variances shrunk toward the pooled
    variance), LDA (pooled within-class covariance over the union of every client's classes, as published; and per task), QDA (per-class covariance);
    shrinkage (1-a) S + a tr(S)/d I, a in ALPHAS. Trained on the server (outside this setting, run for completeness, our implementation from the paper's
    description): FisherMix (cosine head with additive margin on Gaussian samples in the Fisher subspace) and Proto-Hyper (low-rank residual on the
    standardised LDA logits, CE + KD from a QDA teacher, same samples). Two embeddings: 'pre' (CLIP pooled output before the projection; SigLIP 2
    pooled output; DINOv2 [CLS, mean patch]) and 'post' (the L2-normalised embedding the zero-shot head reads). Every head is evaluated with task identity,
    as every merge is (argmax over the task's own classes).
  Ensemble (Guha, Talwalkar, Smith 2019, the one-shot FL baseline): the K client models' softmax averaged ('ens_prob') and their logits averaged
    ('ens_logit'), each client's adapter run separately, at the clients' training temperature (100 CLIP / SigLIP 2, 1 DINOv2); K forward passes.
Client statistics use the client's training draw: 10,000 images per task, seed 0 (all when fewer); the draw our own clients trained on.
Stages (env OFL_STAGE): feat (base-model features of every task's training draw and stored test draw; OFL_SHARD i/n over tasks),
logit (each client adapter's logits on every co-pooled task's test draw; OFL_SHARD over adapters), pool (one pool, CLIPK_SUB: every row, R).
Env OFL_KIND = dc | siglip | dino plus the harness variables (as run_fedfisher_fedlpa.py). Writes results/ofl/{feat,logit}/<kind>/<arch>/ and
results/ofl/<kind>/<arch>/<sub>/ofl.json. Logs carry no retention number. OFL_SMOKE=1: one pool (OFL_SMOKE_SUB), 256 train / 64 test images, short training.
OFL_STATS=cal takes the client statistics from the client's 200 calibration images with their labels (H.load_x(t, "cal"), the inputs the
paper's method and FedFisher/FedLPA use) in place of the 10,000-image training draw. Output root results/ofl_cal; the test features are copied from results/ofl/feat and the client logits are read from
results/ofl/logit (stage logit is not run); the QDA logits are not cached there, because with fewer than two images in a class the class covariance falls back to the pool's union covariance.
Without OFL_STATS nothing changes. Backup run_ghofl_ensemble.py.bak_g140.
OFL_STATS=sub with OFL_NSUB=n takes the client statistics from n of stored training features per task (a fixed permutation
per task, seed 141 and the task name, so the draws are nested over n and the same on every backbone and pool; a task with fewer stored images uses all). Stage pool only, no feature job; output root
results/ofl_sub/n<n>; client logits and test features are read-only; QDA logits are not cached. Backup run_ghofl_ensemble.py.bak_g141.
OFL_SUBSEED (default 141) names the permutation; another seed writes to results/ofl_sub/s<seed>/n<n>. In the sub mode a task whose draw has no class
with two images has no within-class scatter; LDA per task then uses the pool's union covariance (the rule QDA already has for a class with one image), counted in diag.lda_task_union_fallback. Backup run_ghofl_ensemble.py.bak_g142."""
from onemetric.paths import ROOT as _OM_ROOT
import os, sys, json, time, random, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = _OM_ROOT
os.environ.setdefault("POOL", "dc16"); os.environ.setdefault("DINO_ROOT", "dino2")
KIND = os.environ["OFL_KIND"]; STAGE = os.environ["OFL_STAGE"]; assert KIND in ("dc", "siglip", "dino") and STAGE in ("feat", "logit", "pool")

if KIND == "dc": from onemetric import harness_clip as H
elif KIND == "siglip": from onemetric import harness_siglip as H
else: from onemetric import harness_dino as H
ARCH, SUB, log = H.ARCH, H.SUBTAG, H.log
SMOKE = os.environ.get("OFL_SMOKE") == "1"; CHECK = os.environ.get("OFL_CHECK") == "1"   # CHECK: full sizes, one pool (OFL_SMOKE_SUB), scratch output
STATS = os.environ.get("OFL_STATS", "train"); assert STATS in ("train", "cal", "sub") and not (STATS == "cal" and (SMOKE or STAGE == "logit")) and not (STATS == "sub" and (SMOKE or STAGE != "pool"))
NSUB = int(os.environ["OFL_NSUB"]) if STATS == "sub" else None; assert NSUB is None or 1 <= NSUB <= 10000
SUBSEED = int(os.environ.get("OFL_SUBSEED", "141"))
NTR = 256 if SMOKE else (200 if STATS == "cal" else NSUB if STATS == "sub" else 10000); NTE = 64 if SMOKE else None; NSYN = 16 if SMOKE else 128; STEPS = 20 if SMOKE else 400
TEMP = 1.0 if KIND == "dino" else 100.0; ALPHAS = [.01, .05, .2, .5, .9]; FM_CFG = [(16.0, 0.1), (32.0, 0.2)]; PH_K, PH_R, PH_T = 64, 32, 2.0
NEW = ["nall", "nr6_1", "nr6_2", "nr6_3", "mx12_1", "mx12_2", "mx12_3", "mx8_1", "mx8_2", "mx8_3", "mx6_1", "mx6_2"]
BENCH = {"dc": ["r12_10", "r12_11", "r12_13", "r8_10", "r8_11", "r8_12", "r6_13", "r6_14", "r6_15"],
         "siglip": ["all16", "k8", "r12_7", "r12_8", "r12_9", "r8_7", "r8_8", "r8_9", "r6_10", "r6_11", "r6_12"],
         "dino": ["all16", "s6", "r12_1", "r12_2", "r12_3", "r8_1", "r8_2", "r8_3", "r6_1", "r6_2", "r6_3"]}
SUBS = [os.environ["OFL_SMOKE_SUB"]] if (SMOKE or CHECK) else BENCH[KIND] + (NEW if KIND != "dino" else [])
OUTR = os.path.join(os.environ["OFL_SCRATCH"], "ofl") if (SMOKE or CHECK) else os.path.join(ROOT, "results", "ofl") if STATS == "train" else os.path.join(ROOT, "results", "ofl_cal") if STATS == "cal" else os.path.join(ROOT, "results", "ofl_sub", *([] if SUBSEED == 141 else [f"s{SUBSEED}"]), f"n{NSUB}")
FD, LD = os.path.join(OUTR, "feat", KIND, ARCH), os.path.join(OUTR, "logit", KIND, ARCH)
FD0 = os.path.join(ROOT, "results", "ofl", "feat", KIND, ARCH)   # stored features (test draw) and logits, read-only in the cal mode
if STATS in ("cal", "sub"): LD = os.path.join(ROOT, "results", "ofl", "logit", KIND, ARCH)
if STATS == "sub": FD = FD0
else:
    for d in (FD, LD): os.makedirs(d, exist_ok=True)
def sub_draw(t, Z0):
    """NSUB of the stored training features of task t, a fixed nested draw."""
    import zlib; n = len(Z0["tr_y"]); ix = np.sort(np.random.default_rng([SUBSEED, zlib.crc32(t.encode())]).permutation(n)[:NSUB])
    return {"tr_pre": Z0["tr_pre"][ix], "tr_post": Z0["tr_post"][ix], "tr_y": Z0["tr_y"][ix], "te_pre": Z0["te_pre"], "te_post": Z0["te_post"], "te_y": Z0["te_y"]}
POOLS = {s: H.subset(s) for s in SUBS}; UNIV = sorted(set().union(*POOLS.values()))
PAIRS = sorted({(a, t) for s in POOLS for a in POOLS[s] for t in POOLS[s]})
dev = torch.device("cuda:0"); torch.backends.cuda.matmul.allow_tf32 = False
def shard(xs): i, n = map(int, os.environ.get("OFL_SHARD", "0/1").split("/")); return xs[i::n]
def atomic_npz(p, **kw): tmp = p[:-4] + f".tmp{os.getpid()}.npz"; np.savez(tmp, **kw); os.replace(tmp, p)
def atomic_npy(p, a): tmp = p[:-4] + f".tmp{os.getpid()}.npy"; np.save(tmp, a); os.replace(tmp, p)
def head(C, t): return C.head_for_task(t) if KIND == "dino" else C.head(H.names_of(t), H.KNOTS_T[t])
def test_draw(t):
    X, y = H.load_x(t, "test"); y = np.asarray(y)
    return (X[:NTE], y[:NTE]) if NTE else (X, y)

# ---------------- feat ----------------
@torch.no_grad()
def feats(C, X, bs=128):
    dt = next(C.model.parameters()).dtype; pre, post = [], []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + bs])).to(C.dev).permute(0, 3, 1, 2).float().div_(255); xb = ((xb - C.MEAN) / C.STD).to(dt)
        if KIND == "dc": p = C.model.vision_model(pixel_values=xb).pooler_output; q = C.model.visual_projection(p)
        elif KIND == "siglip": p = C.model.get_image_features(pixel_values=xb); q = p
        else: h = C.model(pixel_values=xb).last_hidden_state; p = torch.cat([h[:, 0], h[:, 1:].mean(1)], 1); q = p
        q = q.float(); post.append((q / q.norm(dim=-1, keepdim=True)).cpu()); pre.append(p.float().cpu())
    return torch.cat(pre).numpy(), torch.cat(post).numpy()
def train_getter(t):
    if t in ("caltech101", "tinyimagenet", "aircraft"):
        from onemetric import data_newtasks as NT; n, _, get, _ = NT.source(t); return n, (lambda idx: get("train", idx))
    from onemetric import dino_data as DD
    ds = DD.split(t, "train"); lab, img, _ = DD.cols(t, ds); return len(ds), (lambda idx: (lambda d: (d[img], d[lab]))(ds.select(idx)))
def stage_feat():
    from concurrent.futures import ThreadPoolExecutor
    C = H.clip(); C.set_delta(None)
    for t in shard(UNIV):
        p = os.path.join(FD, f"{t}.npz")
        if os.path.exists(p): continue
        if STATS == "cal":
            t0 = time.time(); Xc, yc = H.load_x(t, "cal"); Xc, yc = Xc[:NTR], np.asarray(yc, dtype=np.int64)[:NTR]; assert len(yc) == NTR, (t, len(yc)); a, b = feats(C, Xc); Z0 = np.load(os.path.join(FD0, f"{t}.npz"))
            atomic_npz(p, tr_pre=a.astype(np.float16), tr_post=b.astype(np.float16), tr_y=yc, te_pre=Z0["te_pre"], te_post=Z0["te_post"], te_y=Z0["te_y"], n_train=np.int64(len(yc)))
            log(f"feat {t}: {len(yc)} calibration images, {len(Z0['te_y'])} test (stored features), {len(set(yc.tolist()))} classes seen ({time.time() - t0:.0f}s)"); continue
        t0 = time.time(); n, get = train_getter(t); idx = sorted(random.Random(0).sample(range(n), min(NTR, n))); P, Q, Y = [], [], []
        for j in range(0, len(idx), 1000):
            im, lab = get(idx[j:j + 1000])
            with ThreadPoolExecutor(16) as ex: X = np.stack(list(ex.map(H.prep, im, chunksize=16)))
            a, b = feats(C, X); P.append(a); Q.append(b); Y.append(np.asarray(lab, dtype=np.int64))
        X, y = test_draw(t); a, b = feats(C, X)
        if KIND != "dino":   # code check: the 'post' embedding is the one the zero-shot head reads
            ref = C.features(X[:64]).float().cpu().numpy(); err = float(np.abs(ref - b[:64]).max()); assert err < 2e-2, (t, err)
        atomic_npz(p, tr_pre=np.concatenate(P).astype(np.float16), tr_post=np.concatenate(Q).astype(np.float16), tr_y=np.concatenate(Y), te_pre=a.astype(np.float16),
                   te_post=b.astype(np.float16), te_y=y.astype(np.int64), n_train=np.int64(n))
        log(f"feat {t}: {len(idx)} of {n} training images, {len(y)} test, {len(set(np.concatenate(Y).tolist()))} classes seen ({time.time() - t0:.0f}s)")

# ---------------- logit ----------------
@torch.no_grad()
def stage_logit():
    C = H.clip(); partners = {}
    for a, t in PAIRS: partners.setdefault(a, []).append(t)
    for a in shard(sorted(partners)):
        todo = [t for t in partners[a] if not os.path.exists(os.path.join(LD, f"{a}__{t}.npy"))]
        if not todo: continue
        t0 = time.time(); C.set_delta(H.load_adapter(a))
        for t in todo:
            X, y = test_draw(t); f = C.features(X).float(); L = (f @ head(C, t).float().T) * TEMP
            atomic_npy(os.path.join(LD, f"{a}__{t}.npy"), L.cpu().numpy().astype(np.float32))
        log(f"logit adapter {a}: {len(todo)} tasks ({time.time() - t0:.0f}s)")

# ---------------- pool ----------------
def stats(Xtr, ytr, C):
    """per-class counts, means, within-class scatter (float64, on the GPU)."""
    X = torch.as_tensor(Xtr, device=dev, dtype=torch.float64); y = torch.as_tensor(ytr, device=dev)
    N = torch.bincount(y, minlength=C).double(); S1 = torch.zeros(C, X.shape[1], device=dev, dtype=torch.float64).index_add_(0, y, X)
    mu = S1 / N.clamp(min=1)[:, None]; R = X - mu[y]; SW = R.T @ R; R = R.float()
    return N, mu, SW, R, y
def shrink(S, a): d = S.shape[0]; return (1 - a) * S + a * (torch.trace(S) / d) * torch.eye(d, device=S.device, dtype=S.dtype)
def lda_logits(Xte, mu, logpi, Sig):
    L = torch.linalg.cholesky(Sig); Wt = torch.cholesky_solve(mu.T, L)   # d x C = Sig^-1 mu^T
    return Xte @ Wt - 0.5 * (mu * Wt.T).sum(1) + logpi
def acc_of(logits, y, ok):
    lg = logits.clone(); lg[:, ~ok] = -float("inf"); return float((lg.argmax(1).cpu().numpy() == y).mean())
def stage_pool():
    TASKS, RET = H.TASKS, getattr(H, "RET", H.TASKS)
    OUT = os.path.join(OUTR, KIND, ARCH, SUB); os.makedirs(OUT, exist_ok=True); RJ = os.path.join(OUT, "ofl.json")
    res = H.jload(RJ) if os.path.exists(RJ) else {"kind": KIND, "arch": ARCH, "sub": SUB, "tasks": TASKS, "ret": RET, "ntr": NTR, "alphas": ALPHAS, "acc": {}, "R": {}, "diag": {}}
    if STATS == "cal": res["stats"] = "cal"
    if STATS == "sub": res["stats"] = "sub"; res["nsub"] = NSUB; res["subseed"] = SUBSEED
    if res.get("done"): log("already done"); return
    t0 = time.time(); home = {t: H.jload(os.path.join(H.EVAL, f"client_{t}.json"))[t] for t in TASKS}
    Z = {t: np.load(os.path.join(FD, f"{t}.npz")) for t in TASKS}
    if STATS == "sub": Z = {t: sub_draw(t, Z[t]) for t in TASKS}; res["diag"]["n_used"] = {t: int(len(Z[t]["tr_y"])) for t in TASKS}
    if KIND == "dino": from onemetric import dino_backend as WM; ncls = {t: int(np.load(WM.probe_path(t))["W"].shape[0]) for t in TASKS}
    else: ncls = {t: len(H.names_of(t)) for t in TASKS}
    acc = {}
    def put(row, t, a): acc.setdefault(row, {})[t] = a
    # ensemble (Guha et al.) and the own-adapter code check
    own_err = {}
    for t in TASKS:
        y = Z[t]["te_y"]; Ls = [torch.as_tensor(np.load(os.path.join(LD, f"{a}__{t}.npy")), device=dev) for a in TASKS]; ok = torch.ones(Ls[0].shape[1], dtype=torch.bool, device=dev)
        put("ens_prob", t, acc_of(torch.stack([l.softmax(1) for l in Ls]).mean(0), y, ok)); put("ens_logit", t, acc_of(torch.stack(Ls).mean(0), y, ok))
        own_err[t] = round(acc_of(Ls[TASKS.index(t)], y, ok) - home[t], 4)
    res["diag"]["own_minus_home"] = own_err
    for fs in ("pre", "post"):
        off, Cs, parts = {}, 0, []
        for t in TASKS: off[t] = Cs; Cs += ncls[t]
        St = {}
        for t in TASKS:
            N, mu, SW, R, y = stats(Z[t][f"tr_{fs}"].astype(np.float32), Z[t]["tr_y"], ncls[t]); St[t] = (N, mu, SW, R, y)
        d = St[TASKS[0]][1].shape[1]; NU = sum(float(St[t][0].sum()) for t in TASKS); CU = sum(int((St[t][0] > 0).sum()) for t in TASKS)
        SWU = sum(St[t][2] for t in TASKS) / (NU - CU); vpool = torch.diag(SWU)
        Te = {t: torch.as_tensor(Z[t][f"te_{fs}"].astype(np.float32), device=dev, dtype=torch.float64) for t in TASKS}
        for t in TASKS:
            N, mu, SW, R, y = St[t]; ok = N > 0; logpi = torch.log(N.clamp(min=1) / N.sum()); yte = Z[t]["te_y"]; X = Te[t]
            Stask = SW / max(float(N.sum()) - int(ok.sum()), 1.0)
            if STATS == "sub" and float(N.max()) < 2: Stask = SWU; res["diag"].setdefault("lda_task_union_fallback", {}).setdefault(fs, []).append(t)
            for a in ALPHAS:
                put(f"gh_lda_union_{fs}_{a}", t, acc_of(lda_logits(X, mu, logpi, shrink(SWU, a)), yte, ok))
                put(f"gh_lda_task_{fs}_{a}", t, acc_of(lda_logits(X, mu, logpi, shrink(Stask, a)), yte, ok))
                v = torch.zeros_like(mu).index_add_(0, y, (R * R).double()) / N.clamp(min=1)[:, None]; v = ((1 - a) * v + a * vpool[None]).clamp(min=1e-12)
                nb = -0.5 * ((X * X) @ (1 / v).T - 2 * X @ (mu / v).T + (mu * mu / v).sum(1)[None]) - 0.5 * torch.log(v).sum(1)[None] + logpi[None]
                put(f"gh_nb_{fs}_{a}", t, acc_of(nb, yte, ok))
                qp = os.path.join(FD, f"qda_{t}_{fs}_{a}.npy")
                if STATS == "train" and os.path.exists(qp): q = torch.as_tensor(np.load(qp), device=dev)
                else:
                    q = torch.empty(X.shape[0], mu.shape[0], device=dev, dtype=torch.float32); Xf = X.float()
                    for c in range(mu.shape[0]):
                        if N[c] < 1: q[:, c] = -float("inf"); continue
                        Rc = R[y == c].float(); Sc = (Rc.T @ Rc) / max(float(N[c]) - 1, 1.0) if N[c] >= 2 else SWU.float()
                        Lc = torch.linalg.cholesky(shrink(Sc, a) + 1e-6 * (float(torch.trace(Sc)) / d + 1e-12) * torch.eye(d, device=dev)); D = (Xf - mu[c].float()).T
                        q[:, c] = -0.5 * (torch.cholesky_solve(D, Lc) * D).sum(0) - torch.log(torch.diagonal(Lc)).sum() + float(logpi[c])
                    if STATS == "train": atomic_npy(qp, q.cpu().numpy())
                put(f"gh_qda_{fs}_{a}", t, acc_of(q.double(), yte, ok))
        # Fisher subspace on the union (FisherMix, Proto-Hyper): S_B v = lam S_W v, S_W = pooled within (a = .05)
        MU = torch.cat([St[t][1] for t in TASKS]); NN = torch.cat([St[t][0] for t in TASKS]); okU = NN > 0; mbar = (NN[:, None] * MU).sum(0) / NN.sum()
        SB = ((MU - mbar)[okU] * NN[okU][:, None]).T @ (MU - mbar)[okU] / NN.sum(); Lw = torch.linalg.cholesky(shrink(SWU, .05))
        M = torch.linalg.solve_triangular(Lw, torch.linalg.solve_triangular(Lw, SB, upper=False).T, upper=False); lam, U = torch.linalg.eigh((M + M.T) / 2)
        lam, U = lam.flip(0).clamp(min=0), U.flip(1); k = int((torch.cumsum(lam, 0) / lam.sum() < .95).sum()) + 1; k = max(2, min(k, 512, CU - 1, d))
        V = torch.linalg.solve_triangular(Lw.T, U, upper=True)   # S_W-orthonormal Fisher directions (d x d); first k used
        res["diag"][f"fisher_k_{fs}"] = k
        g = torch.Generator(device=dev); g.manual_seed(124)
        def sample(kk):
            Vk = V[:, :kk]; Zs, Ys = [], []
            for t in TASKS:
                N, mu, SW, R, y = St[t]
                for c in range(mu.shape[0]):
                    if N[c] < 1: continue
                    Rc = R[y == c].double() @ Vk; Sc = (Rc.T @ Rc) / max(float(N[c]) - 1, 1.0) if N[c] >= 2 else torch.eye(kk, device=dev, dtype=torch.float64)
                    Lc = torch.linalg.cholesky(shrink(Sc, .05) + 1e-9 * torch.eye(kk, device=dev, dtype=torch.float64)).float()
                    Zs.append(((mu[c] @ Vk).float()[None] + torch.randn(NSYN, kk, device=dev, generator=g) @ Lc.T)); Ys.append(torch.full((NSYN,), off[t] + c, device=dev))
            return torch.cat(Zs), torch.cat(Ys)
        Zs, Ys = sample(k); okall = okU
        for s_, m_ in FM_CFG:   # FisherMix: cosine head with additive margin on the samples
            torch.manual_seed(124); W = torch.nn.Parameter(torch.nn.functional.normalize((MU @ V[:, :k]).float(), dim=1).clone()); opt = torch.optim.Adam([W], lr=1e-2)
            for it in range(STEPS):
                ib = torch.randint(0, len(Ys), (min(16384, len(Ys)),), device=dev, generator=g); cos = torch.nn.functional.normalize(Zs[ib], dim=1) @ torch.nn.functional.normalize(W, dim=1).T
                loss = torch.nn.functional.cross_entropy(s_ * (cos - m_ * torch.nn.functional.one_hot(Ys[ib], Cs)), Ys[ib]); opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                Wn = torch.nn.functional.normalize(W, dim=1)
                for t in TASKS:
                    lg = s_ * torch.nn.functional.normalize((Te[t] @ V[:, :k]).float(), dim=1) @ Wn[off[t]:off[t] + ncls[t]].T; put(f"gh_fishermix_{fs}_{int(s_)}_{m_}", t, acc_of(lg, Z[t]["te_y"], St[t][0] > 0))
        # Proto-Hyper: standardised LDA logits in the first PH_K Fisher directions + a rank-PH_R residual; CE + KD (T = PH_T) from the QDA teacher there
        kh = min(PH_K, k); Zh, Yh = sample(kh); Vh = V[:, :kh]; Mh = (MU @ Vh).float(); lp = torch.log(NN.clamp(min=1) / NN.sum()).float()
        def base(z): o = z @ Mh.T - 0.5 * (Mh * Mh).sum(1) + lp; o[:, ~okall] = -1e4; return (o - o.mean(1, keepdim=True)) / o.std(1, keepdim=True).clamp(min=1e-6)
        Pc, ld_ = [], []
        for t in TASKS:
            N, mu, SW, R, y = St[t]
            for c in range(mu.shape[0]):
                Rc = R[y == c].double() @ Vh; Sc = (Rc.T @ Rc) / max(float(N[c]) - 1, 1.0) if N[c] >= 2 else torch.eye(kh, device=dev, dtype=torch.float64)
                Sc = shrink(Sc, .05) + 1e-9 * torch.eye(kh, device=dev, dtype=torch.float64); Pc.append(torch.linalg.inv(Sc).float()); ld_.append(float(torch.logdet(Sc)))
        Pc = torch.stack(Pc); ld_ = torch.tensor(ld_, device=dev)
        with torch.no_grad():
            Tq = torch.empty(len(Yh), Cs, device=dev)
            for c0 in range(0, Cs, 8):
                D = Zh[:, None, :] - Mh[None, c0:c0 + 8]; Tq[:, c0:c0 + 8] = -0.5 * torch.einsum("nci,cij,ncj->nc", D, Pc[c0:c0 + 8], D) - 0.5 * ld_[c0:c0 + 8] + lp[c0:c0 + 8]
            Tq[:, ~okall] = -1e4
        torch.manual_seed(124); U1 = torch.nn.Parameter(0.01 * torch.randn(PH_R, kh, device=dev)); U2 = torch.nn.Parameter(torch.zeros(Cs, PH_R, device=dev)); opt = torch.optim.Adam([U1, U2], lr=1e-3)
        for it in range(STEPS):
            ib = torch.randint(0, len(Yh), (min(16384, len(Yh)),), device=dev, generator=g); o = base(Zh[ib]) + Zh[ib] @ U1.T @ U2.T
            kd = torch.nn.functional.kl_div(torch.log_softmax(o / PH_T, 1), torch.softmax(Tq[ib] / PH_T, 1), reduction="batchmean") * PH_T ** 2
            loss = 0.5 * torch.nn.functional.cross_entropy(o, Yh[ib]) + 0.5 * kd; opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for t in TASKS:
                z = (Te[t] @ Vh).float(); o = base(z) + z @ U1.T @ U2.T; put(f"gh_protohyper_{fs}", t, acc_of(o[:, off[t]:off[t] + ncls[t]], Z[t]["te_y"], St[t][0] > 0))
        res["diag"][f"missing_classes_{fs}"] = {t: int((St[t][0] < 1).sum()) for t in TASKS}
    res["acc"] = acc; res["R"] = {r: float(np.mean([acc[r][t] / max(home[t], 1e-9) for t in RET])) for r in acc}
    res["diag"]["seconds"] = round(time.time() - t0, 1); res["diag"]["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    res["done"] = True; tmp = RJ + ".tmp"; H.jdump(tmp, res); os.replace(tmp, RJ)
    log(f"pool {KIND}/{ARCH}/{SUB}: {len(res['R'])} rows, own-adapter check max |acc - home| {max(abs(v) for v in own_err.values()):.4f}, {res['diag']['seconds']}s")

{"feat": stage_feat, "logit": stage_logit, "pool": stage_pool}[STAGE]()
