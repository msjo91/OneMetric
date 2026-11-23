"""Weight-space merging rules on LoRA factors for the LLaVA rows: concatenation, TIES, DARE, TSV-M, DC-Merge, KnOTS-TIES, Iso-C and Iso-CTS, with the norm scalings some rows apply to the clients first. Taken from the authors' earlier code base for LoRA pools; RULES maps a row name to its function and settings."""
import numpy as np
import torch
def _cells(pool):
    first = next(iter(pool.values()))
    return sorted(first.keys())

def _delta(fs, dtype=np.float32):
    return (fs["B"].astype(dtype) @ fs["A"].astype(dtype))

def _svd_factorize(delta, rank, seed=0):
    
    t = torch.from_numpy(np.ascontiguousarray(delta))
    torch.manual_seed(seed)
    U, S, V = torch.svd_lowrank(t, q=min(rank + 16, min(t.shape)), niter=4)
    U, S, V = U[:, :rank], S[:rank], V[:, :rank]
    B = (U * S.sqrt()).numpy()
    A = (S.sqrt()[:, None] * V.T).numpy()
    return A.astype(np.float64), B.astype(np.float64)

def merge_concat(pool):
    tasks = sorted(pool)
    n = len(tasks)
    out = {}
    for cell in _cells(pool):
        A = np.vstack([pool[t][cell]["A"] for t in tasks])
        B = np.hstack([pool[t][cell]["B"] / n for t in tasks])
        out[cell] = {"A": A, "B": B}
    return out

def merge_ties(pool, rank=64, keep=0.2, elect=True, disjoint=True, mag_pool=None):
    
    tasks = sorted(pool)
    out = {}
    for cell in _cells(pool):
        deltas = [_delta(pool[t][cell]) for t in tasks]
        mags = [_delta(mag_pool[t][cell]) for t in tasks] if mag_pool is not None else deltas
        merged = _ties_combine(deltas, mags, keep, elect, disjoint)
        A, B = _svd_factorize(merged, rank)
        out[cell] = {"A": A, "B": B}
    return out

def _ties_combine(deltas, mags, keep=0.2, elect=True, disjoint=True):
    
    trimmed, trimmed_m = [], []
    for d, m in zip(deltas, mags):
        if keep < 1.0:
            thresh = np.quantile(np.abs(d), 1 - keep)
            mask = np.abs(d) >= thresh
            trimmed.append(np.where(mask, d, 0.0)); trimmed_m.append(np.where(mask, m, 0.0))
        else:
            trimmed.append(d); trimmed_m.append(m)
    stack = np.stack(trimmed); stack_m = np.stack(trimmed_m)
    if elect:
        sign = np.sign(stack.sum(axis=0))
        agree = np.where(np.sign(stack) == sign[None], stack_m, 0.0)
    else:
        agree = stack_m
    cnt = (agree != 0).sum(axis=0)
    if disjoint:
        return agree.sum(axis=0) / np.maximum(cnt, 1)
    mean_cnt = max(float(cnt[cnt > 0].mean()) if (cnt > 0).any() else 1.0, 1.0)
    return agree.sum(axis=0) / mean_cnt

def merge_dare(pool, rank=64, p=0.9, seed=0):
    rng = np.random.default_rng(seed)
    tasks = sorted(pool)
    out = {}
    for cell in _cells(pool):
        kept = []
        for t in tasks:
            d = _delta(pool[t][cell])
            mask = rng.random(d.shape) >= p
            kept.append(d * mask / (1 - p))
        merged = np.mean(kept, axis=0)
        A, B = _svd_factorize(merged, rank)
        out[cell] = {"A": A, "B": B}
    return out

def client_norms(pool, cell):
    
    return {t: float(np.linalg.norm(_delta(pool[t][cell]))) for t in pool}

def normalize_pool(pool, target="median", gamma=1.0, max_up=None):
    
    out = {t: {} for t in pool}
    for cell in _cells(pool):
        norms = client_norms(pool, cell)
        v = np.array(list(norms.values()))
        if target == "median":
            tgt = float(np.median(v))
        elif target == "geomean":
            tgt = float(np.exp(np.mean(np.log(np.maximum(v, 1e-12)))))
        elif target == "mean":
            tgt = float(np.mean(v))
        else:
            raise ValueError(target)
        for t in pool:
            sc = (tgt / max(norms[t], 1e-12)) ** gamma
            if max_up is not None:
                sc = min(sc, max_up)
            out[t][cell] = {"A": pool[t][cell]["A"], "B": pool[t][cell]["B"] * sc}
    return out

def whiten_pool(pool, target="median"):
    
    if target not in ("median", "mean"):
        raise ValueError(target)
    out = {t: {} for t in pool}
    for cell in _cells(pool):
        norms = client_norms(pool, cell)
        v = np.array(list(norms.values()))
        tgt = float(np.median(v)) if target == "median" else float(np.mean(v))
        for t in pool:
            B, A = pool[t][cell]["B"], pool[t][cell]["A"]
            r = A.shape[0]
            U, S, V = _lowrank_svd(B, A)
            k = len(S)
            A_out = np.zeros((r, A.shape[1])); B_out = np.zeros((B.shape[0], r))
            if k:
                scale = tgt / np.sqrt(k)
                A_out[:k] = V.T; B_out[:, :k] = U * scale
            out[t][cell] = {"A": A_out, "B": B_out}
    return out

def merge_whiten(pool, target="median", base="isocts", **kw):
    
    return _BASE_RULES[base](whiten_pool(pool, target), **kw)

def merge_normeq(pool, target="median", base="concat", gamma=1.0, max_up=None, **kw):
    
    return _BASE_RULES[base](normalize_pool(pool, target, gamma=gamma, max_up=max_up), **kw)

def _lowrank_svd(B, A, tol=1e-6):
    
    B = np.asarray(B, dtype=np.float64); A = np.asarray(A, dtype=np.float64)
    Qb, Rb = np.linalg.qr(B)
    Qa, Ra = np.linalg.qr(A.T)
    P, s, Wt = np.linalg.svd(Rb @ Ra.T)
    keep = s > tol * s[0] if s.size and s[0] > 0 else np.zeros(s.shape, bool)
    return Qb @ P[:, keep], s[keep], Qa @ Wt[keep].T

def _orth(M):
    
    P, _, Qt = np.linalg.svd(M, full_matrices=False)
    return P @ Qt

def _factors_from_svd(U, s, V, rank):
    
    order = np.argsort(-s)
    U, s, V = U[:, order], s[order], V[:, order]
    k = min(len(s), rank)
    root = np.sqrt(s[:k])
    B = np.zeros((U.shape[0], rank)); A = np.zeros((rank, V.shape[0]))
    B[:, :k] = U[:, :k] * root
    A[:k] = root[:, None] * V[:, :k].T
    return A, B

def merge_tsvm(pool, rank=64, alpha=1.0):
    
    tasks = sorted(pool)
    n = len(tasks)
    out = {}
    for cell in _cells(pool):
        Us, Ss, Vs = [], [], []
        for t in tasks:
            U, s, V = _lowrank_svd(pool[t][cell]["B"], pool[t][cell]["A"])
            k = min(len(s), min(U.shape[0], V.shape[0]) // n)
            Us.append(U[:, :k]); Ss.append(s[:k]); Vs.append(V[:, :k])
        U, S, V = np.hstack(Us), np.concatenate(Ss), np.hstack(Vs)
        A, B = _factors_from_svd(_orth(U), alpha * S, _orth(V), rank)
        out[cell] = {"A": A, "B": B}
    return out

def merge_isoc(pool, rank=64, alpha=1.0):
    
    tasks = sorted(pool)
    out = {}
    for cell in _cells(pool):
        U, s, V = _lowrank_svd(np.hstack([pool[t][cell]["B"] for t in tasks]),
                               np.vstack([pool[t][cell]["A"] for t in tasks]))
        flat = np.full(len(s), alpha * s.mean()) if len(s) else s
        A, B = _factors_from_svd(U, flat, V, rank)
        out[cell] = {"A": A, "B": B}
    return out

def merge_isocts(pool, rank=64, k_common=32, s=None, alpha=1.0):
    
    tasks = sorted(pool)
    n = len(tasks)
    s_ts = (rank - k_common) // n if s is None else s
    out = {}
    for cell in _cells(pool):
        U, sv, V = _lowrank_svd(np.hstack([pool[t][cell]["B"] for t in tasks]),
                                np.vstack([pool[t][cell]["A"] for t in tasks]))
        kc = min(k_common, len(sv))
        Uc = U[:, :kc]
        Us, Ss, Vs = [Uc], [sv[:kc]], [V[:, :kc]]
        for t in tasks:
            Bt, At = pool[t][cell]["B"], pool[t][cell]["A"]
            Bres = Bt - Uc @ (Uc.T @ Bt)
            Ut, st, Vt = _lowrank_svd(Bres, At)
            m = min(s_ts, len(st))
            Us.append(Ut[:, :m]); Ss.append(st[:m]); Vs.append(Vt[:, :m])
        Ustar, Sstar, Vstar = np.hstack(Us), np.concatenate(Ss), np.hstack(Vs)
        flat = np.full(len(Sstar), alpha * Sstar.mean()) if len(Sstar) else Sstar
        A, B = _factors_from_svd(_orth(Ustar), flat, _orth(Vstar), rank)
        out[cell] = {"A": A, "B": B}
    return out

def _knots_align(pool, cell, tasks):
    
    Qb, Rb = np.linalg.qr(np.hstack([np.asarray(pool[t][cell]["B"], dtype=np.float64) for t in tasks]))
    QRa = [np.linalg.qr(np.asarray(pool[t][cell]["A"], dtype=np.float64).T) for t in tasks]
    offsets = np.cumsum([0] + [q.shape[1] for q, _ in QRa])
    core = np.hstack([Rb[:, offsets[i]:offsets[i + 1]] @ Ra.T for i, (_, Ra) in enumerate(QRa)])
    P, s, Wt = np.linalg.svd(core)
    keep = s > 1e-6 * s[0] if s.size and s[0] > 0 else np.zeros(s.shape, bool)
    U = Qb @ P[:, keep]
    Zs = [s[keep, None] * (Wt[keep][:, offsets[i]:offsets[i + 1]] @ Qa.T) for i, (Qa, _) in enumerate(QRa)]
    return U, Zs

def merge_knots(pool, rank=64, method="ties", keep=0.2, elect=True, disjoint=True):
    
    tasks = sorted(pool)
    out = {}
    for cell in _cells(pool):
        U, Zs = _knots_align(pool, cell, tasks)
        if method == "ta":
            Zm = np.mean(Zs, axis=0)
        elif method == "ties":
            Zm = _ties_combine(Zs, Zs, keep=keep, elect=elect, disjoint=disjoint)
        else:
            raise ValueError(method)
        Uo, so, Vo = _lowrank_svd(U, Zm)
        A, B = _factors_from_svd(Uo, so, Vo, rank)
        out[cell] = {"A": A, "B": B}
    return out

def _dcmerge_smooth(s, smoothing="mean", rho=5.0):
    
    s = np.asarray(s, dtype=np.float64)
    if smoothing == "none" or s.size == 0:
        return s
    if smoothing == "mean":
        w = np.full(s.size, 1.0 / s.size)
    elif smoothing == "linear":
        ratio = min(float(rho), s[0] / s[-1]) if s[-1] > 0 else float(rho)
        w = np.linspace(ratio, 1.0, s.size)
        w = w / w.sum()
    else:
        raise ValueError(smoothing)
    return s.sum() * w

def _dcmerge_trim(mats, percent=1e-3):
    
    out = []
    for m in mats:
        flat = np.abs(m).ravel()
        k = max(1, int(percent * flat.size))
        if k >= flat.size:
            out.append(m); continue
        thr = np.partition(flat, flat.size - k)[flat.size - k]
        out.append(np.where(np.abs(m) >= thr, m, 0.0))
    return out

def merge_dcmerge(pool, rank=64, smoothing="mean", rho=5.0, trim_percent=1e-3,
                  aggregate="ties", mask=True, alpha=1.0):
    
    tasks = sorted(pool)
    out = {}
    for cell in _cells(pool):
        Us, Ss, Vs = [], [], []
        for t in tasks:
            U, s, V = _lowrank_svd(pool[t][cell]["B"], pool[t][cell]["A"])
            Us.append(U); Ss.append(_dcmerge_smooth(s, smoothing, rho)); Vs.append(V)
        off = np.cumsum([0] + [u.shape[1] for u in Us])
        K = int(off[-1])
        if K == 0:
            out[cell] = {"A": np.zeros((rank, pool[tasks[0]][cell]["A"].shape[1])),
                         "B": np.zeros((pool[tasks[0]][cell]["B"].shape[0], rank))}
            continue
        Ut = _orth(np.hstack(Us))
        Vt = _orth(np.hstack(Vs))
        Ms = [((Ut.T @ Us[i]) * Ss[i]) @ (Vs[i].T @ Vt) for i in range(len(tasks))]
        Ms = _dcmerge_trim(Ms, trim_percent)
        if aggregate == "ties":
            Mt = _ties_combine(Ms, Ms, keep=1.0, elect=True, disjoint=True)
        elif aggregate == "ta":
            Mt = np.mean(Ms, axis=0)
        else:
            raise ValueError(aggregate)
        if mask:
            blk = np.zeros((K, K))
            for i in range(len(tasks)):
                blk[off[i]:off[i + 1], off[i]:off[i + 1]] = 1.0
            Mt = Mt * blk
        U, s, V = _lowrank_svd(Ut, (alpha * Mt) @ Vt.T)
        A, B = _factors_from_svd(U, s, V, rank)
        out[cell] = {"A": A, "B": B}
    return out

_BASE_RULES = {"ties": merge_ties, "isoc": merge_isoc, "isocts": merge_isocts, "dcmerge": merge_dcmerge}

RULES = {
    'concat': (merge_concat, {}),
    'ties64': (merge_ties, {'rank': 64, 'keep': 0.2}),
    'dare64': (merge_dare, {'rank': 64, 'p': 0.9}),
    'tsvm': (merge_tsvm, {'rank': 64, 'alpha': 1.0}),
    'dcmerge': (merge_dcmerge, {'rank': 64, 'smoothing': 'mean', 'trim_percent': 0.001, 'aggregate': 'ties', 'mask': True, 'alpha': 1.0}),
    'knots_ties': (merge_knots, {'rank': 64, 'method': 'ties', 'keep': 0.2}),
    'isocts_white': (merge_whiten, {'target': 'median', 'base': 'isocts', 'rank': 64}),
    'isoc_white': (merge_whiten, {'target': 'median', 'base': 'isoc', 'rank': 64}),
    'trim_g50': (merge_normeq, {'target': 'median', 'base': 'ties', 'gamma': 0.5, 'max_up': 1.0, 'rank': 64, 'keep': 0.2, 'elect': False, 'disjoint': False}),
    'dcmerge_g50': (merge_normeq, {'target': 'median', 'base': 'dcmerge', 'gamma': 0.5, 'max_up': 1.0, 'rank': 64}),
}
