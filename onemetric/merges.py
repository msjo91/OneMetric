"""Weight-space merges of LoRA updates on dense per-matrix updates: task arithmetic, TIES, KnOTS-TIES, TSV-M, Iso-C, Iso-CTS, DC-Merge. Written from the papers; each function takes a list of {cell: update} and returns one."""
import numpy as np
def _svd(M): return np.linalg.svd(M.astype(np.float64), full_matrices=False)
def _orth(M): P, _, Qt = np.linalg.svd(M, full_matrices=False); return P @ Qt     # orthogonal Procrustes (TSV-M / Iso-CTS whitening)
def _lowrank(M, tol=1e-8):
    U, s, Vt = _svd(M); keep = s > tol * s[0]; return U[:, keep], s[keep], Vt[keep]
def ta(deltas): return {c: sum(d[c] for d in deltas) / len(deltas) for c in deltas[0]}
def _ties_combine(mats, keep=0.2):
    trimmed = []
    for m in mats:
        thr = np.quantile(np.abs(m), 1 - keep) if keep < 1 else -1.0; trimmed.append(np.where(np.abs(m) >= thr, m, 0.0))
    S = np.stack(trimmed); sign = np.sign(S.sum(0)); sign[sign == 0] = np.sign(S.sum()) or 1.0
    agree = np.where(np.sign(S) == sign[None], S, 0.0); cnt = (agree != 0).sum(0)
    return agree.sum(0) / np.maximum(cnt, 1)
def ties(deltas, keep=0.2): return {c: _ties_combine([d[c] for d in deltas], keep) for c in deltas[0]}
def knots_ties(deltas, keep=0.2):
    """KnOTS (Stoica et al. 2024): SVD of the column-concatenation [dW_1 ... dW_n]; shared U; task i represented by S V_i^T; TIES on those; reconstruct."""
    out = {}
    for c in deltas[0]:
        C = np.hstack([d[c] for d in deltas]); U, s, Vt = _lowrank(C); din = deltas[0][c].shape[1]
        Z = [s[:, None] * Vt[:, i * din:(i + 1) * din] for i in range(len(deltas))]
        out[c] = U @ _ties_combine(Z, keep)
    return out
def tsvm(deltas, frac=None):
    """TSV-M (Gargiulo et al. 2025): per task keep top-k singular triplets (k = d / n), concatenate, orthogonalize U and V by Procrustes, reconstruct."""
    n = len(deltas); out = {}
    for c in deltas[0]:
        Us, Ss, Vs = [], [], []
        for d in deltas:
            U, s, Vt = _lowrank(d[c]); k = min(len(s), min(d[c].shape) // n); Us.append(U[:, :k]); Ss.append(s[:k]); Vs.append(Vt[:k].T)
        out[c] = (_orth(np.hstack(Us)) * np.concatenate(Ss)) @ _orth(np.hstack(Vs)).T
    return out
def isoc(deltas):
    """Iso-C (Marczak et al. 2025): SVD of the task-arithmetic sum, all singular values replaced by their mean."""
    out = {}
    for c in deltas[0]:
        U, s, Vt = _lowrank(sum(d[c] for d in deltas)); out[c] = (U * s.mean()) @ Vt
    return out
def isocts(deltas, k_common=64, s_task=8):
    """Iso-CTS: common subspace = top-k_common of the sum; task-specific = top-s_task of each task's residual after removing the common subspace; whiten; flat spectrum."""
    out = {}
    for c in deltas[0]:
        U, s, Vt = _lowrank(sum(d[c] for d in deltas)); kc = min(k_common, len(s)); Uc = U[:, :kc]
        Us, Ss, Vs = [Uc], [s[:kc]], [Vt[:kc].T]
        for d in deltas:
            R = d[c] - Uc @ (Uc.T @ d[c]); Ut, st, Vtt = _lowrank(R); m = min(s_task, len(st)); Us.append(Ut[:, :m]); Ss.append(st[:m]); Vs.append(Vtt[:m].T)
        S = np.concatenate(Ss); out[c] = (_orth(np.hstack(Us)) * S.mean()) @ _orth(np.hstack(Vs)).T
    return out
def flatten_spectrum(d):
    """DC-Merge/PolCap smoothing: each client's singular values replaced by their mean (energy-preserving in sum, not in Frobenius norm)."""
    out = {}
    for c in d:
        U, s, Vt = _lowrank(d[c]); out[c] = (U * s.mean()) @ Vt
    return out
def cap_pool(deltas, tau=3.0):
    """Norm cap: per cell, scale any client whose Frobenius norm exceeds tau x median down to the ceiling."""
    out = [dict() for _ in deltas]
    for c in deltas[0]:
        norms = np.array([np.linalg.norm(d[c]) for d in deltas]); ceil = tau * np.median(norms)
        for i, d in enumerate(deltas): out[i][c] = d[c] * min(1.0, ceil / max(norms[i], 1e-12))
    return out
def polcap_isocts(deltas, tau=3.0, **kw): return isocts(cap_pool([flatten_spectrum(d) for d in deltas], tau), **kw)
def scale(merged, alpha): return {c: alpha * m for c, m in merged.items()}
def fro(merged): return float(np.sqrt(sum(np.linalg.norm(m) ** 2 for m in merged.values())))
OPS = {"ta": ta, "ties": ties, "knots_ties": knots_ties, "tsvm": tsvm, "isoc": isoc, "isocts": isocts, "polcap_isocts": polcap_isocts}
# --- official-parameter variants (from external/DC-Merge/vision_lora_merge/merging_functions.py, the KnOTS-protocol LoRA code used by DC-Merge) ---
def isocts_official(deltas, k_frac=0.8, r=16):
    """Iso-CTS exactly as in the DC-Merge/KnOTS LoRA code: common = N*r*k_frac rounded so the task-specific part splits evenly (n=8, r=16: common 104, 3 per task)."""
    N = len(deltas); common = int(N * r * k_frac); ts_total = round((N * r - common) / N) * N; common = N * r - ts_total; per = max(1, (N * r - common) // N)
    return isocts(deltas, k_common=common, s_task=per)
def _ties_small(mats):
    S = np.stack(mats); summed = S.sum(0); sign = np.sign(summed); mask = np.sign(S) == sign[None]; cnt = mask.sum(0)
    res = (S * mask).sum(0) / np.maximum(cnt, 1); return np.where(summed == 0, 0.0, res)
def dcmerge_core(deltas, r=16, smoothing="avg", rho=5.0, keep_entries=None):
    """DC-Merge up to the aligned, trimmed, sign-elected, block-masked coefficient matrix: per cell (Uc, M, Vc) with dW = Uc @ M @ Vc."""
    N = len(deltas); out = {}
    for c in deltas[0]:
        Us, Vs, sm = [], [], []
        for d in deltas:
            U, s, Vt = _svd(d[c]); k = min(r, len(s)); e = s[:k]
            dist = np.linspace(min(rho, e[0] / max(e[-1], 1e-12)), 1.0, k) if smoothing == "linear" else np.ones(k)
            dist = dist / dist.sum(); se = e.sum() * dist
            Us.append(U[:, :k]); Vs.append(Vt[:k]); sm.append((U[:, :k] * se) @ Vt[:k])
        Uc = _orth(np.hstack(Us)); Vc = _orth(np.vstack(Vs).T).T
        Ms = [Uc.T @ m @ Vc.T for m in sm]; K = Uc.shape[1]
        kept = []
        for M in Ms:
            flat = np.abs(M).ravel(); kk = keep_entries if keep_entries else max(1, int(1e-3 * flat.size)); thr = np.partition(flat, flat.size - kk)[flat.size - kk]; kept.append(np.where(np.abs(M) >= thr, M, 0.0))
        agg = _ties_small(kept); mask = np.zeros((K, K)); per = K // N
        for i in range(N): mask[i * per:(i + 1) * per, i * per:(i + 1) * per] = 1.0
        out[c] = (Uc, agg * mask, Vc)
    return out
def dcmerge(deltas, r=16, smoothing="avg", rho=5.0, keep_entries=None):
    """DC-Merge (Zhang et al., CVPR 2026) LoRA version, following the official code (see dcmerge_core). Unit scale alpha = 1."""
    return {c: Uc @ M @ Vc for c, (Uc, M, Vc) in dcmerge_core(deltas, r, smoothing, rho, keep_entries).items()}
import os as _os
def dcmerge_fixed16(deltas, **kw): return dcmerge(deltas, keep_entries=int(_os.environ.get("DC_KEEP", "16")), **kw)   # DC_KEEP = the official int(1e-3 K^2) at the full pool size (16 for N=8, 65 for N=16) # official rule keeps int(1e-3 K^2) entries per task (16 for N=8, 12 for N=7); fixed for leave-one-out
OPS.update({"isocts_official": isocts_official, "dcmerge": dcmerge, "dcmerge_fixed16": dcmerge_fixed16})
# --- KnOTS-protocol trimming: TIES's top-k% magnitude threshold is computed over the whole model (all cells of a task), not per cell ---
def _global_trim(per_task, keep):
    out = []
    for d in per_task:
        thr = np.quantile(np.concatenate([np.abs(m).ravel() for m in d.values()]), 1 - keep); out.append({c: np.where(np.abs(m) >= thr, m, 0.0) for c, m in d.items()})
    return out
def ties_global(deltas, keep=0.2):
    T = _global_trim(deltas, keep); return {c: _ties_combine([t[c] for t in T], keep=1.0) for c in deltas[0]}
def knots_ties_global(deltas, keep=0.2):
    Us, Zs = {}, [dict() for _ in deltas]
    for c in deltas[0]:
        C = np.hstack([d[c] for d in deltas]); U, s, Vt = _lowrank(C); din = deltas[0][c].shape[1]; Us[c] = U
        for i in range(len(deltas)): Zs[i][c] = s[:, None] * Vt[:, i * din:(i + 1) * din]
    T = _global_trim(Zs, keep); return {c: Us[c] @ _ties_combine([t[c] for t in T], keep=1.0) for c in deltas[0]}
OPS.update({"ties_global": ties_global, "knots_ties_global": knots_ties_global})
