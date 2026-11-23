"""The spectral end of the method, per adapted matrix: the clients' compact SVDs, the consensus block and the residual blocks, the left stack made orthonormal by its polar factor, the right stack made orthonormal in the metric of the pooled input Gram, and one level per block that matches the block's response energy on its own client's inputs. Pure numpy in float64; no labels."""
import numpy as np
def orth(M):
    """Orthogonal Procrustes: nearest orthonormal matrix to M (TSV-M / Iso-CTS whitening)."""
    P, _, Qt = np.linalg.svd(np.asarray(M, dtype=np.float64), full_matrices=False); return P @ Qt
def compact_svd(B, A, w=1.0):
    """Exact compact SVD of w * B A from the LoRA factors B (d_out x r) and A (r x d_in): QR of B and A^T, SVD of the r x r core."""
    QB, RB = np.linalg.qr(np.asarray(B, dtype=np.float64)); QA, RA = np.linalg.qr(np.asarray(A, dtype=np.float64).T)
    P, s, Qt = np.linalg.svd(w * (RB @ RA.T)); return QB @ P, s, QA @ Qt.T
def dense_svd(Dk, rank):
    """Compact SVD of a dense update truncated to rank (for pools without stored factors)."""
    U, s, Vt = np.linalg.svd(np.asarray(Dk, dtype=np.float64), full_matrices=False); return U[:, :rank], s[:rank], Vt[:rank].T
def tr_q(Ufac, Vfac, G):
    """tr(R G R^T) for R = Ufac @ Vfac^T."""
    return float(np.sum((Ufac.T @ Ufac) * (Vfac.T @ (G @ Vfac))))
def gram_metric(Gsum, ridge):
    """(G^_r^{1/2}, G^_r^{-1/2}) for G^_r = Gsum + ridge * tr(Gsum)/d * I, by eigendecomposition."""
    d = Gsum.shape[0]; lam, E = np.linalg.eigh(Gsum); lam = np.clip(lam, 0, None) + ridge * float(np.trace(Gsum)) / d
    return (E * np.sqrt(lam)) @ E.T, (E / np.sqrt(lam)) @ E.T
def m_cell(F, G_list, Gsum, metric, k_c, r_res, level, factors=False, wk=None, metric_mode="gram"):
    """One cell. F = [(U_k, s_k, V_k)] compact SVDs of the (whitened or raw) client updates; G_list = [G~_k] shrunk Grams (float64) in the same order; Gsum = sum_k G~_k;
    metric = (G^_r^{1/2}, G^_r^{-1/2}); k_c common directions, r_res residual directions per client; level in {'block', 'global'}. Returns the float32 endpoint cell."""
    K = len(F); Q, _, _ = np.linalg.svd(np.hstack([U for U, _, _ in F]), full_matrices=False); Uc = Q[:, :k_c]   # top-k eigenvectors of the averaged projector (1/K) sum_k U_k U_k^T = leading left singular vectors of the stack [U_1 ... U_K]; same subspace as the eigh of the d x d projector, O(d m^2) instead of O(d^3) (the LLaVA down_proj cells have d = 11,008)
    Mk = sum(((Uc.T @ U) * s) @ V.T for U, s, V in F) / K; Pc, sc, Vct = np.linalg.svd(Mk, full_matrices=False); Uc = Uc @ Pc; Vc = Vct.T
    Us, Vs, resid, ew = [Uc], [Vc], [], []
    for U, s, V in F:
        Ur, sr, Vr = U[:, :r_res], s[:r_res], V[:, :r_res]; Z = Ur - Uc @ (Uc.T @ Ur); a = np.linalg.norm(Z, axis=0); Ub = Z / np.where(a > 1e-12, a, 1.0); sb = a * sr
        Us.append(Ub); Vs.append(Vr); resid.append((Ub * sb, Vr)); ew.append(float(np.sum(sb ** 2)))
    Ut = orth(np.hstack(Us)); Vhat = np.hstack(Vs); n_c = Uc.shape[1]; rr = Vhat.shape[1] - n_c; rr_k = [v.shape[1] for _, v in resid]
    blocks, pos = [slice(0, n_c)], n_c
    for r in rr_k: blocks.append(slice(pos, pos + r)); pos += r
    Gh, Gih = metric; VG = (Gih @ orth(Gh @ Vhat)) if metric_mode == "gram" else orth(Vhat); Dv = np.empty(VG.shape[1])   # metric_mode 'euclid' = Euclidean Procrustes of the right stack (CT/TSV-M style)
    e_c = tr_q(Uc * sc, Vc, Gsum); e_k = [tr_q(A_, Vr, Gk) for (A_, Vr), Gk in zip(resid, G_list)]
    if level == "block":
        den_c = float(np.sum((VG[:, blocks[0]].T @ Gsum) * VG[:, blocks[0]].T)); Dv[blocks[0]] = np.sqrt(e_c / max(den_c, 1e-12))
        for i, Gk in enumerate(G_list):
            b = blocks[i + 1]; den = float(np.sum((VG[:, b].T @ Gk) * VG[:, b].T)); Dv[b] = np.sqrt(e_k[i] / max(den, 1e-12))
    elif level == "ct":   # CT-Merging's weight-energy levels, rho_b = sqrt(E_b / n_b) with E_b the squared singular values of the block's source (gamma = 1)
        Dv[blocks[0]] = np.sqrt(float(np.sum(sc ** 2)) / max(n_c, 1))
        for i in range(K): Dv[blocks[i + 1]] = np.sqrt(ew[i] / max(rr_k[i], 1))
    elif level == "global":
        e_tot = sum(tr_q(U * s, V, Gk) for (U, s, V), Gk in zip(F, G_list)); tot = e_c + sum(e_k); f = e_tot / max(tot, 1e-12)
        Dv[blocks[0]] = np.sqrt(e_c / n_c * f)
        for i in range(K): Dv[blocks[i + 1]] = np.sqrt(e_k[i] / max(rr_k[i], 1) * f)
    else: raise ValueError(level)
    if wk is not None:   # per-client level weights on the residual blocks (the consensus block is not weighted)
        for i in range(K): Dv[blocks[i + 1]] *= float(wk[i])
    if factors: return (Ut * Dv).astype(np.float32), VG.T.astype(np.float32)   # B (d_out x m), A (m x d_in) with B A = the endpoint (generative pools: rank <= k_c + K r, no dense 4096^2 cell needed)
    return ((Ut * Dv) @ VG.T).astype(np.float32)
def m_endpoint(cells, factors, norms, grams, gkey, target, level, ridge, kr, whiten=True, as_factors=False, wk=None, metric_mode="gram"):
    """Whole endpoint. cells: list of cell keys; factors[t][c] = (B, A) LoRA factors (scale inside B) or a dense update; norms[t][c] = ||D_t[c]||_F; grams[t][key] = shrunk Gram (float32/64);
    gkey(c) -> Gram key; target 'median' | 'min' (ignored when whiten is False); level, ridge, kr = (k_c, r_res). Returns {cell: float32 endpoint}."""
    tasks = list(factors); K = len(tasks); out = {}; cache = {}
    for c in cells:
        gk = gkey(c)
        if gk not in cache:
            Gl = [np.asarray(grams[t][gk], dtype=np.float64) for t in tasks]; Gs = sum(Gl); cache[gk] = (Gl, Gs, gram_metric(Gs, ridge))
        Gl, Gs, met = cache[gk]; v = np.array([norms[t][c] for t in tasks]); tg = float(np.median(v)) if target == "median" else float(np.min(v))
        F = []
        for t in tasks:
            w = (tg / max(norms[t][c], 1e-12)) if whiten else 1.0; f = factors[t][c]
            F.append(compact_svd(f[0], f[1], w) if isinstance(f, (tuple, list)) else dense_svd(w * np.asarray(f, dtype=np.float64), max(kr[1] + 1, 16)))
        out[c] = m_cell(F, Gl, Gs, met, kr[0], kr[1], level, factors=as_factors, wk=(None if wk is None else [float(wk[t]) for t in tasks]), metric_mode=metric_mode)
    return out
