"""CT-Merging, PACT-Iso-C and ACE-Merging on dense per-matrix updates, written from the papers, with the per-backbone settings the comparison uses."""
import numpy as np
def _svd(d): return np.linalg.svd(np.asarray(d, dtype=np.float64), full_matrices=False)
def _white(X):
    """W(X) = L R^T for X = L S R^T (TSV-style orthogonalisation of the stacked directions)."""
    L, _, Rt = np.linalg.svd(np.asarray(X, dtype=np.float64), full_matrices=False); return L @ Rt
def _qr(X):
    Q, _ = np.linalg.qr(np.asarray(X, dtype=np.float64)); return Q
CT_KR = {"B32": (8, 15), "B16": (8, 15), "L14": (16, 14), "S2B16": (8, 15), "S2L16": (16, 14), "S2SO": (16, 14), "D2B": (8, 15), "D2L": (16, 14), "D3B": (8, 15), "D3L": (16, 14)}   # SigLIP 2: ViT-B setting for base, ViT-L setting for large and So400m
def ct_merging(deltas, arch, gamma=1.0, k=None, r=None, rank=16):
    """CT-Merging: consensus directions from the averaged task projectors (top-k eigenvectors), common right directions from the SVD of the
    averaged projected update, per-task residual directions (leading r left singular vectors with the common part removed), whitening of the stacked
    left and right directions, energy-matched block-isotropic coefficients, Delta = gamma * U~ D V~^T."""
    k0, r0 = CT_KR.get(arch, (8, 15)); k = k or k0; r = r or r0; out = {}
    for c in deltas[0]:
        T = len(deltas); svds = [_svd(d[c]) for d in deltas]
        P = sum(U[:, :rank] @ U[:, :rank].T for U, _, _ in svds) / T                       # Eq. 2
        w, Q = np.linalg.eigh(P); Uc = Q[:, ::-1][:, :k]                                     # Eq. 3: top-k eigenvectors
        M = sum(Uc.T @ np.asarray(d[c], dtype=np.float64) for d in deltas) / T               # Eq. 5: k x in
        Pc, sc, Vct = np.linalg.svd(M, full_matrices=False); Uc = Uc @ Pc; Vc = Vct.T       # Uc <- Uc Pc; common right directions
        Ub, Vb, rho = [Uc], [Vc], [np.sqrt((sc ** 2).sum() / k)]                              # Eq. 12: rho_c
        for U, s, Vt in svds:
            Z = U[:, :r] - Uc @ (Uc.T @ U[:, :r]); a = np.linalg.norm(Z, axis=0)            # Eq. 6-7
            Ubar = Z / np.where(a > 1e-12, a, 1.0); sbar = a * s[:r]
            Ub.append(Ubar); Vb.append(Vt[:r].T); rho.append(np.sqrt((sbar ** 2).sum() / r))  # Eq. 8, 10-11
        Uw, Vw = _white(np.hstack(Ub)), _white(np.hstack(Vb))                                 # Eq. 9
        D = np.concatenate([np.full(k, rho[0])] + [np.full(r, x) for x in rho[1:]])            # Eq. 13
        out[c] = (gamma * (Uw * D) @ Vw.T).astype(np.float32)                                  # Eq. 14
    return out
def pact_filter(deltas, As, W0, K_ratio=0.01, k=4):
    """PACT (LoRA variant, released code lora/pact_isoc.py): pre-trained core V0 = top-K right singular vectors of W0 with K = int(K_ratio * d_in) (7 on ViT-B, 10 on ViT-L); task subspace Vt = top-k right singular vectors of the A factor;
    LBW_t = orth((I - Vt Vt^T) V0); protection basis of task t = orth([LBW_s, s != t]); filtered update = d_t - d_t P P^T."""
    T = len(deltas); out = [dict() for _ in deltas]
    for c in deltas[0]:
        _, _, V0t = _svd(W0[c]); K = max(1, int(K_ratio * W0[c].shape[1])); V0 = V0t[:K].T
        LBW = []
        for A in As:
            _, _, Vat = _svd(A[c]); Vt = Vat[:k].T; LBW.append(_qr(V0 - Vt @ (Vt.T @ V0)))
        for t in range(T):
            P = _qr(np.hstack([LBW[s] for s in range(T) if s != t])); d = np.asarray(deltas[t][c], dtype=np.float64)
            out[t][c] = (d - (d @ P) @ P.T).astype(np.float32)
    return out
def pact_isoc(deltas, As, W0, K_ratio=0.01, k=4):
    import sys, os; from src.baselines import merges as ops
    return ops.isoc(pact_filter(deltas, As, W0, K_ratio, k))

ACE_VIT = {"B32": {"cfg": {"eps": 1e-5, "k_frac": 0.5, "tau": 0.03}, "scale": 1.0}, "B16": {"cfg": {"eps": 1e-5, "k_frac": 0.5, "tau": 0.03}, "scale": 1.0},
           "L14": {"cfg": {"eps": 1e-5, "k_frac": 0.7, "tau": 0.03}, "scale": 0.95},
           "S2B16": {"cfg": {"eps": 1e-5, "k_frac": 0.5, "tau": 0.03}, "scale": 1.0}, "S2L16": {"cfg": {"eps": 1e-5, "k_frac": 0.7, "tau": 0.03}, "scale": 0.95}, "S2SO": {"cfg": {"eps": 1e-5, "k_frac": 0.7, "tau": 0.03}, "scale": 0.95},
           "D2B": {"cfg": {"eps": 1e-5, "k_frac": 0.5, "tau": 0.03}, "scale": 1.0}, "D2L": {"cfg": {"eps": 1e-5, "k_frac": 0.7, "tau": 0.03}, "scale": 0.95}, "D3B": {"cfg": {"eps": 1e-5, "k_frac": 0.5, "tau": 0.03}, "scale": 1.0}, "D3L": {"cfg": {"eps": 1e-5, "k_frac": 0.7, "tau": 0.03}, "scale": 0.95}}   # the released code's ViT configuration (control/config.py + docs reproduction table, read: ViT-B eps 1e-5, k_frac .5, tau .03, scaling 1.0; ViT-L/14 k_frac .7, scaling .95 (paper text: tau .3, k_frac .3)
def ace_merging(deltas, eps=1e-5, k_frac=0.5, tau=0.03, use_cagg=True, refine=True):
    """ACE-Merging exactly as src/merge/strategy.py of the released code: raw traces -> heterogeneity score gamma = var(log tr) / mean(log tr)^2 (unbiased var), flag = gamma > tau;
    per task: column-centre, Sigma = W~^T W~ (trace-normalised with eps/tr when flag), store, ridge eps_t I, accumulate W~ Sigma and Sigma; C_agg = row-mean of the summed
    unregularised Sigmas (a (1, d_in) row broadcast over the rows, as in the code), / mean trace when flag; W = (sum W~ Sigma)(sum Sigma + C_agg)^-1; when flag: spectral
    refinement with Delta_res = sum W_t (Sigma_t - mean Sigma) on the uncentred W_t, SVD of Delta_res + W, k = int(k_frac * min(d)), add mean(s[:k]) U_k V_k^T."""
    out = {}
    for c in deltas[0]:
        Ws = [np.asarray(d[c], dtype=np.float64) for d in deltas]; T = len(Ws); n_in = Ws[0].shape[1]
        traces = np.array([np.trace(W.T @ W) for W in Ws]); lt = np.log(traces + 1e-12)
        gamma = (lt.var(ddof=1) if T > 1 else 0.0) / (lt.mean() ** 2 + 1e-12); flag = bool(gamma > tau); avg_trace = float(traces.mean())
        Sigmas, WS_sum, S_sum = [], 0.0, 0.0
        for W in Ws:
            Wc = W - W.mean(axis=0, keepdims=True); Sraw = Wc.T @ Wc; tr = np.trace(Sraw) + 1e-12
            S, eps_t = (Sraw / tr, eps / tr) if flag else (Sraw, eps)
            Sigmas.append(S); Sreg = S + eps_t * np.eye(n_in); WS_sum = WS_sum + Wc @ Sreg; S_sum = S_sum + Sreg
        C_agg = sum(Sigmas).mean(axis=0, keepdims=True)
        if flag: C_agg = C_agg / (avg_trace + 1e-12)
        A = S_sum + (C_agg if use_cagg else 0.0)   # use_cagg / refine: diagnostic switches (default = the released code)
        try: A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError: A_inv = np.linalg.pinv(A)
        W0m = WS_sum @ A_inv; mv = W0m
        if flag and refine:
            S_mean = S_sum / T; D_res = sum(W @ (S - S_mean) for W, S in zip(Ws, Sigmas)); U, s, Vt = np.linalg.svd(D_res + W0m, full_matrices=False); k = int(len(s) * k_frac)
            if k > 0: mv = mv + s[:k].mean() * (U[:, :k] @ Vt[:k])
        out[c] = mv.astype(np.float32)
    return out



def _flat(M):
    """Iso-C on one matrix: every singular value replaced by their mean (Marczak et al. 2025)."""
    U, s, Vt = _svd(M); return (U * s.mean()) @ Vt
def _tsvm_mats(mats):
    """TSV-M (Gargiulo et al. 2025) on a list of same-shape matrices: per task the top-k singular triplets (k = min(dim) // n), stacks orthogonalised by Procrustes."""
    n = len(mats); Us, Ss, Vs = [], [], []
    for M in mats:
        U, s, Vt = _svd(M); k = max(1, min(len(s), min(M.shape) // n)); Us.append(U[:, :k]); Ss.append(s[:k]); Vs.append(Vt[:k].T)
    def orth(X): P, _, Qt = np.linalg.svd(X, full_matrices=False); return P @ Qt
    return (orth(np.hstack(Us)) * np.concatenate(Ss)) @ orth(np.hstack(Vs)).T
def _core_merge(cores, op, keep=0.3):
    """isoc: Iso-C of the summed cores (basis-invariant, equals plain Iso-C); tsvm_isoc: TSV-M of the cores then Iso-C (basis-equivariant, equals TSV-M + Iso-C in the full space);
    ties_isoc: TIES on the core entries (top-`keep` magnitude, sign election, disjoint mean; the operator that depends on the basis) then Iso-C."""
    if op == "isoc": return _flat(sum(cores))
    if op == "tsvm_isoc": return _flat(_tsvm_mats(cores))
    if op == "ties_isoc":
        from src.baselines import merges as _ops
        return _flat(_ops._ties_combine(cores, keep))
    raise ValueError(op)
def _trim(U, s, tol=1e-10):
    k = int((s > tol * max(s[0], 1e-30)).sum()); return U[:, :max(k, 1)], s[:max(k, 1)]
def corespace(pool_fac, op="isoc", alpha=0.9, keep=0.3):
    """Core Space (Panariello et al. 2025, arXiv 2509.17786): U_ref = left singular vectors of [B_1 ... B_T], V_ref = right singular vectors of [A_1; ...; A_T];
    per-task cores M_t = (U_ref^T B_t)(A_t V_ref); the cores merged by `op` (isoc: Iso-C of the sum; tsvm_isoc: TSV-M then Iso-C, the paper's best); dW = alpha U_ref M* V_ref^T.
    pool_fac = list over tasks of {cell: {"B": (d_out, r), "A": (r, d_in)}} with the LoRA scale folded in. Lossless: span(U_ref) contains every col(B_t)."""
    out = {}
    for c in pool_fac[0]:
        Bs = [np.asarray(f[c]["B"], dtype=np.float64) for f in pool_fac]; As = [np.asarray(f[c]["A"], dtype=np.float64) for f in pool_fac]
        Ub, sb, _ = _svd(np.hstack(Bs)); Ub, _ = _trim(Ub, sb); Vat_full = _svd(np.vstack(As)); Va = _trim(Vat_full[2].T, Vat_full[1])[0]   # (d_in, ka)
        cores = [(Ub.T @ B) @ (A @ Va) for B, A in zip(Bs, As)]
        out[c] = (alpha * (Ub @ _core_merge(cores, op, keep) @ Va.T)).astype(np.float32)
    return out
def pgsb(pool_fac, W0, op="isoc", alpha=0.9, keep=0.3):
    """PGSB (Liu et al. 2026, arXiv 2610.14558), the paper's global form: compact SVDs dW_t = U_t S_t V_t^T; L = [U_t S_t], R = [V_t S_t]; thin SVDs L = U_L S_L (.), R = U_R S_R (.);
    G = U_L^T W0 U_R; T = S_L G S_R = O_L S_T O_R^T; Q_L = U_L O_L, Q_R = U_R O_R; C_t = Q_L^T dW_t Q_R; C* = op(C_t); dW = alpha Q_L C* Q_R^T. W0[cell] = the pre-trained (d_out, d_in) weight.
    The released code additionally restricts the basis per attention head; that variant is not run here."""
    out = {}
    for c in pool_fac[0]:
        Ls, Rs, Ds = [], [], []
        for f in pool_fac:
            B = np.asarray(f[c]["B"], dtype=np.float64); A = np.asarray(f[c]["A"], dtype=np.float64)
            Qb, Rb = np.linalg.qr(B); Qa, Ra = np.linalg.qr(A.T); U, s, Vt = _svd(Rb @ Ra.T); U = Qb @ U; V = Qa @ Vt.T
            Ls.append(U * s); Rs.append(V * s); Ds.append(B @ A)
        UL, sL, _ = _svd(np.hstack(Ls)); UL, sL = _trim(UL, sL); UR, sR, _ = _svd(np.hstack(Rs)); UR, sR = _trim(UR, sR)
        G = UL.T @ np.asarray(W0[c], dtype=np.float64) @ UR; T = (sL[:, None] * G) * sR[None, :]; OL, _, ORt = _svd(T); QL = UL @ OL; QR = UR @ ORt.T
        cores = [QL.T @ D @ QR for D in Ds]
        out[c] = (alpha * (QL @ _core_merge(cores, op, keep) @ QR.T)).astype(np.float32)
    return out
