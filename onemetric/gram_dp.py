#!/usr/bin/env python
"""Release of a client's Grams under (epsilon, delta)-differential privacy for one image: each image's contribution is scaled to a norm bound, and Gaussian noise of the analytic Gaussian mechanism's standard deviation is added to all Grams together."""
import math, zlib, numpy as np, torch
NKEYS = 24
def _phi(x): return 0.5 * math.erfc(-x / math.sqrt(2.0))
def delta_of(eps, sigma, sens):
    """delta of the Gaussian mechanism with L2 sensitivity sens and noise sigma at epsilon (Balle and Wang 2018, Theorem 8)."""
    a = sens / (2.0 * sigma); b = eps * sigma / sens; return _phi(a - b) - math.exp(eps) * _phi(-a - b)
def sigma_analytic(eps, delta, sens):
    """smallest sigma with delta_of(eps, sigma, sens) <= delta, by bisection."""
    lo, hi = 1e-6 * sens, 1e3 * sens
    assert delta_of(eps, hi, sens) <= delta < delta_of(eps, lo, sens)
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if delta_of(eps, mid, sens) <= delta: hi = mid
        else: lo = mid
    return hi
def _hooks(C, layers, fn):
    hs = []
    for l in layers:
        sa = C.model.vision_model.encoder.layers[l].self_attn
        hs.append(sa.q_proj.register_forward_pre_hook(fn((l, "qkv")))); hs.append(sa.out_proj.register_forward_pre_hook(fn((l, "o"))))
    return hs
@torch.no_grad()
def image_norms(C, delta, X, layers, bs=64):
    """{key: array over images of ||X_z^T X_z||_F}."""
    acc = {}
    def mk(key):
        def hook(mod, inp): x = inp[0].double(); K = x @ x.transpose(1, 2); acc.setdefault(key, []).append(torch.linalg.matrix_norm(K).cpu())
        return hook
    hs = _hooks(C, layers, mk); C.set_delta(delta); C.features(X, bs=bs)
    for h in hs: h.remove()
    return {k: torch.cat(v).numpy() for k, v in acc.items()}
@torch.no_grad()
def clipped_sums(C, delta, X, bounds, layers, bs=64):
    """({key: sum_z c_z X_z^T X_z as float64 numpy}, {key: share of images clipped}, tokens per image)."""
    acc, clipped, cnt, T = {}, {}, {}, []
    def mk(key):
        def hook(mod, inp):
            x = inp[0].double(); K = x @ x.transpose(1, 2); nz = torch.linalg.matrix_norm(K); c = torch.clamp(bounds[key] / nz, max=1.0); xs = (x * c.sqrt()[:, None, None]).reshape(-1, x.shape[-1])
            acc[key] = acc.get(key, 0) + xs.T @ xs; clipped[key] = clipped.get(key, 0) + int((c < 1).sum()); cnt[key] = cnt.get(key, 0) + x.shape[0]
            if not T: T.append(x.shape[1])
        return hook
    hs = _hooks(C, layers, mk); C.set_delta(delta); C.features(X, bs=bs)
    for h in hs: h.remove()
    return {k: acc[k].cpu().numpy() for k in acc}, {k: clipped[k] / cnt[k] for k in acc}, T[0]
_DRAW = int(__import__("os").environ.get("DG_DRAW", "0"))   # DG_DRAW > 0 is a further noise draw (appended to the seed); unset or 0 is the draw of
def sym_noise(d, task, key, eps, n):
    """symmetric d x d matrix, upper triangle (with the diagonal) i.i.d. N(0, 1), mirrored; one fixed draw per client, matrix, epsilon and size."""
    rng = np.random.default_rng([154, zlib.crc32(task.encode()), int(key[0]), 0 if key[1] == "qkv" else 1, int(round(1000 * eps)), int(n)] + ([_DRAW] if _DRAW else [])); A = np.triu(rng.standard_normal((d, d))); return A + np.triu(A, 1).T
def psd(M):
    w, V = np.linalg.eigh((M + M.T) / 2.0); return (V * np.clip(w, 0.0, None)) @ V.T
def release(S, Ck, sigma, task, key, eps, n):
    """the released sum before post-processing: S + C_k sigma E. sigma = 0: the clipped sum itself (no guarantee)."""
    return S if sigma == 0 else S + Ck * sigma * sym_noise(S.shape[0], task, key, eps, n)
def spectral_scale(Ck, sigma, d):
    """about the largest eigenvalue of C_k sigma E (2 sigma sqrt(d) for a symmetric Gaussian matrix): the attacker's ridge."""
    return 2.0 * Ck * sigma * math.sqrt(d)
if __name__ == "__main__":
    for eps in (1.0, 3.0, 8.0):
        s = sigma_analytic(eps, 1e-5, math.sqrt(NKEYS)); print(f"epsilon {eps}: sigma {s:.4f} in units of C_k (sensitivity sqrt(24)); check delta {delta_of(eps, s, math.sqrt(NKEYS)):.3e}; one matrix alone would need {sigma_analytic(eps, 1e-5, 1.0):.4f}")
