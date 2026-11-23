"""The spectral end on random clients: rank, orthonormal left directions, right directions orthonormal in the input metric."""
import numpy as np
from onemetric import spectral_end as S


def make(K=8, d_out=256, d_in=192, r=16, seed=0):
    rng = np.random.default_rng(seed)
    F = [S.compact_svd(rng.normal(size=(d_out, r)), rng.normal(size=(r, d_in))) for _ in range(K)]
    G = []
    for _ in range(K):
        X = rng.normal(size=(d_in, 200)) * rng.uniform(0.5, 2.0, size=(d_in, 1))
        G.append(X @ X.T / 200)
    return F, G


def test_spectral_end_cell():
    K, k_c, r_c, ridge = 8, 8, 15, 1e-3
    F, G = make(K)
    Gsum = sum(G)
    B, A = S.m_cell(F, G, Gsum, S.gram_metric(Gsum, ridge), k_c, r_c, "block", factors=True)
    B, A = B.astype(np.float64), A.astype(np.float64)
    m = k_c + K * r_c
    assert B.shape[1] == A.shape[0] == m                                  # rank 8 + 15 K
    levels = np.linalg.norm(B, axis=0)
    U = B / levels
    assert np.allclose(U.T @ U, np.eye(m), atol=1e-4)                     # left directions orthonormal
    Ghat = Gsum + ridge * np.trace(Gsum) / Gsum.shape[0] * np.eye(Gsum.shape[0])
    assert np.allclose(A @ Ghat @ A.T, np.eye(m), atol=1e-3)              # right directions orthonormal in the input metric
    assert np.all(levels > 0)


def test_block_energy_matches_source():
    K, k_c, r_c = 8, 8, 15
    F, G = make(K, seed=1)
    Gsum = sum(G)
    B, A = (x.astype(np.float64) for x in S.m_cell(F, G, Gsum, S.gram_metric(Gsum, 1e-3), k_c, r_c, "block", factors=True))
    Uc = np.linalg.svd(np.hstack([U for U, _, _ in F]), full_matrices=False)[0][:, :k_c]
    for k, (U, s, V) in enumerate(F):                                      # a residual block responds on its own client's inputs as its source does
        b = slice(k_c + k * r_c, k_c + (k + 1) * r_c)
        Z = U[:, :r_c] - Uc @ (Uc.T @ U[:, :r_c])
        source = S.tr_q(Z * s[:r_c], V[:, :r_c], G[k])
        got = float(np.sum((B[:, b].T @ B[:, b]) * (A[b] @ G[k] @ A[b].T)))
        assert abs(got - source) <= 2e-3 * source
