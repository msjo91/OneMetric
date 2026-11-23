"""Gaussian noise of a given relative Frobenius norm on an uploaded Gram, symmetrized and projected back to positive semi-definite; the seed comes from the task, the cell and the draw."""
import zlib, numpy as np
def noisy(G, rho, task, key, draw=0):   # draw > 0 is a further noise draw (its index is appended to the seed); draw 0 is the draw of
    if not rho: return G
    G = np.asarray(G, dtype=np.float64); d = G.shape[0]; rng = np.random.default_rng([147, zlib.crc32(task.encode()), int(key[0]), {"qkv": 0, "o": 1}[key[1]], int(round(1000 * rho))] + ([int(draw)] if draw else []))
    A = rng.standard_normal((d, d)); E = (A + A.T) / np.sqrt(2.0); N = E * (rho * np.linalg.norm(G) / np.linalg.norm(E))
    w, V = np.linalg.eigh(G + N); return (V * np.clip(w, 0.0, None)) @ V.T
def spectral_scale(G, rho): G = np.asarray(G); return rho * float(np.linalg.norm(G)) * 2.0 / np.sqrt(G.shape[0])   # about the spectral norm of the noise; a receiver who knows rho regularises with it
