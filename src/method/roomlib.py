"""Helpers shared by the baseline runners: the published merges of a pool (rebuilt from the client updates), calibration and test arrays, client accuracies, retention. Import after the harness module."""
import os, numpy as np
MIX = [(0.25, "ou25"), (0.5, "ou50"), (0.625, "ou625"), (0.75, "ou75"), (0.875, "ou875")]
class Lib:
    def __init__(self, H, ROOT, kind, ops, merges_ext):
        self.H, self.ROOT, self.kind, self.ops, self.ops_new = H, ROOT, kind, ops, merges_ext
        self.ARCH = H.ARCH; self.SUB = H.SUBTAG; self.UA = os.path.join(ROOT, "results", "unionadopt", kind, self.ARCH, self.SUB); self.E = os.path.join(ROOT, "results", "clipclose", "ext", kind, self.ARCH, self.SUB)
        self.W = H.grid(); self.U = ["isoc"] + [n for _, n in MIX] + ["regmean128"]; self.END = H.end(); self._PD = None; self.RET = H.RET if hasattr(H, "RET") else H.TASKS
        self.home = {t: self.acc_of(f"client_{t}")[t] for t in H.TASKS}
    def a_of(self, m): return self.H.a_of(m) if m in self.W else (0.0 if m == "isoc" else dict((n, a) for a, n in MIX).get(m, float("nan")))
    def is_scaled_union(self, m): return "_x" in m and m.rsplit("_x", 1)[0] in self.W + self.U and m.rsplit("_x", 1)[1].isdigit()
    def acc_of(self, m):
        for d in (os.path.join(self.UA, "eval"), self.H.EVAL, os.path.join(self.E, "eval")):
            p = os.path.join(d, m + ".json")
            if os.path.exists(p): return self.H.jload(p)
        raise FileNotFoundError(m)
    def R_of(self, acc): return float(np.mean([acc[t] / max(self.home[t], 1e-9) for t in self.RET]))
    def pool_deltas(self):
        if self._PD is None: pool, D = self.H.load_deltas(); self._PD = (pool, D)
        return self._PD
    def npz(self, p): z = np.load(p); return {c: z[self.H.cname(c)].astype(np.float32) for c in self.H.CELLS}
    def f32(self, m): return {c: np.asarray(m[c], dtype=np.float32) for c in self.H.CELLS}
    def load(self, m):
        H = self.H
        for p in (os.path.join(self.UA, "merges", m + ".npz"), H.mpath(m)):
            if os.path.exists(p): return self.npz(p)
        if m in self.W and 0 < self.a_of(m) < 1: RM, IW = self.load("regmean128"), self.load(self.END); return H.truncate({c: self.a_of(m) * RM[c] + (1 - self.a_of(m)) * IW[c] for c in H.CELLS}, H.RANK)[0]
        if m in self.U and 0 < self.a_of(m) < 1: RM, IS = self.load("regmean128"), self.load("isoc"); return H.truncate({c: self.a_of(m) * RM[c] + (1 - self.a_of(m)) * IS[c] for c in H.CELLS}, H.RANK)[0]
        raise FileNotFoundError(m)
    def delta_of(self, m):
        if self.is_scaled_union(m): B, s = m.rsplit("_x", 1); d = self.load(B); return {c: (int(s) / 100) * d[c] for c in self.H.CELLS}
        return self.load(m)
    def rebuild_pub(self, m, W0=None):
        H, ops = self.H, self.ops; pool, D = self.pool_deltas(); deltas = [D[t] for t in H.TASKS]
        if m == "ta": return self.f32(ops.ta(deltas))
        if m == "fedavg_w": sizes = H.jload(os.path.join(H.CACHE, "sizes.json")); tot = sum(sizes[t] for t in H.TASKS); return {c: sum(D[t][c] * (sizes[t] / tot) for t in H.TASKS).astype(np.float32) for c in H.CELLS}
        if m == "fedit": return {c: (np.mean([pool[t][c]["B"] for t in H.TASKS], axis=0) @ np.mean([pool[t][c]["A"] for t in H.TASKS], axis=0)).astype(np.float32) for c in H.CELLS}
        if m == "ties": return self.f32(ops.ties(deltas))
        if m == "knots_ties": return self.f32(ops.knots_ties(deltas))
        if m == "tsvm": return self.f32(ops.tsvm(deltas))
        if m == "dcmerge": return self.f32(ops.dcmerge(deltas))
        if m == "isoc": return self.f32(ops.isoc(deltas))
        if m == "isocts": return self.f32(ops.isocts(deltas))
        if m == "ct": return self.f32(self.ops_new.ct_merging(deltas, self.ARCH, gamma=1.0))
        if m == "pact_isoc":
            assert W0 is not None, "pact needs W0"; As = [{c: pool[t][c]["A"] for c in H.CELLS} for t in H.TASKS]; return self.f32(self.ops_new.pact_isoc(deltas, As, W0))
        raise FileNotFoundError(m)
