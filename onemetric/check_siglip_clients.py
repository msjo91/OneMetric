#!/usr/bin/env python
"""§2 amendment: pool-level EXPERTS-OK for one backbone — the rule applied to the SigLIP adapters' logged accuracies on the n = 2000 draw (adapter_config.json: zero_shot_acc,
test_acc_hist[-1]): (a) adapter above zero-shot on >= 15/16 tasks, (b) mean gain >= .03, (c) >= 4 tasks with gain >= .05. Env SIG_ARCH. Writes
results/siglip/experts_pool_<arch>.{json,txt}; exit 1 on EXPERTS-FAIL. CPU only; no merge number is read. The chain's per-task check (sig_experts.py) is left unchanged."""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); os.environ.setdefault("POOL", "dc16")
from onemetric.siglip_backend import ROOT, ARCH, TASKS16, ADAPTERS
rows, missing = {}, []
for t in TASKS16:
    p = os.path.join(ADAPTERS, t, "adapter_config.json")
    if not os.path.exists(p): missing.append(t); continue
    c = json.load(open(p)); zs, fin = c["zero_shot_acc"], c["test_acc_hist"][-1][1]
    rows[t] = {"zero_shot": zs, "final": fin, "gain": fin - zs, "above": fin > zs, "headroom": 1.0 - zs, "gain05": fin - zs >= .05, "peak": max(a for _, a in c["test_acc_hist"]), "train_s": c.get("train_time_s")}
n = len(rows); n_above = sum(r["above"] for r in rows.values()); mean_gain = sum(r["gain"] for r in rows.values()) / max(n, 1); n05 = sum(r["gain05"] for r in rows.values())
A, B, C = n_above >= 15, mean_gain >= .03, n05 >= 4; ok = (not missing) and A and B and C
sub = [t for t, r in rows.items() if not r["gain05"]]
stamp = time.strftime("%Y-%m-%d %H:%M")
L = [f"§2 (amended, pool-level) EXPERTS-OK {ARCH} ({stamp}): {n}/16 adapters" + (f"; MISSING {missing}" if missing else ""), f"{'task':<14}{'zero-shot':>10}{'expert':>8}{'gain':>7}{'headroom':>9}{'peak':>8}  gain>=.05"]
L += [f"{t:<14}{r['zero_shot']:>10.4f}{r['final']:>8.4f}{r['gain']:>+7.3f}{r['headroom']:>9.3f}{r['peak']:>8.4f}  {'yes' if r['gain05'] else 'no'}" for t, r in rows.items()]
L.append(f"(a) above zero-shot on {n_above}/16 (>= 15): {'ok' if A else 'NO'}; (b) mean gain {mean_gain:+.3f} (>= .03): {'ok' if B else 'NO'}; (c) tasks with gain >= .05: {n05}/16 (>= 4): {'ok' if C else 'NO'}")
L.append(f"tasks below +.05: {sub} with zero-shot " + ", ".join(f"{t} {rows[t]['zero_shot']:.4f} (headroom {rows[t]['headroom']:.3f})" for t in sub))
L.append("EXPERTS-OK (pool-level)" if ok else "EXPERTS-FAIL (pool-level)")
txt = "\n".join(L); print(txt); os.makedirs(os.path.join(ROOT, "results", "siglip"), exist_ok=True)
json.dump({"arch": ARCH, "rule": "pool-level : above on >= 15/16, mean gain >= .03, >= 4 tasks gain >= .05", "rows": rows, "n_above": n_above, "mean_gain": mean_gain, "n_gain05": n05, "sub05": sub, "missing": missing, "ok": ok, "stamp": stamp},
          open(os.path.join(ROOT, "results", "siglip", f"experts_pool_{ARCH}.json"), "w"), indent=1)
open(os.path.join(ROOT, "results", "siglip", f"experts_pool_{ARCH}.txt"), "w").write(txt + "\n"); sys.exit(0 if ok else 1)
