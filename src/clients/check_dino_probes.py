#!/usr/bin/env python
"""dino2 line, Step A PROBE-HEADROOM: from the 16 frozen probes of one backbone (results/<DINO_ROOT>/probes/<ARCH>/<task>.npz, test_acc on the
n = 2000 draw), before any adapter is trained: probe accuracy <= .90 on >= 8/16 tasks AND <= .85 on >= 5 of the six VTAB tasks. Writes results/<DINO_ROOT>/probes_check_<ARCH>.{json,txt};
exit 1 on failure or missing probes (the chain then blocks the backbone's adapters and pools; the substitution of a VTAB task above .85 by the next of dino.SUBST is applied by hand in that
order, its probe trained, and the check re-run once). Env DINO_ARCH, DINO_ROOT, DINO_TASKS. Reads probe accuracies only; no merge number exists yet."""
import os, sys, json, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); os.environ.setdefault("POOL", "dc16")
from src.backbones.dino_backend import ARCH, TASKS16, VTAB6, SUBST, RES_ROOT, probe_path
acc, missing = {}, []
for t in TASKS16:
    p = probe_path(t)
    if not os.path.exists(p): missing.append(t); continue
    z = np.load(p); acc[t] = float(z["test_acc"])
vt = [t for t in TASKS16 if t in VTAB6 or t in SUBST]
n_le90 = sum(a <= .90 for a in acc.values()); n_vt85 = sum(acc[t] <= .85 for t in vt if t in acc); over = [t for t in vt if t in acc and acc[t] > .85]
ok = (not missing) and n_le90 >= 8 and n_vt85 >= 5
stamp = time.strftime("%Y-%m-%d %H:%M")
L = [f"dino2 Step A PROBE-HEADROOM {ARCH} ({stamp}): {len(acc)}/16 probes" + (f"; MISSING {missing}" if missing else ""), f"{'task':<24}{'probe acc':>10}  <=.90  VTAB<=.85"]
L += [f"{t:<24}{acc[t]:>10.4f}  {'yes' if acc[t] <= .90 else 'no':>5}  {('yes' if acc[t] <= .85 else 'NO') if t in vt else '-':>9}" for t in TASKS16 if t in acc]
L.append(f"<= .90 on {n_le90}/16 (bar >= 8); VTAB <= .85 on {n_vt85}/{len(vt)} (bar >= 5)" + (f"; VTAB above .85: {over} -> substitutes in order {SUBST}" if over else ""))
L.append("PROBE-HEADROOM-OK" if ok else "PROBE-HEADROOM-FAIL")
txt = "\n".join(L); print(txt); os.makedirs(RES_ROOT, exist_ok=True)
json.dump({"arch": ARCH, "tasks": TASKS16, "acc": acc, "missing": missing, "n_le90": n_le90, "n_vtab_le85": n_vt85, "vtab_over_85": over, "ok": ok, "stamp": stamp}, open(os.path.join(RES_ROOT, f"probes_check_{ARCH}.json"), "w"), indent=1)
open(os.path.join(RES_ROOT, f"probes_check_{ARCH}.txt"), "w").write(txt + "\n"); sys.exit(0 if ok else 1)
