#!/usr/bin/env python
"""§2 pre-condition: EXPERTS-OK for one backbone from the trainers' logged accuracies on the n = 2000 draw
(adapter_config.json: zero_shot_acc = frozen-probe accuracy with zero LoRA, test_acc_hist[-1] = final adapter accuracy). Pool-level rule (the probes of a strong
self-supervised backbone already solve several tasks, so a per-task gain of .05 is not available everywhere): (a) adapter above its probe on >= 15/16 tasks, (b) mean gain
>= .03, (c) >= 4 headroom tasks (gain >= .05). Env DINO_ARCH. Writes results/dino/experts_<arch>.{json,txt}; exit 1 on EXPERTS-FAIL or missing adapters (the recipe is then
re-examined before the backbone's pools are built). CPU only; no merge number is read."""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); os.environ.setdefault("POOL", "dc16")
from src.backbones.dino_backend import ROOT, ARCH, TASKS16, ADAPTERS, RES_ROOT
rows, missing = {}, []
for t in TASKS16:
    p = os.path.join(ADAPTERS, t, "adapter_config.json")
    if not os.path.exists(p): missing.append(t); continue
    c = json.load(open(p)); zs, fin = c["zero_shot_acc"], c["test_acc_hist"][-1][1]
    rows[t] = {"zero_shot": zs, "final": fin, "gain": fin - zs, "ok": fin - zs >= .05, "above": fin > zs, "train_s": c.get("train_time_s"), "n_train": c.get("n_train"), "probe_test_acc": c.get("probe_test_acc")}
n_above = sum(r["above"] for r in rows.values()); n_head = sum(r["ok"] for r in rows.values()); mean_gain = sum(r["gain"] for r in rows.values()) / max(len(rows), 1)
ok = (not missing) and n_above >= 15 and mean_gain >= .03 and n_head >= 4
stamp = time.strftime("%Y-%m-%d %H:%M")
L = [f"§2 EXPERTS-OK (pool-level rule over the frozen-probe accuracy) {ARCH} ({stamp}): {len(rows)}/16 adapters" + (f"; MISSING {missing}" if missing else ""), f"{'task':<14}{'probe':>10}{'expert':>8}{'gain':>7}  headroom"]
L += [f"{t:<14}{r['zero_shot']:>10.4f}{r['final']:>8.4f}{r['gain']:>+7.3f}  {'yes' if r['ok'] else 'no'}{'' if r['above'] else '  BELOW PROBE'}" for t, r in rows.items()]
L.append(f"(a) above probe {n_above}/{len(rows)} (bar >= 15)  (b) mean gain {mean_gain:+.3f} (bar >= .03)  (c) headroom tasks (gain >= .05) {n_head} (bar >= 4)  mean train time {sum(r['train_s'] or 0 for r in rows.values()) / max(len(rows), 1) / 60:.1f} min")
L.append("EXPERTS-OK" if ok else "EXPERTS-FAIL")
txt = "\n".join(L); print(txt); os.makedirs(os.path.join(RES_ROOT), exist_ok=True)
json.dump({"arch": ARCH, "rows": rows, "missing": missing, "n_above": n_above, "n_headroom": n_head, "mean_gain": mean_gain, "ok": ok, "stamp": stamp}, open(os.path.join(RES_ROOT, f"experts_{ARCH}.json"), "w"), indent=1)
open(os.path.join(RES_ROOT, f"experts_{ARCH}.txt"), "w").write(txt + "\n"); sys.exit(0 if ok else 1)
