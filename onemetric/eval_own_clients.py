#!/usr/bin/env python
"""Test accuracy of the zero-shot head and of each of our own new-task adapters, and the adapters' pseudo-answers on the calibration inputs. Writes results/clipnew/own_<backbone>.json and the list of usable new tasks that the new-task pools are drawn from. Env ARCH (or OE_KIND=siglip with SIG_ARCH) and CLIPK_SUB=k8."""
from onemetric.paths import ROOT as _OM_ROOT
import os, sys, json, time, numpy as np
ROOT = _OM_ROOT
os.environ.setdefault("POOL", "dc16"); os.environ.setdefault("CLIPK_SUB", "k8")
KIND = os.environ.get("OE_KIND", "dc")   # OE_KIND=siglip runs the same check on SigLIP 2 (SIG_ARCH), adapters from results/siglip/adapters/<arch>/<task>
if KIND == "siglip": from onemetric import harness_siglip as H; from onemetric import siglip_backend as AD; from onemetric import clip_backend as rep
else: from onemetric import harness_clip as H; from onemetric import clip_backend as rep; AD = rep
C = H.clip(); out = {}; NT = os.environ.get("NT_TASKS", ",".join(rep.NEWTASKS)).split(",")   # NT_TASKS: smoke only
for t in NT:
    head = C.head(H.names_of(t), rep.KNOTS_T[t]); Xt, yt = H.load_x(t, "test"); Xc, yc = H.load_x(t, "cal"); C.set_delta(None); zs = float((H.predict(C, Xt, head) == np.asarray(yt)).mean()); zsc = float((H.predict(C, Xc, head) == np.asarray(yc)).mean())
    pool = AD.load_adapter(t); D = {c: (pool[c]["B"] @ pool[c]["A"]).astype(np.float32) for c in H.CELLS}; C.set_delta(D); own = float((H.predict(C, Xt, head) == np.asarray(yt)).mean()); pr = H.predict(C, Xc, head)
    out[t] = {"zero_shot": zs, "own": own, "gain": own - zs, "classes": len(H.names_of(t)), "zero_shot_cal": zsc, "pseudo": pr.tolist(), "true_unused": np.asarray(yc).tolist(), "agree_true": float((pr == np.asarray(yc)).mean()), "fro": float(np.sqrt(sum(float(np.sum(np.square(D[c], dtype=np.float64))) for c in H.CELLS)))}
    H.log(f"{t}: zero-shot {zs:.4f}, own adapter {own:.4f}, gain {own - zs:+.4f}, classes {out[t]['classes']}, update norm {out[t]['fro']:.2f}")
usable = [t for t in NT if out[t]["gain"] >= .03]; gains = [out[t]["gain"] for t in usable]
H.jdump(os.path.join(ROOT, "results", "clipnew", f"own_{H.ARCH}.json"), out); H.jdump(os.path.join(ROOT, "results", "clipnew", "usable.json" if H.ARCH == "L14" else f"usable_{H.ARCH}.json"), {   # one usable file per backbone; L/14 keeps
        "usable": usable, "rule": "own adapter above the zero-shot head by at least .03 on the test draw", "mean_gain": float(np.mean(gains)) if gains else 0.0, "precondition": bool(len(usable) >= 6 and (np.mean(gains) if gains else 0) >= .05)})
H.log(f"usable {len(usable)} of {len(NT)}: {usable}; mean gain {float(np.mean(gains)) if gains else 0:.4f}")
