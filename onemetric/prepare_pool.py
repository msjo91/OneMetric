#!/usr/bin/env python
"""Prepare one pool for the method and the baselines. No merge is built.

For the pool's clients this writes, under the pool's folder:
  predictor.json       the norm statistics of the uploaded updates and the target norm of the norm-equalized side
  select/pseudo.json   each client's pseudo-answers: its own adapter's predictions on its 200 calibration inputs
  eval/client_<t>.json each client's own test accuracy, the denominator of retention

Environment: PP_KIND = dc (CLIP, default) | knots (the KnOTS pool, ARCH only) | siglip | dino, with ARCH / SIG_ARCH / DINO_ARCH and CLIPK_SUB (the pool). The test and calibration draws of the pool's tasks
must exist (the harness stage `data`, and data_newtasks.py for the new tasks). PP_OUT writes to another folder."""
import os
import numpy as np

KIND = os.environ.get("PP_KIND", "dc"); assert KIND in ("dc", "knots", "siglip", "dino"), KIND
if KIND == "siglip": from onemetric import harness_siglip as H
elif KIND == "dino": from onemetric import harness_dino as H
elif KIND == "knots": from onemetric import harness_knots as H
else: from onemetric import harness_clip as H

OUT = os.environ.get("PP_OUT") or H.RA
SEL, EVAL = os.path.join(OUT, "select"), os.path.join(OUT, "eval")
os.makedirs(SEL, exist_ok=True); os.makedirs(EVAL, exist_ok=True)

pool, D = H.load_deltas()
feat = H.features(D); H.jdump(os.path.join(OUT, "predictor.json"), feat)
C = H.clip(); HD = H.heads(C); pseudo = {}
for t in H.TASKS:
    C.set_delta(D[t])
    X, y = H.load_x(t, "cal"); pr = H.predict(C, X, HD[t])
    pseudo[t] = {"pseudo": pr.tolist(), "true_unused": np.asarray(y).tolist(), "agree_true": float((pr == np.asarray(y)).mean())}
    Xt, yt = H.load_x(t, "test"); acc = float((H.predict(C, Xt, HD[t]) == np.asarray(yt)).mean())
    H.jdump(os.path.join(EVAL, f"client_{t}.json"), {t: acc}); H.log(f"  {t}: own test accuracy {acc:.4f}")
H.jdump(os.path.join(SEL, "pseudo.json"), pseudo)
H.log(f"pool files ready: {H.ARCH} {getattr(H, 'SUBTAG', '8task')}, {len(H.TASKS)} clients, target {feat['target']}")
