#!/usr/bin/env python
"""the zero-shot templates of the eight new tasks, taken unchanged from CLIP Benchmark's published list. Writes external/clip_benchmark/newtask_templates.json."""
from src.paths import ROOT as _OM_ROOT
import json, os
ROOT = _OM_ROOT; CB = os.path.join(ROOT, "external", "clip_benchmark"); T = json.load(open(os.path.join(CB, "en_zeroshot_classification_templates.json")))
M = {"caltech101": "caltech101", "aircraft": "fgvc_aircraft", "tinyimagenet": "imagenet1k", "pcam": "pcam", "clevr_count": "clevr_count_all", "dsprites_orientation": "dsprites_label_orientation", "smallnorb_azimuth": "smallnorb_label_azimuth", "dmlab": "dmlab"}
out = {k: [x.replace("{c}", "{}") for x in T[v]] for k, v in M.items()}; json.dump(out, open(os.path.join(CB, "newtask_templates.json"), "w"), indent=1); print({k: (len(v), v[0]) for k, v in out.items()})
