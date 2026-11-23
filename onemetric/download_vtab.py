#!/usr/bin/env python
"""DINO line, second attempt: download the six VTAB webdataset mirrors (clip-benchmark/wds_vtab-*), save each as a self-contained
arrow dataset under results/dino2/data/<task>/{train,test} with its classnames.txt, verify it loads, then remove ONLY the transient copies this download created (the hub shard
cache and the datasets arrow cache of these six repos; nothing that existed before). Smallest first. Run ONLINE (offline env vars unset), with python -I. Idempotent."""
from onemetric.paths import ROOT as _OM_ROOT
import os, sys, time, json, shutil, glob
from datasets import load_dataset, load_from_disk, config as dcfg
from huggingface_hub import hf_hub_download, constants as hc
ROOT = _OM_ROOT; DATA = os.path.join(ROOT, "results", "dino2", "data"); LOG = os.path.join(ROOT, "results", "dino2", "logs", "dino2_download.log")
TASKS = [("smallnorb_azimuth", "wds_vtab-smallnorb_label_azimuth"), ("dsprites_orientation", "wds_vtab-dsprites_label_orientation"), ("kitti_distance", "wds_vtab-kitti_closest_vehicle_distance"),
         ("pcam", "wds_vtab-pcam"), ("clevr_count", "wds_vtab-clevr_count_all"), ("dmlab", "wds_vtab-dmlab")]
HUB = hc.HF_HUB_CACHE; DSC = dcfg.HF_DATASETS_CACHE
def log(*a):
    s = time.strftime("%Y-%m-%d %H:%M:%S") + " " + " ".join(str(x) for x in a); print(s, flush=True); open(LOG, "a").write(s + "\n")
def free_gb(): st = os.statvfs("/home"); return st.f_bavail * st.f_frsize / 1e9
def du_gb(p): return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs) / 1e9 if os.path.exists(p) else 0.0
for task, name in TASKS:
    repo = f"clip-benchmark/{name}"; out = os.path.join(DATA, task)
    if all(os.path.exists(os.path.join(out, x)) for x in ("train", "test", "classnames.txt")): log(f"{task}: exists ({du_gb(out):.2f} GB), skip"); continue
    hub_dir = os.path.join(HUB, "datasets--clip-benchmark--" + name); dsc_dir = os.path.join(DSC, "clip-benchmark___" + name)
    pre = {p: os.path.exists(p) for p in (hub_dir, dsc_dir)}; t0 = time.time(); log(f"{task}: downloading {repo} (free {free_gb():.1f} GB; pre-existing {pre})")
    tmp = os.path.join(ROOT, "results", "dino2", "tmp_cache", task); shutil.rmtree(tmp, ignore_errors=True)
    ds = load_dataset("webdataset", data_files={"train": f"hf://datasets/{repo}/train/*.tar", "test": f"hf://datasets/{repo}/test/*.tar"}, cache_dir=tmp)   # explicit shards: kitti's test/ holds a stray text file that breaks format inference
    log(f"{task}: splits {list(ds)} sizes {{{', '.join(f'{k}: {len(v)}' for k, v in ds.items())}}} columns {ds[list(ds)[0]].column_names} ({time.time() - t0:.0f}s; shards {du_gb(hub_dir):.2f} GB, arrow {du_gb(tmp):.2f} GB)")
    assert "train" in ds and "test" in ds, list(ds)
    cn = hf_hub_download(repo, "classnames.txt", repo_type="dataset"); names = open(cn).read().strip().split("\n"); log(f"{task}: {len(names)} classes: {names[:6]}{' ...' if len(names) > 6 else ''}")
    if os.path.exists(out): shutil.rmtree(out)
    ds.save_to_disk(out); shutil.copy(cn, os.path.join(out, "classnames.txt"))
    back = load_from_disk(out); assert all(len(back[s]) == len(ds[s]) for s in ds), "reload mismatch"; ex = back["train"][0]; log(f"{task}: saved {du_gb(out):.2f} GB; reload ok; example keys {list(ex)} label {ex.get('cls')} image {getattr(ex.get([c for c in ex if c in ('webp', 'png', 'jpg', 'jpeg')][0]), 'size', None)}")
    del ds, back
    for p, existed in pre.items():
        if not existed and os.path.exists(p): shutil.rmtree(p); log(f"{task}: removed transient {p}")
    for lk in glob.glob(os.path.join(DSC, f"*clip-benchmark___{name}*.lock")): os.remove(lk)
    shutil.rmtree(tmp, ignore_errors=True)
    log(f"{task}: done ({time.time() - t0:.0f}s; free {free_gb():.1f} GB)")
log(f"ALL DONE; data {du_gb(DATA):.2f} GB; free {free_gb():.1f} GB"); open(os.path.join(ROOT, "results", "dino2", "markers", "DINO2_DL_DONE"), "w").close()
