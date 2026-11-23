"""Datasets for the DINOv2 line and for the new tasks: Hugging Face dataset ids, the six VTAB tasks read from results/<DINO_ROOT>/data (saved once by download_vtab.py), the test draw, and a training dataset."""
from src.paths import ROOT as _OM_ROOT
import os, json, random, numpy as np, torch
from datasets import load_dataset
DS = {"stanford_cars": ("tanganke/stanford_cars", None), "dtd": ("tanganke/dtd", None), "eurosat": ("tanganke/eurosat", None), "gtsrb": ("tanganke/gtsrb", None), "mnist": ("ylecun/mnist", None), "resisc45": ("tanganke/resisc45", None),
      "sun397": ("tanganke/sun397", None), "svhn": ("ufldl-stanford/svhn", "cropped_digits"), "cifar10": ("tanganke/cifar10", None), "cifar100": ("tanganke/cifar100", None), "stl10": ("tanganke/stl10", None), "flowers102": ("dpdl-benchmark/oxford_flowers102", None),
      "oxfordpets": ("timm/oxford-iiit-pet", None), "fashionmnist": ("zalando-datasets/fashion_mnist", None), "sst2": ("clip-benchmark/wds_renderedsst2", None), "fer2013": ("clip-benchmark/wds_fer2013", None),
      "clevr_count": ("clip-benchmark/wds_vtab-clevr_count_all", None), "dmlab": ("clip-benchmark/wds_vtab-dmlab", None), "kitti_distance": ("clip-benchmark/wds_vtab-kitti_closest_vehicle_distance", None),
      "dsprites_orientation": ("clip-benchmark/wds_vtab-dsprites_label_orientation", None), "smallnorb_azimuth": ("clip-benchmark/wds_vtab-smallnorb_label_azimuth", None), "pcam": ("clip-benchmark/wds_vtab-pcam", None),
      "clevr_closest_object_distance": ("clip-benchmark/wds_vtab-clevr_closest_object_distance", None), "dsprites_label_x_position": ("clip-benchmark/wds_vtab-dsprites_label_x_position", None),
      "smallnorb_label_elevation": ("clip-benchmark/wds_vtab-smallnorb_label_elevation", None), "dsprites_label_y_position": ("clip-benchmark/wds_vtab-dsprites_label_y_position", None)}
# dino2 line: the VTAB webdatasets are saved once as self-contained arrow datasets (download_vtab.py) under results/<DINO_ROOT>/data/<task>/{train,test} + classnames.txt and read from there
DATA = os.path.join(_OM_ROOT, "results", os.environ.get("DINO_ROOT", "dino2"), "data")
def split(task, name):
    d = os.path.join(DATA, task, name)
    if os.path.isdir(d):
        from datasets import load_from_disk; return load_from_disk(d)
    repo, cfg = DS[task]; return load_dataset(repo, cfg, split=name) if cfg else load_dataset(repo, split=name)
def cols(task, ds):
    wds = DS[task][0].startswith("clip-benchmark/"); c = ds.column_names
    return ("cls" if wds else "label"), (next(k for k in ("webp", "png", "jpg", "jpeg") if k in c) if wds else "image"), wds
def names_of(task, ds, labcol, wds, cache):
    f = os.path.join(cache, f"{task}_names.json")
    if os.path.exists(f): return json.load(open(f))
    if wds:
        loc = os.path.join(DATA, task, "classnames.txt")
        if os.path.exists(loc): names = open(loc).read().strip().split("\n")
        else:
            from huggingface_hub import hf_hub_download; names = open(hf_hub_download(DS[task][0], "classnames.txt", repo_type="dataset")).read().strip().split("\n")
    else: feat = ds.features[labcol]; names = [nm.replace("_", " ") for nm in (feat.names if hasattr(feat, "names") else [str(i) for i in range(int(max(ds[labcol])) + 1)])]
    os.makedirs(cache, exist_ok=True); json.dump(names, open(f, "w")); return names
def test_draw(task, prep, cache, n=2000):
    xf = os.path.join(cache, f"{task}_test_X.npy")
    if os.path.exists(xf): return np.load(xf, mmap_mode="r"), np.load(xf.replace("_X", "_y"))
    te = split(task, "test"); labcol, imgcol, _ = cols(task, te); idx = sorted(random.Random(0).sample(range(len(te)), min(n, len(te)))); te = te.select(idx)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(32) as ex: X = np.stack(list(ex.map(prep, te[imgcol], chunksize=32)))
    y = np.asarray(te[labcol], dtype=np.int64); os.makedirs(cache, exist_ok=True); np.save(xf, X); np.save(xf.replace("_X", "_y"), y); return X, y
class TrainDS(torch.utils.data.Dataset):
    def __init__(self, ds, imgcol, labcol, prep, idx=None): self.ds, self.imgcol, self.labcol, self.prep, self.idx = ds, imgcol, labcol, prep, idx
    def __len__(self): return len(self.idx) if self.idx is not None else len(self.ds)
    def __getitem__(self, i):
        ex = self.ds[int(self.idx[i]) if self.idx is not None else int(i)]; return torch.from_numpy(self.prep(ex[self.imgcol])), int(ex[self.labcol])
