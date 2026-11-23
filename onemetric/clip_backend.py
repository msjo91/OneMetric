"""CLIP backbone (ViT-B/32, B/16, L/14): the model, the LoRA client adapters, zero-shot heads from text templates, image features.
Env ARCH = B32 | B16 | L14. POOL = dc16 (DC-Merge tasks and the eight new tasks; adapters under external/dcmerge_ckpt and results/clipdom/adapters) | knots8 (KnOTS tasks; adapters from the Hugging Face hub)."""
from onemetric.paths import ROOT as _OM_ROOT
import os, sys, json, re, time, numpy as np, torch
from huggingface_hub import snapshot_download
from transformers import CLIPModel, CLIPTokenizer
ROOT = _OM_ROOT; CACHE = f"{ROOT}/results/repair/cache"; CB = f"{ROOT}/external/clip_benchmark"
ARCH = os.environ.get("ARCH", "B32")   # B32 (default) or L14; selects model, cells, adapter repos and the results root
MODEL = os.environ.get("BASE_MODEL") or {"B32": "openai/clip-vit-base-patch32", "B16": "openai/clip-vit-base-patch16", "L14": "openai/clip-vit-large-patch14"}[ARCH];   # BASE_MODEL: alternative pretraining of the same architecture (adapter re-basing)
NLAYERS = {"B32": 12, "B16": 12, "L14": 24}[ARCH]; ADAPTER_ARCH = {"B32": "ViT-B-32", "B16": "ViT-B-16", "L14": "ViT-L-14"}[ARCH]
POOL = os.environ.get("POOL", "knots8")   # knots8 (default) or dc16 (DC-Merge 16-task ViT-B/32 pool, local adapters)
RES = f"{ROOT}/results/repair" + {"B32": "", "B16": "_b16", "L14": "_l14"}[ARCH] + ("" if POOL == "knots8" else "_dc16") + os.environ.get("RES_SUFFIX", ""); SUB = int(os.environ.get("SUB", "0"))   # SUB > 0: fixed random subset of each cached set (seed 0)
TASKS8 = ["stanford_cars", "dtd", "eurosat", "gtsrb", "mnist", "resisc45", "sun397", "svhn"]
TASKS16 = TASKS8 + ["cifar100", "flowers102", "oxfordpets", "stl10", "sst2", "fashionmnist", "fer2013", "cifar10"]
TASKS = TASKS8 if POOL == "knots8" else TASKS16
CACHE_NAME = {"flowers102": "flowers102full", "oxfordpets": "pets", "sst2": "renderedsst2"}   # task -> cache basename where they differ
EXCLUDE_CONTROL = set() if POOL == "knots8" else {"cifar10", "flowers102", "cifar100", "stl10", "pets"}   # merged tasks cannot be controls
EXCLUDE_RET = set(filter(None, os.environ.get("EXCLUDE_RET", "").split(",")))   # clients kept in the merge but left out of the retention mean (fer2013 on the dc16 pool: its adapter does not work on our test set)
RET_TASKS = [t for t in TASKS if t not in EXCLUDE_RET]
DC_ADAPTERS = f"{ROOT}/external/dcmerge_ckpt/lora_adapters"
KNOTS_FT = {"stanford_cars": 74.0, "dtd": 58.3, "eurosat": 99.0, "gtsrb": 92.7, "mnist": 99.3, "resisc45": 88.4, "sun397": 64.5, "svhn": 96.2}   # KnOTS ViT-B/32 individual test acc
KNOTS_FT_L14 = {"stanford_cars": 99.77, "dtd": 70.05, "eurosat": 98.59, "gtsrb": 97.20, "mnist": 99.53, "resisc45": 95.70, "sun397": 79.60, "svhn": 97.72}   # as listed by DC-Merge's repo
DC16_FT = {"stanford_cars": 76.61, "dtd": 67.34, "eurosat": 98.19, "gtsrb": 98.29, "mnist": 99.15, "resisc45": 93.97, "sun397": 72.49, "svhn": 96.50, "oxfordpets": 90.94, "cifar100": 86.98, "stl10": 98.13, "flowers102": 88.17, "sst2": 72.615, "fashionmnist": 94.41, "fer2013": 69.27, "cifar10": 96.96}   # DC-Merge's own ViT-B/32 adapters
DC16_FT_B16 = {"stanford_cars": 82.28, "dtd": 71.54, "eurosat": 98.59, "gtsrb": 98.22, "mnist": 99.44, "resisc45": 95.62, "sun397": 75.05, "svhn": 96.94, "oxfordpets": 93.87, "cifar100": 88.675, "stl10": 99.14, "flowers102": 89.74, "sst2": 75.22, "fashionmnist": 95.06, "fer2013": 69.04, "cifar10": 97.325}
KNOTS_T = {
 "stanford_cars": ["a photo of a {}.", "a photo of the {}.", "a photo of my {}.", "i love my {}!", "a photo of my dirty {}.", "a photo of my clean {}.", "a photo of my new {}.", "a photo of my old {}."],
 "dtd": ["a photo of a {} texture.", "a photo of a {} pattern.", "a photo of a {} thing.", "a photo of a {} object.", "a photo of the {} texture.", "a photo of the {} pattern.", "a photo of the {} thing.", "a photo of the {} object."],
 "gtsrb": ['a zoomed in photo of a "{}" traffic sign.', 'a centered photo of a "{}" traffic sign.', 'a close up photo of a "{}" traffic sign.'],
 "mnist": ['a photo of the number: "{}".'], "svhn": ['a photo of the number: "{}".'],
 "eurosat": ["a centered satellite photo of {}.", "a centered satellite photo of a {}.", "a centered satellite photo of the {}."],
 "resisc45": [(a + " " + b + " {}.").replace("  ", " ") for a in ("satellite imagery of", "aerial imagery of", "satellite photo of", "aerial photo of", "satellite view of", "aerial view of") for b in ("", "a", "the")],
 "sun397": ["a photo of a {}.", "a photo of the {}."]}
if POOL == "dc16": KNOTS_T.update(json.load(open(f"{CB}/dcmerge_templates.json")))   # DC-Merge's templates for its eight extra tasks
NEWTASKS = ["caltech101", "aircraft", "tinyimagenet", "pcam", "clevr_count", "dsprites_orientation", "smallnorb_azimuth", "dmlab"]   # tasks outside DC-Merge's pool; adapters trained by train_clip_client.py under the light recipe
if POOL == "dc16" and os.path.exists(f"{CB}/newtask_templates.json"): KNOTS_T.update(json.load(open(f"{CB}/newtask_templates.json")))
CB_T = json.load(open(f"{CB}/en_zeroshot_classification_templates.json"))
def cb_templates(name):
    if name not in CB_T: return ["a photo of a {}."]
    return [t.replace("{c}", "{}") for t in CB_T[name]]
def suite():
    return json.load(open(f"{CACHE}/suite.json"))
def sub_idx(n): return np.sort(np.random.default_rng(0).choice(n, SUB, replace=False)) if SUB and n > SUB else None
VAL_NAME = {"oxfordpets": "pets", "sst2": "renderedsst2"}   # validation caches keep the task name except these two
def val_file(task, suffix): return f"{CACHE}/{VAL_NAME.get(task, task)}_val_{suffix}.npy"
def load_set(name):
    name = CACHE_NAME.get(name, name)
    X = np.load(f"{CACHE}/{name}_X.npy", mmap_mode="r"); y = np.load(f"{CACHE}/{name}_y.npy"); names = json.load(open(f"{CACHE}/{name}_names.json"))
    idx = sub_idx(len(y))
    if idx is not None: X = np.ascontiguousarray(X[idx]); y = y[idx]
    return X, y, names
KEY = re.compile(r"encoder\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|out_proj)\.lora_(A|B)\.weight")
CELLS = [(l, p) for l in range(NLAYERS) for p in ("q_proj", "k_proj", "v_proj", "out_proj")]
def load_adapter(task, arch=None):
    arch = arch or ADAPTER_ARCH
    p = (f"{ROOT}/results/clipdom/adapters/{ARCH}/light/{task}" if task in NEWTASKS else f"{DC_ADAPTERS}/{arch}/{task}") if POOL == "dc16" else snapshot_download(f"hoffman-lab/KnOTS-{arch}_lora_R16_{task}"); cfg = json.load(open(f"{p}/adapter_config.json")); scale = cfg["lora_alpha"] / cfg["r"]
    sd = torch.load(f"{p}/adapter_model.bin", map_location="cpu", weights_only=True); cells = {}
    for k, v in sd.items():
        m = KEY.search(k)
        if not m: continue
        arr = v.float().numpy().astype(np.float64); arr = arr * scale if m.group(3) == "B" else arr
        cells.setdefault((int(m.group(1)), m.group(2)), {})[m.group(3)] = arr
    assert len(cells) == len(CELLS), (task, len(cells)); return cells
def load_pool(arch=None): return {t: load_adapter(t, arch) for t in TASKS}
class Clip:
    def __init__(self, device="cuda:0", name=None):
        name = name or MODEL
        self.dev = torch.device(device); self.model = CLIPModel.from_pretrained(name, dtype=torch.float16).to(self.dev).eval(); self.tok = CLIPTokenizer.from_pretrained(name)
        self.W0 = {c: self._param(c).detach().clone() for c in CELLS}
        try:
            from transformers import CLIPImageProcessor; _pp = CLIPImageProcessor.from_pretrained(name); _mean, _std = list(_pp.image_mean), list(_pp.image_std)
        except Exception: _mean, _std = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
        self.MEAN = torch.tensor(_mean, device=self.dev).view(1, 3, 1, 1); self.STD = torch.tensor(_std, device=self.dev).view(1, 3, 1, 1)
        self._heads = {}
    def _param(self, c): l, p = c; return getattr(self.model.vision_model.encoder.layers[l].self_attn, p).weight
    def set_delta(self, delta):
        """delta: None (base) or {cell: {"A","B"}} or {cell: ndarray dW}."""
        with torch.no_grad():
            for c in CELLS:
                w = self.W0[c].clone()
                if delta is not None:
                    d = delta[c]; d = (d["B"] @ d["A"]) if isinstance(d, dict) else d
                    w += torch.as_tensor(np.ascontiguousarray(d), dtype=torch.float32, device=self.dev).to(torch.float16)
                self._param(c).copy_(w)
    @torch.no_grad()
    def head(self, names, templates):
        key = (tuple(names), tuple(templates))
        if key in self._heads: return self._heads[key]
        W = []
        for n in names:
            te = self.tok([t.format(n) for t in templates], padding=True, truncation=True, return_tensors="pt").to(self.dev)
            e = self.model.get_text_features(**te).float(); e = e / e.norm(dim=-1, keepdim=True); e = e.mean(0); W.append(e / e.norm())
        H = torch.stack(W).half(); self._heads[key] = H; return H
    @torch.no_grad()
    def features(self, X, bs=512):
        out = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + bs])).to(self.dev).permute(0, 3, 1, 2).float().div_(255); xb = ((xb - self.MEAN) / self.STD).half()
            f = self.model.get_image_features(pixel_values=xb); out.append(f / f.norm(dim=-1, keepdim=True))
        return torch.cat(out)
    @torch.no_grad()
    def correct(self, X, y, H):
        f = self.features(X); return ((f @ H.T).argmax(-1).cpu().numpy() == np.asarray(y))
def eval_model(clip, delta, sets, max_n=None):
    """sets: {name: (X, y, names, templates)}. Returns {name: correctness bool array}."""
    clip.set_delta(delta); out = {}
    for name, (X, y, names, templates) in sets.items():
        H = clip.head(names, templates); n = len(y) if max_n is None else min(max_n, len(y))
        out[name] = clip.correct(X[:n], y[:n], H)
    return out
def summarize(corr, base_corr=None):
    acc = {k: float(v.mean()) for k, v in corr.items()}
    if base_corr is not None: acc.update({f"{k}/base": float(v.mean() / max(base_corr[k].mean(), 1e-9)) for k, v in corr.items()})
    return acc
def paired_bootstrap(a, b, n_boot=2000, seed=0):
    """mean(a) - mean(b) with item-paired bootstrap CI; a, b bool arrays of equal length."""
    rng = np.random.default_rng(seed); d = a.astype(float) - b.astype(float); idx = rng.integers(0, len(d), (n_boot, len(d)))
    boots = d[idx].mean(1); return float(d.mean()), float(np.quantile(boots, .025)), float(np.quantile(boots, .975))
