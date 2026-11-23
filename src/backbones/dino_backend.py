"""DINOv2 backbone (B/14 and L/14) with the interface of clip_backend. A task's head is a linear probe on the frozen features (class token, mean patch token, and a constant), fit once and kept fixed.
Env DINO_ARCH = D2B | D2L. DINO_ROOT is the results sub-folder (default dino2) and DINO_TASKS the task list (default h16: ten DC-Merge tasks and six VTAB tasks)."""
import os, re, sys, json, numpy as np, torch
os.environ.setdefault("POOL", "dc16")
HERE = os.path.dirname(os.path.abspath(__file__))
from src.backbones.clip_backend import ROOT, TASKS8, TASKS16, KNOTS_T
ARCH = os.environ.get("DINO_ARCH", "D2B")
# dino2 line: DINO_ROOT = results sub-folder (default "dino" = untouched; "dino2" = this line); DINO_TASKS = dc16 (DC-Merge 16)
# | h16 (DINO-16H below) | an explicit comma-separated list of 16 tasks (Step-A substitution, applied by hand in the pre-stated order).
DINO_ROOT = os.environ.get("DINO_ROOT", "dino2"); RES_ROOT = os.path.join(ROOT, "results", DINO_ROOT)
TASKS16_H = ["stanford_cars", "dtd", "eurosat", "gtsrb", "resisc45", "sun397", "svhn", "cifar100", "fashionmnist", "fer2013", "clevr_count", "dmlab", "kitti_distance", "dsprites_orientation", "smallnorb_azimuth", "pcam"]
VTAB6 = ["clevr_count", "dmlab", "kitti_distance", "dsprites_orientation", "smallnorb_azimuth", "pcam"]; SUBST = ["clevr_closest_object_distance", "dsprites_label_x_position", "smallnorb_label_elevation", "dsprites_label_y_position"]
_T = os.environ.get("DINO_TASKS", "h16")
if _T == "h16": TASKS16 = list(TASKS16_H)
elif _T != "dc16": TASKS16 = _T.split(","); assert len(TASKS16) == 16, TASKS16
# name, layers, hidden, attribute paths: (layer list path, q, k, v, o) relative to the model
MODELS = {"D2B": ("facebook/dinov2-base", 12, 768, "dinov2"), "D2L": ("facebook/dinov2-large", 24, 1024, "dinov2"),
          "D3B": ("facebook/dinov3-vitb16-pretrain-lvd1689m", 12, 768, "dinov3"), "D3L": ("facebook/dinov3-vitl16-pretrain-lvd1689m", 24, 1024, "dinov3")}
MODEL, NLAYERS, HIDDEN, FAM = MODELS[ARCH]; RES = 224
PROJ = {"dinov2": ("query", "key", "value", "dense"), "dinov3": ("q_proj", "k_proj", "v_proj", "o_proj")}[FAM]
CELLS = [(l, p) for l in range(NLAYERS) for p in PROJ]; QKV, OUT = PROJ[:3], PROJ[3]
KEY = re.compile(r"layer\.(\d+)\.(?:attention\.attention\.|attention\.output\.|attention\.)(query|key|value|dense|q_proj|k_proj|v_proj|o_proj)\.lora_(A|B)\.weight")
ADAPTERS = os.path.join(RES_ROOT, "adapters", ARCH); PROBES = os.path.join(RES_ROOT, "probes", ARCH); DC16_FT, DC16_FT_B16 = {}, {}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
def layers(model): return model.encoder.layer if FAM == "dinov2" else model.layer
def linear(model, c):
    l, p = c; L = layers(model)[l]
    if FAM == "dinov2": return L.attention.output.dense if p == "dense" else getattr(L.attention.attention, p)
    return getattr(L.attention, p)
def lora_key(c, ab):
    l, p = c
    if FAM == "dinov2": return f"base_model.model.encoder.layer.{l}." + ("attention.output.dense" if p == "dense" else f"attention.attention.{p}") + f".lora_{ab}.weight"
    return f"base_model.model.layer.{l}.attention.{p}.lora_{ab}.weight"
def load_adapter(task, arch=None):
    p = os.path.join(ADAPTERS, task); cfg = json.load(open(os.path.join(p, "adapter_config.json"))); scale = cfg["lora_alpha"] / cfg["r"]
    sd = torch.load(os.path.join(p, "adapter_model.bin"), map_location="cpu", weights_only=True); cells = {}
    for k, v in sd.items():
        m = KEY.search(k)
        if not m: continue
        arr = v.float().numpy().astype(np.float64); arr = arr * scale if m.group(3) == "B" else arr
        cells.setdefault((int(m.group(1)), m.group(2)), {})[m.group(3)] = arr
    assert len(cells) == len(CELLS), (task, len(cells)); return cells
def load_pool(arch=None): return {t: load_adapter(t) for t in TASKS16}
def prep(img):
    """uint8 HWC 224: resize shorter side to 256 (bicubic), centre crop 224 (the DINOv2 processor)."""
    from PIL import Image
    img = img.convert("RGB"); w, h = img.size; s = 256 / min(w, h); img = img.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BICUBIC)
    w, h = img.size; l, t = (w - 224) // 2, (h - 224) // 2; return np.array(img.crop((l, t, l + 224, t + 224)), dtype=np.uint8)
def probe_path(task): return os.path.join(PROBES, task + ".npz")
class Clip:
    """Same API as clip_backend.Clip. features(X) -> [CLS, mean patch, 1] (un-normalised); head(names, templates) -> the frozen probe of the task whose class names are `names`
    (probes are keyed by their class-name tuple, written by train_dino_probe.py); head_for_task(t) is the explicit form."""
    def __init__(self, device="cuda:0", name=None, dtype=None):
        from transformers import AutoModel
        name = name or MODEL; self.dev = torch.device(device); self.dtype = dtype or (torch.float16 if self.dev.type == "cuda" else torch.float32)
        self.model = AutoModel.from_pretrained(name, dtype=self.dtype).to(self.dev).eval()
        self.W0 = {c: self._param(c).detach().clone() for c in CELLS}
        self.MEAN = torch.tensor(MEAN, device=self.dev).view(1, 3, 1, 1); self.STD = torch.tensor(STD, device=self.dev).view(1, 3, 1, 1); self._heads = {}; self._by_names = None
    def _param(self, c): return linear(self.model, c).weight
    def set_delta(self, delta):
        with torch.no_grad():
            for c in CELLS:
                w = self.W0[c].clone()
                if delta is not None:
                    d = delta[c]; d = (d["B"] @ d["A"]) if isinstance(d, dict) else d
                    w += torch.as_tensor(np.ascontiguousarray(d), dtype=torch.float32, device=self.dev).to(self.dtype)
                self._param(c).copy_(w)
    def head_for_task(self, task):
        if task not in self._heads:
            z = np.load(probe_path(task)); H = np.concatenate([z["W"], z["b"][:, None]], axis=1); self._heads[task] = torch.as_tensor(H, dtype=self.dtype, device=self.dev)
        return self._heads[task]
    def head(self, names, templates=None):
        if self._by_names is None:
            self._by_names = {}
            for f in os.listdir(PROBES) if os.path.isdir(PROBES) else []:
                if f.endswith(".npz"): self._by_names[tuple(json.loads(str(np.load(os.path.join(PROBES, f))["names"])))] = f[:-4]
        return self.head_for_task(self._by_names[tuple(names)])
    @torch.no_grad()
    def features(self, X, bs=512):
        out = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + bs])).to(self.dev).permute(0, 3, 1, 2).float().div_(255); xb = ((xb - self.MEAN) / self.STD).to(self.dtype)
            h = self.model(pixel_values=xb).last_hidden_state; f = torch.cat([h[:, 0], h[:, 1:].mean(1), torch.ones(h.shape[0], 1, device=self.dev, dtype=h.dtype)], dim=1); out.append(f)
        return torch.cat(out)
    @torch.no_grad()
    def correct(self, X, y, H): return int(((self.features(X) @ H.T).argmax(-1).cpu().numpy() == y).sum())
