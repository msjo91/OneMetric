"""SigLIP 2 backbone (B/16 at 224, L/16 at 256, So400m/14 at 224) with the interface of clip_backend: the model, client adapters under results/siglip/adapters, zero-shot heads, image features. Env SIG_ARCH = S2B16 | S2L16 | S2SO."""
import os, sys, json, numpy as np, torch
os.environ.setdefault("POOL", "dc16")
HERE = os.path.dirname(os.path.abspath(__file__))
from onemetric.clip_backend import ROOT, TASKS8, TASKS16, KNOTS_T, KEY
ARCH = os.environ.get("SIG_ARCH", "S2B16")
MODELS = {"S2B16": ("google/siglip2-base-patch16-224", 12, 768, 224), "S2L16": ("google/siglip2-large-patch16-256", 24, 1024, 256), "S2SO": ("google/siglip2-so400m-patch14-224", 27, 1152, 224)}
MODEL, NLAYERS, HIDDEN, RES = MODELS[ARCH]
CELLS = [(l, p) for l in range(NLAYERS) for p in ("q_proj", "k_proj", "v_proj", "out_proj")]
ADAPTERS = os.path.join(ROOT, "results", "siglip", "adapters", ARCH); DC16_FT, DC16_FT_B16 = {}, {}
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
    """uint8 HWC RES x RES: the SiglipImageProcessor transform (squash-resize, bilinear), no crop."""
    from PIL import Image
    return np.array(img.convert("RGB").resize((RES, RES), Image.BILINEAR), dtype=np.uint8)   # np.array (writable copy): torch.from_numpy in the trainer
class Clip:
    """Same API as clip_backend.Clip: W0, set_delta, head(names, templates), features(X, bs)."""
    def __init__(self, device="cuda:0", name=None, dtype=None):
        from transformers import SiglipModel, AutoTokenizer
        name = name or MODEL; self.dev = torch.device(device); self.dtype = dtype or (torch.float16 if self.dev.type == "cuda" else torch.float32)
        self.model = SiglipModel.from_pretrained(name, dtype=self.dtype).to(self.dev).eval(); self.tok = AutoTokenizer.from_pretrained(name)
        self.W0 = {c: self._param(c).detach().clone() for c in CELLS}
        self.MEAN = torch.full((1, 3, 1, 1), 0.5, device=self.dev); self.STD = torch.full((1, 3, 1, 1), 0.5, device=self.dev); self._heads = {}
    def _param(self, c): l, p = c; return getattr(self.model.vision_model.encoder.layers[l].self_attn, p).weight
    def set_delta(self, delta):
        with torch.no_grad():
            for c in CELLS:
                w = self.W0[c].clone()
                if delta is not None:
                    d = delta[c]; d = (d["B"] @ d["A"]) if isinstance(d, dict) else d
                    w += torch.as_tensor(np.ascontiguousarray(d), dtype=torch.float32, device=self.dev).to(self.dtype)
                self._param(c).copy_(w)
    @torch.no_grad()
    def head(self, names, templates):
        key = (tuple(names), tuple(templates))
        if key in self._heads: return self._heads[key]
        W = []
        for n in names:
            te = self.tok([t.format(n) for t in templates], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(self.dev)
            e = self.model.get_text_features(input_ids=te["input_ids"]).float(); e = e / e.norm(dim=-1, keepdim=True); e = e.mean(0); W.append(e / e.norm())
        H = torch.stack(W).to(self.dtype); self._heads[key] = H; return H
    @torch.no_grad()
    def features(self, X, bs=512):
        out = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + bs])).to(self.dev).permute(0, 3, 1, 2).float().div_(255); xb = ((xb - self.MEAN) / self.STD).to(self.dtype)
            f = self.model.get_image_features(pixel_values=xb); out.append(f / f.norm(dim=-1, keepdim=True))
        return torch.cat(out)
    @torch.no_grad()
    def correct(self, X, y, H): return int(((self.features(X) @ H.T).argmax(-1).cpu().numpy() == y).sum())
