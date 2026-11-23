"""Rank-16 LoRA client trainer for CLIP vision towers (q/k/v/out_proj of every layer), zero-shot text head frozen. Writes a PEFT-compatible adapter dir (adapter_model.bin + adapter_config.json) that clip_backend.py load_adapter reads.
ARCH=B32|B16|L14 (clip_backend.py). Usage: train_clip_client.py --task dtd --steps 2000 --lr 1e-4 --bs 128 --tag light [--seed 0] [--eval 2000]."""
import os, sys, json, time, math, argparse, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
os.environ.setdefault("POOL", "dc16")
HERE = os.path.dirname(os.path.abspath(__file__))
from src.backbones.clip_backend import ROOT, ARCH, NLAYERS, MODEL, KNOTS_T, CELLS
from src.backbones import harness_clip as oc# DS, prep, names via classnames / features
from transformers import CLIPModel, CLIPTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--bs", type=int, default=128); ap.add_argument("--tag", default="light"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--eval", type=int, default=2000)
ap.add_argument("--accum", type=int, default=1); ap.add_argument("--wd", type=float, default=0.0); ap.add_argument("--warmup", type=int, default=100); ap.add_argument("--max_train", type=int, default=10000); ap.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha (client-chosen scaling; saved in adapter_config so the loader applies alpha/r)"); a = ap.parse_args()
OUT = os.path.join(ROOT, "results", "clipdom", "adapters", ARCH, a.tag, a.task); os.makedirs(OUT, exist_ok=True); CACHE = os.path.join(ROOT, "results", "clipdom", "cache")
os.makedirs(CACHE, exist_ok=True); dev = torch.device("cuda:0"); torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
log = oc.log; NOFLIP = {"mnist", "svhn", "sst2", "gtsrb", "fashionmnist"}
def load_split(task, split, n=None):
    """uint8 NHWC 224 crops + labels + names; cached under results/clipdom/cache (train: whole split, test: seed-0 subset of n)."""
    tag = f"train{n}" if split == "train" else f"test{n}"; xf = os.path.join(CACHE, f"{task}_{tag}_X.npy")
    if os.path.exists(xf): return np.load(xf, mmap_mode="r"), np.load(xf.replace("_X", "_y")), json.load(open(os.path.join(CACHE, f"{task}_names.json")))
    from datasets import load_dataset; from concurrent.futures import ThreadPoolExecutor
    repo, cfg = oc.DS[task]; ds = load_dataset(repo, cfg, split=split) if cfg else load_dataset(repo, split=split); wds = repo.startswith("clip-benchmark/")
    labcol = "cls" if wds else "label"; imgcol = ("png" if "png" in ds.column_names else "jpg") if wds else "image"
    if n and len(ds) > n: ds = ds.select(sorted(random.Random(0).sample(range(len(ds)), n)))
    with ThreadPoolExecutor(32) as ex: X = np.stack(list(ex.map(oc.prep, ds[imgcol], chunksize=32)))
    y = np.asarray(ds[labcol], dtype=np.int64)
    if wds:
        from huggingface_hub import hf_hub_download; names = open(hf_hub_download(repo, "classnames.txt", repo_type="dataset")).read().strip().split("\n")
    else: feat = ds.features[labcol]; names = [nm.replace("_", " ") for nm in (feat.names if hasattr(feat, "names") else [str(i) for i in range(int(y.max()) + 1)])]
    np.save(xf, X); np.save(xf.replace("_X", "_y"), y); json.dump(names, open(os.path.join(CACHE, f"{task}_names.json"), "w")); return X, y, names
class LoRALinear(nn.Module):
    def __init__(self, base, r=16, alpha=16, dropout=0.1):
        super().__init__(); self.base = base; self.r = r; self.scale = alpha / r; self.drop = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features)); self.lora_B = nn.Parameter(torch.zeros(base.out_features, r)); nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    def forward(self, x): return self.base(x) + (self.drop(x) @ self.lora_A.T @ self.lora_B.T) * self.scale
t0 = time.time(); model = CLIPModel.from_pretrained(MODEL, dtype=torch.float32).to(dev); tok = CLIPTokenizer.from_pretrained(MODEL)
for p in model.parameters(): p.requires_grad_(False)
loras = {}
for l in range(NLAYERS):
    sa = model.vision_model.encoder.layers[l].self_attn
    for p in ("q_proj", "k_proj", "v_proj", "out_proj"): m = LoRALinear(getattr(sa, p), alpha=a.alpha).to(dev); setattr(sa, p, m); loras[(l, p)] = m
params = [q for m in loras.values() for q in (m.lora_A, m.lora_B)]
Xtr, ytr, names = load_split(a.task, "train", a.max_train); Xte, yte, _ = load_split(a.task, "test", a.eval); log(f"{a.task}: train {len(ytr)} test {len(yte)} classes {len(names)} ({time.time() - t0:.0f}s)")
try: _pp = __import__("transformers").CLIPImageProcessor.from_pretrained(MODEL)
except OSError: _pp = __import__("transformers").CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")   # B/16 snapshot has no preprocessor_config.json; OpenAI CLIP mean/std are identical across B/32, B/16, L/14
MEAN = torch.tensor(_pp.image_mean, device=dev).view(1, 3, 1, 1); STD = torch.tensor(_pp.image_std, device=dev).view(1, 3, 1, 1)
with torch.no_grad():
    W = []
    for n in names:
        te = tok([t.format(n) for t in KNOTS_T[a.task]], padding=True, truncation=True, return_tensors="pt").to(dev); e = model.get_text_features(**te); e = e / e.norm(dim=-1, keepdim=True); e = e.mean(0); W.append(e / e.norm())
    H = torch.stack(W); ls = model.logit_scale.exp().item()
def batch(X, idx, train):
    xb = torch.from_numpy(np.ascontiguousarray(X[np.sort(idx)])).to(dev).permute(0, 3, 1, 2).float().div_(255)
    if train and a.task not in NOFLIP: flip = torch.rand(len(idx), device=dev) < .5; xb[flip] = xb[flip].flip(-1)
    return (xb - MEAN) / STD
@torch.no_grad()
def evaluate():
    model.eval(); c = 0
    for i in range(0, len(yte), 256):
        f = model.get_image_features(pixel_values=batch(Xte, np.arange(i, min(i + 256, len(yte))), False)); f = f / f.norm(dim=-1, keepdim=True); c += int(((f @ H.T).argmax(-1).cpu().numpy() == yte[i:i + 256]).sum())
    model.train(); return c / len(yte)
acc0 = evaluate(); log(f"zero-shot acc {acc0:.4f}")
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
model.train(); rng = np.random.default_rng(a.seed); hist = []
for step in range(1, a.steps + 1):
    idx = np.sort(rng.choice(len(ytr), a.bs, replace=len(ytr) < a.bs)); opt.zero_grad(set_to_none=True); ltot = 0.0
    for ci in np.array_split(idx, a.accum):   # gradient accumulation (--accum > 1 for ViT-L/14: same batch of 128, four micro-batches; added for the )
        yb = torch.from_numpy(ytr[ci]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = model.get_image_features(pixel_values=batch(Xtr, ci, True))
        f = f.float(); f = f / f.norm(dim=-1, keepdim=True); loss = F.cross_entropy(ls * f @ H.T, yb) * (len(ci) / len(idx)); loss.backward(); ltot += loss.item()
    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
    if step % 100 == 0 or step == a.steps: log(f"  step {step} loss {ltot:.4f} lr {sched.get_last_lr()[0]:.2e} ({time.time() - t0:.0f}s)")
    if step % max(a.steps // 4, 1) == 0 or step == a.steps: acc = evaluate(); hist.append((step, acc)); log(f"  step {step} test acc {acc:.4f}")
sd = {}
for (l, p), m in loras.items(): sd[f"base_model.model.vision_model.encoder.layers.{l}.self_attn.{p}.lora_A.weight"] = m.lora_A.detach().cpu().half(); sd[f"base_model.model.vision_model.encoder.layers.{l}.self_attn.{p}.lora_B.weight"] = m.lora_B.detach().cpu().half()
torch.save(sd, os.path.join(OUT, "adapter_model.bin"))
json.dump({"peft_type": "LORA", "r": 16, "lora_alpha": a.alpha, "lora_dropout": 0.1, "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"], "bias": "none", "inference_mode": True, "task_type": None,
           "recipe": {"tag": a.tag, "steps": a.steps, "lr": a.lr, "alpha": a.alpha, "bs": a.bs, "accum": a.accum, "wd": a.wd, "warmup": a.warmup, "seed": a.seed, "arch": ARCH, "model": MODEL, "flip": a.task not in NOFLIP, "head": "zero-shot frozen"},
           "zero_shot_acc": acc0, "test_acc_history": hist, "n_train": int(len(ytr)), "n_test_eval": int(len(yte)), "wall_s": time.time() - t0}, open(os.path.join(OUT, "adapter_config.json"), "w"), indent=1)
log(f"saved {OUT}: zero-shot {acc0:.4f} -> {hist[-1][1]:.4f} ({time.time() - t0:.0f}s)")
