#!/usr/bin/env python
"""Rank-16 LoRA client trainer for SigLIP 2 vision towers (The DC-Merge LoRA recipe, fixed in advance
(external/DC-Merge/vision_lora_merge/vision_training.py): LoRA r 16, alpha 16, dropout .1 on q/k/v/out_proj of every encoder layer; AdamW lr 3e-4, wd .1, warm-up 500,
cosine over max_steps 10,000, batch 32; cross-entropy on 100 x cosine logits against the frozen zero-shot text head (the task's templates), label smoothing 0; no
augmentation (the checkpoint's processor transform: squash-resize, mean/std .5); seed 420; the final weights are saved (DC-Merge's early stopping — patience 5 x eval
every 2,000 steps — cannot trigger inside 10,000 steps). bf16 autocast instead of fp16 GradScaler. Training images are streamed from the HF datasets cache; the
pipeline's n = 2000 test draw (or a seed-0 draw of 2000) is used for monitoring only. Env SIG_ARCH; args --task [--steps 10000 --bs 32 --lr 3e-4 --wd 0.1 --warmup 500 --seed 420 --workers 8]."""
import os, sys, json, time, math, argparse, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
os.environ.setdefault("POOL", "dc16"); HERE = os.path.dirname(os.path.abspath(__file__))
from src.backbones import siglip_backend as sig
from src.backbones.siglip_backend import ROOT, ARCH, MODEL, RES, NLAYERS, CELLS, KNOTS_T, prep
ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--steps", type=int, default=10000); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--wd", type=float, default=0.1); ap.add_argument("--warmup", type=int, default=500); ap.add_argument("--seed", type=int, default=420); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--eval-every", type=int, default=2000)
ap.add_argument("--alpha", type=float, default=16.0); ap.add_argument("--r", type=int, default=16); ap.add_argument("--dropout", type=float, default=0.1); ap.add_argument("--tag", default=""); a = ap.parse_args()
OUT = os.path.join(ROOT, "results", "siglip", "adapters" + (("_" + a.tag) if a.tag else ""), ARCH, a.task); os.makedirs(OUT, exist_ok=True); dev = torch.device("cuda:0")
torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
def log(*x): print(time.strftime("%Y-%m-%d %H:%M:%S"), *x, flush=True)
DS = {"stanford_cars": ("tanganke/stanford_cars", None), "dtd": ("tanganke/dtd", None), "eurosat": ("tanganke/eurosat", None), "gtsrb": ("tanganke/gtsrb", None), "mnist": ("ylecun/mnist", None), "resisc45": ("tanganke/resisc45", None),
      "sun397": ("tanganke/sun397", None), "svhn": ("ufldl-stanford/svhn", "cropped_digits"), "cifar10": ("tanganke/cifar10", None), "cifar100": ("tanganke/cifar100", None), "stl10": ("tanganke/stl10", None), "flowers102": ("dpdl-benchmark/oxford_flowers102", None),
      "oxfordpets": ("timm/oxford-iiit-pet", None), "fashionmnist": ("zalando-datasets/fashion_mnist", None), "sst2": ("clip-benchmark/wds_renderedsst2", None), "fer2013": ("clip-benchmark/wds_fer2013", None)}
from datasets import load_dataset
repo, cfg = DS[a.task]; wds = repo.startswith("clip-benchmark/")
def split(name): return load_dataset(repo, cfg, split=name) if cfg else load_dataset(repo, split=name)
tr = split("train"); cols = tr.column_names; labcol = "cls" if wds else "label"; imgcol = ("png" if "png" in cols else "jpg") if wds else "image"
CACHE = os.path.join(ROOT, "results", "siglip", f"cache{RES}"); os.makedirs(CACHE, exist_ok=True); namesf = os.path.join(CACHE, f"{a.task}_names.json")
if os.path.exists(namesf): names = json.load(open(namesf))
elif wds:
    from huggingface_hub import hf_hub_download; names = open(hf_hub_download(repo, "classnames.txt", repo_type="dataset")).read().strip().split("\n"); json.dump(names, open(namesf, "w"))
else: feat = tr.features[labcol]; names = [nm.replace("_", " ") for nm in (feat.names if hasattr(feat, "names") else [str(i) for i in range(int(max(tr[labcol])) + 1)])]; json.dump(names, open(namesf, "w"))
xf = os.path.join(CACHE, f"{a.task}_test_X.npy")
if os.path.exists(xf): Xte, yte = np.load(xf, mmap_mode="r"), np.load(xf.replace("_X", "_y"))
else:
    te = split("test"); idx = sorted(random.Random(0).sample(range(len(te)), min(2000, len(te)))); te = te.select(idx)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(32) as ex: Xte = np.stack(list(ex.map(prep, te[imgcol], chunksize=32)))
    yte = np.asarray(te[labcol], dtype=np.int64); np.save(xf, Xte); np.save(xf.replace("_X", "_y"), yte)
class TrainDS(torch.utils.data.Dataset):
    def __len__(self): return len(tr)
    def __getitem__(self, i): ex = tr[int(i)]; return torch.from_numpy(prep(ex[imgcol])), int(ex[labcol])
loader = torch.utils.data.DataLoader(TrainDS(), batch_size=a.bs, shuffle=True, drop_last=True, num_workers=a.workers, persistent_workers=a.workers > 0, generator=torch.Generator().manual_seed(a.seed))
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout):
        super().__init__(); self.base = base; self.scale = alpha / r; self.drop = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features)); self.lora_B = nn.Parameter(torch.zeros(base.out_features, r)); nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    def forward(self, x): return self.base(x) + (self.drop(x) @ self.lora_A.T @ self.lora_B.T) * self.scale
t0 = time.time(); from transformers import SiglipModel, AutoTokenizer
model = SiglipModel.from_pretrained(MODEL, dtype=torch.float32).to(dev); tok = AutoTokenizer.from_pretrained(MODEL)
for p in model.parameters(): p.requires_grad_(False)
loras = {}
for l in range(NLAYERS):
    sa = model.vision_model.encoder.layers[l].self_attn
    for p in ("q_proj", "k_proj", "v_proj", "out_proj"): m = LoRALinear(getattr(sa, p), a.r, a.alpha, a.dropout).to(dev); setattr(sa, p, m); loras[(l, p)] = m
params = [q for m in loras.values() for q in (m.lora_A, m.lora_B)]
MEAN = torch.full((1, 3, 1, 1), 0.5, device=dev); STD = torch.full((1, 3, 1, 1), 0.5, device=dev)
with torch.no_grad():
    W = []
    for n in names:
        te_ = tok([t.format(n) for t in KNOTS_T[a.task]], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(dev); e = model.get_text_features(input_ids=te_["input_ids"]); e = e / e.norm(dim=-1, keepdim=True); e = e.mean(0); W.append(e / e.norm())
    H = torch.stack(W)
def norm_px(xb): return (xb.to(dev).permute(0, 3, 1, 2).float().div_(255) - MEAN) / STD
@torch.no_grad()
def evaluate():
    model.eval(); c = 0
    for i in range(0, len(yte), 256):
        with torch.autocast("cuda", dtype=torch.bfloat16): f = model.get_image_features(pixel_values=norm_px(torch.from_numpy(np.ascontiguousarray(Xte[i:i + 256]))))
        f = f.float(); f = f / f.norm(dim=-1, keepdim=True); c += int(((f @ H.T).argmax(-1).cpu().numpy() == yte[i:i + 256]).sum())
    model.train(); return c / len(yte)
log(f"{ARCH} {a.task}: train {len(tr)} test {len(yte)} classes {len(names)} res {RES} ({time.time() - t0:.0f}s)"); acc0 = evaluate(); log(f"zero-shot acc {acc0:.4f}")
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
model.train(); hist = []; step = 0; it = iter(loader); tl = 0.0
while step < a.steps:
    try: xb, yb = next(it)
    except StopIteration: it = iter(loader); xb, yb = next(it)
    step += 1; opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16): f = model.get_image_features(pixel_values=norm_px(xb))
    f = f.float(); f = f / f.norm(dim=-1, keepdim=True); loss = F.cross_entropy(100.0 * f @ H.T, yb.to(dev)); loss.backward(); opt.step(); sched.step(); tl += loss.item()
    if step % 200 == 0: log(f"  step {step} loss {tl / 200:.4f} lr {sched.get_last_lr()[0]:.2e} ({time.time() - t0:.0f}s)"); tl = 0.0
    if step % a.eval_every == 0 or step == a.steps: acc = evaluate(); hist.append((step, acc)); log(f"  step {step} test acc {acc:.4f}")
sd = {}
for (l, p), m in loras.items():
    sd[f"base_model.model.vision_model.encoder.layers.{l}.self_attn.{p}.lora_A.weight"] = m.lora_A.detach().cpu().half(); sd[f"base_model.model.vision_model.encoder.layers.{l}.self_attn.{p}.lora_B.weight"] = m.lora_B.detach().cpu().half()
torch.save(sd, os.path.join(OUT, "adapter_model.bin"))
json.dump({"base_model_name_or_path": MODEL, "peft_type": "LORA", "r": a.r, "lora_alpha": a.alpha, "lora_dropout": a.dropout, "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"], "task_type": None, "recipe": "DC-Merge (vision_training.py): AdamW lr 3e-4 wd .1 warm-up 500 cosine, bs 32, max_steps 10000, CE on 100x cosine logits vs frozen zero-shot head, label smoothing 0, seed 420, no augmentation",
           "steps": a.steps, "bs": a.bs, "lr": a.lr, "wd": a.wd, "warmup": a.warmup, "seed": a.seed, "zero_shot_acc": acc0, "test_acc_hist": hist, "n_train": len(tr), "train_time_s": time.time() - t0}, open(os.path.join(OUT, "adapter_config.json"), "w"), indent=1)
log(f"saved {OUT}: zero-shot {acc0:.4f} -> final {hist[-1][1]:.4f} ({time.time() - t0:.0f}s)")
