#!/usr/bin/env python
"""rank-16 LoRA client trainer for SigLIP 2 vision towers under the light recipe of (train_clip_client.py), with only the backbone
API changed: SiglipModel and its tokenizer (padding to 64 tokens), the checkpoint's preprocessing (squash-resize to RES, mean/std .5; images pre-built by
data_newtasks_siglip.py), and the SigLIP line's fixed logit scale of 100 on the frozen zero-shot head (train_siglip_client.py). Recipe as 1,000 steps, AdamW lr 1e-4,
wd 0, warm-up 100 then cosine, batch 128 in --accum micro-batches, LoRA r 16, alpha 16, dropout .1 on q/k/v/out_proj of every encoder layer, at most 10,000 training images,
horizontal flips except on the NOFLIP tasks, seed 0, bf16 autocast. Writes results/siglip/adapters/<ARCH>/<task>/ (what siglip_backend.load_adapter reads).
Env SIG_ARCH = S2B16 | S2L16 | S2SO. Usage: train_siglip_client_light.py --task pcam --steps 1000 --lr 1e-4 --bs 128 --accum 4 [--out <dir>] [--cache <dir>]."""
import os, sys, json, time, math, argparse, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
os.environ.setdefault("POOL", "dc16"); HERE = os.path.dirname(os.path.abspath(__file__))
from src.backbones.siglip_backend import ROOT, ARCH, MODEL, RES, NLAYERS, KNOTS_T
ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--steps", type=int, default=1000); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--bs", type=int, default=128); ap.add_argument("--accum", type=int, default=4); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--wd", type=float, default=0.0)
ap.add_argument("--warmup", type=int, default=100); ap.add_argument("--alpha", type=float, default=16.0); ap.add_argument("--out", default=""); ap.add_argument("--cache", default=""); a = ap.parse_args()
OUT = a.out or os.path.join(ROOT, "results", "siglip", "adapters", ARCH, a.task); os.makedirs(OUT, exist_ok=True); CACHE = a.cache or os.path.join(ROOT, "results", "siglip", f"cache{RES}")
dev = torch.device("cuda:0"); torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed); NOFLIP = {"mnist", "svhn", "sst2", "gtsrb", "fashionmnist"}   # as train_clip_client.py
def log(*x): print(time.strftime("%Y-%m-%d %H:%M:%S"), *x, flush=True)
Xtr, ytr = np.load(os.path.join(CACHE, f"{a.task}_train10000_X.npy"), mmap_mode="r"), np.load(os.path.join(CACHE, f"{a.task}_train10000_y.npy"))
Xte, yte = np.load(os.path.join(CACHE, f"{a.task}_test_X.npy"), mmap_mode="r"), np.load(os.path.join(CACHE, f"{a.task}_test_y.npy")); names = json.load(open(os.path.join(CACHE, f"{a.task}_names.json")))
assert Xtr.shape[1] == RES and Xte.shape[1] == RES, (Xtr.shape, RES)
class LoRALinear(nn.Module):
    def __init__(self, base, r=16, alpha=16, dropout=0.1):
        super().__init__(); self.base = base; self.r = r; self.scale = alpha / r; self.drop = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features)); self.lora_B = nn.Parameter(torch.zeros(base.out_features, r)); nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    def forward(self, x): return self.base(x) + (self.drop(x) @ self.lora_A.T @ self.lora_B.T) * self.scale
t0 = time.time(); from transformers import SiglipModel, AutoTokenizer
model = SiglipModel.from_pretrained(MODEL, dtype=torch.float32).to(dev); tok = AutoTokenizer.from_pretrained(MODEL)
for p in model.parameters(): p.requires_grad_(False)
loras = {}
for l in range(NLAYERS):
    sa = model.vision_model.encoder.layers[l].self_attn
    for p in ("q_proj", "k_proj", "v_proj", "out_proj"): m = LoRALinear(getattr(sa, p), alpha=a.alpha).to(dev); setattr(sa, p, m); loras[(l, p)] = m
params = [q for m in loras.values() for q in (m.lora_A, m.lora_B)]
MEAN = torch.full((1, 3, 1, 1), 0.5, device=dev); STD = torch.full((1, 3, 1, 1), 0.5, device=dev)
with torch.no_grad():
    W = []
    for n in names:
        te_ = tok([t.format(n) for t in KNOTS_T[a.task]], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(dev); e = model.get_text_features(input_ids=te_["input_ids"]); e = e / e.norm(dim=-1, keepdim=True); e = e.mean(0); W.append(e / e.norm())
    H = torch.stack(W)
log(f"{ARCH} {a.task}: train {len(ytr)} test {len(yte)} classes {len(names)} res {RES} ({time.time() - t0:.0f}s)")
def batch(X, idx, train):
    xb = torch.from_numpy(np.ascontiguousarray(X[np.sort(idx)])).to(dev).permute(0, 3, 1, 2).float().div_(255)
    if train and a.task not in NOFLIP: flip = torch.rand(len(idx), device=dev) < .5; xb[flip] = xb[flip].flip(-1)
    return (xb - MEAN) / STD
@torch.no_grad()
def evaluate():
    model.eval(); c = 0
    for i in range(0, len(yte), 128):
        with torch.autocast("cuda", dtype=torch.bfloat16): f = model.get_image_features(pixel_values=batch(Xte, np.arange(i, min(i + 128, len(yte))), False))
        f = f.float(); f = f / f.norm(dim=-1, keepdim=True); c += int(((f @ H.T).argmax(-1).cpu().numpy() == yte[i:i + 128]).sum())
    model.train(); return c / len(yte)
acc0 = evaluate(); log(f"zero-shot acc {acc0:.4f}")
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
model.train(); rng = np.random.default_rng(a.seed); hist = []
for step in range(1, a.steps + 1):
    idx = np.sort(rng.choice(len(ytr), a.bs, replace=len(ytr) < a.bs)); opt.zero_grad(set_to_none=True); ltot = 0.0
    for ci in np.array_split(idx, a.accum):
        yb = torch.from_numpy(ytr[ci]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16): f = model.get_image_features(pixel_values=batch(Xtr, ci, True))
        f = f.float(); f = f / f.norm(dim=-1, keepdim=True); loss = F.cross_entropy(100.0 * f @ H.T, yb) * (len(ci) / len(idx)); loss.backward(); ltot += loss.item()
    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
    if step % 100 == 0 or step == a.steps: log(f"  step {step} loss {ltot:.4f} lr {sched.get_last_lr()[0]:.2e} ({time.time() - t0:.0f}s)")
    if step % max(a.steps // 4, 1) == 0 or step == a.steps: acc = evaluate(); hist.append((step, acc)); log(f"  step {step} test acc {acc:.4f}")
sd = {}
for (l, p), m in loras.items(): sd[f"base_model.model.vision_model.encoder.layers.{l}.self_attn.{p}.lora_A.weight"] = m.lora_A.detach().cpu().half(); sd[f"base_model.model.vision_model.encoder.layers.{l}.self_attn.{p}.lora_B.weight"] = m.lora_B.detach().cpu().half()
torch.save(sd, os.path.join(OUT, "adapter_model.bin"))
json.dump({"base_model_name_or_path": MODEL, "peft_type": "LORA", "r": 16, "lora_alpha": a.alpha, "lora_dropout": 0.1, "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"], "bias": "none", "inference_mode": True, "task_type": None,
           "recipe": {"tag": "light", "steps": a.steps, "lr": a.lr, "alpha": a.alpha, "bs": a.bs, "accum": a.accum, "wd": a.wd, "warmup": a.warmup, "seed": a.seed, "arch": ARCH, "model": MODEL, "res": RES, "flip": a.task not in NOFLIP, "logit_scale": 100.0, "head": "zero-shot frozen"},
           "zero_shot_acc": acc0, "test_acc_history": hist, "n_train": int(len(ytr)), "n_test_eval": int(len(yte)), "wall_s": time.time() - t0}, open(os.path.join(OUT, "adapter_config.json"), "w"), indent=1)
log(f"saved {OUT}: zero-shot {acc0:.4f} -> {hist[-1][1]:.4f} ({time.time() - t0:.0f}s)")
