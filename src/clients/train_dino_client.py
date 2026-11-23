#!/usr/bin/env python
"""Rank-16 LoRA client trainer for DINO backbones under the FROZEN linear probe of the task (). Recipe = the DC-Merge LoRA recipe as in train_siglip_client.py
(r 16, alpha 16, dropout .1 on the four attention projections of every block; AdamW lr 3e-4, wd .1, warm-up 500, cosine over 10,000 steps, batch 32, cross-entropy, label
smoothing 0, no augmentation, seed 420, final weights, bf16 autocast); logits = [CLS, mean patch, 1] @ probe^T (no temperature: the probe is a trained classifier).
Env DINO_ARCH; --task [--steps 10000 --bs 32 --lr 3e-4 --wd 0.1 --warmup 500 --seed 420 --workers 8 --eval-every 2000 --tag]."""
import os, sys, json, time, math, argparse, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
os.environ.setdefault("POOL", "dc16"); HERE = os.path.dirname(os.path.abspath(__file__))
from src.backbones import dino_backend as dino; from src.backbones import dino_data as DD
ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--steps", type=int, default=10000); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--wd", type=float, default=0.1); ap.add_argument("--warmup", type=int, default=500); ap.add_argument("--seed", type=int, default=420); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--eval-every", type=int, default=2000)
ap.add_argument("--alpha", type=float, default=16.0); ap.add_argument("--r", type=int, default=16); ap.add_argument("--dropout", type=float, default=0.1); ap.add_argument("--tag", default=""); ap.add_argument("--probe-tag", default="")
ap.add_argument("--clip", type=float, default=0.0); ap.add_argument("--stop-at", type=int, default=0); a = ap.parse_args()   # addendum A3: gradient-norm clipping (0 = off, every earlier run unchanged); stop a longer schedule early (pilot)
STOP = a.stop_at if a.stop_at > 0 else a.steps
OUT = os.path.join(dino.RES_ROOT, "adapters" + (("_" + a.tag) if a.tag else ""), dino.ARCH, a.task); os.makedirs(OUT, exist_ok=True); dev = torch.device("cuda:0")
CACHE = os.path.join(dino.RES_ROOT, "cache224"); torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
def log(*x): print(time.strftime("%Y-%m-%d %H:%M:%S"), *x, flush=True)
t0 = time.time(); tr = DD.split(a.task, "train"); labcol, imgcol, wds = DD.cols(a.task, tr); names = DD.names_of(a.task, tr, labcol, wds, CACHE); Xte, yte = DD.test_draw(a.task, dino.prep, CACHE)
loader = torch.utils.data.DataLoader(DD.TrainDS(tr, imgcol, labcol, dino.prep), batch_size=a.bs, shuffle=True, drop_last=True, num_workers=a.workers, persistent_workers=a.workers > 0, generator=torch.Generator().manual_seed(a.seed))
z = np.load(os.path.join(dino.PROBES + (("_" + a.probe_tag) if a.probe_tag else ""), a.task + ".npz")); H = torch.as_tensor(np.concatenate([z["W"], z["b"][:, None]], axis=1), dtype=torch.float32, device=dev); probe_acc = float(z["test_acc"])
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout):
        super().__init__(); self.base = base; self.scale = alpha / r; self.drop = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features)); self.lora_B = nn.Parameter(torch.zeros(base.out_features, r)); nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    def forward(self, x): return self.base(x) + (self.drop(x) @ self.lora_A.T @ self.lora_B.T) * self.scale
from transformers import AutoModel
model = AutoModel.from_pretrained(dino.MODEL, dtype=torch.float32).to(dev)
for p in model.parameters(): p.requires_grad_(False)
loras = {}
for c in dino.CELLS:
    l, p = c; L = dino.layers(model)[l]; m = LoRALinear(dino.linear(model, c), a.r, a.alpha, a.dropout).to(dev); loras[c] = m
    if dino.FAM == "dinov2": (setattr(L.attention.output, "dense", m) if p == "dense" else setattr(L.attention.attention, p, m))
    else: setattr(L.attention, p, m)
params = [q for m in loras.values() for q in (m.lora_A, m.lora_B)]
MEAN = torch.tensor(dino.MEAN, device=dev).view(1, 3, 1, 1); STD = torch.tensor(dino.STD, device=dev).view(1, 3, 1, 1)
def norm_px(xb): return (xb.to(dev).permute(0, 3, 1, 2).float().div_(255) - MEAN) / STD
def feats(xb):
    h = model(pixel_values=norm_px(xb)).last_hidden_state; return torch.cat([h[:, 0], h[:, 1:].mean(1), torch.ones(h.shape[0], 1, device=dev, dtype=h.dtype)], dim=1).float()
@torch.no_grad()
def evaluate():
    model.eval(); c = 0
    for i in range(0, len(yte), 256):
        with torch.autocast("cuda", dtype=torch.bfloat16): f = feats(torch.from_numpy(np.ascontiguousarray(Xte[i:i + 256])))
        c += int(((f @ H.T).argmax(-1).cpu().numpy() == yte[i:i + 256]).sum())
    model.train(); return c / len(yte)
log(f"{dino.ARCH} {a.task}: train {len(tr)} test {len(yte)} classes {len(names)} probe test acc {probe_acc:.4f} ({time.time() - t0:.0f}s)"); acc0 = evaluate(); log(f"frozen-probe acc (zero LoRA) {acc0:.4f}")
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
model.train(); hist = []; step = 0; it = iter(loader); tl = 0.0
while step < STOP:
    try: xb, yb = next(it)
    except StopIteration: it = iter(loader); xb, yb = next(it)
    step += 1; opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16): f = feats(xb)
    loss = F.cross_entropy(f @ H.T, yb.to(dev)); loss.backward()
    if a.clip > 0: torch.nn.utils.clip_grad_norm_(params, a.clip)
    opt.step(); sched.step(); tl += loss.item()
    if step % 200 == 0: log(f"  step {step} loss {tl / 200:.4f} lr {sched.get_last_lr()[0]:.2e} ({time.time() - t0:.0f}s)"); tl = 0.0
    if step % a.eval_every == 0 or step == STOP: acc = evaluate(); hist.append((step, acc)); log(f"  step {step} test acc {acc:.4f}")
sd = {c: None for c in ()}; sd = {}
for c, m in loras.items(): sd[dino.lora_key(c, "A")] = m.lora_A.detach().cpu().half(); sd[dino.lora_key(c, "B")] = m.lora_B.detach().cpu().half()
torch.save(sd, os.path.join(OUT, "adapter_model.bin"))
json.dump({"base_model_name_or_path": dino.MODEL, "peft_type": "LORA", "r": a.r, "lora_alpha": a.alpha, "lora_dropout": a.dropout, "target_modules": list(dino.PROJ), "task_type": None, "head": "frozen linear probe (train_dino_probe.py)", "probe_test_acc": probe_acc,
           "recipe": "DC-Merge LoRA recipe under the frozen probe: AdamW lr 3e-4 wd .1 warm-up 500 cosine, bs 32, max_steps 10000, CE, label smoothing 0, seed 420, no augmentation", "steps": a.steps, "bs": a.bs, "lr": a.lr, "wd": a.wd, "warmup": a.warmup, "seed": a.seed,
           "zero_shot_acc": acc0, "test_acc_hist": hist, "n_train": len(tr), "train_time_s": time.time() - t0, "clip": a.clip, "stop_at": a.stop_at}, open(os.path.join(OUT, "adapter_config.json"), "w"), indent=1)
log(f"saved {OUT}: frozen-probe {acc0:.4f} -> final {hist[-1][1]:.4f} ({time.time() - t0:.0f}s)")
