#!/usr/bin/env python
"""Frozen linear probe per task for the DINO line (): features [CLS, mean patch, 1] of the FROZEN backbone on the train split (streamed; --max-train caps it),
multinomial logistic regression fitted with L-BFGS (full batch, 200 iterations, L2 1e-4 on W), then frozen for every client of that task and for every merge evaluation
(the analogue of CLIP's zero-shot head). Saves results/dino/probes/<ARCH>/<task>.npz (W, b, names, train_acc, test_acc on the n = 2000 draw, n_train). Env DINO_ARCH; --task."""
import os, sys, json, time, argparse, numpy as np, torch, torch.nn.functional as F
os.environ.setdefault("POOL", "dc16"); HERE = os.path.dirname(os.path.abspath(__file__))
from src.backbones import dino_backend as dino; from src.backbones import dino_data as DD
ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--max-train", type=int, default=0); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--iters", type=int, default=200); ap.add_argument("--l2", type=float, default=1e-4); ap.add_argument("--tag", default=""); a = ap.parse_args()
dev = torch.device("cuda:0"); CACHE = os.path.join(dino.RES_ROOT, "cache224"); PROBES = dino.PROBES + (("_" + a.tag) if a.tag else ""); os.makedirs(PROBES, exist_ok=True)
def log(*x): print(time.strftime("%Y-%m-%d %H:%M:%S"), *x, flush=True)
t0 = time.time(); C = dino.Clip("cuda:0"); tr = DD.split(a.task, "train"); labcol, imgcol, wds = DD.cols(a.task, tr); names = DD.names_of(a.task, tr, labcol, wds, CACHE)
idx = None
if a.max_train and len(tr) > a.max_train: idx = sorted(np.random.default_rng(0).choice(len(tr), a.max_train, replace=False).tolist())
loader = torch.utils.data.DataLoader(DD.TrainDS(tr, imgcol, labcol, dino.prep, idx), batch_size=256, shuffle=False, num_workers=a.workers)
feats, ys = [], []
with torch.no_grad():
    for xb, yb in loader: feats.append(C.features(xb.numpy(), bs=256).float()); ys.append(yb)
Ftr = torch.cat(feats); ytr = torch.cat(ys).to(dev); log(f"{dino.ARCH} {a.task}: train features {tuple(Ftr.shape)} classes {len(names)} ({time.time() - t0:.0f}s)")
Xte, yte = DD.test_draw(a.task, dino.prep, CACHE); Fte = C.features(Xte, bs=256).float(); yte_t = torch.as_tensor(yte, device=dev)
d = Ftr.shape[1] - 1; W = torch.zeros(len(names), d + 1, device=dev, requires_grad=True)   # last column = bias (constant feature)
opt = torch.optim.LBFGS([W], lr=1, max_iter=a.iters, history_size=20, line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-9)
def closure():
    opt.zero_grad(); loss = F.cross_entropy(Ftr @ W.T, ytr) + a.l2 * (W[:, :d] ** 2).sum(); loss.backward(); return loss
opt.step(closure)
with torch.no_grad(): tr_acc = float(((Ftr @ W.T).argmax(1) == ytr).float().mean()); te_acc = float(((Fte @ W.T).argmax(1) == yte_t).float().mean())
Wn = W.detach().cpu().numpy(); np.savez(os.path.join(PROBES, a.task + ".npz"), W=Wn[:, :d].astype(np.float32), b=Wn[:, d].astype(np.float32), names=json.dumps(names), train_acc=tr_acc, test_acc=te_acc, n_train=len(Ftr), iters=a.iters, l2=a.l2)
log(f"probe {a.task}: train acc {tr_acc:.4f} test acc {te_acc:.4f} (n_train {len(Ftr)}, {time.time() - t0:.0f}s) -> {PROBES}")
