#!/usr/bin/env python
"""Read the stored runs of a set of pools and print what the server returns on each, next to the best baseline row.

    python -m src.method.summarize --kind dc --arch B32 --pools r8_10 r8_11 r8_12

For every pool this reads the five run records of run_method.py (one-shot, sequential, half-sequential and Chain of Merges Gram ends, and the anchored points),
applies the server's choice, and compares the returned point's test retention with the rows of run_baselines.py and run_chain_of_merges.py.

The choice uses calibration scores only (s_P, the mean agreement with the clients' pseudo-answers):
  1. among the four Gram ends' best points, the highest score; ties go to the one-shot end, then the sequential, the half-sequential, the Chain of Merges end;
  2. an anchored point replaces that point only when its score is strictly higher; ties among anchored points go to the smaller c.
Test labels enter only the reported retention."""
import argparse, json, os
import numpy as np
from src.paths import ROOT

TREES = {"one": "", "full": "v_seq", "half": "v_seqcut0.5", "com": "v_com", "anchor": "v_anchor"}
ORDER = ["one", "full", "half", "com"]          # tie order among the Gram ends
C_GRID = (0.0, 0.03, 0.1, 0.3)                  # anchored points; c = 0 is the base model
BAND = 0.01                                     # two retentions within the band count as a tie


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def server_choice(runs):
    """runs: {tree key: run record}. Returns the returned point's retention and its name, or None if a run is missing."""
    if any(runs[k] is None or not runs[k].get("done") for k in TREES):
        return None
    score = {k: runs[k]["picks"]["P"]["s_P"] for k in ORDER}
    best_end = max(ORDER, key=lambda k: (round(score[k], 12), -ORDER.index(k)))
    candidates = [(score[best_end], 1, 0.0, runs[best_end]["R"]["L1"], best_end)]
    for c in C_GRID:
        point = runs["anchor"]["addon"]["points"][f"N@{c}"]
        candidates.append((point["s_P"], 0, -c, runs["anchor"]["R"][f"A:N@{c}"], f"anchor c={c:g}"))
    s, _, _, retention, name = max(candidates, key=lambda x: (round(x[0], 12), x[1], x[2]))
    pick = runs[best_end]["picks"]["P"]
    return {"R": float(retention), "pick": name, "score": float(s), "a": pick["a"], "side": pick["side"], "length": pick["s"]}


def baseline_rows(kind, arch, pool):
    """Every baseline row stored for the pool: {row name: test retention}. The base model is not a row."""
    rows = {}
    refs = load(os.path.join(ROOT, "results", "clipnew", "refs", arch, pool, "refs.json"))
    if refs:
        rows.update({k: float(v) for k, v in refs["R"].items() if k != "base"})
    com = load(os.path.join(ROOT, "results", "clipclose", "com", kind, arch, pool, "com.json"))
    if com and com.get("done"):
        rows["chain_of_merges"] = float(com["R"]["com"])
    return rows


def wtl(d):
    return [int(sum(x > BAND for x in d)), int(sum(abs(x) <= BAND for x in d)), int(sum(x < -BAND for x in d))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=["dc", "knots", "siglip", "dino"])
    ap.add_argument("--arch", required=True)
    ap.add_argument("--pools", nargs="+", required=True)
    ap.add_argument("--runs", default=os.path.join(ROOT, "results", "clipclose", "closedlen"), help="folder that holds the run trees")
    ap.add_argument("--out", help="write the per-pool numbers to this JSON file")
    a = ap.parse_args()
    out, margins = {}, []
    for pool in a.pools:
        runs = {k: load(os.path.join(a.runs, *([t] if t else []), a.kind, a.arch, pool, "mrun.json")) for k, t in TREES.items()}
        choice = server_choice(runs)
        if choice is None:
            print(f"{pool}: runs missing ({', '.join(k for k in TREES if runs[k] is None or not runs[k].get('done'))})")
            continue
        rows = baseline_rows(a.kind, a.arch, pool)
        best = max(rows, key=rows.get) if rows else None
        choice.update({"best_row": best, "R_best_row": rows.get(best), "rows": rows})
        out[pool] = choice
        line = f"{pool}: OneMetric {choice['R']:.4f} ({choice['pick']})"
        if best:
            margins.append(choice["R"] - rows[best])
            line += f" | best row {best} {rows[best]:.4f} | margin {margins[-1]:+.4f}"
        print(line)
    if out:
        print(f"mean retention over {len(out)} pools: {np.mean([v['R'] for v in out.values()]):.4f}")
    if margins:
        print(f"mean margin over the best row: {np.mean(margins):+.4f}, W/T/L {wtl(margins)}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
