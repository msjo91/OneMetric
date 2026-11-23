"""The server's choice across the five run records: highest score, the stated tie order, and the anchored point only when strictly better."""
from src.method.summarize import server_choice, C_GRID


def run(score, retention):
    return {"done": True, "picks": {"P": {"s_P": score, "a": 0.5, "side": "r", "s": 1.0}}, "R": {"L1": retention}}


def records(scores, anchor_scores):
    r = {k: run(s, 0.1 * i + 0.1) for i, (k, s) in enumerate(zip(["one", "full", "half", "com"], scores))}
    r["anchor"] = {"done": True, "picks": r["one"]["picks"], "R": {f"A:N@{c}": 0.9 - c for c in C_GRID}, "addon": {"points": {f"N@{c}": {"s_P": s} for c, s in zip(C_GRID, anchor_scores)}}}
    return r


def test_highest_score_wins():
    assert server_choice(records([0.80, 0.82, 0.81, 0.79], [0.1] * 4))["pick"] == "full"


def test_ties_go_to_the_earlier_end():
    assert server_choice(records([0.80, 0.82, 0.82, 0.82], [0.1] * 4))["pick"] == "full"


def test_anchor_needs_a_strictly_higher_score():
    assert server_choice(records([0.80, 0.82, 0.81, 0.79], [0.82, 0.82, 0.5, 0.5]))["pick"] == "full"
    c = server_choice(records([0.80, 0.82, 0.81, 0.79], [0.83, 0.83, 0.5, 0.5]))
    assert c["pick"] == "anchor c=0" and abs(c["R"] - 0.9) < 1e-12        # ties among anchored points go to the smaller c


def test_missing_run():
    r = records([0.8] * 4, [0.1] * 4); r["com"] = None
    assert server_choice(r) is None
