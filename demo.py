"""Console demo: watch the AI play, or play against it.

    python demo.py watch --> # Optimal vs Fixed-20, every decision explained
    python demo.py play  -->  # you vs Optimal AI
    python demo.py play --agent mcts  --> # you vs MCTS (--agent expecti | fixed | optimal)
"""
import argparse, random
from pig import *

ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["watch", "play"])
ap.add_argument("--agent", default="optimal", choices=["optimal", "mcts", "expecti", "fixed"])
ap.add_argument("--seed", type=int, default=None); args = ap.parse_args()
solver = PigSolver(100).solve(cache_path="policy_cache_100.npy")
make = {"optimal": lambda: OptimalAgent(solver), "mcts": lambda: MCTSAgent(2000),
        "expecti": lambda: ExpectiminimaxAgent(5), "fixed": lambda: FixedThresholdAgent(20)}
ai = make[args.agent]()
rng = random.Random(args.seed)

def explain(s):
    i, j, k = s
    r, h = solver.q_values(i, j, k)
    return (f"P(win) if roll={r:.3f}  hold={'n/a' if h is None else f'{h:.3f}'}  (exact optimal values)")

names = ["You", ai.name] if args.mode == "play" else ["Optimal", "Fixed-20"]
agents = [None, ai] if args.mode == "play" else [OptimalAgent(solver), FixedThresholdAgent(20)]
g = PigGame(100, rng)
while g.winner is None:
    s = g.state(); who = g.current
    print(f"\n{names[who]}'s turn | score {s[0]} vs {s[1]} | turn total {s[2]}")
    if agents[who] is None:
        if s[2] > 0: print("   hint:", explain(s))
        a = ""
        while a not in g.legal_actions():
            a = input(f"   {'/'.join(g.legal_actions())}? ").strip().lower()
    else:
        a = agents[who].choose_action(s); print("  ", explain(s)); print(f"   -> {names[who]} chooses {a.upper()}")
    die = g.step(a)
    if die: print(f"   rolled a {die}" + ("  (lost the turn!)" if die == 1 else ""))
print(f"\n{names[g.winner]} won!  Final scores: {g.scores}")
