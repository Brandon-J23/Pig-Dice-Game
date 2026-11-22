"""Runs every experiment for the report and writes CSVs/PNGs/JSON to ./results.

    python run_experiments.py --quick     # ~ a few minutes, noisy
    python run_experiments.py             # full run (longer), report-quality
"""
import argparse, csv, json, os, random, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pig import *

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
Q = args.quick
N_GAMES = 1000 if Q else 10000           # games for cheap agents
N_SLOW = 40 if Q else 200               # games for slow agents (MCTS / deep search)
N_STATES = 60 if Q else 100              # states for MCTS-vs-optimal comparison
MCTS_SIMS = [100, 400, 2000] if Q else [100, 400, 2000, 10000]
DEPTHS = [1, 2, 3, 4, 5] if Q else [1, 2, 3, 4, 5, 6]
OUT = "results"; os.makedirs(OUT, exist_ok=True)
TARGET = 100
results = {}

def save_csv(name, header, rows):
    with open(f"{OUT}/{name}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

# ------------------------------------------------------------------ solve
print("== Solving exact optimal policy (value iteration)")
t = time.time()
solver = PigSolver(TARGET).solve(cache_path=f"policy_cache_{TARGET}.npy")
print(f"   done in {time.time()-t:.1f}s (instant if cached; ~10s cold, ~400 sweeps)")
optimal = OptimalAgent(solver)
fixed20 = FixedThresholdAgent(20, TARGET)
results["first_player_win_prob_optimal"] = solver.win_prob(0, 0, 0)

# ----------------------------------------- E1: optimal vs fixed thresholds
print("== E1: Optimal vs fixed-threshold baselines")
rows = []
for thr in [10, 15, 20, 25, 30, 35, 40]:
    p, ci = head_to_head(optimal, FixedThresholdAgent(thr, TARGET), N_GAMES, TARGET, seed=thr)
    rows.append((thr, p, ci)); print(f"   Optimal vs Fixed-{thr}: {p:.3f} +/- {ci:.3f}")
save_csv("e1_optimal_vs_fixed", ["fixed_threshold", "optimal_win_rate", "ci95"], rows)
results["e1_optimal_vs_fixed20"] = [r for r in rows if r[0] == 20][0][1]
plt.figure(figsize=(6, 4))
plt.errorbar([r[0] for r in rows], [r[1] for r in rows], yerr=[r[2] for r in rows], marker="o")
plt.axhline(0.5, color="gray", ls="--"); plt.xlabel("opponent's fixed hold threshold")
plt.ylabel("Optimal agent win rate"); plt.title("Optimal vs fixed-threshold baselines")
plt.tight_layout(); plt.savefig(f"{OUT}/e1_optimal_vs_fixed.png", dpi=150); plt.close()

# Which fixed threshold is best against itself-ish? round robin
print("== E1b: round-robin of fixed thresholds")
ths = [15, 20, 25, 30]
rr = [[head_to_head(FixedThresholdAgent(a, TARGET), FixedThresholdAgent(b, TARGET),
                    N_GAMES // 2, TARGET, seed=a * 100 + b)[0] for b in ths] for a in ths]
save_csv("e1b_fixed_round_robin", ["row_vs_col"] + ths, [[a] + r for a, r in zip(ths, rr)])

# ---------------------------------------------- E2: policy heatmaps
print("== E2: threshold heat-map")
thr = solver.thresholds()
mask = np.arange(TARGET).reshape(-1, 1) + thr > TARGET - 1   # cells where you'd only hold to win
plt.figure(figsize=(6.5, 5))
plt.imshow(np.where(mask, np.nan, thr).T, origin="lower", cmap="viridis", vmin=10, vmax=80)
plt.colorbar(label="optimal hold threshold (turn total)")
plt.xlabel("my score"); plt.ylabel("opponent score")
plt.title("Optimal hold threshold depends on BOTH scores"); plt.tight_layout()
plt.savefig(f"{OUT}/e2_threshold_heatmap.png", dpi=150); plt.close()
plt.figure(figsize=(6.5, 5))
plt.imshow(solver.V[:, :, 0].T, origin="lower", cmap="RdYlGn"); plt.colorbar(label="P(mover wins)")
plt.xlabel("my score"); plt.ylabel("opponent score"); plt.title("Win probability at start of turn")
plt.tight_layout(); plt.savefig(f"{OUT}/e2_winprob_heatmap.png", dpi=150); plt.close()
results["optimal_threshold_examples"] = {"0v0": int(thr[0, 0]), "80v80": int(thr[80, 80]),
                                         "20v90 (far behind)": int(thr[20, 90]), "90v20 (far ahead)": int(thr[90, 20])}
print("   ", results["optimal_threshold_examples"])
# export for the JS front-end
json.dump({"target": TARGET, "thresholds": thr.tolist(),
           "win_prob": np.round(solver.V[:, :, 0], 4).tolist()}, open(f"{OUT}/policy.json", "w"))

# ------------------------- shared: sample decision states from real play
def sample_states(n, seed=1):
    rng = random.Random(seed); states = []
    while len(states) < n * 5:
        a, b = FixedThresholdAgent(rng.choice(range(12, 32)), TARGET), FixedThresholdAgent(rng.choice(range(12, 32)), TARGET)
        g = PigGame(TARGET, rng)
        while g.winner is None:
            s = g.state()
            if s[2] > 0 and s[0] + s[2] < TARGET: states.append(s)
            g.step([a, b][g.current].choose_action(s))
    rng.shuffle(states); return states[:n]
states = sample_states(N_STATES)
gaps = [abs(r - h) for r, h in (solver.q_values(*s) for s in states)]
contested = [g < 0.05 for g in gaps]
print(f"   sampled {len(states)} states, {sum(contested)} contested (|Q_roll-Q_hold|<0.05)")

def regret(s, act):
    r, h = solver.q_values(*s); best = max(r, h)
    return best - (r if act == "roll" else h)

def eval_agent_on_states(agent):
    agree, reg, t0 = [], [], time.time()
    for s in states:
        a = agent.choose_action(s); agree.append(a == solver.best_action(*s)); reg.append(regret(s, a))
    ms = 1000 * (time.time() - t0) / len(states)
    ca = [x for x, c in zip(agree, contested) if c]
    return np.mean(agree), (np.mean(ca) if ca else float("nan")), np.mean(reg), ms

# --------------------------------------- E3: expectiminimax depth sweep
print("== E3: depth-limited expectiminimax")
rows = []
for d in DEPTHS:
    ag = ExpectiminimaxAgent(d, TARGET)
    agree, agree_c, reg, ms = eval_agent_on_states(ag)
    p, ci = head_to_head(ag, fixed20, N_SLOW * 2, TARGET, seed=d)
    rows.append((d, agree, agree_c, reg, ms, p, ci))
    print(f"   depth {d}: agree={agree:.2f} contested={agree_c:.2f} regret={reg:.4f} {ms:.1f}ms/dec  vs Fixed-20 {p:.3f}")
save_csv("e3_expectiminimax_depth", ["depth", "agree_all", "agree_contested", "mean_regret", "ms_per_decision", "winrate_vs_fixed20", "ci95"], rows)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(DEPTHS, [r[1] for r in rows], marker="o", label="all states")
ax[0].plot(DEPTHS, [r[2] for r in rows], marker="s", label="contested states")
ax[0].set_xlabel("search depth"); ax[0].set_ylabel("agreement with optimal action"); ax[0].legend()
ax[1].plot(DEPTHS, [r[5] for r in rows], marker="o"); ax[1].axhline(0.5, color="gray", ls="--")
ax[1].set_xlabel("search depth"); ax[1].set_ylabel("win rate vs Fixed-20")
plt.tight_layout(); plt.savefig(f"{OUT}/e3_depth.png", dpi=150); plt.close()

# ------------------------------------------------------- E4: MCTS study
print("== E4: MCTS convergence toward the optimal policy")
rows = []
for rollout in ["threshold", "random"]:
    for n in MCTS_SIMS:
        ag = MCTSAgent(n, rollout=rollout, target=TARGET, seed=n)
        agree, agree_c, reg, ms = eval_agent_on_states(ag)
        rows.append((rollout, n, agree, agree_c, reg, ms))
        print(f"   {rollout:9s} sims={n:5d}: agree={agree:.2f} contested={agree_c:.2f} regret={reg:.4f} {ms:.0f}ms/dec")
save_csv("e4_mcts_convergence", ["rollout", "n_sim", "agree_all", "agree_contested", "mean_regret", "ms_per_decision"], rows)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for ro in ["threshold", "random"]:
    rr = [r for r in rows if r[0] == ro]
    ax[0].plot([r[1] for r in rr], [r[2] for r in rr], marker="o", label=f"{ro} rollouts")
    ax[1].plot([r[1] for r in rr], [r[4] for r in rr], marker="o", label=f"{ro} rollouts")
for a, yl in zip(ax, ["agreement with optimal action", "mean win-prob regret"]):
    a.set_xscale("log"); a.set_xlabel("MCTS simulations per decision"); a.set_ylabel(yl); a.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/e4_mcts_convergence.png", dpi=150); plt.close()

print("== E4b: MCTS head-to-head")
rows = []
for n in [400, 2000]:
    ag = MCTSAgent(n, target=TARGET, seed=7)
    p1, c1 = head_to_head(ag, fixed20, N_SLOW, TARGET, seed=n)
    p2, c2 = head_to_head(ag, optimal, N_SLOW, TARGET, seed=n)
    rows.append((n, p1, c1, p2, c2)); print(f"   MCTS-{n}: vs Fixed-20 {p1:.3f}  vs Optimal {p2:.3f}")
save_csv("e4b_mcts_head_to_head", ["n_sim", "vs_fixed20", "ci95", "vs_optimal", "ci95"], rows)

# ---------------------------------------------------- E5: runtime summary
print("== E5: runtime per decision (ms)")
rt = {}
for name, ag in [("Fixed-20", fixed20), ("Optimal (lookup)", optimal),
                 ("Expectiminimax d4", ExpectiminimaxAgent(4, TARGET)), ("MCTS-2000", MCTSAgent(2000, target=TARGET, seed=1))]:
    t0 = time.time()
    for s in states: ag.choose_action(s)
    rt[name] = 1000 * (time.time() - t0) / len(states); print(f"   {name}: {rt[name]:.3f} ms")
results["runtime_ms_per_decision"] = rt
json.dump(results, open(f"{OUT}/summary.json", "w"), indent=2, default=float)
print("\nAll results written to ./results")
