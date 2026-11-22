"""Runs every experiment for the report and writes CSVs/PNGs/JSON to ./results.

    python run_experiments.py --quick     # ~ a few minutes, noisy
    python run_experiments.py             # full run (longer), report-quality
"""
import argparse, csv, json, os, random, time  # Import command-line, file, random, and timing tools.
import numpy as np  # Import arrays and numerical operations for policy tables.
import matplotlib  # Import Matplotlib for creating result graphs.
matplotlib.use("Agg")  # Use the non-display backend so plots can be saved in headless runs.
import matplotlib.pyplot as plt  # Import plotting functions.
from pig import *  # Import the game, solver, tournament, and agent classes.

ap = argparse.ArgumentParser()  # Create the command-line argument parser.
ap.add_argument("--quick", action="store_true")  # Add a flag for a shorter test run.
args = ap.parse_args()  # Read arguments supplied when the script is launched.
Q = args.quick  # Store whether quick mode is enabled.
N_GAMES = 1000 if Q else 10000           # games for cheap agents
N_SLOW = 40 if Q else 200               # games for slow agents (MCTS / deep search)
N_STATES = 60 if Q else 100              # states for MCTS-vs-optimal comparison
MCTS_SIMS = [100, 400, 2000] if Q else [100, 400, 2000, 10000]
DEPTHS = [1, 2, 3, 4, 5] if Q else [1, 2, 3, 4, 5, 6]
OUT = "results"; os.makedirs(OUT, exist_ok=True)  # Use a results folder and create it if needed.
TARGET = 100  # Set Pig's winning score for every experiment.
results = {}  # Store summary values that will be written to JSON.

def save_csv(name, header, rows):
    with open(f"{OUT}/{name}.csv", "w", newline="") as f:  # Open a CSV file for writing.
        w = csv.writer(f); w.writerow(header); w.writerows(rows)  # Write the header and every result row.

# ------------------------------------------------------------------ solve
print("== Solving exact optimal policy (value iteration)")  # Announce the first experiment stage.
t = time.time()  # Record the start time for solver timing.
solver = PigSolver(TARGET).solve(cache_path=f"policy_cache_{TARGET}.npy")  # Solve or load the optimal policy.
print(f"   done in {time.time()-t:.1f}s (instant if cached; ~10s cold, ~400 sweeps)")  # Report solve time.
optimal = OptimalAgent(solver)  # Create an agent that follows the solved policy.
fixed20 = FixedThresholdAgent(20, TARGET)  # Create the 20-point fixed-threshold baseline.
results["first_player_win_prob_optimal"] = solver.win_prob(0, 0, 0)  # Record the opening-state win probability.

# ----------------------------------------- E1: optimal vs fixed thresholds
print("== E1: Optimal vs fixed-threshold baselines")
rows = []  # Initialize rows for threshold comparison data.
for thr in [10, 15, 20, 25, 30, 35, 40]:  # Test several fixed hold thresholds.
    p, ci = head_to_head(optimal, FixedThresholdAgent(thr, TARGET), N_GAMES, TARGET, seed=thr)  # Run each head-to-head matchup.
    rows.append((thr, p, ci)); print(f"   Optimal vs Fixed-{thr}: {p:.3f} +/- {ci:.3f}")  # Save and print the result.
save_csv("e1_optimal_vs_fixed", ["fixed_threshold", "optimal_win_rate", "ci95"], rows)  # Export the comparison table.
results["e1_optimal_vs_fixed20"] = [r for r in rows if r[0] == 20][0][1]  # Store the Fixed-20 matchup result.
plt.figure(figsize=(6, 4))  # Create a graph for threshold win rates.
plt.errorbar([r[0] for r in rows], [r[1] for r in rows], yerr=[r[2] for r in rows], marker="o")  # Plot win rates with confidence bars.
plt.axhline(0.5, color="gray", ls="--"); plt.xlabel("opponent's fixed hold threshold")  # Add a reference line and x-axis label.
plt.ylabel("Optimal agent win rate"); plt.title("Optimal vs fixed-threshold baselines")  # Label the y-axis and graph.
plt.tight_layout(); plt.savefig(f"{OUT}/e1_optimal_vs_fixed.png", dpi=150); plt.close()  # Save and close the plot.

# Which fixed threshold is best against itself-ish? round robin
print("== E1b: round-robin of fixed thresholds")  # Announce the fixed-agent round robin.
ths = [15, 20, 25, 30]  # Choose thresholds for every matchup.
rr = [[head_to_head(FixedThresholdAgent(a, TARGET), FixedThresholdAgent(b, TARGET),
                    N_GAMES // 2, TARGET, seed=a * 100 + b)[0] for b in ths] for a in ths]  # Compare every row agent with every column agent.
save_csv("e1b_fixed_round_robin", ["row_vs_col"] + ths, [[a] + r for a, r in zip(ths, rr)])  # Write the matchup matrix.

# ---------------------------------------------- E2: policy heatmaps
print("== E2: threshold heat-map")  # Announce the policy visualization experiment.
thr = solver.thresholds()  # Extract optimal hold thresholds from the solved policy.
mask = np.arange(TARGET).reshape(-1, 1) + thr > TARGET - 1   # cells where you'd only hold to win
plt.figure(figsize=(6.5, 5))  # Create the threshold heat-map figure.
plt.imshow(np.where(mask, np.nan, thr).T, origin="lower", cmap="viridis", vmin=10, vmax=80)  # Plot thresholds, hiding forced-win cells.
plt.colorbar(label="optimal hold threshold (turn total)")  # Add a scale for hold thresholds.
plt.xlabel("my score"); plt.ylabel("opponent score")  # Label both score axes.
plt.title("Optimal hold threshold depends on BOTH scores"); plt.tight_layout()  # Add a title and tighten layout.
plt.savefig(f"{OUT}/e2_threshold_heatmap.png", dpi=150); plt.close()  # Save and close the threshold plot.
plt.figure(figsize=(6.5, 5))  # Create the opening-turn win-probability figure.
plt.imshow(solver.V[:, :, 0].T, origin="lower", cmap="RdYlGn"); plt.colorbar(label="P(mover wins)")  # Plot win probabilities.
plt.xlabel("my score"); plt.ylabel("opponent score"); plt.title("Win probability at start of turn")  # Label and title the plot.
plt.tight_layout(); plt.savefig(f"{OUT}/e2_winprob_heatmap.png", dpi=150); plt.close()  # Save and close the win-probability plot.
results["optimal_threshold_examples"] = {"0v0": int(thr[0, 0]), "80v80": int(thr[80, 80]),
                                         "20v90 (far behind)": int(thr[20, 90]), "90v20 (far ahead)": int(thr[90, 20])}  # Record representative thresholds.
print("   ", results["optimal_threshold_examples"])  # Print the representative threshold values.
# export for the JS front-end
json.dump({"target": TARGET, "thresholds": thr.tolist(),  # Write the target and threshold table to JSON.
           "win_prob": np.round(solver.V[:, :, 0], 4).tolist()}, open(f"{OUT}/policy.json", "w"))  # Include rounded win probabilities.

# ------------------------- shared: sample decision states from real play
def sample_states(n, seed=1):
    rng = random.Random(seed); states = []  # Create a seeded RNG and empty list of sampled states.
    while len(states) < n * 5:  # Generate extra states to allow random downsampling.
        a, b = FixedThresholdAgent(rng.choice(range(12, 32)), TARGET), FixedThresholdAgent(rng.choice(range(12, 32)), TARGET)  # Choose random thresholds for both players.
        g = PigGame(TARGET, rng)  # Create a game that uses the sampling RNG.
        while g.winner is None:  # Run until the sampled game finishes.
            s = g.state()  # Capture the current state.
            if s[2] > 0 and s[0] + s[2] < TARGET: states.append(s)  # Keep only active, non-winning turn states.
            g.step([a, b][g.current].choose_action(s))  # Let the current fixed agent act and update the game.
    rng.shuffle(states); return states[:n]  # Randomize and return exactly n sampled states.
states = sample_states(N_STATES)  # Build the state set used by later agent evaluations.
gaps = [abs(r - h) for r, h in (solver.q_values(*s) for s in states)]  # Calculate each state's roll-versus-hold difference.
contested = [g < 0.05 for g in gaps]  # Identify decisions with a value gap below 0.05.
print(f"   sampled {len(states)} states, {sum(contested)} contested (|Q_roll-Q_hold|<0.05)")  # Report sample and contested-state counts.

def regret(s, act):
    r, h = solver.q_values(*s); best = max(r, h)  # Find the roll value, hold value, and optimum.
    return best - (r if act == "roll" else h)  # Return the value lost by choosing the agent's action.

def eval_agent_on_states(agent):
    agree, reg, t0 = [], [], time.time()  # Initialize agreement, regret, and timing lists.
    for s in states:  # Evaluate the agent in every sampled state.
        a = agent.choose_action(s); agree.append(a == solver.best_action(*s)); reg.append(regret(s, a))  # Compare choice and measure regret.
    ms = 1000 * (time.time() - t0) / len(states)  # Calculate milliseconds used per decision.
    ca = [x for x, c in zip(agree, contested) if c]  # Filter agreement values to contested states.
    return np.mean(agree), (np.mean(ca) if ca else float("nan")), np.mean(reg), ms  # Return all evaluation metrics.

# --------------------------------------- E3: expectiminimax depth sweep
print("== E3: depth-limited expectiminimax")
rows = []  # Initialize depth-sweep result rows.
for d in DEPTHS:  # Test each configured search depth.
    ag = ExpectiminimaxAgent(d, TARGET)  # Create an expectiminimax agent at depth d.
    agree, agree_c, reg, ms = eval_agent_on_states(ag)  # Evaluate policy agreement, regret, and runtime.
    p, ci = head_to_head(ag, fixed20, N_SLOW * 2, TARGET, seed=d)  # Compare the depth-limited agent with Fixed-20.
    rows.append((d, agree, agree_c, reg, ms, p, ci))  # Store the depth experiment results.
    print(f"   depth {d}: agree={agree:.2f} contested={agree_c:.2f} regret={reg:.4f} {ms:.1f}ms/dec  vs Fixed-20 {p:.3f}")  # Print depth metrics.
save_csv("e3_expectiminimax_depth", ["depth", "agree_all", "agree_contested", "mean_regret", "ms_per_decision", "winrate_vs_fixed20", "ci95"], rows)  # Export depth-sweep results.
fig, ax = plt.subplots(1, 2, figsize=(10, 4))  # Create two side-by-side depth plots.
ax[0].plot(DEPTHS, [r[1] for r in rows], marker="o", label="all states")  # Plot agreement across every sampled state.
ax[0].plot(DEPTHS, [r[2] for r in rows], marker="s", label="contested states")  # Plot agreement across contested states.
ax[0].set_xlabel("search depth"); ax[0].set_ylabel("agreement with optimal action"); ax[0].legend()  # Label the agreement graph.
ax[1].plot(DEPTHS, [r[5] for r in rows], marker="o"); ax[1].axhline(0.5, color="gray", ls="--")  # Plot win rate and the 50-percent reference.
ax[1].set_xlabel("search depth"); ax[1].set_ylabel("win rate vs Fixed-20")  # Label the head-to-head graph.
plt.tight_layout(); plt.savefig(f"{OUT}/e3_depth.png", dpi=150); plt.close()  # Save and close the depth plot.

# ------------------------------------------------------- E4: MCTS study
print("== E4: MCTS convergence toward the optimal policy")
rows = []  # Initialize MCTS convergence rows.
for rollout in ["threshold", "random"]:  # Compare threshold and random rollout policies.
    for n in MCTS_SIMS:  # Test each configured simulation count.
        ag = MCTSAgent(n, rollout=rollout, target=TARGET, seed=n)  # Create a seeded MCTS agent.
        agree, agree_c, reg, ms = eval_agent_on_states(ag)  # Measure policy agreement and runtime.
        rows.append((rollout, n, agree, agree_c, reg, ms))  # Store this MCTS configuration.
        print(f"   {rollout:9s} sims={n:5d}: agree={agree:.2f} contested={agree_c:.2f} regret={reg:.4f} {ms:.0f}ms/dec")  # Print MCTS metrics.
save_csv("e4_mcts_convergence", ["rollout", "n_sim", "agree_all", "agree_contested", "mean_regret", "ms_per_decision"], rows)  # Export MCTS convergence data.
fig, ax = plt.subplots(1, 2, figsize=(10, 4))  # Create agreement and regret plots.
for ro in ["threshold", "random"]:  # Draw a series for each rollout mode.
    rr = [r for r in rows if r[0] == ro]  # Filter rows for the current rollout mode.
    ax[0].plot([r[1] for r in rr], [r[2] for r in rr], marker="o", label=f"{ro} rollouts")  # Plot agreement versus simulation count.
    ax[1].plot([r[1] for r in rr], [r[4] for r in rr], marker="o", label=f"{ro} rollouts")  # Plot regret versus simulation count.
for a, yl in zip(ax, ["agreement with optimal action", "mean win-prob regret"]):  # Configure both graph axes.
    a.set_xscale("log"); a.set_xlabel("MCTS simulations per decision"); a.set_ylabel(yl); a.legend()  # Use a log scale and add labels.
plt.tight_layout(); plt.savefig(f"{OUT}/e4_mcts_convergence.png", dpi=150); plt.close()  # Save and close the MCTS plot.

print("== E4b: MCTS head-to-head")
rows = []  # Initialize MCTS head-to-head result rows.
for n in [400, 2000]:  # Test two MCTS search budgets.
    ag = MCTSAgent(n, target=TARGET, seed=7)  # Create an MCTS agent with n simulations.
    p1, c1 = head_to_head(ag, fixed20, N_SLOW, TARGET, seed=n)  # Measure the MCTS win rate against Fixed-20.
    p2, c2 = head_to_head(ag, optimal, N_SLOW, TARGET, seed=n)  # Measure the MCTS win rate against the optimal agent.
    rows.append((n, p1, c1, p2, c2)); print(f"   MCTS-{n}: vs Fixed-20 {p1:.3f}  vs Optimal {p2:.3f}")  # Store and print both matchups.
save_csv("e4b_mcts_head_to_head", ["n_sim", "vs_fixed20", "ci95", "vs_optimal", "ci95"], rows)  # Export MCTS head-to-head results.

# ---------------------------------------------------- E5: runtime summary
print("== E5: runtime per decision (ms)")
rt = {}  # Create a dictionary for runtime measurements.
for name, ag in [("Fixed-20", fixed20), ("Optimal (lookup)", optimal),
                 ("Expectiminimax d4", ExpectiminimaxAgent(4, TARGET)), ("MCTS-2000", MCTSAgent(2000, target=TARGET, seed=1))]:  # Compare representative agent implementations.
    t0 = time.time()  # Record the start time for this agent.
    for s in states: ag.choose_action(s)  # Have the agent choose actions for every shared state.
    rt[name] = 1000 * (time.time() - t0) / len(states); print(f"   {name}: {rt[name]:.3f} ms")  # Calculate and print average milliseconds per decision.
results["runtime_ms_per_decision"] = rt  # Add runtime data to the summary results.
json.dump(results, open(f"{OUT}/summary.json", "w"), indent=2, default=float)  # Write all summary values to JSON.
print("\nAll results written to ./results")  # Confirm that experiment output is complete.
