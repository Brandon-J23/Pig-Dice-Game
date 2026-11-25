# Pig AI: Building and Evaluating Intelligent Agents for Pig Dice

An AI-powered implementation of the **Pig dice game** that explores several approaches to decision-making under uncertainty. This project was developed for a CSE 352 extra-credit assignment and compares rule-based strategies, adversarial search, Monte Carlo Tree Search (MCTS), and value iteration.

The central question behind the project is:

> How can an agent make better decisions when every roll introduces uncertainty, and the best decision depends on the current state of the game?

Rather than relying on a single fixed rule, the project investigates how different algorithms evaluate risk, plan ahead, and maximize their probability of winning.

---

## Table of Contents

- [Project Overview](#project-overview)
- [How Pig Works](#how-pig-works)
- [AI Approaches](#ai-approaches)
  - [Fixed-Threshold Agent](#1-fixed-threshold-agent)
  - [Expectiminimax](#2-expectiminimax)
  - [Monte Carlo Tree Search](#3-monte-carlo-tree-search-mcts)
  - [Value Iteration](#4-value-iteration)
- [Search Problem Formulation](#search-problem-formulation)
- [Experiments and Evaluation](#experiments-and-evaluation)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Running Experiments](#running-experiments)
- [Frontend and Demo](#frontend-and-demo)
- [Implementation Notes](#implementation-notes)

---

## Project Overview

Pig is a two-player dice game in which players must balance the potential reward of rolling again against the risk of losing their accumulated turn points.

This project implements four AI approaches and evaluates their performance across different game states and experimental settings.

| Approach | Decision-making strategy |
|---|---|
| Fixed Threshold | Uses a predefined number of turn points at which to hold |
| Expectiminimax | Searches possible actions and random dice outcomes using a heuristic evaluation function |
| Monte Carlo Tree Search | Uses repeated simulations and UCB1 selection to explore promising actions |
| Value Iteration | Computes an optimal policy for the modeled game through iterative Bellman updates |

The value-iteration solver serves as a reference for evaluating the other approaches. It calculates the expected probability of winning from each valid game state and determines which action provides the highest expected value.

### Project Goals

- Model Pig as a stochastic, turn-based decision problem.
- Implement multiple AI strategies with different approaches to planning and uncertainty.
- Compare heuristic search against simulation-based decision-making.
- Compute a reference optimal policy using value iteration.
- Evaluate agents using win rates, confidence intervals, policy agreement, and computational tradeoffs.
- Provide an interactive implementation that allows users to experience the game against an AI opponent.

---

## How Pig Works

The objective is to be the first player to reach **100 points**.

On each turn, a player chooses between two actions:

**Roll**
- Roll a six-sided die.
- If the result is between 2 and 6, add the result to the current turn total.
- If the result is 1, lose all unbanked points from that turn and pass the turn to the opponent.

**Hold**
- Add the accumulated turn total to the player's permanent score.
- End the turn and pass control to the opponent.
- Holding is unavailable when the turn total is zero.

A player wins immediately when their score plus their current turn total reaches or exceeds 100.

The key strategic decision is knowing when to stop rolling. Rolling can increase a player's score, but every additional roll carries a chance of losing the entire turn total.

---

## AI Approaches

### 1. Fixed-Threshold Agent

The fixed-threshold agent uses a straightforward strategy: continue rolling until the accumulated turn total reaches a predefined threshold, then hold.

For example, an agent with a threshold of 20 will:
- Roll while its turn total is below 20.
- Hold once its turn total reaches 20.
- Roll when its turn total is zero.
- Secure an immediate win whenever possible.

This approach provides a simple baseline for comparison. However, it does not account for the opponent's score or the broader game situation when deciding how much risk to take.

### 2. Expectiminimax

Expectiminimax extends traditional minimax search to environments containing random outcomes.

Unlike ordinary minimax, which alternates between maximizing and minimizing decisions, expectiminimax includes **chance nodes** to account for probabilistic events such as dice rolls.

The algorithm considers:
- **MAX nodes:** The current player chooses between available actions.
- **CHANCE nodes:** The die determines the next state, with each outcome having a probability of 1/6.

The agent evaluates possible sequences of decisions up to a configurable search depth. At the depth limit, it uses a heuristic evaluation function to estimate the desirability of the resulting state.

The heuristic produces a bounded value between 0 and 1. It is a probability-like estimate used for search evaluation, not a calibrated estimate of the actual probability of winning.

**Purpose:** Investigate how looking ahead through possible decisions and random outcomes affects agent performance.

### 3. Monte Carlo Tree Search (MCTS)

Monte Carlo Tree Search evaluates decisions through repeated simulations rather than relying exclusively on a manually designed evaluation function.

The implementation follows four main stages:

1. **Selection:** Traverse the search tree using the Upper Confidence Bound for Trees (UCB1) to balance exploration and exploitation.
2. **Expansion:** Add a new state or action to the tree.
3. **Simulation:** Play out the game using a rollout policy.
4. **Backpropagation:** Update the statistics of visited nodes using the simulation result.

Two rollout strategies were evaluated:

- **Threshold-20 rollout:** Uses a fixed threshold of 20 when deciding when to hold.
- **Random rollout:** Uses randomized hold decisions when holding is legal.

The agent ultimately selects the action with the most visits.

**Purpose:** Examine how simulation budgets and rollout strategies influence decision quality.

### 4. Value Iteration

Value iteration computes an optimal policy for the modeled game by repeatedly updating the expected value of each state.

The value of a state represents the probability that the current player eventually wins when following the optimal policy.

For each state, the solver evaluates the expected value of rolling and holding, then selects the action with the greater value.

The solver accounts for all six possible die outcomes, including the possibility of losing the accumulated turn total and passing control to the opponent.

Updates continue until the maximum change between successive iterations falls below a convergence tolerance of \(10^{-10}\).

**Why value iteration?**

The state graph contains cycles. For example, both players can repeatedly roll a 1, returning the game to the same permanent scores. Consequently, straightforward recursive evaluation can encounter repeated states and fail to terminate without additional handling.

Value iteration addresses this by repeatedly applying Bellman updates until the values converge.

The resulting policy provides a reference for evaluating the other agents. Its optimality is with respect to the modeled game and the numerical convergence tolerance.

---

## Search Problem Formulation

Pig can be represented as a stochastic, turn-based search problem.

### State Representation

Each state is represented from the perspective of the player whose turn it is.

The state representation is player-relative. When the turn changes, the players' scores are swapped to represent the game from the next player's perspective.

The implementation considers approximately **505,000 valid nonterminal states**.

### Actions

The available actions are:

- `ROLL`
- `HOLD`, provided the current turn total is greater than zero

### State Transitions

| Action / Outcome | Result |
|---|---|
| Roll a 1 | Lose the accumulated turn total and pass the turn |
| Roll a 2–6 | Add the die result to the turn total |
| Hold | Bank the turn total and pass the turn |
| Reach 100 points | End the game immediately with a win |

### Objective Function

The objective is to maximize the probability of winning, not simply the number of points accumulated.

This distinction matters because the highest-scoring immediate action is not necessarily the action that gives a player the best chance of winning the game.

---

## Experiments and Evaluation

The project evaluates the agents using several experiments, with results stored in the `results/` directory.

### Evaluation Metrics

**Win Rate**

Measures the percentage of games won by an agent across a collection of matches.

The tournament implementation alternates which player moves first to account for the advantage associated with turn order.

**95% Confidence Interval**

Provides an estimate of the uncertainty surrounding measured win rates.

**Policy Agreement**

Measures the percentage of evaluated states in which an agent selects the same action as the value-iteration reference policy.

Policy agreement and win rate measure different things: an agent can make decisions that frequently match the reference policy without necessarily achieving the same win rate in actual games.

**Computational Cost**

Search depth and simulation budget affect how much computation an agent performs. These parameters help illustrate the tradeoff between decision quality and computational expense.

### Key Findings

The experiments produced several interesting observations:

- **Adaptive decision-making matters.** The value-iteration policy adjusts its decisions based on both players' scores instead of relying on one fixed threshold.
- **Simple strategies can be competitive.** A fixed threshold around 20–25 performed surprisingly well in the tested matchups.
- **Search depth matters.** Expectiminimax policy agreement varied across search depths, illustrating that deeper heuristic search does not guarantee a monotonic improvement.
- **Simulation budgets matter.** MCTS agreement generally improved between the smallest and largest tested budgets, although intermediate results were not strictly monotonic.
- **Rollout policies influence MCTS.** The random rollout strategy achieved higher policy agreement than the threshold-based rollout at the largest tested simulation budget. This is an experimental observation, not a guarantee that random rollouts are always better.

### Selected Results

The following are selected results from the experiments:

| Experiment | Result |
|---|---:|
| First-player win probability under the reference policy | 53.06% |
| Reference policy vs. fixed threshold 20 | 54.46% |
| Expectiminimax policy agreement, depth 1 | 96% |
| Expectiminimax policy agreement, depth 6 | 95% |
| MCTS threshold rollout, 100 simulations | 59% |
| MCTS threshold rollout, 10,000 simulations | 85% |
| MCTS random rollout, 100 simulations | 72% |
| MCTS random rollout, 10,000 simulations | 91% |

These values are drawn from the project's recorded experiment results. They represent particular experimental configurations rather than universal performance guarantees.

---

## Project Structure

```text
Pig-Dice-Game/
│
├── pig/
│   ├── game.py              # Pig game engine and state transitions
│   ├── solver.py            # Value-iteration reference solver
│   ├── agents.py             # AI agent implementations
│   └── tournament.py        # Head-to-head games and win-rate evaluation
│
├── frontend/
│   ├── index.html            # Frontend structure
│   ├── style.css             # Interface styling
│   └── app.js                # Frontend game logic
│
├── results/
│   ├── *.csv                 # Recorded experiment results
│   ├── *.png                 # Experiment plots and visualizations
│   ├── policy.json           # Exported reference policy
│   └── summary.json          # Experiment summary
│
├── policy_cache_100.npy      # Cached solver data
├── server.py                 # HTTP server and backend API
├── demo.py                   # Command-line demonstration
├── test_backend.py           # Backend sanity checks
├── run_experiments.py        # Experiment runner
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Getting Started

### Requirements

- Python 3.9+
- pip

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Pig-Dice-Game
```

Replace `<YOUR_REPOSITORY_URL>` with your repository's Git URL.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Backend Tests

```bash
python test_backend.py
```

The test suite checks the game engine, solver, and reference results, including the expected first-player win probability of approximately 53.06%.

### 4. Run a Demonstration

```bash
python demo.py
```

This runs the command-line demonstration of the game and its agents.

---

## Running Experiments

To run a quicker version of the experiment suite:

```bash
python run_experiments.py --quick
```

For the full experiment configuration used to generate the report results:

```bash
python run_experiments.py
```

The full run takes considerably longer, particularly because MCTS performs repeated simulations across different configurations.

Generated experiment outputs are saved under:

```text
results/
```

This includes CSV data, plots, the exported policy, and a summary of the results.

---

## Frontend and Demo

The project also includes a browser-based frontend that allows users to interact with the Pig game and play against the AI.

The exported reference policy is stored in:

```text
results/policy.json
```

The policy contains:

- `thresholds[my][opp]`: score-dependent hold thresholds.
- `win_prob[my][opp]`: estimated winning probabilities for the corresponding score states.

This allows the frontend to use the precomputed policy without having to run the full value-iteration solver for every decision.

The repository also includes `server.py`, which provides the HTTP server and backend API used by the application.

To start the local server:

```bash
python server.py
```

Open the local address printed by the server in your browser.

---

## Implementation Notes

A few details are important when interpreting the results:

- **The reference policy is score-dependent.** It does not use one universal hold threshold. For example, the optimal threshold is 21 at a tied 0–0 score and 10 when far ahead.
- **The expectiminimax heuristic is approximate.** Its depth-limited evaluations are not equivalent to the value-iteration solver's converged state values.
- **MCTS depends on its rollout policy.** Different simulation policies can produce different estimates and action preferences.
- **Numerical convergence is tolerance-based.** Value iteration stops when the maximum update difference falls below \(10^{-10}\).
- **Reported results are experimental.** Win rates and policy agreement depend on the tested configurations and should not be interpreted as guarantees for every possible game.

---

## What I Learned

This project provided an opportunity to explore how different AI algorithms approach the same decision-making problem.

One of the most interesting takeaways was that a relatively simple strategy can perform well, but a strategy that accounts for the full game state can make more informed decisions. The project also highlighted that adding computational complexity does not automatically produce better results. Search depth, evaluation functions, rollout policies, and simulation budgets all influence performance.

Ultimately, the project demonstrates how probabilistic reasoning, search algorithms, and iterative optimization can be applied to a game where every decision involves balancing risk and reward.

---

**Built for CSE 352: Artificial Intelligence | Stony Brook University**
