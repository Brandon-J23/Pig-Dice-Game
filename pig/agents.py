"""Four interchangeable agents; each exposes choose_action(state) -> 'roll' | 'hold'.
state = (my_score, opp_score, turn_total)."""
import math
import random
import sys

sys.setrecursionlimit(20000)  # Allow deeper game-search recursion.

# Fixed-threshold baseline agent.
class FixedThresholdAgent:
    def __init__(self, threshold=20, target=100):
        self.threshold, self.target = threshold, target
        self.name = f"Fixed-{threshold}"

    def choose_action(self, state):
        i, j, k = state
        # Hold only after accumulating points or reaching the target.
        return "hold" if k > 0 and (k >= self.threshold or i + k >= self.target) else "roll"

# Exact optimal policy agent.
class OptimalAgent:
    def __init__(self, solver):
        self.solver, self.name = solver, "Optimal"

    def choose_action(self, state):
        return self.solver.best_action(*state)

# Depth-limited expectiminimax agent.
class ExpectiminimaxAgent:
    """Searches roll and hold actions up to a limited depth."""

    def __init__(self, depth=4, target=100, scale=25.0):
        self.depth, self.T, self.scale = depth, target, scale
        self.name = f"Expectiminimax-d{depth}"
        self.cache = {}

    def heuristic(self, i, j, k):
        #Estimate win probability when the search reaches its depth limit
        score_difference = (i + k) - j
        return 1.0 / (1.0 + math.exp(-score_difference / self.scale))

    def _roll_hold(self, i, j, k, d):
        # Average all die outcomes for a roll.
        roll = (1.0 - self.value(j, i, 0, d - 1)) / 6.0
        for die in range(2, 7):
            roll += self.value(i, j, k + die, d - 1) / 6.0
        # Holding is illegal when the turn total is zero.
        hold = None if k == 0 else 1.0 - self.value(j, i + k, 0, d - 1)
        return roll, hold

    def value(self, i, j, k, d):
        if i + k >= self.T:
            return 1.0
        if d == 0:
            return self.heuristic(i, j, k)
        key = (i, j, k, d)
        if key not in self.cache:
            roll, hold = self._roll_hold(i, j, k, d)
            # Choose the better legal action.
            self.cache[key] = roll if hold is None else max(roll, hold)
        return self.cache[key]

    def choose_action(self, state):
        i, j, k = state
        if i + k >= self.T:
            return "hold"
        roll, hold = self._roll_hold(i, j, k, self.depth)
        return "hold" if hold is not None and hold > roll else "roll"

# MCTS agent.
class MCTSAgent:
    """Uses UCT Monte Carlo tree search with sampled dice rolls."""

    def __init__(self, n_simulations=2000, c=0.5, rollout="threshold", rollout_threshold=20, target=100, seed=None):
        self.n_sim, self.c, self.rollout = n_simulations, c, rollout
        self.rt, self.T = rollout_threshold, target
        self.rng = random.Random(seed)
        self.name = f"MCTS-{n_simulations}-{rollout}"
        self.last_stats = {}

    def _rollout(self, a, b, k):
        """Complete a random game after leaving the search tree."""
        me, T, rng = True, self.T, self.rng
        while True:
            if a + k >= T:
                return 1.0 if me else 0.0
            # Hold by threshold or randomly, depending on rollout mode.
            if k > 0 and ((k >= self.rt) if self.rollout == "threshold" else rng.random() < 0.5):
                a, b, k = b, a + k, 0
                me = not me
            else:
                die = rng.randint(1, 6)
                if die == 1:
                    a, b, k = b, a, 0
                    me = not me
                else:
                    k += die

    def _simulate(self, tree, s):
        #Run one MCTS simulation and update the tree
        a, b, k = s
        if a + k >= self.T:
            return 1.0
        node = tree.get(s)
        if node is None:
            # Add legal actions, then use a rollout to estimate this state.
            legal_actions = ["roll"] if k == 0 else ["roll", "hold"]
            tree[s] = {"N": 0, "acts": {x: [0, 0.0] for x in legal_actions}}
            return self._rollout(a, b, k)

        # Select the action with the highest UCB1 score.
        best, best_u = None, -1.0
        for act, (n, w) in node["acts"].items():
            u = float("inf") if n == 0 else w / n + self.c * math.sqrt(math.log(node["N"] + 1) / n)
            if u > best_u:
                best, best_u = act, u

        if best == "hold":
            v = 1.0 - self._simulate(tree, (b, a + k, 0))
        else:
            die = self.rng.randint(1, 6)
            if die == 1:
                v = 1.0 - self._simulate(tree, (b, a, 0))
            else:
                v = self._simulate(tree, (a, b, k + die))

        # Update visit counts and win totals for this node and action.
        node["N"] += 1
        node["acts"][best][0] += 1
        node["acts"][best][1] += v
        return v

    def choose_action(self, state):
        i, j, k = state
        if k == 0:
            return "roll"
        if i + k >= self.T:
            return "hold"
        tree = {}
        for _ in range(self.n_sim):
            self._simulate(tree, state)
        acts = tree[state]["acts"]
        self.last_stats = {a: (n, (w / n if n else 0.0)) for a, (n, w) in acts.items()}
        # Choose the action visited most often.
        return max(acts, key=lambda a: acts[a][0])
