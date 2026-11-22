"""Four interchangeable agents; each exposes choose_action(state) -> 'roll' | 'hold'.
state = (my_score, opp_score, turn_total)."""
import math
import random
import sys

sys.setrecursionlimit(20000)


# 1. First agent is the baseline agent, which is Fixed Threshold (hold at 20) with a 
# small improvement: if the turn total is enough to win, hold immediately
class FixedThresholdAgent:
    #Hold as soon as turn_total >= threshold (the classic 'hold at 20' rule)
    def __init__(self, threshold=20, target=100):
        self.threshold, self.target = threshold, target
        self.name = f"Fixed-{threshold}"

    def choose_action(self, state):
        i, j, k = state
        # k > 0 guard: holding at turn_total 0 is illegal, so always roll first
        # i + k >= target: bank the win immediately rather than waiting for the threshold
        return "hold" if k > 0 and (k >= self.threshold or i + k >= self.target) else "roll"

# 2. Second agent is the exact optimal (lookup) agent, which uses the PigSolver to compute the optimal policy for the game
class OptimalAgent:
    #Plays the exact expectiminimax-optimal policy computed by PigSolver
    def __init__(self, solver):
        # `solver` must already have .solve() called on it - this agent does no computation itself, it's a thin wrapper around the precomputed policy table
        self.solver, self.name = solver, "Optimal"

    def choose_action(self, state):
        return self.solver.best_action(*state)

# 3. This agent is depth-limited expectiminimax tree, which is a real tree search with MAX nodes (roll/hold) and CHANCE nodes (die, 1/6 each), cut off at a certain depth and scored by a hand-crafted heuristic. It shows how search depth trades off against the exact solution.
class ExpectiminimaxAgent:
    """Real tree search: MAX nodes (roll/hold) and CHANCE nodes (die, 1/6 each),
    cut off at `depth` decision levels and scored by a hand-crafted heuristic.
    Shows how search depth trades off against the exact solution."""
    def __init__(self, depth=4, target=100, scale=25.0):
        self.depth, self.T, self.scale = depth, target, scale
        self.name = f"Expectiminimax-d{depth}"
        # Memoises value(i, j, k, d) within a single choose_action call. Keyed on depth too,
        # since the same (i, j, k) has a different value at different remaining search depths.
        self.cache = {}

    def heuristic(self, i, j, k):
        """Leaf evaluation when the search is cut off: a logistic function of
        (my banked + turn total) - opponent's score, squashed into [0, 1] so it reads as a
        win probability. `scale` controls how quickly the estimate saturates toward 0 or 1
        as the lead grows -- larger scale = more conservative (slower to call it a blowout)."""
        return 1.0 / (1.0 + math.exp(-((i + k) - j) / self.scale))

    def _roll_hold(self, i, j, k, d):
        # ROLL branch, expectation over the die:
        #   -- die == 1 (prob 1/6): turn is lost, turns passes. Opponent becomes the player to
        #      move, so we evaluate value(j, i, 0, d-1) from *their* perspective and flip it
        #      to 1 - v to get our own win probability back.
        #   -- die in 2..6 (prob 1/6 each): turn total grows, same player continues, one ply
        #      of search depth is consumed (d - 1).
        roll = (1.0 - self.value(j, i, 0, d - 1)) / 6.0
        for die in range(2, 7):
            roll += self.value(i, j, k + die, d - 1) / 6.0
        # HOLD branch: bank k, turn passes to opponent -> again flip their value to ours.
        # Illegal at k == 0, signalled by returning None.
        hold = None if k == 0 else 1.0 - self.value(j, i + k, 0, d - 1)
        return roll, hold

    def value(self, i, j, k, d):
        if i + k >= self.T:
            return 1.0  # already a won position regardless of remaining depth
        if d == 0:
            return self.heuristic(i, j, k)  # search horizon reached -- estimate, don't expand
        key = (i, j, k, d)
        if key not in self.cache:
            roll, hold = self._roll_hold(i, j, k, d)
            # MAX node: player to move picks whichever action is better; hold is None (illegal)
            # exactly when k == 0, in which case roll is forced.
            self.cache[key] = roll if hold is None else max(roll, hold)
        return self.cache[key]

    def choose_action(self, state):
        i, j, k = state
        if i + k >= self.T:
            return "hold"  # take the win, no need to search
        roll, hold = self._roll_hold(i, j, k, self.depth)
        return "hold" if hold is not None and hold > roll else "roll"

# 4. Fourth agent is the MCTS
class MCTSAgent:
    """UCT Monte Carlo Tree Search with sampled chance nodes.

    Nodes are keyed by state (a transposition table). Each simulation:
      selection (UCB1) -> expansion (add one new node) -> rollout (default policy) -> backup.
    Values are stored as win-probability for the player to move at that node, and are
    flipped (1 - v) whenever the turn passes to the opponent.
    rollout: 'threshold' (both sides hold at `rollout_threshold`) or 'random' (hold w.p. 0.5).
    """
    def __init__(self, n_simulations=2000, c=0.5, rollout="threshold", rollout_threshold=20,
                 target=100, seed=None):
        self.n_sim, self.c, self.rollout = n_simulations, c, rollout
        self.rt, self.T = rollout_threshold, target
        # Seeded RNG so a run can be reproduced exactly when seed is given; defaults to
        # nondeterministic play otherwise.
        self.rng = random.Random(seed)
        self.name = f"MCTS-{n_simulations}-{rollout}"
        self.last_stats = {} # populated after choose_action(): {action: (visits, win_rate)}

    # default policy used inside rollouts
    def _rollout(self, a, b, k):
        """Default (random) policy used once we fall off the tree: play the rest of the game
        out to completion using a fixed, cheap rule, and return 1.0/0.0 for whether the
        player who was to move *at the root of this rollout* ultimately won.
        `me` tracks whose turn it currently is relative to that root player."""
        me, T, rng = True, self.T, self.rng
        while True:
            if a + k >= T:
                return 1.0 if me else 0.0
            # Rollout policy: hold once turn_total crosses the threshold ("threshold" mode),
            # or hold with 50/50 odds each decision point ("random" mode).
            if k > 0 and ((k >= self.rt) if self.rollout == "threshold" else rng.random() < 0.5):
                a, b, k = b, a + k, 0   # hold: bank k, swap seats, pass the turn
                me = not me
            else:
                die = rng.randint(1, 6)
                if die == 1:
                    a, b, k = b, a, 0   # bust: turn total lost, pass the turn
                    me = not me
                else:
                    k += die

    def _simulate(self, tree, s):
        """One MCTS simulation from state s. Returns the win probability for whichever
        player is to move *at s* -- note this is a *relative* value, so it gets flipped
        (1 - v) every time the recursion crosses a turn boundary, exactly like in
        ExpectiminimaxAgent._roll_hold."""
        a, b, k = s
        if a + k >= self.T:
            return 1.0
        node = tree.get(s)
        if node is None:
            # Expansion: first visit to this state in this search. Register it with zero
            # stats for each legal action, then fall back to a rollout instead of recursing
            # further (we don't yet know which action to explore).
            tree[s] = {"N": 0, "acts": {x: [0, 0.0] for x in (["roll"] if k == 0 else ["roll", "hold"])}}
            return self._rollout(a, b, k)
        # Selection: UCB1 over this node's actions. An action with n == 0 visits gets
        # infinite priority so every action is tried at least once before any is revisited.
        # c controls exploration vs. exploitation (higher c explores more).
        best, best_u = None, -1.0
        for act, (n, w) in node["acts"].items():
            u = float("inf") if n == 0 else w / n + self.c * math.sqrt(math.log(node["N"] + 1) / n)
            if u > best_u:
                best, best_u = act, u
        if best == "hold":
            # Recurse into the opponent's state, then flip their win-prob to get ours.
            v = 1.0 - self._simulate(tree, (b, a + k, 0))
        else:
            # Chance node: the die isn't selected by UCB1, it's sampled -- this is what makes
            # the search "expecti"-style rather than plain minimax/UCT.
            die = self.rng.randint(1, 6)
            if die == 1:
                v = 1.0 - self._simulate(tree, (b, a, 0))
            else:
                v = self._simulate(tree, (a, b, k + die))
        # Backup: update this node's visit count and the chosen action's running stats.
        node["N"] += 1
        node["acts"][best][0] += 1
        node["acts"][best][1] += v
        return v

    def choose_action(self, state):
        i, j, k = state
        if k == 0:
            return "roll"               # only legal move, skip the search entirely
        if i + k >= self.T:
            return "hold"               # winning move, skip the search entirely
        tree = {}                       # fresh transposition table each call (no reuse across turns)
        for _ in range(self.n_sim):
            self._simulate(tree, state)
        acts = tree[state]["acts"]
        self.last_stats = {a: (n, (w / n if n else 0.0)) for a, (n, w) in acts.items()}
        # Final move selection uses visit count, not win-rate estimate -- the standard,
        # more robust choice in UCT since visits correlate with confidence.
        return max(acts, key=lambda a: acts[a][0])
