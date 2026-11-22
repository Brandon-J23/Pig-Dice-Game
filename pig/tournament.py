import math
import random
from .game import PigGame

def play_game(agents, target=100, rng=None):
    """agents = [agent_for_player0, agent_for_player1]; returns winner index."""
    g = PigGame(target, rng)
    while g.winner is None:
        g.step(agents[g.current].choose_action(g.state()))
    return g.winner


def head_to_head(a, b, n_games=2000, target=100, seed=0):
    """Win rate of agent `a` vs `b`, alternating who moves first. Returns (rate, 95% CI half-width).
    Alternating first-mover matters a lot here: Pig has a first-player win-rate advantage
    (~53% under optimal play, see OptimalAgent), so without alternating, a weaker agent could
    look artificially better (or worse) just from always moving first."""
    rng = random.Random(seed)
    wins = 0
    for n in range(n_games):
        if n % 2 == 0:
            wins += play_game([a, b], target, rng) == 0   # a is player 0 (moves first)
        else:
            wins += play_game([b, a], target, rng) == 1   # a is player 1 (moves second)
    p = wins / n_games
    # Normal approximation to a binomial proportion CI (Wald interval). max(..., 1e-12) just
    # guards against sqrt(0) at p == 0 or p == 1, which would otherwise give a half-width of
    # exactly 0 and look more confident than 2000 games actually warrants.
    return p, 1.96 * math.sqrt(max(p * (1 - p), 1e-12) / n_games)
