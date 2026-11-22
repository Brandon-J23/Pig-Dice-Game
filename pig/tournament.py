import math  
import random  
from .game import PigGame  

def play_game(agents, target=100, rng=None):
    g = PigGame(target, rng)  # Create a game with the requested target and RNG
    while g.winner is None:  # Continue until the game engine records a winner
        g.step(agents[g.current].choose_action(g.state()))  # Ask the current agent to act and apply the action.
    return g.winner  

def head_to_head(a, b, n_games=2000, target=100, seed=0):
    rng = random.Random(seed)  
    wins = 0  # Initialize the number of games won by agent a.
    for n in range(n_games):  # Run every scheduled match.
        if n % 2 == 0:  # Alternate positions on even-numbered games
            wins += play_game([a, b], target, rng) == 0   # a is player 0 (moves first)
        else:  # Place agent a second on odd-numbered games
            wins += play_game([b, a], target, rng) == 1   # a is player 1 (moves second)
    p = wins / n_games  # Calculate agent a's observed win rate

    margin = 1.96 * math.sqrt(max(p * (1 - p), 1e-12) / n_games)  # Calculate the 95% interval half-width.
    return p, margin  # Return the win rate and confidence-interval margin
