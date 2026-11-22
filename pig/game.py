"""Pig game engine (agent-agnostic).

Rules used: on your turn you repeatedly Roll or Hold.
  * Roll a 1  -> turn total is lost, turn passes.
  * Roll 2-6  -> added to turn total.
  * Hold      -> turn total is banked, turn passes.  (Hold is illegal at turn_total == 0.)
The game ends the moment a player's score + turn_total reaches `target`
(equivalent to "hold and win", since holding is always optimal then).

State (always from the point of view of the player to move):
    (my_score, opp_score, turn_total)
"""
import random

class PigGame:
    def __init__(self, target=100, rng=None):
        self.target = target
        self.rng = rng or random.Random()  # inject an rng for reproducible/seeded games
        self.scores = [0, 0]
        self.turn_total = 0
        self.current = 0       # index of the player whose turn it is
        self.winner = None

    def state(self):
        # Always returned from the mover's point of view -- (my_score, opp_score, turn_total) --
        # so a single agent implementation can be used for either seat.
        return (self.scores[self.current], self.scores[1 - self.current], self.turn_total)

    def legal_actions(self):
        # Hold is illegal at turn_total == 0: you can't bank nothing, and this also guarantees
        # every turn involves at least one roll.
        return ["roll"] if self.turn_total == 0 else ["roll", "hold"]

    def step(self, action):
        """Apply an action. Returns the die value if the action was 'roll', else None."""
        if self.winner is not None:
            raise RuntimeError("game is over")
        if action not in self.legal_actions():
            raise ValueError(f"illegal action {action!r} in state {self.state()}")
        die = None
        if action == "roll":
            die = self.rng.randint(1, 6)
            if die == 1:
                # Bust: lose the whole turn total, turn passes to the other player.
                self.turn_total = 0
                self.current = 1 - self.current
            else:
                self.turn_total += die
                # Win check happens immediately after a roll (not just on hold), since
                # reaching target this way is equivalent to "hold and win".
                if self.scores[self.current] + self.turn_total >= self.target:
                    self.scores[self.current] += self.turn_total
                    self.winner = self.current
        else:  # hold
            self.scores[self.current] += self.turn_total
            self.turn_total = 0
            self.current = 1 - self.current
        return die
