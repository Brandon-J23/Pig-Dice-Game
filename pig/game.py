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
import random  # Provide the random number generator used to roll the die.


class PigGame:
    def __init__(self, target=100, rng=None):
        self.target = target  # Store the score needed to win.
        self.rng = rng or random.Random()  # Use the supplied RNG or create a new one.
        self.scores = [0, 0]   # Initialize both players' banked scores to zero.
        self.turn_total = 0    # Initialize the unbanked points for the current turn.
        self.current = 0   # Track that player 0 starts the game.
        self.winner = None  # Store the winning player index, or None until someone wins.

    def state(self):
        # Return the mover's score first and the opponent's score second
        return (self.scores[self.current], self.scores[1 - self.current], self.turn_total)

    def legal_actions(self):
        # At turn_total 0, holding would bank no points, so only roll is allowed.
        if self.turn_total == 0:
            return ["roll"]
        # Once points are accumulated, allow the player to continue rolling or hold.
        return ["roll", "hold"]

    #Apply an action. Returns the die value if the action was 'roll', else None
    def step(self, action):
        if self.winner is not None:
            # Prevent actions after the game has already ended.
            raise RuntimeError("game is over")
        if action not in self.legal_actions():
            # Reject actions that are not valid in the current game state.
            raise ValueError(f"illegal action {action!r} in state {self.state()}")
        die = None  # Track whether a die was rolled and store its value.
        if action == "roll":
            die = self.rng.randint(1, 6) 
            if die == 1:
                # A roll of 1 loses all unbanked points for this turn
                self.turn_total = 0
                # Pass the turn by switching to the other player index
                self.current = 1 - self.current
            else:
                # Add a successful roll to the current turn total.
                self.turn_total += die
                # A player wins immediately if banked and turn points reach the target.
                if self.scores[self.current] + self.turn_total >= self.target:
                    # Bank the turn total before declaring a winner.
                    self.scores[self.current] += self.turn_total
                    self.winner = self.current
        else:  # The only remaining legal action is hold.
            # Add the accumulated turn points to the player's banked score.
            self.scores[self.current] += self.turn_total
            # Clear the turn total because those points are now safely banked.
            self.turn_total = 0
            self.current = 1 - self.current
        return die  
