"""Exact solution of Pig via value iteration on the expectiminimax equations.

V[i, j, k] = P(player to move wins) with own score i, opponent score j, turn total k, assuming both players play optimally.

    hold  = 1 - V[j, i+k, 0] --> (opponent moves next)
    roll  = 1/6 * (1 - V[j, i, 0])  --> (rolled a 1: lose turn total)
          + 1/6 * sum_{d=2..6} V[i, j, k+d] --> (rolled d: same player continues)
    V[i,j,k] = max(hold, roll)   --> hold illegal when k == 0;  V = 1 when i+k >= target)

Pig's state graph has CYCLES (V[a,b,0] -> V[b,a,0] -> V[a,b,0] when both players keep rolling 1s), 
so recursion never bottoms out. Value iteration sweeps the whole table until it stops changing (Neller & Presser, 2004).
"""
import os  # Check whether an existing solver cache is available.
import numpy as np  # Use arrays to perform the value-iteration table updates efficiently.


class PigSolver:
    def __init__(self, target=100, tol=1e-10, max_sweeps=20000):
        self.T = target  # Store winning score
        self.tol = tol  # Store convergence threshold for value changes.
        self.max_sweeps = max_sweeps  # Limit number of full table sweeps
        self.V = None  # Reserve the value table until the solver runs.
        self.policy = None   # policy[i,j,k] is True where HOLD is optimal
        self.sweeps = 0  

    # solve
    def _hold_roll(self, V):
        T = self.T  # Use the target as each table dimension.
        V0 = V[:, :, 0]  # Extract values for states with no turn-total points.
        hold = np.ones((T, T, T))  # Initialize every hold value to a win probability of 1.
        for a in range(T):  #
            hold[a, :, : T - a] = 1.0 - V0[:, a:T]  # Flip the opponent value after banking points.
        roll = ((1.0 - V0.T) / 6.0)[:, :, None]  # Add the probability of rolling a 1
        for d in range(2, 7):  
            roll = roll + V[:, :, d:d + T] / 6.0  # Add 1/6 of that next-state value
        return hold, roll  # Return the hold and roll value tables.

    def solve(self, cache_path=None, verbose=False):
        T = self.T  # Reuse the target for table setup.
        if cache_path and os.path.exists(cache_path):  
            self.V = np.load(cache_path)  
            self._finish()  # Derive the action policy from the loaded values.
            return self  
        
        V = np.zeros((T, T, T + 6))  # Allocate extra columns for immediate post-roll states.
        i = np.arange(T).reshape(T, 1, 1)  
        k = np.arange(T + 6).reshape(1, 1, T + 6)  # Build turn-total columns, including overflow states.
        V[np.broadcast_to(i + k >= T, V.shape)] = 1.0  # Mark every already-winning state as certain victory.
        done = np.broadcast_to(i + k[:, :, :T] >= T, (T, T, T))  
        for sweep in range(1, self.max_sweeps + 1):  # Repeatedly improve every state value.
            hold, roll = self._hold_roll(V)  
            new = np.maximum(hold, roll)  # Select the optimal action value for each state.
            new[:, :, 0] = roll[:, :, 0]  # must roll at k == 0
            new[done] = 1.0  # Force winning states to remain at value 1.
            delta = np.abs(new - V[:, :, :T]).max()  # Find the largest change across the table.
            V[:, :, :T] = new  # Copy updated values into the main table.
            if verbose and sweep % 50 == 0:  
                print(f"  sweep {sweep}: max change {delta:.2e}") 
            if delta < self.tol:  # Stop once changes are smaller than the convergence tolerance.
                break  

        self.sweeps = sweep  
        self.V = V  # Store the solved value table on the solver.
        if cache_path:  # Save only when a cache path was provided.
            np.save(cache_path, V)  # Write the full value table to disk.
        self._finish()  
        return self  

    def _finish(self):
        hold, roll = self._hold_roll(self.V)  # Recalculate action values from the solved table.
        T = self.T  
        pol = hold > roll  # Sates where holding has a strictly larger value.
        pol[:, :, 0] = False  # Disable holding whenever turn total is 0
        i = np.arange(T).reshape(T, 1, 1)  
        k = np.arange(T).reshape(1, 1, T)  
        pol[np.broadcast_to(i + k >= T, pol.shape)] = True  # Force hold in every winning state.
        self.policy = pol  # Save the final optimal policy table.

    # Queries
    def win_prob(self, i, j, k=0):
        return 1.0 if i + k >= self.T else float(self.V[i, j, k])  # Return a certain win or the table value.

    def q_values(self, i, j, k):
        #(roll_value, hold_value); hold_value is None when hold is illegal
        V = self.V  # Refer briefly to the solved value table.
        roll = (1.0 - V[j, i, 0]) / 6.0 + V[i, j, k + 2:k + 7].sum() / 6.0  # Average the bust and successful-roll values
        hold = None if k == 0 else (1.0 if i + k >= self.T else 1.0 - V[j, i + k, 0])  # Evaluate hold or mark it illegal
        return float(roll), (None if hold is None else float(hold))  # Return both values as plain floats

    def best_action(self, i, j, k):
        return "hold" if self.policy[i, j, k] else "roll"  # Select the policy action for this state.

    def thresholds(self):
        """thr[i,j] = smallest turn total at which holding is optimal (T-i = only when winning)."""
        T = self.T  # Reuse the target for table sizes.
        thr = np.zeros((T, T), dtype=int)  
        monotone = True  # Assume hold thresholds follow a monotone pattern until proven otherwise.
        for i in range(T):  
            for j in range(T):  
                arr = self.policy[i, j, : T - i]  # Examine legal turn totals before the player's next win.
                if arr.any(): 
                    f = int(arr.argmax())  # Find the smallest index where hold becomes optimal
                    thr[i, j] = f  #
                    if not arr[f:].all():  # Check whether hold remains optimal above the threshold
                        monotone = False  # Record that the policy is not monotone here
                else:  
                    thr[i, j] = T - i  # Store the smallest turn total that would produce an immediate win.
        self.threshold_is_monotone = monotone  # Save the overall monotonicity result.
        return thr  
