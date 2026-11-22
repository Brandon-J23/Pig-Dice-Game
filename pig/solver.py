"""Exact solution of Pig via value iteration on the expectiminimax equations.

V[i, j, k] = P(player to move wins) with own score i, opponent score j, turn total k,
             assuming both players play optimally.

    hold  = 1 - V[j, i+k, 0]                       (opponent moves next)
    roll  = 1/6 * (1 - V[j, i, 0])                 (rolled a 1: lose turn total)
          + 1/6 * sum_{d=2..6} V[i, j, k+d]        (rolled d: same player continues)
    V[i,j,k] = max(hold, roll)       (hold illegal when k == 0;  V = 1 when i+k >= target)

Why not plain memoised recursion? Pig's state graph has CYCLES (V[a,b,0] -> V[b,a,0]
-> V[a,b,0] when both players keep rolling 1s), so recursion never bottoms out.
Value iteration sweeps the whole table until it stops changing (Neller & Presser, 2004).
"""
import os
import numpy as np


class PigSolver:
    def __init__(self, target=100, tol=1e-10, max_sweeps=20000):
        self.T = target
        self.tol = tol
        self.max_sweeps = max_sweeps
        self.V = None
        self.policy = None   # policy[i,j,k] is True where HOLD is optimal
        self.sweeps = 0

    # solve
    def _hold_roll(self, V):
        T = self.T
        V0 = V[:, :, 0]
        hold = np.ones((T, T, T))
        for a in range(T):
            hold[a, :, : T - a] = 1.0 - V0[:, a:T]
        roll = ((1.0 - V0.T) / 6.0)[:, :, None]
        for d in range(2, 7):
            roll = roll + V[:, :, d:d + T] / 6.0
        return hold, roll

    def solve(self, cache_path=None, verbose=False):
        T = self.T
        if cache_path and os.path.exists(cache_path):
            self.V = np.load(cache_path)
            self._finish()
            return self
        V = np.zeros((T, T, T + 6))
        i = np.arange(T).reshape(T, 1, 1)
        k = np.arange(T + 6).reshape(1, 1, T + 6)
        V[np.broadcast_to(i + k >= T, V.shape)] = 1.0
        done = np.broadcast_to(i + k[:, :, :T] >= T, (T, T, T))
        for sweep in range(1, self.max_sweeps + 1):
            hold, roll = self._hold_roll(V)
            new = np.maximum(hold, roll)
            new[:, :, 0] = roll[:, :, 0]          # must roll at k == 0
            new[done] = 1.0
            delta = np.abs(new - V[:, :, :T]).max()
            V[:, :, :T] = new
            if verbose and sweep % 50 == 0:
                print(f"  sweep {sweep}: max change {delta:.2e}")
            if delta < self.tol:
                break
        self.sweeps = sweep
        self.V = V
        if cache_path:
            np.save(cache_path, V)
        self._finish()
        return self

    def _finish(self):
        hold, roll = self._hold_roll(self.V)
        T = self.T
        pol = hold > roll
        pol[:, :, 0] = False
        i = np.arange(T).reshape(T, 1, 1)
        k = np.arange(T).reshape(1, 1, T)
        pol[np.broadcast_to(i + k >= T, pol.shape)] = True
        self.policy = pol

    # ---------------------------------------------------------------- queries
    def win_prob(self, i, j, k=0):
        return 1.0 if i + k >= self.T else float(self.V[i, j, k])

    def q_values(self, i, j, k):
        """(roll_value, hold_value); hold_value is None when hold is illegal."""
        V = self.V
        roll = (1.0 - V[j, i, 0]) / 6.0 + V[i, j, k + 2:k + 7].sum() / 6.0
        hold = None if k == 0 else (1.0 if i + k >= self.T else 1.0 - V[j, i + k, 0])
        return float(roll), (None if hold is None else float(hold))

    def best_action(self, i, j, k):
        return "hold" if self.policy[i, j, k] else "roll"

    def thresholds(self):
        """thr[i,j] = smallest turn total at which holding is optimal (T-i = only when winning)."""
        T = self.T
        thr = np.zeros((T, T), dtype=int)
        monotone = True
        for i in range(T):
            for j in range(T):
                arr = self.policy[i, j, : T - i]
                if arr.any():
                    f = int(arr.argmax())
                    thr[i, j] = f
                    if not arr[f:].all():
                        monotone = False
                else:
                    thr[i, j] = T - i
        self.threshold_is_monotone = monotone
        return thr
