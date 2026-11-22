"""Sanity checks. Run: python test_backend.py"""
import time, numpy as np
from pig import *

t = time.time()
s = PigSolver(100).solve(cache_path="policy_cache_100.npy", verbose=True)
print(f"solved in {time.time()-t:.1f}s, sweeps={s.sweeps}")

# 1. Known result (Neller & Presser): first player wins ~53.06% under optimal play
p0 = s.win_prob(0, 0, 0)
print("P(first player wins) =", round(p0, 4)); assert abs(p0 - 0.5306) < 0.002

# 2. Bellman consistency: V == max(hold, roll) everywhere
hold, roll = s._hold_roll(s.V)
T = 100
best = np.maximum(hold, roll); best[:, :, 0] = roll[:, :, 0]
i = np.arange(T).reshape(T,1,1); k = np.arange(T).reshape(1,1,T)
mask = (i + k) < T
assert np.abs((best - s.V[:, :, :T])[np.broadcast_to(mask, best.shape)]).max() < 1e-8
print("Bellman residual OK")

# 3. Policy is a threshold policy in turn_total
thr = s.thresholds(); print("threshold policy monotone in k:", s.threshold_is_monotone)
print("thr(0,0) =", thr[0,0], " thr(80,80) =", thr[80,80], " thr(20,90) =", thr[20,90])

# 4. Engine legality + baseline beats nothing weird
g = PigGame(100)
try: g.step("hold"); raise SystemExit("hold at k=0 should be illegal")
except ValueError: pass
print("engine OK")
