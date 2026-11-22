# Pig AI – backend (CSE 352 extra credit)

```
pig/game.py  --> engine: PigGame (state = (my_score, opp_score, turn_total))
pig/solver.py  --> exact optimal policy via value iteration on the expectiminimax equations
pig/agents.py  -->  FixedThresholdAgent | OptimalAgent | ExpectiminimaxAgent(depth) | MCTSAgent(sims)
pig/tournament.py --> play_game, head_to_head (alternates who moves first, returns win rate + 95% CI)
test_backend.py  --> sanity checks (matches published 53.06% first-player win prob)
run_experiments.py  --> all report experiments -> results/*.csv, *.png, summary.json, policy.json
```

Run: `pip install -r requirements.txt && python test_backend.py && python run_experiments.py --quick`
(drop `--quick` for report-quality numbers; it takes much longer because of MCTS).

## Search-problem formulation
* **State**: (my_score, opp_score, turn_total), from the mover's view. ~505,000 valid, nonterminal states.
* **Actions**: Roll; Hold (illegal when turn_total = 0).
* **Transition**: Hold -> bank turn total, pass turn. Roll -> die uniform on 1..6: a 1 loses the turn total and passes the turn; 2-6 add to the turn total.
* **Goal**: score + turn_total >= 100 (game ends immediately; equivalent to holding and winning).
* **Tree**: alternating MAX (roll/hold) and CHANCE (die, 1/6 each) nodes; branching 2 x 6.
* **Utility**: probability of winning (not points).

## Why value iteration instead of memoised recursion
The state graph has cycles (both players rolling 1s returns to the same scores), so plain
recursion never terminates. Value iteration sweeps the table until max change < 1e-10 (~400 sweeps, ~10 s).

## "What makes it intelligent?"
It maximises win probability under the true dice distribution, so its hold threshold adapts
to the score situation (e.g. ~21 at 0-0, ~80 when far behind, ~10 when far ahead) rather than using one fixed number.

## Front end
`results/policy.json` holds `thresholds[my][opp]` (hold when turn_total >= threshold) and
`win_prob[my][opp]`, so a vanilla-JS UI can play the optimal strategy with no Python server.
(Caveat: at extreme turn totals k >= ~97 rolling can beat holding; thresholds ignore those unreachable cases.)
