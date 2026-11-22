"""Front-end server for the Pig AI project.

    python server.py            ->  open http://localhost:8000

It serves the static UI in ./frontend and exposes the *real* backend agents over a tiny JSON API, 
so the browser plays against the exact same Python code used in the experiments. Only the standard 
library is used (no Flask needed).

    GET  /api/info  -> game constants + the agent types the UI can pick from
    POST /api/eval  {state:[my, opp, k]} -> exact optimal values for a position
    POST /api/move  {agent:{type,...}, state:[my,opp,k]} -> the agent's chosen action + its internals

The server is stateless: the browser owns the game state (it is just 3 numbers), the server is
an "oracle" that answers "what would this agent do here?".
"""
import json
import os
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from pig import (ExpectiminimaxAgent, FixedThresholdAgent, MCTSAgent, OptimalAgent,
                 PigSolver)

ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND = os.path.join(ROOT, "frontend")
TARGET = 100  # Score needed to win.
PORT = int(os.environ.get("PORT", 8000))  # Allow the server port to be configured.

# Load the exact solver once at startup.
print("Loading exact solver...")
SOLVER = PigSolver(TARGET).solve(cache_path=os.path.join(ROOT, "policy_cache_100.npy"))
print("Solver ready.")

LOCK = threading.Lock()  # Prevent shared agent state from updating concurrently.
AGENTS = {}  # Reuse agents by their configuration.


def clamp(v, lo, hi, default):
    # Safely convert a value to a bounded integer.
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return default


def get_agent(cfg):
    # Create or reuse the agent selected by the UI.
    kind = cfg.get("type")
    if kind == "fixed":
        threshold = clamp(cfg.get("threshold"), 1, TARGET, 20)
        key, make = ("fixed", threshold), lambda: FixedThresholdAgent(threshold, TARGET)
    elif kind == "optimal":
        key, make = ("optimal",), lambda: OptimalAgent(SOLVER)
    elif kind == "expecti":
        depth = clamp(cfg.get("depth"), 1, 8, 4)
        key, make = ("expecti", depth), lambda: ExpectiminimaxAgent(depth, TARGET)
    elif kind == "mcts":
        sims = clamp(cfg.get("sims"), 50, 20000, 2000)
        rollout = cfg.get("rollout") if cfg.get("rollout") in ("threshold", "random") else "threshold"
        key, make = ("mcts", sims, rollout), lambda: MCTSAgent(sims, rollout=rollout, target=TARGET)
    else:
        raise ValueError(f"unknown agent type {kind!r}")
    if key not in AGENTS:
        AGENTS[key] = make()
    return kind, AGENTS[key]


def parse_state(raw):
    # Convert the UI state into three integers.
    i, j, k = (int(x) for x in raw)
    # Reject states outside the game limits.
    if not (0 <= i < TARGET and 0 <= j < TARGET and 0 <= k):
        raise ValueError("state out of range")
    return i, j, k


def exact_values(i, j, k):
    #Get exact roll, hold, and win values from the solver
    if i + k >= TARGET:
        return {"roll": None, "hold": 1.0, "win_prob": 1.0}
    roll, hold = SOLVER.q_values(i, j, k)
    return {"roll": roll, "hold": hold, "win_prob": SOLVER.win_prob(i, j, k)}


def agent_internals(kind, agent, state):
    #Collect details for the UI insight panel.
    i, j, k = state
    if kind == "mcts":
        if k == 0:  # No search occurs when rolling is the only option.
            return {}
        stats = agent.last_stats or {}
        return {"visits": {a: stats[a][0] for a in stats},
                "winrate": {a: stats[a][1] for a in stats},
                "sims": agent.n_sim, "rollout": agent.rollout}
    if kind == "expecti":
        roll, hold = agent._roll_hold(i, j, k, agent.depth)      # cache hit after choose_action
        return {"roll": roll, "hold": hold, "depth": agent.depth,
                "cache_size": len(agent.cache)}
    if kind == "fixed":
        return {"threshold": agent.threshold}
    return {}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        # Serve files from the frontend directory.
        super().__init__(*a, directory=FRONTEND, **kw)

    def log_message(self, fmt, *args):  # Hide routine static-file log entries.
        if "/api/" not in (args[0] if args else ""):
            super().log_message(fmt, *args)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")  # Do not cache API or UI responses.
        super().end_headers()

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()  # Convert the response into JSON bytes.
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/info":
            # Return game settings and the opening win probability.
            return self._json({"target": TARGET,
                               "first_player_win_prob": SOLVER.win_prob(0, 0, 0)})
        # Serve frontend files for all other paths.
        return super().do_GET()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))  # Read the request body size.
            data = json.loads(self.rfile.read(length) or b"{}")  # Parse the JSON request.
            state = parse_state(data["state"])  # Validate the submitted game state.
            if self.path == "/api/eval":
                # Return exact values for the position.
                return self._json(exact_values(*state))
            if self.path == "/api/move":
                kind, agent = get_agent(data.get("agent", {}))
                with LOCK:
                    t0 = time.perf_counter()  # Start the action timer.
                    action = agent.choose_action(state)  # Ask the selected agent to act.
                    ms = (time.perf_counter() - t0) * 1000  # Calculate decision time in milliseconds.
                    internals = agent_internals(kind, agent, state)  # Gather extra agent details.
                # Return the action and supporting values.
                return self._json({"action": action, "ms": ms, "name": agent.name,
                                   "agent": internals, "exact": exact_values(*state)})
            # Report unknown API paths.
            self._json({"error": "not found"}, 404)
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            # Return invalid requests as JSON errors.
            self._json({"error": str(e)}, 400)


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)  # Start a threaded local server.
    print(f"Pig UI running at http://localhost:{PORT}   (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)
