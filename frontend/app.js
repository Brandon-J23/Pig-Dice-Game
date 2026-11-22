/* Pig front end.
 *
 * The browser owns the game state (scores + turn total) and applies the rules.
 * Whenever an AI player is to move it asks the Python server (server.py), which runs the
 * real agents from ./pig and replies with the action plus what the agent computed.
 *
 * State sent to the server is always from the mover's point of view: [my_score, opp_score, turn_total].
 */
"use strict";

const TARGET = 100;
const FACES = { 1: [5], 2: [1, 9], 3: [1, 5, 9], 4: [1, 3, 7, 9], 5: [1, 3, 5, 7, 9], 6: [1, 3, 4, 6, 7, 9] };
const SPEED = { 0: { think: 0, tumble: 0 }, 1: { think: 300, tumble: 200 }, 2: { think: 800, tumble: 420 }, 3: { think: 1500, tumble: 520 } };
const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const pct = (x) => (x * 100).toFixed(1) + "%";
const nm = (p) => `Player ${p + 1}`;

/* ------------------------------------------------------------------ players */
const cfg = [
  { type: "human" },
  { type: "mcts", sims: 2000, rollout: "threshold" },
];
const AGENT_TYPES = [
  ["human", "Human"],
  ["fixed", "Fixed threshold"],
  ["optimal", "Optimal (exact)"],
  ["expecti", "Expectiminimax"],
  ["mcts", "MCTS"],
];
const DEFAULT_PARAMS = {
  human: {}, optimal: {},
  fixed: { threshold: 20 },
  expecti: { depth: 4 },
  mcts: { sims: 2000, rollout: "threshold" },
};

function describe(c) {
  switch (c.type) {
    case "human": return "Human";
    case "fixed": return `Fixed threshold (hold at ${c.threshold})`;
    case "optimal": return "Optimal (exact policy)";
    case "expecti": return `Expectiminimax (depth ${c.depth})`;
    case "mcts": return `MCTS (${c.sims.toLocaleString()} simulations)`;
  }
}
const isAI = (p) => cfg[p].type !== "human";

/* -------------------------------------------------------------------- state */
const G = {
  gid: 0,              // bumps on every new game so stale async work is dropped
  token: 0,            // bumps whenever pending AI work must be cancelled
  scores: [0, 0],
  turn: 0,             // current turn total
  cur: 0,              // player to move
  winner: null,
  starter: 1,          // flipped at the start of each game => alternates who goes first
  busy: false,         // dice animation / applying a move
  thinking: false,     // waiting on the server
  paused: false,
  status: "",
  eval: null,          // {key, data} exact values for the current position
  evalSeq: 0,
  serverOK: true,
};
const record = [0, 0];
const agreement = [{ n: 0, ok: 0 }, { n: 0, ok: 0 }];
let lastDecision = null;

const stateFor = (p = G.cur) => [G.scores[p], G.scores[1 - p], G.turn];
const stateKey = () => `${G.cur}:${G.scores[0]}:${G.scores[1]}:${G.turn}`;

/* --------------------------------------------------------------------- API */
async function api(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    let msg = r.status;
    try { msg = (await r.json()).error || msg; } catch (_) { /* ignore */ }
    throw new Error(msg);
  }
  return r.json();
}

function setServerOK(ok) {
  G.serverOK = ok;
  const b = $("banner");
  b.hidden = ok;
  if (!ok) {
    b.innerHTML = "Can't reach the Python server. Run <code>python server.py</code> in the project folder, then open <code>http://localhost:8000</code>.";
  }
}

/* -------------------------------------------------------------------- dice */
const dice = $("dice");
function setFace(n) {
  const on = new Set(FACES[n]);
  [...dice.children].forEach((el, i) => el.classList.toggle("on", on.has(i + 1)));
  dice.setAttribute("aria-label", `Die showing ${n}`);
}

async function animateDice(final, ms) {
  if (ms <= 0 || REDUCED_MOTION) { setFace(final); return; }
  dice.classList.add("rolling");
  const end = performance.now() + ms;
  while (performance.now() < end) {
    setFace(1 + Math.floor(Math.random() * 6));
    await sleep(70);
  }
  dice.classList.remove("rolling");
  setFace(final);
}

/* ------------------------------------------------------------------ actions */
function optimalMove(exact, k) {
  if (!exact || k === 0 || exact.roll == null || exact.hold == null) return null;
  return exact.hold > exact.roll ? "hold" : "roll";
}

function recordAgreement(p, action, exact, k) {
  const best = optimalMove(exact, k);
  if (!best) return null;
  agreement[p].n += 1;
  if (best === action) agreement[p].ok += 1;
  return best;
}

/** Apply one action for the current player (roll or hold) with animation. */
async function act(action) {
  if (G.busy || G.winner !== null) return;
  const gid = G.gid;
  const p = G.cur;
  G.busy = true;
  render();

  let event;
  if (action === "roll") {
    const die = 1 + Math.floor(Math.random() * 6);
    await animateDice(die, SPEED[$("speed").value].tumble);
    if (gid !== G.gid) return;
    if (die === 1) {
      const lost = G.turn;
      G.turn = 0;
      G.cur = 1 - p;
      event = `${nm(p)} rolled a 1${lost ? ` and lost ${lost}` : ""}.`;
    } else {
      G.turn += die;
      if (G.scores[p] + G.turn >= TARGET) {
        G.scores[p] += G.turn;
        G.turn = 0;
        G.winner = p;
        record[p] += 1;
        event = `${nm(p)} wins with ${G.scores[p]} points.`;
      } else {
        event = `${nm(p)} rolled a ${die}.`;
      }
    }
  } else {
    G.scores[p] += G.turn;
    event = `${nm(p)} held ${G.turn} and now has ${G.scores[p]}.`;
    G.turn = 0;
    G.cur = 1 - p;
  }

  G.busy = false;
  G.status = event;
  render();
  refreshEval();
  schedule();
}

/** A human clicked roll / hold. */
function humanAct(action) {
  if (G.winner !== null || G.busy || isAI(G.cur)) return;
  if (action === "hold" && G.turn === 0) return;
  if (G.eval && G.eval.key === stateKey()) recordAgreement(G.cur, action, G.eval.data, G.turn);
  act(action);
}

/* ---------------------------------------------------------- AI scheduling */
function cancelPending() {
  G.token += 1;
  G.thinking = false;
}

async function schedule(force = false) {
  render();
  if (G.winner !== null || G.busy || G.thinking || !isAI(G.cur)) return;
  if (G.paused && !force) return;

  const gid = G.gid, tok = G.token, p = G.cur;
  const cfgNow = { ...cfg[p] };
  const state = stateFor(p);
  G.thinking = true;
  render();

  let res;
  try {
    [res] = await Promise.all([
      api("/api/move", { agent: cfgNow, state }),
      sleep(SPEED[$("speed").value].think),
    ]);
  } catch (e) {
    if (gid === G.gid && tok === G.token) {
      G.thinking = false;
      setServerOK(false);
      G.status = "The AI could not move. Check the server and press New game.";
      render();
    }
    return;
  }
  if (gid !== G.gid || tok !== G.token) return;
  setServerOK(true);
  G.thinking = false;

  const best = recordAgreement(p, res.action, res.exact, state[2]);
  lastDecision = { p, cfg: cfgNow, state, res, best };
  renderInsight();
  G.eval = { key: stateKey(), data: res.exact };
  await act(res.action);
}

/* ------------------------------------------------------------------ eval */
async function refreshEval() {
  if (G.winner !== null) { renderPosition(); return; }
  const seq = ++G.evalSeq;
  const key = stateKey();
  try {
    const data = await api("/api/eval", { state: stateFor() });
    if (seq !== G.evalSeq) return;
    G.eval = { key, data };
    setServerOK(true);
  } catch (e) {
    if (seq === G.evalSeq) { G.eval = null; setServerOK(false); }
  }
  renderPosition();
}

/* -------------------------------------------------------------- new game */
function newGame() {
  cancelPending();
  G.gid += 1;
  G.scores = [0, 0];
  G.turn = 0;
  G.winner = null;
  G.busy = false;
  G.starter = 1 - G.starter;
  G.cur = G.starter;
  G.eval = null;
  agreement.forEach((a) => { a.n = 0; a.ok = 0; });
  setFace(1);
  dice.classList.remove("hidden", "rolling");
  G.status = `New game. ${nm(G.cur)} starts.`;
  render();
  refreshEval();
  schedule();
}

/* ------------------------------------------------------------- rendering */
function turnText() {
  const p = G.cur;
  if (G.winner !== null) return "";
  if (G.busy) return "Rolling…";
  if (!isAI(p)) {
    const humans = [0, 1].filter((q) => !isAI(q)).length;
    return humans === 1 ? "Your turn: roll or hold." : `${nm(p)}'s turn.`;
  }
  if (G.thinking) return `${nm(p)} is thinking…`;
  return G.paused ? "AI paused. Press Step or Resume." : "";
}

function render() {
  for (const p of [0, 1]) {
    $(`score${p}`).textContent = G.scores[p];
    $(`currentScore${p}`).textContent = p === G.cur && G.winner === null ? G.turn : 0;
    const panel = $(`player${p}`);
    panel.classList.toggle("player--active", G.winner === null && p === G.cur);
    panel.classList.toggle("player--winner", G.winner === p);
    $(`who${p}`).textContent = describe(cfg[p]);
  }
  dice.classList.toggle("hidden", G.winner !== null);

  const humanTurn = G.winner === null && !isAI(G.cur) && !G.busy;
  $("btnRoll").disabled = !humanTurn;
  $("btnHold").disabled = !humanTurn || G.turn === 0;

  const anyAI = isAI(0) || isAI(1);
  $("btnPause").hidden = !anyAI;
  $("btnPause").textContent = G.paused ? "Resume AI" : "Pause AI";
  $("btnStep").hidden = !(anyAI && G.paused);
  $("btnStep").disabled = G.winner !== null || G.busy || G.thinking || !isAI(G.cur);

  $("status").textContent = [G.status, turnText()].filter(Boolean).join(" ");
  renderSetupStats();
}

function bar(label, share, valueText, best) {
  return `<div class="row${best ? " best" : ""}"><span class="k">${label}</span>` +
    `<div class="bar"><i style="width:${Math.max(0, Math.min(1, share)) * 100}%"></i></div>` +
    `<span class="v">${valueText}</span></div>`;
}

function renderPosition() {
  const el = $("position");
  if (!$("showHints").checked) { el.innerHTML = '<p class="muted">Hidden. Tick the box to show the exact solution.</p>'; return; }
  if (G.winner !== null) { el.innerHTML = `<p class="muted">${nm(G.winner)} won this game.</p>`; return; }
  if (!G.eval || G.eval.key !== stateKey()) { el.innerHTML = '<p class="muted">Looking up the position…</p>'; return; }

  const d = G.eval.data;
  const p1 = G.cur === 0 ? d.win_prob : 1 - d.win_prob;
  const best = optimalMove(d, G.turn);
  let html = `<div class="duel"><div class="p0" style="width:${p1 * 100}%">${Math.round(p1 * 100)}%</div>` +
    `<div class="p1" style="width:${(1 - p1) * 100}%">${Math.round((1 - p1) * 100)}%</div></div>` +
    `<p class="duel-caption">Chance to win from here if both play perfectly: Player 1 on the left, Player 2 on the right.</p>`;
  if (G.turn > 0) {
    html += bar("Roll", d.roll, pct(d.roll), best === "roll") + bar("Hold", d.hold, pct(d.hold), best === "hold") +
      `<p class="note">Win chance for ${nm(G.cur)} after each choice.</p>`;
  } else {
    html += `<p class="note">${nm(G.cur)} has to roll first (holding on 0 is not allowed).</p>`;
  }
  el.innerHTML = html;
}

function renderInsight() {
  const el = $("insight");
  if (!lastDecision) return;
  const { p, cfg: c, state, res, best } = lastDecision;
  const [i, j, k] = state;
  const a = res.agent || {};
  let html = `<div class="ins-head"><strong>${nm(p)}: ${res.name}</strong>` +
    `<span class="pill ${res.action}">${res.action.toUpperCase()}</span></div>` +
    `<p class="ins-sub">Score ${i} to ${j}, turn total ${k}. Decided in ${res.ms < 1 ? "<1" : res.ms.toFixed(res.ms < 10 ? 1 : 0)} ms.</p>`;

  if (c.type === "mcts" && a.visits) {
    const total = Object.values(a.visits).reduce((s, v) => s + v, 0) || 1;
    for (const act of ["roll", "hold"]) {
      if (!(act in a.visits)) continue;
      html += bar(act === "roll" ? "Roll" : "Hold", a.visits[act] / total,
        `${a.visits[act].toLocaleString()} visits`, act === res.action);
    }
    const est = ["roll", "hold"].filter((x) => x in a.winrate).map((x) => `${x} ${pct(a.winrate[x])}`).join(", ");
    html += `<p class="note">${a.sims.toLocaleString()} random playouts (${a.rollout === "random" ? "random" : "hold at 20"} rollouts). It picks the most-visited action. Its win estimates: ${est}.</p>`;
  } else if (c.type === "expecti") {
    const best2 = a.hold != null && a.hold > a.roll ? "hold" : "roll";
    html += bar("Roll", a.roll, pct(a.roll), best2 === "roll");
    if (a.hold != null) html += bar("Hold", a.hold, pct(a.hold), best2 === "hold");
    html += `<p class="note">Searched ${a.depth} levels of roll/hold choices with the die averaged in. Positions beyond that are scored by a heuristic, so these values are estimates.</p>`;
  } else if (c.type === "fixed") {
    html += `<p class="note">Rule: hold once the turn total reaches ${a.threshold}. Turn total is ${k}, so ${res.action}.</p>`;
  } else if (c.type === "optimal") {
    html += `<p class="note">Looked up the exact policy from value iteration.</p>`;
  }

  if (best) {
    const ex = res.exact;
    const ok = best === res.action;
    html += `<p class="verdict ${ok ? "ok" : "miss"}">Exact values: roll ${pct(ex.roll)}, hold ${pct(ex.hold)}. ` +
      (ok ? `Matches the optimal move (${best}).` : `Differs from the optimal move (${best}).`) + `</p>`;
  } else if (k === 0) {
    html += `<p class="verdict ok">Forced roll: holding on 0 is not allowed.</p>`;
  }
  el.innerHTML = html;
}

/* --------------------------------------------------------------- setup UI */
function buildSetup() {
  const host = $("setup");
  host.innerHTML = "";
  for (const p of [0, 1]) {
    const row = document.createElement("div");
    row.className = "setup-row";
    row.innerHTML =
      `<div class="setup-top"><span class="swatch" style="background:var(--${p ? "accent-2" : "accent"})"></span>` +
      `<strong>${nm(p)}</strong><select aria-label="${nm(p)} controller" id="type${p}">` +
      AGENT_TYPES.map(([v, t]) => `<option value="${v}">${t}</option>`).join("") +
      `</select></div><div class="params" id="params${p}"></div><p class="agree" id="agree${p}"></p>`;
    host.appendChild(row);
    $(`type${p}`).value = cfg[p].type;
    $(`type${p}`).addEventListener("change", (e) => {
      cfg[p] = { type: e.target.value, ...DEFAULT_PARAMS[e.target.value] };
      buildParams(p);
      onConfigChange();
    });
    buildParams(p);
  }
}

function numberOptions(values, current) {
  return values.map((v) => `<option value="${v}"${v === current ? " selected" : ""}>${v.toLocaleString()}</option>`).join("");
}

function buildParams(p) {
  const c = cfg[p], host = $(`params${p}`);
  if (c.type === "fixed") {
    host.innerHTML = `<label class="inline">Hold at <input type="number" min="1" max="100" value="${c.threshold}" id="thr${p}"></label>`;
    $(`thr${p}`).addEventListener("change", (e) => {
      c.threshold = Math.max(1, Math.min(100, parseInt(e.target.value, 10) || 20));
      e.target.value = c.threshold;
      onConfigChange();
    });
  } else if (c.type === "expecti") {
    host.innerHTML = `<label class="inline">Search depth <select id="depth${p}">${numberOptions([1, 2, 3, 4, 5, 6, 7, 8], c.depth)}</select></label>`;
    $(`depth${p}`).addEventListener("change", (e) => { c.depth = +e.target.value; onConfigChange(); });
  } else if (c.type === "mcts") {
    host.innerHTML =
      `<label class="inline">Simulations <select id="sims${p}">${numberOptions([100, 500, 1000, 2000, 5000, 10000], c.sims)}</select></label>` +
      `<label class="inline">Rollouts <select id="roll${p}"><option value="threshold">Hold at 20</option><option value="random">Random</option></select></label>`;
    $(`roll${p}`).value = c.rollout;
    $(`sims${p}`).addEventListener("change", (e) => { c.sims = +e.target.value; onConfigChange(); });
    $(`roll${p}`).addEventListener("change", (e) => { c.rollout = e.target.value; onConfigChange(); });
  } else {
    host.innerHTML = "";
  }
}

function onConfigChange() {
  cancelPending();
  agreement.forEach((a) => { a.n = 0; a.ok = 0; });
  render();
  schedule();
}

function renderSetupStats() {
  for (const p of [0, 1]) {
    const a = agreement[p], el = $(`agree${p}`);
    el.textContent = a.n ? `Matched the optimal move on ${a.ok} of ${a.n} choices (${Math.round((a.ok / a.n) * 100)}%).` : "";
  }
  $("record").textContent = record[0] + record[1] ? `This session: Player 1 ${record[0]}, Player 2 ${record[1]}.` : "";
}

/* ----------------------------------------------------------------- wiring */
$("btnNew").addEventListener("click", newGame);
$("btnRoll").addEventListener("click", () => humanAct("roll"));
$("btnHold").addEventListener("click", () => humanAct("hold"));
$("btnPause").addEventListener("click", () => {
  G.paused = !G.paused;
  if (G.paused) cancelPending();
  render();
  if (!G.paused) schedule();
});
$("btnStep").addEventListener("click", () => schedule(true));
$("speed").addEventListener("change", render);
$("showHints").addEventListener("change", renderPosition);

document.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const tag = e.target.tagName;
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
  const k = e.key.toLowerCase();
  if (k === "r") humanAct("roll");
  else if (k === "h") humanAct("hold");
  else if (k === "n") newGame();
});

buildSetup();
fetch("/api/info").then((r) => { if (!r.ok) throw 0; setServerOK(true); }).catch(() => setServerOK(false));
G.starter = 1;       // newGame() flips this, so Player 1 starts the first game
newGame();
