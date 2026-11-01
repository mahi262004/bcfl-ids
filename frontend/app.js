/* Dashboard logic. All server text goes through textContent, never innerHTML. */
(() => {
  const $ = (id) => document.getElementById(id);
  const state = { options: null, results: null, labRuns: [], polling: null, chainRunId: null, selectedRow: null };
  const pct = (v, d = 2) => (v === null || v === undefined ? "n/a" : `${(v * 100).toFixed(d)}%`);
  const num = (v, d = 3) => (v === null || v === undefined ? "n/a" : Number(v).toFixed(d));
  const short = (h) => (h ? `${h.slice(0, 10)}…${h.slice(-6)}` : "");
  const pretty = (s) => s.replaceAll("_", " ");

  async function api(path, options = {}) {
    let res;
    try {
      res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
    } catch {
      throw new Error("The server is not responding. Check that the demo is still running.");
    }
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join("; ") : body.detail;
      throw new Error(detail || `Request failed (${res.status})`);
    }
    return body;
  }

  function banner(message) {
    const b = $("banner");
    b.textContent = message || "";
    b.hidden = !message;
  }
  const guard = (fn) => async (...args) => {
    try {
      banner("");
      return await fn(...args);
    } catch (err) {
      banner(err.message);
    }
  };

  function make(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function tile(host, label, value, hint, long = false) {
    const t = make("div", "tile");
    t.appendChild(make("div", "label", label));
    t.appendChild(make("div", long ? "value long" : "value", value));
    if (hint) t.appendChild(make("div", "hint", hint));
    host.appendChild(t);
  }
  function fillSelect(sel, items, selected) {
    sel.textContent = "";
    for (const item of items) {
      const opt = make("option", "", item.label);
      opt.value = item.value;
      if (String(item.value) === String(selected)) opt.selected = true;
      sel.appendChild(opt);
    }
  }
  function row(cells) {
    const tr = document.createElement("tr");
    for (const c of cells) {
      const td = make("td", c.cls || "", c.node ? undefined : c.text);
      if (c.node) td.appendChild(c.node);
      if (c.title) td.title = c.title;
      tr.appendChild(td);
    }
    return tr;
  }

  /* ---------- navigation and theme ---------- */
  function showView(name) {
    document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.view === name)));
    document.querySelectorAll(".view").forEach((v) => { v.hidden = v.id !== `view-${name}`; });
    if (name === "chain") refreshRunList();
    if (name === "fog") refreshFog();
    if (name === "results") renderResults();
    if (name === "explain") renderExplain();
    if (name === "overview") renderOverview();
    if (name === "lab") redrawLab();
    if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
  }
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));
  $("themeBtn").addEventListener("click", () => {
    const dark = getComputedStyle(document.documentElement).colorScheme.includes("dark");
    document.documentElement.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("theme", document.documentElement.dataset.theme); } catch { /* preference only */ }
    const current = document.querySelector(".tabs button[aria-selected=true]").dataset.view;
    showView(current);
  });
  try {
    const saved = localStorage.getItem("theme");
    if (saved) document.documentElement.dataset.theme = saved;
  } catch { /* preference only */ }

  /* ---------- status ---------- */
  async function pollStatus() {
    try {
      const s = await api("/api/status");
      const pill = $("chainPill");
      pill.textContent = s.chain.connected ? "Chain: connected" : "Chain: offline, using local ledger";
      pill.className = `pill ${s.chain.connected ? "ok" : "off"}`;
      const ipfs = $("ipfsPill");
      ipfs.textContent = s.storage.connected ? "IPFS: private node" : "IPFS: offline, using local folder";
      ipfs.className = `pill ${s.storage.connected ? "ok" : "off"}`;
      if (!$("banner").textContent.startsWith("A simulation") && $("banner").textContent.includes("not responding")) banner("");
    } catch (err) {
      $("chainPill").textContent = "Server: offline";
      $("chainPill").className = "pill off";
      banner(err.message);
    }
  }

  /* ---------- overview ---------- */
  function cleanProposed() {
    const g = state.results?.grid_summary;
    return g ? g.cells.find((c) => c.attack === "none" && c.defence === "proposed") : null;
  }
  function renderOverview() {
    const host = $("overviewTiles");
    host.textContent = "";
    const clean = cleanProposed();
    const meta = state.results?.grid_summary?.meta;
    if (!clean) {
      $("heroValue").textContent = "–";
      $("heroLabel").textContent = "Evaluation results have not been generated yet";
      return;
    }
    $("heroValue").textContent = pct(clean.attack_recall.mean, 2);
    $("heroLabel").textContent = "of botnet records flagged as attacks";
    $("heroNote").textContent = `${Number(meta.test_rows).toLocaleString()} test records, ${meta.seeds} seeds, chronological train/test split, no attacker present.`;
    tile(host, "Benign recall", pct(clean.benign_recall.mean), "normal traffic left alone");
    tile(host, "MCC", num(clean.mcc.mean), "attack vs benign, robust to class imbalance");
    tile(host, "Attack family macro F1", num(clean.macro_f1.mean), `${num(clean.macro_f1_merged.mean)} once tcp/udp and junk/combo, which look alike in this data, are merged`);
    const worst = state.results.grid_summary.headline;
    tile(host, "No defence, worst attack", pct(worst.fedavg_worst.value, 1), `macro F1 under ${pretty(worst.fedavg_worst.attack)}, 3 of 9 clients`);
    tile(host, "Proposed, worst attack", pct(worst.proposed_worst.value, 1), `macro F1 under ${pretty(worst.proposed_worst.attack)}, 3 of 9 clients`);
    const attacked = state.results.grid_summary.cells.filter((c) => c.defence === "proposed" && c.attack !== "none" && c.false_reject_rate.mean !== null);
    if (attacked.length) {
      const worstFalse = Math.max(...attacked.map((c) => c.false_reject_rate.mean));
      tile(host, "Honest updates rejected", pct(worstFalse, 1), "worst case across all attacks, 3 of 9 clients malicious");
    }
  }

  /* ---------- attack lab ---------- */
  function syncLabControls() {
    const attack = $("labAttack").value;
    const none = attack === "none";
    $("labN").disabled = none;
    $("labDesc").textContent = state.options.attacks.find((a) => a.id === attack)?.description || "";
    $("labLie").disabled = none;
    if (none) $("labLie").checked = false;
  }
  function labColors(run, index) {
    return run.config.defence === "fedavg" ? Charts.css("--series-2") : index === 0 && state.labRuns.length === 1 && run.config.defence !== "proposed" ? Charts.css("--series-3") : Charts.css("--series-1");
  }
  function redrawLab() {
    if (!state.options) return;
    const runs = state.labRuns;
    const rounds = runs.length ? runs[0].config.rounds : 10;
    const series = runs.map((r, i) => ({
      name: r.label, color: labColors(r, i),
      points: r.rounds.filter((x) => x.metrics).map((x) => ({ x: x.round, y: x.metrics.macro_f1 })),
    }));
    Charts.legend($("labLegend"), series.map((s) => ({ name: s.name, color: s.color })));
    Charts.line($("labChart"), { series, xMax: Math.max(rounds, 2), yFmt: (v) => v.toFixed(2) });

    const tiles = $("labTiles");
    tiles.textContent = "";
    for (const r of runs) {
      const latest = [...r.rounds].reverse().find((x) => x.metrics)?.metrics;
      if (latest) tile(tiles, r.label, num(latest.macro_f1), `attack recall ${pct(latest.attack_recall, 1)}, benign recall ${pct(latest.benign_recall, 1)}`);
    }
    const defended = [...runs].reverse().find((r) => r.config.defence !== "fedavg") || runs[runs.length - 1];
    if (defended?.summary && defended.summary.attacker_reject_rate !== null) {
      tile(tiles, "Attacker updates rejected", pct(defended.summary.attacker_reject_rate, 0), "by the trust gate");
      tile(tiles, "Honest updates rejected", pct(defended.summary.false_reject_rate, 1), "false rejections");
    }
    drawClients(defended);
    drawReputation(defended);
  }
  function drawClients(run) {
    const host = $("labClients");
    host.textContent = "";
    if (!run || !run.rounds.length) {
      host.appendChild(make("p", "muted", "Nothing yet. Client verdicts appear here as the rounds complete."));
      return;
    }
    const latest = run.rounds[run.rounds.length - 1];
    $("labClientsSub").textContent = `round ${latest.round} of ${run.label}, ${latest.accepted} of 9 updates aggregated`;
    const names = Object.fromEntries(state.options.devices.map((d) => [d.id, d.name]));
    for (const c of latest.clients) {
      const card = make("div", "client");
      const head = make("div", "head");
      const left = make("span", "", names[c.client]);
      head.appendChild(left);
      const tag = c.included ? make("span", "tag ok", "Accepted") : c.on_probation ? make("span", "tag warn", "Probation") : make("span", "tag bad", "Rejected");
      head.appendChild(tag);
      card.appendChild(head);
      if (c.malicious) card.appendChild(make("div", "why", "Simulated attacker"));
      for (const [k, v] of [["Cosine to median", num(c.cosine)], ["Norm ratio", num(c.norm_ratio, 2)], ["Score", c.score], ["Reputation", `${c.rep_before} → ${c.rep_after}`]]) {
        const kv = make("div", "kv");
        kv.appendChild(make("span", "", k));
        kv.appendChild(make("b", "", String(v)));
        card.appendChild(kv);
      }
      if (!c.included && c.reason) card.appendChild(make("div", "why", c.reason));
      host.appendChild(card);
    }
  }
  function drawReputation(run) {
    const host = $("repChart");
    if (!run || !run.rounds.length) {
      host.textContent = "";
      $("repLegend").textContent = "";
      return;
    }
    const clients = run.rounds[0].clients;
    const series = clients.map((c) => {
      const malicious = c.malicious;
      return {
        name: `Client ${c.client}`,
        color: malicious ? Charts.css("--series-2") : Charts.css("--series-1"),
        opacity: malicious ? 1 : 0.55,
        width: malicious ? 2 : 1.5,
        points: run.rounds.map((r) => ({ x: r.round, y: r.clients.find((x) => x.client === c.client).rep_after })),
      };
    });
    const hasAttackers = clients.some((c) => c.malicious);
    Charts.line(host, { series, xMax: Math.max(run.config.rounds, 2), yMin: 0, yMax: 100, yTicks: 4, yFmt: (v) => v.toFixed(0), threshold: { y: 50, label: "probation" }, height: 240, endLabels: false });
    const legend = [{ name: "Honest clients", color: Charts.css("--series-1") }];
    if (hasAttackers) legend.push({ name: "Simulated attackers", color: Charts.css("--series-2") });
    Charts.legend($("repLegend"), legend);
  }
  async function pollJob(job) {
    let done = false;
    const lastRound = new Map();
    state.labRuns = [];
    const tick = async () => {
      const views = [];
      for (const id of job.run_ids) {
        const view = await api(`/api/simulations/${id}`);
        views.push(view);
        lastRound.set(id, view.rounds.length);
      }
      state.labRuns = views.filter((v) => v.status !== "queued");
      redrawLab();
      const active = views.find((v) => v.status === "running");
      const failed = views.find((v) => v.status === "failed");
      if (failed) {
        banner(`Simulation failed: ${failed.error}. You can run it again.`);
      }
      if (active) {
        const ledger = active.backend === "chain" ? "logging to the blockchain" : "logging to the local ledger";
        $("labStatus").textContent = `Running ${active.label}: round ${active.rounds.length} of ${active.config.rounds}, ${ledger}.`;
      }
      done = views.every((v) => ["finished", "failed", "stopped"].includes(v.status));
      if (done) {
        const finished = views.filter((v) => v.status === "finished");
        if (finished.length) {
          const attackers = finished[0].summary.attackers;
          const note = attackers.length ? `Attackers: clients ${attackers.join(", ")}.` : "No attackers.";
          const where = finished[0].backend === "chain" ? "Trust records are on the local chain; open Blockchain audit to verify them." : "The chain was offline, so records went to the local in-memory ledger.";
          const lied = finished.some((v) => v.config.dishonest_server) ? " The server lied on the chain in this run: Blockchain audit will still pass, and Fog and IPFS will catch it." : "";
          $("labStatus").textContent = `Finished. ${note} ${where}${lied} Uses the small bundled demo sample, so values differ from the Evaluation tab.`;
        } else if (!failed) {
          $("labStatus").textContent = "Stopped.";
        }
      }
    };
    while (!done) {
      try {
        await tick();
      } catch (err) {
        banner(err.message);
        break;
      }
      if (!done) await new Promise((r) => setTimeout(r, 500));
    }
    $("labRun").disabled = false;
    $("labStop").hidden = true;
  }
  $("labForm").addEventListener("submit", guard(async (e) => {
    e.preventDefault();
    const attack = $("labAttack").value;
    const payload = {
      attack,
      n_attackers: attack === "none" ? 0 : Number($("labN").value),
      defence: $("labDefence").value,
      rounds: Number($("labRounds").value),
      seed: Number($("labSeed").value),
      compare: $("labCompare").checked,
      dishonest_server: $("labLie").checked,
    };
    $("labRun").disabled = true;
    try {
      const job = await api("/api/simulations", { method: "POST", body: JSON.stringify(payload) });
      $("labStop").hidden = false;
      $("labStatus").textContent = "Starting: deploying a new contract and computing the shared feature scale.";
      pollJob(job);
    } catch (err) {
      $("labRun").disabled = false;
      throw err;
    }
  }));
  $("labStop").addEventListener("click", guard(() => api("/api/simulations/stop", { method: "POST" })));
  $("labAttack").addEventListener("change", syncLabControls);
  $("labCompare").addEventListener("change", syncLabControls);

  /* ---------- detection ---------- */
  function syncFamilies() {
    const device = Number($("detDevice").value);
    const noMirai = device === 3 || device === 7;
    const items = [{ value: "any", label: "Mixed traffic" }, ...state.options.classes
      .filter((c) => !(noMirai && c.startsWith("mirai")))
      .map((c) => ({ value: c, label: pretty(c) }))];
    fillSelect($("detFamily"), items, "any");
  }
  $("detDevice").addEventListener("change", syncFamilies);
  $("detForm").addEventListener("submit", guard(async (e) => {
    e.preventDefault();
    const data = await api("/api/detect/replay", { method: "POST", body: JSON.stringify({ device: Number($("detDevice").value), family: $("detFamily").value, count: Number($("detCount").value), seed: Math.floor(Math.random() * 100000) }) });
    const tiles = $("detTiles");
    tiles.textContent = "";
    const s = data.summary;
    tile(tiles, "Records replayed", String(s.total), data.device_name, false);
    tile(tiles, "Attacks flagged", s.attacks ? `${s.attacks_flagged} of ${s.attacks}` : "none present", "recorded botnet traffic");
    tile(tiles, "False alarms", s.benign ? `${s.false_alarms} of ${s.benign}` : "none present", "normal traffic wrongly flagged");
    tile(tiles, "Exact family correct", `${s.correct_family} of ${s.total}`, "including benign");
    const body = document.querySelector("#detTable tbody");
    body.textContent = "";
    data.rows.forEach((r) => {
      const outcome = r.correct
        ? make("span", "tag ok", "Correct")
        : r.flagged === (r.true !== "benign")
          ? make("span", "tag warn", "Wrong family")
          : make("span", "tag bad", r.flagged ? "False alarm" : "Missed");
      const tr = row([{ text: r.index, cls: "num" }, { text: pretty(r.true) }, { text: pretty(r.predicted) }, { text: pct(r.confidence, 1), cls: "num" }, { node: outcome }]);
      tr.className = "selectable";
      tr.tabIndex = 0;
      const pick = guard(async () => {
        document.querySelectorAll("#detTable tr.selected").forEach((x) => x.classList.remove("selected"));
        tr.classList.add("selected");
        await explainRecord(data.device, r.index);
      });
      tr.addEventListener("click", pick);
      tr.addEventListener("keydown", (evt) => { if (evt.key === "Enter" || evt.key === " ") { evt.preventDefault(); pick(); } });
      body.appendChild(tr);
    });
    const first = data.rows.find((r) => r.flagged) || data.rows[0];
    if (first) {
      const idx = data.rows.indexOf(first);
      body.children[idx].classList.add("selected");
      await explainRecord(data.device, first.index);
    }
  }));
  async function explainRecord(device, index) {
    const data = await api("/api/detect/explain", { method: "POST", body: JSON.stringify({ device, index }) });
    const host = $("detExplain");
    host.textContent = "";
    const head = make("div", "explain-head");
    head.appendChild(make("strong", "", pretty(data.predicted)));
    head.appendChild(make("span", "muted", `${pct(data.confidence, 1)} confidence. True traffic: ${pretty(data.true_class)}.`));
    host.appendChild(head);
    const benign = data.predicted === "benign";
    host.appendChild(make("p", "muted small", benign
      ? `Judged normal. These features show how this record compares with normal traffic, measured against the closest attack family (${pretty(data.explained_against)}).`
      : "Bars to the right pushed the model towards this attack; bars to the left pushed it towards normal traffic."));
    const chart = make("div", "chart");
    host.appendChild(chart);
    Charts.hbars(chart, data.top.map((t) => ({ name: t.feature, value: t.attribution })), { color: Charts.css("--series-2"), negColor: Charts.css("--series-1"), fmt: (v) => v.toFixed(2), labelWidth: 150, tipName: "attribution" });
    Charts.legend(chart.parentElement.appendChild(make("div", "legend")), [{ name: "Towards attack", color: Charts.css("--series-2"), bar: true }, { name: "Towards normal", color: Charts.css("--series-1"), bar: true }]);
    const top = data.top[0];
    host.appendChild(make("p", "muted small", `Strongest signal: ${top.statistic} of the ${top.stream} stream over the ${top.window} window (scaled value ${top.value.toFixed(2)}, typical normal ${top.normal.toFixed(2)}). This describes the model's reasoning, not a proven cause.`));
  }

  /* ---------- explainability ---------- */
  function renderExplain() {
    const e = state.results?.explain_eval;
    const tiles = $("expTiles");
    tiles.textContent = "";
    if (!e) {
      tiles.appendChild(make("p", "muted", "Explanation results have not been generated yet."));
      return;
    }
    const del10 = e.deletion.find((d) => d.k === 10);
    tile(tiles, "Top 10 features deleted", pct(del10.evidence_removed_top, 0), `evidence removed, against ${pct(del10.evidence_removed_random, 0)} for 10 random features`);
    const seeds = Object.values(e.stability_across_seeds_spearman);
    tile(tiles, "Stability across seeds", num(seeds.reduce((a, b) => a + b) / seeds.length, 2), "rank correlation of feature importance");
    tile(tiles, "Untrained model check", num(e.sanity_random_model_spearman, 2), "correlation with random weights; low is good");
    tile(tiles, "Integration error", pct(e.completeness.median_relative_error, 3), "implementation check, not faithfulness");
    tile(tiles, "Reference point", "Average benign traffic", `the average of all traffic reads as ${pretty(e.baseline_check.global_mean_predicted_as)}, so it would be a poor reference`, true);
    Charts.hbars($("expTop"), e.top_features.slice(0, 14).map((f) => ({ name: f.feature, value: f.mean_abs_attribution })), { color: Charts.css("--series-1"), fmt: (v) => v.toFixed(2), labelWidth: 150, tipName: "mean |attribution|" });
    const series = [
      { name: "Top-attributed features", color: Charts.css("--series-1"), points: e.deletion.map((d) => ({ x: d.k, y: d.evidence_removed_top })) },
      { name: "Random features", color: Charts.css("--series-2"), points: e.deletion.map((d) => ({ x: d.k, y: d.evidence_removed_random })) },
    ];
    Charts.legend($("expLegend"), series.map((s) => ({ name: s.name, color: s.color })));
    const peak = Math.max(1, ...e.deletion.map((d) => d.evidence_removed_top));
    Charts.line($("expDeletion"), { series, xMin: 1, xMax: 50, yMax: Math.ceil(peak * 5) / 5, yTicks: Math.ceil(peak * 5), yFmt: (v) => `${Math.round(v * 100)}%`, xLabel: "features deleted", height: 250 });
    const groups = $("expGroups");
    groups.textContent = "";
    for (const [key, title] of [["statistic", "By statistic"], ["stream", "By traffic stream"], ["window", "By time window"]]) {
      const card = make("div", "card");
      card.appendChild(make("h2", "", title));
      const chart = make("div", "chart");
      card.appendChild(chart);
      groups.appendChild(card);
      Charts.hbars(chart, Object.entries(e.groups[key]).map(([k, v]) => ({ name: k, value: v })), { color: Charts.css("--series-1"), fmt: (v) => `${Math.round(v * 100)}%`, labelWidth: 90, tipName: "share of attribution" });
    }
    const body = document.querySelector("#expAttack tbody");
    body.textContent = "";
    for (const [attack, v] of Object.entries(e.under_attack)) {
      body.appendChild(row([{ text: pretty(attack) }, { text: num(v.macro_f1), cls: "num" }, { text: num(v.spearman_to_clean, 2), cls: "num" }, { text: pct(v.top20_overlap_to_clean, 0), cls: "num" }]));
    }
  }

  /* ---------- blockchain ---------- */
  async function refreshRunList() {
    const runs = await api("/api/simulations").catch(() => []);
    const usable = runs.filter((r) => r.status === "finished");
    const sel = $("chainRun");
    const previous = state.chainRunId;
    fillSelect(sel, usable.map((r) => ({ value: r.id, label: `${pretty(r.attack)} (${r.n_attackers} attackers), ${r.label}` })), previous);
    const hasRuns = usable.length > 0;
    for (const id of ["chainVerify", "chainTamper", "chainRestore"]) $(id).disabled = !hasRuns;
    if (!hasRuns) {
      setVerdict("info", "No finished run yet.", "Run a simulation in the Attack lab first. Its trust decisions will appear here.");
      document.querySelector("#chainTable tbody").textContent = "";
      document.querySelector("#chainRecords tbody").textContent = "";
      $("chainTiles").textContent = "";
      return;
    }
    if (!usable.some((r) => r.id === previous)) {
      state.chainRunId = usable.find((r) => r.label.startsWith("Proposed"))?.id || usable[0].id;
      sel.value = state.chainRunId;
      setVerdict("info", "Ready.", "Verify recomputes each record's hash and compares it with the chain.");
    }
    await loadChain();
  }
  function setVerdict(kind, title, detail) {
    const v = $("chainVerdict");
    v.hidden = false;
    v.className = `verdict ${kind}`;
    v.textContent = title;
    if (detail) v.appendChild(make("small", "", detail));
  }
  async function loadChain(highlight) {
    const data = await api(`/api/chain/${state.chainRunId}`);
    const tiles = $("chainTiles");
    tiles.textContent = "";
    if (data.info) {
      tile(tiles, "Contract", short(data.info.address), `chain id ${data.info.chain_id}`, true);
      tile(tiles, "Server (only oracle)", short(data.info.oracle), "clients cannot submit scores", true);
      tile(tiles, "Transactions", String(data.info.transactions), `latest block ${data.info.block}`);
      tile(tiles, "Gas used", Number(data.info.gas_used).toLocaleString(), "measured on the local chain");
    } else {
      tile(tiles, "Ledger", "In-memory", "the chain was offline during this run", true);
    }
    tile(tiles, "Models anchored", String(data.models.length), "one global model hash per round");
    $("chainSub").textContent = data.backend === "chain" ? "read from the contract" : "local ledger; the chain was offline for this run";
    const names = Object.fromEntries(state.options.devices.map((d) => [d.id, d.name]));
    const body = document.querySelector("#chainTable tbody");
    body.textContent = "";
    data.entries.forEach((e) => {
      const tag = e.accepted ? make("span", "tag ok", "Accepted") : make("span", "tag bad", "Rejected");
      const tr = row([{ text: e.round, cls: "num" }, { text: names[e.client] }, { text: e.score, cls: "num" }, { node: tag }, { text: `${e.rep_before} → ${e.rep_after}`, cls: "num" }, { text: short(e.update_hash), cls: "mono", title: e.update_hash }, { text: short(e.record_hash), cls: "mono", title: e.record_hash }]);
      if (highlight?.some((m) => m.round === e.round && m.client === e.client)) tr.classList.add("highlight");
      body.appendChild(tr);
    });
    const recs = document.querySelector("#chainRecords tbody");
    recs.textContent = "";
    data.records.forEach((r) => {
      const tr = row([{ text: r.round, cls: "num" }, { text: names[r.client] }, { text: num(r.cosine), cls: "num" }, { text: num(r.norm_ratio, 2), cls: "num" }, { text: r.reason, cls: "wrap" }]);
      if (highlight?.some((m) => m.round === r.round && m.client === r.client)) tr.classList.add("highlight");
      recs.appendChild(tr);
    });
  }
  $("chainRun").addEventListener("change", guard(async () => {
    state.chainRunId = $("chainRun").value;
    setVerdict("info", "Ready.", "Verify recomputes each record's hash and compares it with the chain.");
    await loadChain();
  }));
  $("chainVerify").addEventListener("click", guard(async () => {
    setVerdict("info", "Checking…", "Recomputing every record's hash and reading the chain.");
    const res = await api(`/api/chain/${state.chainRunId}/verify`, { method: "POST" });
    if (res.valid) {
      setVerdict("ok", `Verified: all ${res.checked} records match the chain.`, "Every recomputed hash equals the one stored on-chain.");
      await loadChain();
    } else {
      const m = res.mismatches[0];
      setVerdict("bad", `Tampering detected: ${res.mismatches.length} of ${res.checked} records do not match.`, `Round ${m.round}, client ${m.client}: ${m.problem}.`);
      await loadChain(res.mismatches);
    }
  }));
  $("chainTamper").addEventListener("click", guard(async () => {
    const res = await api(`/api/chain/${state.chainRunId}/tamper`, { method: "POST" });
    setVerdict("info", `Record changed: round ${res.round}, client ${res.client} was ${res.changed}.`, "This edited the off-chain record, as a dishonest operator might. Now press Verify.");
    await loadChain();
  }));
  $("chainRestore").addEventListener("click", guard(async () => {
    await api(`/api/chain/${state.chainRunId}/restore`, { method: "POST" });
    setVerdict("info", "Original records restored.", "Press Verify to confirm.");
    await loadChain();
  }));

  /* ---------- fog and IPFS ---------- */
  function fogVerdict(kind, title, detail, id = "fogVerdict") {
    const v = $(id);
    v.hidden = false;
    v.className = `verdict ${kind}`;
    v.textContent = title;
    if (detail) v.appendChild(make("small", "", detail));
  }
  const whereLabel = { ipfs: "IPFS node", local: "Local folder", swapped: "Swapped file", missing: "Missing" };
  async function refreshFog() {
    const runs = await api("/api/simulations").catch(() => []);
    const usable = runs.filter((r) => r.status === "finished");
    const previous = state.fogRunId;
    fillSelect($("fogRun"), usable.map((r) => ({ value: r.id, label: `${pretty(r.attack)} (${r.n_attackers} attackers), ${r.label}` })), previous);
    const hasRuns = usable.length > 0;
    for (const id of ["fogRecheck", "fogSwap", "fogRestore", "fogCheckFile"]) $(id).disabled = !hasRuns;
    $("fogDownload").setAttribute("aria-disabled", String(!hasRuns));
    if (!hasRuns) {
      fogVerdict("info", "No finished run yet.", "Run a simulation in the Attack lab first. For fog nodes, choose the defence Fog nodes + full stack.");
      for (const id of ["fogTiles", "fogNodes", "fogModel"]) $(id).textContent = "";
      document.querySelector("#fogFiles tbody").textContent = "";
      return;
    }
    if (!usable.some((r) => r.id === previous)) {
      state.fogRunId = usable.find((r) => r.label.startsWith("Fog"))?.id || usable.find((r) => r.label.startsWith("Proposed"))?.id || usable[0].id;
      $("fogRun").value = state.fogRunId;
      fogVerdict("info", "Ready.", "Re-check fetches every stored update by its on-chain hash, reruns the gate and aggregation, and compares the outcome with the chain.");
    }
    await loadFog();
  }
  function resetRecheck() {
    const body = document.querySelector("#fogRecheckTable tbody");
    body.textContent = "";
    const tr = document.createElement("tr");
    const td = make("td", "muted", "Press Re-check to fill this in.");
    td.colSpan = 4;
    tr.appendChild(td);
    body.appendChild(tr);
  }
  async function loadFog() {
    const [data, view] = await Promise.all([api(`/api/audit/${state.fogRunId}`), api(`/api/simulations/${state.fogRunId}`)]);
    state.fogData = data;
    state.fogView = view;
    const tiles = $("fogTiles");
    tiles.textContent = "";
    const inIpfs = data.files.filter((f) => f.where === "ipfs").length;
    tile(tiles, "Update store", data.storage.connected ? "Private IPFS node" : "Local folder", data.storage.connected ? `Kubo ${data.storage.version}, no network access` : "start the IPFS node to use it; addresses stay the same", true);
    tile(tiles, "Update files", String(data.files.length), `${inIpfs} on IPFS, ${data.files.length - inIpfs} elsewhere`);
    tile(tiles, "Gating", data.fog_mode ? "3 fog nodes" : "Central", data.fog_mode ? "each gates its own 3 devices" : "one gate over all 9 devices", true);
    tile(tiles, "Trust ledger", data.backend === "chain" ? "Blockchain" : "In-memory", data.backend === "chain" ? "hashes read from the contract" : "the chain was offline for this run", true);
    if (data.dishonest_server) tile(tiles, "Server behaviour", "Lying", "this run wrote false verdicts on purpose", true);
    const rounds = view.rounds.map((r) => r.round);
    fillSelect($("fogRound"), rounds.map((r) => ({ value: r, label: `Round ${r}` })), rounds[rounds.length - 1]);
    drawFogNodes();

    const names = Object.fromEntries(state.options.devices.map((d) => [d.id, d.name]));
    const model = $("fogModel");
    model.textContent = "";
    const download = $("fogDownload");
    if (data.model && !data.model.error) {
      for (const [k, v] of [["From round", data.model.round], ["Size", `${(data.model.size / 1024).toFixed(0)} KB of weights`], ["On-chain hash", short(data.model.model_hash)], ["IPFS address", short(data.model.cid)]]) {
        const kv = make("div", "kv");
        kv.appendChild(make("span", "", k));
        kv.appendChild(make("b", "", String(v)));
        model.appendChild(kv);
      }
      download.href = `/api/model/${state.fogRunId}/file`;
      download.setAttribute("aria-disabled", "false");
    } else {
      model.appendChild(make("p", "muted", data.model?.error || "No model available."));
      download.removeAttribute("href");
      download.setAttribute("aria-disabled", "true");
    }
    $("fogFilesSub").textContent = `${data.files.length} updates, one per client per round`;
    const body = document.querySelector("#fogFiles tbody");
    body.textContent = "";
    data.files.forEach((f) => {
      const verdict = f.accepted ? make("span", "tag ok", "Accepted") : make("span", "tag bad", "Rejected");
      const where = make("span", f.where === "ipfs" ? "tag ok" : f.where === "local" ? "tag warn" : "tag bad", whereLabel[f.where] || f.where);
      body.appendChild(row([{ text: f.round, cls: "num" }, { text: names[f.client] }, { node: verdict }, { text: short(f.cid), cls: "mono", title: f.cid }, { node: where }]));
    });
  }
  function drawFogNodes() {
    const host = $("fogNodes");
    host.textContent = "";
    const data = state.fogData;
    const round = state.fogView?.rounds.find((r) => r.round === Number($("fogRound").value));
    if (!data || !round) return;
    const names = Object.fromEntries(state.options.devices.map((d) => [d.id, d.name]));
    $("fogSub").textContent = data.fog_mode ? "each node gates only its own devices, then forwards what passes" : "this run used one central gate; shown by group for comparison. Choose Fog nodes + full stack in the Attack lab";
    for (const [fog, devices] of Object.entries(data.fog_groups)) {
      const clients = devices.map((d) => round.clients.find((c) => c.client === d));
      const card = make("div", "fog-node");
      const head = make("div", "head");
      head.appendChild(make("span", "", `Fog node ${fog}`));
      head.appendChild(make("span", "muted", `${clients.filter((c) => c.included).length} of ${clients.length} forwarded`));
      card.appendChild(head);
      for (const c of clients) {
        const line = make("div", "fog-device");
        const left = make("span", "", names[c.client]);
        left.appendChild(make("small", "", `${c.malicious ? "Simulated attacker, " : ""}cosine ${num(c.cosine)}, size ${num(c.norm_ratio, 2)}`));
        line.appendChild(left);
        line.appendChild(c.included ? make("span", "tag ok", "Forwarded") : c.on_probation ? make("span", "tag warn", "Probation") : make("span", "tag bad", "Rejected"));
        card.appendChild(line);
      }
      host.appendChild(card);
    }
  }
  $("fogRound").addEventListener("change", drawFogNodes);
  $("fogRun").addEventListener("change", guard(async () => {
    state.fogRunId = $("fogRun").value;
    fogVerdict("info", "Ready.", "Re-check fetches every stored update by its on-chain hash, reruns the gate and aggregation, and compares the outcome with the chain.");
    $("fogCheckResult").hidden = true;
    resetRecheck();
    await loadFog();
  }));
  $("fogRecheck").addEventListener("click", guard(async () => {
    fogVerdict("info", "Re-checking…", "Fetching every stored update and rerunning the decisions.");
    const res = await api(`/api/audit/${state.fogRunId}/recheck`, { method: "POST" });
    const records = res.chain_records_check ? "The record check on Blockchain audit passes for this run." : "The record check on Blockchain audit also fails for this run.";
    if (res.valid) {
      fogVerdict("ok", `Verified: ${res.rounds_checked} rounds recomputed from ${res.files_checked} stored files give the same decisions and models.`, records);
    } else {
      const p = res.problems[0];
      const who = p.client ? `Round ${p.round}, client ${p.client}` : `Round ${p.round}`;
      fogVerdict("bad", `Re-check failed: ${res.problems.length} ${res.problems.length === 1 ? "problem" : "problems"} across ${res.rounds_checked} rounds.`, `${who}: ${p.problem}. ${records}`);
    }
    const body = document.querySelector("#fogRecheckTable tbody");
    body.textContent = "";
    res.rounds.forEach((r) => {
      const differ = r.decisions_differ === null ? "n/a" : String(r.decisions_differ);
      const model = make("span", r.model === "matches" ? "tag ok" : r.model === "differs" ? "tag bad" : "tag warn", r.model === "matches" ? "Matches" : r.model === "differs" ? "Differs" : "Not checked");
      const tr = row([{ text: r.round, cls: "num" }, { text: `${r.files} of ${r.files_expected}`, cls: "num" }, { text: differ, cls: "num" }, { node: model }]);
      if (r.decisions_differ || r.model !== "matches" || r.files !== r.files_expected) tr.classList.add("highlight");
      body.appendChild(tr);
    });
    await loadFog();
  }));
  $("fogSwap").addEventListener("click", guard(async () => {
    const res = await api(`/api/audit/${state.fogRunId}/swap`, { method: "POST" });
    fogVerdict("info", `Now serving a different file for round ${res.round}, client ${res.client}.`, "A dishonest operator might do this. Now press Re-check.");
    await loadFog();
  }));
  $("fogRestore").addEventListener("click", guard(async () => {
    await api("/api/audit/restore", { method: "POST" });
    fogVerdict("info", "Original files restored.", "Press Re-check to confirm.");
    resetRecheck();
    await loadFog();
  }));
  $("fogCheckFile").addEventListener("change", guard(async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      const res = await api(`/api/model/${state.fogRunId}/check`, { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: await file.arrayBuffer() });
      if (res.matches) fogVerdict("ok", "This file holds the weights recorded on the chain.", `SHA-256 ${short(res.computed)}`, "fogCheckResult");
      else fogVerdict("bad", "This file does not match the chain.", `File ${short(res.computed)}, chain ${short(res.on_chain)}`, "fogCheckResult");
    } catch (err) {
      fogVerdict("bad", "Could not check this file.", err.message, "fogCheckResult");
    }
    e.target.value = "";
  }));

  /* ---------- evaluation ---------- */
  function renderResults() {
    const g = state.results?.grid_summary;
    if (!g) {
      $("resHeat").textContent = "Evaluation results have not been generated yet.";
      return;
    }
    const metric = $("resMetric").value;
    const lower = metric === "target_family_evasion";
    const attacks = g.attacks, defences = g.defences;
    const cell = (a, d) => g.cells.find((c) => c.attack === a && c.defence === d);
    const cells = attacks.map((a) => defences.map((d) => {
      const m = cell(a, d)?.[metric];
      return m && m.mean !== null ? m.mean : null;
    }));
    const colLabels = defences.map((d) => g.defence_labels[d]);
    const shortLabels = { fedavg: "No defence", median: "Median", trimmed: "Trim mean", multikrum: "Multi-Krum", gate: "Gate", gate_rep: "Gate + rep", gate_trim: "Gate + trim", proposed: "Proposed" };
    $("resSub").textContent = `${g.meta.seeds} seeds, 3 of 9 clients malicious, ${(g.meta.train_fraction * 100).toFixed(0)}% training sample, ${lower ? "lower" : "higher"} is better`;
    const host = $("resHeat");
    if ($("resTable").checked) {
      host.textContent = "";
      const wrap = make("div", "table-wrap");
      const table = document.createElement("table");
      const head = document.createElement("tr");
      head.appendChild(make("th", "", "Attack"));
      colLabels.forEach((c) => head.appendChild(make("th", "num", c)));
      const thead = document.createElement("thead");
      thead.appendChild(head);
      table.appendChild(thead);
      const body = document.createElement("tbody");
      attacks.forEach((a, i) => {
        body.appendChild(row([{ text: pretty(a) }, ...defences.map((d, j) => {
          const m = cell(a, d)?.[metric];
          return { text: cells[i][j] === null ? "n/a" : `${cells[i][j].toFixed(3)} ± ${m.std.toFixed(3)}`, cls: "num" };
        })]));
      });
      table.appendChild(body);
      wrap.appendChild(table);
      host.appendChild(wrap);
    } else {
      Charts.heatmap(host, {
        rows: attacks.map(pretty), cols: defences.map((d) => shortLabels[d]), cells, fmt: (v) => v.toFixed(2), labelWidth: 112, colHeight: 26, rotate: false, cellMinW: 84, highlightCol: defences.indexOf("proposed"),
        tip: (i, j, v) => {
          if (v === null || v === undefined) return "n/a";
          const c = cell(attacks[i], defences[j]);
          return `${v.toFixed(3)} ± ${c[metric].std.toFixed(3)}; attackers rejected ${c.attacker_reject_rate?.mean === null || c.attacker_reject_rate?.mean === undefined ? "n/a" : pct(c.attacker_reject_rate.mean, 0)}`;
        },
      });
    }
    const clean = cleanProposed();
    const classes = state.options.classes;
    const conf = clean.confusion;
    const norm = conf.map((r) => { const t = r.reduce((a, b) => a + b, 0) || 1; return r.map((v) => v / t); });
    Charts.heatmap($("resConfusion"), { rows: classes.map(pretty), cols: classes.map(pretty), cells: norm, fmt: (v) => (v < 0.005 ? "" : v.toFixed(2)), labelWidth: 112, colHeight: 86, cellMinW: 42, cellH: 26, tip: (i, j, v) => `${pct(v, 1)} of ${pretty(classes[i])} detected as ${pretty(classes[j])}` });
    const names = Object.fromEntries(state.options.devices.map((d) => [d.id, d.name]));
    const devBody = document.querySelector("#resDevices tbody");
    devBody.textContent = "";
    for (const [id, d] of Object.entries(clean.devices)) {
      devBody.appendChild(row([{ text: names[id] }, { text: pct(d.benign_recall), cls: "num" }, { text: d.benign_recall_ci ? `${pct(d.benign_recall_ci[0])} to ${pct(d.benign_recall_ci[1])}` : "n/a", cls: "num" }, { text: Number(d.benign_test_rows).toLocaleString(), cls: "num" }]));
    }
    const a = state.results.data_audit;
    const auditBody = document.querySelector("#resAudit tbody");
    auditBody.textContent = "";
    if (a) {
      const rows = [
        ["Always say attack", a.always_attack.binary_accuracy, null],
        [`One feature (${a.best_single_feature.feature})`, a.best_single_feature.binary_accuracy, null],
        ["Decision tree, depth 3", a.tree_depth3.binary_accuracy, a.tree_depth3.macro_f1],
        ["Logistic regression", a.logistic_regression.binary_accuracy, a.logistic_regression.macro_f1],
        ["Random forest, pooled data", a.random_forest.binary_accuracy, a.random_forest.macro_f1],
      ];
      if (clean) rows.push(["This project: federated, defended", clean.binary_accuracy.mean, clean.macro_f1.mean]);
      for (const [name, acc, f1] of rows) auditBody.appendChild(row([{ text: name }, { text: pct(acc), cls: "num" }, { text: f1 === null ? "n/a" : num(f1), cls: "num" }]));
      $("resAuditNote").textContent = "Attack versus benign is close to solved by simple models on this dataset, so that number alone says little. The family classification and the robustness under attack are the meaningful results.";
    }
    const limits = $("resLimits");
    limits.textContent = "";
    for (const text of g.limits) limits.appendChild(make("li", "", text));
  }
  $("resMetric").addEventListener("change", renderResults);
  $("resTable").addEventListener("change", renderResults);

  /* ---------- start ---------- */
  guard(async () => {
    state.options = await api("/api/options");
    state.results = await api("/api/results");
    fillSelect($("labAttack"), state.options.attacks.map((a) => ({ value: a.id, label: pretty(a.id) })), "sign_flip");
    fillSelect($("labN"), [1, 2, 3, 4].map((n) => ({ value: n, label: `${n} of 9` })), 3);
    fillSelect($("labDefence"), state.options.defences.map((d) => ({ value: d.id, label: d.label })), "proposed");
    fillSelect($("detDevice"), state.options.devices.map((d) => ({ value: d.id, label: d.name })), 5);
    syncFamilies();
    syncLabControls();
    await pollStatus();
    setInterval(pollStatus, 5000);
    const start = location.hash.slice(1);
    showView(document.getElementById(`view-${start}`) ? start : "overview");
    const job = await api("/api/jobs/current");
    if (job.status === "running") { $("labRun").disabled = true; $("labStop").hidden = false; pollJob(job); }
  })();
})();
