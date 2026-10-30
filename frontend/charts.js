/* Small SVG chart helpers: no external libraries, so the demo works with no internet. */
const Charts = (() => {
  const NS = "http://www.w3.org/2000/svg";
  const el = (name, attrs = {}, parent) => {
    const node = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  };
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  const tooltip = () => document.getElementById("tooltip");
  function showTip(evt, title, rows) {
    const tip = tooltip();
    tip.textContent = "";
    const head = document.createElement("div");
    head.className = "t-title";
    head.textContent = title;
    tip.appendChild(head);
    for (const row of rows) {
      const line = document.createElement("div");
      line.className = "t-row";
      if (row.color) {
        const key = document.createElement("i");
        key.style.background = row.color;
        line.appendChild(key);
      }
      const value = document.createElement("b");
      value.textContent = row.value;
      line.appendChild(value);
      const name = document.createElement("span");
      name.textContent = row.name;
      line.appendChild(name);
      tip.appendChild(line);
    }
    tip.hidden = false;
    const pad = 14;
    const box = tip.getBoundingClientRect();
    const x = evt.clientX ?? evt.target.getBoundingClientRect().right;
    const y = evt.clientY ?? evt.target.getBoundingClientRect().top;
    tip.style.left = `${Math.max(4, Math.min(window.innerWidth - box.width - 4, x + pad))}px`;
    tip.style.top = `${Math.max(4, Math.min(window.innerHeight - box.height - 4, y + pad))}px`;
  }
  const hideTip = () => { tooltip().hidden = true; };

  function legend(host, items) {
    host.textContent = "";
    for (const item of items) {
      const span = document.createElement("span");
      const key = document.createElement("i");
      if (item.bar) key.className = "dot";
      key.style.background = item.color;
      span.appendChild(key);
      span.appendChild(document.createTextNode(item.name));
      host.appendChild(span);
    }
  }

  /* series: [{name, color, points: [{x, y}]}]. y is drawn on [yMin, yMax]. */
  function line(host, { series, xMax, xMin = 1, yMin = 0, yMax = 1, yFmt = (v) => v.toFixed(2), yTicks = 5, xLabel = "round", threshold = null, height = 300, endLabels = true }) {
    const W = Math.max(420, host.clientWidth || 680), H = height, m = { l: 42, r: threshold ? 76 : endLabels ? 56 : 16, t: 10, b: 34 };
    host.textContent = "";
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": series.map((s) => s.name).join(", ") }, host);
    const px = (x) => m.l + ((x - xMin) / Math.max(1e-9, xMax - xMin)) * (W - m.l - m.r);
    const py = (y) => H - m.b - ((y - yMin) / (yMax - yMin)) * (H - m.t - m.b);
    for (let i = 0; i <= yTicks; i++) {
      const v = yMin + ((yMax - yMin) * i) / yTicks;
      el("line", { x1: m.l, x2: W - m.r, y1: py(v), y2: py(v), stroke: i === 0 ? css("--axis") : css("--grid"), "stroke-width": 1 }, svg);
      el("text", { x: m.l - 6, y: py(v) + 4, "text-anchor": "end" }, svg).textContent = yFmt(v);
    }
    const step = Math.max(1, Math.ceil((xMax - xMin) / 10));
    for (let x = xMin; x <= xMax; x += step) el("text", { x: px(x), y: H - m.b + 16, "text-anchor": "middle" }, svg).textContent = x;
    el("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle" }, svg).textContent = xLabel;
    if (threshold !== null) {
      el("line", { x1: m.l, x2: W - m.r, y1: py(threshold.y), y2: py(threshold.y), stroke: css("--text-2"), "stroke-width": 1 }, svg);
      el("text", { x: W - m.r + 4, y: py(threshold.y) + 4 }, svg).textContent = threshold.label;
    }
    for (const s of series) {
      if (!s.points.length) continue;
      const d = s.points.map((p, i) => `${i ? "L" : "M"}${px(p.x).toFixed(1)} ${py(p.y).toFixed(1)}`).join(" ");
      el("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width ?? 2, "stroke-linejoin": "round", "stroke-linecap": "round", opacity: s.opacity ?? 1 }, svg);
      const last = s.points[s.points.length - 1];
      el("circle", { cx: px(last.x), cy: py(last.y), r: 4, fill: s.color, stroke: css("--surface"), "stroke-width": 2 }, svg);
    }
    if (endLabels) {
      const placed = [];
      const ends = series.filter((s) => s.points.length).map((s) => ({ s, y: py(s.points[s.points.length - 1].y), v: s.points[s.points.length - 1].y }));
      ends.sort((a, b) => a.y - b.y);
      for (const e of ends) {
        if (placed.some((y) => Math.abs(y - e.y) < 14)) continue; // the legend and tooltip carry a converged series
        placed.push(e.y);
        const t = el("text", { x: W - m.r + 8, y: e.y + 4, class: "value" }, svg);
        t.textContent = yFmt(e.v);
      }
    }
    // Crosshair and tooltip: snap to the nearest x position.
    const cross = el("line", { y1: m.t, y2: H - m.b, stroke: css("--axis"), "stroke-width": 1, visibility: "hidden" }, svg);
    const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent", tabindex: 0 }, svg);
    const onMove = (evt) => {
      const box = svg.getBoundingClientRect();
      const sx = ((evt.clientX - box.left) / box.width) * W;
      const x = Math.max(xMin, Math.min(xMax, Math.round(xMin + ((sx - m.l) / (W - m.l - m.r)) * (xMax - xMin))));
      const rows = series.map((s) => ({ s, p: s.points.find((q) => q.x === x) })).filter((r) => r.p).map((r) => ({ color: r.s.color, value: yFmt(r.p.y), name: r.s.name }));
      if (!rows.length) return hideTip();
      cross.setAttribute("x1", px(x));
      cross.setAttribute("x2", px(x));
      cross.setAttribute("visibility", "visible");
      showTip(evt, `${xLabel} ${x}`, rows);
    };
    hit.addEventListener("pointermove", onMove);
    hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  }

  /* rows: [{name, value, label?}]. Signed values grow from a shared zero. */
  function hbars(host, rows, { color, negColor, fmt = (v) => v.toFixed(3), labelWidth = 170, barHeight = 14, gap = 8, tipName = "" } = {}) {
    host.textContent = "";
    const W = Math.max(420, host.clientWidth || 640), m = { r: 56 };
    const H = rows.length * (barHeight + gap) + 4;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "bar chart" }, host);
    const min = Math.min(0, ...rows.map((r) => r.value));
    const max = Math.max(0, ...rows.map((r) => r.value), 1e-9);
    const plot = W - labelWidth - m.r;
    const px = (v) => labelWidth + ((v - min) / (max - min)) * plot;
    el("line", { x1: px(0), x2: px(0), y1: 0, y2: H, stroke: css("--axis"), "stroke-width": 1 }, svg);
    rows.forEach((r, i) => {
      const y = i * (barHeight + gap) + 2;
      const x0 = Math.min(px(0), px(r.value));
      const width = Math.max(1, Math.abs(px(r.value) - px(0)));
      el("text", { x: labelWidth - 8, y: y + barHeight - 3, "text-anchor": "end", class: "name" }, svg).textContent = r.name;
      const fill = r.value < 0 ? (negColor ?? color) : color;
      const radius = Math.min(4, width / 2);
      const d = r.value >= 0
        ? `M${x0} ${y}h${width - radius}a${radius} ${radius} 0 0 1 ${radius} ${radius}v${barHeight - 2 * radius}a${radius} ${radius} 0 0 1 -${radius} ${radius}h-${width - radius}z`
        : `M${x0 + width} ${y}h-${width - radius}a${radius} ${radius} 0 0 0 -${radius} ${radius}v${barHeight - 2 * radius}a${radius} ${radius} 0 0 0 ${radius} ${radius}h${width - radius}z`;
      el("path", { d, fill }, svg);
      const label = el("text", { y: y + barHeight - 3, class: "value" }, svg);
      label.textContent = fmt(r.value);
      if (r.value >= 0) label.setAttribute("x", px(r.value) + 5);
      else { label.setAttribute("x", px(r.value) - 5); label.setAttribute("text-anchor", "end"); }
      const hit = el("rect", { x: 0, y: y - gap / 2, width: W, height: barHeight + gap, fill: "transparent" }, svg);
      hit.addEventListener("pointermove", (evt) => showTip(evt, r.name, [{ color: fill, value: fmt(r.value), name: tipName }]));
      hit.addEventListener("pointerleave", hideTip);
    });
  }

  function mix(a, b, t) {
    const parse = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
    const [x, y] = [parse(a), parse(b)];
    return `rgb(${x.map((v, i) => Math.round(v + (y[i] - v) * t)).join(",")})`;
  }

  /* cells: matrix of numbers (null = not available). */
  function heatmap(host, { rows, cols, cells, fmt, min = 0, max = 1, invert = false, showValues = true, labelWidth = 170, colHeight = 70, cellH = 30, cellMinW = 58, highlightCol = null, tip }) {
    host.textContent = "";
    const low = css("--seq-low"), high = css("--seq-high");
    const cellW = Math.max(cellMinW, Math.floor(560 / cols.length));
    const W = labelWidth + cols.length * cellW + 4, H = colHeight + rows.length * cellH + 4;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "heatmap" }, host);
    cols.forEach((c, j) => {
      const x = labelWidth + j * cellW + cellW / 2;
      const t = el("text", { transform: `translate(${x + 3},${colHeight - 6}) rotate(-35)`, "text-anchor": "start" }, svg);
      t.textContent = c;
      if (j === highlightCol) t.setAttribute("style", `font-weight:700;fill:${css("--text")}`);
    });
    rows.forEach((r, i) => {
      el("text", { x: labelWidth - 8, y: colHeight + i * cellH + cellH / 2 + 4, "text-anchor": "end" }, svg).textContent = r;
      cols.forEach((c, j) => {
        const v = cells[i][j];
        const x = labelWidth + j * cellW, y = colHeight + i * cellH;
        const t = v === null || v === undefined ? null : Math.max(0, Math.min(1, (v - min) / (max - min)));
        const shade = t === null ? css("--raised") : mix(low, high, invert ? 1 - t : t);
        const rect = el("rect", { x: x + 1, y: y + 1, width: cellW - 2, height: cellH - 2, rx: 3, fill: shade, class: "heat-cell", tabindex: 0 }, svg);
        const body = tip ? tip(i, j, v) : fmt(v);
        const open = (evt) => showTip(evt, `${r} / ${c}`, [{ value: v === null || v === undefined ? "n/a" : body, name: "" }]);
        rect.addEventListener("pointermove", open);
        rect.addEventListener("focus", open);
        rect.addEventListener("pointerleave", hideTip);
        rect.addEventListener("blur", hideTip);
        if (showValues && t !== null) {
          const shadeT = invert ? 1 - t : t;
          const darkTheme = css("--surface") === "#1a1a19";
          const dark = darkTheme ? shadeT < 0.55 : shadeT > 0.5;
          const label = el("text", { x: x + cellW / 2, y: y + cellH / 2 + 4, "text-anchor": "middle", "pointer-events": "none" }, svg);
          label.textContent = fmt(v);
          label.setAttribute("style", `fill:${dark ? "#ffffff" : "#0b0b0b"};font-size:11px`);
        }
      });
    });
  }

  return { line, hbars, heatmap, legend, css, showTip, hideTip };
})();
