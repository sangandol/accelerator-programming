// Tiny SVG drawing library for the book's figures. API summary: tools/fig/README.md.
// A figure is a Fig: call drawing methods with absolute coordinates (px), then fig.svg().
// The canvas grows to fit what was drawn, so figures never set their own width or height.

export const INK = '#222';
export const MUTED = '#666';
// Palette names give [light fill, dark stroke/text]; anywhere a color is accepted, a name or a hex works.
export const PAL = {
  blue: ['#e8f0fb', '#3a5a8c'],
  orange: ['#fdebd3', '#b45f06'],
  green: ['#e3f2e5', '#2e7d32'],
  red: ['#fbe4e1', '#c0392b'],
  purple: ['#efe6f8', '#6a3d9a'],
  yellow: ['#fff5cc', '#8a6d00'],
  teal: ['#dff3f1', '#00796b'],
  gray: ['#efefef', '#555555'],
  white: ['#ffffff', '#222222'],
};
const FONT = "'PingFang SC','Microsoft YaHei','Noto Sans CJK SC',system-ui,sans-serif";
const fillOf = (c) => (c === 'none' ? 'none' : PAL[c]?.[0] ?? c);
const inkOf = (c) => (c ? PAL[c]?.[1] ?? c : INK);
const r1 = (n) => Math.round(n * 10) / 10;
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// ---- text: "plain $math$ plain"; in math, letters are italic, _x _{..} ^x ^{..} make sub/superscripts.
const SYM = {
  otimes: '⊗', oplus: '⊕', wedge: '∧', vee: '∨', neg: '¬', cdot: '·', times: '×', le: '≤', ge: '≥',
  ne: '≠', to: '→', gets: '←', leftarrow: '←', rightarrow: '→', dots: '…', ldots: '…', infty: '∞',
  in: '∈', sum: 'Σ', alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', tau: 'τ', mu: 'μ', pi: 'π', sigma: 'σ',
  lfloor: '⌊', rfloor: '⌋', lceil: '⌈', rceil: '⌉', mid: '∣', approx: '≈', log: 'log', max: 'max', min: 'min',
};

function runs(str) {
  const out = [];
  const push = (s, lvl, it) => {
    const last = out[out.length - 1];
    if (last && last.lvl === lvl && last.it === it) last.s += s;
    else out.push({ s, lvl, it });
  };
  const emit = (s, lvl) => {
    for (let j = 0; j < s.length; j++) {
      const m = s[j] === '\\' && /^\\([a-zA-Z]+)/.exec(s.slice(j));
      if (m) {
        const sym = SYM[m[1]] ?? m[1];
        push(sym, lvl, false);
        j += m[0].length - 1;
      } else if (s[j] === "'") push('′', lvl, false);
      else if (s[j] !== '{' && s[j] !== '}') push(s[j], lvl, /[A-Za-z]/.test(s[j]));
    }
  };
  String(str).split('$').forEach((part, k) => {
    if (k % 2 === 0) {
      if (part) push(part, 0, false);
      return;
    }
    let i = 0;
    const group = () => {
      if (part[i] === '{') {
        let depth = 1;
        let j = i + 1;
        for (; j < part.length && depth; j++) depth += part[j] === '{' ? 1 : part[j] === '}' ? -1 : 0;
        const g = part.slice(i + 1, j - 1);
        i = j;
        return g;
      }
      const m = part[i] === '\\' && /^\\[a-zA-Z]+/.exec(part.slice(i));
      const g = m ? m[0] : part[i];
      i += g.length;
      return g;
    };
    while (i < part.length) {
      const ch = part[i];
      if (ch === '_' || ch === '^') {
        i++;
        emit(group(), ch === '_' ? -1 : 1);
      } else {
        emit(group(), 0);
      }
    }
  });
  return out;
}

function charW(ch) {
  const c = ch.codePointAt(0);
  if ((c >= 0x2e80 && c <= 0x9fff) || (c >= 0xff00 && c <= 0xffef) || (c >= 0x3000 && c <= 0x303f)) return 1;
  if (ch === ' ') return 0.3;
  if (/[A-Z]/.test(ch)) return 0.66;
  if (/[a-z0-9]/.test(ch)) return 0.55;
  if (/[.,:;′|!()[\]{}]/.test(ch)) return 0.32;
  return 0.75;
}

// Returns SVG markup for the runs and the estimated width in px.
function typeset(str, size) {
  let html = '';
  let width = 0;
  let cur = 0;
  for (const { s, lvl, it } of runs(str)) {
    const shift = lvl < 0 ? 0.3 * size : lvl > 0 ? -0.42 * size : 0;
    const fs = lvl ? 0.72 : 1;
    const attrs = [];
    if (shift !== cur) attrs.push(`dy="${r1(shift - cur)}"`);
    if (fs !== 1) attrs.push(`font-size="${r1(size * fs)}"`);
    if (it) attrs.push('font-style="italic"');
    html += attrs.length ? `<tspan ${attrs.join(' ')}>${esc(s)}</tspan>` : esc(s);
    cur = shift;
    for (const ch of s) width += charW(ch) * size * fs;
  }
  return { html, width };
}

// ---- anchors: boxes and nodes expose their geometry and side points.
function rectAnchors(x, y, w, h) {
  return {
    kind: 'rect', x, y, w, h, cx: x + w / 2, cy: y + h / 2,
    L: (t = 0.5) => [x, y + h * t],
    R: (t = 0.5) => [x + w, y + h * t],
    T: (t = 0.5) => [x + w * t, y],
    B: (t = 0.5) => [x + w * t, y + h],
  };
}
const center = (a) => (Array.isArray(a) ? a : [a.cx, a.cy]);

// Point where the ray from a's center toward p leaves a (points stay put).
function clip(a, p) {
  if (Array.isArray(a)) return a;
  const [dx, dy] = [p[0] - a.cx, p[1] - a.cy];
  const len = Math.hypot(dx, dy) || 1;
  if (a.kind === 'circle') return [a.cx + (dx / len) * a.r, a.cy + (dy / len) * a.r];
  const s = Math.min(dx ? a.w / 2 / Math.abs(dx) : Infinity, dy ? a.h / 2 / Math.abs(dy) : Infinity);
  return [a.cx + dx * s, a.cy + dy * s];
}

const GATE_SYM = { and: '∧', or: '∨', xor: '⊕', not: '¬', nand: '⊼', nor: '⊽' };

export class Fig {
  constructor({ pad = 12 } = {}) {
    this.pad = pad;
    this.els = [];
    this.texts = [];
    this.segs = [];
    this.markers = new Map();
    this.pins = {};
    this.bb = [Infinity, Infinity, -Infinity, -Infinity];
  }

  _bb(x0, y0, x1, y1) {
    const b = this.bb;
    this.bb = [Math.min(b[0], x0, x1), Math.min(b[1], y0, y1), Math.max(b[2], x0, x1), Math.max(b[3], y0, y1)];
  }

  // Raw SVG markup with its bounding box. z: 1 shapes, 2 lines, 3 dots, 4 text (drawn in that order).
  raw(markup, [x0, y0, x1, y1], z = 2) {
    this.els.push({ z, markup });
    this._bb(x0, y0, x1, y1);
    return this;
  }

  _marker(color, size) {
    const id = `ah-${color.replace('#', '')}-${size}`;
    if (!this.markers.has(id))
      this.markers.set(id, `<marker id="${id}" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="${size}" markerHeight="${size}" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${color}"/></marker>`);
    return id;
  }

  _stroke({ color, width = 1.6, dash, arrow } = {}) {
    const c = inkOf(color);
    let a = ` fill="none" stroke="${c}" stroke-width="${width}" stroke-linejoin="round"`;
    if (dash) a += ` stroke-dasharray="${dash}"`;
    if (arrow) {
      const id = this._marker(c, Math.max(10, Math.round(width * 4)));
      if (arrow === true || arrow === 'end' || arrow === 'both') a += ` marker-end="url(#${id})"`;
      if (arrow === 'start' || arrow === 'both') a += ` marker-start="url(#${id})"`;
    }
    return a;
  }

  // ---- primitives
  text(x, y, s, { size = 14, anchor = 'middle', color, weight, z = 4 } = {}) {
    const { html, width } = typeset(s, size);
    const w = weight ? ` font-weight="${weight}"` : '';
    this.els.push({ z, markup: `<text x="${r1(x)}" y="${r1(y + size * 0.35)}" text-anchor="${anchor}" font-size="${size}" fill="${inkOf(color)}"${w}>${html}</text>` });
    const x0 = anchor === 'middle' ? x - width / 2 : anchor === 'end' ? x - width : x;
    this._bb(x0, y - size * 0.65, x0 + width, y + size * 0.65);
    this.texts.push({ s, x0, y0: y - size * 0.5, x1: x0 + width, y1: y + size * 0.5 });
    return this;
  }

  note(x, y, s, opts = {}) {
    return this.text(x, y, s, { size: 12, color: MUTED, ...opts });
  }

  rect(x, y, w, h, { fill = 'none', stroke, width = 1.6, rx = 6, dash, z = 1 } = {}) {
    const d = dash ? ` stroke-dasharray="${dash}"` : '';
    const s = stroke === 'none' ? ' stroke="none"' : ` stroke="${inkOf(stroke)}" stroke-width="${width}"`;
    this.raw(`<rect x="${r1(x)}" y="${r1(y)}" width="${r1(w)}" height="${r1(h)}" rx="${rx}" fill="${fillOf(fill)}"${s}${d}/>`, [x, y, x + w, y + h], z);
    return rectAnchors(x, y, w, h);
  }

  // Filled box with a centered label. hollow: outline only (for grouping frames).
  box(x, y, w, h, label = '', { color = 'blue', size = 14, rx = 6, dash, width = 1.6, hollow = false, textColor, z = 1 } = {}) {
    const [fill, stroke] = PAL[color] ?? [color, INK];
    const a = this.rect(x, y, w, h, { fill: hollow ? 'none' : fill, stroke, width, rx, dash, z });
    if (label) this.text(x + w / 2, y + h / 2, label, { size, color: textColor ?? stroke });
    return a;
  }

  circle(cx, cy, r, { fill = 'none', stroke, width = 1.6, z = 1 } = {}) {
    const s = stroke === 'none' ? ' stroke="none"' : ` stroke="${inkOf(stroke)}" stroke-width="${width}"`;
    this.raw(`<circle cx="${r1(cx)}" cy="${r1(cy)}" r="${r}" fill="${fillOf(fill)}"${s}/>`, [cx - r, cy - r, cx + r, cy + r], z);
    return { kind: 'circle', cx, cy, r, x: cx - r, y: cy - r, w: 2 * r, h: 2 * r };
  }

  // Circle with a label: graph nodes, gates in trees.
  node(cx, cy, label = '', { r = 13, color = 'blue', size = 16, solid = false } = {}) {
    const [fill, stroke] = PAL[color] ?? [color, INK];
    const a = this.circle(cx, cy, r, { fill: solid ? stroke : fill, stroke });
    if (label) this.text(cx, cy, label, { size, color: solid ? '#fff' : stroke });
    return a;
  }

  dot(x, y, r = 3.2, color) {
    this.raw(`<circle cx="${r1(x)}" cy="${r1(y)}" r="${r}" fill="${inkOf(color)}"/>`, [x - r, y - r, x + r, y + r], 3);
    return [x, y];
  }

  line(points, opts = {}) {
    const pts = points.map(center);
    for (let i = 1; i < pts.length; i++) this.segs.push([pts[i - 1], pts[i]]);
    this.raw(`<polyline points="${pts.map(([x, y]) => `${r1(x)},${r1(y)}`).join(' ')}"${this._stroke(opts)}/>`, [
      Math.min(...pts.map((p) => p[0])), Math.min(...pts.map((p) => p[1])),
      Math.max(...pts.map((p) => p[0])), Math.max(...pts.map((p) => p[1])),
    ], opts.z ?? 2);
    return this;
  }

  arrow(p, q, opts = {}) {
    return this.line([p, q], { arrow: 'end', ...opts });
  }

  // Connect two shapes (or points) edge to edge; bend > 0 curves to the left of p→q.
  link(a, b, { bend = 0, ...opts } = {}) {
    if (!bend) return this.line([clip(a, center(b)), clip(b, center(a))], opts);
    const [ca, cb] = [center(a), center(b)];
    const [mx, my] = [(ca[0] + cb[0]) / 2, (ca[1] + cb[1]) / 2];
    const len = Math.hypot(cb[0] - ca[0], cb[1] - ca[1]);
    const c = [mx + ((cb[1] - ca[1]) / len) * bend * len, my - ((cb[0] - ca[0]) / len) * bend * len];
    const [p, q] = [clip(a, c), clip(b, c)];
    this.raw(`<path d="M${r1(p[0])},${r1(p[1])} Q${r1(c[0])},${r1(c[1])} ${r1(q[0])},${r1(q[1])}"${this._stroke(opts)}/>`,
      [Math.min(p[0], q[0], c[0]), Math.min(p[1], q[1], c[1]), Math.max(p[0], q[0], c[0]), Math.max(p[1], q[1], c[1])], opts.z ?? 2);
    return this;
  }

  // Bracket under (or beside) a span, with a label: brace(x0, y, x1, '长为 $2^k$') or {vertical: true} for (x, y0, y1).
  brace(a0, at, a1, label = '', { vertical = false, flip = false, size = 12, color = MUTED, d = 6 } = {}) {
    const s = flip ? -d : d;
    if (!vertical) {
      this.line([[a0, at], [a0, at + s], [a1, at + s], [a1, at]], { color, width: 1.2 });
      if (label) this.text((a0 + a1) / 2, at + s + Math.sign(s) * (size * 0.5 + 6), label, { size, color });
    } else {
      this.line([[at, a0], [at + s, a0], [at + s, a1], [at, a1]], { color, width: 1.2 });
      if (label) this.text(at + s + Math.sign(s) * 6, (a0 + a1) / 2, label, { size, color, anchor: s > 0 ? 'start' : 'end' });
    }
    return this;
  }

  // ---- circuits: gates are 44×44; pins name.in0 / name.in1 (y+14, y+30), name.in (single), name.out.
  gate(name, kind, x, y, { color = 'blue' } = {}) {
    const [fill, stroke] = PAL[color];
    this.rect(x, y, 44, 44, { fill, stroke });
    this.text(x + 22, y + 22, GATE_SYM[kind] ?? kind, { size: GATE_SYM[kind] ? 20 : 12, color: stroke });
    Object.assign(this.pins, {
      [`${name}.in0`]: [x, y + 14], [`${name}.in1`]: [x, y + 30], [`${name}.in`]: [x, y + 22], [`${name}.out`]: [x + 44, y + 22],
    });
    return rectAnchors(x, y, 44, 44);
  }

  // Named terminal with a label: side 'in' (label left), 'out' (right), 'top', 'bottom'.
  pin(name, x, y, label = '', side = 'in') {
    this.pins[name] = [x, y];
    if (label) {
      const pos = { in: [x - 8, y, 'end'], out: [x + 8, y, 'start'], top: [x, y - 12, 'middle'], bottom: [x, y + 14, 'middle'] }[side];
      this.text(pos[0], pos[1], label, { size: 15, anchor: pos[2] });
    }
    return [x, y];
  }

  P(ref) {
    if (typeof ref !== 'string') return center(ref);
    if (!this.pins[ref]) throw new Error(`unknown pin ${ref}`);
    return this.pins[ref];
  }

  // Wire from src to one or more dsts through a vertical spine at x (default: halfway); dots mark T-junctions.
  net(src, dsts, x, opts = {}) {
    const s = this.P(src);
    const ds = [].concat(dsts).map((d) => this.P(d));
    const sx = x ?? (s[0] + Math.min(...ds.map((d) => d[0]))) / 2;
    const ys = [s[1], ...ds.map((d) => d[1])];
    const [lo, hi] = [Math.min(...ys), Math.max(...ys)];
    if (s[0] !== sx) this.line([s, [sx, s[1]]], { ...opts, arrow: false });
    if (lo < hi) this.line([[sx, lo], [sx, hi]], { ...opts, arrow: false });
    for (const d of ds) if (d[0] !== sx) this.line([[sx, d[1]], d], opts);
    for (const y of new Set(ys)) {
      const deg = (s[1] === y && s[0] !== sx) + ds.filter((d) => d[1] === y && d[0] !== sx).length + (y > lo) + (y < hi);
      if (deg >= 3) this.dot(sx, y, 3.2, opts.color);
    }
    return this;
  }

  // ---- column networks (prefix / reduction / butterfly): ops are [level, i, j] = column i combines column j.
  colnet(x0, y0, { n, ops, dx = 52, dy = 56, tap = 36, top = 'x', bottom = 'y', levelLabel = (L) => `第 ${L + 1} 级`, color = 'blue' }) {
    const levels = Math.max(...ops.map((o) => o[0])) + 1;
    const X = (i) => x0 + dx * i;
    const Y = (L) => y0 + 12 + dy * (L + 1);
    const end = Y(levels - 1) + 30;
    const lab = (spec, i) => (typeof spec === 'function' ? spec(i) : `$${spec}_{${i}}$`);
    for (let i = 0; i < n; i++) {
      if (top) this.text(X(i), y0, lab(top, i), { size: 15 });
      this.line([[X(i), y0 + 12], [X(i), end]]);
      if (bottom) this.text(X(i), end + 18, lab(bottom, i), { size: 15 });
    }
    for (let L = 0; L < levels; L++) if (levelLabel) this.note(x0 - 28, Y(L), levelLabel(L), { anchor: 'end' });
    for (const [L, i, j] of ops) {
      this.line([[X(j), Y(L) - tap], [X(i), Y(L)]]);
      this.dot(X(j), Y(L) - tap, 2.6);
      this.dot(X(i), Y(L), 6.5, PAL[color][1]);
    }
    return { X, Y, end };
  }

  // ---- grids (matrices, layouts, tiles): fill/label are (r, c) => value or null.
  grid(x0, y0, { rows, cols, cw = 26, ch = 26, fill, label, size = 12, stroke = '#999', rowLabels, colLabels, labelSize = 12 }) {
    const cell = (r, c) => rectAnchors(x0 + c * cw, y0 + r * ch, cw, ch);
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++) {
        const f = fill?.(r, c);
        const a = this.rect(x0 + c * cw, y0 + r * ch, cw, ch, { fill: f ?? '#fff', stroke, width: 0.8, rx: 0 });
        const t = label?.(r, c);
        if (t !== null && t !== undefined && t !== '') this.text(a.cx, a.cy, String(t), { size, color: f ? inkOf(f) : INK });
      }
    const lab = (spec, k) => (typeof spec === 'function' ? spec(k) : spec[k]);
    if (rowLabels) for (let r = 0; r < rows; r++) this.text(x0 - 6, y0 + (r + 0.5) * ch, lab(rowLabels, r), { size: labelSize, anchor: 'end', color: MUTED });
    if (colLabels) for (let c = 0; c < cols; c++) this.text(x0 + (c + 0.5) * cw, y0 - 10, lab(colLabels, c), { size: labelSize, color: MUTED });
    return {
      cell, x: x0, y: y0, w: cols * cw, h: rows * ch,
      // Rectangle covering cells [r0, r1] × [c0, c1] (inclusive).
      region: (r0, c0, r1, c1) => rectAnchors(x0 + c0 * cw, y0 + r0 * ch, (c1 - c0 + 1) * cw, (r1 - r0 + 1) * ch),
    };
  }

  // Thick outline around a region (e.g. grid.region(...)) to highlight a block.
  // side: where the label goes ('top', 'bottom', 'left', 'right'); outside the grid, use 'right' or 'bottom'.
  frame(a, { color = 'red', width = 2.5, dash, label, size = 12, rx = 2, side = 'top' } = {}) {
    this.rect(a.x, a.y, a.w, a.h, { stroke: color, width, dash, rx, z: 3 });
    const pos = {
      top: [a.cx, a.y - 10, 'middle'], bottom: [a.cx, a.y + a.h + 12, 'middle'],
      left: [a.x - 8, a.cy, 'end'], right: [a.x + a.w + 8, a.cy, 'start'],
    }[side];
    if (label) this.text(pos[0], pos[1], label, { size, color, anchor: pos[2] });
    return a;
  }

  // ---- timelines (pipelines, DMA/compute overlap): bars are [lane, t0, t1, label, color].
  timeline(x0, y0, { lanes, bars, unit = 36, lh = 30, gap = 10, size = 13, axis = true, ticks = 1, tickLabel = (t) => String(t), axisLabel = '时间', deps = [], end }) {
    const tx = (t) => x0 + t * unit;
    const ly = (k) => y0 + k * (lh + gap);
    lanes.forEach((name, k) => this.text(x0 - 10, ly(k) + lh / 2, name, { size, anchor: 'end' }));
    const out = bars.map(([k, t0, t1, label = '', color = 'blue']) => this.box(tx(t0), ly(k), (t1 - t0) * unit, lh, label, { color, size, rx: 3 }));
    const T = end ?? Math.max(...bars.map((b) => b[2]));
    if (axis) {
      const ya = ly(lanes.length) - gap + 10;
      this.line([[x0, ya], [tx(T) + 18, ya]], { arrow: 'end', color: MUTED });
      if (ticks) for (let t = 0; t <= T; t += ticks) {
        this.line([[tx(t), ya - 3], [tx(t), ya + 3]], { color: MUTED });
        this.note(tx(t), ya + 13, tickLabel(t));
      }
      if (axisLabel) this.note(tx(T) + 22, ya, axisLabel, { anchor: 'start' });
    }
    for (const [a, b] of deps) this.arrow(out[a].R(), out[b].L(), { color: 'red', width: 1.4 });
    return { bars: out, tx, ly };
  }

  // ---- plots (roofline etc.): axis = {min, max, log, label, ticks: [values], fmt}; series = [{pts, color, dash, label}].
  plot(x0, y0, { w = 360, h = 220, x: X, y: Y, series = [], marks = [], size = 12 }) {
    const scale = (A, p0, p1) => (v) => {
      const f = A.log ? (Math.log10(v) - Math.log10(A.min)) / (Math.log10(A.max) - Math.log10(A.min)) : (v - A.min) / (A.max - A.min);
      return p0 + f * (p1 - p0);
    };
    const sx = scale(X, x0, x0 + w);
    const sy = scale(Y, y0 + h, y0);
    this.line([[x0, y0 + h], [x0 + w + 12, y0 + h]], { arrow: 'end' });
    this.line([[x0, y0 + h], [x0, y0 - 12]], { arrow: 'end' });
    const fmt = (A, v) => (A.fmt ? A.fmt(v) : String(v));
    for (const v of X.ticks ?? []) {
      this.line([[sx(v), y0 + h], [sx(v), y0 + h + 4]]);
      this.text(sx(v), y0 + h + 14, fmt(X, v), { size, color: MUTED });
    }
    for (const v of Y.ticks ?? []) {
      this.line([[x0 - 4, sy(v)], [x0, sy(v)]]);
      this.text(x0 - 8, sy(v), fmt(Y, v), { size, color: MUTED, anchor: 'end' });
    }
    if (X.label) this.text(x0 + w + 16, y0 + h, X.label, { size, anchor: 'start' });
    if (Y.label) this.text(x0, y0 - 24, Y.label, { size });
    for (const { pts, color = 'blue', dash, width = 2.2, label, at = pts.length - 1, dx = 6, dy = -10 } of series) {
      this.line(pts.map(([a, b]) => [sx(a), sy(b)]), { color, dash, width });
      if (label) this.text(sx(pts[at][0]) + dx, sy(pts[at][1]) + dy, label, { size, color, anchor: 'start' });
    }
    for (const { at: [a, b], label, color = 'red', dx = 6, dy = -10 } of marks) {
      this.dot(sx(a), sy(b), 4, color);
      if (label) this.text(sx(a) + dx, sy(b) + dy, label, { size, color, anchor: 'start' });
    }
    return { sx, sy };
  }

  // ---- topologies: nodes on a ring / a 2-D mesh; returned node(i) works with link().
  ring(cx, cy, { n, r = 80, nodeR = 16, label = (i) => String(i), color = 'blue', links = true, start = -90 }) {
    const nodes = [];
    for (let i = 0; i < n; i++) {
      const t = ((start + (360 * i) / n) * Math.PI) / 180;
      nodes.push({ kind: 'circle', cx: cx + r * Math.cos(t), cy: cy + r * Math.sin(t), r: nodeR });
    }
    if (links) for (let i = 0; i < n; i++) this.link(nodes[i], nodes[(i + 1) % n], { color: MUTED });
    nodes.forEach((a, i) => this.node(a.cx, a.cy, label(i), { r: nodeR, color, size: 13 }));
    return { node: (i) => nodes[((i % n) + n) % n] };
  }

  mesh(x0, y0, { rows, cols, d = 64, nodeR = 15, label = (r, c) => `${r},${c}`, color = 'blue', wrap = false }) {
    const at = (r, c) => ({ kind: 'circle', cx: x0 + c * d, cy: y0 + r * d, r: nodeR });
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++) {
        if (c + 1 < cols) this.link(at(r, c), at(r, c + 1), { color: MUTED });
        if (r + 1 < rows) this.link(at(r, c), at(r + 1, c), { color: MUTED });
      }
    if (wrap) {
      const s = d * 0.45;
      for (let r = 0; r < rows; r++) {
        const y = y0 + r * d;
        this.line([[x0 - nodeR, y], [x0 - s, y]], { color: MUTED, dash: '3 3' });
        this.line([[x0 + (cols - 1) * d + nodeR, y], [x0 + (cols - 1) * d + s, y]], { color: MUTED, dash: '3 3' });
      }
      for (let c = 0; c < cols; c++) {
        const x = x0 + c * d;
        this.line([[x, y0 - nodeR], [x, y0 - s]], { color: MUTED, dash: '3 3' });
        this.line([[x, y0 + (rows - 1) * d + nodeR], [x, y0 + (rows - 1) * d + s]], { color: MUTED, dash: '3 3' });
      }
    }
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) this.node(x0 + c * d, y0 + r * d, label(r, c), { r: nodeR, color, size: 11 });
    return { node: at };
  }

  // ---- bit fields (number formats): fields = [[bits, label, color]], most significant first.
  bitfield(x0, y0, { fields, unit = 16, h = 30, size = 13, index = true }) {
    const total = fields.reduce((s, f) => s + f[0], 0);
    let x = x0;
    let hi = total - 1;
    return fields.map(([bits, label, color = 'blue']) => {
      const a = this.box(x, y0, bits * unit, h, label, { color, size, rx: 0 });
      for (let k = 1; k < bits; k++) this.line([[x + k * unit, y0 + h - 5], [x + k * unit, y0 + h]], { color: inkOf(color), width: 0.8 });
      if (index) {
        this.note(x + unit / 2, y0 - 9, String(hi), { size: 10 });
        if (bits > 1) this.note(x + (bits - 0.5) * unit, y0 - 9, String(hi - bits + 1), { size: 10 });
      }
      x += bits * unit;
      hi -= bits;
      return a;
    });
  }

  // ---- output
  // Warnings for labels that overlap each other (cheap check instead of looking at a render).
  lint() {
    const t = this.texts;
    const out = [];
    for (let i = 0; i < t.length; i++)
      for (let j = i + 1; j < t.length; j++) {
        const [a, b] = [t[i], t[j]];
        const w = Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0);
        const h = Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0);
        if (w > 1 && h > 1 && w * h > 0.2 * Math.min((a.x1 - a.x0) * (a.y1 - a.y0), (b.x1 - b.x0) * (b.y1 - b.y0)))
          out.push(`text overlap: "${a.s}" / "${b.s}"`);
      }
    // Liang–Barsky: does segment pq pass through the (slightly shrunk) text box?
    const hits = (b, [p, q]) => {
      let [t0, t1] = [0, 1];
      const d = [q[0] - p[0], q[1] - p[1]];
      for (const [num, den] of [[p[0] - b.x0 - 2, -d[0]], [b.x1 - 2 - p[0], d[0]], [p[1] - b.y0 - 2, -d[1]], [b.y1 - 2 - p[1], d[1]]]) {
        if (den === 0) { if (num < 0) return false; continue; }
        const t = num / den;
        if (den < 0) t0 = Math.max(t0, t); else t1 = Math.min(t1, t);
        if (t0 > t1) return false;
      }
      return true;
    };
    for (const a of t) if (this.segs.some((sg) => hits(a, sg))) out.push(`line through text: "${a.s}"`);
    return out;
  }

  svg() {
    const p = this.pad;
    const [x0, y0, x1, y1] = this.bb;
    const vx = Math.floor(x0 - p);
    const vy = Math.floor(y0 - p);
    const W = Math.ceil(x1 + p) - vx;
    const H = Math.ceil(y1 + p) - vy;
    const body = [...this.els].sort((a, b) => a.z - b.z).map((e) => e.markup);
    const defs = this.markers.size ? `<defs>${[...this.markers.values()].join('')}</defs>\n` : '';
    return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="${vx} ${vy} ${W} ${H}" font-family="${FONT}" font-size="14">
${defs}<rect x="${vx}" y="${vy}" width="${W}" height="${H}" rx="8" fill="#ffffff"/>
${body.join('\n')}
</svg>
`;
  }
}

export const fig = (opts) => new Fig(opts);
