// Generates the book's SVG figures into book/fig/. Run: node tools/figures.mjs
// Each figure is a function returning SVG text; edit the coordinates here, not the .svg files.
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const OUT = join(dirname(fileURLToPath(import.meta.url)), '..', 'book', 'fig');

const INK = '#222';
const MUTED = '#666';
const GATE_FILL = '#eef3fb';
const GATE_STROKE = '#3a5a8c';
const NODE = '#2f5597';
const CARRY = '#c0392b';
const FONT = "'PingFang SC','Microsoft YaHei','Noto Sans CJK SC',system-ui,sans-serif";

// White card background so the figures stay readable on dark pages.
function svg(w, h, body) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" font-family="${FONT}" font-size="14">
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${INK}"/></marker><marker id="arrow-carry" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${CARRY}"/></marker></defs>
<rect width="${w}" height="${h}" rx="8" fill="#ffffff"/>
${body.join('\n')}
</svg>
`;
}

// Text with an optional subscript: label('x', '0') renders x₀; variables are italic.
function label(x, y, base, sub = '', { anchor = 'middle', size = 15, color = INK, italic = true } = {}) {
  const style = italic ? ' font-style="italic"' : '';
  const subPart = sub ? `<tspan dy="4" font-size="${Math.round(size * 0.7)}">${sub}</tspan>` : '';
  return `<text x="${x}" y="${y}" text-anchor="${anchor}" font-size="${size}" fill="${color}"${style}>${base}${subPart}</text>`;
}

function note(x, y, s, { anchor = 'middle', size = 12, color = MUTED } = {}) {
  return `<text x="${x}" y="${y}" text-anchor="${anchor}" font-size="${size}" fill="${color}">${s}</text>`;
}

function wire(points, { color = INK, width = 1.6, arrow = false, dash = '' } = {}) {
  const pts = points.map(([x, y]) => `${x},${y}`).join(' ');
  const extra = (arrow ? ` marker-end="url(#${color === CARRY ? 'arrow-carry' : 'arrow'})"` : '') + (dash ? ` stroke-dasharray="${dash}"` : '');
  return `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linejoin="round"${extra}/>`;
}

function dot(x, y, r = 3.2, color = INK) {
  return `<circle cx="${x}" cy="${y}" r="${r}" fill="${color}"/>`;
}

// A 44×44 gate box at (x, y). Pins: in0 = y+14, in1 = y+30, single input = y+22, out = (x+44, y+22).
function gate(x, y, symbol) {
  return [
    `<rect x="${x}" y="${y}" width="44" height="44" rx="6" fill="${GATE_FILL}" stroke="${GATE_STROKE}" stroke-width="1.6"/>`,
    `<text x="${x + 22}" y="${y + 29}" text-anchor="middle" font-size="20" fill="${GATE_STROKE}">${symbol}</text>`,
  ].join('\n');
}

function box(x, y, w, h, text, { size = 14, dash = '' } = {}) {
  const d = dash ? ` stroke-dasharray="${dash}"` : '';
  const fill = dash ? 'none' : GATE_FILL;
  const t = text ? `\n<text x="${x + w / 2}" y="${y + h / 2 + 5}" text-anchor="middle" font-size="${size}" fill="${GATE_STROKE}">${text}</text>` : '';
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6" fill="${fill}" stroke="${GATE_STROKE}" stroke-width="1.6"${d}/>${t}`;
}

const AND = '∧';
const OR = '∨';
const XOR = '⊕';
const NOT = '¬';

// 图 1.1: 2-to-1 multiplexer.
function mux() {
  return svg(560, 240, [
    label(32, 49, 'a', '', { anchor: 'end' }),
    label(32, 125, 's', '', { anchor: 'end' }),
    label(32, 185, 'b', '', { anchor: 'end' }),
    wire([[40, 44], [230, 44]]),
    wire([[40, 120], [80, 120]]),
    wire([[80, 120], [80, 90], [110, 90]]),
    wire([[80, 120], [80, 164], [230, 164]]),
    dot(80, 120),
    wire([[154, 90], [195, 90], [195, 60], [230, 60]]),
    note(174, 83, '¬s', { size: 13, color: INK }),
    wire([[40, 180], [230, 180]]),
    wire([[274, 52], [305, 52], [305, 104], [340, 104]]),
    wire([[274, 172], [315, 172], [315, 120], [340, 120]]),
    wire([[384, 112], [430, 112]]),
    `<text x="438" y="117" font-size="15" fill="${INK}">mux(<tspan font-style="italic">s</tspan>, <tspan font-style="italic">a</tspan>, <tspan font-style="italic">b</tspan>)</text>`,
    gate(110, 68, NOT),
    gate(230, 30, AND),
    gate(230, 150, AND),
    gate(340, 90, OR),
    note(132, 226, '第 1 级'),
    note(252, 226, '第 2 级'),
    note(362, 226, '第 3 级'),
  ]);
}

// 图 1.2: 8-input AND as a chain and as a binary tree.
function andTree() {
  const col = (ox, i) => ox + 30 + 38 * i;
  const r = 13;
  const node = (x, y) => [
    `<circle cx="${x}" cy="${y}" r="${r}" fill="${GATE_FILL}" stroke="${GATE_STROKE}" stroke-width="1.6"/>`,
    `<text x="${x}" y="${y + 6}" text-anchor="middle" font-size="16" fill="${GATE_STROKE}">${AND}</text>`,
  ].join('\n');
  const lines = [];
  const nodes = [];
  const inputs = (ox) => {
    for (let i = 0; i < 8; i++) lines.push(label(col(ox, i), 22, 'x', String(i)));
  };

  // Chain: gate k combines the running result with x_k.
  const ox1 = 10;
  inputs(ox1);
  const yk = (k) => 34 + 36 * k;
  let prev = [col(ox1, 0), 36];
  for (let k = 1; k < 8; k++) {
    const here = [col(ox1, k), yk(k)];
    lines.push(wire([[col(ox1, k), 36], here]));
    lines.push(wire([prev, here]));
    nodes.push(node(...here));
    prev = here;
  }
  lines.push(wire([prev, [prev[0], yk(7) + 34]], { arrow: true }));
  lines.push(note(col(ox1, 3.5), 352, '逐个相与：7 个门，深度 7', { size: 13, color: INK }));

  // Tree: three levels of pairwise ANDs.
  const ox2 = 380;
  inputs(ox2);
  let level = Array.from({ length: 8 }, (_, i) => [col(ox2, i), 36]);
  let depth = 0;
  while (level.length > 1) {
    depth++;
    const next = [];
    for (let j = 0; j < level.length; j += 2) {
      const here = [(level[j][0] + level[j + 1][0]) / 2, 34 + 60 * depth];
      lines.push(wire([level[j], here]), wire([level[j + 1], here]));
      nodes.push(node(...here));
      next.push(here);
    }
    level = next;
  }
  lines.push(wire([level[0], [level[0][0], 34 + 60 * depth + 50]], { arrow: true }));
  lines.push(note(col(ox2, 3.5), 352, '二叉树：7 个门，深度 3', { size: 13, color: INK }));

  return svg(740, 370, [...lines, ...nodes]);
}

// 图 1.3: full adder = two half adders + OR.
function fullAdder() {
  return svg(540, 240, [
    box(88, 22, 98, 184, '', { dash: '5 4' }),
    note(137, 224, '半加器'),
    box(232, 28, 114, 132, '', { dash: '5 4' }),
    note(289, 178, '半加器'),
    label(32, 51, 'a', '', { anchor: 'end' }),
    label(32, 67, 'b', '', { anchor: 'end' }),
    wire([[40, 46], [130, 46]]),
    wire([[100, 46], [100, 164], [130, 164]]),
    dot(100, 46),
    wire([[40, 62], [130, 62]]),
    wire([[115, 62], [115, 180], [130, 180]]),
    dot(115, 62),
    label(260, 14, 'c'),
    wire([[260, 20], [260, 140], [290, 140]]),
    wire([[260, 70], [290, 70]]),
    dot(260, 70),
    wire([[174, 54], [290, 54]]),
    wire([[245, 54], [245, 124], [290, 124]]),
    dot(245, 54),
    label(205, 48, 'p'),
    wire([[174, 172], [380, 172], [380, 148], [400, 148]]),
    label(205, 166, 'g'),
    wire([[334, 132], [400, 132]]),
    label(366, 126, 't'),
    wire([[334, 62], [490, 62]]),
    label(500, 67, 's', '', { anchor: 'start' }),
    wire([[444, 140], [490, 140]]),
    label(500, 145, 'c′', '', { anchor: 'start' }),
    gate(130, 32, XOR),
    gate(130, 150, AND),
    gate(290, 40, XOR),
    gate(290, 110, AND),
    gate(400, 118, OR),
  ]);
}

// 图 1.4: 4-bit ripple-carry adder, least significant bit on the right.
function rippleCarry() {
  const bx = (i) => 70 + (3 - i) * 120;
  const parts = [0, 1, 2, 3].map((i) => box(bx(i), 70, 72, 56, '全加器', { size: 13 }));
  for (let i = 0; i < 4; i++) {
    const x = bx(i);
    parts.push(wire([[x + 22, 34], [x + 22, 70]], { arrow: true }));
    parts.push(wire([[x + 50, 34], [x + 50, 70]], { arrow: true }));
    parts.push(label(x + 22, 26, 'a', String(i)));
    parts.push(label(x + 50, 26, 'b', String(i)));
    parts.push(wire([[x + 36, 126], [x + 36, 158]], { arrow: true }));
    parts.push(label(x + 36, 178, 's', String(i)));
    // Carry out of bit i goes left into bit i+1.
    const to = i < 3 ? bx(i + 1) + 72 : 24;
    parts.push(wire([[x, 98], [to, 98]], { color: CARRY, width: 3, arrow: true }));
    parts.push(label(i < 3 ? (x + to) / 2 : 30, 90, 'c', String(i + 1), { color: CARRY }));
  }
  parts.push(wire([[548, 98], [bx(0) + 72, 98]], { color: CARRY, width: 3, arrow: true }));
  parts.push(label(536, 90, 'c', '0', { color: CARRY }));
  parts.push(note(536, 118, '= 0', { size: 13, color: CARRY }));
  return svg(580, 192, parts);
}

// 图 1.5: accumulator, register + adder in a loop.
function accumulator() {
  return svg(460, 250, [
    box(200, 40, 120, 50, '加法器'),
    box(200, 140, 120, 44, '寄存器 <tspan font-style="italic">s</tspan>'),
    label(62, 60, 'x', '', { anchor: 'end' }),
    wire([[70, 55], [200, 55]], { arrow: true }),
    wire([[320, 65], [370, 65], [370, 162], [322, 162]], { arrow: true }),
    `<text x="380" y="118" font-size="14" fill="${INK}"><tspan font-style="italic">s</tspan> + <tspan font-style="italic">x</tspan></text>`,
    wire([[200, 162], [150, 162], [150, 75], [198, 75]], { arrow: true }),
    wire([[150, 162], [70, 162]], { arrow: true }),
    dot(150, 162),
    label(60, 167, 's', '', { anchor: 'end' }),
    wire([[260, 226], [260, 186]], { arrow: true }),
    note(260, 243, '时钟', { size: 13, color: INK }),
    `<path d="M252,184 L268,184 L260,174 z" fill="none" stroke="${GATE_STROKE}" stroke-width="1.4"/>`,
  ]);
}

// 图 1.6–1.8: prefix networks on n = 8 inputs. levels[L] lists [i, j]: column i combines with column j < i.
function prefixNetwork(levels) {
  const n = 8;
  const x = (i) => 96 + 52 * i;
  const top = 34;
  const gap = 56;
  const yl = (L) => top + gap * (L + 1);
  const bottom = yl(levels.length - 1) + 30;
  const parts = [];
  for (let i = 0; i < n; i++) {
    parts.push(label(x(i), 24, 'x', String(i)));
    parts.push(wire([[x(i), top], [x(i), bottom]]));
    parts.push(label(x(i), bottom + 20, 'y', String(i)));
  }
  const nodes = [];
  levels.forEach((pairs, L) => {
    parts.push(note(12, yl(L) + 4, `第 ${L + 1} 级`, { anchor: 'start' }));
    for (const [i, j] of pairs) {
      parts.push(wire([[x(j), yl(L) - 36], [x(i), yl(L)]]));
      nodes.push(dot(x(j), yl(L) - 36, 2.6));
      nodes.push(dot(x(i), yl(L), 6.5, NODE));
    }
  });
  return svg(x(n - 1) + 40, bottom + 34, [...parts, ...nodes]);
}

const sklansky = () => prefixNetwork([
  [[1, 0], [3, 2], [5, 4], [7, 6]],
  [[2, 1], [3, 1], [6, 5], [7, 5]],
  [[4, 3], [5, 3], [6, 3], [7, 3]],
]);

const koggeStone = () => prefixNetwork([
  [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6]],
  [[2, 0], [3, 1], [4, 2], [5, 3], [6, 4], [7, 5]],
  [[4, 0], [5, 1], [6, 2], [7, 3]],
]);

const brentKung = () => prefixNetwork([
  [[1, 0], [3, 2], [5, 4], [7, 6]],
  [[3, 1], [7, 5]],
  [[7, 3], [5, 3]],
  [[2, 1], [4, 3], [6, 5]],
]);

const FIGURES = {
  '01-mux.svg': mux,
  '01-and-tree.svg': andTree,
  '01-full-adder.svg': fullAdder,
  '01-ripple-carry.svg': rippleCarry,
  '01-accumulator.svg': accumulator,
  '01-prefix-sklansky.svg': sklansky,
  '01-prefix-kogge-stone.svg': koggeStone,
  '01-prefix-brent-kung.svg': brentKung,
};

await mkdir(OUT, { recursive: true });
for (const [name, make] of Object.entries(FIGURES)) {
  await writeFile(join(OUT, name), make());
  console.log(`wrote book/fig/${name}`);
}
