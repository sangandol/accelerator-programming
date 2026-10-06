// 第 2 章的插图。每个函数接收一个新的 Fig f 并在上面作图（见 README.md）。
export default {
  // 图 2.1 乘法器：竖式中的部分积，以及乘法器的结构
  '02-multiplier'(f) {
    const X = (pos) => 250 - pos * 26;
    const Y = (r) => 20 + r * 28;
    const row = (r, bits, shift, label, fill, right) => {
      [...bits].forEach((ch, k) => {
        const pos = shift + bits.length - 1 - k;
        if (fill) f.rect(X(pos) - 12, Y(r) - 12, 24, 24, { fill, stroke: 'none', rx: 3 });
        f.text(X(pos), Y(r), ch, { size: 15 });
      });
      f.text(X(7) - 30, Y(r), label, { anchor: 'end' });
      if (right) f.note(X(0) + 26, Y(r), right, { anchor: 'start', size: 13 });
    };
    row(0, '1011', 0, '$a$', null, '= 11');
    row(1, '1101', 0, '$b$', null, '= 13');
    f.line([[X(7) - 14, Y(1) + 15], [X(0) + 14, Y(1) + 15]]);
    const b = [1, 0, 1, 1];
    b.forEach((bj, j) => row(2 + j, bj ? '1011' : '0000', j, `$a \\cdot b_${j}$`, bj ? 'blue' : 'gray'));
    f.line([[X(7) - 14, Y(5) + 15], [X(0) + 14, Y(5) + 15]]);
    row(6, '10001111', 0, '$a \\cdot b$', null, '= 143');

    const steps = ['与门阵列：$p$ 个部分积', '3:2 压缩树：$O(\\log p)$ 级', '超前进位加法器', '乘积（$2p$ 位）'];
    const boxes = steps.map((s, i) => f.box(390, 14 + i * 50, 190, 32, s, { size: 13, color: i === 3 ? 'gray' : 'blue' }));
    for (let i = 1; i < 4; i++) f.arrow(boxes[i - 1].B(), boxes[i].T());
  },

  // 图 2.2 一个玩具浮点系统 F(3, -1, 2) 的全部非负数
  '02-float-line'(f) {
    const y = 80;
    const normal = [-1, 0, 1, 2].flatMap((e) => [4, 5, 6, 7].map((m) => m * 2 ** (e - 2)));
    const sx = f.numline(40, y, {
      min: 0, max: 8, w: 560, ticks: normal,
      labels: (v) => ([0.5, 1, 1.5, 2, 3, 4, 5, 6, 7].includes(v) ? String(v) : ''),
    });
    for (const v of [0, 0.125, 0.25, 0.375]) f.line([[sx(v), y - 6], [sx(v), y + 6]], { color: 'red' });
    f.text(sx(0), y + 18, '0');
    [[0, 0.5, '1/8'], [0.5, 1, '1/8'], [1, 2, '1/4'], [2, 4, '1/2'], [4, 8, '1']].forEach(([a, b, s]) =>
      f.brace(sx(a) + 1, y - 14, sx(b) - 1, s, { flip: true, color: a === 0 ? 'red' : undefined }));
    f.note(sx(0) - 8, y - 33, '间距', { anchor: 'end' });
    f.note(sx(0.25), y + 40, '次正规数', { color: 'red' });
  },

  // 图 2.3 常用格式的位分配
  '02-formats'(f) {
    const u = 11;
    const rows = [['f32', 8, 23], ['tf32', 8, 10], ['bf16', 8, 7], ['fp16', 5, 10], ['fp8 E4M3', 4, 3], ['fp8 E5M2', 5, 2], ['fp4 E2M1', 2, 1]];
    rows.forEach(([name, e, m], r) => {
      const y = 50 + r * 36;
      f.text(92, y + 11, name, { anchor: 'end', size: 13 });
      f.bitfield(100, y, { fields: [[1, '', 'red'], [e, String(e), 'green'], [m, m > 1 ? String(m) : '', 'blue']], unit: u, h: 22, size: 11, index: false });
    });
    [['符号', 'red'], ['指数', 'green'], ['尾数', 'blue']].forEach(([s, c], i) => {
      f.box(100 + i * 80, 8, 14, 14, '', { color: c, rx: 2 });
      f.text(120 + i * 80, 15, s, { size: 12, anchor: 'start' });
    });
  },

  // 图 2.4 块缩放：每 b 个元素共用一个缩放因子
  '02-block-scaling'(f) {
    const g = f.grid(40, 70, {
      rows: 1, cols: 16, cw: 30, ch: 30,
      fill: (r, c) => (c < 8 ? 'blue' : 'orange'),
      label: (r, c) => `$q_{${c}}$`, size: 11,
    });
    [0, 1].forEach((j) => {
      const blk = g.region(0, 8 * j, 0, 8 * j + 7);
      const s = f.box(blk.cx - 22, 14, 44, 28, `$s_${j}$`, { color: j ? 'orange' : 'blue' });
      f.arrow(s.B(), [blk.cx, blk.y]);
      f.brace(blk.x + 2, blk.y + blk.h + 6, blk.x + blk.w - 2, `第 ${j} 块：$b = 8$ 个元素`);
    });
    f.text(g.x + g.w + 16, g.y + 15, '元素的值 = $s_j \\cdot q_i$', { anchor: 'start' });
  },

  // 图 2.5 例 2.13：bf16 累加器的停滞
  '02-bf16-stagnation'(f) {
    const t = [0, 1024, 2048, 3072, 4096];
    f.plot(70, 30, {
      w: 360, h: 220,
      x: { min: 0, max: 4096, ticks: t, label: '加的次数' },
      y: { min: 0, max: 4096, ticks: t, label: '累加结果' },
      series: [
        { pts: [[0, 0], [4096, 4096]], color: 'blue', label: 'f32 累加器', at: 1, dx: -96, dy: -4 },
        { pts: [[0, 0], [256, 256], [4096, 256]], color: 'red', label: 'bf16 累加器', at: 2, dx: -84, dy: -14 },
      ],
      marks: [{ at: [256, 256], label: '停在 256', color: 'red', dx: 150, dy: -14 }],
    });
  },

  // 图 2.6 命题 2.14：f32 的 24 位尾数拆成三个 bf16 的 8 位尾数
  '02-f32-split'(f) {
    const colors = ['blue', 'green', 'orange'];
    const cw = 16;
    f.grid(80, 20, { rows: 1, cols: 24, cw, ch: 24, fill: (r, c) => colors[Math.floor(c / 8)] });
    f.text(72, 32, '$x$', { anchor: 'end' });
    f.note(80 + 12 * cw, 58, 'f32 的尾数：24 位', { size: 12 });
    [0, 1, 2].forEach((i) => {
      const y = 82 + i * 36;
      f.grid(80 + 8 * i * cw, y, { rows: 1, cols: 8, cw, ch: 24, fill: () => colors[i] });
      f.text(72, y + 12, `$x_${i + 1}$`, { anchor: 'end' });
      f.note(80 + 8 * (i + 1) * cw + 10, y + 12, `bf16，约 $2^{-${8 * i}} \\lvert x \\rvert$`, { anchor: 'start', size: 12 });
    });
  },

  // 图 2.7 随机舍入
  '02-stochastic-rounding'(f) {
    const y = 90;
    const sx = f.numline(40, y, { min: 0, max: 1, w: 360, ticks: [0, 1], labels: { 0: '$a$', 1: '$b$' }, arrow: false });
    const x = [sx(0.3), y];
    f.dot(...x, 4, 'red');
    f.text(x[0], y + 18, '$x$', { color: 'red' });
    f.link(x, [sx(0), y], { bend: -0.45, arrow: 'end', color: 'blue' });
    f.link(x, [sx(1), y], { bend: 0.3, arrow: 'end', color: 'blue' });
    f.text(sx(0.15), y - 50, '概率 $(b - x)/(b - a)$', { size: 12, color: 'blue' });
    f.text(sx(0.68), y - 70, '概率 $(x - a)/(b - a)$', { size: 12, color: 'blue' });
  },

  // 图：11 × 13 的比特点阵，两级 3:2 压缩之后只剩两行
  '02-dots'(f) {
    const X = (col) => 130 + (7 - col) * 24;
    const row = (y, bits, low, label, color, value) => {
      [...bits].forEach((ch, k) => f.box(X(low + bits.length - 1 - k) - 10, y - 10, 20, 20, ch, { color, size: 12, rx: 2 }));
      f.text(X(7) - 24, y, label, { anchor: 'end', size: 13 });
      if (value) f.note(X(0) + 22, y, value, { anchor: 'start' });
    };
    for (let c = 0; c < 8; c++) f.note(X(c), 8, String(c));
    f.note(X(7) - 24, 8, '位', { anchor: 'end' });
    const step = (y, s) => { f.arrow([X(3.5), y], [X(3.5), y + 26]); f.text(X(3.5) + 12, y + 13, s, { anchor: 'start', size: 12 }); };
    row(30, '1011', 0, '$q_0$', 'blue', '11');
    row(54, '0000', 1, '$q_1$', 'blue', '0');
    row(78, '1011', 2, '$q_2$', 'blue', '44');
    row(102, '1011', 3, '$q_3$', 'blue', '88');
    step(118, '第一级 3:2，压缩 $q_0, q_1, q_2$');
    row(162, '100111', 0, '$s$', 'green', '39');
    row(186, '001000', 1, '$2c$', 'orange', '16');
    row(210, '1011', 3, '$q_3$', 'blue', '88');
    step(226, '第二级 3:2');
    row(270, '1101111', 0, '$s$', 'green', '111');
    row(294, '0010000', 1, '$2c$', 'orange', '32');
    step(310, '超前进位加法');
    row(354, '10001111', 0, '积', 'gray', '143');
  },

  // 图：bf16 的 16 个比特（13.0）
  '02-bf16-decode'(f) {
    const bits = '0100000101010000';
    const field = (c) => (c === 0 ? 'red' : c <= 8 ? 'green' : 'blue');
    const g = f.grid(40, 60, { rows: 1, cols: 16, cw: 26, ch: 28, fill: (r, c) => field(c), label: (r, c) => bits[c], size: 13 });
    for (let c = 0; c < 16; c++) f.note(g.cell(0, c).cx, 46, String(15 - c), { size: 10 });
    f.brace(g.cell(0, 0).x + 2, 34, g.cell(0, 0).x + 24, '符号 0：正数', { flip: true });
    f.brace(g.cell(0, 1).x + 2, 96, g.cell(0, 8).x + 24, '指数字段 $10000010_2 = 130$，$e = 130 - 127 = 3$');
    f.brace(g.cell(0, 9).x + 2, 128, g.cell(0, 15).x + 24, '尾数字段，有效数字 $(1.1010000)_2 = 1.625$');
    f.text(g.cx, 170, '值 $= +1.625 \\times 2^3 = 13$', { size: 14 });
  },

  // 图：8 位右移器，k = 5
  '02-shifter'(f) {
    const X = (i) => 70 + (7 - i) * 52;
    const Y = [20, 96, 172, 248];
    const content = [
      [7, 6, 5, 4, 3, 2, 1, 0].map((i) => `$x_${i}$`),
      ['0', '$x_7$', '$x_6$', '$x_5$', '$x_4$', '$x_3$', '$x_2$', '$x_1$'],
      ['0', '$x_7$', '$x_6$', '$x_5$', '$x_4$', '$x_3$', '$x_2$', '$x_1$'],
      ['0', '0', '0', '0', '0', '$x_7$', '$x_6$', '$x_5$'],
    ];
    const k = [1, 0, 1];
    for (let r = 0; r < 3; r++)
      for (let i = 0; i < 8; i++) {
        const top = [X(i), Y[r + 1]];
        const straight = [[X(i), Y[r] + 24], top];
        const src = i + 2 ** r;
        const diag = src <= 7 ? [[X(src), Y[r] + 24], top] : null;
        const pick = k[r] ? diag : straight;
        const other = k[r] ? straight : diag;
        if (other) f.line(other, { color: '#c8c8c8', width: 1, dash: '3 3' });
        if (pick) f.line(pick, { color: 'blue', width: 1.4 });
      }
    const onPath = [7, 6, 6, 2];
    for (let r = 0; r < 3; r++) f.line([[X(onPath[r]), Y[r] + 24], [X(onPath[r + 1]), Y[r + 1]]], { color: 'red', width: 2.6 });
    content.forEach((labels, r) => labels.forEach((s, c) => f.box(X(7 - c) - 17, Y[r], 34, 24, s, { color: 7 - c === onPath[r] ? 'red' : r ? 'blue' : 'green', size: 12 })));
    ['第 0 级：$k_0 = 1$，右移 1 位', '第 1 级：$k_1 = 0$，不移', '第 2 级：$k_2 = 1$，右移 4 位'].forEach((s, r) =>
      f.text(X(0) + 30, (Y[r] + Y[r + 1] + 24) / 2, s, { anchor: 'start', size: 12 }));
  },

  // 图：浮点乘法与浮点加法的数据通路
  '02-fp-units'(f) {
    const col = (x0, title, steps) => {
      f.text(x0 + 100, 14, title, { size: 14, weight: 'bold' });
      const boxes = steps.map(([s, color, w = 200, dx = 0], i) => f.box(x0 + dx, 34 + i * 52, w, 32, s, { color, size: 12 }));
      return boxes;
    };
    const m = col(30, '浮点乘法', [['尾数相乘：$p \\times p$ 乘法器', 'orange', 168], ['规格化：至多右移 1 位', 'blue'], ['舍入', 'blue']]);
    const e = f.box(30 + 176, 34, 62, 32, '指数相加', { color: 'gray', size: 11 });
    f.arrow(m[0].B(), [m[0].cx, m[1].y]);
    f.arrow(e.B(), [e.cx, m[1].y]);
    f.arrow(m[1].B(), m[2].T());
    f.note(m[0].cx, m[2].y + 56, '面积 $\\Theta(p^2)$，几乎全在尾数乘法器');
    const a = col(350, '浮点加法', [['比较指数，求差 $d$', 'gray'], ['对阶：右移 $d$ 位', 'orange'], ['尾数相加', 'orange'], ['数前导零', 'blue'], ['规格化：左移', 'orange'], ['舍入', 'blue']]);
    for (let i = 1; i < a.length; i++) f.arrow(a[i - 1].B(), a[i].T());
    f.note(a[0].cx, a[5].y + 56, '面积 $O(p \\log p)$：两个移位器和一个加法器');
  },

  // 图：顺序求和与成对求和中，每个数经过几次舍入
  '02-sum-tree'(f) {
    const X = (ox, i) => ox + 38 * i;
    const inputs = (ox) => Array.from({ length: 8 }, (_, i) => (f.text(X(ox, i), 22, `$x_${i + 1}$`, { size: 15 }), [X(ox, i), 34]));
    const tops = inputs(0);
    let prev = tops[0];
    for (let k = 1; k < 8; k++) {
      const g = f.node(X(0, k), 34 + 36 * k, '+');
      f.link(tops[k], g);
      f.link(prev, g, { color: 'red', width: 2.4 });
      prev = g;
    }
    f.text(X(0, 3.5), 340, '顺序求和：$x_1$ 经过 7 次舍入', { size: 13 });
    let level = inputs(370);
    for (let d = 1; level.length > 1; d++) {
      const next = [];
      for (let j = 0; j < level.length; j += 2) {
        const [p, q] = [level[j], level[j + 1]];
        const g = f.node(((p.cx ?? p[0]) + (q.cx ?? q[0])) / 2, 34 + 60 * d, '+');
        f.link(p, g, { color: j === 0 ? 'red' : undefined, width: j === 0 ? 2.4 : undefined });
        f.link(q, g);
        next.push(g);
      }
      level = next;
    }
    f.text(X(370, 3.5), 340, '成对求和：每个数经过 3 次舍入', { size: 13 });
  },

  // 图：xy 的 9 个部分乘积 x_i y_j 的大小
  '02-split-products'(f) {
    f.grid(90, 50, {
      rows: 3, cols: 3, cw: 80, ch: 40, size: 13,
      fill: (r, c) => (r + c <= 1 ? 'orange' : r + c === 2 ? 'yellow' : null),
      label: (r, c) => (r + c === 0 ? '$\\approx 1$' : `$\\approx 2^{-${8 * (r + c)}}$`),
      rowLabels: ['$x_1$', '$x_2$', '$x_3$'], colLabels: ['$y_1$', '$y_2$', '$y_3$'],
    });
    f.note(210, 192, '每格是 $x_i y_j$ 相对于 $\\lvert xy \\rvert$ 的大小');
  },
};
