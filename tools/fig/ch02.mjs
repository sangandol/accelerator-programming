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
};
