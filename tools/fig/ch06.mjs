// 第 6 章的插图。
import { oblique } from './lib.mjs';

export default {
  // 图 6.1 抽象芯片
  '06-chip'(f) {
    f.box(20, 20, 520, 196, '', { color: 'gray', hollow: true });
    f.note(40, 36, '芯片', { anchor: 'start', color: '#222', size: 13 });
    const srams = [0, 1].map((c) => {
      const x = 40 + c * 250;
      f.box(x, 50, 230, 150, '', { color: 'blue', hollow: true, dash: '5 4' });
      f.note(x + 115, 64, `核心 ${c}`, { color: '#222' });
      f.box(x + 10, 78, 64, 40, '标量控制', { color: 'orange', size: 11 });
      f.box(x + 83, 78, 64, 40, '矩阵单元', { color: 'blue', size: 11 });
      f.box(x + 156, 78, 64, 40, '向量单元', { color: 'green', size: 11 });
      return f.box(x + 10, 130, 210, 56, '片上存储（容量 $C_S$）', { color: 'yellow', size: 12 });
    });
    const hbm = f.box(150, 260, 260, 40, 'HBM（容量 $C_H$）', { color: 'gray' });
    srams.forEach((s) => f.link(hbm, s, { arrow: 'both' }));
    f.note(420, 280, '共享带宽 $B$；单次传输 $\\alpha + S/\\beta$', { color: '#222', size: 12, anchor: 'start' });
  },

  // 图 6.2 参考芯片 X 的屋顶线
  '06-roofline'(f) {
    const ridge = 262;
    const { sx, sy } = f.plot(70, 30, {
      w: 420, h: 250,
      x: { min: 0.1, max: 10000, log: true, label: '算术强度 $I$（FLOP/字节）', ticks: [0.1, 1, 10, 100, 1000, 10000] },
      y: { min: 0.1, max: 1000, log: true, label: '可达算力（TFLOP/s）', ticks: [0.1, 1, 10, 100, 1000] },
      series: [{ pts: [[0.1, 0.1], [ridge, ridge], [10000, ridge]], color: 'blue', width: 2.6 }],
      marks: [
        { at: [0.25, 0.25], label: '逐元素运算', dx: 8, dy: 10 },
        { at: [1, 1], label: '矩阵乘向量', dx: 8, dy: 10 },
        { at: [64, 64], label: '批量 64 的矩阵乘法', dx: -118, dy: -14 },
        { at: [2000, ridge], label: '大矩阵乘法', dx: -30, dy: 16 },
      ],
    });
    f.line([[sx(ridge), sy(ridge)], [sx(ridge), sy(0.1)]], { color: 'gray', dash: '4 4' });
    f.text(sx(ridge) + 4, sy(0.1) - 12, '拐点 $I^* = P/B = 262$', { anchor: 'start', size: 12 });
    f.text(sx(1500), sy(ridge) - 14, '$P = 262$ TFLOP/s', { size: 12, color: 'blue' });
    f.text(sx(3), sy(12), '斜率 = 带宽 $B$', { size: 12, color: 'blue' });
    f.note(sx(0.8), sy(400), '访存受限', { size: 13 });
    f.note(sx(1500), sy(30), '计算受限', { size: 13 });
  },

  // 图 6.3 矩阵乘法的分块
  '06-tiling'(f) {
    const c = 16;
    const LO = { o: '#fde3c0', g: '#d6eed6' };
    const HI = { o: '#f0a64a', g: '#7cc47c' };
    const A = f.grid(40, 260, { rows: 8, cols: 12, cw: c, ch: c, stroke: '#ccc', fill: (r, k) => (r >= 2 && r <= 3 ? (k >= 4 && k <= 5 ? HI.o : LO.o) : null) });
    const B = f.grid(262, 38, { rows: 12, cols: 10, cw: c, ch: c, stroke: '#ccc', fill: (k, j) => (j >= 6 && j <= 7 ? (k >= 4 && k <= 5 ? HI.g : LO.g) : null) });
    const C = f.grid(262, 260, { rows: 8, cols: 10, cw: c, ch: c, stroke: '#ccc', fill: (r, j) => (r >= 2 && r <= 3 && j >= 6 && j <= 7 ? 'blue' : null) });
    f.text(A.cx, A.y + A.h + 16, '$A$：$M \\times K$');
    f.text(B.x - 10, B.cy, '$B$：$K \\times N$', { anchor: 'end' });
    f.text(C.cx, C.y + C.h + 16, '$C$：$M \\times N$');
    f.frame(C.region(2, 6, 3, 7), { color: 'blue', width: 2 });
    f.text(C.x + C.w + 10, C.region(2, 6, 3, 7).cy, '$b_M \\times b_N$ 的 $C$ 块：累加器留在片上', { anchor: 'start', size: 12, color: 'blue' });
    f.frame(A.region(2, 4, 3, 5), { color: '#b45f06', width: 2 });
    f.frame(B.region(4, 6, 5, 7), { color: '#2e7d32', width: 2 });
    f.text(A.x, A.y - 14, '深色：$K$ 循环的一步读入的 $b_M \\times b_K$ 与 $b_K \\times b_N$ 块', { anchor: 'start', size: 12 });
  },

  // 图 6.4 Loomis–Whitney：一段时间内执行的乘加 V 与它在三个坐标平面上的投影
  '06-loomis-whitney'(f) {
    const p = oblique(80, 60, [0, 34], [40, 0], [24, -18]);
    const S = 6;
    const face = (pts, color) => f.poly(pts.map((q) => p(...q)), { color, width: 1 });
    face([[0, 0, S], [0, S, S], [S, S, S], [S, 0, S]], '#f6f6f6');
    face([[0, 0, 0], [0, 0, S], [S, 0, S], [S, 0, 0]], '#f0f0f0');
    face([[S, 0, 0], [S, 0, S], [S, S, S], [S, S, 0]], '#e9e9e9');
    face([[2, 2, S], [2, 4, S], [4, 4, S], [4, 2, S]], 'blue');
    face([[2, 0, 2], [2, 0, 4], [4, 0, 4], [4, 0, 2]], 'orange');
    face([[S, 2, 2], [S, 2, 4], [S, 4, 4], [S, 4, 2]], 'green');
    for (const [a, b] of [[[3, 3, 3], [3, 3, S]], [[3, 3, 3], [3, 0, 3]], [[3, 3, 3], [S, 3, 3]]]) f.line([p(...a), p(...b)], { color: 'gray', dash: '3 3', width: 1.2 });
    face([[2, 2, 2], [2, 4, 2], [4, 4, 2], [4, 2, 2]], 'purple');
    face([[2, 2, 2], [2, 2, 4], [2, 4, 4], [2, 4, 2]], 'purple');
    face([[2, 4, 2], [2, 4, 4], [4, 4, 4], [4, 4, 2]], 'purple');
    f.text(...p(3, 3, 2), '$V$', { size: 15 });
    f.text(p(0, S, S)[0] + 10, p(1, S, S)[1], '$V_{ij}$：更新的 $C$ 元素', { anchor: 'start', size: 12, color: 'blue' });
    f.text(p(3, 0, 0)[0] - 10, p(3, 0, 1)[1], '$V_{ik}$：用到的 $A$ 元素', { anchor: 'end', size: 12, color: 'orange' });
    f.text(p(S, 3, 0)[0], p(S, 3, 0)[1] + 18, '$V_{kj}$：用到的 $B$ 元素', { size: 12, color: 'green' });
  },

  // 图 6.5 波次量化：8 个块分给 6 个核心
  '06-waves'(f) {
    const bars = [];
    for (let c = 0; c < 6; c++) bars.push([c, 0, 1, `块 ${c}`]);
    bars.push([0, 1, 2, '块 6'], [1, 1, 2, '块 7']);
    for (let c = 2; c < 6; c++) bars.push([c, 1, 2, '空闲', 'gray']);
    f.timeline(80, 20, { lanes: [0, 1, 2, 3, 4, 5].map((c) => `核心 ${c}`), bars, unit: 90, lh: 22, gap: 5, ticks: 1, tickLabel: (t) => (t ? `第 ${t} 轮结束` : '0'), axisLabel: '' });
  },

  // 图 6.6 沿 K 切分（split-K）
  '06-split-k'(f) {
    const cs = ['blue', 'orange', 'green', 'purple'];
    for (let r = 0; r < 4; r++) {
      f.box(40 + r * 50, 60, 50, 40, `$A_{${r}}$`, { color: cs[r], rx: 0, size: 12 });
      f.box(270, 10 + r * 30, 40, 30, `$B_{${r}}$`, { color: cs[r], rx: 0, size: 12 });
      const c = f.box(360 + r * 70, 60, 46, 40, `$C^{(${r})}$`, { color: cs[r], size: 12 });
      f.note(c.cx, 116, `核心 ${r}`);
      if (r < 3) f.text(c.x + 58, c.cy, '+', { size: 16 });
    }
    f.text(40 + 100, 116, '$A$：沿 $K$ 切成 4 段', { size: 12 });
    f.text(330, 140, '', {});
    f.arrow([360 + 3 * 70 + 52, 80], [360 + 3 * 70 + 82, 80]);
    f.box(360 + 3 * 70 + 88, 60, 46, 40, '$C$', { color: 'gray' });
    f.note(400 + 2 * 70, 150, '每个核心算一个 f32 部分和，最后再做一次归约', { color: '#222', size: 12 });
  },

  // 图：y = tanh(2x + 1) 不融合与融合时的 HBM 读写
  '06-fusion'(f) {
    const panel = (y0, programs, title, count) => {
      f.note(40, y0, title, { anchor: 'start', color: '#222', size: 13 });
      const hbm = f.box(40, y0 + 92, 560, 26, 'HBM', { color: 'gray', size: 12 });
      programs.forEach(([x, w, label]) => {
        const b = f.box(x, y0 + 20, w, 36, label, { color: 'orange', size: 12 });
        f.arrow([x + 24, hbm.y], [x + 24, b.y + b.h], { color: 'blue' });
        f.arrow([x + w - 24, b.y + b.h], [x + w - 24, hbm.y], { color: 'green' });
        f.note(x + 34, y0 + 74, '读', { anchor: 'start', color: 'blue' });
        f.note(x + w - 34, y0 + 74, '写', { anchor: 'end', color: 'green' });
      });
      f.note(610, y0 + 105, count, { anchor: 'start' });
    };
    panel(10, [[50, 150, '程序 1：$t_1 = 2x$'], [240, 150, '程序 2：$t_2 = t_1 + 1$'], [430, 150, '程序 3：$y$ = tanh $t_2$']], '不融合：三个程序', 'HBM 读写 6 次');
    panel(160, [[160, 330, '一个程序：$2x$、$+1$、tanh，中间结果留在片上']], '融合：一个程序', 'HBM 读写 2 次');
  },

  // 图：8192³ 矩阵乘法的搬运时间随块边长的变化（芯片 X）
  '06-tile-sweep'(f) {
    const bs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
    const mem = (b) => (2.199e12 / b + 1.342e8) / 1e9;
    const { sx, sy } = f.plot(70, 30, {
      w: 400, h: 240,
      x: { min: 1, max: 4096, log: true, label: '块边长 $b$', ticks: [1, 4, 16, 64, 256, 1024, 4096] },
      y: { min: 0.5, max: 5000, log: true, label: '时间（ms）', ticks: [1, 10, 100, 1000] },
      series: [
        { pts: bs.map((b) => [b, mem(b)]), color: 'blue', label: '搬运时间', at: 3, dx: 8, dy: -6 },
        { pts: [[1, 4.2], [4096, 4.2]], color: 'orange', dash: '6 4', label: '计算时间 4.2 ms', at: 0, dx: 6, dy: 14 },
      ],
      marks: [{ at: [540, 4.2], label: '$b \\approx 540$', dx: 10, dy: -14 }],
    });
    f.note(sx(16), sy(0.8), '← 访存受限', { anchor: 'start' });
    f.note(sx(640), sy(0.8), '计算受限 →', { anchor: 'start' });
  },
};
