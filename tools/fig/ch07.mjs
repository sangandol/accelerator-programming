// 第 7 章的插图。
const RANK = ['blue', 'orange', 'green', 'purple'];
const LET = ['A', 'B', 'C', 'D'];

export default {
  // 图 7.1 三种拓扑；环上的二分
  '07-topologies'(f) {
    f.ring(110, 110, { n: 8, r: 72, nodeR: 14, start: -67.5 });
    f.line([[110, 22], [110, 198]], { color: 'red', dash: '5 4', width: 2 });
    f.note(110, 222, '(a) 环：一刀切成两半只断 2 条链路', { color: '#222', size: 12 });
    f.mesh(270, 50, { rows: 4, cols: 4, d: 40, nodeR: 11, wrap: true, label: () => '' });
    f.note(330, 222, '(b) 4×4 二维环面（虚线回绕）', { color: '#222', size: 12 });
    const sw = f.box(500, 92, 80, 36, '交换机', { color: 'gray', size: 12 });
    for (let i = 0; i < 8; i++) {
      const t = (Math.PI * 2 * i) / 8;
      const n = f.node(540 + 82 * Math.cos(t), 110 + 82 * Math.sin(t), '', { r: 12 });
      f.link(n, sw, { color: 'gray' });
    }
    f.note(540, 222, '(c) 交换机：任意两颗芯片全带宽', { color: '#222', size: 12 });
  },

  // 图 7.2 四种集合操作前后的数据（4 个 rank，每份数据 4 块）
  '07-collectives'(f) {
    const panel = (x0, y0, title, before, after) => {
      f.text(x0 + 135, y0, title, { size: 14, weight: 'bold' });
      const grid = (x, spec) => f.grid(x, y0 + 36, {
        rows: 4, cols: 4, cw: 30, ch: 24, size: 12,
        fill: (b, r) => spec(b, r)?.[1] ?? null,
        label: (b, r) => spec(b, r)?.[0] ?? '',
        colLabels: (r) => `${r}`,
      });
      const g1 = grid(x0, before);
      f.arrow([x0 + 132, g1.cy], [x0 + 166, g1.cy]);
      grid(x0 + 172, after);
      f.note(x0 - 6, y0 + 26, 'rank', { size: 11, anchor: 'end' });
    };
    const own = (b, r) => [`$${LET[r]}_${b}$`, RANK[r]];
    const sum = (b) => [`$\\Sigma_${b}$`, 'gray'];
    panel(30, 20, 'all-gather', (b, r) => (b === r ? own(b, r) : null), (b) => own(b, b));
    panel(360, 20, 'reduce-scatter', own, (b, r) => (b === r ? sum(b) : null));
    panel(30, 190, 'all-reduce', own, sum);
    panel(360, 190, 'all-to-all', own, (b, r) => own(r, b));
    f.note(330, 350, '列是 rank，行是块；$\\Sigma_b = A_b + B_b + C_b + D_b$', { color: '#222', size: 12 });
  },

  // 图 7.3 环形 all-gather：每步把上一步收到的块转发给右邻居
  '07-ring-allgather'(f) {
    const rows = ['开始', '第 1 步后', '第 2 步后', '第 3 步后'];
    rows.forEach((name, t) => {
      const y = 30 + t * 62;
      f.text(70, y + 11, name, { anchor: 'end', size: 12 });
      for (let r = 0; r < 4; r++) {
        const has = new Set(Array.from({ length: t + 1 }, (_, s) => (r - s + 4) % 4));
        const x = 90 + r * 130;
        const g = f.grid(x, y, { rows: 1, cols: 4, cw: 22, ch: 22, size: 11, fill: (_, b) => (has.has(b) ? RANK[b] : null), label: (_, b) => (has.has(b) ? `$${LET[b]}$` : '') });
        if (t === 0) f.note(g.cx, y - 14, `rank ${r}`);
        if (t < 3) {
          const sent = (r - t + 4) % 4;
          f.frame(g.cell(0, sent), { color: 'red', width: 2 });
          if (r < 3) f.arrow([x + 92, y + 11], [x + 126, y + 11], { color: 'red', width: 1.4 });
        }
      }
    });
    f.note(330, 268, '红框：这一步发给右邻居的块（rank 3 发给 rank 0）', { color: '#222', size: 12 });
  },

  // 图 7.4 递归倍增：第 t 步与距离 2^t 的 rank 交换全部数据
  '07-recursive-doubling'(f) {
    const ops = [0, 1, 2].flatMap((L) => Array.from({ length: 8 }, (_, i) => [L, i, i ^ (1 << L)]));
    f.colnet(110, 24, { n: 8, ops, top: (i) => `${i}`, bottom: (i) => `${i}`, levelLabel: (L) => `第 ${L + 1} 步：距离 ${1 << L}`, color: 'orange' });
  },

  // 图 7.5 二维环面上的 all-reduce：先沿行、再沿列 reduce-scatter
  '07-torus-2d'(f) {
    const ringArrows = (M, k, rowWise, color) => {
      for (let a = 0; a < k; a++)
        for (let b = 0; b + 1 < k; b++) {
          const [p, q] = rowWise ? [M.node(a, b), M.node(a, b + 1)] : [M.node(b, a), M.node(b + 1, a)];
          f.link(p, q, { arrow: 'end', color, width: 2.2 });
        }
    };
    const A = f.mesh(60, 50, { rows: 3, cols: 3, d: 64, label: () => '', wrap: true });
    ringArrows(A, 3, true, 'red');
    f.note(124, 220, '(a) 每一行是一个环：对 $n$ 字节做 reduce-scatter', { color: '#222', size: 12 });
    const B = f.mesh(380, 50, { rows: 3, cols: 3, d: 64, label: () => '', wrap: true });
    ringArrows(B, 3, false, 'blue');
    f.note(444, 220, '(b) 每一列是一个环：只剩 $n/k$ 字节', { color: '#222', size: 12 });
  },

  // 图 7.6 命题 7.10：环形 all-gather 与矩阵乘法重叠（t_c = 2，t_m = 3）
  '07-overlap'(f) {
    f.timeline(70, 20, {
      lanes: ['通信', '计算'], unit: 36, ticks: 3,
      bars: [
        [0, 0, 2, '收第 1 块', 'orange'], [0, 2, 4, '收第 2 块', 'orange'], [0, 4, 6, '收第 3 块', 'orange'],
        [1, 0, 3, '算第 0 块'], [1, 3, 6, '算第 1 块'], [1, 6, 9, '算第 2 块'], [1, 9, 12, '算第 3 块'],
      ],
      deps: [[0, 4], [1, 5], [2, 6]],
    });
  },

  // 图 7.7 跨芯片传输的双向约定
  '07-credit'(f) {
    const s = f.box(30, 40, 120, 70, '发送方', { color: 'blue' });
    const r = f.box(400, 40, 150, 70, '接收方的缓冲', { color: 'green' });
    f.arrow(s.R(0.3), r.L(0.3), { color: 'blue' });
    f.text(275, 46, '① 写入数据，并给对方的完成计数器加数', { size: 12, color: 'blue' });
    f.arrow(r.L(0.75), s.R(0.75), { color: 'orange' });
    f.text(275, 106, '② 用完缓冲后，回送"信用"：可以再写了', { size: 12, color: 'orange' });
  },

  // 图：例 7.x 中 all-reduce 的时间随数据量变化（64 个 rank，α = 5 µs，β = 100 GB/s）
  '07-crossover'(f) {
    const a = 5e-6;
    const beta = 1e11;
    const ns = Array.from({ length: 31 }, (_, k) => 10 ** (3 + k / 5));
    const band = (n) => (2 * 63 / 64) * n / beta;
    const ring = (n) => (126 * a + band(n)) * 1e6;
    const dbl = (n) => (12 * a + band(n)) * 1e6;
    f.plot(80, 30, {
      w: 400, h: 230,
      x: { min: 1e3, max: 1e9, log: true, label: '数据量 $n$（字节）', ticks: [1e3, 1e5, 1e7, 1e9] },
      y: { min: 10, max: 1e5, log: true, label: '时间（µs）', ticks: [10, 100, 1e3, 1e4, 1e5] },
      series: [
        { pts: ns.map((n) => [n, ring(n)]), color: 'red', label: '环形', at: 5, dx: 6, dy: -12 },
        { pts: ns.map((n) => [n, dbl(n)]), color: 'blue', label: '倍增', at: 20, dx: -10, dy: 16 },
      ],
    });
  },

  // 图：缓冲被覆盖与信用（上：错误；下：正确）
  '07-overwrite'(f) {
    f.note(40, 8, '(a) 发送方不等待：块 $D$ 覆盖了还在使用的块 $A$', { anchor: 'start', color: '#222', size: 13 });
    f.timeline(80, 26, {
      lanes: ['发送方', '接收方的缓冲', '接收方'], unit: 40, lh: 24, gap: 6, ticks: 1, axisLabel: '时间',
      bars: [[0, 0, 1, '写 $A$', 'orange'], [0, 1, 2, '写 $D$', 'red'], [1, 1, 2, '$A$', 'blue'], [1, 2, 3, '$D$', 'red'], [2, 1, 3, '用 $A$ 计算', 'green']],
    });
    f.note(40, 188, '(b) 接收方用完后回送信用，发送方才写入', { anchor: 'start', color: '#222', size: 13 });
    f.timeline(80, 206, {
      lanes: ['发送方', '接收方的缓冲', '接收方'], unit: 40, lh: 24, gap: 6, ticks: 1, axisLabel: '时间',
      bars: [[0, 0, 1, '写 $A$', 'orange'], [0, 3, 4, '写 $D$', 'orange'], [1, 1, 3, '$A$', 'blue'], [1, 4, 5, '$D$', 'blue'], [2, 1, 3, '用 $A$ 计算', 'green'], [2, 3, 4, '信用', 'yellow']],
      deps: [[4, 5]],
    });
  },
};
