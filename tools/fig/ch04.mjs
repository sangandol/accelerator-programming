// 第 4 章的插图。
const OPS = ['blue', 'orange', 'green', 'purple', 'teal', 'red'];

export default {
  // 图 4.1 在组合电路中插入寄存器，切成 3 级
  '04-pipeline-cut'(f) {
    const reg = (x, y) => f.box(x, y, 14, 50, '', { color: 'gray', rx: 2 });
    const chain = (y, parts, label) => {
      let x = 40;
      reg(x, y);
      for (const [w, s] of parts) {
        f.arrow([x + 14, y + 25], [x + 30, y + 25]);
        f.box(x + 30, y, w, 50, s, { size: 13 });
        f.arrow([x + 30 + w, y + 25], [x + 46 + w, y + 25]);
        x += 46 + w;
        reg(x, y);
      }
      f.text(x + 30, y + 25, label, { anchor: 'start', size: 13 });
    };
    chain(30, [[330, '组合电路，深度 $d$']], '周期 $\\ge d\\tau$');
    chain(120, [[90, '深度 $d/3$'], [90, '深度 $d/3$'], [90, '深度 $d/3$']], '周期 $\\approx d\\tau/3$');
    f.note(47, 18, '寄存器');
  },

  // 图 4.2 4 级流水线：每个周期进入一个新操作
  '04-pipeline'(f) {
    const bars = [];
    for (let j = 0; j < 5; j++) for (let s = 0; s < 4; s++) bars.push([s, j + s, j + s + 1, `$o_${j + 1}$`, OPS[j]]);
    f.timeline(80, 20, { lanes: ['第 1 级', '第 2 级', '第 3 级', '第 4 级'], bars, unit: 40, axisLabel: '周期' });
  },

  // 图 4.3 互相独立与互相依赖的操作（L = 4）
  '04-dependent'(f) {
    const lanes = ['$o_1$', '$o_2$', '$o_3$', '$o_4$'];
    f.note(40, 8, '(a) 互相独立：每个周期发出一个', { anchor: 'start', color: '#222', size: 13 });
    f.timeline(70, 24, { lanes, bars: lanes.map((_, k) => [k, k, k + 4, '', OPS[k]]), unit: 22, lh: 22, gap: 6, end: 16, ticks: 4, axisLabel: '周期' });
    f.note(40, 170, '(b) 互相依赖：每个都等前一个的结果', { anchor: 'start', color: '#222', size: 13 });
    f.timeline(70, 186, {
      lanes, bars: lanes.map((_, k) => [k, 4 * k, 4 * k + 4, '', OPS[k]]), unit: 22, lh: 22, gap: 6, ticks: 4, axisLabel: '周期',
      deps: [[0, 1], [1, 2], [2, 3]],
    });
  },

  // 图 4.4 Little 定律：在途的数据量 = 带宽 × 延迟
  '04-little'(f) {
    const cpu = f.box(20, 40, 90, 60, '处理器');
    const mem = f.box(470, 40, 90, 60, 'HBM', { color: 'gray' });
    f.box(130, 52, 320, 36, '', { hollow: true, color: 'gray' });
    for (let i = 0; i < 12; i++) f.box(138 + i * 26, 60, 18, 20, '', { color: 'orange', rx: 2 });
    f.arrow([300, 30], [330, 30]);
    f.note(290, 30, '数据流动方向', { anchor: 'end' });
    f.brace(130, 98, 450, '延迟 $L$：一个请求从发出到返回的时间');
    f.text(290, 145, '管道中的数据量 = 带宽 $B$ × 延迟 $L$', { size: 14 });
    f.link(cpu, [130, 70]);
    f.link(mem, [450, 70]);
  },

  // 图 4.5 SIMD：一条指令驱动 8 个通道，掩码关掉一部分通道
  '04-simd'(f) {
    const mask = [1, 1, 0, 1, 1, 0, 0, 1];
    const ins = f.box(60, 10, 448, 30, '一条指令：$v_3 \\gets v_1 + v_2$（带掩码）', { size: 13, color: 'orange' });
    const g = f.grid(60, 80, {
      rows: 4, cols: 8, cw: 56, ch: 26, size: 12,
      fill: (r, c) => (r === 3 ? (mask[c] ? 'yellow' : 'gray') : r === 2 && !mask[c] ? 'gray' : r === 2 ? 'green' : null),
      label: (r, c) => (r === 3 ? mask[c] : r === 2 ? (mask[c] ? `$a_${c} + b_${c}$` : '不变') : r === 0 ? `$a_${c}$` : `$b_${c}$`),
      rowLabels: ['$v_1$', '$v_2$', '$v_3$', '掩码'],
    });
    for (let c = 0; c < 8; c++) f.arrow([g.cell(0, c).cx, ins.y + ins.h], [g.cell(0, c).cx, g.y], { color: 'orange', width: 1.2 });
    f.note(g.x + g.w + 10, g.y + 13, '通道 0–7', { anchor: 'start' });
  },

  // 图 4.6 8 个通道上的任意置换：交叉开关与 Beneš 网络
  '04-crossbar-benes'(f) {
    const n = 8;
    const d = 22;
    // (a) crossbar
    for (let i = 0; i < n; i++) {
      f.line([[30, 40 + i * d], [30 + n * d, 40 + i * d]], { color: 'gray' });
      f.line([[40 + i * d, 30], [40 + i * d, 30 + n * d]], { color: 'gray' });
      f.note(22, 40 + i * d, String(i), { anchor: 'end' });
      for (let j = 0; j < n; j++) f.dot(40 + j * d, 40 + i * d, 2.5, 'blue');
    }
    f.note(30 + (n * d) / 2, 30 + n * d + 20, '(a) 交叉开关：$8^2 = 64$ 个交叉点', { color: '#222', size: 13 });

    // (b) Beneš network: stages 0..4, ports 0..7; links unshuffle into halves, then shuffle back.
    const X = (s) => 290 + s * 64;
    const Y = (q) => 40 + q * d;
    const link = (s, q, size, inverse) => {
      const o = q - (q % size);
      const r = q % size;
      const half = size / 2;
      return inverse ? o + 2 * (r % half) + Math.floor(r / half) : o + Math.floor(r / 2) + (r % 2) * half;
    };
    const sizes = [8, 4, 4, 8];
    for (let s = 0; s < 4; s++)
      for (let q = 0; q < n; q++) f.line([[X(s) + 22, Y(q)], [X(s + 1), Y(link(s, q, sizes[s], s >= 2))]], { width: 1.2 });
    for (let q = 0; q < n; q++) {
      f.line([[X(0) - 14, Y(q)], [X(0), Y(q)]], { width: 1.2 });
      f.line([[X(4) + 22, Y(q)], [X(4) + 36, Y(q)]], { width: 1.2 });
      f.note(X(0) - 18, Y(q), String(q), { anchor: 'end' });
    }
    for (let s = 0; s < 5; s++) for (let i = 0; i < n / 2; i++) f.box(X(s), Y(2 * i) - 7, 22, d + 14, '', { color: 'blue', rx: 3 });
    f.note((X(0) + X(4) + 22) / 2, 30 + n * d + 20, '(b) Beneš 网络：5 级，每级 4 个 2×2 开关', { color: '#222', size: 13 });
  },

  // 图 4.7 静态调度：指令包与槽
  '04-vliw'(f) {
    const slots = ['标量', '向量 0', '向量 1', '加载', '存储', '矩阵'];
    const prog = [
      ['加地址', '乘', 'nop', '读 $v_1$', 'nop', '推入 $v_0$'],
      ['比较', '加', '乘', '读 $v_2$', 'nop', 'nop'],
      ['跳转', 'nop', '加', 'nop', '写 $v_3$', '取结果'],
    ];
    f.grid(90, 40, {
      rows: 3, cols: 6, cw: 70, ch: 30, size: 12,
      fill: (r, c) => (prog[r][c] === 'nop' ? 'gray' : 'blue'),
      label: (r, c) => prog[r][c],
      rowLabels: ['指令包 0', '指令包 1', '指令包 2'], colLabels: slots,
    });
    f.note(90 + 210, 148, '每个周期发射一个指令包；每个单元在包里有固定的槽，没事做的槽填 nop', { size: 12 });
  },

  // 图 4.8 多线程隐藏延迟：4 个线程轮流发射
  '04-multithread'(f) {
    const L = 3;
    const bars = [];
    for (let t = 0; t < 4; t++)
      for (let k = 0; k < 3; k++) {
        const c = t + 4 * k;
        bars.push([t + 1, c, c + 1, '', OPS[t]]);
        if (c + 1 + L <= 12) bars.push([t + 1, c + 1, c + 1 + L, '等待', 'gray']);
      }
    for (let c = 0; c < 12; c++) bars.push([0, c, c + 1, String(c % 4), OPS[c % 4]]);
    f.timeline(90, 20, { lanes: ['发射', '线程 0', '线程 1', '线程 2', '线程 3'], bars, unit: 34, lh: 24, gap: 6, axisLabel: '周期', ticks: 2 });
  },

  // 图 4.9 DMA：发起与等待之间，处理器做别的事
  '04-dma'(f) {
    f.timeline(100, 20, {
      lanes: ['处理器', 'DMA 引擎'], unit: 46, lh: 30, ticks: 0,
      bars: [
        [0, 0, 1, '发起', 'orange'], [0, 1, 6, '与这次拷贝无关的工作', 'blue'], [0, 6, 6.6, '', 'gray'], [0, 6.6, 9, '使用数据', 'green'],
        [1, 1, 6.6, '后台拷贝：$\\alpha + S/\\beta$', 'teal'],
      ],
      deps: [[0, 4], [4, 3]],
    });
    f.note(100 + 6.3 * 46, 8, '等待 $c \\ge n$', { color: '#222' });
  },

  // 图 4.10 单次拷贝的效率 η = S / (S + αβ)
  '04-alpha-beta'(f) {
    const pts = Array.from({ length: 41 }, (_, k) => {
      const x = 10 ** (k / 10 - 2);
      return [x, x / (1 + x)];
    });
    f.plot(80, 30, {
      w: 380, h: 200,
      x: { min: 0.01, max: 100, log: true, label: '$S / (\\alpha\\beta)$', ticks: [0.01, 0.1, 1, 10, 100] },
      y: { min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], label: '有效带宽占 $\\beta$ 的比例 $\\eta$' },
      series: [{ pts, color: 'blue' }],
      marks: [
        { at: [1, 0.5], label: '$S = \\alpha\\beta$：50%', dx: 10, dy: 8 },
        { at: [9, 0.9], label: '$S = 9\\alpha\\beta$：90%', dx: 8, dy: 16 },
      ],
    });
  },

  // 图：一个累加器与四个累加器（L = 4）
  '04-accumulators'(f) {
    f.note(40, 8, '(a) 一个累加器：每次加法等上一次的结果', { anchor: 'start', color: '#222', size: 13 });
    f.timeline(90, 24, { lanes: ['$s$'], bars: [0, 1, 2].map((k) => [0, 4 * k, 4 * k + 4, `$+x_${k}$`, OPS[0]]), unit: 24, lh: 24, end: 15, ticks: 4, axisLabel: '周期' });
    f.note(40, 104, '(b) 四个累加器轮流使用：每个周期都有一次加法开始', { anchor: 'start', color: '#222', size: 13 });
    const bars = Array.from({ length: 12 }, (_, i) => [i % 4, i, i + 4, `$+x_{${i}}$`, OPS[i % 4]]);
    f.timeline(90, 120, { lanes: ['$s_0$', '$s_1$', '$s_2$', '$s_3$'], bars, unit: 24, lh: 24, gap: 6, ticks: 4, axisLabel: '周期' });
  },

  // 图：按点存放与按分量存放（SIMD 宽度 4）
  '04-aos-soa'(f) {
    const C = { x: 'blue', y: 'green', z: 'orange' };
    const cells = (x0, y0, names, label) => {
      const g = f.grid(x0, y0, { rows: 1, cols: names.length, cw: 34, ch: 24, size: 12, fill: (r, c) => C[names[c][0]], label: (r, c) => `$${names[c][0]}_${names[c][1]}$` });
      if (label) f.text(x0 - 10, y0 + 12, label, { anchor: 'end', size: 12 });
      return g;
    };
    const aos = ['x0', 'y0', 'z0', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3'];
    f.note(110, 12, '按点存放（结构的数组）', { anchor: 'start', color: '#222', size: 13 });
    cells(110, 26, aos, '存储');
    [0, 1, 2].forEach((k) => cells(110 + k * 150, 76, aos.slice(4 * k, 4 * k + 4), k ? '' : '寄存器'));
    f.note(110, 116, '每个寄存器里混着 $x, y, z$，要先重新排列', { anchor: 'start' });
    const soa = ['x0', 'x1', 'x2', 'x3', 'y0', 'y1', 'y2', 'y3', 'z0', 'z1', 'z2', 'z3'];
    f.note(110, 152, '按分量存放（数组的结构）', { anchor: 'start', color: '#222', size: 13 });
    cells(110, 166, soa, '存储');
    [0, 1, 2].forEach((k) => cells(110 + k * 150, 216, soa.slice(4 * k, 4 * k + 4), k ? '' : '寄存器'));
    [0, 1].forEach((k) => f.text(110 + k * 150 + 143, 228, '+', { size: 16 }));
    f.note(110, 256, '逐通道相加：两条向量加法算出 4 个点的 $x + y + z$', { anchor: 'start' });
  },

  // 图：用循环移位在 8 个通道上求和
  '04-lane-sum'(f) {
    const rows = [[3, 1, 4, 1, 5, 9, 2, 6], [8, 10, 6, 7, 8, 10, 6, 7], [14, 17, 14, 17, 14, 17, 14, 17], [31, 31, 31, 31, 31, 31, 31, 31]];
    const names = ['开始', '移 4 位再加', '移 2 位再加', '移 1 位再加'];
    const G = rows.map((v, r) => {
      const g = f.grid(130, 30 + r * 62, { rows: 1, cols: 8, cw: 40, ch: 28, size: 13, fill: (_, c) => (c === 0 ? 'orange' : 'blue'), label: (_, c) => v[c], colLabels: r === 0 ? (c) => `通道 ${c}` : undefined });
      f.text(120, 30 + r * 62 + 14, names[r], { anchor: 'end', size: 12 });
      return g;
    });
    [4, 2, 1].forEach((d, r) => {
      f.line([G[r].cell(0, 0).B(), G[r + 1].cell(0, 0).T()], { color: 'red', width: 2 });
      f.line([G[r].cell(0, d).B(), G[r + 1].cell(0, 0).T()], { color: 'red', width: 2 });
    });
  },

  // 图：DMA 的二维拷贝：大矩阵中的一块拷到片上并紧密排列
  '04-dma-tile'(f) {
    const inBlock = (r, c) => r >= 3 && r <= 8 && c >= 4 && c <= 11;
    const A = f.grid(40, 40, { rows: 12, cols: 16, cw: 14, ch: 14, stroke: '#ccc', fill: (r, c) => (inBlock(r, c) ? 'orange' : null) });
    f.note(A.cx, 18, 'HBM：按行存放的大矩阵', { color: '#222', size: 13 });
    f.brace(A.cell(3, 4).x, A.y + A.h + 8, A.cell(3, 11).x + 14, '每段 1024 字节，连续');
    f.brace(A.cell(3, 0).y, A.x + A.w + 8, A.cell(4, 0).y, '源步长：一整行', { vertical: true });
    const B = f.grid(430, 82, { rows: 6, cols: 8, cw: 14, ch: 14, stroke: '#999', fill: () => 'orange' });
    f.note(B.cx, 60, '片上存储：紧密排列', { color: '#222', size: 13 });
    f.brace(B.y, B.x + B.w + 8, B.y + 14, '目的步长：一段', { vertical: true });
    f.arrow([A.x + A.w + 110, B.cy], [B.x - 12, B.cy], { color: 'teal', width: 2 });
    f.note(A.x + A.w + 110 + (B.x - 12 - A.x - A.w - 110) / 2, B.cy - 14, 'DMA', { color: 'teal' });
  },

  // 图：解耦访存与执行
  '04-decouple'(f) {
    const acc = f.box(20, 100, 120, 44, '访存流', { color: 'orange' });
    const hbm = f.box(230, 10, 120, 40, 'HBM', { color: 'gray' });
    const dma = f.box(230, 100, 120, 44, 'DMA 引擎', { color: 'teal' });
    const sram = f.box(440, 100, 120, 44, '片上存储', { color: 'yellow' });
    const cnt = f.box(230, 200, 120, 44, '完成计数器', { color: 'gray' });
    const cmp = f.box(440, 200, 120, 44, '计算流', { color: 'blue' });
    f.arrow(acc.R(), dma.L());
    f.note((acc.x + acc.w + dma.x) / 2, acc.cy - 12, '描述符');
    f.arrow(hbm.B(), dma.T());
    f.arrow(dma.R(), sram.L());
    f.note((dma.x + dma.w + sram.x) / 2, dma.cy - 12, '数据');
    f.arrow(dma.B(), cnt.T());
    f.note(dma.cx + 8, (dma.y + dma.h + cnt.y) / 2, '完成时加数', { anchor: 'start' });
    f.arrow(cnt.R(), cmp.L());
    f.note((cnt.x + cnt.w + cmp.x) / 2, cnt.cy - 12, '等待');
    f.arrow(sram.B(), cmp.T());
    f.note(acc.cx, acc.y + acc.h + 16, '算地址、发拷贝，跑在前面');
    f.note(cmp.cx, cmp.y + cmp.h + 16, '按序计算已到达的块');
  },
};
