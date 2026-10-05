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
};
