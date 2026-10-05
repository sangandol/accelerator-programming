// 第 3 章的插图。
export default {
  // 图 3.1 SRAM 阵列：行译码选中一行，整行读出，再选列
  '03-sram'(f) {
    const g = f.grid(130, 40, { rows: 8, cols: 16, cw: 18, ch: 18, fill: (r) => (r === 3 ? 'orange' : null), stroke: '#bbb' });
    const dec = f.box(50, 40, 60, 144, '行译码', { size: 13 });
    f.arrow([0, 112], dec.L());
    f.note(24, 98, '行地址', { color: '#222' });
    const wl = g.cell(3, 0).L();
    f.arrow([dec.x + dec.w, wl[1]], wl, { color: 'orange' });
    f.text(g.x + g.w + 8, wl[1], '字线：选中一行', { anchor: 'start', size: 12, color: 'orange' });
    const bl = g.cell(0, 6).cx;
    f.line([[bl, g.y - 10], [bl, g.y + g.h]], { color: 'blue', width: 2 });
    f.text(bl, g.y - 20, '位线（每列一条）', { size: 12, color: 'blue' });
    const amp = f.box(g.x, g.y + g.h + 14, g.w, 26, '读出放大：整行同时读出', { size: 12, color: 'green' });
    const mux = f.box(g.cx - 70, amp.y + 50, 140, 28, '列选择器', { size: 12 });
    f.arrow(amp.B(), mux.T());
    f.arrow([0, mux.cy], mux.L());
    f.note(24, mux.cy - 14, '列地址', { color: '#222' });
    f.arrow(mux.B(), [mux.cx, mux.y + mux.h + 30]);
    f.text(mux.cx + 8, mux.y + mux.h + 22, '数据', { anchor: 'start', size: 13 });
  },

  // 图 3.2 8 个存储体的交错存储：不同步长的访问落在哪些存储体
  '03-banks'(f) {
    const k = 8;
    [[1, 0, 0], [2, 1, 0], [8, 0, 1], [9, 1, 1]].forEach(([s, px, py]) => {
      const hit = new Set(Array.from({ length: k }, (_, i) => i * s));
      const x0 = 50 + px * 230;
      const y0 = 40 + py * 250;
      const g = f.grid(x0, y0, {
        rows: 8, cols: k, cw: 21, ch: 21, size: 9,
        fill: (r, c) => (hit.has(k * r + c) ? 'orange' : null),
        label: (r, c) => k * r + c,
        colLabels: (c) => c,
      });
      const conflicts = Math.max(...Array.from({ length: k }, (_, b) => [...hit].filter((a) => a % k === b).length));
      f.text(g.x + g.w / 2, g.y + g.h + 18, `步长 ${s}：冲突度 ${conflicts}`, { size: 13 });
    });
    f.note(50 + 84, 14, '存储体编号（地址 mod 8）');
    f.note(50 + 230 + 84, 14, '存储体编号（地址 mod 8）');
  },

  // 图 3.3 DRAM 的存储体与行缓冲；HBM 的封装（侧视）
  '03-dram-hbm'(f) {
    const g = f.grid(80, 30, { rows: 10, cols: 16, cw: 12, ch: 12, fill: (r) => (r === 6 ? 'orange' : null), stroke: '#bbb' });
    f.box(40, 30, 30, 120, '', { color: 'gray' });
    f.note(55, 160, '行译码');
    const buf = f.grid(80, 175, { rows: 1, cols: 16, cw: 12, ch: 14, fill: (r, c) => (c >= 4 && c < 8 ? 'red' : 'orange'), stroke: '#999' });
    f.arrow([g.x + g.w + 10, g.cell(6, 0).cy], [g.x + g.w + 10, buf.y + 7]);
    f.text(g.x + g.w + 16, 120, '① 激活：整行读入行缓冲', { anchor: 'start', size: 12 });
    f.text(g.x + g.w + 16, 182, '行缓冲', { anchor: 'start', size: 12 });
    const burst = buf.region(0, 4, 0, 7);
    f.arrow(burst.B(), [burst.cx, burst.y + burst.h + 30], { color: 'red' });
    f.text(burst.cx + 8, burst.y + burst.h + 22, '② 读一个突发（如 32 字节）', { anchor: 'start', size: 12, color: 'red' });
    f.text(g.x + g.w + 16, 140, '③ 换行之前先预充电', { anchor: 'start', size: 12 });
    f.note(g.cx, 14, '(a) 一个 DRAM 存储体', { color: '#222', size: 13 });

    // (b) side view of a package: processor and HBM stacks on a silicon interposer
    const ox = 480;
    f.box(ox, 150, 330, 16, '', { color: 'gray', rx: 2 });
    f.note(ox + 165, 182, '硅中介层：上千条短而密的连线');
    f.box(ox + 110, 80, 110, 70, '处理器', { color: 'blue' });
    [ox + 10, ox + 240].forEach((x) => {
      for (let l = 0; l < 5; l++) f.box(x, 130 - l * 14, 80, 12, '', { color: l === 0 ? 'gray' : 'green', rx: 1 });
      f.text(x + 40, 56, 'HBM 堆栈', { size: 12 });
    });
    for (let i = 0; i < 4; i++) {
      f.line([[ox + 70 + i * 6, 152 + i * 3], [ox + 130 - i * 6, 152 + i * 3]], { color: 'blue', width: 1 });
      f.line([[ox + 200 + i * 6, 152 + i * 3], [ox + 260 - i * 6, 152 + i * 3]], { color: 'blue', width: 1 });
    }
    f.note(ox + 165, 14, '(b) HBM 的封装（侧视）', { color: '#222', size: 13 });
  },

  // 图 3.4 每次操作的能耗（45 nm，对数刻度）
  '03-energy'(f) {
    f.bars(200, 20, {
      log: true, min: 0.01, max: 3000, w: 360,
      items: [
        ['8 位整数加法', 0.03], ['32 位整数加法', 0.1], ['8 位整数乘法', 0.2], ['16 位浮点乘法', 1.1],
        ['32 位整数乘法', 3.1], ['32 位浮点乘法', 3.7],
        ['读 8 KiB SRAM（64 位）', 10, 'green'], ['读 1 MiB SRAM（64 位）', 100, 'green'], ['读 DRAM（64 位）', 2000, 'red'],
      ],
      fmt: (v) => (v >= 1000 ? '1.3–2.6 nJ' : `${v} pJ`),
    });
  },

  // 图 3.5 存储层次
  '03-hierarchy'(f) {
    const levels = [
      ['寄存器', '几百 KiB', '最高', '1 周期', 'orange'],
      ['片上 SRAM', '几十到几百 MiB', '几十 TB/s', '几到几十周期', 'green'],
      ['HBM', '几十到几百 GB', '几 TB/s', '几百周期', 'blue'],
    ];
    const cx = 170;
    const h = 56;
    const half = (y) => 30 + (y - 30) * 0.75;
    levels.forEach(([name, cap, bw, lat, color], i) => {
      const [y0, y1] = [40 + i * h, 40 + (i + 1) * h];
      f.poly([[cx - half(y0), y0], [cx + half(y0), y0], [cx + half(y1), y1], [cx - half(y1), y1]], { color });
      f.text(cx, (y0 + y1) / 2, name, { size: 14 });
      [cap, bw, lat].forEach((s, j) => f.text(390 + j * 120, (y0 + y1) / 2, s, { size: 12 }));
    });
    ['容量', '带宽', '延迟'].forEach((s, j) => f.text(390 + j * 120, 22, s, { size: 13, weight: 'bold' }));
    f.arrow([cx - 200, 50], [cx - 200, 40 + 3 * h]);
    f.text(cx - 210, 40 + 1.5 * h, '更大、更慢、更费能', { size: 12, anchor: 'end' });
  },
};
