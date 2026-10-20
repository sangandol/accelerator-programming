export default {
  '21-fix-costs'(f) {
    f.bars(230, 20, {
      w: 330,
      max: 2600,
      items: [
        ['官方 Pallas（会出错）', 685, 'gray'],
        ['朴素屏蔽：修复重选', 1355, 'red'],
        ['抬升：修复重选', 697, 'blue'],
        ['MXU 后缀名次：修复重选', 685, 'blue'],
        ['分段整数键：逐位正确', 717, 'green'],
        ['原生 XLA（正确）', 2564, 'orange'],
      ],
      fmt: (v) => `${v}`,
    });
  },
  '21-topk-vs-k'(f) {
    const ks = [1, 8, 16, 32, 128];
    const off = [117, 685, 1341, 2653, 10525];
    const xla = [1992, 2564, 2564.5, 2564.5, 2560.5];
    const ours = [149, 717, 830, 894, 1278];
    f.plot(70, 30, {
      w: 420, h: 250,
      x: { min: 1, max: 128, log: true, label: '$k$', ticks: [1, 8, 16, 32, 128] },
      y: { min: 100, max: 20000, log: true, label: '周期', ticks: [100, 1000, 10000] },
      series: [
        { pts: ks.map((k, i) => [k, off[i]]), color: 'red', label: '官方 Pallas', at: 4, dx: -78, dy: -12 },
        { pts: ks.map((k, i) => [k, xla[i]]), color: 'orange', label: '原生 XLA', at: 0, dx: 8, dy: -12 },
        { pts: ks.map((k, i) => [k, ours[i]]), color: 'blue', label: '按形状选择', at: 4, dx: -70, dy: 16 },
      ],
    });
  },
};
