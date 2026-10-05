// Samples of every component: smoke test for lib.mjs and a visual catalog (node tools/figures.mjs --demo).
export default {
  // Matrix tiled into 4×4 blocks; one block highlighted.
  'demo-grid'(f) {
    const colors = ['blue', 'orange', 'green', 'purple'];
    const g = f.grid(30, 30, {
      rows: 8, cols: 8,
      fill: (r, c) => colors[2 * Math.floor(r / 4) + Math.floor(c / 4)],
      label: (r, c) => 8 * r + c,
      rowLabels: (r) => r, colLabels: (c) => c,
    });
    f.frame(g.region(4, 4, 7, 7), { label: '块 $(1, 1)$', side: 'right' });
    f.brace(g.x, g.y + g.h + 6, g.x + g.w, '$N = 8$');
  },

  // Double buffering: DMA of block i+1 overlaps compute on block i.
  'demo-timeline'(f) {
    f.timeline(70, 20, {
      lanes: ['DMA', '计算'],
      bars: [
        [0, 0, 2, '读 $A_0$'], [0, 2, 4, '读 $A_1$'], [0, 4, 6, '读 $A_2$'],
        [1, 2, 5, '算 $A_0$', 'orange'], [1, 5, 8, '算 $A_1$', 'orange'],
      ],
      deps: [[0, 3], [1, 4]],
    });
  },

  // Roofline on log-log axes.
  'demo-plot'(f) {
    const ridge = 240;
    f.plot(60, 30, {
      x: { min: 1, max: 10000, log: true, label: 'FLOP/字节', ticks: [1, 10, 100, 1000, 10000] },
      y: { min: 1, max: 2000, log: true, label: 'TFLOP/s', ticks: [1, 10, 100, 1000] },
      series: [{ pts: [[1, 1000 / ridge], [ridge, 1000], [10000, 1000]], label: '屋顶线', at: 2, dx: -50 }],
      marks: [{ at: [ridge, 1000], label: '拐点 $I^* = P/B$' }],
    });
  },

  // Ring of 4 devices with one step of a ring algorithm.
  'demo-ring'(f) {
    const R = f.ring(110, 100, { n: 4, r: 70, label: (i) => `$d_${i}$` });
    for (let i = 0; i < 4; i++) f.link(R.node(i), R.node(i + 1), { arrow: 'end', bend: 0.25, color: 'red' });
  },

  // 3×4 torus.
  'demo-mesh'(f) {
    f.mesh(40, 40, { rows: 3, cols: 4, wrap: true });
  },

  // bf16 bit layout.
  'demo-bitfield'(f) {
    f.bitfield(20, 30, { fields: [[1, '$s$', 'red'], [8, '指数', 'green'], [7, '尾数', 'blue']] });
  },

  // Boxes and links: a small block diagram.
  'demo-blocks'(f) {
    const hbm = f.box(20, 20, 120, 40, 'HBM', { color: 'gray' });
    const vmem = f.box(200, 20, 120, 40, 'VMEM');
    const mxu = f.box(380, 0, 100, 36, 'MXU', { color: 'orange' });
    const vpu = f.box(380, 46, 100, 36, 'VPU', { color: 'green' });
    f.link(hbm, vmem, { arrow: 'both' });
    f.note(170, 30, 'DMA');
    f.link(vmem, mxu, { arrow: 'end' });
    f.link(vmem, vpu, { arrow: 'end' });
  },
};
