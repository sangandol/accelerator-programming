export default {
  '34-budget'(f) {
    const k = 2; // 每 GiB 画 2 像素
    const x0 = 120;
    const rows = [
      ['H100 + bf16', 74.5, [[130.4, 'blue']], '权重 130.4 GiB 超过卡容量 74.5 GiB：装不下'],
      ['H100 + fp8', 74.5, [[65.2, 'blue'], [8, 'gray'], [1.3, 'green']], '约 4300 个词元'],
      ['B200 + bf16', 178.8, [[130.4, 'blue'], [8, 'gray'], [40.4, 'green']], '约 132000 个词元'],
      ['B200 + fp8', 178.8, [[65.2, 'blue'], [8, 'gray'], [105.6, 'green']], '约 346000 个词元'],
    ];
    const stroke = { blue: '#3a5a8c', gray: '#555555', green: '#2e7d32' };
    rows.forEach(([name, cap, segs, note], i) => {
      const y = 20 + i * 58;
      f.text(20, y + 15, name, { anchor: 'start', size: 14 });
      let x = x0;
      for (const [g, c] of segs) {
        f.rect(x, y, g * k, 30, { fill: c, stroke: stroke[c] });
        x += g * k;
      }
      const xc = x0 + cap * k;
      f.line([[xc, y - 4], [xc, y + 34]], { color: 'red', width: 2 });
      f.note(Math.max(x, xc) + 10, y + 15, note, { anchor: 'start' });
      f.note(xc, y + 44, `卡容量 ${cap} GiB`, { anchor: 'middle' });
    });
    const ly = 20 + 4 * 58 + 6;
    [['权重', 'blue'], ['预留 8 GiB', 'gray'], ['留给 KV cache', 'green']].forEach(([s, c], j) => {
      const lx = x0 + j * 150;
      f.rect(lx, ly, 16, 16, { fill: c, stroke: stroke[c] });
      f.text(lx + 24, ly + 8, s, { anchor: 'start', size: 13 });
    });
  },
};
