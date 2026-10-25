export default {
  '36-dag'(f) {
    const a = f.box(20, 85, 130, 48, 'A 读文件：2 秒', { color: 'gray' });
    const b = f.box(210, 15, 150, 48, 'B 抽取字段：6 秒', { color: 'blue' });
    const c = f.box(210, 155, 150, 48, 'C 核对硬件：10 秒', { color: 'orange' });
    const d = f.box(430, 85, 130, 48, 'D 汇总：3 秒', { color: 'gray' });
    [[a, b], [b, d]].forEach(([u, v]) => f.link(u, v, { arrow: 'end', color: 'blue' }));
    [[a, c], [c, d]].forEach(([u, v]) => f.link(u, v, { arrow: 'end', color: 'orange' }));
    f.text(290, 250, '关键路径 A → C → D：15 秒');
    f.note(290, 277, '总工作量 21 秒；两个工作者可达到 15 秒');
  },
};
