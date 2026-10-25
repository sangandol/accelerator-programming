export default {
  '35-tool-loop'(f) {
    const context = f.box(20, 30, 150, 54, '当前上下文', { color: 'gray' });
    const model = f.box(230, 30, 120, 54, '模型', { color: 'blue' });
    const proposal = f.box(410, 30, 180, 54, '提出工具与参数', { color: 'orange' });
    const check = f.box(410, 175, 180, 54, '参数与权限检查', { color: 'orange' });
    const tool = f.box(230, 175, 120, 54, '实际工具', { color: 'green' });
    const receipt = f.box(20, 175, 150, 54, '记录实际返回', { color: 'gray' });
    const final = f.box(215, -85, 150, 48, '最终答案与验收', { color: 'purple' });
    [[context, model], [model, proposal], [proposal, check], [check, tool],
      [tool, receipt], [receipt, context], [model, final]]
      .forEach(([a, b]) => f.link(a, b, { arrow: 'end' }));
    f.note(305, 128, '调用提议不等于已经执行');
  },
};
