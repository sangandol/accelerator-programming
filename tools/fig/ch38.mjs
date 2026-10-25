export default {
  '38-control'(f) {
    const planner = f.box(20, 30, 165, 54, '规划者：选择方向', { color: 'blue' });
    const intents = f.box(245, 30, 150, 54, '可领取的意图', { color: 'orange' });
    const workers = f.box(455, 30, 165, 54, '工作者：调用工具', { color: 'green' });
    const results = f.box(455, 180, 165, 54, '结论与原始证据', { color: 'green' });
    const store = f.box(245, 180, 150, 54, '任务与图存储', { color: 'gray' });
    const events = f.box(20, 180, 165, 54, '具体变化的事件', { color: 'purple' });
    [[planner, intents], [intents, workers], [workers, results],
      [results, store], [store, events], [events, planner]]
      .forEach(([a, b]) => f.link(a, b, { arrow: 'end' }));
    f.note(320, 132, '意图、结论与目标各有编号');
    f.note(320, 280, '图示为常规控制路径；心跳和用户干预见正文');
  },
};
