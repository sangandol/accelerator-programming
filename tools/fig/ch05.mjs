// 第 5 章的插图。
import { oblique } from './lib.mjs';

export default {
  // 图 5.1 操作数只从边界进入的乘加阵列
  '05-array'(f) {
    const pe = (r, c) => f.box(120 + c * 64, 50 + r * 54, 40, 34, '', { color: 'blue' });
    const P = [0, 1, 2, 3].map((r) => [0, 1, 2, 3].map((c) => pe(r, c)));
    for (let r = 0; r < 4; r++)
      for (let c = 0; c < 4; c++) {
        if (c < 3) f.arrow(P[r][c].R(), P[r][c + 1].L(), { color: 'orange', width: 1.4 });
        if (r < 3) f.arrow(P[r][c].B(), P[r + 1][c].T(), { color: 'green', width: 1.4 });
      }
    for (let r = 0; r < 4; r++) f.arrow([60, P[r][0].cy], P[r][0].L(), { color: 'orange', width: 1.4 });
    for (let c = 0; c < 4; c++) f.arrow([P[0][c].cx, 14], P[0][c].T(), { color: 'green', width: 1.4 });
    f.text(50, P[1][0].y + 44, '从左边进入', { anchor: 'end', size: 12, color: 'orange' });
    f.text(P[0][3].x + 60, 22, '从上边进入', { anchor: 'start', size: 12, color: 'green' });
    f.note(P[3][0].x + 130, P[3][0].y + 58, '每周期：边上进入 $O(R + C)$ 个数，内部完成 $RC$ 次乘加', { color: '#222', size: 12 });
  },

  // 图 5.2 矩阵乘法的迭代空间与沿 u = (1, 0, 0) 的投影
  '05-spacetime'(f) {
    const p = oblique(70, 60, [0, 52], [56, 0], [30, -24]);
    const R = [0, 1, 2];
    for (const a of R) for (const b of R) {
      f.line([p(0, a, b), p(2, a, b)], { color: '#ccc', width: 1 });
      f.line([p(a, 0, b), p(a, 2, b)], { color: '#ccc', width: 1 });
      f.line([p(a, b, 0), p(a, b, 2)], { color: '#ccc', width: 1 });
    }
    for (const i of R) for (const j of R) for (const k of R) f.dot(...p(i, j, k), 3, j === 1 && k === 1 ? 'red' : '#666');
    f.line([p(-0.3, 1, 1), p(2.3, 1, 1)], { color: 'red', width: 2 });
    f.arrow(p(0, 0, 0), p(2.7, 0, 0), { color: 'gray' });
    f.arrow(p(0, 0, 0), p(0, 2.7, 0), { color: 'gray' });
    f.arrow(p(0, 0, 0), p(0, 0, 2.7), { color: 'gray' });
    f.text(p(2.7, 0, 0)[0] - 12, p(2.7, 0, 0)[1], '$i$');
    f.text(p(0, 2.7, 0)[0] + 6, p(0, 2.7, 0)[1] - 12, '$j$');
    f.text(p(0, 0, 2.7)[0] - 10, p(0, 0, 2.7)[1] - 6, '$k$');
    // dependencies into the point (2, 2, 2)
    f.arrow(p(2, 1, 2), p(2, 2, 2), { color: 'orange', width: 2.2 });
    f.arrow(p(1, 2, 2), p(2, 2, 2), { color: 'green', width: 2.2 });
    f.arrow(p(2, 2, 1), p(2, 2, 2), { color: 'blue', width: 2.2 });
    [['orange', '$d_a = (0, 1, 0)$：$a$ 沿 $j$ 传递'], ['green', '$d_b = (1, 0, 0)$：$b$ 沿 $i$ 传递'], ['blue', '$d_c = (0, 0, 1)$：$c$ 沿 $k$ 累加']].forEach(([c, s], n) => {
      f.arrow([40, 220 + n * 22], [70, 220 + n * 22], { color: c, width: 2.2 });
      f.text(78, 220 + n * 22, s, { anchor: 'start', size: 12 });
    });

    const g = f.grid(380, 70, { rows: 3, cols: 3, cw: 40, ch: 40, fill: (r, c) => (r === 1 && c === 1 ? 'red' : 'blue'), colLabels: (c) => `$j = ${c}$` });
    [0, 1, 2].forEach((r) => f.note(g.x + g.w + 8, g.cell(r, 0).cy, `$k = ${r}$`, { anchor: 'start' }));
    f.arrow([p(2.3, 1, 1)[0] + 30, p(2.3, 1, 1)[1] - 10], g.cell(1, 1).L(), { color: 'red', bend: 0 });
    f.note(g.cx, g.y + g.h + 18, '处理单元阵列 $(j, k)$', { color: '#222', size: 12 });
  },

  // 图 5.3 权重驻留的脉动阵列：A 的行错开一拍进入，部分和向下流动
  '05-weight-stationary'(f) {
    const K = 3;
    const N = 3;
    const pe = [];
    for (let k = 0; k < K; k++) {
      pe.push([]);
      for (let j = 0; j < N; j++) pe[k].push(f.box(300 + j * 84, 50 + k * 68, 56, 40, `$B_{${k}${j}}$`, { color: 'blue', size: 13 }));
    }
    for (let k = 0; k < K; k++) {
      for (let i = 0; i < 3; i++) f.box(300 - 46 - (i + k) * 40, pe[k][0].cy - 13, 32, 26, `$a_{${i}${k}}$`, { color: 'orange', size: 12, rx: 3 });
      f.arrow([300 - 12, pe[k][0].cy], pe[k][0].L(), { color: 'orange' });
      for (let j = 0; j + 1 < N; j++) f.arrow(pe[k][j].R(), pe[k][j + 1].L(), { color: 'orange' });
    }
    for (let j = 0; j < N; j++) {
      f.arrow([pe[0][j].cx, 18], pe[0][j].T(), { color: 'green' });
      f.text(pe[0][j].cx + 8, 16, '0', { anchor: 'start', size: 12, color: 'green' });
      for (let k = 0; k + 1 < K; k++) f.arrow(pe[k][j].B(), pe[k + 1][j].T(), { color: 'green' });
      f.arrow(pe[K - 1][j].B(), [pe[K - 1][j].cx, pe[K - 1][j].y + 74], { color: 'green' });
      f.text(pe[K - 1][j].cx, pe[K - 1][j].y + 84, `$C_{i${j}}$`, { size: 13, color: 'green' });
    }
    f.note(150, 40, '$A$ 的第 $k$ 列错开 $k$ 拍进入', { color: 'orange', size: 12 });
    f.note(300 + 84 * 3, pe[1][2].cy, '部分和向下流动并累加', { color: 'green', anchor: 'start', size: 12 });
    f.note(300 + 84 * 3, pe[0][2].cy, '权重驻留', { color: 'blue', anchor: 'start', size: 12 });
  },

  // 图 5.4 权重双缓冲：计算一个权重块时，在后台装入下一个
  '05-weight-double-buffer'(f) {
    f.timeline(100, 20, {
      lanes: ['装入权重', '计算'], unit: 22, ticks: 4, axisLabel: '周期',
      bars: [
        [0, 0, 4, '$W_0$', 'orange'], [0, 4, 8, '$W_1$', 'orange'], [0, 12, 16, '$W_2$', 'orange'],
        [1, 4, 12, '用 $W_0$ 算 $m$ 行'], [1, 12, 20, '用 $W_1$ 算 $m$ 行'], [1, 20, 28, '用 $W_2$ 算 $m$ 行'],
      ],
    });
  },

  // 图 5.5 填充的浪费：N = 100 与 N = 129 在宽 128 的阵列上
  '05-padding'(f) {
    const s = 1.4;
    const row = (y, n, tiles, label) => {
      for (let t = 0; t < tiles; t++) f.box(150 + t * 128 * s, y, 128 * s, 30, '', { color: 'gray', rx: 0 });
      let left = n;
      for (let t = 0; t < tiles; t++) {
        const used = Math.min(128, left);
        f.box(150 + t * 128 * s, y, used * s, 30, '', { color: 'blue', rx: 0 });
        left -= used;
      }
      f.text(140, y + 15, label, { anchor: 'end', size: 13 });
      f.text(150 + tiles * 128 * s + 10, y + 15, `利用率 ${Math.round((100 * n) / (128 * tiles))}%`, { anchor: 'start', size: 13 });
    };
    row(20, 100, 1, '$N = 100$');
    row(70, 129, 2, '$N = 129$');
    f.box(150, 120, 14, 14, '', { color: 'blue', rx: 2 });
    f.text(170, 127, '有效的列', { anchor: 'start', size: 12 });
    f.box(250, 120, 14, 14, '', { color: 'gray', rx: 2 });
    f.text(270, 127, '补零（浪费）', { anchor: 'start', size: 12 });
  },

  // 图：2 × 2 输出驻留阵列的四拍（A = [[1,2],[3,4]]，B = [[5,6],[7,8]]）
  '05-os-snapshots'(f) {
    const A = [[1, 2], [3, 4]];
    const B = [[5, 6], [7, 8]];
    const c = [[0, 0], [0, 0]];
    for (let t = 0; t < 4; t++) {
      const x0 = 30 + t * 170;
      for (let i = 0; i < 2; i++)
        for (let j = 0; j < 2; j++) {
          const k = t - i - j;
          const active = k >= 0 && k < 2;
          if (active) c[i][j] += A[i][k] * B[k][j];
          f.box(x0 + 44 + j * 54, 64 + i * 50, 42, 36, String(c[i][j]), { color: active ? 'orange' : 'blue', size: 13 });
        }
      for (let i = 0; i < 2; i++) {
        const k = t - i;
        if (k >= 0 && k < 2) f.text(x0 + 26, 64 + i * 50 + 18, String(A[i][k]), { size: 13, color: '#b45f06' });
      }
      for (let j = 0; j < 2; j++) {
        const k = t - j;
        if (k >= 0 && k < 2) f.text(x0 + 44 + j * 54 + 21, 46, String(B[k][j]), { size: 13, color: '#2e7d32' });
      }
      f.note(x0 + 92, 180, `第 ${t} 拍`, { color: '#222', size: 13 });
    }
    f.note(30, 12, '方框中是这一拍结束时的累加器；左边是这一拍进入的 $A$ 元素，上边是进入的 $B$ 元素', { anchor: 'start' });
  },

  // 图：点积单元算一个输出元素（k = 4）
  '05-dot-unit'(f) {
    const xs = [60, 150, 240, 330];
    const muls = xs.map((x, t) => {
      f.text(x, 18, `$a_${t}, b_${t}$`, { size: 13 });
      const n = f.node(x, 66, '×', { r: 15 });
      f.arrow([x, 30], [x, 51]);
      return n;
    });
    const s1 = [f.node(105, 126, '+', { r: 15 }), f.node(285, 126, '+', { r: 15 })];
    [0, 1].forEach((h) => { f.link(muls[2 * h], s1[h]); f.link(muls[2 * h + 1], s1[h]); });
    const s2 = f.node(195, 186, '+', { r: 15 });
    s1.forEach((n) => f.link(n, s2));
    const acc = f.node(195, 246, '+', { r: 15, color: 'orange' });
    f.link(s2, acc);
    const reg = f.box(290, 230, 110, 32, '累加器 $c$', { color: 'orange', size: 12 });
    f.link(reg, acc, { arrow: 'end' });
    f.line([[195, 261], [195, 296], [345, 296], [345, reg.y + reg.h]], { arrow: 'end' });
    f.note(430, 66, '4 个乘法器同时工作', { anchor: 'start' });
    f.note(430, 156, '加法树：$\\log_2 4 = 2$ 级', { anchor: 'start' });
    f.note(430, 246, '再加到累加器上', { anchor: 'start' });
  },

  // 图：2:4 稀疏
  '05-sparse'(f) {
    const row = [0, 3, 0, -1, 2, 0, 0, 5];
    const g = f.grid(60, 34, { rows: 1, cols: 8, cw: 36, ch: 28, size: 13, fill: (r, c) => (row[c] ? 'blue' : 'gray'), label: (r, c) => row[c] });
    f.line([[g.cell(0, 4).x, g.y - 6], [g.cell(0, 4).x, g.y + g.h + 6]], { color: 'red', width: 2 });
    f.note(g.cx, 18, '$A$ 的一行：每 4 个元素中至多 2 个非零', { color: '#222' });
    const vals = [3, -1, 2, 5];
    const pos = [1, 3, 0, 3];
    const v = f.grid(110, 110, { rows: 1, cols: 4, cw: 36, ch: 28, size: 13, fill: () => 'blue', label: (r, c) => vals[c] });
    const p = f.grid(110, 150, { rows: 1, cols: 4, cw: 36, ch: 28, size: 13, fill: () => 'yellow', label: (r, c) => pos[c] });
    f.text(v.x - 10, v.y + 14, '非零值', { anchor: 'end', size: 12 });
    f.text(p.x + p.w + 10, p.y + 14, '组内位置（每个 2 位）', { anchor: 'start', size: 12 });
    const pick = [1, 3, 4, 7];
    const Bg = f.grid(440, 20, { rows: 8, cols: 4, cw: 22, ch: 20, stroke: '#bbb', fill: (r) => (pick.includes(r) ? 'orange' : null) });
    for (let r = 0; r < 8; r++) f.note(Bg.x + Bg.w + 10, Bg.cell(r, 0).cy, `第 ${r} 行`, { anchor: 'start' });
    f.note(Bg.cx, Bg.y + Bg.h + 16, '$B$：只取 4 行', { color: '#222' });
    vals.forEach((_, c) => f.line([v.cell(0, c).T(), [Bg.x, Bg.cell(pick[c], 0).cy]], { color: 'orange', width: 1.4, arrow: 'end' }));
  },
};
