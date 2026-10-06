// 第 1 章的插图。
// 每个函数接收一个新的 Fig f 并在上面作图（见 README.md）。
const CARRY = 'red';

export default {
  // 图 1.1 二选一选择器
  '01-mux'(f) {
    f.pin('a', 40, 44, '$a$');
    f.pin('s', 40, 120, '$s$');
    f.pin('b', 40, 180, '$b$');
    f.gate('n', 'not', 110, 68);
    f.gate('g1', 'and', 230, 30);
    f.gate('g2', 'and', 230, 150);
    f.gate('g3', 'or', 340, 90);
    f.net('a', 'g1.in0');
    f.net('s', ['n.in', 'g2.in0'], 80);
    f.net('n.out', 'g1.in1', 195);
    f.net('b', 'g2.in1');
    f.net('g1.out', 'g3.in0', 305);
    f.net('g2.out', 'g3.in1', 315);
    f.pin('y', 430, 112, 'mux($s$, $a$, $b$)', 'out');
    f.net('g3.out', 'y');
    f.text(174, 81, '$\\neg s$', { size: 13 });
    // depth of each gate (定义 1.6)
    [[132, 58, 1], [252, 20, 2], [252, 140, 1], [362, 80, 3]].forEach(([x, y, d]) => f.note(x, y, `深度 ${d}`));
  },

  // 图 1.2 八个输入的与：逐个相与与二叉树
  '01-and-tree'(f) {
    const X = (ox, i) => ox + 38 * i;
    const inputs = (ox) => Array.from({ length: 8 }, (_, i) => (f.text(X(ox, i), 22, `$x_${i}$`, { size: 15 }), [X(ox, i), 34]));

    const tops = inputs(0);
    let prev = tops[0];
    for (let k = 1; k < 8; k++) {
      const g = f.node(X(0, k), 34 + 36 * k, '∧');
      f.link(tops[k], g);
      f.link(prev, g);
      prev = g;
    }
    f.arrow([prev.cx, prev.cy + 13], [prev.cx, prev.cy + 46]);
    f.text(X(0, 3.5), 350, '逐个相与：7 个门，深度 7', { size: 13 });

    let level = inputs(370);
    for (let d = 1; level.length > 1; d++) {
      const next = [];
      for (let j = 0; j < level.length; j += 2) {
        const [a, b] = [level[j], level[j + 1]];
        const g = f.node(((a.cx ?? a[0]) + (b.cx ?? b[0])) / 2, 34 + 60 * d, '∧');
        f.link(a, g);
        f.link(b, g);
        next.push(g);
      }
      level = next;
    }
    f.arrow([level[0].cx, level[0].cy + 13], [level[0].cx, level[0].cy + 50]);
    f.text(X(370, 3.5), 350, '二叉树：7 个门，深度 3', { size: 13 });
  },

  // 图 1.3 全加器 = 两个半加器 + 或门
  '01-full-adder'(f) {
    f.box(88, 22, 98, 184, '', { hollow: true, dash: '5 4' });
    f.note(137, 222, '半加器');
    f.box(232, 28, 114, 132, '', { hollow: true, dash: '5 4' });
    f.note(318, 18, '半加器');
    f.pin('a', 40, 46, '$a$');
    f.pin('b', 40, 62, '$b$');
    f.pin('c', 260, 20, '$c$', 'top');
    f.gate('x1', 'xor', 130, 32);
    f.gate('a1', 'and', 130, 150);
    f.gate('x2', 'xor', 290, 40);
    f.gate('a2', 'and', 290, 110);
    f.gate('o', 'or', 400, 118);
    f.net('a', ['x1.in0', 'a1.in0'], 100);
    f.net('b', ['x1.in1', 'a1.in1'], 115);
    f.net('c', ['x2.in1', 'a2.in1'], 260);
    f.net('x1.out', ['x2.in0', 'a2.in0'], 245);
    f.net('a1.out', 'o.in1', 380);
    f.net('a2.out', 'o.in0');
    f.pin('s', 490, 62, '$s$', 'out');
    f.net('x2.out', 's');
    f.pin('co', 490, 140, "$c'$", 'out');
    f.net('o.out', 'co');
    f.text(205, 44, '$p$');
    f.text(205, 162, '$g$');
    f.text(366, 122, '$t$');
  },

  // 图 1.4 4 位行波进位加法器，最低位在右
  '01-ripple-carry'(f) {
    const bx = (i) => 70 + (3 - i) * 120;
    for (let i = 0; i < 4; i++) {
      const x = bx(i);
      f.box(x, 70, 72, 56, '全加器', { size: 13 });
      f.arrow([x + 22, 34], [x + 22, 70]);
      f.arrow([x + 50, 34], [x + 50, 70]);
      f.text(x + 22, 22, `$a_${i}$`, { size: 15 });
      f.text(x + 50, 22, `$b_${i}$`, { size: 15 });
      f.arrow([x + 36, 126], [x + 36, 158]);
      f.text(x + 36, 172, `$s_${i}$`, { size: 15 });
      const to = i < 3 ? bx(i + 1) + 72 : 24;
      f.arrow([x, 98], [to, 98], { color: CARRY, width: 3 });
      f.text(i < 3 ? (x + to) / 2 : 34, 85, `$c_${i + 1}$`, { size: 15, color: CARRY });
    }
    f.arrow([560, 98], [bx(0) + 72, 98], { color: CARRY, width: 3 });
    f.text(538, 85, '$c_0 = 0$', { size: 15, color: CARRY });
  },

  // 图 1.5 累加器
  '01-accumulator'(f) {
    const add = f.box(200, 40, 120, 50, '加法器');
    const reg = f.box(200, 140, 120, 44, '寄存器 $s$');
    f.text(62, 55, '$x$', { size: 15, anchor: 'end' });
    f.arrow([70, 55], add.L(0.3));
    f.line([add.R(), [370, 65], [370, 162], reg.R()], { arrow: 'end' });
    f.text(380, 113, '$s + x$', { anchor: 'start' });
    f.line([reg.L(), [150, 162], [150, 75], add.L(0.7)], { arrow: 'end' });
    f.arrow([150, 162], [70, 162]);
    f.dot(150, 162);
    f.text(62, 162, '$s$', { size: 15, anchor: 'end' });
    f.arrow([260, 226], [260, 186]);
    f.text(260, 240, '时钟', { size: 13 });
    f.raw('<path d="M252,184 L268,184 L260,174 z" fill="none" stroke="#3a5a8c" stroke-width="1.4"/>', [252, 174, 268, 184], 3);
  },

  // 图 1.9 超前进位加法器：半加器 → 前缀网络（n = 4 的 Kogge–Stone）→ 异或；位置 0 在左
  '01-cla'(f) {
    const X = (i) => 80 + 160 * i;
    const Y = (L) => 165 + 68 * L;
    for (let i = 0; i < 4; i++) {
      const x = X(i);
      f.text(x - 16, 16, `$a_${i}$`, { size: 15 });
      f.text(x + 16, 16, `$b_${i}$`, { size: 15 });
      f.arrow([x - 16, 28], [x - 16, 44]);
      f.arrow([x + 16, 28], [x + 16, 44]);
      f.box(x - 38, 44, 76, 36, '半加器', { size: 13 });
      f.line([[x, 80], [x, 282]]);
      f.text(x - 8, 104, `$(g_${i}, p_${i})$`, { size: 13, anchor: 'end' });
      f.text(x - 8, 268, `$G_${i}$`, { size: 13, anchor: 'end' });
      if (i === 0) {
        f.arrow([x + 28, 80], [x + 28, 368], { color: 'teal' });
        f.text(x + 28, 384, '$s_0 = p_0$', { size: 15 });
      } else {
        f.box(x - 6, 336, 40, 32, '$\\oplus$', { color: 'orange', size: 16 });
        f.arrow([x + 28, 80], [x + 28, 336], { color: 'teal' });
        f.arrow([X(i - 1), 282], [x + 2, 336], { color: CARRY });
        f.arrow([x + 14, 368], [x + 14, 380]);
        f.text(x + 14, 394, `$s_${i}$`, { size: 15 });
      }
      f.text(x + 34, 104, `$p_${i}$`, { size: 13, anchor: 'start', color: 'teal' });
    }
    f.arrow([X(3), 282], [X(3) + 90, 336], { color: CARRY });
    f.text(X(3) + 96, 344, '$c_4 = G_3$', { size: 15, anchor: 'start', color: CARRY });
    for (const [L, i, j] of [[0, 1, 0], [0, 2, 1], [0, 3, 2], [1, 2, 0], [1, 3, 1]]) {
      f.line([[X(j), Y(L) - 40], [X(i), Y(L)]]);
      f.dot(X(j), Y(L) - 40, 2.6);
      f.dot(X(i), Y(L), 6.5, '#3a5a8c');
    }
    f.box(X(0) - 32, 118, X(3) - X(0) + 70, 132, '', { hollow: true, dash: '5 4' });
    f.text(X(0) - 44, 176, '前缀网络', { size: 13, anchor: 'end' });
    f.note(X(0) - 44, 196, '（运算 $\\otimes$）', { anchor: 'end' });
  },

  // 图 1.6–1.8 前缀网络（n = 8）；ops 的每项 [级, i, j]：第 i 列结合第 j 列
  '01-prefix-sklansky': (f) => f.colnet(96, 24, { n: 8, ops: [
    [0, 1, 0], [0, 3, 2], [0, 5, 4], [0, 7, 6],
    [1, 2, 1], [1, 3, 1], [1, 6, 5], [1, 7, 5],
    [2, 4, 3], [2, 5, 3], [2, 6, 3], [2, 7, 3],
  ] }),
  '01-prefix-kogge-stone': (f) => f.colnet(96, 24, { n: 8, ops: [0, 1, 2].flatMap((L) =>
    Array.from({ length: 8 - 2 ** L }, (_, k) => [L, k + 2 ** L, k])) }),
  '01-prefix-brent-kung': (f) => f.colnet(96, 24, { n: 8, ops: [
    [0, 1, 0], [0, 3, 2], [0, 5, 4], [0, 7, 6],
    [1, 3, 1], [1, 7, 5],
    [2, 7, 3], [2, 5, 3],
    [3, 2, 1], [3, 4, 3], [3, 6, 5],
  ] }),
  // 图：一个时钟周期内的信号（命题 1.24 的证明）
  '01-timing'(f) {
    const [e1, e2, xa, xb] = [130, 490, 70, 560];
    const bus = (x0, x1, y, label, color = 'white') => {
      const [h, b] = [26, 6];
      f.poly([[x0, y + h / 2], [x0 + b, y], [x1 - b, y], [x1, y + h / 2], [x1 - b, y + h], [x0 + b, y + h]], { color });
      if (label) f.text((x0 + x1) / 2, y + h / 2, label, { size: 13 });
    };
    f.line([[xa, 58], [e1, 58], [e1, 30], [310, 30], [310, 58], [e2, 58], [e2, 30], [xb, 30]], { width: 2 });
    for (const [x, s] of [[e1, '第 $t$ 个上升沿'], [e2, '第 $t + 1$ 个上升沿']]) {
      f.line([[x, 30], [x, 164]], { color: 'gray', dash: '4 4', width: 1 });
      f.note(x, 14, s);
    }
    bus(xa, 160, 84, '$s_t$');
    bus(160, 520, 84, '$s_{t+1}$');
    bus(520, xb, 84, '');
    bus(xa, 160, 134, '旧值');
    bus(160, 400, 134, '变化中', 'gray');
    bus(400, 520, 134, '稳定');
    bus(520, xb, 134, '', 'gray');
    [['时钟', 44], ['$Q$', 97], ['组合电路输出', 147]].forEach(([s, y]) => f.text(xa - 10, y, s, { anchor: 'end', size: 13 }));
    f.brace(e1, 172, 160, '$t_1$');
    f.brace(160, 172, 400, '至多 $D\\tau$');
    f.brace(460, 172, e2, '$t_2$');
    f.brace(e1, 212, e2, '时钟周期 $T$，要求 $D\\tau + t_1 + t_2 \\le T$');
  },

  // 图：三种进位函数（命题 1.28）
  '01-carry-functions'(f) {
    const maps = [['吸收：送出恒为 0', [0, 0]], ['传递：收到什么送出什么', [0, 1]], ['产生：送出恒为 1', [1, 1]]];
    maps.forEach(([title, out], k) => {
      const x0 = 50 + k * 210;
      f.note(x0, 22, '收到');
      f.note(x0 + 100, 22, '送出');
      const L = [0, 1].map((v) => f.node(x0, 52 + 56 * v, String(v)));
      const R = [0, 1].map((v) => f.node(x0 + 100, 52 + 56 * v, String(v)));
      [0, 1].forEach((v) => f.link(L[v], R[out[v]], { arrow: 'end', color: 'blue' }));
      f.text(x0 + 50, 150, title, { size: 13 });
    });
  },

  // 图：9 个数经 3:2 压缩化为 2 个数（Wallace 树）
  '01-wallace'(f) {
    const item = (x, y, label) => f.box(x - 12, y, 24, 18, label, { color: 'green', size: 11, rx: 3 });
    let xs = Array.from({ length: 9 }, (_, i) => 40 + i * 52);
    let y = 20;
    xs.forEach((x, i) => item(x, y, `$x_${i + 1}$`));
    f.note(520, y + 9, '9 个数', { anchor: 'start' });
    while (xs.length > 2) {
      const yb = y + 46;
      const outY = yb + 28 + 28;
      const next = [];
      const full = 3 * Math.floor(xs.length / 3);
      for (let g = 0; g < full; g += 3) {
        const [a, b, c] = xs.slice(g, g + 3);
        f.box(a - 16, yb, c - a + 32, 28, '3:2', { size: 12 });
        for (const x of [a, b, c]) f.line([[x, y + 18], [x, yb]]);
        for (const x of [(a + b) / 2, (b + c) / 2]) { f.line([[x, yb + 28], [x, outY]]); next.push(x); }
      }
      for (const x of xs.slice(full)) { f.line([[x, y + 18], [x, outY]]); next.push(x); }
      xs = next.sort((p, q) => p - q);
      y = outY;
      xs.forEach((x) => item(x, y, ''));
      f.note(520, y + 9, `${xs.length} 个数`, { anchor: 'start' });
    }
    const [a, b] = xs;
    const cpa = f.box(a - 40, y + 46, b - a + 80, 30, '超前进位加法器', { size: 12, color: 'orange' });
    for (const x of xs) f.line([[x, y + 18], [x, y + 46]]);
    f.arrow(cpa.B(), [cpa.cx, cpa.y + cpa.h + 26]);
    f.text(cpa.cx + 10, cpa.y + cpa.h + 18, '和', { anchor: 'start', size: 13 });
  },

  // 图：两个单元按层调度归约 8 个数（定理 1.57 的例子）
  '01-brent-schedule'(f) {
    const names = ['a', 'b', 'c', 'd', 'e', 'f', 'g'];
    let level = Array.from({ length: 8 }, (_, i) => {
      f.text(30 + 36 * i, 18, `$x_${i}$`, { size: 14 });
      return [30 + 36 * i, 30];
    });
    let k = 0;
    for (let d = 1; level.length > 1; d++) {
      const next = [];
      for (let j = 0; j < level.length; j += 2) {
        const [p, q] = [level[j], level[j + 1]];
        const n = f.node(((p.cx ?? p[0]) + (q.cx ?? q[0])) / 2, 30 + 50 * d, `$${names[k++]}$`);
        f.link(p, n);
        f.link(q, n);
        next.push(n);
      }
      level = next;
    }
    f.timeline(390, 50, {
      lanes: ['单元 0', '单元 1'], unit: 52, lh: 30, ticks: 1, axisLabel: '时间步',
      bars: [[0, 0, 1, '$a$'], [1, 0, 1, '$b$'], [0, 1, 2, '$c$'], [1, 1, 2, '$d$'], [0, 2, 3, '$e$', 'orange'], [1, 2, 3, '$f$', 'orange'], [0, 3, 4, '$g$', 'green'], [1, 3, 4, '空闲', 'gray']],
    });
  },
};