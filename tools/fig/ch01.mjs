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
    [132, 252, 362].forEach((x, k) => f.note(x, 222, `第 ${k + 1} 级`));
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
};