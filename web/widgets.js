// Interactive figures, mounted on <div class="widget" data-widget="NAME"> by name. Loaded by reader.js.
(() => {
  window.bookWidgets = true;
  const h = (tag, attrs = {}, ...kids) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
      else if (v !== null && v !== undefined && v !== false) e.setAttribute(k, v === true ? '' : v);
    }
    e.append(...kids.flat().filter((k) => k !== null && k !== undefined && k !== false));
    return e;
  };
  const svg = (tag, attrs = {}, ...kids) => {
    const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    e.append(...kids);
    return e;
  };
  const sub = (base, i) => h('span', {}, h('i', {}, base), h('sub', {}, String(i)));
  const pair = (g, p) => `(${g}, ${p})`;

  // ---- carry: a + b bit by bit, with carry functions and prefixes (§1.6) ----------------------
  function carry(root) {
    let n = 8;
    let a = 0b01111111;
    let b = 0b00000001;
    const bitsSel = h('select', { 'aria-label': '位数' }, [4, 8, 16].map((k) => h('option', { value: k, selected: k === n }, `${k} 位`)));
    const aIn = h('input', { type: 'number', min: 0, 'aria-label': 'a' });
    const bIn = h('input', { type: 'number', min: 0, 'aria-label': 'b' });
    const table = h('table', { class: 'w-table' });
    const note = h('div', { class: 'w-note' });
    const set = (x, y) => {
      const max = 2 ** n - 1;
      a = Math.max(0, Math.min(max, x | 0));
      b = Math.max(0, Math.min(max, y | 0));
      draw();
    };
    bitsSel.addEventListener('change', () => { n = +bitsSel.value; set(a, b); });
    aIn.addEventListener('input', () => set(+aIn.value, b));
    bIn.addEventListener('input', () => set(a, +bIn.value));
    root.append(
      h('div', { class: 'w-title' }, '逐位计算进位：进位函数与前缀'),
      h('div', { class: 'w-controls' },
        bitsSel,
        h('label', {}, h('i', {}, 'a'), ' = ', aIn),
        h('label', {}, h('i', {}, 'b'), ' = ', bIn),
        h('button', { type: 'button', onclick: () => set(2 ** (n - 1) - 1, 1) }, '最长进位链'),
        h('button', { type: 'button', onclick: () => set(Math.floor(Math.random() * 2 ** n), Math.floor(Math.random() * 2 ** n)) }, '随机')),
      h('div', { class: 'w-scroll' }, table),
      note);

    function draw() {
      aIn.value = a;
      bIn.value = b;
      aIn.max = bIn.max = 2 ** n - 1;
      const bit = (x, i) => (x >> i) & 1;
      const rows = [];
      let G = 0;
      let P = 1;
      const cols = [];
      for (let i = 0; i < n; i++) {
        const ai = bit(a, i);
        const bi = bit(b, i);
        const g = ai & bi;
        const p = ai ^ bi;
        const cin = i === 0 ? 0 : cols[i - 1].G;
        if (i === 0) { G = g; P = p; } else { G = g | (p & G); P = p & P; }
        cols.push({ ai, bi, g, p, G, P, cin, s: p ^ cin });
      }
      // The longest run "generate, then propagate…" that a carry actually travels.
      let best = null;
      for (let j = 0; j < n; j++) {
        if (!cols[j].g) continue;
        let k = j;
        while (k + 1 < n && cols[k + 1].p) k++;
        if (!best || k - j > best[1] - best[0]) best = [j, k];
      }
      const inChain = (i) => best && i >= best[0] && i <= best[1] + 1 && i < n;
      const fn = (c) => (c.g ? h('span', { class: 'w-g' }, h('u', {}, '1')) : c.p ? h('span', { class: 'w-p' }, 'id') : h('span', { class: 'w-k' }, h('u', {}, '0')));
      const name = (c) => (c.g ? '产生' : c.p ? '传递' : '吸收');
      const head = h('tr', {}, h('th', {}, h('i', {}, 'i')), cols.map((_, i) => h('th', { class: inChain(i) ? 'w-chain' : null }, String(i))));
      const row = (label, cell, attrs) => h('tr', {}, h('td', {}, label), cols.map((c, i) => h('td', { class: [inChain(i) ? 'w-chain' : '', attrs?.(c, i) ?? ''].join(' ').trim() || null }, cell(c, i))));
      const flip = (which, i) => () => (which === 'a' ? set(a ^ (1 << i), b) : set(a, b ^ (1 << i)));
      rows.push(head,
        row(sub('a', 'i'), (c, i) => h('span', { class: 'w-bit', title: '点击翻转', role: 'button', tabindex: 0, onclick: flip('a', i) }, String(c.ai))),
        row(sub('b', 'i'), (c, i) => h('span', { class: 'w-bit', title: '点击翻转', role: 'button', tabindex: 0, onclick: flip('b', i) }, String(c.bi))),
        row(h('span', {}, '(', sub('g', 'i'), ', ', sub('p', 'i'), ')'), (c) => pair(c.g, c.p)),
        row(h('span', {}, 'φ', h('sub', {}, h('i', {}, 'i'))), (c) => h('span', { title: name(c) }, fn(c))),
        row(h('span', {}, '(', sub('G', 'i'), ', ', sub('P', 'i'), ')'), (c) => pair(c.G, c.P)),
        row(h('span', {}, h('i', {}, 'c'), h('sub', {}, h('i', {}, 'i'), '+1'), ' = ', sub('G', 'i')), (c) => String(c.G)),
        row(h('span', {}, sub('s', 'i'), ' = ', sub('p', 'i'), ' ⊕ ', sub('c', 'i')), (c) => h('b', {}, String(c.s))));
      table.replaceChildren(...rows);
      const sum = cols.reduce((acc, c, i) => acc + c.s * 2 ** i, 0) + cols[n - 1].G * 2 ** n;
      note.replaceChildren(
        `${a} + ${b} = ${a + b}；由各位的 `, sub('s', 'i'), ' 与 ', sub('c', n), ` 读出 ${sum}${sum === a + b ? '，一致' : '（不一致！）'}。`,
        best
          ? h('span', {}, ` 进位传得最远的一段：第 ${best[0]} 位产生进位${best[1] > best[0] ? `，第 ${best[0] + 1}–${best[1]} 位传递` : ''}${best[1] + 1 < n ? `，到第 ${best[1] + 1} 位为止` : '，一直送出最高位'}（底色标出）。行波进位要等它逐位走完；前缀网络在 log ${n} = ${Math.log2(n)} 级节点内算出全部 `, sub('G', 'i'), '。')
          : ' 没有任何一位产生进位。',
        ' 点击 ', sub('a', 'i'), ' 或 ', sub('b', 'i'), ' 翻转该位。');
    }
    draw();
  }

  // ---- prefix: step through Sklansky / Kogge–Stone / Brent–Kung level by level (§1.8) ------------
  const NETS = {
    serial: { name: '逐个计算', ops: (n) => Array.from({ length: n - 1 }, (_, k) => [k + 1, k + 1, k]) },
    sklansky: {
      name: 'Sklansky',
      ops: (n) => {
        const ops = [];
        for (let k = 1; 2 ** k <= n; k++) {
          for (let base = 0; base < n; base += 2 ** k) {
            const mid = base + 2 ** (k - 1);
            for (let i = mid; i < base + 2 ** k; i++) ops.push([k, i, mid - 1]);
          }
        }
        return ops;
      },
    },
    koggeStone: {
      name: 'Kogge–Stone',
      ops: (n) => {
        const ops = [];
        for (let k = 1; 2 ** (k - 1) < n; k++) for (let i = 2 ** (k - 1); i < n; i++) ops.push([k, i, i - 2 ** (k - 1)]);
        return ops;
      },
    },
    brentKung: {
      name: 'Brent–Kung',
      ops: (n) => {
        const m = Math.log2(n);
        const ops = [];
        for (let k = 1; k <= m; k++) for (let i = 2 ** k - 1; i < n; i += 2 ** k) ops.push([k, i, i - 2 ** (k - 1)]);
        for (let k = m - 1; k >= 1; k--) {
          for (let r = 1; r * 2 ** k + 2 ** (k - 1) - 1 < n; r++) ops.push([2 * m - 1 - k, r * 2 ** k + 2 ** (k - 1) - 1, r * 2 ** k - 1]);
        }
        return ops;
      },
    },
  };

  function prefix(root) {
    let net = 'koggeStone';
    let n = 8;
    let level = 0;
    let timer = 0;
    const netSel = h('select', { 'aria-label': '网络' }, Object.entries(NETS).map(([k, v]) => h('option', { value: k, selected: k === net }, v.name)));
    const nSel = h('select', { 'aria-label': 'n' }, [8, 16].map((k) => h('option', { value: k, selected: k === n }, `n = ${k}`)));
    const prev = h('button', { type: 'button', onclick: () => go(level - 1) }, '◀ 上一级');
    const next = h('button', { type: 'button', onclick: () => go(level + 1) }, '下一级 ▶');
    const play = h('button', { type: 'button', onclick: () => toggle() }, '播放');
    const box = h('div', { class: 'w-scroll' });
    const note = h('div', { class: 'w-note' });
    netSel.addEventListener('change', () => { net = netSel.value; go(0); });
    nSel.addEventListener('change', () => { n = +nSel.value; go(0); });
    root.append(
      h('div', { class: 'w-title' }, '前缀网络逐级演示：每个位置保存哪个区间'),
      h('div', { class: 'w-controls' }, netSel, nSel, prev, next, play),
      box, note);

    let ops = [];
    let depth = 0;
    function toggle() {
      if (timer) { clearInterval(timer); timer = 0; play.textContent = '播放'; return; }
      if (level >= depth) go(0);
      play.textContent = '暂停';
      timer = setInterval(() => { if (level >= depth) toggle(); else go(level + 1); }, 900);
    }
    function go(L) {
      ops = NETS[net].ops(n);
      depth = Math.max(0, ...ops.map((o) => o[0]));
      level = Math.max(0, Math.min(depth, L));
      draw();
    }
    // Interval held at each position after each level, computed by simulation.
    function simulate() {
      const states = [Array.from({ length: n }, (_, i) => [i, i])];
      for (let L = 1; L <= depth; L++) {
        const cur = states[L - 1].map((x) => x.slice());
        for (const [lv, i, j] of ops) {
          if (lv !== L) continue;
          const left = states[L - 1][j];
          const right = states[L - 1][i];
          cur[i] = left[1] + 1 === right[0] ? [left[0], right[1]] : [NaN, NaN];
        }
        states.push(cur);
      }
      return states;
    }
    function draw() {
      const states = simulate();
      const dx = n === 8 ? 58 : 36;
      const rowH = depth > 8 ? 26 : 36;
      const left = 64;
      const top = 30;
      const X = (i) => left + dx * i;
      const Y = (L) => top + rowH * L;
      const bottom = Y(depth) + 14;
      const W = X(n - 1) + 40;
      const Hh = bottom + (n === 16 ? 52 : 34);
      const s = svg('svg', { class: 'w-svg', viewBox: `0 0 ${W} ${Hh}`, width: W, role: 'img', 'aria-label': `${NETS[net].name} 网络，n = ${n}` });
      for (let i = 0; i < n; i++) {
        s.append(svg('line', { class: 'col', x1: X(i), y1: top - 8, x2: X(i), y2: bottom }));
        s.append(svg('text', { x: X(i), y: top - 14, 'text-anchor': 'middle', 'font-size': 12 }, `x${i}`));
      }
      for (let L = 1; L <= depth; L++) {
        s.append(svg('text', { class: 'lvl', x: left - 22, y: Y(L) + 4, 'text-anchor': 'end' }, `第 ${L} 级`));
      }
      if (level > 0) s.append(svg('rect', { x: left - 60, y: Y(level) - rowH / 2, width: W - left + 70, height: rowH, fill: 'var(--hit)', opacity: 0.35 }));
      const tap = rowH * 0.62;
      for (const [L, i, j] of ops) {
        const on = L === level;
        const done = L <= level;
        const g = svg('g', { opacity: done ? 1 : 0.28 });
        g.append(svg('line', { class: `wire${on ? ' on' : ''}`, x1: X(j), y1: Y(L) - tap, x2: X(i), y2: Y(L) }));
        g.append(svg('circle', { cx: X(j), cy: Y(L) - tap, r: 2.4, fill: on ? 'var(--w-on)' : 'var(--muted)' }));
        const a = states[L - 1][j];
        const b = states[L - 1][i];
        const c = states[L][i];
        g.append(svg('circle', { class: `node${on ? ' on' : done ? ' done' : ''}`, cx: X(i), cy: Y(L), r: 6 },
          svg('title', {}, `第 ${L} 级：位置 ${i} 取位置 ${j} 的值放在左边\n[${a}] ⊗ [${b}] = [${c}]`)));
        s.append(g);
      }
      const held = states[level];
      held.forEach(([lo, hi], i) => {
        const full = lo === 0 && hi === i;
        const y = bottom + 16 + (n === 16 && i % 2 ? 16 : 0);
        s.append(svg('text', { class: `seg${full ? ' full' : ''}`, x: X(i), y, 'text-anchor': 'middle' }, lo === hi ? `${lo}` : `${lo}–${hi}`));
      });
      box.replaceChildren(s);
      const used = ops.filter((o) => o[0] <= level).length;
      const now = ops.filter((o) => o[0] === level).length;
      const fullCount = held.filter(([lo, hi], i) => lo === 0 && hi === i).length;
      prev.disabled = level === 0;
      next.disabled = level === depth;
      note.replaceChildren(
        `${NETS[net].name}，n = ${n}：深度 ${depth}，共 ${ops.length} 个节点。`,
        level === 0 ? '第 0 级：位置 i 保存 xᵢ 本身。' : `第 ${level} 级有 ${now} 个节点（红色），到此共用 ${used} 个。`,
        ` 底部一行是此时各位置保存的区间 [j, i]（写作 j–i），红色的已是完整前缀（${fullCount}/${n}）。把鼠标停在节点上可以看它合并的两个区间。`);
    }
    go(0);
  }

  const WIDGETS = { carry, prefix };
  for (const el of document.querySelectorAll('[data-widget]')) WIDGETS[el.dataset.widget]?.(el);
})();
