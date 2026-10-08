// Interactive figures, mounted on <div class="widget" data-widget="NAME"> by name. Loaded by reader.js.
(() => {
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

  const M = window.bookWidgetModels;
  const fmt = (x, digits = 3) => Number.isFinite(x) ? String(Number(x.toFixed(digits))) : x === -Infinity ? '−∞' : '未定义';
  function choice(name, options, initial, draw) {
    const input = h('select', { 'aria-label': name }, options.map(o => {
      const [value, label] = Array.isArray(o) ? o : [o, String(o)];
      return h('option', { value, selected: value === initial }, label);
    }));
    input.addEventListener('change', draw);
    return { input, label: h('label', {}, `${name} `, input), value: () => input.value };
  }
  function panel(root, title) {
    const controls = h('div', { class: 'w-controls' });
    const body = h('div', { class: 'w-scroll' });
    const note = h('div', { class: 'w-note', 'aria-live': 'polite' });
    root.append(h('div', { class: 'w-title' }, title), controls, body, note);
    return { controls, body, note };
  }
  function table(headers, rows) {
    return h('table', { class: 'w-table' }, h('thead', {}, h('tr', {}, headers.map(x => h('th', { scope: 'col' }, String(x))))),
      h('tbody', {}, rows.map(row => h('tr', {}, row.map(x => h('td', {}, String(x)))))));
  }
  function stepper(controls, max, draw) {
    let step = 0;
    const prev = h('button', { type: 'button', onclick: () => set(step - 1) }, '上一步');
    const next = h('button', { type: 'button', onclick: () => set(step + 1) }, '下一步');
    const status = h('span');
    const set = x => { step = Math.max(0, Math.min(max(), x)); prev.disabled = step === 0; next.disabled = step === max(); status.textContent = `${step}/${max()}`; draw(); };
    controls.append(prev, next, h('button', { type: 'button', onclick: () => set(0) }, '重置'), status);
    return { value: () => step, set };
  }
  function timePlot(lanes, bars, end) {
    const left = 115, width = 530, row = 45, top = 20, scale = width / Math.max(1, end);
    const s = svg('svg', { class: 'w-svg', viewBox: `0 0 670 ${top + lanes.length * row + 32}`, role: 'img', 'aria-label': '根据当前参数计算的时间线' });
    lanes.forEach((name, i) => {
      s.append(svg('text', { x: left - 10, y: top + i * row + 19, 'text-anchor': 'end', 'font-size': 12 }, name));
      s.append(svg('line', { x1: left, x2: left + width, y1: top + i * row + 33, y2: top + i * row + 33, class: 'col' }));
    });
    for (const [lane, from, to, name] of bars) {
      const x = left + from * scale, y = top + lane * row;
      const g = svg('g', {}, svg('rect', { x, y, width: (to - from) * scale, height: 30, fill: 'var(--hit)', stroke: 'var(--link)', 'stroke-width': 1 }),
        svg('title', {}, `${name}：[${fmt(from)}, ${fmt(to)})`));
      if ((to - from) * scale > 24) g.append(svg('text', { x: x + (to - from) * scale / 2, y: y + 19, 'text-anchor': 'middle', 'font-size': 11 }, name));
      s.append(g);
    }
    for (let k = 0; k <= 4; k++) s.append(svg('text', { x: left + width * k / 4, y: top + lanes.length * row + 16, 'text-anchor': 'middle', 'font-size': 11 }, fmt(end * k / 4)));
    return s;
  }
  function layout(root) {
    const p = panel(root, '行、列与存储体 · Layout / bank conflict');
    const mode = choice('布局', [['row', '行优先'], ['xor', 'XOR 交错']], 'row', draw);
    const axis = choice('同时读取', [['row', '一行'], ['col', '一列']], 'col', draw);
    const index = choice('行或列号', [0, 1, 2, 3], 1, draw);
    p.controls.append(mode.label, axis.label, index.label);
    function draw() {
      const n = +index.value(), xor = mode.value() === 'xor';
      const banks = Array.from({ length: 4 }, (_, i) => M.bank(axis.value() === 'row' ? n : i, axis.value() === 'row' ? i : n, xor));
      p.body.replaceChildren(table(['行 / 列', '0', '1', '2', '3'], Array.from({ length: 4 }, (_, r) => [r, ...Array.from({ length: 4 }, (_, c) => `${M.bank(r, c, xor)}${axis.value() === 'row' ? r === n ? ' ●' : '' : c === n ? ' ●' : ''}`)])));
      const counts = banks.map(b => banks.filter(x => x === b).length);
      p.note.textContent = `所读存储体：${banks.join(', ')}；最大冲突度 ${Math.max(...counts)}。假设四个单端口存储体、每个逻辑元素一个位置；● 标出本次访问。`;
    }
    draw();
  }
  function pipeline(root) {
    const p = panel(root, '输入槽与延迟隐藏 · Software pipeline');
    const latency = choice('读入延迟 λ', [1, 3, 7], 3, draw), compute = choice('计算时间 τ', [1, 2, 4], 2, draw), slots = choice('输入槽数', [1, 2, 3, 5, 8], 3, draw);
    p.controls.append(latency.label, compute.label, slots.label);
    function draw() {
      const rows = M.pipeline({ latency: +latency.value(), compute: +compute.value(), slots: +slots.value() });
      const end = rows.at(-1).end;
      const n = +slots.value();
      const bars = rows.flatMap(x => [[x.slot, x.load, x.ready, `L${x.i}`],
        ...(x.start > x.ready ? [[x.slot, x.ready, x.start, `等${x.i}`]] : []),
        [x.slot, x.start, x.end, `用${x.i}`], [n, x.start, x.end, `C${x.i}`]]);
      p.body.replaceChildren(timePlot([...Array.from({ length: n }, (_, i) => `输入槽 ${i}`), '计算单元'], bars, end),
        table(['块', '槽', '发起', '就绪', '开始算', '释放槽'], rows.map(x => [x.i, x.slot, x.load, x.ready, x.start, x.end])));
      const min = 1 + Math.ceil(+latency.value() / +compute.value());
      p.note.textContent = `8 块完成于 ${end}；维持每 ${compute.value()} 时间一块，定理给出的槽数至少 ${min}。本模型读入服务间隔为 1、各请求可在途重叠；槽一直占用到计算结束。表按槽复用约束实际排程，未计写回。`;
    }
    draw();
  }
  function softmax(root) {
    const p = panel(root, '逐块合并状态 · Online softmax / attention');
    const block = choice('每块元素', [1, 2, 3], 2, () => steps.set(0));
    const mask = choice('掩码', [['none', '全部有效'], ['last', '末两项无效'], ['all', '全部无效']], 'none', () => steps.set(0));
    p.controls.append(block.label, mask.label);
    const x = [2, -1, 4, 0, 3, 1], v = [10, 20, 30, 40, 50, 60];
    const history = () => M.online(x, v, +block.value(), mask.value() === 'all' ? [0,1,2,3,4,5] : mask.value() === 'last' ? [4,5] : []);
    const steps = stepper(p.controls, () => history().length, draw);
    function draw() {
      const hist = history().slice(0, steps.value());
      p.body.replaceChildren(table(['块范围', '局部最大', '旧状态缩放', 'm', 'ℓ', 'u', 'u/ℓ'], hist.map(x => [
        `[${x.from},${x.to})`, fmt(x.local.m), x.scale === null ? '空状态' : fmt(x.scale), fmt(x.state.m), fmt(x.state.l), fmt(x.state.u), x.state.count ? fmt(x.state.u / x.state.l) : '无有效项'
      ])));
      p.note.textContent = `固定 logits = (${x.join(', ')})，值 = (${v.join(', ')})。空状态直接合并，避免 −∞−(−∞)；全无效行没有归一化输出。步数 ${steps.value()}；分块改变合并次序，实数结果相同，浮点末位允许差异。`;
    }
    steps.set(0);
  }
  function kv(root) {
    const p = panel(root, 'KV 容量如何增长 · KV cache');
    const batch = choice('序列数 B', [1, 4, 16], 1, draw), length = choice('长度 T', [1024, 8192, 32768], 8192, draw), heads = choice('KV 头数', [1, 8, 32], 8, draw);
    p.controls.append(batch.label, length.label, heads.label);
    function draw() {
      const bytes = 2 * 32 * +batch.value() * +length.value() * +heads.value() * 128 * 2;
      p.body.replaceChildren(table(['因子', '取值'], [['K 与 V', 2], ['层数', 32], ['B', batch.value()], ['T', length.value()], ['KV 头', heads.value()], ['头维度', 128], ['每数 bytes', 2], ['总 GiB', fmt(bytes / 2 ** 30)]]));
      p.note.textContent = '只计有效 KV 载荷，不计页池空隙、量化尺度、元数据、权重和工作区；改变 KV 头数需模型本身使用相应的 GQA/MQA 结构。';
    }
    draw();
  }
  function parallel(root) {
    const p = panel(root, '微批量与空泡 · Pipeline parallelism');
    const stages = choice('阶段数 p', [2, 4, 8], 4, draw), batches = choice('微批量 m', [2, 4, 8, 16], 8, draw);
    p.controls.append(stages.label, batches.label);
    function draw() {
      const n = +stages.value(), m = +batches.value(), end = n + m - 1;
      p.body.replaceChildren(timePlot(Array.from({ length: n }, (_, i) => `阶段 ${i}`), Array.from({ length: n }, (_, i) => Array.from({ length: m }, (_, j) => [i, i+j, i+j+1, String(j)])).flat(), end));
      p.note.textContent = `等长的单向抽象任务流水线：总时间 ${end}，空泡比例 (p−1)/(m+p−1) = ${fmt(100*(n-1)/end)}%。这不是具体训练的完整前后向排程，也未计通信。`;
    }
    draw();
  }
  function trace(root) {
    const p = panel(root, '抽象类型与编译缓存 · Tracing / jit cache');
    const shape = choice('长度', [4, 8, 16], 4, () => {}), dtype = choice('dtype', ['f32','bf16'], 'f32', () => {}), staticValue = choice('静态参数 a', [1, 2, 3], 2, () => {});
    const cache = new Map(); let calls = 0;
    const value = choice('设备输入值', [1, 5, 9], 1, () => {});
    p.controls.append(shape.label, dtype.label, staticValue.label, value.label,
      h('button', { type: 'button', onclick: run }, '调用 f(x)=a·x'),
      h('button', { type: 'button', onclick: () => { cache.clear(); calls=0; p.body.replaceChildren(); p.note.textContent='缓存已清空。'; } }, '清空缓存'));
    function run() {
      const key = `${dtype.value()}[${shape.value()}], a=${staticValue.value()}`;
      const hit = cache.has(key); if (!hit) cache.set(key, cache.size+1); calls++;
      p.body.replaceChildren(table(['缓存键', '示意程序编号'], [...cache].map(([k, n]) => [k, n])));
      p.note.textContent = `第 ${calls} 次调用：${hit ? '复用已有程序' : '建立一个新程序'}；a·x = ${+staticValue.value()*+value.value()}。这是固定函数、目标与其他配置的教学缓存模型；实际 JAX 缓存还受函数身份、分片及编译选项等影响。改变输入值不改变此处的键。`;
    }
    run();
  }
  function sharding(root) {
    const p = panel(root, '全局形状与每设备容量 · PartitionSpec');
    const spec = choice('分片', [['xy','P(x,y)'], ['x','P(x,None)'], ['y','P(None,y)'], ['','P(None,None)']], 'xy', draw);
    p.controls.append(spec.label);
    function draw() {
      const shards = Array.from({length:2}, (_,x)=>Array.from({length:4}, (_,y)=>({x,y,...M.shard(spec.value(),x,y)}))).flat();
      p.body.replaceChildren(table(['设备 (x,y)', '行区间', '列区间', '元素数'], shards.map(s=>[`(${s.x},${s.y})`, `[${s.rows})`, `[${s.cols})`, s.elements])));
      const total = shards.reduce((s,a)=>s+a.elements,0);
      p.note.textContent = `全局 8×16 = 128 个元素；mesh 2×4；物理总量 ${total}，复制倍数 ${total/128}。None 让未使用的网格轴复制数据；各区间左闭右开。`;
    }
    draw();
  }
  function blockmap(root) {
    const p = panel(root, '循环次序决定驻留与读取 · Grid / index_map');
    const order = choice('grid 次序', ['ijk','jik','kij'], 'ijk', () => steps.set(0));
    p.controls.append(order.label);
    const steps = stepper(p.controls, () => 12, draw);
    function draw() {
      const model = M.grid(order.value());
      p.body.replaceChildren(table(['步', '(i,j,k)', 'A 块', 'B 块', '输出块'], model.points.slice(0,steps.value()).map((p,t)=>[t,`(${p.i},${p.j},${p.k})`,`(${p.i},${p.k})`,`(${p.k},${p.j})`,`(${p.i},${p.j})`])));
      p.note.textContent = `完整遍历 A 读 ${model.aLoads} 次，B 读 ${model.bLoads} 次；${model.contiguous ? '同一输出的 K 步连续，仍须把 K 声明为顺序维' : '同一输出被分成多个不连续区间，不能直接套用自动驻留累加'}。这是定义 16.1 的教学模型，未计多核心或尾块。`;
    }
    steps.set(0);
  }
  function rng(root) {
    const p = panel(root, '逻辑计数器与重放 · Counter-based PRNG');
    const cols = choice('全局列数 C', [128,256], 128, draw), tileRows = choice('行块大小', [4,8], 8, draw), numbering = choice('计数方式', [['global','全局编号'], ['local','错误：每块从零编号']], 'global', draw);
    p.controls.append(cols.label,tileRows.label,numbering.label);
    function draw() {
      const rows = Array.from({length:16},(_,i)=>[i,Math.floor(i/+tileRows.value()), numbering.value()==='global' ? i*+cols.value() : (i%+tileRows.value())*+cols.value()]);
      p.body.replaceChildren(table(['全局行 i', '行块', '第 0 列的计数器 c'],rows));
      const unique = new Set(rows.map(r=>r[2])).size;
      p.note.textContent = `密钥固定，r=F(k,c)。16 个逻辑位置只有 ${unique} 个不同计数器；${unique===16?'改行块大小仍保持位置身份':'多个块重用同一随机样本'}。这里只显示寻址，不执行或冒充某种 PRNG；改变全局列数会改变第二行起的 c。`;
    }
    draw();
  }
  function tilebudget(root) {
    const gpu = root.dataset.backend === 'gpu';
    const p = panel(root, gpu ? '分块复用与共享容量 · GEMM tile budget' : '分块强度与 VMEM 容量 · GEMM tile budget');
    const bm=choice('bM',gpu?[64,128,256]:[128,256,512],128,draw), bn=choice('bN',gpu?[64,128,256,512]:[128,256,512],128,draw), bk=choice('bK',gpu?[32,64,128,256]:[128,512,1024,4096],gpu?32:128,draw), slots=choice('输入缓冲数',[2,3],2,draw);
    p.controls.append(bm.label,bn.label,bk.label,slots.label);
    function draw() {
      const b=M.tile(+bm.value(),+bn.value(),+bk.value(),+slots.value());
      p.body.replaceChildren(table(['资源', 'KiB'], gpu ? [['共享内存的输入槽',fmt(b.input/1024)],['整个输出 tile 的 f32 累加值',fmt(b.accumulator/1024)]] : [['VMEM 输入槽',fmt(b.input/1024)],['VMEM 输出双/三槽',fmt(b.output/1024)],['f32 scratch 累加器',fmt(b.accumulator/1024)],['显式总量',fmt(b.bytes/1024)]]));
      p.note.textContent = gpu ? `只计 bf16 输入，输入读取强度 ${fmt(b.intensity)} FLOP/byte（未计输出）。假设共享上限 64 KiB：${b.input<=65536?'输入槽通过总量检查':'输入槽超限'}。累加值分配到线程寄存器或矩阵存储，不能与共享字节合并；未计硬件片段与分配粒度。` : `bf16 输入/输出、f32 累加器，输入读取强度 ${fmt(b.intensity)} FLOP/byte（未计输出）。按 v4 的每 TC 16 MiB：${b.bytes<=16*2**20?'显式总量可放入':'超出容量'}；还需给布局填充、临时值、编译器工作区留余量。`;
    }
    draw();
  }
  function issue(root) {
    const p=panel(root,'延迟与发射间隔 · Latency / initiation interval');
    const count=choice('操作数',[4,8,16],4,draw), mode=choice('依赖',[['independent','互相独立'],['chain','串行依赖']], 'independent',draw);
    p.controls.append(count.label,mode.label);
    function draw() {
      const rows=M.issue(+count.value(),7,2,mode.value()==='chain');
      p.body.replaceChildren(timePlot(rows.map(x=>`操作 ${x.i}`),rows.map(x=>[x.i,x.start,x.ready,String(x.i)]),rows.at(-1).ready),table(['操作','发射','结果就绪'],rows.map(x=>[x.i,x.start,x.ready])));
      p.note.textContent=`L=7，I=2，最后结果在 ${rows.at(-1).ready}。这是单单元的简化模型，忽略 VIF、槽冲突、载入、取回和后端的具体排程；独立操作按间隔推进，依赖链按 max(L,I) 推进。`;
    }
    draw();
  }
  function ring(root) {
    const p=panel(root,'沿环转发源块 · Ring all-gather');
    const count=choice('设备数',[3,4,8],4,()=>steps.set(0)); p.controls.append(count.label);
    const steps=stepper(p.controls,()=>+count.value()-1,draw);
    function draw() {
      const n=+count.value(),round=steps.value(),state=M.ring(n,round);
      p.body.replaceChildren(table(['设备','已有全局块号','本轮向右发的块'],state.map((blocks,d)=>[d,blocks.join(', '),round<n-1?(d-round+n)%n:'完成'])));
      p.note.textContent=`右邻居 (d+1) mod ${n}，已执行 ${round} 轮；每轮转发上一轮新收的源块，${n-1} 轮后每设备有全部块。这里只演示块身份，实际实现还要证明发送、接收、消费与覆盖的完成边。`;
    }
    steps.set(0);
  }
  function coalescing(root) {
    const p=panel(root,'地址、段数与有效字节 · Coalescing');
    const stride=choice('元素步长',[1,2,8,32],1,draw),offset=choice('起点 bytes',[0,4,28],0,draw);p.controls.append(stride.label,offset.label);
    function draw() {
      const m=M.segments(+stride.value(),+offset.value());
      p.body.replaceChildren(table(['lane','字节地址','32-byte 段'],m.addresses.map((a,i)=>[i,a,Math.floor(a/32)])));
      p.note.textContent=`32 lane 各读一个 f32，有用载荷 128 bytes；触及 ${m.touched.length} 段，理想传输 ${32*m.touched.length} bytes，效率 ${fmt(100*m.efficiency)}%。本模型不计缓存、重放和其他层次的事务。`;
    }
    draw();
  }
  function occupancy(root) {
    const p=panel(root,'资源上限取最小值 · Occupancy');
    const registers=choice('每线程寄存器',[32,64,80,128],64,draw),threads=choice('每块线程',[128,256,512],256,draw),shared=choice('每块共享 KiB',[16,64,80,112],80,draw);p.controls.append(registers.label,threads.label,shared.label);
    function draw() {
      const m=M.occupancy({registers:+registers.value(),threads:+threads.value(),sharedKiB:+shared.value()});
      p.body.replaceChildren(table(['限制','最多块'],['寄存器','共享内存','线程 / warp','块数硬上限'].map((x,i)=>[x,m.limits[i]])));
      p.note.textContent=`假设每 SM：65536 寄存器、224 KiB 共享、64 warp、16 块上限。算术上界 ${m.blocks} 块、${m.warps} warp、${fmt(m.fraction*100)}%。忽略分配粒度与其他限制；高占用率不单独保证更快。`;
    }
    draw();
  }
  function gpupipeline(root) {
    const p=panel(root,'满与空保护两条边 · Producer / consumer');
    const steps=stepper(p.controls,()=>4,draw);
    function draw() {
      const states=['生产者可写','异步写入中','写入完成，消费者可读','消费者 / MMA 读者在途','所有读者完成，可复用'];
      p.body.replaceChildren(table(['阶段','槽状态','此时复用'],states.slice(0,steps.value()+1).map((s,i)=>[i,s,i===0||i===4?'允许':'不允许'])));
      p.note.textContent=`当前：${states[steps.value()]}。满信号证明写完；空信号证明旧读者全读完。MMA 提交返回不等于已经读完共享 tile；实际 barrier/phase 与代理可见性按目标 ISA 建立。`;
    }
    steps.set(0);
  }
  const WIDGETS = { carry, prefix, layout, pipeline, softmax, kv, parallel, trace, sharding, blockmap, rng, tilebudget, issue, ring, coalescing, occupancy, gpupipeline };
  for (const el of document.querySelectorAll('[data-widget]')) {
    if (!WIDGETS[el.dataset.widget]) throw new Error(`Unknown widget: ${el.dataset.widget}`);
    WIDGETS[el.dataset.widget](el);
  }
  window.bookWidgets = true;
})();
