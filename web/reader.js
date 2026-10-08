// Reading aids for the pages rendered by server.mjs. The page reads fine without this script; with it:
// an outline of sections and results, previews of references and terms on hover (tap on phones),
// a glossary, "cited by" lists, foldable proofs, English names after terms, reader settings
// (font, size, width, line height, background), interactive figures.
(() => {
  const main = document.querySelector('main');
  if (!main) return;
  const widgetSrc = new URL('widgets.js', document.currentScript?.src || new URL('/assets/reader.js', location.href)).href;
  // This page's chapter file (as in data-ref="01-x.md#…"), also when the page is an exported copy.
  const file = main.dataset.file || decodeURIComponent(location.pathname.split('/').pop() || 'README.md');
  const root = document.documentElement;
  const store = {
    get: (k) => { try { return localStorage.getItem(k); } catch { return null; } },
    set: (k, v) => { try { if (v === null) localStorage.removeItem(k); else localStorage.setItem(k, v); } catch { /* private mode */ } },
  };
  const h = (tag, attrs = {}, ...kids) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) e.setAttribute(k, v);
    e.append(...kids.flat().filter((k) => k !== null && k !== undefined));
    return e;
  };
  const LABEL = /^(定义|命题|定理|引理|推论|例|注|习题|图)\s?(\d+|[A-E])\.(\d+)/;

  // Pages from an older server lack the head script and font links that server.mjs now writes: add them here.
  if (!document.querySelector('link[href*="fonts.googleapis.com"]')) {
    for (const href of ['https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&display=swap',
      'https://cdn.jsdelivr.net/npm/lxgw-wenkai-screen-webfont@1.7.0/lxgwwenkaiscreen.css']) {
      document.head.append(h('link', { rel: 'stylesheet', href }));
    }
    for (const k of ['font', 'width', 'lh', 'theme']) { const v = store.get(`rd-${k}`); if (v) root.setAttribute(`data-${k}`, v); }
    if (store.get('rd-fs')) root.style.setProperty('--fs', `${store.get('rd-fs')}px`);
  }

  // ---- toolbar and panels -----------------------------------------------------------------------
  let topbar = document.querySelector('.topbar');
  if (!topbar) document.body.prepend((topbar = h('header', { class: 'topbar' })));
  const tools = h('div', { class: 'tools' });
  topbar.append(tools);
  const tool = (label, title) => {
    const b = h('button', { type: 'button', title }, label);
    tools.append(b);
    return b;
  };
  // A remembered on/off switch that sets a class on <body>.
  const toggle = (key, cls, button, labels, onChange) => {
    let on = store.get(key) === '1';
    const apply = () => {
      document.body.classList.toggle(cls, on);
      button.setAttribute('aria-pressed', String(on));
      if (labels) button.textContent = labels[on ? 1 : 0];
      onChange?.(on);
    };
    button.addEventListener('click', () => { on = !on; store.set(key, on ? '1' : null); apply(); });
    apply();
  };
  // Panels drop down from the top bar; one at a time, closed by the button, Esc or a click elsewhere.
  const panels = [];
  const closePanels = (except) => {
    for (const p of panels) if (p.box !== except) { p.box.hidden = true; p.button.setAttribute('aria-expanded', 'false'); }
  };
  const panel = (label, title, cls, build) => {
    const button = tool(label, title);
    const close = h('button', { type: 'button', class: 'panel-close', title: '关闭' }, '×');
    const box = h('aside', { class: `panel ${cls}`, 'aria-label': title, hidden: '' }, h('div', { class: 'panel-head' }, h('span', {}, label), close));
    build(box);
    document.body.append(box);
    button.setAttribute('aria-expanded', 'false');
    const entry = { box, button, onOpen: null };
    panels.push(entry);
    button.addEventListener('click', () => {
      const opening = box.hidden;
      closePanels(box);
      box.hidden = !opening;
      button.setAttribute('aria-expanded', String(opening));
      if (opening) entry.onOpen?.();
    });
    close.addEventListener('click', () => closePanels());
    return entry;
  };
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.panel, .tools, .pop')) closePanels();
  });

  // ---- proofs: the word 证明 folds the proof --------------------------------------------------
  const proofs = [...main.querySelectorAll('.proof')];
  for (const p of proofs) {
    const body = h('div', { class: 'proof-body' });
    body.append(...p.childNodes);
    const head = body.querySelector('.proof-head');
    if (head) {
      head.classList.add('proof-toggle');
      head.title = '折叠这段证明';
    }
    const line = h('p', { class: 'proof-fold-line' }, h('span', { class: 'proof-head proof-toggle', title: '展开这段证明' }, '证明'), '　（已折叠，点击展开）');
    p.append(line, body);
    p.addEventListener('click', (e) => { if (e.target.closest('.proof-toggle')) p.classList.toggle('folded'); });
  }
  if (proofs.length) {
    const b = tool('折叠证明', '折叠本章所有证明，只读定义、定理与例（初读时的"骨架"读法）');
    toggle('fold-proofs', 'fold-proofs', b, ['折叠证明', '展开证明'], (on) => proofs.forEach((p) => p.classList.toggle('folded', on)));
  }

  // The name in （…） after a label, with its math: "定义 1.30（运算 ⊗）" → "运算 ⊗".
  const titleOf = (e) => {
    const name = e.querySelector('.env-name');
    if (!name) return e.dataset.title || '';
    const c = name.cloneNode(true);
    const texts = [];
    const walker = document.createTreeWalker(c, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) if (!walker.currentNode.parentElement.closest('.katex')) texts.push(walker.currentNode);
    if (!texts.length || !texts[0].data.includes('（')) return '';
    texts[0].data = texts[0].data.slice(texts[0].data.indexOf('（') + 1);
    const last = texts.at(-1);
    last.data = last.data.replace(/）\s*$/, '');
    const span = h('span', {});
    span.append(...c.childNodes);
    return span;
  };

  // ---- outline: sections and the numbered results under each ----------------------------------
  const heads = [...main.querySelectorAll('h2[id]')];
  if (heads.length > 2) {
    const links = new Map();
    const outline = panel('大纲', '本章的节与编号条目', 'outline', (box) => {
      const list = h('ol');
      let sub = null;
      for (const e of main.querySelectorAll('h2[id], [data-kind]')) {
        if (e.tagName === 'H2') {
          const a = h('a', { href: `#${e.id}` }, e.textContent);
          links.set(e, a);
          sub = h('ol');
          list.append(h('li', { class: 'outline-sec' }, a, sub));
          continue;
        }
        const kind = e.dataset.kind;
        if (!sub || !['定义', '命题', '定理', '引理', '推论', '例'].includes(kind)) continue;
        const num = e.id.slice(kind.length + 1);
        sub.append(h('li', {}, h('a', { href: `#${e.id}` }, h('span', { class: 'outline-num' }, `${kind} ${num}`), ' ', titleOf(e))));
      }
      box.append(list);
      box.addEventListener('click', (e) => { if (e.target.closest('a')) closePanels(); });
    });
    // Mark the section being read, and scroll it into view in the outline.
    outline.onOpen = () => {
      let cur = heads[0];
      for (const e of heads) if (e.getBoundingClientRect().top < innerHeight * 0.3) cur = e;
      for (const [e, a] of links) a.classList.toggle('current', e === cur);
      links.get(cur)?.scrollIntoView({ block: 'center' });
    };
  }

  // ---- glossary -------------------------------------------------------------------------------
  const defs = [...main.querySelectorAll('.term-def')];
  const used = new Map();
  for (const t of main.querySelectorAll('.term')) if (!t.dataset.ref.startsWith(`${file}#`) && !used.has(t.textContent)) used.set(t.textContent, t);
  if (defs.length || used.size) {
    panel('术语', '本章的术语与英文对照', 'glossary', (box) => {
      const filter = h('input', { type: 'search', placeholder: '筛选：中文或英文', 'aria-label': '筛选术语' });
      box.append(filter);
      const section = (title, rows) => {
        if (!rows.length) return;
        box.append(h('div', { class: 'glossary-group' }, title), h('dl', {}, rows));
      };
      const secOf = (e) => {
        let n = e.closest('main > *') ?? e;
        while (n && n.tagName !== 'H2') n = n.previousElementSibling;
        return n?.textContent.match(/^\S+/)?.[0] ?? '';
      };
      section('本章定义', defs.map((d) => h('div', { class: 'g-row', 'data-key': `${d.textContent} ${d.dataset.en}`.toLowerCase() },
        h('dt', {}, h('a', { class: 'xref', href: `#${d.id}`, 'data-ref': `${file}#${d.id}` }, d.textContent)),
        h('dd', {}, d.dataset.en, h('span', { class: 'g-sec' }, secOf(d))))));
      section('前面各章的术语', [...used.values()].map((t) => h('div', { class: 'g-row', 'data-key': `${t.textContent} ${t.dataset.en}`.toLowerCase() },
        h('dt', {}, h('a', { class: 'xref', href: t.dataset.ref, 'data-ref': t.dataset.ref }, t.textContent)),
        h('dd', {}, t.dataset.en, h('span', { class: 'g-sec' }, t.dataset.where)))));
      filter.addEventListener('input', () => {
        const q = filter.value.trim().toLowerCase();
        box.querySelectorAll('.g-row').forEach((r) => { r.hidden = q && !r.dataset.key.includes(q); });
      });
    });
  }

  // ---- English names after terms ----------------------------------------------------------------
  for (const term of main.querySelectorAll('.term')) {
    if (/^\s*（[A-Za-z]/.test(term.nextSibling?.textContent ?? '')) term.classList.add('term-inline-en');
  }
  if (main.querySelector('.term, .term-def')) toggle('show-en', 'show-en', tool('英文', '在术语后面显示英文名'));

  // ---- reader settings ------------------------------------------------------------------------
  // Stored as rd-<key>; only values that differ from the default are kept (the page head applies them before paint).
  const CHOICES = [
    ['font', '字体', 'kai', [['song', '宋体', 'sample-song'], ['hei', '黑体', 'sample-hei'], ['kai', '楷体', 'sample-kai']]],
    ['width', '行宽', 'wide', [['narrow', '窄'], ['medium', '中'], ['wide', '宽'], ['full', '满']]],
    ['lh', '行距', 'normal', [['tight', '紧'], ['normal', '中'], ['loose', '松']]],
    ['theme', '背景', 'sepia', [['auto', '跟随系统'], ['light', '白'], ['sepia', '米黄'], ['dark', '黑']]],
  ];
  const FS = { min: 14, max: 24, def: 18 };
  const setChoice = (key, value, def) => {
    if (value === def) root.removeAttribute(`data-${key}`);
    else root.setAttribute(`data-${key}`, value);
    store.set(`rd-${key}`, value === def ? null : value);
  };
  const fontSize = () => Number(store.get('rd-fs')) || FS.def;
  const setFontSize = (px) => {
    const v = Math.min(FS.max, Math.max(FS.min, px));
    if (v === FS.def) root.style.removeProperty('--fs');
    else root.style.setProperty('--fs', `${v}px`);
    store.set('rd-fs', v === FS.def ? null : String(v));
    return v;
  };
  panel('版式', '字体、字号、行宽、行距与背景', 'settings', (box) => {
    const refresh = [];
    for (const [key, label, def, opts] of CHOICES) {
      const row = h('div', { class: 'opts' });
      for (const [value, text, cls] of opts) {
        const b = h('button', { type: 'button', class: cls ?? null }, text);
        b.addEventListener('click', () => { setChoice(key, value, def); refresh.forEach((f) => f()); barHeight(); });
        refresh.push(() => b.setAttribute('aria-pressed', String((root.getAttribute(`data-${key}`) ?? def) === value)));
        row.append(b);
      }
      box.append(h('div', { class: 'row' }, h('span', {}, label), row));
    }
    const size = h('span', { class: 'size' });
    const step = (d) => { size.textContent = setFontSize(fontSize() + d); barHeight(); };
    const minus = h('button', { type: 'button', title: '缩小' }, 'A−');
    const plus = h('button', { type: 'button', title: '放大' }, 'A+');
    minus.addEventListener('click', () => step(-1));
    plus.addEventListener('click', () => step(1));
    refresh.push(() => { size.textContent = fontSize(); });
    box.append(h('div', { class: 'row' }, h('span', {}, '字号'), h('div', { class: 'opts' }, minus, size, plus)));
    const reset = h('button', { type: 'button', class: 'reset' }, '恢复默认');
    reset.addEventListener('click', () => {
      for (const [key, , def] of CHOICES) setChoice(key, def, def);
      setFontSize(FS.def);
      refresh.forEach((f) => f());
      barHeight();
    });
    box.append(reset);
    refresh.forEach((f) => f());
  });

  // ---- previews -------------------------------------------------------------------------------
  const pages = new Map();
  const load = (f) => {
    if (!f || f === file) return Promise.resolve(document);
    if (!pages.has(f)) {
      pages.set(f, fetch(f)
        .then((r) => (r.ok ? r.text() : Promise.reject(new Error(r.statusText))))
        .then((t) => new DOMParser().parseFromString(t, 'text/html'))
        .catch(() => null));
    }
    return pages.get(f);
  };
  const chapterName = (doc) => doc.querySelector('h1')?.textContent ?? '';

  // The part of a page that a reference points to, cloned for display.
  function extract(doc, id, f) {
    const pieces = [];
    let head = '';
    const target = id ? doc.getElementById(id) : null;
    if (!id) {
      head = chapterName(doc);
      const box = doc.querySelector('main .position-box') ?? doc.querySelector('main > p');
      if (box) pieces.push(box);
    } else if (!target) {
      return null;
    } else if (id.startsWith('term-')) {
      const box = target.closest('.env, .box') ?? target.closest('li, td, th, p') ?? target;
      pieces.push(box);
      head = `${target.textContent}　${target.dataset.en ?? ''}`;
    } else if (/^H[1-6]$/.test(target.tagName)) {
      // A section: its opening paragraphs, which say what it does.
      head = target.textContent;
      for (let n = target.nextElementSibling; n && pieces.length < 2 && !/^H[1-6]$/.test(n.tagName); n = n.nextElementSibling) {
        if (!n.matches('p')) break;
        pieces.push(n);
      }
    } else {
      pieces.push(target);
      // An old-style numbered paragraph: take the display math or table that completes the statement.
      if (target.matches('p.label-para')) {
        for (let n = target.nextElementSibling; n && (n.matches('.katex-block, table') || (n.matches('p') && pieces.length > 1 && pieces.at(-1).matches('.katex-block') && !LABEL.test(n.textContent) && n.textContent.length < 120)); n = n.nextElementSibling) {
          pieces.push(n);
          if (n.matches('p:not(.katex-block)')) break;
        }
      }
    }
    const box = h('div', { class: 'pop-body' });
    for (const p of pieces) {
      const c = document.importNode(p, true);
      c.querySelectorAll('.cited, .proof-fold-line').forEach((n) => n.remove());
      c.removeAttribute('id');
      c.querySelectorAll('[id]').forEach((n) => n.removeAttribute('id'));
      if (f && f !== file) {
        c.querySelectorAll('a[href^="#"]').forEach((a) => a.setAttribute('href', f + a.getAttribute('href')));
        c.querySelectorAll('img[src]').forEach((img) => img.setAttribute('src', new URL(img.getAttribute('src'), new URL(f, location.href)).href));
      }
      if (id?.startsWith('term-')) c.querySelectorAll('.term-def').forEach((n) => { if (n.textContent === target.textContent) n.classList.add('hit'); });
      box.append(c);
    }
    return { head, box };
  }

  const stack = [];
  const closeFrom = (level) => { while (stack.length > level) stack.pop().pop.remove(); };
  function place(pop, trigger) {
    const r = trigger.getBoundingClientRect();
    const w = Math.min(pop.offsetWidth, innerWidth - 16);
    const left = Math.max(8, Math.min(r.left, innerWidth - w - 8));
    const below = innerHeight - r.bottom;
    const top = below < pop.offsetHeight + 12 && r.top > below ? r.top - pop.offsetHeight - 6 : r.bottom + 6;
    pop.style.left = `${left + scrollX}px`;
    pop.style.top = `${Math.max(scrollY + 4, top + scrollY)}px`;
  }
  async function open(trigger) {
    const parent = trigger.closest('.pop');
    const level = parent ? stack.findIndex((s) => s.pop === parent) + 1 : 0;
    if (stack[level]?.trigger === trigger) return;
    closeFrom(level);
    const pop = h('div', { class: 'pop', role: 'dialog' });
    stack.push({ trigger, pop });
    let content;
    if (trigger.matches('.cited')) {
      const list = h('ul', { class: 'cited-list' });
      for (const [href, text] of JSON.parse(trigger.dataset.list)) {
        const [f, id] = href.split('#');
        list.append(h('li', {}, h('a', { class: 'xref', href, 'data-ref': `${f || file}#${id}` }, text)));
      }
      content = { head: '本书中引用它的地方（其他章）', box: h('div', { class: 'pop-body' }, list) };
    } else {
      const [f, id] = trigger.dataset.ref.split('#');
      const doc = await load(f);
      content = doc ? extract(doc, id, f) : { head: trigger.textContent, box: h('div', { class: 'pop-body' }, h('p', {}, '这一章不在当前文件中（单章导出的页面）。')) };
      if (!content) return;
      const where = [];
      if (trigger.matches('.term')) where.push(trigger.dataset.where ? `定义于 ${trigger.dataset.where}` : '');
      if (f && f !== file && doc) where.push(chapterName(doc));
      const go = h('a', { href: f === file ? `#${id}` : trigger.getAttribute('href') ?? `${f}${id ? `#${id}` : ''}`, class: 'pop-go' }, '前往 →');
      content.head = h('span', {}, h('b', {}, content.head || trigger.textContent), where.filter(Boolean).length ? h('span', { class: 'pop-where' }, ` · ${where.filter(Boolean).join(' · ')}`) : null);
      content.go = go;
    }
    if (stack.at(-1)?.pop !== pop) return;
    pop.append(h('div', { class: 'pop-head' }, content.head, content.go ?? null), content.box);
    document.body.append(pop);
    place(pop, trigger);
  }

  const TRIGGER = '.xref[data-ref], .term[data-ref], .cited';
  const canHover = matchMedia('(hover: hover)').matches;
  let openTimer = 0;
  let closeTimer = 0;
  let hovered = null;
  if (canHover) {
    document.addEventListener('mouseover', (e) => {
      hovered = e.target;
      clearTimeout(closeTimer);
      const t = e.target.closest(TRIGGER);
      clearTimeout(openTimer);
      if (t && !t.matches('.cited') && !t.closest('.panel')) openTimer = setTimeout(() => open(t), 220);
      else if (!t) {
        closeTimer = setTimeout(() => {
          let keep = -1;
          stack.forEach((s, i) => { if (s.pop.contains(hovered) || s.trigger.contains(hovered)) keep = i; });
          closeFrom(keep + 1);
        }, 380);
      }
    });
  }
  document.addEventListener('click', (e) => {
    const t = e.target.closest(TRIGGER);
    if (t && !t.closest('.panel') && (t.matches('.term, .cited') || (!canHover && t.matches('.xref') && !t.closest('.pop')))) {
      e.preventDefault();
      clearTimeout(openTimer);
      open(t);
      return;
    }
    if (!e.target.closest('.pop')) closeFrom(0);
  });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { closeFrom(0); closePanels(); } });
  addEventListener('resize', () => closeFrom(0));

  // ---- "cited by" notes -----------------------------------------------------------------------
  for (const e of main.querySelectorAll('[data-cited]')) {
    const list = JSON.parse(e.dataset.cited);
    const b = h('button', { type: 'button', class: 'cited', title: '本书其他章中引用它的地方', 'data-list': e.dataset.cited }, `被引用 ${list.length} 处`);
    if (e.matches('blockquote')) e.prepend(b);
    else e.append(' ', b);
  }

  // Panels start below the sticky top bar.
  function barHeight() { root.style.setProperty('--bar-h', `${topbar.offsetHeight}px`); }
  barHeight();
  addEventListener('resize', barHeight);

  // ---- interactive figures --------------------------------------------------------------------
  if (main.querySelector('[data-widget]') && !window.bookWidgets) {
    const loadWidgets = () => document.head.append(h('script', { src: widgetSrc }));
    if (window.bookWidgetModels) loadWidgets();
    else {
      const models = h('script', { src: new URL('widget-models.js', widgetSrc).href });
      models.addEventListener('load', loadWidgets, { once: true });
      document.head.append(models);
    }
  }
})();
