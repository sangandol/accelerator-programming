// Local reader for the book in book/: renders Markdown + KaTeX on request and adds the reading aids
// (boxed definitions and theorems, cross-reference and term previews, outline; see web/reader.js).
import { createServer } from 'node:http';
import { readFile, readdir, stat, writeFile } from 'node:fs/promises';
import { dirname, extname, join, normalize, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import MarkdownIt from 'markdown-it';
import markdownItKatex from '@vscode/markdown-it-katex';

const ROOT = dirname(fileURLToPath(import.meta.url));
const BOOK = join(ROOT, 'book');
const KATEX = join(ROOT, 'node_modules', 'katex', 'dist');
const WEB = join(ROOT, 'web');
const TYPES = {
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.woff2': 'font/woff2',
  '.woff': 'font/woff',
  '.ttf': 'font/ttf',
  '.json': 'application/json; charset=utf-8',
};

const md = new MarkdownIt({ html: true, linkify: false, typographer: false })
  .use(markdownItKatex.default ?? markdownItKatex, { throwOnError: false, strict: 'ignore', output: 'html' });
const escapeHtml = md.utils.escapeHtml;

// GitHub-compatible heading ids, so the same #anchors work on GitHub and here.
md.core.ruler.push('heading_ids', (state) => {
  const seen = new Map();
  const tokens = state.tokens;
  for (let i = 0; i < tokens.length; i++) {
    if (tokens[i].type !== 'heading_open') continue;
    const text = tokens[i + 1].children
      .filter((t) => t.type === 'text' || t.type === 'code_inline')
      .map((t) => t.content)
      .join('');
    let id = text.trim().toLowerCase().replace(/[^\p{L}\p{M}\p{N}\s_-]/gu, '').replace(/\s/g, '-');
    const count = seen.get(id) ?? 0;
    seen.set(id, count + 1);
    if (count) id = `${id}-${count}`;
    tokens[i].attrSet('id', id);
  }
});

// Empty text tokens (markdown-it leaves one before a leading **…**) would hide a paragraph's label.
md.core.ruler.push('drop_empty_text', (state) => {
  for (const t of state.tokens) if (t.children) t.children = t.children.filter((c) => c.type !== 'text' || c.content !== '');
});

// ---------------------------------------------------------------------------------------------
// Book structure. Numbered items are paragraphs that start with a bold label such as
// **定义 1.4（组合电路）**; written inside a blockquote, the whole blockquote becomes a box.
// A term is defined as **术语**（English）. Proofs run from *证明* to the paragraph ending in ∎.

const KINDS = { 定义: 'def', 命题: 'prop', 定理: 'thm', 引理: 'lemma', 推论: 'cor', 例: 'example', 注: 'remark', 习题: 'exercise', 图: 'figure' };
const LABEL = /^(定义|命题|定理|引理|推论|例|注|习题|图)\s?(\d+|[A-E])\.(\d+)/;
const BOXES = [[/^在体系中的位置/, 'position-box'], [/^现状/, 'status-box']];
const REF_LABEL = /(定义|命题|定理|引理|推论|例|注|习题|图)\s?((?:\d+|[A-E])\.\d+)((?:\s?[、，,和与及–]\s?(?:\d+|[A-E])\.\d+(?![\d.]|\s?节))*)/gu;
const REF_SECTION = /(?<![\d.A-Za-z])((?:\d+|[A-E])\.\d+)((?:\s?[、，–—-]\s?(?:\d+|[A-E])\.\d+)*)\s?节/gu;
const REF_CHAPTER = /第\s?(\d+)((?:\s?[、，和与及–]\s?\d+)*)\s?章/gu;
const REF_APPENDIX = /附录\s?([A-E])(?![A-Za-z])/gu;
const NUMBER = /(?:\d+|[A-E])\.\d+/g;

function chapterKey(file) {
  const m = file.match(/^(\d+)-/) ?? file.match(/^([A-E])-/);
  return m ? (/^\d+$/.test(m[1]) ? String(Number(m[1])) : m[1]) : null;
}

function leading(inline, open) {
  const c = inline?.children;
  if (!c || c.length < 3 || c[0].type !== open || c[1].type !== 'text') return null;
  return c[1].content;
}

const inlineText = (inline) => (inline?.children ?? []).filter((t) => t.type === 'text' || t.type === 'code_inline').map((t) => t.content).join('');
const labelId = (kind, num) => `${kind}-${num}`;
const labelTitle = (head) => head.match(/（(.+)）\s*$/)?.[1] ?? head.match(/（(.+?)）/)?.[1] ?? '';

// **术语**（English） at children[k]; returns the term and its English name, or null.
function termAt(c, k) {
  if (c[k]?.type !== 'strong_open' || c[k + 1]?.type !== 'text' || c[k + 2]?.type !== 'strong_close' || c[k + 3]?.type !== 'text') return null;
  const term = c[k + 1].content.trim();
  const m = c[k + 3].content.match(/^（([A-Za-z][A-Za-z0-9 .\-–'’/+&()]*?)\s*[，,；;）]/);
  if (!m || term.length < 2 || term.length > 16 || LABEL.test(term) || !/[一-鿿]/.test(term)) return null;
  return { term, en: m[1] };
}

// Visits the paragraphs and headings of a token list in order, tracking the current section.
function* walk(tokens) {
  let section = { num: null, id: null, title: '' };
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i];
    if (t.type === 'heading_open') {
      const title = inlineText(tokens[i + 1]);
      if (t.tag === 'h2') section = { num: title.match(/^((?:\d+|[A-E])\.\d+)\s/)?.[1] ?? null, id: t.attrGet('id'), title };
      yield { kind: 'heading', i, t, title, section };
    } else if (t.type === 'inline' && tokens[i - 1]?.type !== 'heading_open') {
      yield { kind: 'inline', i, t, section };
    }
  }
}

// Splits text into pieces with refs as links; `resolve(kind, key)` gives { href, ref } or null.
function linkRefs(text, resolve) {
  let pieces = [{ text }];
  const pass = (re, handle) => {
    pieces = pieces.flatMap((p) => {
      if (p.html !== undefined) return [p];
      const out = [];
      let last = 0;
      for (const m of p.text.matchAll(re)) {
        if (m.index > last) out.push({ text: p.text.slice(last, m.index) });
        out.push(...handle(m));
        last = m.index + m[0].length;
      }
      if (last < p.text.length) out.push({ text: p.text.slice(last) });
      return out;
    });
  };
  const link = (display, target) => (target
    ? { html: `<a class="xref" href="${escapeHtml(target.href)}" data-ref="${escapeHtml(target.ref)}">${escapeHtml(display)}</a>`, target }
    : { text: display });
  // "命题 1.24、1.26", "1.5–1.8 节", "第 7、10 章": every number is its own link; the first link takes the
  // leading word, the last takes a trailing 节 or 章. Listed labels must be in the first one's chapter.
  const list = (m, numbers, kind) => {
    const s = m[0];
    const nums = [...s.matchAll(numbers)];
    const chapter = nums[0][0].split('.')[0];
    const out = [];
    let last = 0;
    nums.forEach((n, idx) => {
      const start = idx === 0 ? 0 : n.index;
      if (start > last) out.push({ text: s.slice(last, start) });
      let end = n.index + n[0].length;
      if (idx === nums.length - 1 && /^\s?[节章]$/.test(s.slice(end))) end = s.length;
      out.push(link(s.slice(start, end), n[0].split('.')[0] === chapter || kind === '章' ? resolve(kind, n[0]) : null));
      last = end;
    });
    if (last < s.length) out.push({ text: s.slice(last) });
    return out;
  };
  pass(REF_LABEL, (m) => list(m, NUMBER, m[1]));
  pass(REF_SECTION, (m) => list(m, NUMBER, '节'));
  pass(REF_CHAPTER, (m) => list(m, /\d+/g, '章'));
  pass(REF_APPENDIX, (m) => [link(m[0], resolve('章', m[1]))]);
  return pieces;
}

// Pass 1 over every file: chapters, sections, numbered items, terms and who cites what.
let bookCache = null;
async function bookIndex() {
  const order = await chapterOrder();
  const files = (await readdir(BOOK)).filter((f) => f.endsWith('.md')).sort();
  const stamps = (await Promise.all(files.map((f) => stat(join(BOOK, f))))).map((s) => s.mtimeMs).join();
  if (bookCache?.stamps === stamps + order.join()) return bookCache;
  const book = { stamps: stamps + order.join(), order, chapters: new Map(), sections: new Map(), labels: new Map(), terms: [], refs: [], citedBy: new Map(), problems: [] };
  const rank = (f) => (order.includes(f) ? order.indexOf(f) : order.length);
  for (const file of [...files].sort((a, b) => rank(a) - rank(b))) {
    const ch = chapterKey(file);
    const tokens = md.parse(await readFile(join(BOOK, file), 'utf8'), {});
    const termIds = new Map();
    let pos = 0;
    for (const v of walk(tokens)) {
      pos++;
      if (v.kind === 'heading') {
        if (v.t.tag === 'h1' && ch !== null) book.chapters.set(ch, { file, title: v.title.replace(/[*`$]/g, '') });
        if (v.t.tag === 'h2' && v.section.num) book.sections.set(v.section.num, { file, id: v.section.id, title: v.title });
        continue;
      }
      const c = v.t.children;
      const head = tokens[v.i - 1].type === 'paragraph_open' ? leading(v.t, 'strong_open') : null;
      const m = head?.match(LABEL);
      if (m) {
        const id = labelId(m[1], `${m[2]}.${m[3]}`);
        if (book.labels.has(id)) book.problems.push(`${file}: duplicate label: ${m[1]} ${m[2]}.${m[3]}`);
        book.labels.set(id, { file, id, kind: m[1], ch: m[2], n: Number(m[3]), title: labelTitle(head), section: v.section.num });
      }
      let inLink = 0;
      for (let k = 0; k < c.length; k++) {
        if (c[k].type === 'link_open') inLink++;
        if (c[k].type === 'link_close') inLink--;
        const def = termAt(c, k);
        if (def) {
          const count = termIds.get(def.term) ?? 0;
          termIds.set(def.term, count + 1);
          book.terms.push({ ...def, file, id: `term-${def.term}${count ? `-${count + 1}` : ''}`, section: v.section.num, rank: rank(file), pos });
        }
        if (c[k].type !== 'text' || inLink || (k === 1 && m)) continue; // k === 1: the label itself
        for (const piece of linkRefs(c[k].content, (kind, key) => ({ href: '', ref: `${kind}|${key}` }))) {
          if (piece.target) book.refs.push({ file, section: v.section, ref: piece.target.ref });
        }
      }
    }
  }
  for (const r of book.refs) {
    const [kind, key] = r.ref.split('|');
    if (kind === '章' || kind === '节') continue;
    const id = labelId(kind, key);
    const target = book.labels.get(id);
    if (!target) continue;
    if (target.file === r.file) continue; // only uses in other chapters: where the layer above relies on it
    const list = book.citedBy.get(id) ?? [];
    const href = `${r.file}#${r.section.id ?? ''}`;
    const chapter = book.chapters.get(chapterKey(r.file))?.title ?? r.file;
    if (!list.some((x) => x[0] === href)) list.push([href, r.section.num ? r.section.title : `${chapter}（${r.section.title || '开头'}）`]);
    book.citedBy.set(id, list);
  }
  bookCache = book;
  return book;
}

function resolver(book, file) {
  return (kind, key) => {
    let target = null;
    if (kind === '章') {
      const c = book.chapters.get(key);
      if (c) target = { file: c.file, id: '' };
    } else if (kind === '节') {
      target = book.sections.get(key) ?? null;
    } else {
      target = book.labels.get(labelId(kind, key)) ?? null;
    }
    if (!target) return null;
    const hash = target.id ? `#${target.id}` : '';
    return { href: target.file === file ? hash || '#' : `${target.file}${hash}`, ref: `${target.file}${hash}` };
  };
}

// Pass 2 for one file: boxes and anchors, links for refs and terms, wrappers for proofs, figures, exercises.
md.core.ruler.push('book_structure', (state) => {
  const { book, file } = state.env;
  if (!book) return;
  const tokens = state.tokens;
  const resolve = resolver(book, file);
  const token = (type, content) => Object.assign(new state.Token(type, '', 0), { content });
  const at = book.order.includes(file) ? book.order.indexOf(file) : book.order.length;
  const boxed = new Set();

  // Boxes: a blockquote whose first paragraph starts with a label or a box name.
  const mark = (open, strong, head, m) => {
    const id = labelId(m[1], `${m[2]}.${m[3]}`);
    open.attrSet('id', id);
    open.attrSet('data-kind', m[1]);
    open.attrSet('data-title', labelTitle(head));
    const cited = book.citedBy.get(id);
    if (cited?.length) open.attrSet('data-cited', JSON.stringify(cited.map(([href, text]) => [href.startsWith(`${file}#`) ? href.slice(file.length) : href, text])));
    strong.attrSet('class', 'env-name');
  };
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i];
    if (t.type === 'blockquote_open' && tokens[i + 1]?.type === 'paragraph_open') {
      const head = leading(tokens[i + 2], 'strong_open');
      if (!head) continue;
      const m = head.match(LABEL);
      if (m && KINDS[m[1]] && m[1] !== '习题' && m[1] !== '图') {
        t.attrJoin('class', `env env-${KINDS[m[1]]}`);
        mark(t, tokens[i + 2].children[0], head, m);
        boxed.add(i + 1);
        continue;
      }
      const box = BOXES.find(([re]) => re.test(head));
      if (box) {
        t.attrJoin('class', `box ${box[1]}`);
        tokens[i + 2].children[0].attrSet('class', 'box-name');
      }
    } else if (t.type === 'paragraph_open' && !boxed.has(i)) {
      const head = leading(tokens[i + 1], 'strong_open');
      const m = head?.match(LABEL);
      if (m && m[1] !== '图') {
        t.attrJoin('class', `label-para label-${KINDS[m[1]]}`);
        mark(t, tokens[i + 1].children[0], head, m);
      }
    }
  }

  // Links for refs, and the first use of each term in a section linked to its definition.
  // Terms from earlier chapters only when 3+ characters: short ones (深度, 传递) are often ordinary words there.
  const active = new Map();
  for (const d of book.terms) if (d.rank < at && d.term.length >= 3) active.set(d.term, d);
  const own = book.terms.filter((d) => d.file === file);
  const placed = new Set();
  let pattern = null;
  let seen = new Set();
  let pos = 0;
  for (const v of walk(tokens)) {
    pos++;
    if (v.kind === 'heading') {
      if (v.t.tag === 'h2') seen = new Set();
      continue;
    }
    const c = v.t.children;
    const out = [];
    let inLink = 0;
    let skip = 0;
    for (let k = 0; k < c.length; k++) {
      const tok = c[k];
      if (tok.type === 'link_open') inLink++;
      if (tok.type === 'link_close') inLink--;
      if (tok.type === 'strong_open' && tok.attrGet('class')) skip++;
      if (tok.type === 'strong_close' && skip) skip--;
      const def = termAt(c, k);
      if (def) {
        const d = own.find((x) => x.term === def.term && x.pos === pos && !placed.has(x));
        if (d) {
          placed.add(d);
          tok.attrSet('class', 'term-def');
          tok.attrSet('id', d.id);
          tok.attrSet('data-en', d.en);
          active.set(d.term, d);
          seen.add(d.term);
          pattern = null;
          skip++;
        }
      }
      if (tok.type !== 'text' || inLink || skip) {
        out.push(tok);
        continue;
      }
      if (!pattern && active.size) {
        const words = [...active.keys()].sort((a, b) => b.length - a.length).map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
        pattern = new RegExp(words.join('|'), 'gu');
      }
      const pieces = linkRefs(tok.content, resolve).flatMap((p) => {
        if (p.html !== undefined || !pattern) return [p];
        const parts = [];
        let last = 0;
        for (const m of p.text.matchAll(pattern)) {
          const d = active.get(m[0]);
          if (seen.has(m[0]) || !d) continue;
          seen.add(m[0]);
          if (m.index > last) parts.push({ text: p.text.slice(last, m.index) });
          const ref = `${d.file}#${d.id}`;
          parts.push({ html: `<span class="term" tabindex="0" data-ref="${escapeHtml(ref)}" data-en="${escapeHtml(d.en)}" data-where="${escapeHtml(d.section ? `${d.section} 节` : '')}">${escapeHtml(m[0])}</span>` });
          last = m.index + m[0].length;
        }
        if (last < p.text.length) parts.push({ text: p.text.slice(last) });
        return parts;
      });
      if (pieces.length === 1 && pieces[0].html === undefined) {
        out.push(tok);
        continue;
      }
      for (const p of pieces) out.push(p.html !== undefined ? token('html_inline', p.html) : token('text', p.text));
    }
    v.t.children = out;
  }

  // Wrappers. Ranges are [first, last] token indices at the same nesting level.
  const wraps = [];
  const paragraphEnd = (i) => {
    let j = i;
    while (tokens[j].type !== 'paragraph_close') j++;
    return j;
  };
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i];
    if (t.type !== 'paragraph_open') continue;
    const inline = tokens[i + 1];
    if (leading(inline, 'em_open') === '证明') {
      inline.children[0].attrSet('class', 'proof-head');
      for (let j = i; j < tokens.length; j++) {
        const u = tokens[j];
        if (u.type === 'heading_open' || u.level < t.level) break;
        if (j > i && u.type === 'paragraph_open' && u.level === t.level && LABEL.test(leading(tokens[j + 1], 'strong_open') ?? '')) break;
        if (u.type === 'paragraph_close' && u.level === t.level && inlineText(tokens[j - 1]).trimEnd().endsWith('∎')) {
          wraps.push([i, j, '<div class="proof">', '</div>']);
          break;
        }
      }
      continue;
    }
    const kids = inline.children.filter((k) => !(k.type === 'text' && !k.content.trim()));
    if (kids.length === 1 && kids[0].type === 'image' && tokens[i + 3]?.type === 'paragraph_open') {
      const m = leading(tokens[i + 4], 'strong_open')?.match(LABEL);
      if (m?.[1] === '图') {
        tokens[i + 3].attrSet('class', 'caption');
        wraps.push([i, paragraphEnd(i + 3), `<figure class="fig" id="${escapeHtml(labelId('图', `${m[2]}.${m[3]}`))}">`, '</figure>']);
      }
      continue;
    }
    const m = leading(inline, 'strong_open')?.match(LABEL);
    if (m?.[1] === '习题' && t.level === 0) {
      let j = i + 1;
      for (; j < tokens.length; j++) {
        const u = tokens[j];
        if (u.type === 'heading_open') break;
        if (u.type === 'paragraph_open' && u.level === 0 && leading(tokens[j + 1], 'strong_open')?.match(LABEL)?.[1] === '习题') break;
      }
      wraps.push([i, j - 1, '<section class="exercise">', '</section>']);
    }
  }
  for (const [first, last, open, close] of wraps.sort((a, b) => b[0] - a[0])) {
    tokens.splice(last + 1, 0, token('html_block', `${close}\n`));
    tokens.splice(first, 0, token('html_block', `${open}\n`));
  }
});

function inside(base, path) {
  const full = resolve(base, path);
  return full === base || full.startsWith(base + sep) ? full : null;
}

// Reading order comes from the links in book/README.md.
async function chapterOrder() {
  const index = await readFile(join(BOOK, 'README.md'), 'utf8');
  return [...new Set([...index.matchAll(/\]\(([^)#\s]+\.md)\)/g)].map((m) => m[1]))];
}

// Web fonts for the reader's 宋体 and 楷体 settings; a font's files load only when the page uses it, and
// the stacks in style.css fall back to system fonts when offline.
const FONT_CSS = {
  song: 'https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&display=swap',
  kai: 'https://cdn.jsdelivr.net/npm/lxgw-wenkai-screen-webfont@1.7.0/lxgwwenkaiscreen.css',
};
// Applies the reader settings saved by web/reader.js before the first paint.
const SETTINGS_SCRIPT = "try{var r=document.documentElement,s=localStorage;['font','width','lh','theme'].forEach(function(k){var v=s.getItem('rd-'+k);if(v)r.setAttribute('data-'+k,v)});var f=s.getItem('rd-fs');if(f)r.style.setProperty('--fs',f+'px')}catch(e){}";

function page(title, body, nav, file = '') {
  return `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${title}</title>
<script>${SETTINGS_SCRIPT}</script>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="${FONT_CSS.song}">
<link rel="stylesheet" href="${FONT_CSS.kai}">
<link rel="stylesheet" href="/assets/katex/katex.min.css">
<link rel="stylesheet" href="/assets/style.css">
<script src="/assets/reader.js" defer></script>
</head>
<body>
<header class="topbar"><nav>${nav}</nav></header>
<main data-file="${escapeHtml(file)}">
${body}
</main>
<nav class="bottom">${nav}</nav>
</body>
</html>`;
}

async function renderChapter(name) {
  const file = inside(BOOK, name);
  if (!file || extname(file) !== '.md') return null;
  let source;
  try {
    source = await readFile(file, 'utf8');
  } catch {
    return null;
  }
  const book = await bookIndex();
  const order = book.order;
  const at = order.indexOf(name);
  const links = ['<a href="/book/">目录</a>'];
  if (at > 0) links.push(`<a href="/book/${order[at - 1]}" rel="prev">上一章</a>`);
  if (at >= 0 && at + 1 < order.length) links.push(`<a href="/book/${order[at + 1]}" rel="next">下一章</a>`);
  const title = (source.match(/^#\s+(.+)$/m)?.[1] ?? name).replace(/[*`$]/g, '');
  return { title, html: page(title, md.render(source, { book, file: name }), links.join(' · '), name) };
}

async function serveFile(res, file) {
  try {
    const info = await stat(file);
    if (!info.isFile()) throw new Error('not a file');
    res.writeHead(200, { 'Content-Type': TYPES[extname(file)] ?? 'application/octet-stream' });
    res.end(await readFile(file));
  } catch {
    res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('Not found');
  }
}

async function handle(req, res) {
  const url = new URL(req.url, 'http://localhost');
  const path = decodeURIComponent(url.pathname);
  if (path === '/api/health') {
    const order = await chapterOrder();
    res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({ ok: true, chapters: order.length }));
    return;
  }
  if (path === '/' || path === '/book') {
    res.writeHead(302, { Location: '/book/' });
    res.end();
    return;
  }
  if (path.startsWith('/assets/katex/')) return serveFile(res, inside(KATEX, path.slice('/assets/katex/'.length)) ?? '');
  if (path.startsWith('/assets/')) return serveFile(res, inside(WEB, path.slice('/assets/'.length)) ?? '');
  if (path === '/book/fig/' || path === '/book/fig') {
    // Gallery of all figures; ?f=09 shows only names starting with 09.
    const prefix = url.searchParams.get('f') ?? '';
    const svgs = (await readdir(join(BOOK, 'fig'))).filter((f) => f.endsWith('.svg') && f.startsWith(prefix)).sort();
    const body = `<h1>插图</h1>\n` + svgs.map((f) => `<h3>${f}</h3>\n<p><img src="/book/fig/${f}" alt="${f}"></p>`).join('\n');
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(page('插图', body, '<a href="/book/">目录</a>'));
    return;
  }
  if (path.startsWith('/book/')) {
    const name = path.slice('/book/'.length) || 'README.md';
    if (name.endsWith('.md')) {
      const chapter = await renderChapter(normalize(name).split(sep).join('/'));
      if (chapter) {
        res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
        res.end(chapter.html);
        return;
      }
    } else {
      return serveFile(res, inside(BOOK, name) ?? '');
    }
  }
  res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
  res.end('Not found');
}

// --check: render every Markdown file and report KaTeX errors, broken local links, missing images,
// numbered references to items that do not exist, and items numbered out of sequence.
async function check() {
  const files = (await readdir(BOOK)).filter((f) => f.endsWith('.md')).sort();
  const book = await bookIndex();
  const order = book.order;
  let problems = 0;
  const report = (line) => {
    console.log(line);
    problems++;
  };
  for (const missing of order.filter((name) => !files.includes(name))) report(`README.md: listed but missing: ${missing}`);
  book.problems.forEach(report);
  for (const name of files) {
    const source = await readFile(join(BOOK, name), 'utf8');
    const html = md.render(source, { book, file: name });
    for (const m of html.matchAll(/class="katex-error"[^>]*title="([^"]*)"/g)) report(`${name}: KaTeX: ${m[1].slice(0, 160)}`);
    for (const m of source.matchAll(/!\[[^\]]*\]\(([^)\s]+)\)/g)) {
      if (/^[a-z]+:/i.test(m[1])) continue;
      try {
        await stat(join(BOOK, m[1]));
      } catch {
        report(`${name}: missing image: ${m[1]}`);
      }
    }
    for (const m of source.matchAll(/\]\(([^)\s#]+\.md)(#[^)\s]*)?\)/g)) {
      if (/^[a-z]+:/i.test(m[1])) continue;
      const target = m[1];
      if (!files.includes(target)) {
        report(`${name}: broken link: ${target}`);
        continue;
      }
      if (m[2]) {
        const targetHtml = md.render(await readFile(join(BOOK, target), 'utf8'), { book, file: target });
        const id = decodeURIComponent(m[2].slice(1));
        if (!targetHtml.includes(`id="${id}"`)) report(`${name}: missing anchor: ${target}${m[2]}`);
      }
    }
    // Numbered items: one counter for 定义…注, one for 习题, one for 图, each 1, 2, 3, … in order.
    const ch = chapterKey(name);
    const next = { env: 1, 习题: 1, 图: 1 };
    for (const l of [...book.labels.values()].filter((l) => l.file === name)) {
      const counter = l.kind === '习题' || l.kind === '图' ? l.kind : 'env';
      if (l.ch !== ch) report(`${name}: ${l.kind} ${l.ch}.${l.n} is numbered for another chapter`);
      else if (l.n !== next[counter]) report(`${name}: ${l.kind} ${l.ch}.${l.n} out of sequence (expected ${ch}.${next[counter]})`);
      next[counter] = l.n + 1;
    }
  }
  for (const r of book.refs) {
    const [kind, key] = r.ref.split('|');
    const found = kind === '章' ? book.chapters.has(key) : kind === '节' ? book.sections.has(key) : book.labels.has(labelId(kind, key));
    if (!found) report(`${r.file}: reference to a missing item: ${kind === '节' ? `${key} 节` : kind === '章' ? `第 ${key} 章 / 附录 ${key}` : `${kind} ${key}`}`);
  }
  console.log(`${files.length} files checked, ${problems} problem(s).`);
  process.exitCode = problems ? 1 : 0;
}

// --export 01-x.md out.html: one chapter as a self-contained page (styles, fonts, scripts, figures inlined).
// Links to other chapters stay as they are; their previews say the chapter is not in the file.
async function exportChapter(name, out) {
  const chapter = await renderChapter(name);
  if (!chapter) throw new Error(`no such chapter: ${name}`);
  const data = async (file, type) => `data:${type};base64,${(await readFile(file)).toString('base64')}`;
  let katex = await readFile(join(KATEX, 'katex.min.css'), 'utf8');
  const fonts = new Map();
  for (const [, font] of katex.matchAll(/url\(fonts\/([\w-]+)\.woff2\)/g)) fonts.set(font, await data(join(KATEX, 'fonts', `${font}.woff2`), 'font/woff2'));
  katex = katex.replace(/src:url\(fonts\/([\w-]+)\.woff2\) format\("woff2"\)[^;}]*/g, (_, font) => `src:url(${fonts.get(font)}) format("woff2")`);
  let html = chapter.html
    .replace('<link rel="stylesheet" href="/assets/katex/katex.min.css">', () => `<style>${katex}</style>`)
    .replace('<link rel="stylesheet" href="/assets/style.css">', '')
    .replace('<script src="/assets/reader.js" defer></script>', '');
  html = html.replace('</head>', `<style>${await readFile(join(WEB, 'style.css'), 'utf8')}</style>\n</head>`);
  html = html.replace(/<nav>(.*?)<\/nav>/gs, `<nav>${chapter.title}</nav>`);
  for (const [tag, src] of [...html.matchAll(/<img src="(fig\/[^"]+\.svg)"/g)].map((m) => [m[0], m[1]])) {
    html = html.replace(tag, `<img src="${await data(join(BOOK, src), 'image/svg+xml')}"`);
  }
  const scripts = await Promise.all(['widgets.js', 'reader.js'].map((f) => readFile(join(WEB, f), 'utf8')));
  html = html.replace('</body>', `${scripts.map((js) => `<script>${js.replace(/<\/script/g, '<\\/script')}</script>`).join('\n')}\n</body>`);
  await writeFile(out, html);
  console.log(`${out}: ${(html.length / 1024).toFixed(0)} KiB`);
}

if (process.argv.includes('--check')) {
  await check();
} else if (process.argv.includes('--export')) {
  const at = process.argv.indexOf('--export');
  await exportChapter(process.argv[at + 1], process.argv[at + 2] ?? process.argv[at + 1].replace(/\.md$/, '.html'));
} else {
  const portArg = process.argv.indexOf('--port');
  const port = Number(portArg > 0 ? process.argv[portArg + 1] : process.env.PORT ?? 43202);
  createServer((req, res) => {
    handle(req, res).catch((error) => {
      console.error(error);
      res.writeHead(500, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('Internal error');
    });
  }).listen(port, '127.0.0.1', () => console.log(`[ready] http://127.0.0.1:${port}/book/`));
}
