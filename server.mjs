// Local reader for the book in book/: renders Markdown + KaTeX on request.
import { createServer } from 'node:http';
import { readFile, readdir, stat } from 'node:fs/promises';
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

function inside(base, path) {
  const full = resolve(base, path);
  return full === base || full.startsWith(base + sep) ? full : null;
}

// Reading order comes from the links in book/README.md.
async function chapterOrder() {
  const index = await readFile(join(BOOK, 'README.md'), 'utf8');
  return [...new Set([...index.matchAll(/\]\(([^)#\s]+\.md)\)/g)].map((m) => m[1]))];
}

function page(title, body, nav) {
  return `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${title}</title>
<link rel="stylesheet" href="/assets/katex/katex.min.css">
<link rel="stylesheet" href="/assets/style.css">
</head>
<body>
<nav>${nav}</nav>
<main>
${body}
</main>
<nav>${nav}</nav>
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
  const order = await chapterOrder();
  const at = order.indexOf(name);
  const links = ['<a href="/book/">目录</a>'];
  if (at > 0) links.push(`<a href="/book/${order[at - 1]}" rel="prev">上一章</a>`);
  if (at >= 0 && at + 1 < order.length) links.push(`<a href="/book/${order[at + 1]}" rel="next">下一章</a>`);
  const title = (source.match(/^#\s+(.+)$/m)?.[1] ?? name).replace(/[*`$]/g, '');
  return { title, html: page(title, md.render(source), links.join(' · ')) };
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

// --check: render every Markdown file and report KaTeX errors and broken local links.
async function check() {
  const files = (await readdir(BOOK)).filter((f) => f.endsWith('.md')).sort();
  const order = await chapterOrder();
  let problems = 0;
  for (const missing of order.filter((name) => !files.includes(name))) {
    console.log(`README.md: listed but missing: ${missing}`);
    problems++;
  }
  for (const name of files) {
    const source = await readFile(join(BOOK, name), 'utf8');
    const html = md.render(source);
    for (const m of html.matchAll(/class="katex-error"[^>]*title="([^"]*)"/g)) {
      console.log(`${name}: KaTeX: ${m[1].slice(0, 160)}`);
      problems++;
    }
    for (const m of source.matchAll(/\]\(([^)\s#]+\.md)(#[^)\s]*)?\)/g)) {
      if (/^[a-z]+:/i.test(m[1])) continue;
      const target = m[1];
      if (!files.includes(target)) {
        console.log(`${name}: broken link: ${target}`);
        problems++;
        continue;
      }
      if (m[2]) {
        const targetHtml = md.render(await readFile(join(BOOK, target), 'utf8'));
        const id = decodeURIComponent(m[2].slice(1));
        if (!targetHtml.includes(`id="${id}"`)) {
          console.log(`${name}: missing anchor: ${target}${m[2]}`);
          problems++;
        }
      }
    }
  }
  console.log(`${files.length} files checked, ${problems} problem(s).`);
  process.exitCode = problems ? 1 : 0;
}

if (process.argv.includes('--check')) {
  await check();
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
