// Verify the generated site's links, preview metadata, anchors and KaTeX font files.
import { readFile, readdir, stat } from 'node:fs/promises';
import { extname, join, resolve, sep } from 'node:path';
import MarkdownIt from 'markdown-it';

const directory = resolve(process.argv[2] ?? 'dist');
const base = `/${(process.argv[3] ?? '/').split('/').filter(Boolean).join('/')}/`.replace(/\/+/g, '/');
const origin = 'https://static-site.invalid';
const unescape = new MarkdownIt().utils.unescapeAll;
const files = await readdir(directory, { recursive: true });
const pages = new Map();
for (const file of files.filter((file) => extname(file) === '.html')) {
  const html = await readFile(join(directory, file), 'utf8');
  pages.set(file.split(sep).join('/'), {
    html,
    ids: new Set([...html.matchAll(/\bid="([^"]*)"/g)].map((match) => unescape(match[1]))),
  });
}
let links = 0;
let problems = 0;
const report = (file, message) => { console.error(`${file}: ${message}`); problems++; };
async function checkLink(file, href) {
  const url = new URL(href, `${origin}${base}${file}`);
  if (url.origin !== origin) return;
  links++;
  if (!url.pathname.startsWith(base)) return report(file, `link escapes base path: ${href}`);
  const target = decodeURIComponent(url.pathname.slice(base.length)) + (url.pathname.endsWith('/') ? 'index.html' : '');
  try {
    if (!(await stat(join(directory, target))).isFile()) throw new Error('not a file');
  } catch {
    return report(file, `missing file: ${href}`);
  }
  if (url.hash && pages.has(target) && !pages.get(target).ids.has(decodeURIComponent(url.hash.slice(1)))) {
    report(file, `missing anchor: ${href}`);
  }
}
for (const [file, { html }] of pages) {
  for (const [, attr, encoded] of html.matchAll(/\b(href|src|data-ref|data-cited)="([^"]*)"/g)) {
    const value = unescape(encoded);
    if (attr === 'data-cited') {
      for (const [href] of JSON.parse(value)) await checkLink(file, href);
    } else {
      await checkLink(file, value);
    }
  }
}
const cssFile = 'assets/katex/katex.min.css';
const css = await readFile(join(directory, cssFile), 'utf8');
for (const [, href] of css.matchAll(/url\((fonts\/[^)]+)\)/g)) await checkLink(cssFile, href);
console.log(`${pages.size} pages and ${links} local links checked, ${problems} problem(s).`);
process.exitCode = problems ? 1 : 0;
