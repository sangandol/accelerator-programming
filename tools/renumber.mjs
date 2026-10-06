// Numbers a chapter written with symbolic labels, then updates references to it in the other files.
//
//   node tools/renumber.mjs label book/02-multipliers-floats.md <source.md>
//   node tools/renumber.mjs <source.md> book/02-multipliers-floats.md [--refs]
//
// "label" turns a numbered chapter into a source whose labels are its current numbers, ready for editing.
// Writes the numbered chapter. With --refs it also rewrites the other files' references to the chapter's old
// numbers; run that exactly once, since afterwards those files already carry the new numbers.
//
// In the source, every numbered thing is written with a label ⟪name⟫ instead of its number:
//   ## ⟪name⟫ 标题                       a section          → "## 2.3 标题"
//   > **命题 ⟪name⟫（…）**               an item (定义 命题 定理 引理 推论 例 注 share one counter)
//   ![图 ⟪name⟫](fig/…) / **图 ⟪name⟫**   a figure
//   **习题 ⟪name⟫**                       an exercise
// Every other ⟪name⟫ is a reference and becomes the number ("命题 ⟪a⟫、⟪b⟫" → "命题 2.4、2.5").
// A label named after the old number (⟪2.7⟫ for an item, ⟪图2.3⟫, ⟪习题2.6⟫, ⟪2.5节⟫) records where an old
// item went; the other files' references to old numbers are then rewritten. References to old items that
// no longer exist, or whose kind changed inside a list, are reported for a manual fix.
import { readFile, writeFile, readdir } from 'node:fs/promises';
import { join, basename, dirname } from 'node:path';

const KIND = '定义|命题|定理|引理|推论|例|注';
// Same grammar as server.mjs.
const REF_LABEL = /(定义|命题|定理|引理|推论|例|注|习题|图)(\s?)((?:\d+|[A-E])\.\d+)((?:\s?[、，,和与及–]\s?(?:\d+|[A-E])\.\d+(?![\d.]|\s?节))*)/gu;
const REF_SECTION = /(?<![\d.A-Za-z])((?:\d+|[A-E])\.\d+)((?:\s?[、，–—-]\s?(?:\d+|[A-E])\.\d+)*)(\s?节)/gu;
const NUM = /(?:\d+|[A-E])\.\d+/g;

if (process.argv[2] === 'label') {
  const [, , , from, to] = process.argv;
  const ch = Number(basename(from).slice(0, 2));
  const mine = (n) => Number(n.split('.')[0]) === ch;
  let t = await readFile(from, 'utf8');
  t = t.replace(REF_LABEL, (all, kind, sp, first, rest) => {
    if (!mine(first)) return all;
    const tag = kind === '图' ? '图' : kind === '习题' ? '习题' : '';
    return kind + sp + (first + rest).replace(NUM, (n) => `⟪${tag}${n}⟫`);
  });
  t = t.replace(REF_SECTION, (all, first, rest, tail) => (mine(first) ? (first + rest).replace(NUM, (n) => `⟪${n}节⟫`) + tail : all));
  t = t.replace(new RegExp(`^## (${ch}\\.\\d+) `, 'gm'), '## ⟪$1节⟫ ');
  await writeFile(to, t);
  process.exit(0);
}
const [src, out, flag] = process.argv.slice(2);
if (!src || !out) throw new Error('usage: node tools/renumber.mjs <source.md> book/NN-name.md [--refs]');
const chapter = Number(basename(out).slice(0, 2));
let text = await readFile(src, 'utf8');

const num = new Map(); // label → { type, kind, n }
const counters = { item: 0, fig: 0, ex: 0, sec: 0 };
const define = (name, type, kind) => {
  if (num.has(name)) {
    if (type === 'fig' && num.get(name).type === 'fig') return; // the caption repeats the image's label
    throw new Error(`label defined twice: ${name}`);
  }
  num.set(name, { type, kind, n: `${chapter}.${++counters[type]}` });
};
for (const line of text.split('\n')) {
  let m;
  if ((m = line.match(/^## ⟪([^⟫]+)⟫ /))) define(m[1], 'sec');
  else if ((m = line.match(/^!\[图 ⟪([^⟫]+)⟫\]/))) define(m[1], 'fig', '图');
  else if ((m = line.match(/^\*\*习题 ⟪([^⟫]+)⟫\*\*/))) define(m[1], 'ex', '习题');
  else if ((m = line.match(new RegExp(`^(?:> )?\\*\\*(${KIND}) ⟪([^⟫]+)⟫`)))) define(m[2], 'item', m[1]);
}
const missing = new Set();
text = text.replace(/⟪([^⟫]+)⟫/g, (_, name) => {
  if (!num.has(name)) { missing.add(name); return `??${name}??`; }
  return num.get(name).n;
});
if (missing.size) throw new Error(`undefined labels: ${[...missing].join(', ')}`);
await writeFile(out, text);
console.log(`${out}: ${counters.sec} sections, ${counters.item} items, ${counters.fig} figures, ${counters.ex} exercises`);

if (flag !== '--refs') process.exit(0);

// Old number → new number, from labels named after old numbers.
const map = { item: new Map(), fig: new Map(), ex: new Map(), sec: new Map() };
for (const [name, v] of num) {
  let m;
  if ((m = name.match(/^图(\d+\.\d+)$/))) map.fig.set(m[1], v);
  else if ((m = name.match(/^习题(\d+\.\d+)$/))) map.ex.set(m[1], v);
  else if ((m = name.match(/^(\d+\.\d+)节$/))) map.sec.set(m[1], v);
  else if ((m = name.match(/^(\d+\.\d+)$/))) map.item.set(m[1], v);
}

// Rewrite references to this chapter in every other file.
const dir = dirname(out);
const files = (await readdir(dir)).filter((f) => f.endsWith('.md') && f !== basename(out));
const problems = [];
const ours = (n) => Number(n.split('.')[0]) === chapter;
for (const f of files) {
  const path = join(dir, f);
  const before = await readFile(path, 'utf8');
  const lineOf = (i) => before.slice(0, i).split('\n').length;
  let after = before.replace(REF_LABEL, (all, kind, sp, first, rest, idx) => {
    if (!ours(first)) return all;
    const type = kind === '图' ? 'fig' : kind === '习题' ? 'ex' : 'item';
    const nums = [first, ...[...rest.matchAll(/(?:\d+|[A-E])\.\d+/g)].map((x) => x[0])];
    const news = nums.map((n) => map[type].get(n));
    if (news.some((v) => !v)) { problems.push(`${f}:${lineOf(idx)}: ${all} → old item has no new home`); return all; }
    const kinds = new Set(news.map((v) => v.kind));
    if (type === 'item' && (kinds.size > 1 || (nums.length > 1 && !kinds.has(kind)))) {
      problems.push(`${f}:${lineOf(idx)}: ${all} → ${news.map((v) => `${v.kind} ${v.n}`).join('、')} (fix by hand)`);
      return all;
    }
    let i = 0;
    const newKind = type === 'item' ? news[0].kind : kind;
    return `${newKind}${sp}${news[0].n}` + rest.replace(/(?:\d+|[A-E])\.\d+/g, () => news[++i].n);
  });
  after = after.replace(REF_SECTION, (all, first, rest, tail, idx) => {
    if (!ours(first)) return all;
    const nums = [first, ...[...rest.matchAll(/(?:\d+|[A-E])\.\d+/g)].map((x) => x[0])];
    const news = nums.map((n) => map.sec.get(n));
    if (news.some((v) => !v)) { problems.push(`${f}:${lineOf(idx)}: ${all} → old section has no new home`); return all; }
    let i = 0;
    return news[0].n + rest.replace(/(?:\d+|[A-E])\.\d+/g, () => news[++i].n) + tail;
  });
  if (after !== before) {
    await writeFile(path, after);
    console.log(`updated references in ${f}`);
  }
}
for (const p of problems) console.log(`CHECK ${p}`);
