// Builds book/fig/*.svg from the specs in tools/fig/chNN.mjs.
//   node tools/figures.mjs            build all figures
//   node tools/figures.mjs 09 01-mux  build only figures whose names start with one of the prefixes
//   node tools/figures.mjs --demo     build the component samples in tools/fig/demo.mjs into tools/fig/demo/
// Prints label-overlap warnings, and lists SVGs that no spec produces or no chapter uses.
import { readdir, readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Fig } from './fig/lib.mjs';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const SPECS = join(ROOT, 'tools', 'fig');
const demo = process.argv.includes('--demo');
const OUT = demo ? join(SPECS, 'demo') : join(ROOT, 'book', 'fig');
const filters = process.argv.slice(2).filter((a) => a !== '--demo');

await mkdir(OUT, { recursive: true });
const names = new Set();
let built = 0;
let warnings = 0;
for (const file of (await readdir(SPECS)).filter((f) => (demo ? f === 'demo.mjs' : /^ch\w+\.mjs$/.test(f))).sort()) {
  const specs = (await import(pathToFileURL(join(SPECS, file)))).default;
  for (const [name, make] of Object.entries(specs)) {
    names.add(`${name}.svg`);
    if (filters.length && !filters.some((p) => name.startsWith(p))) continue;
    const f = new Fig();
    const result = make(f);
    const drawn = result instanceof Fig ? result : f;
    await writeFile(join(OUT, `${name}.svg`), drawn.svg());
    for (const w of drawn.lint()) {
      console.log(`${name}: ${w}`);
      warnings++;
    }
    built++;
  }
}

if (demo) {
  console.log(`${built} demo figure(s) built in tools/fig/demo, ${warnings} warning(s).`);
  process.exit(0);
}
const book = (await Promise.all((await readdir(join(ROOT, 'book'))).filter((f) => f.endsWith('.md')).map((f) => readFile(join(ROOT, 'book', f), 'utf8')))).join('\n');
for (const svg of (await readdir(OUT)).filter((f) => f.endsWith('.svg'))) {
  if (!names.has(svg)) console.log(`${svg}: no spec produces it (stale?)`);
  else if (!book.includes(`fig/${svg}`)) console.log(`${svg}: not used in any chapter`);
}
console.log(`${built} figure(s) built, ${warnings} warning(s).`);
