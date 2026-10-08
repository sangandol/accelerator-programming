// Verify mathematical invariants, chapter wiring and figure-spec uniqueness.
import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import { runInNewContext } from 'node:vm';
const context = { window: {} };
runInNewContext(await readFile('web/widget-models.js','utf8'),context);
const M=context.window.bookWidgetModels;
let checks=0;
const check=(value,message)=>{assert.ok(value,message);checks++;};
const near=(a,b)=>Math.abs(a-b)<=1e-10*Math.max(1,Math.abs(b));
check(M.segments(1,0).touched.length===4,'aligned warp');
check(M.segments(1,4).touched.length===5,'one f32 offset adds a segment');
check(M.segments(32,0).touched.length===32,'stride 32');
for(let r=0;r<4;r++) {
  check(new Set(Array.from({length:4},(_,c)=>M.bank(r,c,true))).size===4,'XOR row is a permutation');
  check(new Set(Array.from({length:4},(_,c)=>M.bank(c,r,true))).size===4,'XOR column is a permutation');
}
for(const latency of [1,3,7])for(const compute of [1,2,4])for(const slots of [1,2,3,5,8]) {
  const rows=M.pipeline({latency,compute,slots}),last=new Map();
  for(const [i,x] of rows.entries()) {
    check(x.start>=x.ready,'read-before-compute');
    check(x.load>=(last.get(x.slot)??0),'reuse after all readers finish');
    if(i) {check(x.start>=rows[i-1].end,'one compute resource');check(x.load>=rows[i-1].load+1,'load service interval');}
    last.set(x.slot,x.end);
  }
  if(slots===1)check(rows.at(-1).end===8*(latency+compute),'serial boundary');
  if(slots>=1+Math.ceil(latency/compute))check(rows.at(-1).end===latency+8*compute,'sufficient buffering attains steady cadence');
}
for(const logits of [[2,-1,4,0,3,1],[1000,1001,999,-1000],[1]]) {
  const values=logits.map((_,i)=>10*(i+1)),max=Math.max(...logits);
  const denominator=logits.reduce((s,x)=>s+Math.exp(x-max),0);
  const numerator=logits.reduce((s,x,i)=>s+Math.exp(x-max)*values[i],0);
  for(let size=1;size<=logits.length;size++) {
    const state=M.online(logits,values,size).at(-1).state;
    check(near(state.u/state.l,numerator/denominator),'online attention agrees with direct stable evaluation');
  }
  const empty=M.online(logits,values,2,logits.map((_,i)=>i)).at(-1).state;
  check(empty.count===0&&empty.l===0&&empty.u===0&&empty.m===-Infinity,'all-masked identity avoids NaN');
}
for(const order of ['ijk','jik','kij']) {
  const grid=M.grid(order);
  check(grid.points.length===12,'grid cardinality');
  check(new Set(grid.points.map(x=>`${x.i},${x.j},${x.k}`)).size===12,'each iteration covered once');
  check(grid.contiguous===(order!=='kij'),'K residency condition');
}
for(const spec of ['xy','x','y','']) {
  const cover=Array.from({length:8},()=>Array(16).fill(0));
  for(let x=0;x<2;x++)for(let y=0;y<4;y++) {
    const s=M.shard(spec,x,y);
    for(let i=s.rows[0];i<s.rows[1];i++)for(let j=s.cols[0];j<s.cols[1];j++)cover[i][j]++;
  }
  const copies=(spec.includes('x')?1:2)*(spec.includes('y')?1:4);
  check(cover.flat().every(x=>x===copies),'global coverage and replication');
}
for(const n of [3,4,8]) {
  check(M.ring(n,0).every((row,d)=>row.length===1&&row[0]===d),'ring initial owners');
  check(M.ring(n,n-1).every(row=>row.length===n&&new Set(row).size===n),'all-gather final coverage');
}
const occupancy=M.occupancy();
check(occupancy.blocks===2&&occupancy.warps===16&&occupancy.fraction===.25,'worked SM budget');
check(M.issue(4,7,2,false).at(-1).ready===13,'independent service cadence');
check(M.issue(4,7,2,true).at(-1).ready===28,'dependent critical path');
const tile=M.tile(512,512,1024,2);
check(tile.bytes===6*2**20,'chapter 16 six-MiB buffer budget');
check(M.tile(512,512,4096,2).bytes===18*2**20,'larger K exceeds v4 VMEM');

const widgets=await readFile('web/widgets.js','utf8');
const names=new Set(widgets.match(/const WIDGETS = \{([^}]+)\}/)[1].split(',').map(x=>x.trim()));
let mounts=0;
for(const file of (await readdir('book')).filter(f=>f.endsWith('.md'))) {
  const text=await readFile(`book/${file}`,'utf8');
  for(const [,name] of text.matchAll(/data-widget="([^"]+)"/g)) {
    check(names.has(name),`${file}: registered widget ${name}`);mounts++;
    const place=text.indexOf(`data-widget="${name}"`);
    const section=text.slice(text.lastIndexOf('\n## ',place),text.indexOf('\n## ',place)>0?text.indexOf('\n## ',place):text.length);
    if (!file.startsWith('01-')) check(section.includes('![图 '),`${file}: static figure fallback for ${name}`);
  }
}
const seen=new Set();
for(const file of (await readdir('tools/fig')).filter(f=>/^ch\w+\.mjs$/.test(f))) {
  const source=await readFile(`tools/fig/${file}`,'utf8');
  for(const [, , name] of source.matchAll(/^\s*(['"])(\d{2}-[\w-]+)\1\s*(?:\(|:)/gm)) {
    check(!seen.has(name),`unique figure spec ${name}`);seen.add(name);
  }
}
// Branch independence: GPU chapters may use part III only in the explicitly optional §24.4.
for(const file of (await readdir('book')).filter(f=>/^(2[1-7])-/.test(f))) {
  let source=await readFile(`book/${file}`,'utf8');
  if(file.startsWith('24-'))source=source.replace(/## 24\.4 [\s\S]*?(?=\n## )/,'');
  check(!/(?:定义|命题|定理|引理|推论|例|注)\s?(?:1[3-9]|20)\.\d+/.test(source),`${file}: no required TPU/JAX item dependency`);
}
console.log(`${checks} invariant / wiring checks passed; ${mounts} widget mounts, ${seen.size} unique figure specs.`);
