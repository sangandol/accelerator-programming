// Pure mathematical models shared by the reader's interactive figures and their checks.
// Loaded before widgets.js, including in a self-contained chapter export.
(() => {
  const bank = (row, col, xor = false) => xor ? row ^ col : col;
  function segments(stride = 1, offset = 0) {
    const addresses = Array.from({ length: 32 }, (_, i) => offset + 4 * stride * i);
    const touched = [...new Set(addresses.map(a => Math.floor(a / 32)))];
    return { addresses, touched, efficiency: 128 / (32 * touched.length) };
  }
  function pipeline({ count = 8, latency = 3, service = 1, compute = 2, slots = 3 } = {}) {
    const free = Array(slots).fill(0);
    let loadFree = 0, computeFree = 0;
    return Array.from({ length: count }, (_, i) => {
      const slot = i % slots;
      const load = Math.max(loadFree, free[slot]);
      const ready = load + latency;
      const start = Math.max(ready, computeFree);
      const end = start + compute;
      loadFree = load + service;
      computeFree = free[slot] = end;
      return { i, slot, load, ready, start, end };
    });
  }
  function merge(a, b) {
    if (!a.count) return { ...b };
    if (!b.count) return { ...a };
    const m = Math.max(a.m, b.m), x = Math.exp(a.m - m), y = Math.exp(b.m - m);
    return { m, l: a.l * x + b.l * y, u: a.u * x + b.u * y, count: a.count + b.count };
  }
  const empty = () => ({ m: -Infinity, l: 0, u: 0, count: 0 });
  function online(logits, values, block = 2, masked = []) {
    let state = empty();
    const history = [];
    for (let i = 0; i < logits.length; i += block) {
      let local = empty();
      for (let j = i; j < Math.min(i + block, logits.length); j++) {
        if (!masked.includes(j)) local = merge(local, { m: logits[j], l: 1, u: values[j], count: 1 });
      }
      const old = state;
      state = merge(state, local);
      history.push({ from: i, to: Math.min(i + block, logits.length), local, state,
        scale: old.count && state.count ? Math.exp(old.m - state.m) : null });
    }
    return history;
  }
  function shard(spec, x, y) {
    const rows = spec.includes('x') ? [4 * x, 4 * x + 4] : [0, 8];
    const cols = spec.includes('y') ? [4 * y, 4 * y + 4] : [0, 16];
    return { rows, cols, elements: (rows[1] - rows[0]) * (cols[1] - cols[0]) };
  }
  function grid(order = 'ijk') {
    const out = [];
    const visit = (level, p) => {
      if (level === 3) { out.push({ ...p }); return; }
      const axis = order[level];
      for (let v = 0; v < (axis === 'k' ? 3 : 2); v++) visit(level + 1, { ...p, [axis]: v });
    };
    visit(0, {});
    const runs = new Map();
    let prev = '', aPrev = '', bPrev = '', aLoads = 0, bLoads = 0;
    for (const p of out) {
      const key = `${p.i},${p.j}`;
      if (key !== prev) runs.set(key, (runs.get(key) ?? 0) + 1);
      if (`${p.i},${p.k}` !== aPrev) aLoads++;
      if (`${p.k},${p.j}` !== bPrev) bLoads++;
      prev = key; aPrev = `${p.i},${p.k}`; bPrev = `${p.k},${p.j}`;
    }
    return { points: out, aLoads, bLoads, contiguous: [...runs.values()].every(v => v === 1) };
  }
  function occupancy({ registers = 64, threads = 256, sharedKiB = 80 } = {}) {
    const limits = [Math.floor(65536 / (registers * threads)), Math.floor(224 / sharedKiB), Math.floor(2048 / threads), 16];
    const blocks = Math.min(...limits), warps = blocks * Math.ceil(threads / 32);
    return { limits, blocks, warps, fraction: warps / 64 };
  }
  function issue(count = 4, latency = 7, interval = 2, dependent = false) {
    return Array.from({ length: count }, (_, i) => {
      const start = i * (dependent ? Math.max(latency, interval) : interval);
      return { i, start, ready: start + latency };
    });
  }
  function ring(devices = 4, rounds = 0) {
    return Array.from({ length: devices }, (_, d) => Array.from({ length: rounds + 1 }, (_, r) => (d - r + devices) % devices).sort((a, b) => a - b));
  }
  function tile(bm = 128, bn = 128, bk = 128, buffers = 2) {
    const input = buffers * 2 * (bm * bk + bk * bn);
    const output = buffers * 2 * bm * bn;
    const accumulator = 4 * bm * bn;
    const bytes = input + output + accumulator;
    return { input, output, accumulator, bytes, intensity: 2 * bm * bn * bk / (2 * (bm * bk + bk * bn)) };
  }
  window.bookWidgetModels = { bank, segments, pipeline, empty, merge, online, shard, grid, occupancy, issue, ring, tile };
})();
