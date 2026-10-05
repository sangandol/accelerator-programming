# 画图库

书中插图全部由这里的代码生成：`lib.mjs` 是画图库，`chNN.mjs` 是第 NN 章的插图，输出到 `book/fig/<名字>.svg`。**画新图只需读本文件和对应章的 chNN.mjs，不必读 lib.mjs。**

## 流程

1. 在 `tools/fig/chNN.mjs`（没有就新建，照 ch01.mjs 的样子）的 `export default {}` 里加一项：`'NN-name'(f) { ... }`。`f` 是一张空白的图；坐标单位是像素，y 向下；画布自动包住所有内容，不用设宽高。
2. `npm run fig -- NN-name`（或 `node tools/figures.mjs NN`，按前缀只生成这些图）。输出的警告：`text overlap`（两段文字重叠）、`line through text`（线穿过文字）、`label wider than its box/cell`（文字比框窄不下，要求两边各留约 4 像素）、`unknown math command`（数学段里用了不支持的命令）、`NaN coordinates`（用了不存在的锚点）、`not used in any chapter`（还没在正文中引用）。警告为 0 通常就不必再看渲染结果。
3. 正文中写 `![图 N.k](fig/NN-name.svg)`，下一行写 `**图 N.k**　说明……`。
4. 需要看效果时打开阅读器的插图页 `/book/fig/?f=NN`（按前缀筛选）。
5. `node tools/figures.mjs --demo` 生成 `demo.mjs` 中各组件的样例到 `tools/fig/demo/`，改了 lib.mjs 之后用它做回归。

## 约定

- **文字**：`'普通文字 $数学$'`。数学段里字母自动斜体，`x_0`、`x_{i+1}`、`2^k`、`c'`（撇号）可用，常用命令有 `\otimes \oplus \wedge \vee \neg \cdot \times \le \ge \ne \to \gets \dots \infty \alpha \tau \log \max` 等。中文写在 `$` 外。
- **颜色**：名字 `blue orange green red purple yellow teal gray white`（浅色填充 + 深色描边），或任意十六进制色。线的颜色用名字时取深色。
- **层次**：先画形状，再画线、圆点、文字，与调用顺序无关（`z` 参数可以改）。
- **锚点**：`box`/`rect`/`grid.cell` 返回的对象有 `x y w h cx cy`，以及边上的点 `L(t) R(t) T(t) B(t)`（`t` 是沿边的比例，默认 0.5）。`node`/`circle` 返回圆。凡是要一个点的地方，都可以传 `[x, y]` 或这些对象（取中心）。

## API

| 调用 | 作用 |
| --- | --- |
| `f.text(x, y, s, {size=14, anchor='middle', color, weight})` | 文字，`(x, y)` 是竖直中心 |
| `f.note(x, y, s, opts)` | 灰色小字（12 号） |
| `f.box(x, y, w, h, label, {color='blue', size, rx, dash, hollow})` | 带居中文字的方框；`hollow: true` 只画边框（分组框，配 `dash: '5 4'`） |
| `f.rect(x, y, w, h, {fill, stroke, width, rx, dash})` | 裸矩形 |
| `f.node(cx, cy, label, {r=13, color, size, solid})` | 带文字的圆（图的节点） |
| `f.circle(cx, cy, r, {fill, stroke})`、`f.dot(x, y, r=3.2, color)` | 圆、实心点 |
| `f.line(points, {color, width=1.6, dash, arrow})` | 折线；`arrow: 'end' / 'start' / 'both'` |
| `f.arrow(p, q, opts)` | 箭头 |
| `f.link(a, b, {bend, arrow, color, dash})` | 从形状 a 的边到形状 b 的边连线；`bend` 非零时为弧线（正值向 a→b 的左侧弯） |
| `f.brace(x0, y, x1, label, {flip})`；`{vertical: true}` 时为 `(x, y0, y1)` | 标注一段范围的方括号 |
| `f.poly(points, {color, fill, stroke, dash})` | 填充多边形（梯形、楔形等） |
| `f.raw(svg, [x0, y0, x1, y1], z)` | 直接写 SVG（给出外框以便自动定尺寸） |

**电路**：`f.gate(name, 'and'|'or'|'xor'|'not'|文字, x, y)` 画 44×44 的门，引脚为 `name.in0`、`name.in1`（左侧上下）、`name.in`（单输入）、`name.out`。`f.pin(name, x, y, label, 'in'|'out'|'top'|'bottom')` 定义端点并标字。`f.net(src, [dst...], x)` 从 src 横走到竖线 `x`，再分别横走到各 dst，T 形分叉处自动画圆点；`x` 省略时取中点。

**组件**（返回对象供后续标注）：

| 调用 | 用途 | 返回 |
| --- | --- | --- |
| `f.colnet(x0, y0, {n, ops: [[级, i, j], ...], dx=52, dy=56, top='x', bottom='y', levelLabel})` | 前缀网络、归约树、蝶形：每列一个位置，`[级, i, j]` 表示第 i 列在该级结合第 j 列 | `{X(i), Y(级), end}` |
| `f.grid(x0, y0, {rows, cols, cw=26, ch=26, fill(r,c), label(r,c), rowLabels, colLabels})` | 矩阵、布局、分块；`fill`/`label` 返回 null 表示空白 | `{cell(r,c), region(r0,c0,r1,c1), x, y, w, h}` |
| `f.frame(region, {color='red', label, side='top', dash})` | 粗框标出一块；格子里的块用 `side: 'right'` 或 `'bottom'` | — |
| `f.timeline(x0, y0, {lanes, bars: [[道, t0, t1, label, color]], unit=36, deps: [[i, j]], ticks, axisLabel})` | 流水线、搬运与计算的重叠；`deps` 画第 i 条到第 j 条的依赖箭头 | `{bars, tx(t), ly(道)}` |
| `f.plot(x0, y0, {w, h, x: {min, max, log, label, ticks}, y: {...}, series: [{pts, color, dash, label, at}], marks: [{at, label}]})` | 屋顶线等坐标图 | `{sx, sy}` |
| `f.ring(cx, cy, {n, r, label(i)})` | 环形拓扑；配 `f.link(R.node(i), R.node(j), {bend, arrow})` 画通信 | `{node(i)}` |
| `f.mesh(x0, y0, {rows, cols, d=64, wrap, label(r,c)})` | 网格 / 环面（`wrap` 画回绕的虚线） | `{node(r,c)}` |
| `f.bitfield(x0, y0, {fields: [[位数, label, color]], unit=16})` | 数值格式的位布局，高位在左 | 各字段的方框 |
| `f.numline(x0, y, {min, max, w, ticks, labels, minor})` | 数轴（浮点数的分布、舍入） | 值→x 的映射 |
| `f.bars(x0, y0, {items: [[label, value, color]], w, log, fmt})` | 横向条形图（能耗、带宽的比较） | `{len, y(i)}` |

**三维示意**：`import { oblique } from './lib.mjs'`，`const p = oblique(x0, y0, ei, ej, ek)`，`p(i, j, k)` 给出斜投影后的点（例：ch05.mjs 的迭代空间、ch06.mjs 的 Loomis–Whitney 图）。

## 省 token 的写法

- 有规律的图用循环和 `map` 生成，不逐个写坐标（例：ch01.mjs 的 Kogge–Stone 网络一行写完）。
- 先选组件；组件不够时用 `box`、`node`、`link`、`net` 拼；同一种结构第二次出现时，把它做成 lib.mjs 里的新组件，并在 demo.mjs 加一个样例、在本表加一行。
- 靠 lint 检查，不必每次截图；只在新组件或复杂的图上看一次插图页（把整章的图排在一页上一起看）。
- 往正文插图时，锚点句必须是段落的最后一句，否则说明会和后面的文字连成一段。
