# 第 7 章　多颗芯片：互联与集合通信

一颗芯片的存储放不下大模型，算力也不够在可接受的时间内训练它。于是要把许多芯片连起来。本章回答：芯片之间怎样连接？多颗芯片一起完成一次归约、一次收集，需要多少时间？有没有下界？

> **在体系中的位置**：下层是第 4 章的 α–β 传输模型和第 6 章的抽象芯片。本章给出链路、拓扑、集合操作的定义、算法与代价。第 12 章用它分析各种并行策略；第 14 章的 JAX 集合原语、第 19 章的 TPU 多芯片编程、第 27 章的多 GPU 编程都是本章的具体实现。

## 7.1 链路与拓扑

**定义 7.1（链路）** 两颗芯片之间的**链路**是双向的点对点连接，每个方向的带宽为 $\beta$，延迟为 $\alpha$。通过一条链路发送 $m$ 字节需要 $\alpha + m/\beta$。一颗芯片的所有链路可以同时收发。

芯片之间的链路带宽通常比 HBM 带宽小一到两个数量级（几十到几百 GB/s 对每秒数 TB），所以多芯片程序的首要问题是少传输、以及让传输与计算重叠。

**定义 7.2（拓扑与二分带宽）** 芯片与链路组成的图称为**拓扑**。把芯片分成数量相等的两半，穿过这个划分的链路在一个方向上的总带宽，对所有划分取最小值，称为**二分带宽**。

常见的拓扑：

| 拓扑 | 每颗芯片的链路数 | 直径（跳数） | 二分带宽 |
| --- | ---: | ---: | ---: |
| 环，$p$ 颗芯片 | 2 | $\lfloor p/2 \rfloor$ | $2\beta$ |
| $d$ 维环面 $k^d$（$p = k^d$） | $2d$ | $d \lfloor k/2 \rfloor$ | $2 k^{d-1} \beta$ |
| 交换机（无阻塞），$p$ 颗芯片 | 1（接到交换机） | 2 | $\frac{p}{2} \beta$ |

环面（torus）是每一维都首尾相接的网格：$k^d$ 环面中，芯片 $(x_1, \dots, x_d)$ 与每一维上 $x_i \pm 1 \pmod k$ 的芯片相连。环面的链路都很短，便于大规模铺开，但二分带宽只随 $p^{(d-1)/d}$ 增长。交换机让任意两颗芯片之间都能以全带宽通信，但交换机本身的规模有限，大的系统要多级交换。

![图 7.1](fig/07-topologies.svg)

**图 7.1**　三种拓扑。(a) 8 颗芯片的环，红色虚线把它分成两半，只切断 2 条链路，所以二分带宽是 $2\beta$；(b) 二维环面，每一行、每一列都首尾相接；(c) 所有芯片接到一台交换机。

实际系统常是分层的：一组芯片以高带宽互联（一个环面切片，或一台交换机下的一组），组与组之间通过较慢的数据中心网络相连。

**命题 7.3（全交换的二分下界）** 若 $p$ 颗芯片中每一颗都要向其余每一颗发送 $m$ 字节（**全交换**，all-to-all），则所需时间至少为

$$
\frac{(p/2)^2\, m}{W},
$$

其中 $W$ 是二分带宽。

*证明* 取达到二分带宽的划分。一半中的 $p/2$ 颗芯片各要向另一半的 $p/2$ 颗芯片各发 $m$ 字节，共 $(p/2)^2 m$ 字节，全部要穿过这个划分，单方向带宽为 $W$。∎

在环上 $W = 2\beta$，时间至少 $p^2 m / (8\beta)$，随 $p^2$ 增长；在交换机上 $W = p\beta/2$，时间至少 $pm/(2\beta)$，与每颗芯片要发出的总量 $(p-1)m$ 同阶。全交换是环和环面的弱项。

## 7.2 集合操作

**定义 7.4（集合操作）** 设有 $p$ 个参与者（rank），编号 $0, \dots, p - 1$。把长度为 $n$ 的向量分成 $p$ 块：$x = (x^0, x^1, \dots, x^{p-1})$，每块长度 $n/p$。rank $r$ 持有的数据记作 $x_r$。

- **all-gather**：rank $r$ 只持有一块 $x_r^r$；结束时每个 rank 都得到 $(x_0^0, x_1^1, \dots, x_{p-1}^{p-1})$。
- **reduce-scatter**：rank $r$ 持有完整的 $x_r$；结束时 rank $r$ 得到 $\sum_q x_q^r$，即总和的第 $r$ 块。
- **all-reduce**：rank $r$ 持有 $x_r$；结束时每个 rank 都得到 $\sum_q x_q$。
- **all-to-all**：rank $r$ 持有 $x_r$；结束时 rank $q$ 得到 $(x_0^q, x_1^q, \dots, x_{p-1}^q)$。把 $p \times p$ 个块排成矩阵，第 $r$ 行是 rank $r$ 的数据，all-to-all 就是这个块矩阵的转置。
- **broadcast**、**reduce**：从一个 rank 发给所有 rank；把所有 rank 的数据归约到一个 rank。
- **置换**（collective permute）：每个 rank 把自己的数据发给 $\sigma(r)$，$\sigma$ 是一个置换。

![图 7.2](fig/07-collectives.svg)

**图 7.2**　4 个 rank 上的四种集合操作，箭头左边是之前、右边是之后。每一列是一个 rank 的数据，$A, B, C, D$ 分别是 rank 0–3 原有的数据，下标是块号。

这里的"求和"可以换成任何满足结合律的运算（最大值、按位或等），与第 1 章相同。浮点数的求和顺序会影响结果的末位（2.6 节）。

**命题 7.5** all-reduce 等于先做 reduce-scatter、再做 all-gather。

*证明* reduce-scatter 之后 rank $r$ 持有 $\sum_q x_q^r$；all-gather 把这 $p$ 块拼起来，每个 rank 得到 $(\sum_q x_q^0, \dots, \sum_q x_q^{p-1}) = \sum_q x_q$。∎

## 7.3 代价与算法

本节假定每个 rank 每个方向的总注入带宽为 $\beta$（例如环上只用一个方向的一条链路），并用 α–β 模型计时。

**命题 7.6（带宽下界）** all-gather 和 reduce-scatter 的时间至少为 $\frac{p-1}{p} \cdot \frac{n}{\beta}$。

*证明* all-gather：每个 rank 必须收到其余 $p - 1$ 块，共 $\frac{p-1}{p} n$ 字节，接收带宽为 $\beta$。reduce-scatter：rank $r$ 的结果依赖于其余每个 rank 的第 $r$ 块；一般而言这些块不可压缩，所以每个 rank 必须为其余 $p - 1$ 个 rank 各发出至少 $n/p$ 字节的信息，共 $\frac{p-1}{p} n$ 字节。∎

由命题 7.5，all-reduce 的带宽项不超过 $2\frac{p-1}{p} \frac{n}{\beta}$；可以证明这也是下界（Patarasuk 与 Yuan，2009）。

**环形算法。** 把 rank 排成环，每个 rank 只向右邻居发送。

- **环形 all-gather**：第 $t$ 步（$t = 1, \dots, p - 1$），每个 rank 把上一步收到的块（第一步是自己的块）转发给右邻居。$p - 1$ 步后每个 rank 都收到了全部块。
- **环形 reduce-scatter**：同样的 $p - 1$ 步，但每个 rank 把收到的部分和加上自己对应的块，再转发。
- **环形 all-reduce**：二者接连进行。

![图 7.3](fig/07-ring-allgather.svg)

**图 7.3**　4 个 rank 的环形 all-gather。每一步每个 rank 把上一步收到的块（红框）发给右邻居，3 步之后每个 rank 都有了全部 4 块。

**命题 7.7（环形算法的时间）** 环形 all-gather 与 reduce-scatter 各用时 $(p-1)\left(\alpha + \frac{n}{p\beta}\right)$，环形 all-reduce 用时 $2(p-1)\left(\alpha + \frac{n}{p\beta}\right)$。带宽项达到命题 7.6 的下界。

*证明* 每步每个 rank 发送一块 $n/p$ 字节，所有 rank 并行。∎

若链路是双向的，把数据分成两半，在两个方向上各跑一个环，带宽项再减半。在 $d$ 维环面上，同样的思路可以用上全部 $2d$ 条链路。

**树形与倍增算法。** 环形算法有 $p - 1$ 步，每步付一次 $\alpha$；当 $n$ 很小时，$\alpha$ 项主导。**递归倍增**的 all-gather 用 $\log p$ 步：第 $t$ 步与距离 $2^{t}$ 的 rank 交换目前持有的全部数据，数据量逐步加倍。（图 7.4）

![图 7.4](fig/07-recursive-doubling.svg)

**图 7.4**　8 个 rank 的递归倍增。每一步 rank $i$ 与 rank $i \oplus 2^t$（按位异或）交换数据，3 步之后每个 rank 都汇集了全部数据。

**命题 7.8** 递归倍增 all-gather 用时 $\alpha \log p + \frac{p-1}{p} \frac{n}{\beta}$（与距离 $2^t$ 的 rank 之间有直接链路时）。递归减半 reduce-scatter 对称；二者组合得到 all-reduce，用时 $2\alpha \log p + 2\frac{p-1}{p}\frac{n}{\beta}$。

*证明* 第 $t$ 步交换 $2^t n/p$ 字节，$\sum_{t=0}^{\log p - 1} 2^t n/p = (p-1) n/p$。∎

在只有相邻链路的环上，"距离 $2^t$"的交换要经过多跳，与别的交换争用链路，所以环上大数据量时仍用环形算法。一般的选择规则是：**小数据量时减少步数（$\alpha$ 项），大数据量时用满带宽（$\beta$ 项）**。

**多维环面上的 all-reduce。** 在 $k \times k$ 的二维环面上，先沿第一维（每一行是一个环）做 reduce-scatter，每个 rank 剩下 $n/k$；再沿第二维对这 $n/k$ 做 reduce-scatter，剩下 $n/k^2$；然后按相反的顺序做两次 all-gather。

![图 7.5](fig/07-torus-2d.svg)

**图 7.5**　$3 \times 3$ 环面上 all-reduce 的前半部分：先在每一行的环上做 reduce-scatter，再在每一列的环上对剩下的 $n/k$ 字节做 reduce-scatter。

**命题 7.9** 上述算法的带宽项为 $2\frac{k-1}{k}\left(\frac{n}{\beta} + \frac{n}{k\beta}\right)$，约为 $2n/\beta$，与同样 $p = k^2$ 个 rank 的单个大环相同，而步数从 $2(k^2 - 1)$ 降到 $4(k - 1)$。

*证明* 第一维的 reduce-scatter 作用在 $n$ 字节上，带宽项 $\frac{k-1}{k} \frac{n}{\beta}$；第二维作用在 $n/k$ 上，带宽项 $\frac{k-1}{k} \frac{n}{k\beta}$。all-gather 对称。∎

## 7.4 通信与计算的重叠

集合操作由一步步的点对点传输组成，每一步之间可以插入计算。以环形 all-gather 为例：若最终要用收集到的全部块去做矩阵乘法，那么每收到一块就可以先乘这一块，同时接收下一块。

**命题 7.10** 环形 all-gather 共 $p - 1$ 步，每步通信时间为 $t_c$；$p$ 块（含自己的一块）每块的计算时间为 $t_m$。若第 $j$ 块的计算可以与之后各块的通信同时进行，则总时间为

$$
\max\{\, p\,t_m,\ (p-1)\,t_c + t_m \,\}.
$$

当 $t_m \ge t_c$ 时，总时间就是纯计算的时间 $p\,t_m$，通信被完全隐藏。

*证明* 第 $j$ 块（$j = 0$ 是自己的块）在时刻 $j t_c$ 到达，第 $j$ 块的计算在它到达且第 $j - 1$ 块算完之后开始。记 $F_j$ 为第 $j$ 块算完的时刻，则 $F_j = \max(F_{j-1}, j t_c) + t_m$，展开得 $F_{p-1} = \max_j \{ j t_c + (p - j) t_m \}$。括号内是 $j$ 的线性函数，最大值在 $j = 0$ 或 $j = p - 1$ 处取到。∎

![图 7.6](fig/07-overlap.svg)

**图 7.6**　命题 7.10 的时间线（$p = 4$，$t_c = 2$，$t_m = 3$）。每块到达之后就开始计算；因为 $t_m \ge t_c$，计算单元从不空等，总时间是 $4 t_m = 12$。

第 12 章会把这一点用到具体的并行策略上。

## 7.5 谁来搬运，怎样同步

芯片之间的数据由谁来搬？常见的有三种：

- 芯片上的 DMA 引擎直接把数据写进另一颗芯片的存储（**远程 DMA**），并在接收方的完成计数器上加数（定义 4.11），接收方据此知道数据已到；
- 处理器的线程直接对另一颗芯片的存储发出读写指令；
- 经由主机和网卡转发，用于跨越较慢的网络。

无论哪种，跨芯片的传输都需要双向的约定：接收方要知道数据**已经到了**（完成计数）；发送方要知道接收方的缓冲**已经空出来了**，否则会覆盖对方还没用完的数据。后者通常靠接收方反向发一个信号（"信用"）来实现。第 9 章会把这类约定写成先行发生关系，严格地讨论它们。

![图 7.7](fig/07-credit.svg)

**图 7.7**　跨芯片传输的双向约定：数据和完成信号从发送方流向接收方，信用从接收方流回发送方。

## 接口小结

1. 链路：每方向带宽 $\beta$、延迟 $\alpha$；芯片间带宽比 HBM 小一到两个数量级。（定义 7.1）
2. 环、环面、交换机的直径与二分带宽见 7.1 节的表；全交换的时间至少 $(p/2)^2 m / W$，环和环面上随 $p$ 增长很快。（定义 7.2、命题 7.3）
3. 集合操作：all-gather、reduce-scatter、all-reduce、all-to-all（块矩阵的转置）、broadcast、reduce、置换；all-reduce = reduce-scatter 后接 all-gather。（定义 7.4、命题 7.5）
4. all-gather 与 reduce-scatter 的带宽项至少 $\frac{p-1}{p}\frac{n}{\beta}$，all-reduce 至少两倍；环形算法达到它，用时 $(p - 1)(\alpha + n/(p\beta))$ 每次。小数据量用倍增算法减少 $\alpha$ 项。（命题 7.6–7.8）
5. 多维环面逐维做集合操作，带宽项不变、步数大减。（命题 7.9）
6. 集合操作逐步进行，可与计算流水重叠；每步计算不少于通信时，通信几乎被隐藏。（命题 7.10）
7. 跨芯片传输需要"数据已到"与"缓冲已空"两个方向的同步。（7.5 节）

## 习题

**习题 7.1** ★ 求下列拓扑的直径和二分带宽（每条链路每方向带宽 $\beta$）：16 颗芯片的环；$4 \times 4$ 环面；$4 \times 4 \times 4$ 环面；64 颗芯片接在一台无阻塞交换机上。

<details><summary>提示</summary>

用 7.1 节的表。环面的二分要切断 $k^{d-1}$ 条链路两次（因为有回绕）。

</details>
<details><summary>答案</summary>

16 环：直径 8，二分 $2\beta$。$4 \times 4$ 环面：直径 4，二分 $2 \cdot 4\beta = 8\beta$。$4^3$ 环面：直径 6，二分 $2 \cdot 16 \beta = 32\beta$。64 交换机：直径 2，二分 $32\beta$。$4^3$ 环面与 64 交换机的二分带宽相同，但环面每颗芯片要 6 条链路。

</details>

**习题 7.2** ★ 8 颗芯片组成环，$\beta = 50$ GB/s，$\alpha = 5$ µs。对 $n = 1$ GB 的数据做 all-reduce。(a) 用环形算法需要多少时间？(b) 若链路双向、分两半各跑一个环呢？(c) 若 $n = 8$ KB 呢？此时 $\alpha$ 项占多少？

<details><summary>提示</summary>

命题 7.7：$2(p-1)(\alpha + n/(p\beta))$。

</details>
<details><summary>答案</summary>

(a) $n/(p\beta) = 10^9 / (8 \times 5 \times 10^{10}) = 2.5$ ms；$14 \times (5\ \mu\text{s} + 2.5\ \text{ms}) \approx 35.1$ ms。(b) 带宽项减半，约 17.6 ms。(c) $n/(p\beta) = 0.02$ µs，$14 \times 5.02 \approx 70$ µs，几乎全是 $\alpha$。小消息应当改用步数少的算法，或者把多个小的集合操作合并成一个大的。

</details>

**习题 7.3** ★ 在每对 rank 之间都有直接链路的网络上，比较环形 all-reduce 与递归减半/倍增 all-reduce。当 $p = 64$、$\alpha = 5$ µs、$\beta = 100$ GB/s 时，数据量多大时两者相同？

<details><summary>提示</summary>

两者的带宽项相同（命题 7.7、7.8），只有 $\alpha$ 项不同。

</details>
<details><summary>答案</summary>

带宽项都是 $2\frac{p-1}{p}\frac{n}{\beta}$，$\alpha$ 项分别是 $2(p-1)\alpha = 126\alpha$ 和 $2\alpha\log p = 12\alpha$。所以在这种网络上递归算法总是不差于环形算法。环形算法的优势只在于它只需要相邻链路：在环或环面上，递归算法的远距离交换会多跳并争用链路，带宽项变差。

</details>

**习题 7.4** ★ 64 颗芯片，每颗芯片要向其余每颗芯片各发 1 MB（例如混合专家模型中把词元分发给专家）。$\beta = 100$ GB/s。分别在 64 环和无阻塞交换机上，用命题 7.3 求时间下界。

<details><summary>提示</summary>

$W$ 分别为 $2\beta$ 和 $32\beta$。

</details>
<details><summary>答案</summary>

$(p/2)^2 m = 1024 \times 1\ \text{MB} \approx 1.07 \times 10^9$ 字节。环：$/ (2 \times 10^{11}) \approx 5.4$ ms。交换机：$/(3.2 \times 10^{12}) \approx 0.34$ ms。相差 16 倍。所以大量全交换的工作负载偏好交换机或高维的拓扑。

</details>

**习题 7.5** ★ 在 $8 \times 8$ 的二维环面上对 $n$ 字节做 all-reduce，用命题 7.9 的算法。(a) 每一阶段每个 rank 持有多少数据？(b) 总步数是多少？与 64 个 rank 的单环比较。

<details><summary>提示</summary>

四个阶段：第一维 reduce-scatter、第二维 reduce-scatter、第二维 all-gather、第一维 all-gather。

</details>
<details><summary>答案</summary>

(a) 开始 $n$；第一维 reduce-scatter 后 $n/8$；第二维 reduce-scatter 后 $n/64$；第二维 all-gather 后 $n/8$；第一维 all-gather 后 $n$。(b) 每阶段 7 步，共 28 步；单环为 $2 \times 63 = 126$ 步。带宽项约 $2 \cdot \frac{7}{8}(1 + \frac{1}{8}) \frac{n}{\beta} \approx 1.97 \frac{n}{\beta}$，与单环的 $2 \cdot \frac{63}{64}\frac{n}{\beta} \approx 1.97 \frac{n}{\beta}$ 相同。

</details>

**习题 7.6** ★（审查题）AI 为 $p$ 颗芯片的 all-reduce 写了这样的方案："每个 rank 把自己的完整向量发给其余每个 rank，然后各自求和。"在环上和在无阻塞交换机上，分别估算它的时间，与环形算法比较。

<details><summary>提示</summary>

每个 rank 要发出 $(p-1)n$ 字节。在环上，这还是一次全交换（命题 7.3），$m = n$。

</details>
<details><summary>答案</summary>

交换机上：每个 rank 发出 $(p-1)n$ 字节，至少 $(p-1)n/\beta$，是环形算法 $2\frac{p-1}{p}\frac{n}{\beta}$ 的 $p/2$ 倍。环上：由命题 7.3 至少 $p^2 n / (8\beta)$，是环形算法的约 $p^2/16$ 倍。$p = 64$ 时分别慢 32 倍和 256 倍。正确的做法是 reduce-scatter 加 all-gather。

</details>

**习题 7.7** ☆ 把环形 all-gather 与矩阵乘法重叠（命题 7.10）：$p = 8$，每步通信 $t_c = 1$ ms，每块计算 $t_m = 1.5$ ms。总时间是多少？与不重叠相比节省多少？

<details><summary>提示</summary>

不重叠时是 $(p-1)t_c + p\,t_m$（自己那块也要算）。重叠时用命题 7.10。

</details>
<details><summary>答案</summary>

不重叠：$7 + 12 = 19$ ms。重叠：$\max\{8 \times 1.5,\ 7 \times 1 + 1.5\} = \max\{12, 8.5\} = 12$ ms，通信完全隐藏，节省约 37%。

</details>
