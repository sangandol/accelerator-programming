# 第 26 章　GPU kernel：注意力

注意力的中间矩阵是 $T \times T$ 的，不能写回显存；第 10 章的在线 softmax（命题 10.5）让我们一块一块地处理键和值，只保留每个查询的状态 $(m, \ell, u)$。本章把它写成 GPU 上的 kernel。26.1 节先估出上限；26.2 节讲 FlashAttention 的基本结构与 warp 的分工；26.3、26.4 节说明它在 Hopper 与 Blackwell 上为什么要重新设计（指数运算成了瓶颈），以及怎样设计；26.5 节讲 decode 时的注意力与分页 KV cache；26.6 节讲反向传播。

> **在体系中的位置**：下层是第 10 章的在线 softmax、第 11 章的注意力算术、第 21–25 章的 GPU 结构与 kernel 写法；反向传播所需的求导在 26.6 节直接推出。本章给出 FlashAttention 的结构与负载分析、Hopper 与 Blackwell 上的设计、decode 注意力、分页 KV、反向传播。

## 26.1 注意力的屋顶线

按惯例，写 kernel 之前先估上限。考虑一个头：$Q, K, V \in \mathbb{R}^{T \times d}$，$O = \operatorname{softmax}(QK^{\mathsf T}/\sqrt{d})\, V$，运算量约 $4T^2 d$（不计掩码）。若中间矩阵不离开片上存储，每个查询块（$b_q$ 行）要读一遍全部的 $K$、$V$，强度约为 $\frac{4T^2d}{2sTd \cdot T/b_q} = \frac{2b_q}{s}$，与 $T$ 无关。所以：

- **prefill**（或训练）：$b_q$ 取 128 左右即可计算受限；
- **decode**：每个序列只有一个（或少数几个）查询，强度约为 $2 \times (\text{同时处理的查询数}) / s$，访存受限，时间由读 KV cache 决定。

## 26.2 FlashAttention 的结构

本节把命题 10.5 落实为线程块的工作流程，再说明一个线程块内的 warp 应当怎样分工。

**网格**：每个线程块负责一个（批量，头，查询块）。线程块把它的 $Q$ 块载入共享内存，然后依次处理各个键值块 $j$：

1. 载入 $K_j$、$V_j$（多级流水线，第 23 章）；
2. $S = Q K_j^{\mathsf T}$（Tensor Core）；
3. 按命题 10.5 更新每行的状态：$m' = \max(m, \operatorname{rowmax}(S))$，$P = e^{S - m'}$，$\ell' = \ell e^{m - m'} + \operatorname{rowsum}(P)$，$u' = u\, e^{m - m'} + P V_j$（最后一项又是一次 Tensor Core 乘法）；
4. 处理完所有键值块后输出 $u / \ell$。

**warp 的分工。** 一个线程块有若干个 warp（或 warpgroup）。把**查询的行**分给各个 warp：每个 warp 负责查询块中的若干行，与所有的键值块相乘。这样每一行的 rowmax、rowsum 都在一个 warp 内完成（warp 洗牌，命题 22.2），处理一个键值块时 warp 之间不需要同步。若反过来把**键值块的列**分给各个 warp，每一行的最大值与和就要在 warp 之间汇总，每个键值块都要经过共享内存同步一次。前者是 FlashAttention-2 相对于第一版的主要改进之一。

**因果掩码**：完全在对角线以上的键值块直接跳过，约省一半的运算；跨越对角线的块逐元素屏蔽。注意若某一行在已处理的块中全被屏蔽，$m = -\infty$，$e^{m - m'}$ 会得到 NaN，要特别处理（把 $-\infty$ 的参考点替换为 0）。

**长序列、小批量**：（批量 × 头 × 查询块）的个数可能少于 SM 数；这时沿序列长度切分更细的查询块，或在 decode 中沿键值方向切分（26.5 节）。

## 26.3 指数运算为什么成为瓶颈

26.2 节的结构在 Ampere 上已经够好。在更新的 GPU 上，瓶颈出现在一个意想不到的地方：矩阵乘法之外的逐元素运算。本节用一个简单的比较说明原因（命题 26.1）。

对分数矩阵 $S$ 的每个元素，Tensor Core 要做 $QK^{\mathsf T}$ 与 $PV$ 中的各 $d$ 次乘加（$4d$ FLOP），专门的函数单元（SFU）要做一次指数，CUDA 核心还要做几次减法、乘法与比较（最大值、缩放）。

> **命题 26.1** 设每个 SM 每周期的 Tensor Core 算力为 $P$（FLOP），指数运算为 $E$ 次。注意力的指数运算不成为瓶颈的条件是 $4d \ge P / E$。

*证明* 每个元素的 Tensor Core 时间为 $4d/P$，指数时间为 $1/E$。∎

以 Hopper 为例，每个 SM 的 bf16 Tensor Core 约为每周期 4096 FLOP，指数等超越函数为每周期 16 次，$P / E = 256$：$d = 128$ 时 $4d = 512$，指数的时间是矩阵乘法的一半；再加上 CUDA 核心上的其他逐元素运算，非矩阵乘法的部分已经占了可观的比例。Blackwell 每个 SM 的 Tensor Core 算力翻倍，而指数单元的吞吐不变，$P / E$ 约为 512：$d = 128$ 时两者相当，$d = 64$ 时指数运算成为主要瓶颈。

所以在新的 GPU 上，注意力 kernel 的设计重点不再只是矩阵乘法，而是：**让 softmax 的逐元素运算与矩阵乘法同时进行，并减少逐元素运算本身**。

## 26.4 Hopper 与 Blackwell 上的设计

26.3 节的结论给出了新设计的两个方向：让逐元素运算与矩阵乘法同时进行，以及减少逐元素运算本身。两代 GPU 上的公开设计分别侧重其一：

**Hopper（FlashAttention-3 的思路）**：

- warp 专门化：一个生产者 warp 用 TMA 载入 $K$、$V$；两个或更多的消费者 warpgroup 用 wgmma 做矩阵乘法（23.4 节）；
- **乒乓**：两个消费者 warpgroup 交替进行，一个在做 softmax（SFU 与 CUDA 核心）时，另一个在做矩阵乘法（Tensor Core）；
- warpgroup 内部也做流水：在算第 $j$ 块的 softmax 时，已经发出第 $j + 1$ 块的 $QK^{\mathsf T}$；
- FP8：$QK^{\mathsf T}$ 与 $PV$ 都可以用 FP8，配合按块的缩放（25.2 节）。

**Blackwell（FlashAttention-4 的思路）**：

- $S$ 与输出的累加 $u$ 都放在 TMEM 中，tcgen05 由单个线程发起，softmax 由专门的 warp 从 TMEM 读出 $S$、写回 $P$（23.5 节）；
- **减少指数单元的负担**：把一部分指数改用 CUDA 核心上的多项式近似计算（先做区间约化，把 $2^x$ 拆成整数次幂与 $[0, 1)$ 内的小数次幂，后者用低次多项式），使指数运算分摊到两种单元上；
- **减少重新缩放**：由注 10.4，参考点 $m$ 不必是真正的最大值，只要不溢出即可；只在新的最大值超过旧参考点一定的幅度时才重新缩放 $u$，大多数块可以跳过这一步。

> **现状（2026-10）**：上述两版的公开实现分别用 CUTLASS（C++）和 CuTe DSL（Python）写成。指数近似与条件重新缩放的具体参数（多项式次数、阈值、分摊比例）以其论文与代码为准。

## 26.5 解码注意力与分页 KV cache

decode 时，目标是以接近显存带宽的速度读完 KV cache。

**利用 GQA。** 共享同一个键值头的 $H / H_{kv}$ 个查询头放进同一个线程块一起处理：键值只读一次，强度提高 $H / H_{kv}$ 倍。投机解码的多个草稿词元也一起处理。

**沿键值方向切分（split-KV，也称 flash-decoding）。** 批量小、上下文长时，（批量 × 键值头）的个数可能远少于 SM 数，每个线程块又要读很长的 KV，并发不够（命题 21.4）。把每个序列的 KV 切成若干段，每段由一个线程块算出部分状态 $(m, \ell, u)$，再用一个小 kernel 按命题 10.5 合并。

**分页 KV cache。** 服务系统中，各序列的长度不同且不断增长，若为每个序列预留连续的最大长度，显存浪费严重。把 KV cache 分成固定大小的**页**（例如 16 到 128 个词元），每个序列用一张**页表**记录它的各页在一个大的页池中的位置。kernel 读 KV 时按页表取页：用 cp.async 或 TMA 逐页载入（页内连续，页间不连续）。页越大，每次载入越高效，但每个序列最后一页的浪费（至多一页减一个词元）越大。

**MLA 的 decode。** 多头潜在注意力（11.4 节的现状框）把每个词元的键值压缩成一个共享的潜向量，所有查询头共用它。于是 decode 时一个线程块可以处理很多个查询头（例如 128 个）对同一段潜向量的注意力，强度远高于普通的 MHA 或 GQA（习题 26.6），这类 kernel 接近计算受限。

## 26.6 反向传播

训练还需要注意力的反向传播。GPU 路线不依赖第 13 章，所以这里直接推出所需的公式，再说明 kernel 的结构。设损失对输出的梯度为 $\bar{O}$（与 $O$ 同形状）。只需要两条求导规则：

- 矩阵乘法 $Y = XW$：$\bar{X} = \bar{Y} W^{\mathsf T}$，$\bar{W} = X^{\mathsf T} \bar{Y}$（由 $\langle \bar{Y}, XW \rangle = \operatorname{tr}(\bar{Y}^{\mathsf T} X W)$ 分别对 $X$、$W$ 求梯度）；
- 逐行 softmax $y = \operatorname{softmax}(x)$：雅可比矩阵是对称的 $\operatorname{diag}(y) - y y^{\mathsf T}$，所以 $\bar{x} = y \odot (\bar{y} - \langle y, \bar{y} \rangle)$。

由此，一个头的反向为（省略缩放 $1/\sqrt{d}$）：

$$
\bar{V} = P^{\mathsf T} \bar{O}, \qquad \bar{P} = \bar{O} V^{\mathsf T}, \qquad \bar{S} = P \odot (\bar{P} - D\,\mathbf{1}^{\mathsf T}), \qquad \bar{Q} = \bar{S} K, \qquad \bar{K} = \bar{S}^{\mathsf T} Q,
$$

其中 $D_i = \langle O_{i,:}, \bar{O}_{i,:} \rangle$。（$D_i = \sum_j P_{ij} \bar{P}_{ij}$，代入 $\bar{P} = \bar{O} V^{\mathsf T}$ 与 $O = PV$ 即得。）

由公式得到 GPU 上的结构：

- 前向为每行保存 $\operatorname{lse}_i = m_i + \log \ell_i$；反向时用 $P_{ij} = e^{S_{ij} - \operatorname{lse}_i}$ 重新算出 $P$ 的每一块。预先用一个小 kernel 算出 $D$。
- 每个线程块负责一个键值块，持有它的 $\bar{K}$、$\bar{V}$ 累加器，遍历所有查询块。
- $\bar{Q}$ 的每一块要对所有键值块累加，而这些键值块分布在不同的线程块上：常见的做法是用 f32 的原子加法累加到全局内存中，代价是结果不确定（命题 10.11）；要确定的结果，就把各线程块的部分 $\bar{Q}$ 写出、再按固定次序归约，或者另写一个以查询块为单位的 kernel 计算 $\bar{Q}$。

反向的运算量约为前向的 2.5 倍（重算 $QK^{\mathsf T}$，再加四次矩阵乘法）。

## 接口小结

1. prefill 的注意力强度约 $2b_q/s$，计算受限；decode 访存受限，时间由读 KV cache 决定。（26.1 节）
2. FlashAttention：线程块负责一个查询块，遍历键值块，按在线 softmax 更新每行状态；按查询的行分给 warp，使行归约在 warp 内完成；因果掩码跳过对角线以上的块；注意全屏蔽行的 NaN。（26.2 节）
3. 每个分数元素要 $4d$ 次 Tensor Core FLOP 与一次指数；$4d < P/E$ 时指数成为瓶颈；Hopper 的 $P/E \approx 256$，Blackwell 约 512。（命题 26.1）
4. Hopper：warp 专门化、warpgroup 乒乓，使 softmax 与矩阵乘法重叠；Blackwell：$S$ 与 $u$ 在 TMEM，部分指数用多项式近似，条件重新缩放。（26.4 节）
5. decode：GQA 的查询头一起处理；split-KV 后合并状态；分页 KV 按页表取页，页的大小在载入效率与浪费之间权衡；MLA 的 decode 强度很高。（26.5 节）
6. 反向：保存 lse 重算 $P$；$\bar{S} = P \odot (\bar{P} - D)$，$D_i = \langle O_i, \bar{O}_i \rangle$；按键值块分线程块，$\bar{Q}$ 用原子加法（不确定）或额外的归约；约为前向的 2.5 倍。（26.6 节）

## 习题

**习题 26.1** ★ 用命题 26.1，分别对 Hopper（$P/E = 256$）与 Blackwell（$P/E = 512$）、$d = 64$ 与 $d = 128$，计算指数时间与矩阵乘法时间之比。

<details><summary>提示</summary>

比值为 $(P/E) / (4d)$。

</details>
<details><summary>答案</summary>

Hopper：$d = 64$ 时 $256/256 = 1$，$d = 128$ 时 $0.5$。Blackwell：$d = 64$ 时 2，$d = 128$ 时 1。Blackwell 上 $d = 64$ 的注意力若不减少指数运算，时间约为矩阵乘法的两倍；这还没有计入 CUDA 核心上的其他逐元素运算。

</details>

**习题 26.2** ★ 一个线程块有 4 个 warp，处理 $128$ 行的查询块与一个 $128$ 列的键值块。(a) 若按行分给 warp（每个 warp 32 行），每个键值块需要几次 warp 之间的同步来完成 rowmax 与 rowsum？(b) 若按键值块的列分给 warp（每个 warp 32 列）呢？

<details><summary>提示</summary>

行归约需要该行的所有列。

</details>
<details><summary>答案</summary>

(a) 0 次：每行的 128 列都在同一个 warp 内，用洗牌归约。(b) 每行的 128 列分布在 4 个 warp 中，rowmax 与 rowsum 各要经过共享内存汇总一次，且汇总之后才能算 $P$ 与更新 $u$，每个键值块至少两次块内同步。这就是 26.2 节"按行分工"的理由。

</details>

**习题 26.3** ★ decode：批量 4，$H_{kv} = 8$，上下文 128k 词元，GPU 有 132 个 SM。(a) 不切分时有几个线程块？SM 的利用率如何？(b) 沿键值方向每个序列切成 8 段呢？(c) 合并的代价是多少（$d = 128$，f32 状态）？

<details><summary>提示</summary>

不切分时每个（序列，键值头）一个线程块。合并时每段写出 $(m, \ell, u)$。

</details>
<details><summary>答案</summary>

(a) $4 \times 8 = 32$ 个，只用 32 个 SM，约 24%；而且每个线程块要读 128k 词元的 KV，单个 SM 的在途量不足以跑满它分到的带宽。(b) 256 个线程块，约两轮，每个读 16k 词元。(c) 每段每个查询头写出 $2 + 128$ 个 f32，共 $256 \times (H/H_{kv}) \times 130 \times 4$ 字节，若 $H / H_{kv} = 4$ 约 0.5 MB，与读 KV cache 的数 GB 相比可以忽略。

</details>

**习题 26.4** ★ 分页 KV cache 的页大小取 16 或 256 个词元。设每个词元每层的 KV 是 2 KB（例如 $H_{kv} = 8$、$d = 128$、bf16，键值各 1 KB）。(a) 每页多大？(b) 1000 个序列的平均浪费（每个序列最后一页平均浪费半页）各是多少？(c) 页太小有什么代价？

<details><summary>提示</summary>

(c) 每页一次载入，对比命题 4.12 与定义 8.11。

</details>
<details><summary>答案</summary>

(a) 32 KB 与 512 KB（每层）。(b) 平均每序列浪费半页：每层 16 KB 与 256 KB，1000 个序列每层 16 MB 与 256 MB。(c) 页太小时，每页一次载入，固定开销占比大，页表也更长；页太大时浪费显存。常见的选择在 16 到 128 个词元之间。

</details>

**习题 26.5** ★（审查题）一个注意力反向 kernel 用 bf16 的 `atomicAdd` 把各线程块的部分 $\bar{Q}$ 累加到全局内存中。指出问题。

<details><summary>提示</summary>

例 2.13 与命题 10.11。

</details>
<details><summary>答案</summary>

一，bf16 累加会损失精度，长序列时甚至停滞；累加应当在 f32 中进行，最后再转换。二，原子累加的次序不确定，结果每次运行不同；若要可复现，应写出部分结果再按固定次序归约。三，bf16 的原子加法在一些硬件上要按对（两个 bf16）进行，性能也不好。

</details>

**习题 26.6** ★ MLA 的 decode：128 个查询头共享每个词元的一个 512 维潜向量（外加 64 维位置编码分量），bf16。对每个缓存的词元，计算注意力的运算量与读取的字节数，求强度。与 H100 的拐点（约 295 FLOP/字节）比较。

<details><summary>提示</summary>

每个查询头：分数用 $512 + 64 = 576$ 维的点积，加权和用 512 维。

</details>
<details><summary>答案</summary>

每个词元读 $576 \times 2 = 1152$ 字节；运算量 $128 \times (2 \times 576 + 2 \times 512) = 278528$ FLOP；强度约 242 FLOP/字节，接近拐点。与普通 MHA 的约 1 FLOP/字节相比，MLA 的 decode 注意力几乎是计算受限的，所以它的 kernel 要按计算受限的方式设计（大的 Tensor Core 块、流水线），而不只是追求带宽。

</details>

**习题 26.7** ☆ 在 Hopper 上用 FP8 计算注意力的 $QK^{\mathsf T}$ 与 $PV$。$P \in [0, 1]$ 转为 FP8 E4M3 时，最小规格化数是 $2^{-6}$。对很长的行（例如 32k 个键），这会带来什么问题？可以怎样缓解？

<details><summary>提示</summary>

长行中大部分 $P_{ij}$ 远小于 $2^{-6}$。命题 2.10。

</details>
<details><summary>答案</summary>

softmax 之后大部分概率很小，低于 $2^{-6}$ 的值落入次正规数区间甚至变为 0，它们的总和却可能不小，丢失会使输出偏差。缓解：在转为 FP8 之前把 $P$ 乘以一个缩放因子 $c$（例如 $c = 256$：$P$ 的最大值 1 变为 256，仍小于 E4M3 的最大值 448），与 $V$ 相乘之后再除以 $c$，这把能以规格化数表示的最小概率从 $2^{-6}$ 降到 $2^{-14}$；或者 $PV$ 保留 bf16，只对 $QK^{\mathsf T}$ 用 FP8。

</details>
