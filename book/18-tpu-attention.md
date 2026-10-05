# 第 18 章　TPU kernel：注意力

注意力是 Transformer 中唯一不是"矩阵乘法加逐元素运算"的部分。第 11 章说明了它的问题：中间的分数矩阵是 $T \times T$ 的，不能写回 HBM；第 10 章给出了解决办法：在线 softmax 让我们一块一块地处理键和值，只保留每个查询的状态 $(m, \ell, u)$。本章把它写成 TPU 上的 kernel，并讨论因果掩码与块稀疏、decode 时的注意力，以及反向传播。

> **在体系中的位置**：下层是第 10 章的在线 softmax（命题 10.5）、第 11 章的注意力算术、第 13 章的自动微分、第 15 章的 TensorCore、第 16、17 章的 Pallas 与矩阵乘法 kernel。本章给出 flash attention 的 TPU 形式、各单元的负载平衡、掩码的块级跳过、decode 注意力与反向传播。

## 18.1 注意力的屋顶线

考虑一个头：$Q, K, V \in \mathbb{R}^{T \times d}$，输出 $O = \operatorname{softmax}(QK^{\mathsf T}/\sqrt{d})\,V$。

- 运算量：$QK^{\mathsf T}$ 与 $PV$ 各 $2T^2 d$，不计掩码共 $4T^2 d$；
- 若把 $S$、$P$ 写回 HBM：数据量约 $2 s T^2$ 的量级，强度约 $d/s$，访存受限；
- 若中间结果不离开 VMEM（flash attention）：只读 $Q, K, V$ 写 $O$，但键值要被每个查询块读一遍。查询块大小为 $b_q$ 时，$K$、$V$ 被读 $T / b_q$ 次，数据量约 $2 s T d \cdot T / b_q$，强度约 $\frac{4 T^2 d}{2 s T^2 d / b_q} = \frac{2 b_q}{s}$。

**命题 18.1** flash attention 在 prefill 中的强度约为 $2b_q / s$，与 $T$ 无关；$b_q$ 达到 $sI^*/2$ 时计算受限。decode 时每个序列只有一个（或少数几个）查询，强度约为 $2 \cdot (\text{同时处理的查询数}) / s$，访存受限，时间由读 KV cache 决定。

*证明* 上面的计数；decode 时 $b_q$ 换成同时处理的查询数。∎

所以 prefill 的注意力 kernel 要选足够大的 $b_q$，decode 的注意力 kernel 要以最高效率读 KV cache（18.4 节）。

## 18.2 Flash attention 的 TPU 形式

grid 取（批量 × 头，查询块，键值块）。对每个查询块，在 VMEM 的 scratch 中保留 f32 的状态：每行的 $m$、$\ell$，以及 $b_q \times d$ 的累加器 $u$。沿键值块的维逐块执行命题 10.5 的合并，这一维必须是 `"arbitrary"`（命题 16.5），最后一块时输出 $u / \ell$。

```python
def flash_kernel(q_ref, k_ref, v_ref, o_ref, m_ref, l_ref, acc_ref, *, causal, bq, bk, scale):
    qi, ki = pl.program_id(1), pl.program_id(2)

    @pl.when(ki == 0)
    def _():
        m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
        l_ref[...] = jnp.zeros_like(l_ref)
        acc_ref[...] = jnp.zeros_like(acc_ref)

    def update():
        q, k = q_ref[0], k_ref[0]                                   # (bq, d), (bk, d)
        s = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())),     # q k^T，不显式转置 k
                                preferred_element_type=jnp.float32) * scale
        if causal:
            row = qi * bq + jax.lax.broadcasted_iota(jnp.int32, (bq, bk), 0)
            col = ki * bk + jax.lax.broadcasted_iota(jnp.int32, (bq, bk), 1)
            s = jnp.where(col <= row, s, -jnp.inf)
        m_prev = m_ref[...]
        m_new = jnp.maximum(m_prev, s.max(axis=1, keepdims=True))
        alpha = jnp.exp(m_prev - m_new)                             # 旧状态的缩放因子
        p = jnp.exp(s - m_new)
        l_ref[...] = alpha * l_ref[...] + p.sum(axis=1, keepdims=True)
        acc_ref[...] = alpha * acc_ref[...] + jnp.dot(
            p.astype(v_ref.dtype), v_ref[0], preferred_element_type=jnp.float32)
        m_ref[...] = m_new

    if causal:
        pl.when(ki * bk <= qi * bq + bq - 1)(update)                # 跳过完全在对角线以上的块
    else:
        update()

    @pl.when(ki == pl.num_programs(2) - 1)
    def _():
        o_ref[0] = (acc_ref[...] / l_ref[...]).astype(o_ref.dtype)
```

BlockSpec：$Q$ 与输出取 `(1, bq, d)`，下标 `(b, i, 0)`；$K$、$V$ 取 `(1, bk, d)`，下标 `(b, j, 0)`；scratch 为 `VMEM((bq, 1), f32)` 两个、`VMEM((bq, d), f32)` 一个；`dimension_semantics=("parallel", "parallel", "arbitrary")`。这个 kernel 在解释器中与直接计算的结果一致（误差来自把 $P$ 转为 bf16 后再与 $V$ 相乘）。

几点说明：

- **不显式转置 $K$**：`dot_general` 收缩两者的第 1 维，即 $QK^{\mathsf T}$；MXU 在推入权重时可以顺便转置（15.4 节），所以不需要 XLU 做转置。
- **$P$ 转为 bf16 再乘 $V$**：MXU 以 bf16 计算，$P \in [0, 1]$ 转为 bf16 的相对误差约 $2^{-8}$。累加仍在 f32 中。
- **$m$、$\ell$ 的存放**：上面为了清楚，把它们存为 $(b_q, 1)$。在真实的 TPU 上，一列的数组会被补齐到 128 个通道；实用的 kernel 把每行的 $m$、$\ell$ 复制到全部 128 个通道中存放，运算时再按需要截取，以符合 $\mathrm{T}(8, 128)$ 的布局。

**各单元的负载。** 对 $S$ 的每个元素，MXU 要做 $QK^{\mathsf T}$ 与 $PV$ 中的各 $d$ 次乘加，即 $4d$ FLOP；EUP 要做一次指数；XLU 要参与行最大值与行和两次跨通道归约。

**命题 18.2（MXU 与 EUP 的平衡）** 设 MXU 每周期 $P_{\text{mxu}}$ FLOP，EUP 每周期 $E$ 个指数。flash attention 不被 EUP 限制的条件是

$$
4d \ge \frac{P_{\text{mxu}}}{E}.
$$

*证明* 每个分数元素的 MXU 时间为 $4d / P_{\text{mxu}}$，EUP 时间为 $1 / E$。∎

以 v4 的一个 TensorCore 为例，$P_{\text{mxu}} = 2 \times 65536$，EUP 每 2 个周期一个寄存器即 $E = 512$，比值为 256，要求 $d \ge 64$。$d = 128$ 时 MXU 是瓶颈，$d = 64$ 时两者持平，更小的头维度就会被指数运算拖住。这是注意力 kernel 特有的现象：它的瓶颈可能不在矩阵单元。

XLU 的负载更需要小心：若每个 $8 \times 128$ 的分数寄存器都做两次跨通道归约，在 v4 上每个寄存器约要 8 个周期（两个 XLU），远慢于 MXU。办法与习题 15.4 相同：先在 $b_k / 128$ 个寄存器之间逐元素求最大值与和，每 8 行只做两次跨通道归约。另一种办法是改变布局：计算 $S^{\mathsf T} = K Q^{\mathsf T}$，让键沿子通道和寄存器排列，行方向的归约就变成寄存器之间的逐元素运算（8.3 节的原则）。

## 18.3 掩码与块稀疏

**因果掩码。** 查询块 $i$ 与键值块 $j$ 的关系只有三种：完全在对角线以下（全部可见），跨越对角线（部分可见），完全在对角线以上（全部屏蔽）。上面的 kernel 用 `pl.when` 跳过第三种，对第二种用 `where` 逐元素屏蔽。

**命题 18.3** 取 $b_q = b_k = b$，$n = T/b$，因果掩码下需要计算的块数为 $n(n+1)/2$，占全部 $n^2$ 块的 $\frac{1}{2} + \frac{1}{2n}$。

*证明* 第 $i$ 个查询块需要第 $0, \dots, i$ 个键值块。∎

跳过的块仍然占用 grid 的步。若 grid 很大，可以只遍历需要的块：

**块稀疏注意力。** 一般的掩码（滑动窗口、文档边界、前缀可见等）事先在块的粒度上分类：对每个（查询块，键值块），标记为"全部屏蔽""全部可见""部分可见"。用标量预取传入每个查询块需要访问的键值块列表（以及列表长度、每块是否需要逐元素掩码），grid 的键值维只遍历列表中的块，下标映射查表得到实际的键值块号（16.7 节）。

**命题 18.4** 块稀疏注意力的运算量与非空块的个数成正比；部分可见块额外付出逐元素掩码的代价。

例如窗口为 $w$ 的滑动窗口注意力，每个查询块只需约 $w / b + 1$ 个键值块，运算量从 $O(T^2)$ 降到 $O(Tw)$。

## 18.4 解码注意力

decode 时每个序列只有一个新的查询，要与它的整个 KV cache 计算注意力（11.4 节）。由命题 18.1，这是访存受限的，目标是**以接近 HBM 带宽的速度读完 KV cache**。

**利用 GQA。** 在分组查询注意力中，$H / H_{kv}$ 个查询头共享一个键值头。把共享同一个键值头的所有查询头放进同一个块一起处理，键值只读一次，强度提高 $H / H_{kv}$ 倍。这时"查询块"的行数就是 $H / H_{kv}$（再乘以投机解码时的草稿词元数）。

**分页与长度不同的批量。** KV cache 通常按页存放（习题 16.5），每个序列的长度不同。用标量预取传入页表和每个序列的长度：下标映射查页表得到每一页的位置；超过序列长度的页用 `pl.when` 跳过，最后一页中多余的位置用掩码屏蔽。

**切分长序列。** 若批量很小而序列很长，按序列分 grid 用不满所有核心，读 KV 的并发也不够（推论 4.5）。这时沿序列长度把 KV cache 切成若干段，各段独立地计算部分状态 $(m, \ell, u)$，最后用命题 10.5 合并（这一做法常称为 split-K 或 flash-decoding）。合并的代价是写出、读入每段的状态，每段只有 $O(d)$ 个数，可以忽略。

> **现状（2026-10）**：TPU 上的推理框架（如 vLLM 的 TPU 后端）使用"ragged paged attention" kernel：一次调用中同时处理 prefill 的长查询和 decode 的单个查询，按页读取 KV，并用标量预取传入每个序列的长度与页表。这正是本节各项技术的组合。

## 18.5 反向传播

设损失对输出的梯度为 $\bar{O}$。对一个头（省略缩放 $1/\sqrt{d}$，它只是乘到 $S$ 上），由第 13 章：

**命题 18.5（注意力的反向）**

$$
\bar{V} = P^{\mathsf T} \bar{O}, \qquad \bar{P} = \bar{O} V^{\mathsf T}, \qquad \bar{S} = P \odot (\bar{P} - D\,\mathbf{1}^{\mathsf T}), \qquad \bar{Q} = \bar{S} K, \qquad \bar{K} = \bar{S}^{\mathsf T} Q,
$$

其中 $D_i = \sum_j P_{ij} \bar{P}_{ij} = \sum_c O_{ic} \bar{O}_{ic}$。

*证明* 前两个与后两个是矩阵乘法的 VJP（命题 13.4）。$\bar{S}$ 是逐行 softmax 的 VJP（习题 13.3）：$\bar{s} = p \odot (\bar{p} - \langle p, \bar{p} \rangle)$。最后，$\langle P_{i,:}, \bar{P}_{i,:} \rangle = \sum_j P_{ij} (\bar{O} V^{\mathsf T})_{ij} = \sum_c \bar{O}_{ic} (PV)_{ic} = \langle \bar{O}_{i,:}, O_{i,:} \rangle$。∎

由此得到反向 kernel 的结构：

- 前向时为每一行保存 $\operatorname{lse}_i = m_i + \log \ell_i$（每行一个数），反向时用 $P_{ij} = \exp(S_{ij} - \operatorname{lse}_i)$ 重新算出 $P$ 的每一块，而不保存 $T \times T$ 的 $P$；
- 预先算出 $D_i = \langle O_{i,:}, \bar{O}_{i,:} \rangle$（一次逐行的点积）；
- $\bar{Q}$ 的每一块要对所有键值块累加，$\bar{K}$、$\bar{V}$ 的每一块要对所有查询块累加。为了让每个输出块只在一个按顺序执行的 grid 维上累加（命题 16.2），常见的做法是写两个 kernel：一个以查询块为外层、遍历键值块，累加 $\bar{Q}$；另一个以键值块为外层、遍历查询块，累加 $\bar{K}$ 与 $\bar{V}$。

反向的运算量：$QK^{\mathsf T}$ 重算一次，$\bar{V}, \bar{P}, \bar{Q}, \bar{K}$ 各一次矩阵乘法，共 5 次 $2T^2d$，约为前向（2 次）的 2.5 倍。

把前向与反向的 kernel 连接起来，用 `jax.custom_vjp`（13.4 节）：前向函数返回输出与残差（$Q, K, V, O, \operatorname{lse}$），反向函数调用反向的 kernel。

## 接口小结

1. flash attention 的强度约 $2b_q/s$，与序列长度无关；decode 时访存受限，由读 KV cache 决定。（命题 18.1）
2. TPU 上的 flash attention：grid（批量×头，查询块，键值块），键值维 `"arbitrary"`，f32 的 $(m, \ell, u)$ 放在 scratch 中，按命题 10.5 合并；用 `dot_general` 收缩第 1 维避免显式转置。（18.2 节）
3. 每个分数元素需要 $4d$ 次 MXU FLOP、一次指数、两次参与跨通道归约；$4d < P_{\text{mxu}}/E$ 时被 EUP 限制；跨通道归约要先在寄存器之间部分完成，或改变布局。（命题 18.2）
4. 因果掩码跳过约一半的块；一般的掩码按块分类，用标量预取只遍历非空块。（命题 18.3、18.4）
5. decode：把共享键值头的查询头放在一起；用标量预取处理页表与长度；长序列沿长度切分再合并状态。（18.4 节）
6. 反向：保存每行的 lse，重算 $P$；$\bar{S} = P \odot (\bar{P} - D)$，$D_i = \langle O_i, \bar{O}_i \rangle$；分两个 kernel 累加 $\bar{Q}$ 与 $\bar{K}, \bar{V}$；运算量约为前向的 2.5 倍；用 `custom_vjp` 连接。（命题 18.5）

## 习题

**习题 18.1** ★ 一个头 $d = 128$，$T = 8192$，bf16。(a) 不融合（把 $S$ 与 $P$ 写回 HBM）时，注意力的数据量与强度各是多少？(b) flash attention 取 $b_q = 512$ 时呢？(c) 在 v5e 上（拐点约 240）二者各是什么受限？

<details><summary>提示</summary>

(a) $S$ 与 $P$ 各写一次、读一次（f32 或 bf16，这里按 bf16 计）。(b) 命题 18.1。

</details>
<details><summary>答案</summary>

(a) $S$、$P$ 各 $T^2 = 6.7 \times 10^7$ 个元素，各写读一次约 $4 \times 2 \times 6.7 \times 10^7 \approx 5.4 \times 10^8$ 字节；运算 $4T^2 d \approx 3.4 \times 10^{10}$；强度约 64。(b) 约 $2 \times 512 / 2 = 512$。(c) 前者访存受限，后者计算受限。

</details>

**习题 18.2** ★ 用命题 18.2，分别对 $d = 64, 128, 256$ 判断 v4 上的 flash attention 是 MXU 还是 EUP 受限。若某代芯片的 MXU 算力翻倍而 EUP 不变呢？

<details><summary>提示</summary>

v4：$P_{\text{mxu}}/E = 256$。

</details>
<details><summary>答案</summary>

$4d = 256, 512, 1024$：$d = 64$ 两者持平，$d = 128, 256$ MXU 受限。MXU 翻倍后比值为 512：$d = 64$ 变为 EUP 受限，$d = 128$ 持平。随着矩阵单元越来越快，指数运算越来越可能成为瓶颈；可以用更便宜的指数近似、把部分指数交给其他单元，或减少不必要的重新缩放（注 10.4）。

</details>

**习题 18.3** ★ $T = 8192$，$b = 512$，因果掩码。需要计算多少个块？跨越对角线、需要逐元素掩码的有多少个？

<details><summary>提示</summary>

$n = 16$。

</details>
<details><summary>答案</summary>

$16 \times 17 / 2 = 136$ 个块，占 256 个的 53%。跨越对角线的是 $i = j$ 的 16 个块，其余 120 个全部可见，不需要掩码。

</details>

**习题 18.4** ★ 滑动窗口注意力：每个查询只看它之前的 $w = 1024$ 个键（含自己）。$T = 32768$，$b = 512$。每个查询块需要多少个键值块？总共多少块？与完整的因果注意力相比运算量少多少？

<details><summary>提示</summary>

查询块 $i$ 的行覆盖 $[ib, (i+1)b)$，需要的键覆盖 $[ib - w + 1, (i+1)b)$。

</details>
<details><summary>答案</summary>

键的区间长 $b + w - 1 = 1535$，至多跨越 $\lceil 1535 / 512 \rceil + 1 = 4$ 个块（对齐时为 3 个块，即块 $i - 2, i - 1, i$）。$n = 64$ 个查询块，约 $64 \times 3 = 192$ 块；完整因果注意力为 $64 \times 65 / 2 = 2080$ 块，约少 91%。

</details>

**习题 18.5** ★ decode 时 $H = 32$、$H_{kv} = 8$、$d = 128$，bf16 的 KV cache。(a) 每个查询头单独处理时，注意力的强度是多少？(b) 把共享同一键值头的 4 个查询头放在一起呢？(c) 再加上 4 个投机解码的草稿词元呢？

<details><summary>提示</summary>

强度约为 $2 \times$（同时处理的查询数）$/ s$。

</details>
<details><summary>答案</summary>

(a) $2 \times 1 / 2 = 1$。(b) 4。(c) 16 个查询共享一次读取，强度 16。都远低于拐点，仍是访存受限，但每读一个字节完成的工作多了 16 倍：在同样的带宽下，每秒能服务的词元数相应增加。

</details>

**习题 18.6** ★ 验证命题 18.5 中 $D_i$ 的两种表达式相等，并说明为什么反向 kernel 要预先算出 $D$，而不是在每个块中用 $\sum_j P_{ij}\bar{P}_{ij}$ 计算。

<details><summary>提示</summary>

$\sum_j P_{ij} \bar{P}_{ij}$ 需要整行的 $P$ 与 $\bar{P}$。

</details>
<details><summary>答案</summary>

等式见命题 18.5 的证明。$\sum_j P_{ij}\bar{P}_{ij}$ 要遍历整行的所有键值块才能得到，而 $\bar{S}$ 的每一块都需要 $D_i$；若在块中计算，就要先完整地遍历一遍。$\langle O_{i,:}, \bar{O}_{i,:} \rangle$ 只需要 $O$ 与 $\bar{O}$ 的第 $i$ 行，可以在进入主循环之前用一次很便宜的逐行点积算好。

</details>

**习题 18.7** ★（审查题）AI 写的 flash attention 把 grid 的三维都标为 `"parallel"`。它在单核心的芯片上正确，在双核心的芯片上错误。为什么？

<details><summary>提示</summary>

命题 16.5。状态 $(m, \ell, u)$ 在哪一维上被累加？

</details>
<details><summary>答案</summary>

$(m, \ell, u)$ 与输出块在键值块的维上保持不变并被累加。这一维标为 `"parallel"` 后，可能被分给两个核心，或打乱次序（例如 $k_i = 0$ 的初始化不再先执行），状态的合并出现竞争。应当标为 `"arbitrary"`。

</details>

**习题 18.8** ★（审查题）有人把 18.2 节的 kernel 改为 $b_q = 256$、$b_k = 128$，并把跳过条件写成 `ki <= qi`。在因果掩码下会出现什么问题？若某一行在当前块之前的所有块中都被完全屏蔽，`m_prev` 与 `m_new` 都是 $-\infty$，`alpha = exp(m_prev - m_new)` 会得到什么？

<details><summary>提示</summary>

$b_q \ne b_k$ 时，查询块 $i$ 覆盖的行是 $[256 i, 256 i + 256)$，键值块 $j$ 覆盖 $[128 j, 128 j + 128)$。

</details>
<details><summary>答案</summary>

`ki <= qi` 只在 $b_q = b_k$ 时正确；$b_q = 256$ 时查询块 $i$ 需要键值块 $0, \dots, 2i + 1$，条件应为 `ki * bk <= qi * bq + bq - 1`，原条件会漏掉一半需要的块，结果错误。第二个问题：$-\infty - (-\infty)$ 是 NaN，`alpha` 为 NaN，会污染 $\ell$ 与 $u$。按 18.2 节的条件，处理的第一个块总有可见的元素，不会出现这种情况；但在一般的掩码（例如文档边界）下可能出现。稳妥的写法是在 `m_new` 为 $-\infty$ 时把 `alpha` 与 `p` 置零（例如先把 `m_new` 中的 $-\infty$ 替换为 0 再计算指数）。

</details>
