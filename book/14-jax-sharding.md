# 第 14 章　JAX 的分布式编程

第 12 章用记号 $A[I_x, J]$ 描述分片，用命题 12.3 判断一次分布式矩阵乘法需要什么通信。JAX 的分布式接口就是这套记号的程序形式。本章讲三件事：怎样描述网格和分片；谁来决定中间结果的分片（JAX 提供三种模式）；怎样在需要完全控制时，以每台设备的视角写程序并显式地调用集合通信。

14.1 节讲网格与分片的写法；14.2 节讲三种模式；14.3 节讲 shard_map 与集合原语；14.4 节把第 12 章的 FSDP 与张量并行写成程序；14.5、14.6 节讲显存与多主机。

本章的代码都在 8 台模拟设备上运行过（2026-10，JAX 0.11）。

> **在体系中的位置**：下层是第 7 章的集合操作、第 12 章的分片记号与并行策略、第 13 章的追踪与 `jit`。本章给出 mesh 与 PartitionSpec、Explicit/Auto/Manual 三种模式、`shard_map` 与集合原语、FSDP 与张量并行的写法以及显存工具。第 16 章的 Pallas kernel 可以放在 `shard_map` 之内；第 19 章的 TPU 多芯片 kernel 是 `shard_map` 中集合通信的手写版本。

## 14.1 网格与 PartitionSpec

第 12 章的网格与分片（定义 12.1、12.2）在 JAX 中各有一个对象。本节给出它们的写法，以及它们与第 12 章记号的对应。

```python
from jax.sharding import AxisType
mesh = jax.make_mesh((2, 4), ("x", "y"), (AxisType.Explicit, AxisType.Explicit))
```

建立一个 $\{x: 2, y: 4\}$ 的网格（定义 12.1）。`make_mesh` 会按设备的物理拓扑排列它们，使同一网格轴上的设备在物理上相邻。第三个参数为每个轴指定**轴类型**（14.2 节）。

**PartitionSpec**（`jax.P`）为数组的每一维指定分片：`None` 表示不分片（复制），一个轴名表示沿该轴分片，一个轴名的元组表示同时沿几个轴分片。它与第 12 章的记号一一对应：

| 第 12 章 | JAX |
| --- | --- |
| $A[I_x, J]$ | `P("x", None)` |
| $A[I, J_y]$ | `P(None, "y")` |
| $A[I_{xy}, J]$ | `P(("x", "y"), None)` |
| $A[I, J]$（完全复制） | `P()` 或 `P(None, None)` |

```python
with jax.set_mesh(mesh):
    a = jax.device_put(jnp.ones((8, 16)), jax.P("x", None))
    print(jax.typeof(a))      # float32[8@x,16]
```

`jax.set_mesh` 设定当前使用的网格；`device_put` 按分片把数组放到各设备上。在 Explicit 模式下，分片是数组**类型**的一部分：`float32[8@x,16]` 表示第一维沿 $x$ 分片。

## 14.2 三种模式

中间结果的分片由谁决定？JAX 按网格轴提供三种答案。本节给出它们的定义（定义 14.1），在矩阵乘法上比较它们，并说明怎样取舍。

> **定义 14.1（轴类型）**
>
> - **Explicit**：分片是类型的一部分，在追踪时按类型规则逐个运算地确定。规则有歧义的运算（例如收缩维被分片的矩阵乘法）必须由程序员指定输出的分片；不相容的分片是类型错误。
> - **Auto**：中间结果的分片由编译器的分区器决定，它根据输入的分片和程序中的约束推断，并自动插入集合通信。
> - **Manual**：在 `shard_map` 之内，程序看到的是每台设备上的局部块，通信由程序员显式调用集合原语完成（14.3 节）。

**Explicit 模式下的矩阵乘法**正是命题 12.3：

```python
with jax.set_mesh(mesh):
    a = jax.device_put(jnp.ones((8, 16)), jax.P("x", None))      # A[I_x, K]
    b = jax.device_put(jnp.ones((16, 32)), jax.P(None, "y"))     # B[K, J_y]
    jax.typeof(a @ b)                                            # float32[8@x,32@y]：情形 1，无通信

    a2 = jax.device_put(jnp.ones((8, 16)), jax.P(None, "x"))     # A[I, K_x]
    b2 = jax.device_put(jnp.ones((16, 32)), jax.P("x", None))    # B[K_x, J]
    a2 @ b2              # ShardingTypeError：收缩维被分片，输出的分片有歧义
    jnp.dot(a2, b2, out_sharding=jax.P(None, None))              # float32[8,32]：all-reduce
    jnp.dot(a2, b2, out_sharding=jax.P("x", None))               # float32[8@x,32]：reduce-scatter
```

第二个乘法属于情形 2：每台设备得到部分和，可以 all-reduce 成复制的结果，也可以 reduce-scatter 成分片的结果。JAX 不替程序员选择，而是要求用 `out_sharding` 写明。

不相容的分片同样在追踪时报错。例如 $A[I_x, J]$ 与 $A[I, J_x]$ 逐元素相加，结果将是 $[I_x, J_x]$，同一个轴用于两维（定义 12.2 不允许），JAX 报 `ShardingTypeError`。要改变分片，用 `jax.reshard(a, jax.P(None, "y"))` 显式地重新分片，它会生成相应的集合通信。

**Auto 模式**下，同样的 `a2 @ b2` 不会报错：编译器自己选择输出的分片（在这个例子中它选择了 all-reduce 得到复制的结果）。用 `jax.lax.with_sharding_constraint(x, sharding)` 可以约束某个中间结果的分片。

三种模式的取舍：

- Explicit：分片的错误在追踪时就暴露，程序中每个数组的分片一目了然，通信的位置清楚；代价是要多写分片的标注。
- Auto：代码最少；但性能取决于分区器的选择，要查看编译后的 HLO 才知道它插入了哪些集合通信（13.3 节）。
- Manual：完全控制，适合需要特殊通信模式或通信与计算重叠的地方；代价是代码最多。

三种模式可以按网格轴混用，例如数据并行的轴用 Explicit，某一段需要手写通信的代码对某个轴用 Manual。

## 14.3 shard_map 与集合原语

Manual 模式下，程序以每台设备的视角写成，通信由集合原语显式完成。本节给出 shard_map 的用法、集合原语与第 7 章集合操作的对应，以及一个把通信与计算重叠起来的例子（例 14.2）。

`jax.shard_map` 让一个函数以每台设备的视角运行：

```python
@jax.jit
@jax.shard_map(in_specs=jax.P("x", None), out_specs=jax.P(None, None))
def column_sum(x_blk):                                   # x_blk 是本设备的局部块
    return jax.lax.psum(x_blk.sum(0, keepdims=True), "x")
```

若全局输入的形状是 $(8, 4)$、沿 $x$（大小 2）分片，函数收到的 `x_blk` 是 $(4, 4)$。`in_specs` 说明全局数组怎样切成局部块，`out_specs` 说明局部输出怎样拼成全局数组。

集合原语以轴名指定参与的设备（定义 7.4）：

| 原语 | 集合操作 |
| --- | --- |
| `lax.psum(x, "x")` | all-reduce（求和） |
| `lax.all_gather(x, "x", tiled=True)` | all-gather，沿某一维拼接 |
| `lax.psum_scatter(x, "x", scatter_dimension=d, tiled=True)` | reduce-scatter |
| `lax.ppermute(x, "x", perm)` | 置换，`perm` 是（源，目的）对的列表 |
| `lax.all_to_all(x, "x", split_axis, concat_axis)` | all-to-all |
| `lax.axis_index("x")`、`lax.axis_size("x")` | 本设备在轴上的坐标、轴的大小 |

**复制性的检查。** `out_specs` 中没有出现某个轴，意味着声明输出在该轴上是复制的：各设备上的值相同。JAX 为 `shard_map` 中的每个值记录它可能在哪些轴上不同，并检查这个声明。`psum` 的结果在被求和的轴上相同，可以声明为复制的；直接由局部输入算出的值则被认为可能不同。若检查过于保守（例如某些集合原语的结果实际上相同），可以用 `check_vma=False` 关闭它，此时正确性由程序员负责。

**一个常见的错误**：对一个本来就复制的值做 `psum`，结果是它的 $\lvert x \rvert$ 倍，因为每台设备都贡献了同一个值（习题 14.3）。

> **例 14.2（通信与计算重叠的矩阵乘法）** 计算 $Y[N, F_x] = X[N_x, D] \cdot W[D, F_x]$：直接的做法是先 all-gather $X$ 再相乘。按命题 7.10，可以把 all-gather 拆成环形的 $p - 1$ 次置换，每收到一块就先乘这一块：
>
> ```python
> @jax.jit
> @jax.shard_map(in_specs=(jax.P("x", None), jax.P(None, "x")), out_specs=jax.P(None, "x"))
> def ag_matmul(x_blk, w_blk):              # x_blk: [N/p, D]，w_blk: [D, F/p]
>     p = jax.lax.axis_size("x")            # Python 整数，下面的循环在追踪时展开
>     r = jax.lax.axis_index("x")
>     perm = [(i, (i + 1) % p) for i in range(p)]
>     out = jnp.zeros((p, x_blk.shape[0], w_blk.shape[1]), x_blk.dtype)
>     blk = x_blk
>     for t in range(p):
>         src = (r - t) % p                 # 第 t 步手中是第 src 块行
>         out = out.at[src].set(blk @ w_blk)
>         if t + 1 < p:
>             blk = jax.lax.ppermute(blk, "x", perm)
>     return out.reshape(-1, w_blk.shape[1])
> ```
>
> 每一步的矩阵乘法与下一块的传递互不依赖，编译器可以让它们同时进行。正确性只依赖于一个不变量：第 $t$ 步时，设备 $r$ 手中的是第 $(r - t) \bmod p$ 块行。

## 14.4 FSDP 与张量并行的写法

作为前几节的综合，把第 12 章的方案写成 Explicit 模式的程序。网格 $\{d: 2, t: 4\}$，$d$ 是数据并行与 FSDP 的轴，$t$ 是张量并行的轴。MLP 的激活沿 $d$ 分片批量维；权重沿 $d$ 做 FSDP 分片，同时沿 $t$ 做张量并行分片：

```python
x  = jax.device_put(x,  jax.P("d", None))     # X[N_d, D]
w1 = jax.device_put(w1, jax.P("d", "t"))      # W1[D_d, F_t]
w2 = jax.device_put(w2, jax.P("t", "d"))      # W2[F_t, D_d]

@jax.jit
def mlp(x, w1, w2):
    w1 = jax.reshard(w1, jax.P(None, "t"))    # 沿 d 做 all-gather：W1[D, F_t]
    w2 = jax.reshard(w2, jax.P("t", None))    # 沿 d 做 all-gather：W2[F_t, D]
    h = jax.nn.relu(x @ w1)                   # H[N_d, F_t]：情形 1，无通信
    return jnp.dot(h, w2, out_sharding=jax.P("d", None))   # 沿 t 的部分和：all-reduce
```

编译后的程序中恰好出现两类集合通信：all-gather（FSDP 取回权重）与 all-reduce（张量并行合并部分和），与命题 12.3 的预测一致。反向传播由 `jax.grad` 自动生成：权重梯度的收缩维是批量维（沿 $d$ 分片），按命题 12.3 得到部分和；而梯度要与参数一样沿 $d$ 分片（实际得到的梯度类型正是 `float32[32@d,64@t]`），所以语义上这是一次沿 $d$ 的 reduce-scatter，正是 FSDP 的梯度规约。编译器可以把它实现为 reduce-scatter，也可以实现为 all-reduce 之后再取本地的一段：在模拟设备（CPU）上编译的程序就是后者，通信量是前者的两倍。加速器的编译器通常会把后者改写为前者，但这正是审查时应当在 HLO 中确认的一点。

## 14.5 显存

第 11 章说明，训练时显存往往比运算更早成为限制。JAX 为此提供了下列工具：

- **捐赠**（`donate_argnums`，13.3 节）：让更新后的参数和优化器状态复用旧的存储，避免同时存在两份。
- **重计算**（`jax.checkpoint`，13.4 节）：以运算换激活显存。
- **估计显存**：`jax.jit(f).lower(*args).compile().memory_analysis()` 给出编译后程序的参数、输出、临时存储的大小，可以在不运行的情况下检查是否放得下。
- **卸载到主机内存**：JAX 允许把数组放在主机内存中（不同的"内存种类"），在需要时取回，用于放不下的优化器状态或激活。主机与设备之间的带宽远低于 HBM，只适合不频繁访问的数据。

## 14.6 多主机

大规模系统由许多台主机组成，每台主机连接若干台设备。JAX 的做法是每台主机运行同一个程序（多进程的 SPMD）：先调用 `jax.distributed.initialize()` 建立联系，此后 `jax.devices()` 返回所有主机的设备，网格可以跨越主机，一个全局数组的各个分片分布在不同主机的设备上。每个进程只负责把**本地**设备需要的那部分输入数据装进去（例如用 `jax.make_array_from_process_local_data`）。程序的其余部分与单主机时相同。

## 接口小结

1. `jax.make_mesh(形状, 轴名, 轴类型)` 建立网格；`jax.P(...)` 与第 12 章的分片记号一一对应；`jax.set_mesh` 设定当前网格，`jax.device_put` 按分片放置数组；Explicit 模式下 `jax.typeof` 显示分片，如 `float32[8@x,16]`。（14.1 节）
2. 三种轴类型：Explicit（分片是类型，歧义处用 `out_sharding` 指定，不相容是类型错误，`jax.reshard` 显式重分片）、Auto（编译器决定，`with_sharding_constraint` 约束）、Manual（`shard_map` 中以局部视角编程）。（定义 14.1）
3. `shard_map` 的 `in_specs`/`out_specs` 描述局部块与全局数组的关系；集合原语 `psum`、`all_gather`、`psum_scatter`、`ppermute`、`all_to_all`、`axis_index`；输出声明为复制时会被检查；对复制的值做 `psum` 会乘以轴的大小。（14.3 节）
4. 通信与计算的重叠可以用 `ppermute` 循环手写，正确性由"第 $t$ 步持有哪一块"的不变量保证。（例 14.2）
5. FSDP + 张量并行在 Explicit 模式下就是对权重 `reshard`（all-gather）和对输出指定 `out_sharding`（all-reduce）；反向的梯度规约在语义上是 reduce-scatter，要在 HLO 中确认它没有被实现成 all-reduce 加切片。（14.4 节）
6. 显存工具：捐赠、重计算、`memory_analysis`、主机卸载；多主机时每个进程运行同一程序、装入本地数据。（14.5、14.6 节）

## 习题

**习题 14.1** ★ 写出下列分片的 PartitionSpec，以及在网格 $\{x: 2, y: 4\}$ 上每台设备的局部形状（全局形状 $(64, 128)$）：$A[I_x, J_y]$；$A[I_{xy}, J]$；$A[I, J_x]$；$A[I_y, J_x]$。

<details><summary>提示</summary>

沿大小为 $q$ 的轴（或轴组）分片，那一维除以 $q$。

</details>
<details><summary>答案</summary>

`P("x", "y")`，$(32, 32)$；`P(("x", "y"), None)`，$(8, 128)$；`P(None, "x")`，$(64, 64)$；`P("y", "x")`，$(16, 64)$。

</details>

**习题 14.2** ★ 在 Explicit 模式下，下列运算的结果类型是什么？哪些会报错？(a) `P("x", None)` 的 $(8, 16)$ 数组乘以 `P(None, None)` 的 $(16, 32)$ 数组；(b) `P(None, "y")` 的 $(8, 16)$ 数组乘以 `P("y", None)` 的 $(16, 32)$ 数组，不给 `out_sharding`；(c) 同 (b)，`out_sharding=P(None, "y")`；(d) `P("x", None)` 与 `P("x", None)` 的两个 $(8, 16)$ 数组相加。

<details><summary>提示</summary>

对照命题 12.3 与 14.2 节的例子。

</details>
<details><summary>答案</summary>

(a) `float32[8@x,32]`，无通信。(b) `ShardingTypeError`：收缩维被分片，必须指定输出分片。(c) `float32[8,32@y]`：沿 $y$ 做 reduce-scatter，散布到结果的第二维。(d) `float32[8@x,16]`，逐元素相加无通信。

</details>

**习题 14.3** ★（审查题）下面的 `shard_map` 想计算全局数组 `w`（在所有设备上复制）的元素和。在 $\{x: 4\}$ 的网格上，结果是正确值的几倍？怎样修改？

```python
@jax.shard_map(in_specs=jax.P(), out_specs=jax.P())
def total(w):
    return jax.lax.psum(w.sum(), "x")
```

<details><summary>提示</summary>

`in_specs=P()` 意味着每台设备拿到的是完整的 `w`。

</details>
<details><summary>答案</summary>

每台设备都算出完整的和，`psum` 把 4 个相同的值相加，得到 4 倍。修改：既然每台设备已有完整的 `w`，直接 `return w.sum()`；若想分摊计算，就让输入沿 $x$ 分片（`in_specs=P("x")`），每台设备只求局部和，再 `psum`。

</details>

**习题 14.4** ★ 在 `shard_map` 中用 `psum_scatter` 和 `all_gather` 实现 all-reduce，并说明它与直接用 `psum` 的关系。

<details><summary>提示</summary>

命题 7.5。

</details>
<details><summary>答案</summary>

```python
def all_reduce(x):                         # x: 本地的完整向量，形状 (n,)，n 能被轴大小整除
    part = jax.lax.psum_scatter(x, "x", scatter_dimension=0, tiled=True)   # (n/p,)
    return jax.lax.all_gather(part, "x", tiled=True)                        # (n,)
```

由命题 7.5，结果与 `psum(x, "x")` 相同；编译器对 `psum` 通常也是这样实现的。显式拆开的好处是可以在两步之间插入计算，例如序列并行中在 reduce-scatter 与 all-gather 之间做归一化（12.4 节），使这部分计算只在 $1/p$ 的数据上进行。

</details>

**习题 14.5** ★ 对例 14.2，设 $p = 8$，每台设备上 $X$ 的块为 $1024 \times 8192$（bf16），$W$ 的块为 $8192 \times 1024$，芯片 $P = 2.6 \times 10^{14}$ FLOP/s，链路 $\beta = 10^{11}$ 字节/秒。每一步的计算与通信各要多少时间？通信能被掩盖吗？

<details><summary>提示</summary>

每步计算一块 $1024 \times 8192$ 乘 $8192 \times 1024$；每步传递一块 $X$。

</details>
<details><summary>答案</summary>

计算 $2 \times 1024 \times 8192 \times 1024 \approx 1.7 \times 10^{10}$ FLOP，约 66 µs；通信 $1024 \times 8192 \times 2 = 16$ MiB，约 168 µs。通信比计算慢，不能完全掩盖，总时间约为 $7 \times 168 + 66 \approx 1.24$ ms（命题 7.10），而不重叠时约 $7 \times 168 + 8 \times 66 \approx 1.70$ ms。要完全掩盖，需要每块的计算更多（例如更大的 $F/p$），即张量并行的规模更小（命题 12.5）。

</details>

**习题 14.6** ★（审查题）一个 Auto 模式的训练程序在编译后的 HLO 中出现了对一个 $8192 \times 28672$ 的权重做 all-gather 的操作，而作者原本的意图是张量并行（权重沿 $F$ 分片、不应被收集）。可能的原因是什么？怎样排查和修改？

<details><summary>提示</summary>

Auto 模式下中间结果的分片由分区器推断。哪一步的分片可能与意图不一致？

</details>
<details><summary>答案</summary>

很可能某个与该权重相乘的激活被推断成了与意图不同的分片（例如沿 $F$ 而不是沿批量维分片，或者完全复制），使分区器认为收集权重比搬运激活便宜，或者遇到了命题 12.3 情形 4 那样的冲突。排查：在 HLO 中找到这个 all-gather 对应的源代码位置，检查它前后的数组分片。修改：用 `with_sharding_constraint` 固定激活的分片（批量维沿数据轴、隐藏维沿 $t$），或者把这些轴改为 Explicit 模式，让不一致在追踪时就报错。

</details>

**习题 14.7** ☆ 16 台主机，每台 8 台设备，网格 $\{d: 16, t: 8\}$，$t$ 轴在主机内。全局批量 $N \times D$ 的输入沿 $d$ 分片。每台主机需要装入多大的数据？若某台主机装入了全局批量，会发生什么？

<details><summary>提示</summary>

每台主机的 8 台设备在 $d$ 轴上的坐标相同。

</details>
<details><summary>答案</summary>

每台主机上的设备共享同一个 $d$ 坐标，所以只需要装入 $N/16 \times D$ 的数据（在主机内的 8 台设备上复制）。若每台主机都装入全局批量，数据加载的工作和主机内存都增加 16 倍，而且 `make_array_from_process_local_data` 的形状约定被违反，得到的全局数组会是错的。

</details>
