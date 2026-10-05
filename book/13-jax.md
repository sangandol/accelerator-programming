# 第 13 章　JAX 的计算模型

前两部分讲的是机器与算法，没有涉及具体的编程语言。从本章起进入程序。JAX 让人用 NumPy 的写法写数值函数，再对函数做**变换**：编译（`jit`）、求导（`grad`）、向量化（`vmap`）、分布到多台设备（第 14 章）。理解 JAX 只需抓住一件事：**JAX 先把 Python 函数追踪成一个有类型的一阶程序（jaxpr），所有变换都是对这个程序的解释**。JAX 的各种规则，例如形状必须静态、函数必须是纯的、控制流要用专门的原语、随机数要显式传递密钥，都由此而来。

> **在体系中的位置**：下层是前言的数组记号、第 2 章的精度、第 6 章的融合、第 10 章的扫描。本章给出 JAX 的数组与纯函数、追踪与 jaxpr、jit 与 XLA 编译、自动微分（JVP 与 VJP）、vmap、控制流原语、随机数与精度。第 14 章在此之上加入分片；第 16 章的 Pallas kernel 也是在 jaxpr 层面上定义的函数。GPU 路线的读者若不用 JAX，可以跳过本部分。

## 13.1 数组与纯函数

JAX 的数组（`jax.Array`）是**不可变**的 $n$ 维数组，带有形状、数据类型，以及它所在的设备（第 14 章还会加上分片）。`jnp` 与 `jax.lax` 中的运算都返回新数组。看似"原地修改"的写法 `x.at[i].set(v)` 返回一个新数组；编译器在确认旧数组不再使用时会复用它的存储。

**定义 13.1（纯函数）** 一个函数是**纯的**，若它的输出只依赖于输入，且没有副作用（不修改全局状态、不做输入输出）。

JAX 的变换只对纯函数有定义。原因在下一节：变换可能只运行一次 Python 函数（追踪），之后反复执行追踪得到的程序；也可能重新排列、复制或删除其中的计算。副作用在追踪时发生一次，之后就不再发生。

**pytree**：由元组、列表、字典嵌套而成、叶子是数组的结构。JAX 的变换把 pytree 当作一个结构化的参数，对其叶子逐一处理。模型的参数通常就是一个 pytree。

## 13.2 追踪与 jaxpr

**定义 13.2（抽象值与追踪）** 一个**抽象值**是数组的类型：形状与数据类型（以及第 14 章的分片），例如 `f32[4,8]`。**追踪**是用带抽象值的占位对象（tracer）调用 Python 函数：每执行一个 JAX 运算，就记下一条"原语作用于哪些变量、产生什么类型的结果"的方程。记下的程序称为 **jaxpr**。

`jax.make_jaxpr` 打印追踪的结果：

```python
jax.make_jaxpr(lambda x, w: jnp.tanh(x @ w).sum())(x, w)   # x: f32[4,8], w: f32[8,3]
```

```text
{ lambda ; a:f32[4,8] b:f32[8,3]. let
    c:f32[4,3] = dot_general[
      dimension_numbers=(([1], [0]), ([], []))
      preferred_element_type=float32
    ] a b
    d:f32[4,3] = tanh c
    e:f32[] = reduce_sum[axes=(0, 1) out_sharding=None] d
  in (e,) }
```

jaxpr 是一个由原语组成的直线程序，每个变量都有静态的类型。`x @ w` 变成了 `dot_general`，它的参数说明收缩的是 `a` 的第 1 维和 `b` 的第 0 维。

追踪带来三条规则：

1. **不能依据数组的值做 Python 控制流。** 追踪时数组只是占位对象，没有值。`if x.sum() > 0:` 在 `jit` 中会报错（`TracerBoolConversionError`）。依据值的选择要写成 `jnp.where`（两边都算，逐元素选择）或 `lax.cond`（13.6 节）。依据形状、数据类型这些静态信息的 Python 控制流则没有问题。
2. **形状必须静态。** 输出的形状不能依赖于输入的值。例如 `x[x > 0]` 的长度取决于有多少个正数，在 `jit` 中不允许（`NonConcreteBooleanIndexError`）。通常的替代是固定形状加掩码，例如 `jnp.where(x > 0, x, 0)`。
3. **Python 循环会被展开。** `for i in range(1000):` 在追踪时执行 1000 次，记下 1000 份方程，编译慢、程序大。需要在设备上执行的循环要写成 `lax.fori_loop` 或 `lax.scan`。

**静态参数。** 用 `jax.jit(f, static_argnums=...)` 声明某些参数是静态的：它们的值在追踪时就确定，参与 Python 控制流；每个不同的值会触发一次重新追踪与编译。

## 13.3 jit 与编译

`jax.jit(f)` 的执行过程：

1. 第一次以某种抽象类型调用时，追踪得到 jaxpr；
2. 把 jaxpr 降低为 StableHLO（一种张量程序的中间表示）；
3. 交给 XLA 编译器，为目标设备生成机器程序；
4. 以（抽象类型，静态参数）为键缓存编译结果。之后用同样类型的参数调用时直接执行。

编译要几秒到几分钟，所以要避免反复编译：例如每一步输入的形状都不同时，每一步都会重新编译，常用的办法是把长度补齐到少数几个固定的档位。

XLA 在编译时完成了前两部分讲到的许多事：把逐元素运算与相邻的运算**融合**（定义 6.7），为每个数组选择**布局**（第 8 章），为矩阵乘法选择**分块**（第 6 章），安排计算与访存的次序，规划内存的复用，以及在多设备时插入**集合通信**（第 14 章）。

审查一个 JAX 程序时，可以查看编译的各个阶段：`jax.jit(f).lower(x).as_text()` 给出 StableHLO，`.compile().as_text()` 给出优化后的 HLO。值得检查的是：逐元素运算是否被融合；矩阵乘法的输入输出类型是否符合预期（例如是否在 f32 中累加）；是否出现了意外的转置、拷贝或集合通信。

**两个实用的细节**：`donate_argnums` 声明某个输入在调用后不再使用，允许输出复用它的存储，例如训练中旧参数与新参数共用存储。调用 `jit` 函数是**异步**的：函数立即返回，设备在后台执行，只有在读取结果时才等待它完成。

## 13.4 自动微分

设 $f: \mathbb{R}^n \to \mathbb{R}^m$ 在 $x$ 处的雅可比矩阵为 $J_f(x) \in \mathbb{R}^{m \times n}$。

**定义 13.3（JVP 与 VJP）** **JVP**（前向模式）计算 $v \mapsto J_f(x)\, v$；**VJP**（反向模式）计算 $u \mapsto u^{\mathsf T} J_f(x)$。`jax.jvp(f, (x,), (v,))` 返回 $(f(x), J_f(x) v)$；`jax.vjp(f, x)` 返回 $f(x)$ 和函数 $u \mapsto J_f(x)^{\mathsf T} u$。对标量函数 $f$，$\nabla f(x) = J_f(x)^{\mathsf T} \cdot 1$，这就是 `jax.grad`。

两种模式的代价都只是 $f$ 本身代价的常数倍。区别在于：一次 JVP 给出雅可比矩阵的一列，一次 VJP 给出一行；神经网络的损失是标量，参数有上亿个，所以训练用 VJP。VJP 的代价是要保存前向计算的中间结果（称为残差），供反向使用，这就是第 11 章的"激活显存"。

**JAX 怎样实现 VJP。** 先对 $f$ 做 JVP，并把只依赖于 $x$ 的部分与依赖于切向量 $v$ 的部分分开，得到线性映射 $v \mapsto J_f(x) v$ 的程序；再对这个线性程序做**转置**。每个线性原语都有转置规则，例如：

| 线性原语 | 转置 |
| --- | --- |
| $v \mapsto v W$ | $u \mapsto u W^{\mathsf T}$ |
| $v \mapsto W v$ | $u \mapsto W^{\mathsf T} u$ |
| 广播 | 沿广播的维求和 |
| 切片 | 补零 |
| 求和 | 广播 |

**命题 13.4** 对 $y = xW$，VJP 给出 $\bar{x} = \bar{y} W^{\mathsf T}$ 与 $\bar{W} = x^{\mathsf T} \bar{y}$，其中 $\bar{y}$ 是输出的余切向量。

*证明* $y$ 对 $(x, W)$ 是双线性的。固定 $W$，$x \mapsto xW$ 的转置是 $\bar{y} \mapsto \bar{y} W^{\mathsf T}$；固定 $x$，$W \mapsto xW$ 的转置是 $\bar{y} \mapsto x^{\mathsf T} \bar{y}$（对内积 $\langle \bar{y}, xW \rangle = \operatorname{tr}(\bar{y}^{\mathsf T} x W) = \langle x^{\mathsf T} \bar{y}, W \rangle$ 验证）。∎

这就是命题 11.3 中"反向是前向的两倍"的来源。

**重计算与自定义导数。**

- `jax.checkpoint`（也称 remat）：被它包装的函数在前向时不保存中间结果，反向时重新计算。可以指定策略，例如只保存矩阵乘法的结果、重算其余部分。这是 11.3 节"以运算换显存"的实现。
- `jax.custom_vjp`：为一个函数手写反向传播。在数值上需要特别处理时（例如防止溢出），或者前向是一个手写 kernel、反向也要用手写 kernel 时（第 18 章的 flash attention），都要用到它。

变换可以任意复合：`jax.grad(jax.grad(f))` 是二阶导数，`jax.jit(jax.vmap(jax.grad(f)))` 是编译后的逐样本梯度。

## 13.5 vmap

**定义 13.5（vmap）** `jax.vmap(f, in_axes)` 给 $f$ 的输入加上一个批量维：对所有 $b$，

$$
\operatorname{vmap}(f)(X)[b] = f(X[b]).
$$

它不是用循环实现的：每个原语都有**批处理规则**，把"对一批输入各做一次"改写成"对整批做一次"。例如对矩阵 $M$ 的每一行与同一个向量 $v$ 做点积，`jax.vmap(jnp.dot, in_axes=(0, None))(M, v)` 追踪得到的就是一个矩阵乘向量的 `dot_general`，而不是若干次点积。所以可以放心地为一个样本写函数，再用 `vmap` 得到高效的批量版本。

## 13.6 控制流原语

| 原语 | 语义 | 说明 |
| --- | --- | --- |
| `lax.cond(p, f, g, *ops)` | $p$ 为真时算 $f$，否则算 $g$ | 两个分支都被追踪；运行时只执行一个。在 `vmap` 下若 $p$ 是批量的，退化为两边都算再选择，与 SIMD 的掩码（4.3 节）相同 |
| `lax.while_loop(c, body, init)` | 当 $c$ 为真时反复执行 `body` | 迭代次数在运行时决定；不能做反向模式求导，因为迭代次数未知、无法预先安排保存残差 |
| `lax.fori_loop(lo, hi, body, init)` | $i$ 从 `lo` 到 `hi` 的循环 | 边界为静态时可以反向求导 |
| `lax.scan(f, c, xs)` | $(c, y_t) = f(c, x_t)$，返回最终的 $c$ 与堆叠的 $y$ | 循环体只编译一次；反向求导时保存每一步的残差 |

`lax.scan` 是顺序执行的。若循环体是一个满足结合律的合并（第 10 章），可以用 `lax.associative_scan` 得到并行的前缀算法（命题 10.9）。在训练中，常用 `scan` 依次执行结构相同的各层，这样每层只编译一次。

## 13.7 随机数

JAX 的随机数用显式的**密钥**：

```python
key = jax.random.key(0)
k1, k2 = jax.random.split(key)
x = jax.random.normal(k1, (1024,))
```

同一个密钥总产生同样的随机数；要得到新的随机数，就要用 `split` 派生新的密钥。这是纯函数的要求：随机数生成器没有隐藏的状态。默认的生成器是**基于计数器**的：第 $i$ 个随机数是 $\operatorname{hash}(\text{key}, i)$，所以各个位置可以独立、并行地计算，结果与在几台设备上计算无关。代价是每个随机数要计算一次哈希；第 2 章说过，有的向量单元没有整数乘法器，以乘法为主的哈希在那里很贵，以加法、异或、移位为主的哈希则便宜得多。

## 13.8 数据类型与精度

- 默认的浮点类型是 f32（64 位类型默认关闭）。`x.astype(jnp.bfloat16)` 转为 bf16。
- **类型提升**：bf16 与 f32 的数组运算得到 f32；但 Python 的数字字面量是"弱类型"的，不会提升数组的类型：`x_bf16 * 2.0` 仍是 bf16，`x_bf16 * jnp.float32(2)` 是 f32。
- **矩阵乘法**：两个 bf16 相乘默认得到 bf16。用 `preferred_element_type=jnp.float32` 得到 f32 结果（习题 2.6）。`precision` 参数控制 f32 输入时的计算精度：`lax.Precision.DEFAULT`、`HIGH`、`HIGHEST`（推论 2.15）。

## 接口小结

1. JAX 数组不可变；变换只对纯函数有定义；参数是 pytree。（13.1 节）
2. 追踪用抽象值（形状、类型）运行 Python 函数，得到 jaxpr。因此不能依据值做 Python 控制流，形状必须静态，Python 循环会展开。（定义 13.2、13.2 节）
3. `jit` 追踪、降低、由 XLA 编译并按类型缓存；XLA 负责融合、布局、分块、调度和集合通信；形状变化会导致重新编译。（13.3 节）
4. 自动微分：JVP 给出雅可比矩阵乘向量，VJP 给出向量乘雅可比矩阵；VJP 由 JVP 加线性转置得到，需要保存残差。$y = xW$ 的 VJP 是 $\bar{y}W^{\mathsf T}$ 与 $x^{\mathsf T}\bar{y}$。`checkpoint` 以重算换显存，`custom_vjp` 手写反向。（定义 13.3、命题 13.4）
5. `vmap` 按原语的批处理规则加批量维，语义是逐个应用。（定义 13.5）
6. 控制流用 `cond`、`while_loop`、`fori_loop`、`scan`；`while_loop` 不能反向求导；`associative_scan` 是并行前缀。（13.6 节）
7. 随机数用显式密钥与 `split`，生成器基于计数器。（13.7 节）
8. bf16 与 f32 混合得 f32，Python 字面量是弱类型；bf16 矩阵乘法默认输出 bf16，用 `preferred_element_type` 指定 f32。（13.8 节）

## 习题

**习题 13.1** ★ 不运行代码，写出 `jax.make_jaxpr(lambda x: jnp.exp(x - x.max()))(x)` 的 jaxpr 中依次出现的原语（$x$ 为 `f32[8]`）。

<details><summary>提示</summary>

`x.max()` 是一次归约；`x - (标量)` 需要先把标量广播成 `f32[8]`（或由减法原语直接处理标量）。

</details>
<details><summary>答案</summary>

`reduce_max`（得到 `f32[]`），`sub`（`f32[8]` 减 `f32[]`，结果 `f32[8]`；标量直接作为操作数，jaxpr 中没有单独的广播），`exp`。实际输出为：

```text
{ lambda ; a:f32[8]. let
    b:f32[] = reduce_max[axes=(0,) out_sharding=None] a
    c:f32[8] = sub a b
    d:f32[8] = exp c
  in (d,) }
```

</details>

**习题 13.2** ★（审查题）下面的代码在 `jit` 下报错。说明原因，给出两种修改，并说明它们在语义和代价上的区别。

```python
@jax.jit
def clip_or_negate(x):
    if x.sum() > 0:
        return jnp.minimum(x, 1.0)
    return -x
```

<details><summary>提示</summary>

规则 1（13.2 节）。`jnp.where` 与 `lax.cond` 各算了几个分支？

</details>
<details><summary>答案</summary>

追踪时 `x.sum() > 0` 是一个没有值的占位对象，不能转换为 Python 的布尔值。修改一：`return jnp.where(x.sum() > 0, jnp.minimum(x, 1.0), -x)`，两个分支都计算，再选择。修改二：`return lax.cond(x.sum() > 0, lambda x: jnp.minimum(x, 1.0), lambda x: -x, x)`，运行时只执行一个分支。这里两个分支都很便宜，`where` 更简单；分支很贵时用 `cond`。若函数之后被 `vmap` 且条件是批量的，`cond` 也会退化为两边都算。

</details>

**习题 13.3** ★ 设 $y = \operatorname{softmax}(x)$，$x \in \mathbb{R}^n$。求 VJP：给定 $\bar{y}$，求 $\bar{x}$。

<details><summary>提示</summary>

$\partial y_i / \partial x_j = y_i(\delta_{ij} - y_j)$。

</details>
<details><summary>答案</summary>

$J = \operatorname{diag}(y) - y y^{\mathsf T}$，对称，所以 $\bar{x} = J \bar{y} = y \odot \bar{y} - y\,(y^{\mathsf T}\bar{y}) = y \odot (\bar{y} - \langle y, \bar{y} \rangle)$。只需要一次点积和逐元素运算，不必构造 $n \times n$ 的雅可比矩阵。注意力的反向传播（第 18、26 章）用的就是这个公式。

</details>

**习题 13.4** ★ 设 `M` 是 `f32[B, D]`，`W` 是 `f32[D, F]`。`jax.vmap(lambda m: m @ W)(M)` 与 `jax.vmap(lambda m, w: m @ w)(M, Ws)`（`Ws` 是 `f32[B, D, F]`）分别会被批处理成什么运算？哪一个能用到矩阵单元的高效形式？

<details><summary>提示</summary>

第一个中 `W` 没有批量维；第二个中每个样本有自己的权重。

</details>
<details><summary>答案</summary>

第一个变成一次 $B \times D$ 乘 $D \times F$ 的矩阵乘法（`dot_general` 的批量维为空），正是矩阵单元擅长的形式。第二个变成 $B$ 个独立的向量乘矩阵（`dot_general` 带批量维 `([0], [0])`），每个样本只有一行，强度约为 $2/s$（命题 6.16），访存受限。`vmap` 保持语义，但不能把访存受限的计算变成计算受限的计算。

</details>

**习题 13.5** ★ 下面的函数用 `while_loop` 计算 $x^n$（$n = 3$），对它求导时报错。为什么？怎样修改才能求导？

```python
def power(x):
    return lax.while_loop(lambda c: c[0] < 3, lambda c: (c[0] + 1, c[1] * x), (0, 1.0))[1]
```

<details><summary>提示</summary>

13.6 节的表。

</details>
<details><summary>答案</summary>

`while_loop` 的迭代次数在运行时才知道，反向模式无法事先安排保存每一步的残差，所以 JAX 拒绝对它做反向求导。迭代次数实际上是静态的 3，改写为 `lax.fori_loop(0, 3, lambda i, c: c * x, 1.0)` 即可，导数在 $x = 2$ 处为 $3x^2 = 12$。

</details>

**习题 13.6** ★（审查题）下面的代码为两层网络初始化权重。找出问题。

```python
key = jax.random.key(0)
w1 = jax.random.normal(key, (D, F))
w2 = jax.random.normal(key, (F, D))
```

<details><summary>提示</summary>

同一个密钥产生同样的随机数。

</details>
<details><summary>答案</summary>

两次使用了同一个密钥。若 $D = F$，`w2` 与 `w1` 完全相同；即使形状不同，两者也高度相关（取自同一串随机数）。应当先 `k1, k2 = jax.random.split(key)`，分别用 `k1`、`k2`。

</details>

**习题 13.7** ☆ 一个训练循环每一步的输入序列长度不同，在 1 到 4096 之间任意取值，每一步都调用同一个 `jit` 函数。会发生什么？怎样修改？

<details><summary>提示</summary>

编译缓存的键是抽象类型。

</details>
<details><summary>答案</summary>

每出现一个新的长度就重新追踪和编译一次，最多可达数千次编译。修改：把长度向上补齐到少数几个档位（例如 2 的幂），用掩码忽略补齐的部分。编译次数降到十几次，代价是补齐部分的无用计算。

</details>

**习题 13.8** ★ 设 `x` 是 bf16 数组，`y` 是 f32 数组。下列表达式的结果类型各是什么？`x * 2.0`；`x * jnp.float32(2.0)`；`x + y`；`x @ x.T`；`jnp.dot(x, x.T, preferred_element_type=jnp.float32)`。

<details><summary>提示</summary>

13.8 节。

</details>
<details><summary>答案</summary>

bf16；f32；f32；bf16；f32。第三、四个最容易出错：前者可能意外地把大数组提升为 f32，使显存和带宽加倍；后者可能意外地在 bf16 中得到矩阵乘法的结果。

</details>
