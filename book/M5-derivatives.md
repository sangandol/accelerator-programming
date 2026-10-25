# 间章 M5　导数、梯度与链式法则

训练一个模型，就是一遍遍地问：几十亿个参数，每个往哪边动一点，损失会变小？回答这个问题的是导数。第 11 章要用到四件事：一元函数的导数是"局部的斜率"；多元函数的梯度指向上升最快的方向，所以沿负梯度走一小步损失下降；复合函数的导数由链式法则一段一段乘起来；把链式法则从后往前算，就是反向传播，而"往后传"的那一步恰好是乘以一个矩阵的转置（命题 M3.5）。第 13 章的自动微分又把这些写成程序变换。这个间章把四件事讲清楚，每件都配一个带数字的例子。

## M5.1 导数是局部的斜率

函数 $f$ 在 $x$ 处的**导数**（derivative）$f'(x)$ 是

$$
f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h},
$$

即 $x$ 附近曲线的斜率。它最有用的读法是一个近似式：

$$
f(x + h) \approx f(x) + f'(x)\, h \qquad (h \text{ 很小}).
$$

输入改变一点点 $h$，输出大约改变 $f'(x) h$。

> **例 M5.1（几个常用的导数）**
>
> | $f(x)$ | $f'(x)$ | 在一点检验 |
> | --- | --- | --- |
> | $x^2$ | $2x$ | $x = 3$：$f(3.01) = 9.0601$，$f(3) + 6 \times 0.01 = 9.06$ |
> | $e^x$ | $e^x$ | $x = 0$：$e^{0.01} \approx 1.01005$，$1 + 1 \times 0.01 = 1.01$ |
> | $\ln x$ | $1/x$ | $x = 2$：$\ln 2.01 \approx 0.69813$，$0.69315 + 0.5 \times 0.01 = 0.69815$ |
> | $cx$（$c$ 是常数） | $c$ | 直线的斜率处处相同 |
>
> 和的导数是导数的和，常数倍的导数是导数的常数倍；乘积 $(uv)' = u'v + uv'$，商 $(u/v)' = (u'v - uv')/v^2$。

## M5.2 偏导数与梯度

多个输入的函数 $f(w_1, \dots, w_n)$，固定其余输入、只让 $w_i$ 变化，得到的导数称为**偏导数**（partial derivative）$\partial f / \partial w_i$。把全部偏导数排成一个向量，就是**梯度**（gradient）

$$
\nabla f = \Big(\frac{\partial f}{\partial w_1}, \dots, \frac{\partial f}{\partial w_n}\Big).
$$

一元的近似式推广成：输入改变一个小向量 $h$ 时，

$$
f(w + h) \approx f(w) + \sum_i \frac{\partial f}{\partial w_i} h_i = f(w) + \langle \nabla f(w), h \rangle .
$$

> **命题 M5.2（梯度是上升最快的方向）** 在所有长度为 $\varepsilon$ 的小步 $h$ 中，使近似的变化量 $\langle \nabla f, h \rangle$ 最大的是 $h = \varepsilon \nabla f / \lVert \nabla f \rVert$，最小的是它的相反方向。

*证明* 由柯西—施瓦茨不等式（命题 M2.6），$\lvert \langle \nabla f, h \rangle \rvert \le \lVert \nabla f \rVert \, \lVert h \rVert = \varepsilon \lVert \nabla f \rVert$，等号当且仅当 $h$ 与 $\nabla f$ 成比例。∎

所以要让 $f$ 变小，就沿 $-\nabla f$ 走一小步：$w \leftarrow w - \eta \nabla f(w)$，$\eta$ 是一个小的正数。这就是**梯度下降**，第 11 章的训练就是它。

> **例 M5.3（一步梯度下降）** $f(w_1, w_2) = w_1^2 + 3 w_2^2$。偏导数 $\partial f / \partial w_1 = 2 w_1$，$\partial f / \partial w_2 = 6 w_2$。在 $w = (1, 1)$ 处 $f = 4$，$\nabla f = (2, 6)$。取 $\eta = 0.1$，走一步到 $(1 - 0.2,\ 1 - 0.6) = (0.8, 0.4)$，$f = 0.64 + 0.48 = 1.12$，下降了。$w_2$ 方向弯得更厉害（系数 3），梯度在这个方向上更大，走得也更多。

## M5.3 链式法则

模型是一长串函数的复合。复合函数的导数怎样求？先看一元的情形：$y = g(x)$，$z = f(y)$。$x$ 改变 $h$，$y$ 约改变 $g'(x) h$；$y$ 改变这么多，$z$ 又约改变 $f'(y) \cdot g'(x) h$。所以

$$
\frac{dz}{dx} = f'(g(x))\, g'(x) = \frac{dz}{dy} \cdot \frac{dy}{dx} .
$$

导数一段一段地乘起来，这就是**链式法则**（chain rule）。多元的情形同理，只是一个输入可能经过好几条路影响输出，要把每条路的贡献加起来：

> **命题 M5.4（多元链式法则）** 设 $y_1, \dots, y_m$ 都是 $x_1, \dots, x_n$ 的函数，$z$ 是 $y_1, \dots, y_m$ 的函数。则
>
> $$
> \frac{\partial z}{\partial x_i} = \sum_{j=1}^{m} \frac{\partial z}{\partial y_j} \cdot \frac{\partial y_j}{\partial x_i} .
> $$

$x_i$ 改变一点，通过每个 $y_j$ 各带来一份变化，总变化是各份之和。证明见任何一本微积分教材，下面用数字验证一次。

> **例 M5.5（一个变量用了两次）** $z = (xy + x)^2$，在 $x = 2$、$y = 3$ 处求 $\partial z / \partial x$。拆成几步：$a = xy = 6$，$b = a + x = 8$，$z = b^2 = 64$。
>
> 从后往前：$\partial z / \partial b = 2b = 16$。$b = a + x$，所以 $\partial z / \partial a = 16$，并且 $x$ 通过"直接加进 $b$"这条路得到 $16 \times 1 = 16$。$a = xy$，$x$ 通过 $a$ 这条路得到 $16 \times y = 48$。两条路加起来，$\partial z / \partial x = 16 + 48 = 64$。
>
> 直接求导验证：$z = x^2 (y + 1)^2$，$\partial z / \partial x = 2x (y + 1)^2 = 2 \times 2 \times 16 = 64$。一致。$x$ 在计算中被用了两次，它的导数就是两份之和：前向"复制"一个值，反向就"加起来"（例 M3.6）。

## M5.4 雅可比矩阵与反向传播

把一步运算看成从 $\mathbb{R}^n$ 到 $\mathbb{R}^m$ 的映射 $y = f(x)$，它的全部偏导数排成一个 $m \times n$ 的矩阵，称为**雅可比矩阵**（Jacobian matrix）：

$$
J_f[j, i] = \frac{\partial y_j}{\partial x_i} .
$$

近似式写成 $f(x + h) \approx f(x) + J_f h$：在一点附近，任何光滑的映射都近似为一个线性映射，矩阵就是 $J_f$。命题 M5.4 用矩阵说就是：复合的雅可比矩阵是各段雅可比矩阵的乘积，$J_{g \circ f} = J_g J_f$。

> **例 M5.6（线性层的雅可比矩阵）** $y = Wx$，$W$ 是 $m \times n$ 矩阵。$y_j = \sum_i W[j, i] x_i$，所以 $\partial y_j / \partial x_i = W[j, i]$：雅可比矩阵就是 $W$ 本身。逐元素的运算 $y_i = \sigma(x_i)$ 的雅可比矩阵是对角矩阵，对角线上是 $\sigma'(x_i)$。

训练中最后总是一个数：损失 $L$。设计算是一串 $x \to y \to \dots \to L$。记 $\bar{y} = \nabla_y L$（损失对 $y$ 的梯度，与 $y$ 同形）。由命题 M5.4，

$$
\frac{\partial L}{\partial x_i} = \sum_j \frac{\partial L}{\partial y_j} \frac{\partial y_j}{\partial x_i} = \sum_j J_f[j, i]\, \bar{y}_j, \qquad \text{即} \qquad \bar{x} = J_f^{\mathsf T}\, \bar{y} .
$$

> **命题 M5.7（反向一步是乘以转置）** 设 $y = f(x)$，损失 $L$ 通过 $y$ 依赖于 $x$。则 $\nabla_x L = J_f^{\mathsf T}\, \nabla_y L$。对线性层 $y = Wx$，$\nabla_x L = W^{\mathsf T} \nabla_y L$。

于是求整串计算的梯度，就是从 $\nabla_L L = 1$ 出发，从后往前每一步乘以那一步雅可比矩阵的转置。这个从后往前的过程称为**反向传播**（backpropagation）。它不需要把任何雅可比矩阵真的写出来：线性层乘 $W^{\mathsf T}$，逐元素运算乘对角线上的 $\sigma'(x_i)$，每一步的代价和前向那一步相当。

> **命题 M5.8（为什么从后往前）** 设计算是 $k$ 段的复合，每段的雅可比矩阵是 $n \times n$，最后一段把 $n$ 维向量变成一个数（损失），$1 \times n$。则总梯度是乘积 $J_k J_{k-1} \cdots J_1$（最左边是 $1 \times n$ 的行向量）。从左往右乘，每一步是"行向量乘矩阵"，约 $n^2$ 次运算，共约 $k n^2$；从右往左乘，每一步是"矩阵乘矩阵"，约 $n^3$ 次，共约 $k n^3$。

*证明* 矩阵乘法满足结合律，两种次序结果相同，代价不同：$1 \times n$ 乘 $n \times n$ 用 $n^2$ 次乘加，$n \times n$ 乘 $n \times n$ 用 $n^3$ 次。∎

从左往右，就是从损失那一端往回乘，这正是反向传播。参数有几十亿个而损失只有一个数时，这个差别是决定性的：反向传播的代价只是前向的常数倍。第 13 章会把"从前往后乘一个向量"和"从后往前乘一个向量"分别称为 JVP 和 VJP。

> **例 M5.9（两层的反向，带数字）** $x = (1, 2)$，$y = Wx$，$W = \begin{pmatrix} 1 & 0 \\ 2 & -1 \end{pmatrix}$，于是 $y = (1, 0)$；$L = \frac12 (y_1^2 + y_2^2) = 0.5$。
>
> 反向：$\nabla_y L = (y_1, y_2) = (1, 0)$；$\nabla_x L = W^{\mathsf T} (1, 0) = (1 \cdot 1 + 2 \cdot 0,\ 0 \cdot 1 + (-1) \cdot 0) = (1, 0)$。
>
> 验证：$L = \frac12 \big(x_1^2 + (2x_1 - x_2)^2\big)$，$\partial L / \partial x_1 = x_1 + 2(2x_1 - x_2) = 1 + 0 = 1$，$\partial L / \partial x_2 = -(2x_1 - x_2) = 0$。一致。

## 本间章小结

1. 导数是局部斜率：$f(x + h) \approx f(x) + f'(x) h$；$(x^2)' = 2x$，$(e^x)' = e^x$，$(\ln x)' = 1/x$。（M5.1 节）
2. 梯度是全部偏导数组成的向量，指向上升最快的方向；梯度下降 $w \leftarrow w - \eta \nabla f$ 沿负梯度走一小步。（命题 M5.2、例 M5.3）
3. 链式法则：导数沿计算一段段相乘，一个输入经过几条路影响输出时，各条路的贡献相加。（命题 M5.4、例 M5.5）
4. 雅可比矩阵是一步运算的全部偏导数；复合的雅可比矩阵是乘积；损失是一个数时，反向一步是乘以雅可比矩阵的转置，线性层就是乘 $W^{\mathsf T}$；从损失往回乘，代价只是前向的常数倍。（命题 M5.7、M5.8）

## 习题

**习题 M5.1** ★ 用近似式 $f(x + h) \approx f(x) + f'(x) h$ 估计 $\sqrt{4.04}$，并与真值比较。

<details><summary>提示</summary>

$f(x) = \sqrt{x}$，$f'(x) = \frac{1}{2\sqrt{x}}$，取 $x = 4$、$h = 0.04$。

</details>
<details><summary>答案</summary>

$2 + \frac{1}{4} \times 0.04 = 2.01$。真值 $\sqrt{4.04} \approx 2.00998$，误差约 $2 \times 10^{-5}$，比 $h$ 小得多。

</details>

**习题 M5.2** ★ 对例 M5.3 的 $f$，从 $w = (1, 1)$ 出发，分别取 $\eta = 0.3$ 与 $\eta = 0.4$ 走一步，$f$ 各变成多少？哪一个出了问题，为什么？

<details><summary>提示</summary>

一步之后 $w_1 \leftarrow (1 - 2\eta) w_1$，$w_2 \leftarrow (1 - 6\eta) w_2$。

</details>
<details><summary>答案</summary>

$\eta = 0.3$：$w = (0.4, -0.8)$，$f = 0.16 + 1.92 = 2.08$。$\eta = 0.4$：$w = (0.2, -1.4)$，$f = 0.04 + 5.88 = 5.92$，比出发时的 4 还大。$w_2$ 方向每步乘以 $1 - 6\eta$，$\eta > 1/3$ 时绝对值超过 1，来回跳着越跳越远。步长要服从弯得最厉害的方向（第 11 章的例 11.13 会再遇到这件事）。

</details>

**习题 M5.3** ★ 设 $z = \sigma(u)$，$u = wx + b$，其中 $\sigma(u) = 1/(1 + e^{-u})$，$\sigma'(u) = \sigma(u)(1 - \sigma(u))$。在 $w = 2, x = 1, b = -2$ 处，求 $\partial z / \partial w$ 与 $\partial z / \partial b$。

<details><summary>提示</summary>

先算 $u = 0$，$\sigma(0) = 1/2$，$\sigma'(0) = 1/4$。再用链式法则：$\partial u / \partial w = x$，$\partial u / \partial b = 1$。

</details>
<details><summary>答案</summary>

$\partial z / \partial w = \sigma'(0) \cdot x = 1/4$，$\partial z / \partial b = \sigma'(0) \cdot 1 = 1/4$。

</details>

**习题 M5.4** ★ 一串计算有 $k = 50$ 段，每段的雅可比矩阵是 $4096 \times 4096$，最后得到一个数。按命题 M5.8，从损失往回乘与从输入往前乘，运算量各约是多少？差几倍？

<details><summary>提示</summary>

$n = 4096 = 2^{12}$。

</details>
<details><summary>答案</summary>

往回乘约 $50 \times 4096^2 \approx 8.4 \times 10^8$ 次乘加；往前乘约 $50 \times 4096^3 \approx 3.4 \times 10^{12}$ 次，差 $n = 4096$ 倍。

</details>
