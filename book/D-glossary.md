# 附录 D　术语对照

按首次详细介绍的章节排列。硬件单元、指令与 API 的名字在正文中保留英文。间章（数学工具）的术语单列在最前面，"首见"一栏写作 M1.2 这样的形式。

## 间章　数学工具

| 中文 | 英文 | 首见 |
| --- | --- | --- |
| 下取整、上取整 | floor, ceiling | M1.2 |
| 数学归纳法 | mathematical induction | M1.4 |
| 渐近记号（大 O、$\Omega$、$\Theta$） | asymptotic notation (big O, Omega, Theta) | M1.5 |
| 均值不等式 | AM–GM inequality | M2.2 |
| 柯西—施瓦茨不等式 | Cauchy–Schwarz inequality | M2.3 |
| 单射、满射、双射、置换 | injection, surjection, bijection, permutation | M3.1 |
| 线性映射、转置 | linear map, transpose | M3.2 |
| 二元域 $\mathbb{F}_2$、比特向量 | field with two elements, bit vector | M3.3 |
| 子空间、维数、核、像、秩 | subspace, dimension, kernel, image, rank | M3.4 |
| 陪集、纤维 | coset, fiber | M3.5 |
| 分布、事件、抽样 | distribution, event, sampling | M4.1 |
| 联合分布、条件概率、独立 | joint distribution, conditional probability, independence | M4.2 |
| 无偏、尾和公式 | unbiased, tail-sum formula | M4.3 |
| 熵、交叉熵、KL 散度 | entropy, cross-entropy, KL divergence | M4.4 |
| 总变差距离 | total variation distance | M4.5 |
| 导数、偏导数、梯度 | derivative, partial derivative, gradient | M5.1 |
| 链式法则 | chain rule | M5.3 |
| 雅可比矩阵、反向传播 | Jacobian matrix, backpropagation | M5.4 |

## 第一部分　硬件

| 中文 | 英文 | 首见 |
| --- | --- | --- |
| 比特、逻辑门、真值表 | bit, logic gate, truth table | 1.1 |
| 与、或、异或、非 | AND, OR, XOR, NOT | 1.1 |
| 选择器 | multiplexer (mux) | 1.1 |
| 组合电路、扇出、规模、深度 | combinational circuit, fan-out, size, depth | 1.2 |
| 门延迟、关键路径 | gate delay, critical path | 1.3 |
| 半加器、全加器、多数函数 | half adder, full adder, majority | 1.4 |
| 行波进位加法器 | ripple-carry adder | 1.4 |
| 时钟、时钟周期、上升沿 | clock, clock period, rising edge | 1.5 |
| 寄存器、触发器、同步电路 | register, flip-flop, synchronous circuit | 1.5 |
| 进位产生、传递、吸收 | generate, propagate, kill | 1.6 |
| 产生信号、传递信号 | generate signal, propagate signal | 1.6 |
| 前缀问题（扫描）、归约 | prefix problem (scan), reduction | 1.8 |
| 前缀网络、节点 | prefix network, node | 1.8 |
| 超前进位加法器 | carry-lookahead adder | 1.10 |
| 进位保留加法、3:2 压缩、Wallace 树 | carry-save addition, 3:2 compression, Wallace tree | 1.11 |
| 工作量、深度、调度 | work, depth (span), schedule | 1.12 |
| 部分积 | partial product | 2.1 |
| 尾数、指数、次正规数 | significand, exponent, subnormal | 2.2 |
| 偏置、ulp | (exponent) bias, unit in the last place | 2.2 |
| 单位舍入误差 | unit roundoff | 2.3 |
| 相消、融合乘加 | cancellation, fused multiply-add (FMA) | 2.4 |
| 混合精度 | mixed precision | 2.7 |
| 块缩放 | block scaling (microscaling) | 2.8 |
| 随机舍入 | stochastic rounding | 2.10 |
| 随机变量、期望、方差、标准差 | random variable, expectation, variance, standard deviation | 2.10 |
| 全序、整数键 | total order (totalOrder), integer key | 2.11 |
| 寄存器堆、端口 | register file, port | 3.1 |
| 译码器、写使能 | decoder, write enable | 3.1 |
| 字线、位线 | word line, bit line | 3.2 |
| 存储体、冲突度、交错 | bank, conflict degree, interleaving | 3.3 |
| 行缓冲、突发 | row buffer, burst | 3.4 |
| 高带宽存储 | high-bandwidth memory (HBM) | 3.4 |
| 存储层次、缓存、缓存行、便签存储 | memory hierarchy, cache, cache line, scratchpad | 3.6 |
| 延迟、发射间隔 | latency, issue interval | 4.1 |
| 通道、掩码、向量寄存器 | lane, mask, vector register | 4.3 |
| 循环移位 | rotation | 4.4 |
| 静态调度、超长指令字、指令包、槽 | static scheduling, VLIW, bundle, slot | 4.6 |
| 记分板、线程 | scoreboard, thread | 4.7 |
| 完成计数器（同步标志、信号量） | completion counter (sync flag, semaphore) | 4.5 |
| 脉动阵列、斜排 | systolic array, skew | 5.2 |
| 时空映射、广播 | space-time mapping, broadcast | 5.3 |
| 输出驻留、权重驻留、输入驻留 | output / weight / input stationary | 5.3 |
| MMA 操作、点积单元 | matrix multiply-accumulate (MMA), dot-product unit | 5.5 |
| 结构化稀疏 | structured sparsity (2:4) | 5.6 |
| 算术强度、屋顶线、拐点 | arithmetic intensity, roofline, ridge point | 6.2 |
| 访存受限、计算受限 | memory-bound, compute-bound | 6.2 |
| 融合 | fusion | 6.3 |
| 波次量化 | wave quantization | 6.8 |
| 链路 | link | 7.1 |
| 拓扑、环面、直径、二分带宽 | topology, torus, diameter, bisection bandwidth | 7.2 |
| 集合操作 | collective operation | 7.3 |
| 信用 | credit | 7.9 |

## 第二部分　计算的共同结构

| 中文 | 英文 | 首见 |
| --- | --- | --- |
| 布局、步长、视图 | layout, stride, view | 8.1 |
| 分块布局、填充 | tiled layout, padding | 8.2 |
| 子通道 | sublane | 8.3 |
| 打包 | packing | 8.4 |
| 交错 | swizzle | 8.5 |
| 线性布局 | linear layout | 8.6 |
| 访存合并 | (memory) coalescing | 8.7 |
| 先行发生、数据竞争 | happens-before, data race | 9.1 |
| 屏障 | barrier | 9.2 |
| 软件流水线、序幕、尾声 | software pipelining, prologue, epilogue | 9.4 |
| 释放、获取、栅栏、作用域 | release, acquire, fence, scope | 9.6 |
| 幺半群 | monoid | 10.1 |
| 在线 softmax | online softmax | 10.2 |
| 名次计数、败者树 | rank counting, loser tree | 10.4 |
| 可复现 | reproducible | 10.6 |
| 批不变 | batch-invariant | 10.6 |
| 语言模型、参数、logits | language model, parameters, logits | 11.1 |
| 词元、头 | token, head | 11.2 |
| 残差、激活函数 | residual connection, activation function | 11.2 |
| 分组查询注意力、多头潜在注意力 | GQA, MLA | 11.2、11.8 |
| 损失、交叉熵损失、困惑度 | loss, cross-entropy loss, perplexity | 11.4 |
| 小批量梯度下降、步、学习率 | minibatch gradient descent, step, learning rate | 11.4 |
| 前向传播、反向传播、激活 | forward pass, backpropagation, activations | 11.4 |
| 模型 FLOP 利用率 | model FLOPs utilization (MFU) | 11.5 |
| 优化器、动量、Adam、主参数 | optimizer, momentum, Adam, master weights | 11.6 |
| 重计算 | rematerialization (activation checkpointing) | 11.6 |
| 预训练、监督微调、强化学习、蒸馏 | pretraining, supervised fine-tuning (SFT), reinforcement learning (RL), distillation | 11.7 |
| 策略梯度、基线、同策略 | policy gradient, baseline, on-policy | 11.7 |
| 预填充、解码 | prefill, decode | 11.8 |
| 混合专家、路由器、分组矩阵乘法 | mixture of experts, router, grouped matmul | 11.9 |
| 投机解码、草稿模型、总变差距离 | speculative decoding, draft model, total variation distance | 11.11 |
| 连续批处理、分段 prefill | continuous batching, chunked prefill | 11.12 |
| 分页 KV cache、页、页表 | paged KV cache, page, block table | 11.12 |
| prefill–decode 分离 | prefill–decode disaggregation | 11.12 |
| 网格、网格轴、分片、复制 | mesh, mesh axis, sharding, replication | 12.1 |
| 收缩维 | contracting dimension | 12.2 |
| 数据并行、张量并行、流水线并行、专家并行、上下文并行 | DP, TP, PP, EP, CP | 12.3–12.7 |
| 空泡 | (pipeline) bubble | 12.6 |

## 第三部分　JAX

| 中文 | 英文 | 首见 |
| --- | --- | --- |
| 纯函数 | pure function | 13.1 |
| 追踪、抽象值 | tracing, abstract value | 13.2 |
| 雅可比向量积、向量雅可比积 | JVP, VJP | 13.4 |
| 线性化、转置 | linearize, transpose | 13.4 |
| 批处理规则 | batching rule | 13.5 |
| 密钥 | (PRNG) key | 13.7 |
| 弱类型 | weak type | 13.8 |
| 轴类型 | axis type (Explicit / Auto / Manual) | 14.2 |

## 第四部分　TPU

| 中文 | 英文 | 首见 |
| --- | --- | --- |
| 标量单元、向量单元、矩阵单元 | scalar unit, VPU, MXU | 15.1 |
| 扩展一元流水线、跨通道单元 | EUP, XLU | 15.3 |
| 暂存区、权重寄存器 | staging register, gains (weight) register | 15.4 |
| 向量存储、标量存储 | VMEM, SMEM | 15.5 |
| 芯粒 | chiplet | 15.6 |
| 切片 | slice | 15.8 |
| 下标映射 | index map | 16.1 |
| 收集；散布 | gather; scatter | 定义 16.14 |
| 有状态生成器；计数器式生成器 | stateful PRNG; counter-based PRNG | 定义 16.19 |
| 精确往返 | exact round trip | 定义 20.9 |
| 局部周期计数器；全局时间计数器 | local cycle counter (LCC); global time counter (GTC) | 定义 20.14 |
| 维度语义 | dimension semantics | 16.5 |
| 标量预取 | scalar prefetch | 16.7 |
| 解释器 | interpreter (interpret mode) | 16.8 |
| 参差的批 | ragged batch | 18.6 |
| 远程 DMA | remote DMA | 19.1 |
| 向量发射队列 | vector issue FIFO (VIF) | 20.2 |

## 第五部分　GPU

| 中文 | 英文 | 首见 |
| --- | --- | --- |
| 流式多处理器 | streaming multiprocessor (SM) | 22.1 |
| 线程、warp、线程块 | thread, warp, thread block (CTA) | 22.2 |
| 共享内存、溢出 | shared memory, spill | 22.3 |
| 占用率 | occupancy | 22.4 |
| warpgroup、张量存储 | warpgroup, tensor memory (TMEM) | 22.5 |
| 片段 | fragment | 22.5 |
| 张量搬运引擎、张量描述符 | TMA, tensor map (descriptor) | 22.6 |
| 线程块簇、分布式共享内存 | thread block cluster, distributed shared memory | 22.7 |
| 网格步长循环 | grid-stride loop | 23.1 |
| 洗牌 | shuffle | 23.3 |
| 流、事件、图 | stream, event, CUDA graph | 23.8 |
| 生产者、消费者、warp 专门化 | producer, consumer, warp specialization | 24.4 |
| 乒乓 | ping-pong | 24.4 |
| CTA 对 | CTA pair | 24.5 |
| 持久化 kernel | persistent kernel | 24.6 |
| 光栅化 | rasterization (tile order) | 24.6 |
| megakernel | megakernel | 24.8 |
| tile 程序 | tile program | 25.1 |
| 结尾 | epilogue | 26.4 |
| 沿键值方向切分 | split-KV (flash-decoding) | 27.5 |
| 块稀疏矩阵、注意力状态 | block-sparse row (BSR), attention state | 27.7 |
| 对称内存 | symmetric memory | 28.2 |

## 第六部分　大模型推理系统

| 中文 | 英文 | 首见 |
| --- | --- | --- |
| 检查点 | checkpoint (safetensors) | 29.1 |
| 字节对编码、合并表、预切分 | byte-pair encoding (BPE), merges, pre-tokenization | 29.2 |
| 对话模板、特殊词元 | chat template, special token | 29.2 |
| 旋转位置编码 | rotary position embedding (RoPE) | 29.3 |
| 温度、贪心解码、top-k、top-p | temperature, greedy decoding, top-k, top-p (nucleus) sampling | 29.4 |
| 逆分布函数法、Gumbel-max | inverse-CDF sampling, Gumbel-max trick | 29.4 |
| 约束输出 | constrained (structured) decoding | 29.4 |
| 停止串 | stop sequence | 29.5 |
| 吸收 | (weight) absorption in MLA | 30.2 |
| 滑动窗口注意力、环形缓冲、注意力汇 | sliding window attention (SWA), ring buffer, attention sink | 30.3 |
| 压缩、索引器、稀疏选择 | compression, indexer, sparse selection | 30.4 |
| 跨层共享、因果编码器—解码器 | cross-layer KV sharing, YOCO, causal encoder-decoder (CED) | 30.5 |
| 线性注意力、状态、delta 规则 | linear attention, state, delta rule | 30.6 |
| 推理引擎、接入层 | inference engine, API server | 31.1 |
| 锁页内存 | pinned (page-locked) memory | 31.2 |
| 写入槽号 | slot mapping | 31.3 |
| 抢占 | preemption | 31.4 |
| 录制、异步调度 | graph capture, asynchronous (zero-overhead) scheduling | 31.5 |
| 首词元时间、词元间隔、有效吞吐 | time to first token (TTFT), time per output token (TPOT), goodput | 31.7 |
| 引用计数、写时复制、并行抽样 | reference count, copy-on-write, parallel sampling | 32.1 |
| 前缀缓存、链式哈希、基数树 | prefix caching, chained hash, radix tree | 32.2 |
| 持平带宽 | break-even bandwidth | 32.3 |
| 快照、近似重建 | snapshot, bounded replay | 32.5 |
| 宽专家并行 | wide expert parallelism (wide EP) | 33.2 |
| 冗余专家、列表调度、LPT | redundant experts (EPLB), list scheduling, longest processing time first | 33.3 |
| 置信头 | confidence head | 33.5 |

## 第七部分　模型调用与智能体编排

| 中文 | 英文 | 首见 |
| --- | --- | --- |
| 上下文、调用 | context, call | 34.2 |
| 上下文预算 | context budget | 34.4 |
| 提示、任务规格、少样本提示 | prompt, task specification, few-shot prompting | 34.5 |
| 结构化输出、模式 | structured output, schema | 34.6 |
| 工具调用 | tool calling | 35.1 |
| 模型上下文协议 | Model Context Protocol (MCP) | 35.1 |
| 检索增强生成、向量检索、提示注入 | retrieval-augmented generation (RAG), vector retrieval, prompt injection | 35.3 |
| 任务状态、长期记忆 | task state, long-term memory | 35.4 |
| 技能 | skill | 35.5 |
| 智能体、编排、规划者、工作者、工作流 | agent, orchestration, planner, worker, workflow | 36.1 |
| 子任务约定、任务依赖图、关键路径 | sub-task contract, task dependency graph, critical path | 36.2 |
| 任务检查点、事件记录 | task checkpoint, event log | 37.2 |
| 幂等、去重标识 | idempotent, deduplication key | 37.3 |
| 带来源的工作结论、黑板式协作 | provenance-carrying conclusion, blackboard architecture | 37.4 |
| 检查时与使用时不一致 | time-of-check to time-of-use (TOCTOU) | 37.5 |
| 可验证的任务成功 | verifiable task success | 37.6 |
| 资产图、探索图 | asset graph, exploration graph | 38.3 |

## 读代码时要区分的层次

| 对象 | 意义 | 首见或复查位置 |
| --- | --- | --- |
| 指令集体系结构 | instruction set architecture (ISA)：规定可见状态、操作和执行约束 | 定义 23.3 |
| PTX、目标机器指令 | virtual ISA、target machine instructions (SASS)：公开虚拟操作与目标实现 | 23.2、B.5 节 |
| grid 坐标、块坐标、元素偏移 | program index、block index、element offset：单位不同的三次定位 | 16.1、25.1 节 |
| 数组布局、执行者到数据映射 | layout、ownership mapping：物理位置与谁负责该元素 | 8.1、23.1 节 |
| 发起、传输完成、消费完成 | issue、transfer completion、consumer completion：三个不同事件 | 9.1、19.1、24.4 节 |
| 有用载荷、请求流量、HBM 流量 | useful payload、request traffic、HBM traffic：缓存或粒度造成差异 | 8.7、26.1 节 |
| 逻辑大小、物理预算 | logical size、physical footprint：布局填充与同时活跃对象分别计入 | 16.2、18.2 节 |
