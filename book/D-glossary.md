# 附录 D　术语对照

按首次详细介绍的章节排列。硬件单元、指令与 API 的名字在正文中保留英文。

## 第一部分　硬件

| 中文 | 英文 | 首见 |
| --- | --- | --- |
| 组合电路、规模、深度 | combinational circuit, size, depth | 1.1 |
| 寄存器、同步电路、关键路径 | register, synchronous circuit, critical path | 1.2 |
| 全加器、行波进位加法器 | full adder, ripple-carry adder | 1.3 |
| 进位产生、传递、吸收 | generate, propagate, kill | 1.4 |
| 前缀问题、归约 | prefix problem, reduction | 1.4 |
| 超前进位加法器 | carry-lookahead adder | 1.5 |
| 进位保留加法、3:2 压缩 | carry-save addition, 3:2 compressor | 1.6 |
| 工作量、深度 | work, depth (span) | 1.7 |
| 部分积 | partial product | 2.1 |
| 尾数、指数、次正规数 | significand, exponent, subnormal | 2.2 |
| 单位舍入误差 | unit roundoff | 2.2 |
| 融合乘加 | fused multiply-add (FMA) | 2.3 |
| 混合精度 | mixed precision | 2.3 |
| 块缩放 | block scaling (microscaling) | 2.5 |
| 随机舍入 | stochastic rounding | 2.8 |
| 寄存器堆、端口 | register file, port | 3.1 |
| 存储体、冲突度 | bank, conflict degree | 3.2 |
| 行缓冲、突发 | row buffer, burst | 3.3 |
| 高带宽存储 | high-bandwidth memory (HBM) | 3.3 |
| 缓存、便签存储 | cache, scratchpad | 3.5 |
| 延迟、发射间隔 | latency, issue interval | 4.1 |
| 通道、掩码 | lane, mask | 4.3 |
| 超长指令字、指令包、槽 | VLIW, bundle, slot | 4.4 |
| 记分板 | scoreboard | 4.4 |
| 完成计数器（同步标志、信号量） | completion counter (sync flag, semaphore) | 4.5 |
| 一致递推、时空映射 | uniform recurrence, space-time mapping | 5.2 |
| 输出驻留、权重驻留、输入驻留 | output / weight / input stationary | 5.2 |
| 脉动阵列 | systolic array | 5.3 |
| 结构化稀疏 | structured sparsity (2:4) | 5.5 |
| 算术强度、屋顶线、拐点 | arithmetic intensity, roofline, ridge point | 6.2 |
| 融合 | fusion | 6.2 |
| 波次量化 | wave quantization | 6.6 |
| 链路、拓扑、环面、二分带宽 | link, topology, torus, bisection bandwidth | 7.1 |
| 集合操作 | collective operation | 7.2 |

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
| 可复现 | reproducible | 10.6 |
| 词元、头 | token, head | 11.1 |
| 分组查询注意力、多头潜在注意力 | GQA, MLA | 11.1、11.4 |
| 模型 FLOP 利用率 | model FLOPs utilization (MFU) | 11.3 |
| 重计算 | rematerialization (activation checkpointing) | 11.3 |
| 预填充、解码 | prefill, decode | 11.4 |
| 混合专家、路由器、分组矩阵乘法 | mixture of experts, router, grouped matmul | 11.5 |
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
| 维度语义 | dimension semantics | 16.5 |
| 标量预取 | scalar prefetch | 16.7 |
| 解释器 | interpreter (interpret mode) | 16.8 |
| 远程 DMA | remote DMA | 19.1 |
| 向量发射队列 | vector issue FIFO (VIF) | 20.2 |

## 第五部分　GPU

| 中文 | 英文 | 首见 |
| --- | --- | --- |
| 流式多处理器 | streaming multiprocessor (SM) | 21.1 |
| 线程、warp、线程块 | thread, warp, thread block (CTA) | 21.2 |
| 共享内存、溢出 | shared memory, spill | 21.3 |
| 占用率 | occupancy | 21.4 |
| warpgroup、张量存储 | warpgroup, tensor memory (TMEM) | 21.5 |
| 张量搬运引擎、张量描述符 | TMA, tensor map (descriptor) | 21.6 |
| 线程块簇、分布式共享内存 | thread block cluster, distributed shared memory | 21.7 |
| 网格步长循环 | grid-stride loop | 22.1 |
| 洗牌 | shuffle | 22.3 |
| 流、事件、图 | stream, event, CUDA graph | 22.8 |
| 生产者、消费者、warp 专门化 | producer, consumer, warp specialization | 23.4 |
| 乒乓 | ping-pong | 23.4 |
| CTA 对 | CTA pair | 23.5 |
| 持久化 kernel | persistent kernel | 23.6 |
| 光栅化 | rasterization (tile order) | 23.6 |
| tile 程序 | tile program | 24.1 |
| 结尾 | epilogue | 25.4 |
| 分页 KV cache、页表 | paged KV cache, block table | 26.5 |
| 对称内存 | symmetric memory | 27.2 |
