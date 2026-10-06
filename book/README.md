# 加速器编程：TPU、GPU、CUDA 与 JAX

本书写给有数学训练、懂一点计算机体系结构、想尽快学会整套加速器编程体系的读者。全书自底向上：从逻辑门和加法器出发，经过乘法器、存储、流水线、矩阵单元，搭起一颗芯片和一组芯片；再讲在这样的机器上组织计算的共同结构；最后分成 TPU 和 GPU 两条路线，讲各自的结构、编程模型和典型 kernel。

每一层都写成一个模型：有什么状态、能做什么操作、各要付出多少代价。每章末尾的"接口小结"列出上层会用到的结论，初读时可以只读接口、跳过实现，像引用定理一样往上走。

先读[前言](00-preface.md)，其中给出阅读路线、记号和习题体例。

## 阅读路线

- **TPU 路线**：第一部分 → 第二部分 → 第三部分 → 第四部分。
- **GPU 路线**：第一部分 → 第二部分 → 第五部分。第三部分（JAX）选读，只在第 24 章讲 Pallas GPU 的一节用到。
- 第四部分和第五部分互不依赖。其中偶尔出现的"注（对照）"可以跳过。

## 目录

[前言](00-preface.md)

### 第一部分　硬件：从逻辑门到多芯片

1. [逻辑门与加法器](01-circuits-adders.md)：逻辑门与组合电路；规模、深度与代价模型；全加器与行波进位；寄存器与时钟周期；进位函数与结合运算；前缀问题与并行前缀网络；超前进位加法器；进位保留加法；工作量、深度与 Brent 定理。
2. [乘法器与浮点运算](02-multipliers-floats.md)：部分积与 3:2 压缩；浮点数的表示与舍入；浮点运算单元；求和的误差；机器学习的数值格式；低精度相乘、高精度累加；块缩放；用低精度拼出高精度；随机舍入。
3. [存储](03-memory.md)：寄存器堆与译码器；SRAM 的阵列与访问距离；分体与存储体冲突；DRAM 与 HBM、访问粒度；按距离计的能耗；存储层次、缓存与便签存储。
4. [流水线、并行与控制](04-pipelines-control.md)：流水线与多个累加器；Little 定律；SIMD 摊薄控制；跨通道的数据交换；DMA 引擎与完成计数；静态调度与解耦访存；动态调度与多线程。
5. [矩阵单元](05-matrix-units.md)：2×2 阵列的逐拍计算；一致递推的时空映射与三种驻留方式；权重驻留阵列的时间；固定形状的 MMA 与补零；低精度、块缩放与 2:4 稀疏。
6. [一颗芯片：屋顶线与数据复用](06-chip-model.md)：抽象机器、屋顶线、传输的 α–β 模型、矩阵乘法的分块与 I/O 下界、多级分块。
7. [多颗芯片：互联与集合通信](07-interconnect.md)：环、环面与交换网络；集合操作的代数；环形算法与带宽下界；多维环面上的集合通信。

### 第二部分　计算的共同结构

8. [布局](08-layouts.md)：布局即函数、分块与打包、存储体冲突、XOR 交错、$\mathbb{F}_2$ 上的线性布局。
9. [异步程序与软件流水线](09-async.md)：先行发生关系、信号量、多缓冲、软件流水线、内存一致性。
10. [归约、扫描与在线算法](10-reductions.md)：在线 softmax、Welford、top-k、线性递推、可复现的归约。
11. [Transformer 的算术](11-transformer.md)：各算子的 FLOPs 与字节、训练、prefill 与 decode、KV cache、MoE、量化。
12. [并行策略](12-parallelism.md)：分片矩阵乘法的代数；DP、FSDP、TP、SP/CP、PP、EP；通信与计算重叠；估算一步训练和一步解码。

### 第三部分　JAX

13. [JAX 的计算模型](13-jax.md)：追踪与 jaxpr、jit、自动微分、vmap、控制流、随机数、精度。
14. [JAX 的分布式编程](14-jax-sharding.md)：mesh 与 PartitionSpec、Explicit/Auto/Manual 三种模式、shard_map 与集合原语、内存。

### 第四部分　TPU

15. [TPU 的结构](15-tpu-architecture.md)：TensorCore（标量单元、向量单元、MXU）、存储、DMA、SparseCore、ICI，v4 到第八代。
16. [Pallas TPU 编程](16-pallas-tpu.md)：grid 与下标映射、BlockSpec 与自动流水线、内存空间、手动 DMA、标量预取、布局约束。
17. [TPU kernel：矩阵乘法与访存受限算子](17-tpu-matmul.md)：分块矩阵乘法、量化、结尾融合、归一化与 softmax、分组矩阵乘法。
18. [TPU kernel：注意力](18-tpu-attention.md)：flash attention、块稀疏掩码、分页 KV 的解码注意力、反向传播。
19. [TPU 多芯片编程](19-tpu-multichip.md)：远程 DMA、在 Pallas 中写集合通信、集合矩阵乘法、all-to-all。
20. [TPU 性能的静态分析](20-tpu-static-analysis.md)：指令包与发射槽、由清单推算周期、性能剖析工具的读法。

### 第五部分　GPU

21. [GPU 的结构](21-gpu-architecture.md)：SM、SIMT、存储层次、延迟隐藏、Tensor Core 与 TMA 的演变、互联，Ampere 到 Rubin。
22. [CUDA 编程：SIMT 模型](22-cuda-simt.md)：线程层级、内存空间、同步、warp 级原语、归约与扫描、分块矩阵乘法、占用率。
23. [CUDA 编程：现代 GPU 的 kernel 结构](23-cuda-modern.md)：异步拷贝与 mbarrier、TMA、wgmma、warp 专门化、tcgen05 与 TMEM、持久化 kernel、CuTe。
24. [GPU 上的 tile 语言](24-gpu-tile-languages.md)：tile 模型、Triton、cuTile、Pallas GPU、如何选择抽象层级。
25. [GPU kernel：矩阵乘法与访存受限算子](25-gpu-matmul.md)：GEMM 的逐级优化、FP8/FP4 块缩放 GEMM、分组 GEMM、归一化与 softmax。
26. [GPU kernel：注意力](26-gpu-attention.md)：FlashAttention 2/3/4 的结构、解码注意力、分页 KV。
27. [多 GPU 编程](27-multi-gpu.md)：NCCL 与拓扑、对称内存与设备端通信、通信与计算重叠、MoE 的 all-to-all。

### 附录

- [A　硬件参数表](A-hardware.md)
- [B　概念与 API 对照表](B-api.md)
- [C　与 AI 协作：规格与审查](C-ai-workflow.md)
- [D　术语对照](D-glossary.md)
- [E　参考文献](E-references.md)
