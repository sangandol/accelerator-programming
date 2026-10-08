# 附录 E　参考文献

按主题排列。网页资料的内容会更新，引用时以 2026 年 10 月的版本为准。

## 教材与综述

- D. Harris, S. Harris. *Digital Design and Computer Architecture*. Morgan Kaufmann.（第 1–3 章的背景）
- J. Hennessy, D. Patterson. *Computer Architecture: A Quantitative Approach*, 6th ed. Morgan Kaufmann, 2017.
- N. Higham. *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM, 2002.（第 2 章）
- V. Sze, Y.-H. Chen, T.-J. Yang, J. Emer. *Efficient Processing of Deep Neural Networks*. Morgan & Claypool, 2020.
- T. Aamodt, W. Fung, T. Rogers. *General-Purpose Graphics Processor Architecture*. Morgan & Claypool, 2018.
- W. Hwu, D. Kirk, I. El Hajj. *Programming Massively Parallel Processors*, 4th ed. Morgan Kaufmann, 2022.
- J. Austin 等. *How to Scale Your Model*. JAX 团队，2025.（第 6、11、12 章的思路与 TPU、GPU 的参数）

## 电路与算术（第 1、2 章）

- J. Sklansky. Conditional-sum addition logic. *IRE Trans. Electronic Computers*, 1960.
- P. Kogge, H. Stone. A parallel algorithm for the efficient solution of a general class of recurrence equations. *IEEE Trans. Computers*, 1973.
- R. Brent, H. T. Kung. A regular layout for parallel adders. *IEEE Trans. Computers*, 1982.
- M. Snir. Depth-size trade-offs for parallel prefix computation. *J. Algorithms*, 1986.
- C. Wallace. A suggestion for a fast multiplier. *IEEE Trans. Electronic Computers*, 1964.
- R. Brent. The parallel evaluation of general arithmetic expressions. *J. ACM*, 1974.
- Open Compute Project. *OCP Microscaling Formats (MX) Specification*, v1.0, 2023.
- K. Ozaki, T. Ogita, S. Oishi, S. Rump. Error-free transformations of matrix multiplication by using fast routines of matrix multiplication and its applications. *Numerical Algorithms*, 2012.

## 存储、流水线与矩阵单元（第 3–5 章）

- M. Horowitz. Computing's energy problem (and what we can do about it). *ISSCC*, 2014.
- J. Little. A proof for the queuing formula $L = \lambda W$. *Operations Research*, 1961.
- V. Beneš. *Mathematical Theory of Connecting Networks and Telephone Traffic*. Academic Press, 1965.
- R. Karp, R. Miller, S. Winograd. The organization of computations for uniform recurrence equations. *J. ACM*, 1967.
- H. T. Kung, C. Leiserson. Systolic arrays (for VLSI). 1978.
- H. T. Kung. Why systolic architectures? *IEEE Computer*, 1982.

## 芯片模型与分块（第 6 章）

- S. Williams, A. Waterman, D. Patterson. Roofline: an insightful visual performance model for multicore architectures. *CACM*, 2009.
- J.-W. Hong, H. T. Kung. I/O complexity: the red-blue pebble game. *STOC*, 1981.
- L. Loomis, H. Whitney. An inequality related to the isoperimetric inequality. *Bull. AMS*, 1949.
- D. Irony, S. Toledo, A. Tiskin. Communication lower bounds for distributed-memory matrix multiplication. *JPDC*, 2004.
- T. Smith, B. Lowery, J. Langou, R. van de Geijn. A tight I/O lower bound for matrix multiplication. arXiv:1702.02017.

## 通信与并行（第 7、12 章）

- R. Thakur, R. Rabenseifner, W. Gropp. Optimization of collective communication operations in MPICH. *IJHPCA*, 2005.
- P. Patarasuk, X. Yuan. Bandwidth optimal all-reduce algorithms for clusters of workstations. *JPDC*, 2009.
- M. Shoeybi 等. Megatron-LM: training multi-billion parameter language models using model parallelism. 2019.
- S. Rajbhandari 等. ZeRO: memory optimizations toward training trillion parameter models. *SC*, 2020.
- Y. Huang 等. GPipe: efficient training of giant neural networks using pipeline parallelism. *NeurIPS*, 2019.
- D. Lepikhin 等. GShard: scaling giant models with conditional computation and automatic sharding. 2020.
- H. Liu 等. Ring attention with blockwise transformers for near-infinite context. 2023.

## 布局、异步与归约（第 8–10 章）

- K. Zhou 等. Linear layouts: robust code generation of efficient tensor computation using $\mathbb{F}_2$. *ASPLOS*, 2026.
- L. Lamport. Time, clocks, and the ordering of events in a distributed system. *CACM*, 1978.
- G. Blelloch. Prefix sums and their applications. 1990.
- M. Milakov, N. Gimelshein. Online normalizer calculation for softmax. 2018.
- B. Welford. Note on a method for calculating corrected sums of squares and products. *Technometrics*, 1962.
- T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample variance: analysis and recommendations. *The American Statistician*, 1983.

## Transformer 与推理（第 11 章）

- A. Vaswani 等. Attention is all you need. *NeurIPS*, 2017.
- N. Shazeer. Fast transformer decoding: one write-head is all you need. 2019.
- J. Ainslie 等. GQA: training generalized multi-query transformer models from multi-head checkpoints. 2023.
- DeepSeek-AI. DeepSeek-V2（多头潜在注意力）, 2024；DeepSeek-V3 技术报告（FP8 训练）, 2024.
- W. Fedus, B. Zoph, N. Shazeer. Switch transformers. 2021.
- W. Kwon 等. Efficient memory management for large language model serving with PagedAttention. *SOSP*, 2023.

## JAX 与 Pallas（第 13、14、16–20 章）

- R. Frostig, M. Johnson, C. Leary. Compiling machine learning programs via high-level tracing. *SysML*, 2018.
- JAX 文档：*Distributed arrays and automatic parallelization*（mesh、Explicit/Auto/Manual）、*Pallas* 与 *Pallas TPU*、*Mosaic GPU*。
- ayaka14732. [*Pallas TPU Kernel 开发教程*](https://github.com/ayaka14732/pallas-tpu-tutorial)，GitHub，2026.（来源报告的 TPU v4 硬件观察与静态分析）
- ayaka14732. [*tpuasm*](https://github.com/ayaka14732/tpuasm)：TPU 指令包的汇编器与反汇编器，GitHub，2026.

## TPU（第 15 章）

- N. Jouppi 等. In-datacenter performance analysis of a tensor processing unit. *ISCA*, 2017.
- T. Norrie 等. The design process for Google's training chips: TPUv2 and TPUv3. *IEEE Micro*, 2021.
- N. Jouppi 等. Ten lessons from three generations shaped Google's TPUv4i. *ISCA*, 2021.
- N. Jouppi 等. TPU v4: an optically reconfigurable supercomputer for machine learning with hardware support for embeddings. *ISCA*, 2023.
- N. Jouppi, S. Lakshmanamurthy, C. Young, D. Patterson. Google's training supercomputers from TPU v2 to Ironwood. *IEEE Micro*, 2026.
- Google Cloud TPU 文档（各代系统架构）；Google Cloud 博客：Ironwood 与 TPU 8t/8i 的技术介绍，2025–2026.

## GPU 与 CUDA（第 21–27 章）

- NVIDIA. *CUDA C++ Programming Guide*（含 tile kernel 一章）；*PTX ISA*；*CUTLASS* 与 *CuTe* 文档；*cuTile Python* 文档。
- NVIDIA 技术博客：Hopper、Blackwell、Rubin 架构介绍。
- P. Tillet, H. T. Kung, D. Cox. Triton: an intermediate language and compiler for tiled neural network computations. *MAPL*, 2019.
- M. Osama 等. Stream-K: work-centric parallel decomposition for dense matrix-matrix multiplication on the GPU. *PPoPP*, 2023.
- T. Dao 等. FlashAttention: fast and memory-efficient exact attention with IO-awareness. *NeurIPS*, 2022.
- T. Dao. FlashAttention-2: faster attention with better parallelism and work partitioning. 2023.
- J. Shah 等. FlashAttention-3: fast and accurate attention with asynchrony and low-precision. 2024.
- FlashAttention 的 Blackwell 版本（用 CuTe DSL 写成）的公开代码与介绍，2025–2026.
- NVIDIA. *NCCL* 与 *NVSHMEM* 文档。

## 本次实例与指令衔接使用的在线资料

以下链接于 **2026-10-07** 核对。型号表给的是特定产品或后端参考参数；论文和 ISA 给的是算法或操作语义；它们都不是本书作者的真机测量。各章的微型算例和推导为本书按这些接口独立构造。

| 来源 | 本书使用的内容 |
| --- | --- |
| Google Cloud，[TPU v5e](https://docs.cloud.google.com/tpu/docs/v5e) | TensorCore、四 MXU 与产品口径；15.1 节、附录 A |
| JAX，[TPU Hardware Reference](https://docs.jax.dev/en/latest/pallas/tpu/hardware.html) | 按 TensorCore 的后端参数、v5e 的 VMEM 与带宽参考值；附录 A |
| Google Cloud，[TPU 8t/8i technical deep dive](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive/) | 第八代的存储、CAE 与 Boardfly；15.9 节、附录 A |
| JAX，[Writing TPU kernels with Pallas](https://docs.jax.dev/en/latest/pallas/tpu/details.html)、[TPU Pipelining](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html) | 块到寄存器、布局与自动流水线的接口；16.1–16.8 节 |
| JAX，[Manual parallelism with shard_map](https://docs.jax.dev/en/latest/notebooks/shard_map.html) | 全局与本地视角、集合原语与复制声明；第 14 章 |
| NVIDIA，[H100 产品表](https://www.nvidia.com/en-us/data-center/h100/) | SXM 带宽、稀疏峰值标注与接口口径；附录 A |
| NVIDIA，[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) | 虚拟 ISA、MMA 形态、异步提交等待与目标要求；21.5、22.2、第 23 章、附录 B |
| NVIDIA，[Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)、[Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) | SM/块资源上限、代际编程接口；第 21、23 章 |
| NVIDIA，[Rubin GPU 架构介绍](https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/) | 新一代 HBM、互联与部件组织；21.9 节、附录 A |
| NVIDIA CUTLASS，[CuTe Layouts](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html) | 形状步长与嵌套坐标，不混同物理连续分块；23.7 节 |
| Triton，[Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)；NVIDIA，[cuTile Python](https://docs.nvidia.com/cuda/cutile-python/) | 指针块、tile 坐标与编译器分工；第 24 章 |
| Shah 等，[FlashAttention-3](https://arxiv.org/abs/2407.08608)；[FlashAttention-4](https://arxiv.org/abs/2603.05451) | 异步流水线与不均衡硬件扩展的算法设计；26.3、26.4 节 |
| NVIDIA，[NCCL CUDA Stream Semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html)、[Using NVSHMEM](https://docs.nvidia.com/nvshmem/api/latest/using.html) | 提交与完成的区别、单边通信次序与可见性；第 27 章、附录 B |

## Pallas 教程与 tpuasm 的覆盖核对（2026-10-08）

这次按教程的四章 **36 个主题小节**与 tpuasm 的公开格式、设计及兼容性文档核对。教程固定为 `b4fc5b11713aa34eef1fa824cadf994c9d67edb0`，tpuasm 固定为 `a6e3d927deb1dc5a906574e6b6bbfc8bc175d30d`；后续更新应重新核对，不能把 main 的新内容冒充这个快照。

| 原来欠缺或需修正的内容 | 固定快照的出处 | 补入位置 |
| --- | --- | --- |
| 数据移动原语与接口限制 | [DMA](https://github.com/ayaka14732/pallas-tpu-tutorial/blob/b4fc5b11713aa34eef1fa824cadf994c9d67edb0/chapter01/03_local_dma/README.md)、[gather/scatter](https://github.com/ayaka14732/pallas-tpu-tutorial/blob/b4fc5b11713aa34eef1fa824cadf994c9d67edb0/chapter01/08_gather_scatter/README.md)、[扫描](https://github.com/ayaka14732/pallas-tpu-tutorial/blob/b4fc5b11713aa34eef1fa824cadf994c9d67edb0/chapter01/12_prefix_scan/README.md)、[top-k](https://github.com/ayaka14732/pallas-tpu-tutorial/blob/b4fc5b11713aa34eef1fa824cadf994c9d67edb0/chapter01/14_top_k/README.md) | 16.9 节与附录 B 的边界表 |
| CMEM 的直接读、共享与远端路由 | [CMEM](https://github.com/ayaka14732/pallas-tpu-tutorial/blob/b4fc5b11713aa34eef1fa824cadf994c9d67edb0/chapter02/03_cmem/README.md)、[远端 CMEM](https://github.com/ayaka14732/pallas-tpu-tutorial/blob/b4fc5b11713aa34eef1fa824cadf994c9d67edb0/chapter02/06_remote_cmem/README.md) | 15.5、19.8 节 |
| 主机内存与持久循环协议 | [主机内存](https://github.com/ayaka14732/pallas-tpu-tutorial/blob/b4fc5b11713aa34eef1fa824cadf994c9d67edb0/chapter02/08_host_memory/README.md) | 19.8 节；私有接口限制留在附录 B |
| 共享带宽下的发起顺序、内部 dot 与写回尾巴 | [大矩阵乘法](https://github.com/ayaka14732/pallas-tpu-tutorial/blob/b4fc5b11713aa34eef1fa824cadf994c9d67edb0/chapter02/09_large_matmul/README.md) | 17.7 节 |
| 时间边界、sfence 与两种时钟 | [五种时间](https://github.com/ayaka14732/pallas-tpu-tutorial/blob/b4fc5b11713aa34eef1fa824cadf994c9d67edb0/chapter03/01_five_times/README.md)、[sfence/VIF](https://github.com/ayaka14732/pallas-tpu-tutorial/blob/b4fc5b11713aa34eef1fa824cadf994c9d67edb0/chapter03/04_sfence_vif/README.md)、[GTC](https://github.com/ayaka14732/pallas-tpu-tutorial/blob/b4fc5b11713aa34eef1fa824cadf994c9d67edb0/chapter03/06_gtc/README.md) | 20.2、20.7 节 |
| XProf 跟踪事件确有 GTC 时间戳；不是全部插值 | [XProf/vtrace](https://github.com/ayaka14732/pallas-tpu-tutorial/blob/b4fc5b11713aa34eef1fa824cadf994c9d67edb0/chapter03/07_xprof_vtrace/README.md) | 修正 20.4 节与接口小结 |
| 硬件随机状态、key、计数器、分布、代价 | [随机数全章](https://github.com/ayaka14732/pallas-tpu-tutorial/tree/b4fc5b11713aa34eef1fa824cadf994c9d67edb0/chapter04) | 13.7、16.10 节与附录 B |
| 精确编码与 TC/BCS/TEC 目标差别 | [v4 TC](https://github.com/ayaka14732/tpuasm/blob/a6e3d927deb1dc5a906574e6b6bbfc8bc175d30d/docs/references/tpu_v4_tc.md)、[v4 BCS](https://github.com/ayaka14732/tpuasm/blob/a6e3d927deb1dc5a906574e6b6bbfc8bc175d30d/docs/references/tpu_v4_bcs.md)、[v6e TC](https://github.com/ayaka14732/tpuasm/blob/a6e3d927deb1dc5a906574e6b6bbfc8bc175d30d/docs/references/tpu_v6e_tc.md)、[v6e TEC](https://github.com/ayaka14732/tpuasm/blob/a6e3d927deb1dc5a906574e6b6bbfc8bc175d30d/docs/references/tpu_v6e_tec.md) | 20.6 节与附录 B |
| 来源映射、回灌、插入与兼容性 | [来源映射](https://github.com/ayaka14732/tpuasm/blob/a6e3d927deb1dc5a906574e6b6bbfc8bc175d30d/docs/design/tc_source_mapping.md)、[回灌](https://github.com/ayaka14732/tpuasm/blob/a6e3d927deb1dc5a906574e6b6bbfc8bc175d30d/docs/design/executable_replacement.md)、[兼容性](https://github.com/ayaka14732/tpuasm/blob/a6e3d927deb1dc5a906574e6b6bbfc8bc175d30d/docs/compatibility.md) | 20.6 节与附录 B |
| 编码往返与设备语义是不同证据 | [v6e 执行语义核对](https://github.com/ayaka14732/tpuasm/blob/a6e3d927deb1dc5a906574e6b6bbfc8bc175d30d/docs/design/tpu_v6e_execution.md) | 20.6、20.7 节的证据与计时边界；附录 B |

本书按接口和代价模型重新构造证明、算例与图，没有复制教程文字、代码或图。安装、实验载体、基准输出、工具内部 ABI 偏移和全助记符索引没有逐条搬进正文：它们不参与本书的线性推导，需使用工具时从上述固定来源查阅。接口失败与逆向得到的参数均注明来源环境，不列为跨代保证。
