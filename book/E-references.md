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
- ayaka14732. *Pallas TPU Kernel 开发教程*，GitHub，2026.（TPU v4 的实测数据与静态分析的方法）
- ayaka14732. *tpuasm*：TPU 指令包的汇编器与反汇编器，GitHub，2026.

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
