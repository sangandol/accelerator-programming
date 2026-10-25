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
- IEEE. *IEEE Standard for Floating-Point Arithmetic*, IEEE Std 754-2019, §5.10 totalOrder.（2.11 节的全序）
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
- K. Batcher. Sorting networks and their applications. *AFIPS Spring Joint Computer Conference*, 1968.（10.4 节与第 21 章的比较网络）
- D. Knuth. *The Art of Computer Programming*, Vol. 3: *Sorting and Searching*, 2nd ed. Addison-Wesley, 1998.（败者树与多路归并）

## Transformer 与推理（第 11 章）

- A. Vaswani 等. Attention is all you need. *NeurIPS*, 2017.
- N. Shazeer. Fast transformer decoding: one write-head is all you need. 2019.
- J. Ainslie 等. GQA: training generalized multi-query transformer models from multi-head checkpoints. 2023.
- DeepSeek-AI. DeepSeek-V2（多头潜在注意力）, 2024；DeepSeek-V3 技术报告（FP8 训练）, 2024.
- W. Fedus, B. Zoph, N. Shazeer. Switch transformers. 2021.
- W. Kwon 等. Efficient memory management for large language model serving with PagedAttention. *SOSP*, 2023.

## JAX 与 Pallas（第 13、14、16–21 章）

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

## 近期论文与报告

这些论文在正文中被详细讲到或引用了其中的数字，按章排列。面试和读论文时，它们是"现在业界在做什么"的好样本；附录 F 给出了阅读顺序和自测题。

- M. Zhong. Bitwise-correct and faster top-$k$ on TPU v4. 2026. 代码、原始结果与中文版：[github.com/ayaka14732/tpu-v4-top-k](https://github.com/ayaka14732/tpu-v4-top-k)。（2.11 节、10.4 节、15.3 节、第 20、21 章）
- A. Chowdhery 等. PaLM: scaling language modeling with Pathways. arXiv:2204.02311, 2022.（MFU 的定义与 TPU v4 上的数字，11.5 节）
- R. Pope 等. Efficiently scaling Transformer inference. *MLSys*, 2023（arXiv:2211.05102）。（prefill 与 decode 的分片，11.8 节）
- Z. Jiang 等. MegaScale: scaling large language model training to more than 10,000 GPUs. *NSDI*, 2024.（11.5 节）
- Llama Team, AI @ Meta. The Llama 3 herd of models. arXiv:2407.21783, 2024.（MFU、四维并行与故障统计，11.5 节、第 12 章）
- DeepSeek-AI. DeepSeek-V3 technical report. arXiv:2412.19437, 2024.（DualPipe，第 12 章；FP8 训练与累加精度，26.2 节）
- NVIDIA. Pretraining large language models with NVFP4. arXiv:2509.25149, 2025.（26.2 节）
- SonicMoE（Guo、Mishra、Cheng、Stoica、Dao，*ICLR*, 2026）：MoE 的高效 GPU 实现，含路由 top-$k$ 的专门 kernel。（例 26.8）
- A. Ouyang 等. KernelBench: can LLMs write efficient GPU kernels? *ICML*, 2025.（注 25.9、C.9 节）
- Sakana AI. The AI CUDA Engineer（2025 年 2 月）及其后对评测漏洞的更正；EvoEngineer 等后续复现研究，2025.（注 25.9、C.9 节）
- METR. Recent frontier models are reward hacking. 2025 年 6 月.（C.9 节）

以下是 2026 年 10 月补入的论文，内容在正文中有完整的讲解（附录 F.3 有导读）。数字据各论文的 arXiv 版本，2026-10-20 核对。

- H. He, Thinking Machines Lab. Defeating nondeterminism in LLM inference. *Thinking Machines Lab: Connectionism*, 2025 年 9 月. 代码：[batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops)。（10.6 节、11.12 节）
- P. Qi, Z. Liu 等. Defeating the training-inference mismatch via FP16. arXiv:2510.26788, 2025.（注 11.33）
- Y. Leviathan, M. Kalman, Y. Matias. Fast inference from Transformers via speculative decoding. *ICML*, 2023（arXiv:2211.17192）.（11.11 节）
- C. Chen 等. Accelerating large language model decoding with speculative sampling. arXiv:2302.01318, 2023.（11.11 节）
- Y. Zhong 等. DistServe: disaggregating prefill and decoding for goodput-optimized large language model serving. *OSDI*, 2024（arXiv:2401.09670）.（11.12 节）
- R. Qin 等. Mooncake: a KVCache-centric disaggregated architecture for LLM serving. *FAST*, 2025（arXiv:2407.00079）.（11.12 节）
- C. Zhao 等（DeepSeek-AI）. Insights into DeepSeek-V3: scaling challenges and reflections on hardware for AI architectures. *ISCA*, 2025（arXiv:2505.09343）.（11.8 节现状框、12.7 节、例 12.15）
- DeepSeek-AI. DeepSeek-V4: towards highly efficient million-token context intelligence. arXiv:2606.19348, 2026；DeepSeek-V4.1-Flash, arXiv:2609.19969, 2026.（11.8、11.11、11.12 节，命题 12.16，24.8、28.4 节，注 27.12）
- J. Jiang, Y. Chen, B. Hechtman, F. Zhang, Y. Mu. Ragged paged attention: a high-performance and flexible LLM inference kernel for TPU. arXiv:2604.15464, 2026. 代码在 vLLM 的 tpu-inference 仓库。（18.6 节）
- Z. Ye 等. FlashInfer: efficient and customizable attention engine for LLM inference serving. *MLSys*, 2025（arXiv:2501.01005）.（27.7 节）
- B. Spector, J. Juravsky 等. Look ma, no bubbles! Designing a low-latency megakernel for Llama-1B. Hazy Research 博客, 2025 年 5 月.（24.8 节）
- X. Cheng 等. MPK: a compiler and runtime for mega-kernelizing tensor programs. arXiv:2512.22219, 2025.（24.8 节）
- Chowdhary 等（AMD Research）. Fleet: hierarchical task-based abstraction for megakernels on multi-die GPUs. arXiv:2604.15379, 2026.（24.8 节）
- A. Novikov 等（Google DeepMind）. AlphaEvolve: a coding agent for scientific and algorithmic discovery. arXiv:2506.13131, 2025.（17.1 节、C.9 节）
- A. Tschand, C. Hong 等. JAXBench: benchmarking autonomous TPU kernel optimization. arXiv:2607.20466, 2026.（18.6 节、C.9 节）
- S. Wang, N. Cai 等. MaxKernel: agentic kernel generation for TPUs. arXiv:2609.04523, 2026.（18.6 节、C.9 节）
- Y. Zhang 等. KernelBench-Verified: do LLM-generated kernels actually beat PyTorch? arXiv:2607.16241, 2026.（注 25.9、C.9 节）

## GPU 与 CUDA（第 22–28 章）

- NVIDIA. *CUDA C++ Programming Guide*（含 tile kernel 一章）；*PTX ISA*；*CUTLASS* 与 *CuTe* 文档；*cuTile Python* 文档。
- NVIDIA 技术博客：Hopper、Blackwell、Rubin 架构介绍。
- P. Tillet, H. T. Kung, D. Cox. Triton: an intermediate language and compiler for tiled neural network computations. *MAPL*, 2019.
- M. Osama 等. Stream-K: work-centric parallel decomposition for dense matrix-matrix multiplication on the GPU. *PPoPP*, 2023.
- T. Dao 等. FlashAttention: fast and memory-efficient exact attention with IO-awareness. *NeurIPS*, 2022.
- T. Dao. FlashAttention-2: faster attention with better parallelism and work partitioning. 2023.
- J. Shah 等. FlashAttention-3: fast and accurate attention with asynchrony and low-precision. 2024.
- FlashAttention 的 Blackwell 版本（用 CuTe DSL 写成）的公开代码与介绍，2025–2026.
- NVIDIA. *NCCL* 与 *NVSHMEM* 文档。

## 大模型推理系统（第 29–33 章）

第六部分的主要实例来自 2026 年 10 月公开的两份一手材料，2026-10-23 核对：

- DeepSeek-AI. [*DeepInfer*](https://github.com/deepseek-ai/DeepInfer): high-performance, large-scale inference framework for DeepSeek models. GitHub，2026 年 10 月 20 日开源，Apache 2.0；其中的部署说明 `doc/inference_deployment_qa.md` 给出了 E/P/D 分离、专家负载均衡、负载均衡器、KV cache 存储、异步调度、结构化输出和投机解码的参数与实测数字。（第 31–33 章）
- DeepSeek-AI. DeepSeek-V4.1-Flash: pushing the limits of KV cache compression. 技术报告，2026（见上文"近期论文与报告"）；Hugging Face 上的 `config.json`、权重文件头与参考推理代码。（例 29.3、30.5 节、例 30.15、32.5 节、33.5 节）
- DeepSeek-AI. *deepseek-recipe*：提示编码与输出解析的 Rust 库，GitHub，2026.（29.2 节、31.1 节）
- DeepSeek-AI. *3FS*（Fire-Flyer File System），GitHub，2025.（32.3 节）

模型与词元：

- Hugging Face. *safetensors* 格式说明。（29.1 节）
- R. Sennrich, B. Haddow, A. Birch. Neural machine translation of rare words with subword units. *ACL*, 2016.（BPE）
- A. Radford 等. Language models are unsupervised multitask learners. OpenAI, 2019.（字节级 BPE）
- J. Su 等. RoFormer: enhanced Transformer with rotary position embedding. arXiv:2104.09864, 2021.
- B. Peng 等. YaRN: efficient context window extension of large language models. arXiv:2309.00071, 2023.
- A. Holtzman 等. The curious case of neural text degeneration. *ICLR*, 2020.（top-p）
- E. J. Gumbel. *Statistical Theory of Extreme Values and Some Practical Applications*. 1954；C. Maddison, D. Tarlow, T. Minka. A* sampling. *NeurIPS*, 2014.（Gumbel-max）
- B. Willard, R. Louf. Efficient guided generation for large language models. arXiv:2307.09702, 2023；Y. Dong 等. XGrammar: flexible and efficient structured generation engine for large language models. arXiv:2411.15100, 2024.（约束输出）

注意力的变体：

- DeepSeek-AI. DeepSeek LLM: scaling open-source language models with longtermism. arXiv:2401.02954, 2024.（67B 的结构，30.7 节）
- W. Brandon 等. Reducing Transformer key-value cache size with cross-layer attention. arXiv:2405.12981, 2024.
- Y. Sun 等. You only cache once: decoder-decoder architectures for language models. arXiv:2405.05254, 2024.（YOCO）
- G. Xiao 等. Efficient streaming language models with attention sinks. *ICLR*, 2024.
- Gemma Team. Gemma 3 technical report, 2025；OpenAI. gpt-oss model card, 2025.（30.3 节的现状框）
- DeepSeek-AI. DeepSeek-V3.2（稀疏注意力与索引器），2025.
- A. Katharopoulos 等. Transformers are RNNs: fast autoregressive Transformers with linear attention. *ICML*, 2020.
- T. Dao, A. Gu. Transformers are SSMs: generalized models and efficient algorithms through structured state space duality. *ICML*, 2024.（Mamba-2）
- S. Yang, J. Kautz, A. Hatamizadeh. Gated delta networks: improving Mamba2 with delta rule. *ICLR*, 2025.
- Qwen Team. Qwen3-Next, 2025；Kimi Team. Kimi Linear, 2025.（30.6 节的现状框）

引擎、缓存与部署：

- L. Zheng 等. SGLang: efficient execution of structured language model programs. *NeurIPS*, 2024.（RadixAttention，32.2 节）
- A. Agrawal 等. Taming throughput-latency tradeoff in LLM inference with Sarathi-Serve. *OSDI*, 2024.（分段 prefill）
- K. Zhu 等. NanoFlow: towards optimal large language model serving throughput. arXiv:2408.12757, 2024.（异步调度，31.5 节）
- vLLM 项目文档：自动前缀缓存、调度器、CUDA Graph 的批量档位，2026.
- L. Wang 等. Auxiliary-loss-free load balancing strategy for mixture-of-experts. arXiv:2408.15664, 2024.（33.3 节）
- R. L. Graham. Bounds on multiprocessing timing anomalies. *SIAM J. Applied Mathematics*, 1969.（定理 33.5 与 LPT 的界）
- Engram 与 DSpark：Cheng 等（2026），见 DeepSeek-V4.1-Flash 技术报告的参考文献。

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
| NVIDIA，[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) | 虚拟 ISA、MMA 形态、异步提交等待与目标要求；22.5、23.2 节，第 24 章，附录 B |
| NVIDIA，[Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)、[Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) | SM/块资源上限、代际编程接口；第 22、24 章 |
| NVIDIA，[Rubin GPU 架构介绍](https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/) | 新一代 HBM、互联与部件组织；22.9 节、附录 A |
| NVIDIA CUTLASS，[CuTe Layouts](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html) | 形状步长与嵌套坐标，不混同物理连续分块；24.7 节 |
| Triton，[Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)；NVIDIA，[cuTile Python](https://docs.nvidia.com/cuda/cutile-python/) | 指针块、tile 坐标与编译器分工；第 25 章 |
| Shah 等，[FlashAttention-3](https://arxiv.org/abs/2407.08608)；[FlashAttention-4](https://arxiv.org/abs/2603.05451) | 异步流水线与不均衡硬件扩展的算法设计；27.3、27.4 节 |
| NVIDIA，[NCCL CUDA Stream Semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html)、[Using NVSHMEM](https://docs.nvidia.com/nvshmem/api/latest/using.html) | 提交与完成的区别、单边通信次序与可见性；第 28 章、附录 B |

## Pallas 教程与 tpuasm 的覆盖核对（2026-10-08）

这次按教程的四章 **36 个主题小节**与 tpuasm 的公开格式、设计及兼容性文档核对。教程固定为 `b4fc5b11713aa34eef1fa824cadf994c9d67edb0`，tpuasm 固定为 `a6e3d927deb1dc5a906574e6b6bbfc8bc175d30d`；后续更新应重新核对，不能把 main 的新内容冒充这个快照。

| 原来欠缺或需修正的内容 | 固定快照的出处 | 补入位置 |
| --- | --- | --- |
| 数据移动原语与接口限制 | [DMA](https://github.com/ayaka14732/pallas-tpu-tutorial/blob/b4fc5b11713aa34eef1fa824cadf994c9d67edb0/chapter01/03_local_dma/README.md)、[gather/scatter](https://github.com/ayaka14732/pallas-tpu-tutorial/blob/b4fc5b11713aa34eef1fa824cadf994c9d67edb0/chapter01/08_gather_scatter/README.md)、[扫描](https://github.com/ayaka14732/pallas-tpu-tutorial/blob/b4fc5b11713aa34eef1fa824cadf994c9d67edb0/chapter01/12_prefix_scan/README.md)、[top-k](https://github.com/ayaka14732/pallas-tpu-tutorial/blob/b4fc5b11713aa34eef1fa824cadf994c9d67edb0/chapter01/14_top_k/README.md) | 16.9 节与附录 B 的边界表 |
| CMEM 的直接读、共享与远端路由 | [CMEM](https://github.com/ayaka14732/pallas-tpu-tutorial/blob/b4fc5b11713aa34eef1fa824cadf994c9d67edb0/chapter02/03_cmem/README.md)、[远端 CMEM](https://github.com/ayaka14732/pallas-tpu-tutorial/blob/b4fc5b11713aa34eef1fa824cadf994c9d67edb0/chapter02/06_remote_cmem/README.md) | 15.5、19.8 节 |
| 主机内存与持久循环协议 | [主机内存](https://github.com/ayaka14732/pallas-tpu-tutorial/blob/b4fc5b11713aa34eef1fa824cadf994c9d67edb0/chapter02/08_host_memory/README.md) | 19.8 节；私有接口限制留在附录 B |
| 共享带宽下的发起顺序、内部 dot 与写回尾巴 | [大矩阵乘法](https://github.com/ayaka14732/pallas-tpu-tutorial/blob/b4fc5b11713aa34eef1fa824cadf994c9d67edb0/chapter02/09_large_matmul/README.md) | 17.7 节 |
| 时间边界、sfence 与两种时钟 | [五种时间](https://github.com/ayaka14732/pallas-tpu-tutorial/blob/b4fc5b11713aa34eef1fa824cadf994c9d67edb0/chapter03/01_five_times/README.md)、[sfence/VIF](https://github.com/ayaka14732/pallas-tpu-tutorial/blob/b4fc5b11713aa34eef1fa824cadf994c9d67edb0/chapter03/04_sfence_vif/README.md)、[GTC](https://github.com/ayaka14732/pallas-tpu-tutorial/blob/b4fc5b11713aa34eef1fa824cadf994c9d67edb0/chapter03/06_gtc/README.md) | 20.2、20.7 节 |
| XProf 跟踪事件确有 GTC 时间戳；不是全部插值 | [XProf/vtrace](https://github.com/ayaka14732/pallas-tpu-tutorial/blob/b4fc5b11713aa34eef1fa824cadf994c9d67edb0/chapter03/07_xprof_vtrace/README.md) | 修正 20.4 节与本章小结 |
| 硬件随机状态、key、计数器、分布、代价 | [随机数全章](https://github.com/ayaka14732/pallas-tpu-tutorial/tree/b4fc5b11713aa34eef1fa824cadf994c9d67edb0/chapter04) | 13.7、16.10 节与附录 B |
| 精确编码与 TC/BCS/TEC 目标差别 | [v4 TC](https://github.com/ayaka14732/tpuasm/blob/a6e3d927deb1dc5a906574e6b6bbfc8bc175d30d/docs/references/tpu_v4_tc.md)、[v4 BCS](https://github.com/ayaka14732/tpuasm/blob/a6e3d927deb1dc5a906574e6b6bbfc8bc175d30d/docs/references/tpu_v4_bcs.md)、[v6e TC](https://github.com/ayaka14732/tpuasm/blob/a6e3d927deb1dc5a906574e6b6bbfc8bc175d30d/docs/references/tpu_v6e_tc.md)、[v6e TEC](https://github.com/ayaka14732/tpuasm/blob/a6e3d927deb1dc5a906574e6b6bbfc8bc175d30d/docs/references/tpu_v6e_tec.md) | 20.6 节与附录 B |
| 来源映射、回灌、插入与兼容性 | [来源映射](https://github.com/ayaka14732/tpuasm/blob/a6e3d927deb1dc5a906574e6b6bbfc8bc175d30d/docs/design/tc_source_mapping.md)、[回灌](https://github.com/ayaka14732/tpuasm/blob/a6e3d927deb1dc5a906574e6b6bbfc8bc175d30d/docs/design/executable_replacement.md)、[兼容性](https://github.com/ayaka14732/tpuasm/blob/a6e3d927deb1dc5a906574e6b6bbfc8bc175d30d/docs/compatibility.md) | 20.6 节与附录 B |
| 编码往返与设备语义是不同证据 | [v6e 执行语义核对](https://github.com/ayaka14732/tpuasm/blob/a6e3d927deb1dc5a906574e6b6bbfc8bc175d30d/docs/design/tpu_v6e_execution.md) | 20.6、20.7 节的证据与计时边界；附录 B |

本书按接口和代价模型重新构造证明、算例与图，没有复制教程文字、代码或图。安装、实验载体、基准输出、工具内部 ABI 偏移和全助记符索引没有逐条搬进正文：它们不参与本书的线性推导，需使用工具时从上述固定来源查阅。接口失败与逆向得到的参数均注明来源环境，不列为跨代保证。
