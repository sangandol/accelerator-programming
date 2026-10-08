# 附录 A　硬件参数表

本附录集中列出正文用到的硬件数字。**所有数字都会过时**，以 2026 年 10 月为准；更新本书时首先更新这里。数字取自厂商的公开资料与公开的测量，标为"约"的是取整或推算的值，"—"表示没有找到可靠的公开数字。正文的推导只依赖这些数字的量级。

## A.1 参考芯片 X

第一、二部分的例子使用的假想芯片（例 6.2）：

| 项目 | 值 |
| --- | --- |
| 时钟 | 1 GHz |
| 核心 | 2 个，每个有 4 个 $128 \times 128$ 的 bf16 权重驻留阵列 |
| 矩阵峰值 $P_{\text{mat}}$ | $2.6 \times 10^{14}$ FLOP/s |
| 向量峰值 $P_{\text{vec}}$ | $4.1 \times 10^{12}$ FLOP/s |
| 片上存储 | 每核心 32 MiB |
| HBM | 64 GB，$B = 1$ TB/s |
| 拐点 $P/B$ | 262 FLOP/字节 |
| 芯片间链路（第 12 章的例子） | 每芯片注入带宽 $10^{11}$ 字节/秒，$P/\beta = 2600$ |

## A.2 TPU

| | v4 | v5e | v5p | v6e（Trillium） | Ironwood（TPU7x） | TPU 8t | TPU 8i |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 每芯片 TensorCore | 2（megacore） | 1 | 2（megacore） | 1 | 2（两个芯粒，JAX 中为两个设备） | — | — |
| MXU | $128 \times 128$ | $128 \times 128$ | $128 \times 128$ | $256 \times 256$ | $256 \times 256$ | — | — |
| bf16 峰值（每芯片） | 275 TFLOP/s | 197 | 459 | 918 | 2307 | — | — |
| 更窄格式的峰值 | — | int8 394 | int8 918 | int8 1836 | fp8 4614 | fp4 12.6 PFLOP/s | fp4 10.1 PFLOP/s |
| HBM 容量 | 32 GiB | 16 GB | 95 GB | 32 GB | 192 GB | 216 GB | 288 GB |
| HBM 带宽 | 1.2 TB/s | 0.82 TB/s | 2.8 TB/s | 1.64 TB/s | 7.4 TB/s | 6.5 TB/s | 8.6 TB/s |
| 片上 VMEM | 16 MiB/TC，另有 128 MiB CMEM | 约 128 MiB | — | — | — | 128 MB | 384 MB |
| SparseCore | 有 | — | 4 | 2 | 4 | 有 | 以集合通信加速单元取代 |
| ICI 拓扑 | 3D 环面 + OCS | 2D 环面 | 3D 环面 | 2D 环面，至多 256 芯片 | 3D 环面（$4^3$ 立方体）+ OCS，至多 9216 芯片 | 3D 环面，超级节点 9600 芯片 | Boardfly，1152 芯片 |
| 每芯片 ICI | 6 链路，每链路每方向约 45 GB/s | — | — | 4 端口，双向合计 800 GB/s | 双向合计 1200 GB/s | — | — |

**TPU v4 的微架构参数**（第 15、20 章；来自 ayaka14732 的 pallas-tpu-tutorial 的实测，时钟约 1.05 GHz）：

| 项目 | 值 |
| --- | --- |
| 向量寄存器 | 32 个，每个 $8 \times 128 \times 32$ 位 |
| 指令包 | 至多 12 条指令（`s0`、`s1`、`va0`、`va1`、`vld`、`vst`、`vx0`、`vx1`、`vr0`、`vr1`、`misc`、`cld`） |
| SMEM | 1 MiB/TC |
| EUP | 延迟 7 周期，发射间隔 2 |
| MXU | 乘法到取回约 83 周期（部分 MXU 约 101），每 8 周期接收或取回一个寄存器 |
| XLU | 跨通道归约约 79 周期，通道置换约 69 周期，发射间隔 8 |
| CMEM 直接读取 | `cld` → `crf` 约 53 周期，提交与取回间隔各 2 |
| 硬件随机比特 | `vrng` 每 8 周期至多提交一次；状态与代价讨论见 16.10 节 |
| 向量到标量 | 约 43 周期 |
| HBM → VMEM 的 DMA | 约 $484 + 1.1K$ 周期（$K$ 为 KiB），约 970 GB/s |
| DMA 粒度 | 512 字节 |

## A.3 GPU

| | A100 | H100（SXM） | B200 | Rubin | RTX 3060 Ti |
| --- | --- | --- | --- | --- | --- |
| 架构 / 计算能力 | Ampere / 8.0 | Hopper / 9.0 | Blackwell / 10.0 | Rubin / — | Ampere / 8.6 |
| SM 数 | 108 | 132 | 148 | 224 | 38 |
| bf16 稠密峰值 | 312 TFLOP/s | 约 990 | 约 2300 | — | — |
| 更窄格式的峰值 | int8 624 | fp8 约 2000 | fp8 约 4500 | NVFP4 约 50 PFLOP/s | — |
| f32（CUDA 核心） | 19.5 TFLOP/s | 约 67 | — | — | 约 16.2 |
| 每 SM 寄存器 | 256 KB | 256 KB | 256 KB | — | 256 KB |
| 每 SM 共享内存上限 | 164 KB | 228 KB | 228 KB | — | 100 KB |
| 每 SM 张量存储 | — | — | 256 KB（TMEM） | — | — |
| L2 | 40 MB | 50 MB | 126 MB | — | 4 MB |
| 显存 | 80 GB HBM2e，约 2 TB/s | 80 GB HBM3，3.35 TB/s | 192 GB HBM3e，8 TB/s | 288 GB HBM4，22 TB/s | 8 GB GDDR6，448 GB/s |
| 每 GPU 的 NVLink（双向） | 600 GB/s | 900 GB/s | 1.8 TB/s | 3.6 TB/s | — |
| NVLink 域 | 8 | 8 | 8；NVL72 为 72 | — | — |

**通用的 SM 参数**（第 21、22 章）：warp 32 个线程；每块至多 1024 个线程；每 SM 至多 64 个 warp（计算能力 8.6 为 48 个）；每线程至多 255 个寄存器；共享内存 32 个 4 字节宽的存储体；全局内存访问以 32 字节为段；超越函数单元每 SM 每周期约 16 次（Hopper）。

## A.4 能耗（45 nm，Horowitz 2014）

见 3.5 节的表。绝对数值已随工艺下降，比例关系大体不变：从 DRAM 读一个 64 位字约相当于数百次 32 位浮点乘法。

## A.5 主要来源

- Google Cloud TPU 文档（各代的系统架构页面）与 Google Cloud 博客（Ironwood、TPU 8t/8i 的技术介绍）；
- Austin 等，*How to Scale Your Model*（JAX 团队，2025），TPU 与 GPU 两章的参数表；
- NVIDIA 的产品资料与技术博客（Hopper、Blackwell、Rubin 架构介绍），CUDA C++ Programming Guide 中的计算能力参数表；
- ayaka14732，*Pallas TPU Kernel 开发教程*（2026），TPU v4 的实测数据。

完整的文献信息见[附录 E](E-references.md)。

## A.6 怎样把参数表代入书中的模型

参数表给的是某种硬件口径，公式需要的是所用资源的口径。代入前先确定三件事：按芯片还是按核心、按单向还是按双向、按稠密还是按稀疏。GB 为 $10^9$ 字节，GiB 为 $2^{30}$ 字节；GB/s 与 GiB/s 也要作相同换算。

> **现状（2026-10-07）**：JAX 的 [TPU Hardware Reference](https://docs.jax.dev/en/latest/pallas/tpu/hardware.html) 按 TensorCore 给数，列 v5e 的 VMEM 为 128 MiB、HBM 带宽为 820 GB/s。本书 v5e 的 $0.82$ TB/s 算例使用这组后端参考值。Google Cloud 的 [v5e 产品文档](https://docs.cloud.google.com/tpu/docs/v5e) 列 800 GiB/s，换成十进制约为 0.859 TB/s；两份资料的口径有差异，不应把数字无说明地混用。实际部署以目标设备与后端报告为准。

例如按本书参考值，v5e 的 bf16 拐点为 $197/0.82\approx240$ FLOP/字节；正方形 tile 的强度为 $b/2$，故取 $b=512$ 的候选略高于这个拐点。它只说明该理想模型的计算与输入带宽主项可能平衡，不保证 DMA、向量单元、输出流量和首尾开销都被掩盖。

> **现状（2026-10-07）**：NVIDIA 的 [H100 产品表](https://www.nvidia.com/en-us/data-center/h100/) 给 SXM 的 bf16 1979 TFLOP/s，并注明使用稀疏；本书的稠密例子取其一半，约 990 TFLOP/s。使用 3.35 TB/s HBM 时，稠密拐点约 296 FLOP/字节。产品表中的 900 GB/s NVLink 是双向合计，不能直接作为单方向的 $\beta$。

更新数字时要把衍生量一起更新：$P/B$ 决定块强度，$BL$ 决定在途量，$\alpha+S/\beta$ 决定搬运延迟，容量决定缓冲数。每项数字都应保留单位、对象、日期和来源；假设的延迟须标为假设，不能与厂商公开参数混列。

> **现状（2026-10-07）**：第八代 TPU 的系统级参数来自 Google 的 [TPU 8t/8i 技术介绍](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive/)；Rubin 的 HBM、互联与单元组织来自 NVIDIA 的 [Rubin 架构介绍](https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/)。这里的参数表示公开设计与标称能力，不表示本书做过真机测量。
