# 附录 B　概念与 API 对照表

本附录把正文中的概念对应到各个编程接口的写法。**接口的拼写会变**，以 2026 年 10 月的版本为准：JAX 0.11（正文的 JAX 与 Pallas 代码都在 JAX 0.11.2 上运行过）、CUDA 13.x、cuTile Python（CUDA 13.1 起）、当时的 Triton。具体参数请查各自的文档；本附录只是索引。

## B.1 JAX

| 概念 | 写法 | 正文 |
| --- | --- | --- |
| 编译 | `jax.jit(f, static_argnums=..., donate_argnums=...)` | 13.3 节 |
| 查看追踪结果 | `jax.make_jaxpr(f)(*args)` | 13.2 节 |
| 查看编译结果 | `jax.jit(f).lower(*args).as_text()`、`.compile().as_text()`、`.compile().memory_analysis()` | 13.3、14.5 节 |
| 求导 | `jax.grad`、`jax.value_and_grad`、`jax.jvp`、`jax.vjp`、`jax.linearize` | 13.4 节 |
| 自定义反向、重计算 | `jax.custom_vjp`、`jax.checkpoint` | 13.4 节 |
| 向量化 | `jax.vmap(f, in_axes=...)` | 13.5 节 |
| 控制流 | `lax.cond`、`lax.while_loop`、`lax.fori_loop`、`lax.scan`、`lax.associative_scan` | 13.6 节 |
| 随机数 | `jax.random.key(seed)`、`jax.random.split(key)` | 13.7 节 |
| 精度 | `jnp.dot(a, b, preferred_element_type=jnp.float32, precision=lax.Precision.HIGHEST)` | 13.8 节、2.9 节 |
| 网格 | `jax.make_mesh(形状, 轴名, (AxisType.Explicit, ...))`，`jax.set_mesh(mesh)` | 14.1 节 |
| 分片 | `jax.P("x", None)`、`NamedSharding(mesh, spec)`、`jax.device_put(x, spec)`、`jax.typeof(x)` | 14.1 节 |
| 重新分片、指定输出分片 | `jax.reshard(x, spec)`、`jnp.dot(..., out_sharding=spec)`、`lax.with_sharding_constraint` | 14.2 节 |
| 手动 SPMD | `jax.shard_map(f, in_specs=..., out_specs=..., check_vma=...)` | 14.3 节 |
| 集合原语 | `lax.psum`、`lax.all_gather(..., tiled=True)`、`lax.psum_scatter`、`lax.ppermute`、`lax.all_to_all`、`lax.axis_index`、`lax.axis_size` | 14.3 节 |
| 多主机 | `jax.distributed.initialize()`、`jax.make_array_from_process_local_data` | 14.6 节 |

## B.2 Pallas TPU

| 概念 | 写法 | 正文 |
| --- | --- | --- |
| kernel | `pl.pallas_call(kernel, out_shape=..., grid=..., in_specs=..., out_specs=..., scratch_shapes=..., compiler_params=...)` | 16.1、16.2 节 |
| 块与下标映射 | `pl.BlockSpec(block_shape, index_map)`；更深的预取 `pipeline_mode=pl.Buffered(n)` | 16.2 节 |
| 程序编号 | `pl.program_id(d)`、`pl.num_programs(d)` | 16.3 节 |
| 条件、动态切片 | `pl.when(cond)`、`pl.ds(start, size)` | 16.3 节 |
| 内存空间 | `pltpu.VMEM`、`pltpu.SMEM`、`pl.ANY`（或 `pltpu.HBM`） | 16.4 节 |
| scratch | `pltpu.VMEM(shape, dtype)`、`pltpu.SemaphoreType.DMA((n,))`、`pltpu.SemaphoreType.REGULAR` | 16.4 节 |
| 维度语义 | `pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary"))` | 16.5 节 |
| 本地 DMA | `pltpu.make_async_copy(src, dst, sem)` 的 `.start()`、`.wait()`；`pltpu.sync_copy(src, dst)` | 16.6 节 |
| 嵌套流水线 | `pltpu.emit_pipeline(body, grid=..., in_specs=..., out_specs=...)` | 16.6 节 |
| 标量预取 | `pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=k, grid=..., in_specs=..., out_specs=...)` | 16.7 节 |
| 远程 DMA | `pltpu.make_async_remote_copy(src, dst, send_sem, recv_sem, device_id=(...), device_id_type=pl.DeviceIdType.MESH)` 的 `.start()`、`.wait_send()`、`.wait_recv()` | 19.1 节 |
| 信号与屏障 | `pl.semaphore_signal(sem, n, device_id=...)`、`pl.semaphore_wait(sem, n)`、`pltpu.get_barrier_semaphore()` 配合 `CompilerParams(collective_id=0)` | 19.1、19.2 节 |
| 解释器 | `interpret=pltpu.InterpretParams(detect_races=..., dma_execution_mode="on_wait" 或 "eager")` | 16.8 节、例 19.5 |
| 硬件信息 | `pltpu.get_tpu_info()`（只在 TPU 上可用） | 15.4 节 |

注意：JAX 0.11 中 `DeviceIdType`、`semaphore_signal`、`semaphore_wait` 已从 `pltpu` 移到 `pl`；旧代码中的 `pltpu.DeviceIdType` 等会报错。

## B.3 GPU 上的 kernel 语言

| 概念 | CUDA C++ | Triton | cuTile Python |
| --- | --- | --- | --- |
| 启动 | `kernel<<<grid, block, smem, stream>>>(...)` | `kernel[grid](..., num_warps=..., num_stages=...)` | `ct.launch(stream, grid, kernel, args)` |
| 程序编号 | `blockIdx`、`threadIdx`、`blockDim`、`gridDim` | `tl.program_id(d)` | `ct.bid(d)` |
| 下标 | 手算 | `tl.arange(0, B)` 构造指针块 | 块坐标 `index=(i, j)` |
| 载入 | 普通读写、`cp.async`、TMA | `tl.load(ptrs, mask=..., other=...)` | `ct.load(A, index=..., shape=..., padding_mode=...)` |
| 写回 | 普通写、TMA 写回 | `tl.store(ptrs, v, mask=...)` | `ct.store(C, index=..., tile=...)` |
| 片上存储 | `__shared__` | 编译器管理 | 编译器管理 |
| 块内同步 | `__syncthreads()`、mbarrier | 不需要 | 不需要 |
| warp 内交换 | `__shfl_sync`、`__shfl_xor_sync`、`__ballot_sync` | 编译器管理 | 编译器管理 |
| 原子操作 | `atomicAdd`、`cuda::atomic_ref<T, scope>` | `tl.atomic_add` | — |
| 矩阵乘法 | 经 CUTLASS/CuTe 使用 `mma.sync`、wgmma、tcgen05 | `tl.dot(a, b, acc)` | `ct.mma(a, b, acc)` |
| 归约 | 手写（22.4 节） | `tl.max`、`tl.sum` | 块上的归约函数（名称见 cuTile 文档） |
| 调试 | `compute-sanitizer`（含竞争检查） | `TRITON_INTERPRET=1` | — |

Pallas GPU：BlockSpec 级的 `pl.pallas_call` 可以经 Triton 后端运行；Mosaic GPU 后端（`from jax.experimental.pallas import mosaic_gpu as plgpu`）提供 `plgpu.kernel`、`plgpu.copy_gmem_to_smem`、`plgpu.Barrier`、`plgpu.barrier_wait`、`plgpu.wgmma`、`plgpu.tcgen05_mma`、`plgpu.emit_pipeline`、`plgpu.emit_pipeline_warp_specialized` 等（24.4 节）。

## B.4 通信

| 概念 | JAX | Pallas TPU | GPU |
| --- | --- | --- | --- |
| all-reduce | `lax.psum` | 自己用远程 DMA 写（19.4 节加 all-gather） | NCCL `ncclAllReduce` |
| all-gather | `lax.all_gather` | 19.3 节 | `ncclAllGather` |
| reduce-scatter | `lax.psum_scatter` | 19.4 节 | `ncclReduceScatter` |
| 置换 | `lax.ppermute` | 远程 DMA | `ncclSend`、`ncclRecv` |
| all-to-all | `lax.all_to_all` | 每个目标一个远程 DMA（19.6 节） | 用发送与接收组成；或专门的库（27.4 节） |
| 设备端直接访问 | — | 远程 DMA | 对称内存、NVSHMEM 的放与取（27.2 节） |

## B.5 从源运算到指令与硬件

本节按第 22 章的 ISA 区分编程接口、虚拟指令和物理单元。表中的指令族是阅读索引；具体类型、目标架构、参与者和完成规则都必须查对应 ISA。GPU 的 PTX 会继续降低为目标机器程序，不承诺逐条对应。TPU 的完整机器 ISA 未在本书中作为公开稳定接口给出，下面用操作含义描述它，避免把 Pallas API 名当成机器指令。

| 数学或程序操作 | 数据与硬件路径 | 指令或完成接口 | 应审查的条件 |
| --- | --- | --- | --- |
| TPU 块载入 | HBM → DMA → VMEM | Pallas 自动流水线，或 `make_async_copy` 的 start/wait | 块号、字节数、完成身份、复用前的全部读者 |
| TPU 逐元素运算 | VMEM → 向量寄存器 → VPU/EUP → VMEM | 向量 ALU 或一元流水线操作 | 打包、通道位置、源寄存器就绪、发射槽 |
| TPU 矩阵乘法 | 权重暂存 → MXU 活动权重；左输入 → 结果队列 → 寄存器 | 推权重、装入、乘法、取回四类操作 | 在途量、权重生命周期、K 部分积的 f32 合并 |
| GPU 普通加载/算术/存储 | 每线程寄存器与全局存储路径 | `ld.global`、`add`/`fma`、`st.global` 指令族 | warp 请求是否合并、dtype、寄存器溢出 |
| GPU warp 数据交换 | 同一 warp 的寄存器通道之间 | `shfl.sync` 指令族 | mask 中参与者一致、源 lane 有效 |
| Ampere 异步载入 | 全局 → 共享内存，不经过数值寄存器 | `cp.async`、commit/wait group | 发起线程完成自己的组后，其他消费者还需合适的协作同步 |
| warp MMA | 寄存器片段 → Tensor Core → 寄存器 | `mma.sync`，如 `m16n8k16` | 整 warp 参与、片段布局、输入与累加类型 |
| Hopper 张量载入 | TMA 描述符 → 共享内存交错块 | `cp.async.bulk.tensor` 指令族 + mbarrier | 描述符、预期字节、到达计数、代理可见性 |
| Hopper 矩阵乘法 | 共享内存/寄存器 → Tensor Core → 寄存器累加 | `wgmma.mma_async`、commit/wait group | 整 warpgroup 参与、等待相关组后才能读取或复用 |
| Blackwell 矩阵乘法 | 共享内存/TMEM → Tensor Core → TMEM | `tcgen05.mma`、`tcgen05.commit` + mbarrier | 目标支持、TMEM 分配、完成绑定、结尾读者的生命周期 |

> **现状（2026-10-07）**：GPU 指令形态与目标限制据 [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)；Hopper、Blackwell 资源与调优分别查 [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) 和 [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)。本次扩写的指令示意用于推理，没有 CUDA Toolkit 的编译验证。

“等数据到齐”和“让某类异步执行器看见之前的写”还可能是两项操作。不同代理（普通线程访存、TMA、矩阵执行器）之间的可见性按 ISA 的 fence/proxy 规则处理；高层库通常封装这些细节，低层示意省略的规则不能直接从代码中删去。

## B.6 通信中的完成与可见性

| 路径 | 完成对象的含义 | 覆盖缓冲前还缺什么 |
| --- | --- | --- |
| Pallas 远程 DMA | send 表示源读完；recv 表示目的写完 | 目的消费者读完后的信用 |
| CUDA stream 上的 NCCL | 操作已提交到指定 stream；完成由该 stream/事件依赖建立 | 下一次覆盖必须在通信读完与消费者用完之后 |
| 直接对等访问 | 受支持映射上的读写；发布使用合适的系统作用域同步 | 原子能力与双方执行进展保证、消费确认 |
| NVSHMEM 单边操作 | 由库规定 fence 的次序、quiet 的完成和信号的可见性 | 按库协议的目标消费确认，不可仅观察本地普通标志 |

> **现状（2026-10-07）**：参见 [NCCL 的 stream 语义](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html) 与 [Using NVSHMEM](https://docs.nvidia.com/nvshmem/api/latest/using.html)。`fence` 用于次序并不等于交付已完成；普通 CUDA 原子操作也不能自动跟踪一次网络 put。选择路径之后，再证明第 9 章的读前与覆盖前两条边。

## B.7 Pallas 与 tpuasm 的来源快照

本节于 **2026-10-08** 核对。教程使用 TPU v4、JAX `0.12.0.dev20261002+7fc69a22c2`、jaxlib `0.12.0.dev20261002`、libtpu `0.0.49`；这与正文已有示例的 CPU 解释器环境 JAX 0.11.2 是两套环境。以下是阅读索引，不声称这些新接口在本机运行过。

| 任务 | 来源中的接口或字段 | 正文使用 |
| --- | --- | --- |
| 显式 kernel 与设备核心轴 | `pl.kernel`、`pltpu.TensorCoreMesh`、`scratch_types` | 与定义 16.1 同样先写块、状态与所有权；不整体迁移已有 API |
| 循环、局部可变状态、对齐承诺 | `pl.loop`、JAX Ref、`ref.at[pl.ds(...)]`、`pl.multiple_of` | 16.3、16.9 节；对齐承诺必须由实际索引保证 |
| 更新既有缓冲的局部窗口 | `jax.new_ref`，把 Ref 传入显式 kernel | 16.9 节；旧状态、写区间及所有权显式给出，与数组的函数式更新分开 |
| 有状态随机数 | `pltpu.prng_seed`、`pltpu.prng_random_bits`、`stateful_uniform`、`stateful_bernoulli` | 16.10 节；比特、种子混合与分布变换分开计账 |
| 按块随机数 | Pallas key、`sample_block`；普通 `jax.random` 的 threefry key | 固定全局计数器映射；不同 key 实现不承诺序列相同 |
| 导出与精确往返 | `executable_programs`、`dump_compiled`、`dump_executable`、`format_assembly(..., encoding='exact')`、`assemble_listing` | 定义 20.8；raw image 需显式 target |
| 修改机器程序 | `replace_executable_programs`、`insert_executable_bundles`、`BundleInsertion`、`load_executable` | 命题 20.10；保留调用约定、分配与程序身份 |
| 编译来源 | `compiler_source_mapping`、`executable_source_maps` | 20.6 节；按 (PC, slot) 追溯，未知保持未知 |
| 局部计时 | `srdreg.lcclo`、`srdreg.lcchi`、`sfence` | 定义 20.11；先等完成，再跨栅栏边界读数 |
| 全局事件 | `srdreg.gtclo`、`srdreg.gtchi`、`vtrace` | 定义 20.13；不能与 LCC 或原始低位直接混算 |

tpuasm 还绑定 Python、libtpu 的 GNU build-id 和原生桥接后端，只有 `.target` 相同不能保证可用。其当前兼容性表列 Linux x86-64、CPython 3.14t 与特定 libtpu 0.0.48/0.0.49 组合；编解码可离线进行，实际装载仍需要相应设备与运行时。具体组合以[兼容性表](https://github.com/ayaka14732/tpuasm/blob/a6e3d927deb1dc5a906574e6b6bbfc8bc175d30d/docs/compatibility.md)为准。

| target | 执行对象 | 格式中的关键差别 |
| --- | --- | --- |
| `tpu-v4-tc` | TensorCore | 12 槽；完整映像包数为 10 的正整数倍；`.align 10` |
| `tpu-v4-bcs` | BarnaCore Sequencer | 两个标量槽；完整映像包数为 16 的正整数倍；semantic protobuf 与机器字节可互转 |
| `tpu-v6e-tc` | TensorCore | 15 槽，四个向量算术槽、两个向量读槽；包数为 8 的正整数倍；DMA 与标量槽不能任意同包 |
| `tpu-v6e-tec` | SparseCore 向量子核 | 12 槽，含 stream；不要求包数对齐；导出需显式 target，当前只支持等长替换，无来源注释 |

这张表是工具的格式约束，不是吞吐表。v6e 的向量整数乘法形式还会占邻槽，不能把 v4“没有向量整数乘法器”的分析搬过去。汇编器会联合求解共享立即数、选择字段与端口，但不重排指令、不补分支延迟或同步。

## B.8 教程报告的能力与接口边界

以下均是 **2026-10-08 核对的来源报告**，没有本书的真机复测。目的是在审查时知道该查哪一层；特定版本的失败不写成永久定理。

| 操作 | 教程报告的边界 | 本书的处理 |
| --- | --- | --- |
| 非对齐多列 tile 行窗口 | Mosaic 的单窗口降低拒绝；可拆跨步 DMA | 例 16.17 同时写源、目的布局和总载荷 |
| 数值转换与位型重解释 | 窄格式需打包；浮点到整数有舍入/饱和语义 | 16.9 节要求规格给转换规则；bitcast 不代替数值转换 |
| 整数逐元素运算 | v4 向量整数乘法需合成；部分无符号比较降低受限 | 16.10 节按机器操作计数，不按算法轮数估时 |
| 子通道 gather 与 scatter | 子通道 gather 可移位选择合成；重复 scatter 需冲突语义 | 命题 16.15、例 16.16 |
| 转置、拼接、扫描 | 打包转置的提交间隔不同；拼接跨边界要合并；cumsum 降低受限 | 16.9 节给数学构造，20.2 节限定参数适用形式 |
| 显式 MXU FIFO | v4 的 `matmul_push_rhs`/`matmul_lhs_fifo` 路径曾选择错误权重来源；int8 路径受限 | 20.6 节核对实际 selector；保留已有 dot 示例 |
| top-k | 有效项不足时，用值清除可能重复下标 | 例 16.18 独立维护有效性、选择历史与 tie-break |
| CMEM | 硬件有 staging、直接读与远端路径；公开 scratch 分配受限 | 15.5、19.8 节分开可达性、分配、路由、完成 |
| pinned host / 持久主机流 | 动态主机窗口和并发主机访问含公开接口限制或私有依赖 | 19.8 节只建立端点与信用协议，不提供未经验证的运行代码 |
| 随机数、随机舍入 | 硬件状态作用域、Pallas/JAX key 差别；v4 stochastic_round 降低受限 | 16.10 节规定重放、端点、特殊值与生成代价 |
| v6e selector 与 formatter | 字段宽度、显示常量和实际消费位宽未必相同；部分 encoder 槽形式受限 | 20.6 节分开文本、字节与执行证据；具体观察查附录 E 的执行语义文档 |

逐节来源与审计范围见附录 E。
