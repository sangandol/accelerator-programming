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
| 解释器 | `interpret=pltpu.InterpretParams(detect_races=..., dma_execution_mode="on_wait" 或 "eager")` | 16.8 节、例 19.3 |
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
