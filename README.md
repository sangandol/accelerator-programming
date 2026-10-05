# 加速器编程：TPU、GPU、CUDA 与 JAX

一本按数学书体例写的中文教程，从逻辑门出发自底向上讲到 TPU 与 GPU 上的 kernel 和分布式程序。正文在 [`book/`](book/README.md)，在 GitHub 上可以直接阅读（公式由 GitHub 渲染）。

## 本地阅读

需要 Node.js 20 或更新版本。

```powershell
.\start.ps1
```

首次运行会安装锁定的依赖，然后打开 <http://127.0.0.1:43202/book/>。参数：`-NoBrowser` 不打开浏览器，`-Port <n>` 换端口，`-Setup` 重装依赖。

## 检查

```powershell
npm run check
```

渲染全书，报告公式错误、目录中缺失的章节、失效的链接和锚点。
