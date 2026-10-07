# 加速器编程：TPU、GPU、CUDA 与 JAX

一本按数学书体例写的中文教程，从逻辑门出发自底向上讲到 TPU 与 GPU 上的 kernel 和分布式程序。正文在 [`book/`](book/README.md)，在 GitHub 上可以直接阅读（公式由 GitHub 渲染）。

在线阅读：<https://sangandol.github.io/accelerator-programming/>。包含公式、章节导航、引用与术语预览、版式设置和互动图。默认使用米黄色背景和霞鹜文楷，可在「版式」中调整。

## 本地阅读

需要 Node.js 20 或更新版本。

```powershell
.\start.ps1
```

首次运行会安装锁定的依赖，然后打开 <http://127.0.0.1:43202/book/>。参数：`-NoBrowser` 不打开浏览器，`-Port <n>` 换端口，`-Setup` 重装依赖。

阅读器比 GitHub 多几样辅助：定义、命题、例、注按种类显示为不同颜色的方框；左侧是本章大纲（各节及其中的定义与命题）；正文中的"命题 1.17""1.3 节""第 2 章"可以悬停预览，术语悬停显示它的定义；顶栏可以打开术语表（中英对照）、在术语后显示英文、折叠全部证明（只读陈述）；定理旁的"被引用"列出后面哪些章用到它；部分插图有可以逐步操作的互动版。

`node server.mjs --export 01-circuits-adders.md out.html` 把一章导出为一个自包含的 HTML 文件。

## 检查

```powershell
npm run check
```

渲染全书，报告公式错误、目录中缺失的章节、失效的链接和锚点、指向不存在条目的编号引用（如"命题 9.4"）和编号不连续的条目。

## GitHub Pages

推送到 `main` 后，GitHub Actions 自动检查书稿、生成静态阅读器并发布到 GitHub Pages；也可在 Actions 页面手动运行 `Deploy GitHub Pages`。

```powershell
npm run build -- --base-path /accelerator-programming/
npm run check:site -- /accelerator-programming/
```

生成结果在 `dist/`，不需要 Node 服务即可阅读。静态构建复用本地阅读器的渲染，检查覆盖章节链接、锚点、悬停预览、插图和公式字体。部署到域名根目录时省略 `--base-path`。
