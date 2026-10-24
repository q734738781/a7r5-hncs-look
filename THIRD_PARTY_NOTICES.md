# 第三方来源与许可范围

本仓库新增的网页、适配与求解代码按根目录 MIT 许可发布；具体第三方来源与各自权利保留如下。调用外部项目或借鉴数学方法不表示取得其所有数据、商标、商业软件或固件的再许可。

| 来源 | 本项目使用方式 | 上游许可／说明 |
| --- | --- | --- |
| [751K/sony-a7m5-looks](https://github.com/751K/sony-a7m5-looks/tree/042e2c31a9d75803f2af2f3aaa5ede5e6030e448) | 自行取得的 Python Sony Edit 重建函数；本项目提供 RAW 标定与原生增量适配 | MIT，保留于 [licenses/sony-a7m5-looks.txt](licenses/sony-a7m5-looks.txt) |
| [fishvivfish/HNCS-Lightroom](https://github.com/fishvivfish/HNCS-Lightroom/tree/38fbbfc9c593e3ba65ced3a7c08e1d3e84f91ca0) | 自行取得的 A7R V DCP/XMP、白平衡和色表处理函数；用作组合教师资源 | Apache-2.0，见 [licenses/HNCS-Lightroom.txt](licenses/HNCS-Lightroom.txt)；资源来源及第三方权利仍需查看上游说明 |
| [shenmintao/V-Log-Alchemy](https://github.com/shenmintao/V-Log-Alchemy/tree/e51ba9a23458ff5c316f630b860eb0201139d126) | 自行取得的 Standard Film Curve、工作空间描述及显示尾链参考 | Apache-2.0，见 [licenses/V-Log-Alchemy.txt](licenses/V-Log-Alchemy.txt)；不在本仓库再分发其 LUT／曲线资源 |
| [QAQLeonard/a7c2-fl23-research](https://github.com/QAQLeonard/a7c2-fl23-research/tree/782785641ca9f8b984940c7274b48706219e0da4) | `src/hncs_native/grid.py` 适配其坐标、插值与平滑结构；文件保留来源与版权说明 | Copyright (c) 2026 A7C II FL2/FL3 preview contributors；MIT，见 [licenses/a7c2-fl23-research.txt](licenses/a7c2-fl23-research.txt) |
| [Liwenru88/a7r5-rea](https://github.com/Liwenru88/a7r5-rea/tree/031960ba5b737ca38d8ebbb68cc78a80193efb45) | 原生参数数学和 `.a7look` 交换格式的参考；这里独立打包，不执行或分发其固件构建器 | 根许可副本 [licenses/a7r5-rea.txt](licenses/a7r5-rea.txt)；上游 README 另有非商业使用声明，应按实际使用部分核对，不能仅凭本项目 MIT 覆盖上游条件 |
| [Jannchie/llr](https://github.com/Jannchie/llr) | Sony RAW 和颜色算法交叉参考 | 本仓库没有复制其代码或数据 |

NumPy、SciPy、Pillow、tifffile、colour-science、rawpy/LibRaw、Numba/llvmlite 和 pytest 由用户的包管理器安装，保留各自许可。本仓库不捆绑这些依赖的二进制分发。

`method/source-port.json` 记录从研究代码整理到公开包时的文件摘要；数值核心仅作包名和可移植性调整，并移除未使用的历史接口。原始研究目录及作者本机路径没有进入该记录。

## 未随仓库发布

- 原始 RAW、私人标定缓存、训练照片全尺寸文件或作者安装路径。
- Sony 固件、CP/AP 镜像、Adobe／Sony／Phocus 可执行程序、密钥或授权凭据。
- 上游 DCP、XMP、Film Curve、商业 LUT 或由商业程序提取的原始二进制资源。

`tools/fetch_upstream.py --fetch` 是用户明确选择后从对应公开仓库获取固定版本的便利脚本，不改变上游资源的权利状态，也不表示本项目为它们提供授权保证。
