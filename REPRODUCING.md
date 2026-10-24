# 复现步骤

从仓库检出目录运行以下命令。推荐 Python 3.12、Git，以及支持对应 RAW 的 LibRaw/rawpy。Windows / Python 3.12 已做本地验证；其他系统的 C99 math 适配已保留，但本次没有对应系统验收。

## 1．安装核心

```sh
git clone https://github.com/q734738781/a7r5-hncs-look.git
cd a7r5-hncs-look
python -m venv .venv
```

激活环境：

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```sh
# macOS / Linux
source .venv/bin/activate
```

安装及不依赖上游资源的检查：

```sh
python -m pip install -e ".[fast,dev]"
python -m hncs_native inspect
python tools/verify_release.py
python -m pytest -q
```

`fast` 提供 Numba 加速；不安装时使用 NumPy 参照路径。测试使用合成标定和合成颜色，不需要仓库维护者的 RAW 或任何相机连接。只从仓库用 editable 安装，不提供独立 PyPI 二进制分发。

## 2．自行取得固定上游资源

先阅读 [第三方说明](THIRD_PARTY_NOTICES.md) 及各上游许可，确认拥有使用所需资源的权利，再运行：

```sh
python tools/fetch_upstream.py --fetch
python -m hncs_native doctor
```

脚本只克隆 `upstream.lock.json` 指定的三个公开仓库，在 `.external/` 检出固定提交；没有自动安装软件、下载商业程序或操作相机。已有目录若不是指定提交会停止，不会 reset 或覆盖用户更改。`--fetch` 才访问网络；不带该选项只检查本地资源。

固定版本：

| 仓库 | 提交 |
| --- | --- |
| 751K/sony-a7m5-looks | `042e2c31a9d75803f2af2f3aaa5ede5e6030e448` |
| fishvivfish/HNCS-Lightroom | `38fbbfc9c593e3ba65ced3a7c08e1d3e84f91ca0` |
| shenmintao/V-Log-Alchemy | `e51ba9a23458ff5c316f630b860eb0201139d126` |

`doctor` 检查 49 份所需代码／资源的 SHA-256，文本按 LF 统一后比较，避免 Windows 换行差异。已有上游目录也可通过 **HNCS_UPSTREAM_DIR** 指定；代码没有预设作者的磁盘、用户名或安装路径。该目录下应保留上述三个仓库目录名。

本项目只读取上游公开资源，不重新发布 DCP、XMP、Film Curve、LUT 数据或其二进制。它们仍受各自许可和来源权利约束。

## 3．用自己的 RAW 渲染

把自己的 A7R V RAW 放在已忽略的 `data/local/`，下面用 `scene-01.ARW` 作为占位文件名：

```sh
python -m hncs_native render --raw data/local/scene-01.ARW --mode look --output runs/look.png
python -m hncs_native render --raw data/local/scene-01.ARW --mode teacher --output runs/teacher.png
python -m hncs_native render --raw data/local/scene-01.ARW --mode sony --output runs/sony-baseline.png
```

- `look`：发布的统一 Look，包括完整网格。
- `teacher`：METHOD 中公开的组合教师，**不是 Adobe 实际导出**。
- `sony`：零增量的软件 Sony 基线，**不是 Sony Edit 实际导出文件**。

完整物理 RAW 处理可能耗时并占用较多内存。默认 `--scale 12` 在颜色运算完成后作 BOX 平均；不会先把 RAW 缩小。快速检查可指定原始方向下的有效输出区域：

```sh
python -m hncs_native render --raw data/local/scene-01.ARW --crop 2000 2000 2192 2192 --scale 4 --output runs/crop.png
```

输出必须是新文件。程序读取 RAW 自己的白平衡与标定，包含 RawNR、ITP、一次 DRO 和几何处理；不提供额外 DRO 叠加选项。结果只含颜色路径所需处理，不等于 Sony 完整锐化／JPEG 输出链。每个输出附有只使用文件名和摘要的 JSON 回执。

## 4．采样并建立自己的数据集

```sh
python -m hncs_native prepare --raw data/local/scene-01.ARW --output data/local/scene-01
python -m hncs_native prepare --raw data/local/scene-02.ARW --output data/local/scene-02
python -m hncs_native prepare --raw data/local/scene-03.ARW --output data/local/scene-03
```

每张生成同前缀 `.npz` 与 `.json`：实际物理像素、教师目标、裁剪有效性、空间块编号及颜色标定。RAW 不被复制。缓存仅为用户本地材料；`.gitignore` 排除了这些目录、RAW、NPY、NPZ 与运行结果。

复制并编辑 [`examples/dataset.json`](examples/dataset.json)，按你自己的文件安排 construction 与 validation。路径相对于清单文件；`calibration_state_ids` 明确指定数学积分使用的标定状态。验证图片不参与照片损失，默认标定状态也从 construction 选择。

建议覆盖日光、暖光、较高色温，以及绿叶、天空、食物、正常高光。图像内容只决定固定物理采样与观感统计，不会变成运行时场景分类。

## 5．数学拟合与评价

```sh
python -m hncs_native fit --dataset examples/dataset.json --output runs/own-fit --maxiter 600 --grid-iterations 600 --integer-sweeps 3 --seconds 1200
python -m hncs_native evaluate --dataset examples/dataset.json --look runs/own-fit/looks/HNCS_UserFit.a7look --output runs/own-fit-evaluation.json
```

默认以发布 Look 作初猜；也可用 `--start` 指定兼容的 A7R V 参数文件。初猜不进入颜色参考损失。`--strength` 是算子自稳定损失系数，默认历史值 0.05，**不是运行时网格强度或去紫滑块**。

`--seconds` 限制连续联合阶段；后续网格优化和整数扫描分别由 `--grid-iterations`、`--integer-sweeps` 限制。想先验收流程，可把三者降到 2、2、0；这只检查链路，不代表拟合质量收敛。

产物包括合同／目标设置、连续历史、检查点、整数改进记录、评价指标和 `HNCS_UserFit.a7look`。输出目录必须不存在，避免覆盖已有实验。多次运行使用不同目录。

`evaluate` 中的 ΔE00 相对**组合教师**计算，不是相对 Adobe 输出，更不是哈苏实机误差。普通有效点和全部样本分别报告；验证集反复查看后不能宣称盲测。优化停止条件或整数邻域无改进也不构成全局最优证明。

## 6．Adobe 对照与发布文件核验

Adobe 实际全流程需要用户在自己的 Camera Raw / Lightroom 中加载来源匹配的配置并真实导出。保持可追踪的 As Shot 白平衡、基础调整、Process Version、几何设置、输出色域与软件版本；不要把 `--mode teacher` 的文件冒充 Adobe 导出。本仓库没有商业软件自动化或许可证绕过代码。

```sh
python tools/verify_release.py
python tools/check_publication.py
```

第一项核验原 Look 和 27 个预览；第二项按 Git 的拟发布文件清单检查原始 RAW、数据缓存、商业二进制、个人路径与凭据标记。检查脚本只报告，不删除文件。它们不能代替人工检查照片中的人脸、文字或其他隐私。

**完整历史参数重算限制：**私人训练 RAW、七个历史标定状态及原初猜未公开，因此不承诺新数据／不同库版本重新拟合后得到同一参数摘要。现有 Look 的精确下载核验、数值方法复现，以及使用自有数据重新求解是三种不同层次的复现。
