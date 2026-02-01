# ColorCard Kit（色卡识别与特征导出工具）

本项目用于对“溶液 + 多色块色卡编码”拍摄图像进行**批量色卡识别、色块采样与特征导出**。程序会在每张图片中自动定位上下两块色卡区域（`ref` / `sample`），按网格切分色块并提取 RGB 数据，然后构建 `ratio / log-ratio` 等特征，最终输出多组 `.npy` 文件与可视化结果，便于后续的光谱重建/回归建模与数据质检。

---

## 功能概览

- **批量处理（GUI）**：选择输入/输出目录，一键处理文件夹内所有图片  
- **自动检测两块色卡区域**：上方色卡为 `ref`，下方色卡为 `sample`  
- **两种采样模式（normal / aberration）**
  - `normal`：每个色块中心区域随机采样 `N` 个像素并求均值（鲁棒剔除离群点）
  - `aberration`：每个色块中心区域随机采样 `K` 个像素点，不求均值（保留多点信息）
- **特征构建（features）**
  - `ratio = sample / ref`
  - `log_ratio = log(sample / ref)`
  - 支持 `log_ratio / ratio / multi` 等模式（由 `features.py` 控制）
- **可视化输出**：保存检测框、网格、采样区域与特征展示图，便于快速检查  
- **手动回退**：自动检测失败时可手动框选两块区域（ref/sample）  
- **采样数据源可选（原图/调节后）**
  - 可选择采样/保存使用**原始数据**或**调节后数据**（适用于过暗/曝光不佳图片）
  - 支持“自动失败进入手动框选时使用调节后数据”的开关（normal/aberration 通用）

---

## 输出文件说明

每张输入图片会在输出目录生成多组 `.npy` 文件（默认 5 组），并生成可视化图片。文件前缀/命名由 `io_utils.py` 中的输出规则决定。

### 1) ref / sample 提取数据（RGB）

- `ref_*_346.npy`：上色卡（ref）提取的 RGB 数据  
- `sample_*_346.npy`：下色卡（sample）提取的 RGB 数据  

输出形状取决于提取模式：

- **normal**：`(3, m, n)`（默认 `m=6, n=12`）
- **aberration**：`(3, m, n, K)`（`K = aberration_points`）

其中 `3` 表示 RGB 三通道。

### 2) 中间量与最终特征

- `ratio_*_346.npy`：`ratio = sample / ref`
- `logratio_*_346.npy`：`log_ratio = log(sample / ref)`
- `features_<feature_mode>_*.npy`：最终特征 `X`（由 `feature_mode` 决定）

### 3) 可视化输出

- `output/vis/`：每张图生成一张可视化结果图，用于检查检测框、网格与采样区域是否正确。

---

## 安装依赖

建议 Python 3.9+：

- `numpy`
- `opencv-python`
- `Pillow`

可选 RAW 支持（CR2/CR3/NEF/ARW/DNG 等）：

- `rawpy`

安装示例：

```bash
pip install numpy opencv-python pillow
pip install rawpy
```

> RAW 支持取决于系统环境与 rawpy/LibRaw 可用性。

---

## 运行方式（GUI）

在项目根目录运行：

```bash
python ui_main.py
```

使用流程：

1. 选择【输入目录】（包含待处理图片）
2. 选择【输出目录】
3. 在各 Tab 设置参数（网格大小、采样方式、检测增强、RAW 与手动回退等）
4. 点击【开始批处理】

---

## 参数说明（与 UI 对应）

### 基础参数

- 色块行数 `m` / 列数 `n`：默认 `6 × 12`
- 检测缩放高度：仅影响检测速度与稳定性
- 中心采样区域：每个色块只在中心区域采样，降低边缘混色与反光影响
- 内缩裁剪：对色卡整体区域向内收缩，避免边框污染

### 提取方式

- `normal`：每色块采样 `sample_count` 点并求均值（鲁棒剔除离群）
- `aberration`：每色块采样 `aberration_points` 点，不求均值，输出多一维 `K`

### 检测增强（仅用于“找框”）

亮度/对比度/饱和度/Gamma/CLAHE/自动增亮等参数用于提升检测成功率。默认情况下这些增强只影响检测，不影响最终保存的数据；如需让最终保存的数据也使用调节后结果，请在 UI 中选择相应的“采样数据源”选项。

### RAW 与手动回退

- 优先 RAW 线性解码：对 CR2/CR3 等优先读 RAW，提高数据一致性
- 允许手动框选回退：自动检测失败时进入手动选区
- 强制手动框选：跳过自动检测，直接手动选区

---

## 项目结构（主要文件）

- `ui_main.py`：Tkinter GUI 批处理入口
- `pipeline.py`：单张图像处理主流程（读取 → 检测 → 提取 → 特征 → 保存 → 可视化）
- `detect.py`：区域检测（Sobel + 连通域 + 规则筛选），包含 RAW 读取支持
- `extract.py`：色卡网格切分与采样（normal/aberration）
- `features.py`：特征构建（ratio/log_ratio/multi）
- `visualize.py`：可视化输出
- `manual_select.py`：手动选择两个区域（自动失败回退）
- `config.py`：`PipelineConfig` 参数配置
- `io_utils.py`：文件遍历与输出路径工具

---

## 备注

- `ref` 为上色卡，`sample` 为下色卡；`ratio / log_ratio` 等特征均由两者组合计算得到。
- 输出文件名规则与输出目录结构以实际代码为准（见 `io_utils.py` 与 `pipeline.py`）。

---

## License

请根据你的实际需求在仓库中添加许可证（例如 MIT / Apache-2.0 / GPL 等）。
