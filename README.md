# Color Card Detection & Patch RGB Extraction

本项目用于从“相机拍摄的含多色块色卡图像”中，自动/手动定位色卡并切分为多个色块（patch），从每个色块 ROI 中采样像素并提取 RGB 编码数据，同时提供可视化结果用于质检与数据清洗。

> 本仓库产出的是 **色卡色块的 RGB 数据与可视化**，用于后续任何需要色卡编码向量的任务（例如光谱重建等）。

---
<img width="2816" height="1536" alt="1766388579925-019b44f6-1628-76d5-a492-e49007a69950" src="https://github.com/user-attachments/assets/3aaba90f-973e-4352-a499-defb0008f2a9" />

## 1. 功能概述

### 1.1 自动处理（Auto）
- 自动检测色卡所在区域并进行对齐/裁剪
- 将色卡划分为固定网格（例如 4×6 共 24 个色块）
- 对每个色块 ROI 进行像素采样（可选随机采样）并输出 RGB 数据
- 输出可视化图，辅助检查色卡定位与色块切分是否正确

### 1.2 手动处理（Manual）
当自动检测失败或结果不稳定时，可使用手动模式：
- 人工框选色卡区域（或关键边界）
- 按同样的网格规则切分 patch 并输出 RGB
- 输出可视化图用于确认结果

---

## 2. 输入与输出

### 2.1 输入（Input）
- 单张或批量图像（建议同一拍摄条件：光源、相机参数、色卡位置相对稳定）
- 图像中应包含完整或尽可能完整的色卡区域

### 2.2 输出（Output）
本项目输出两类结果：

1) **RGB 编码数据（核心产物）**  
- 每张图像输出一份色卡 RGB 数据
- 典型组织方式（示例）：  
  - `RGB.npy` shape: `(R, C, K, 3)`  
    - `R, C`：色卡网格行列数（如 4×6）  
    - `K`：每个色块采样点数（如随机采样 K=100）  
    - `3`：RGB 三通道  
- 也可以输出为：
  - 每个 patch 的均值 RGB（shape: `(R, C, 3)`）
  - 或扁平化向量（shape: `(R*C*K*3,)` / `(R*C*3,)`）

2) **质检可视化（辅助产物）**  
- 色卡定位框/透视矫正结果
- patch 网格叠加图（能直观看到每个色块切分是否正确）
- patch 颜色拼图或均值色块可视化（便于发现曝光/偏色/检测错误）

---

## 3. 快速开始（Quick Start）

> 说明：仓库内可能提供“自动检测脚本 / 手动检测脚本 / 随机采样版本脚本”等文件。下面以“自动模式”和“手动模式”的典型运行方式描述；具体脚本名与参数请以仓库文件为准。

### 3.1 安装依赖
```bash
pip install -r requirements.txt



![image](https://github.com/user-attachments/assets/fa25009a-73d8-4b8b-becb-f169fa4a2b30)
![image](https://github.com/user-attachments/assets/0fc656e7-1437-4772-9396-b4dfb48799bc)
<img width="2816" height="1536" alt="image" src="https://github.com/user-attachments/assets/8c4ff0e0-0081-4899-971d-cbdfbdbe4beb" />

