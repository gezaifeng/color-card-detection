from dataclasses import dataclass
import math

@dataclass
class PipelineConfig:
    # 网格设置
    grid_rows: int = 6
    grid_cols: int = 12
    target_height: int = 512

    # 色卡边缘裁剪比例（对检测四边形做等比内缩）
    card_crop_long: float = 0.01
    card_crop_short: float = 0.02

    # 每格中心采样
    sample_count: int = 100
    sample_center_area: float = 0.40  # (0,1] 中心采样面积比例

    # 边缘检测
    sobel_ksize: int = 3
    edge_thresh: int = 50

    # 特征输出模式：'log_ratio' | 'ratio' | 'multi'
    feature_mode: str = "log_ratio"
    per_image_channel_norm: bool = True
    save_extras: bool = True

    # —— 新增：手动框选回退 & 强制手动
    allow_manual: bool = True
    force_manual: bool = False
    manual_downscale: int = 1200  # 手动标注时的最长边显示尺寸

    # —— 新增：RAW 读取选项
    prefer_raw_linear: bool = True         # 有 rawpy 时，尽量用线性无伽马解码
    raw_use_camera_wb: bool = True         # 使用相机白平衡
    raw_output_bps: int = 8                # 输出 8-bit（与现有流程对齐）

    @property
    def sample_center_side_ratio(self) -> float:
        a = max(0.0, min(1.0, float(self.sample_center_area)))
        return math.sqrt(a)
