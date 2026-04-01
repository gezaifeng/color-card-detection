# config.py
from dataclasses import dataclass
import math

@dataclass
class PipelineConfig:
    # ---- UI/输入 ----
    input_mode: str = "auto"
    prefer_raw_linear: bool = True
    raw_use_camera_wb: bool = True
    raw_output_bps: int = 8

    # —— 新增：采样源选择
    # False: 从原始读取到的图像直接采样（默认，保持原行为不变）
    # True : 从“自动白平衡”后的图像采样（适用于原图偏色/曝光异常导致提取不稳定的情况）
    sample_from_adjusted: bool = False

    # ---- 提取模式（UI传参）----
    extract_mode: str = "normal"        # normal / aberration
    aberration_points: int = 9          # 像差：每色块采样点数
    aberration_seed: int = 0            # 像差：随机种子

    # ---- 手动回退 ----
    allow_manual: bool = True
    force_manual: bool = False
    manual_downscale: int = 1200

    # --- 检测增强（仅用于识别，不影响提取保存） ---
    det_tune_enable: bool = True  # 是否启用手动调节（UI开关）
    det_brightness: float = 1.0  # 亮度倍增（V通道），推荐范围 0.3~3.0
    det_contrast: float = 1.0  # 对比度（alpha），推荐范围 0.3~3.0
    det_saturation: float = 1.0  # 饱和度倍增（S通道），推荐范围 0.0~3.0
    det_gamma: float = 1.0  # gamma（<1提亮暗部），推荐范围 0.3~2.5

    det_use_clahe: bool = False  # 是否对灰度做CLAHE
    clahe_clip_limit: float = 2.0
    clahe_grid: int = 8

    # 自动增亮（你上次要的“过暗自动提亮”）是否启用
    det_auto_brighten: bool = True
    bright_target_median: float = 110.0
    bright_max_gain: float = 4.0
    bright_clip_low: float = 1.0
    bright_clip_high: float = 99.0
    bright_gamma_dark: float = 0.85
    bright_dark_thresh: float = 70.0

    # ---- 检测参数 ----
    target_height: int = 512
    sobel_ksize: int = 3
    edge_thresh: int = 50

    # ---- 网格 ----
    grid_rows: int = 6
    grid_cols: int = 12

    card_crop_long: float = 0.01
    card_crop_short: float = 0.02

    # 常规提取：每块采样点数（取均值）
    sample_count: int = 100
    sample_center_area: float = 0.40

    # ---- 特征 ----
    feature_mode: str = "log_ratio"
    per_image_channel_norm: bool = True
    save_extras: bool = True

    debug: bool = False

    @property
    def sample_center_side_ratio(self) -> float:
        a = max(0.0, min(1.0, float(self.sample_center_area)))
        return math.sqrt(a)
