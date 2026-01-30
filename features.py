# colorcard_kit/features.py
import numpy as np

def srgb_to_linear(x):  # x in [0,1]
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)

def build_features(ref_346, sam_346, mode="log_ratio",
                   per_image_channel_norm=True,
                   eps=1e-6, clip_min=1e-6, clip_max=1e6):
    """
    将 (3,4,6) 的 ref/sample RGB（0~255）转换为用于学习的特征张量。
    参数:
      mode: 'log_ratio' | 'ratio' | 'multi'
      per_image_channel_norm: 是否做每图每通道标准化（对 log_ratio/ratio 有利）
    返回:
      X: 特征张量（'log_ratio'/'ratio' => (3,4,6); 'multi' => (15,4,6)）
      extras: dict，包含 ref_lin/sam_lin/ratio/log_ratio 便于保存与可视化
    """
    ref = ref_346.astype(np.float32) / 255.0
    sam = sam_346.astype(np.float32) / 255.0
    ref_lin = srgb_to_linear(ref)
    sam_lin = srgb_to_linear(sam)

    ratio = sam_lin / np.clip(ref_lin, eps, None)
    ratio = np.clip(ratio, clip_min, clip_max)
    log_ratio = np.log(np.clip(sam_lin, eps, None)) - np.log(np.clip(ref_lin, eps, None))
    delta = sam_lin - ref_lin  # 线性域差分

    if mode == "log_ratio":
        X = log_ratio.copy()
        if per_image_channel_norm:
            X = (X - X.mean(axis=(1,2), keepdims=True)) / (X.std(axis=(1,2), keepdims=True) + eps)
    elif mode == "ratio":
        X = ratio.copy()
        if per_image_channel_norm:
            X = X / (X.mean(axis=(1,2), keepdims=True) + eps)
    elif mode == "multi":
        # 按 [ref_lin, sam_lin, ratio, log_ratio, delta] 维度拼接（3*5=15 通道）
        X = np.concatenate([ref_lin, sam_lin, ratio, log_ratio, delta], axis=0)
        # multi 通常不做通道内标准化，保留原始比例
    else:
        raise ValueError(f"Unknown feature_mode: {mode}")

    extras = {
        "ref_lin": ref_lin,
        "sam_lin": sam_lin,
        "ratio": ratio,
        "log_ratio": log_ratio
    }
    return X.astype(np.float32), extras
