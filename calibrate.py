import numpy as np
from config import PipelineConfig

def apply_gray_calibration(rgb_array, gray_targets, gray_means, mode='linear'):
    """
    基于灰度条的逐通道标定：
    - linear: y = a1*x + a0，求逆 (x = (y - a0)/a1)
    - poly2 : y = a2*x^2 + a1*x + a0，牛顿迭代近似反解
    输入:
      rgb_array: (R,C,N,3) 的原始 RGB 样本
      gray_targets: 理想灰度步进（0~255 或按你的灰条定标）
      gray_means:   实测每段灰度的均值 (S,3)
    """
    rgb_array = rgb_array.astype(np.float32)
    corrected = rgb_array.copy()

    x = gray_targets.astype(np.float32)
    X_lin = np.vstack([x, np.ones_like(x)]).T
    X_poly = np.vstack([x**2, x, np.ones_like(x)]).T

    for ch in range(3):
        y = gray_means[:, ch].astype(np.float32)
        if mode == 'poly2' and len(x) >= 3:
            coef, *_ = np.linalg.lstsq(X_poly, y, rcond=None)
            a2, a1, a0 = coef

            def invert_channel(v):
                # 解 a2*t^2 + a1*t + a0 = v
                t = (v - a0) / max(a1, 1e-4)  # 初值
                for _ in range(5):
                    f = a2*t*t + a1*t + a0 - v
                    df = 2*a2*t + a1
                    t -= f / (df + 1e-6)
                return t

            vec_inv = np.vectorize(invert_channel, otypes=[np.float32])
            corrected[..., ch] = vec_inv(rgb_array[..., ch])
        else:
            coef, *_ = np.linalg.lstsq(X_lin, y, rcond=None)
            a1, a0 = coef
            a1 = a1 if abs(a1) > 1e-6 else 1e-6
            corrected[..., ch] = (rgb_array[..., ch] - a0) / a1

    return np.clip(corrected, 0, 255).astype(np.float32)
