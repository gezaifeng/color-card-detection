import numpy as np
import cv2
from config import PipelineConfig


def shrink_quad(box, crop_long_ratio, crop_short_ratio):
    x_min, y_min = np.min(box, axis=0)
    x_max, y_max = np.max(box, axis=0)
    w, h = x_max - x_min, y_max - y_min
    dw = int(w * crop_long_ratio)
    dh = int(h * crop_short_ratio)
    return np.array([
        [x_min + dw, y_min + dh],
        [x_max - dw, y_min + dh],
        [x_max - dw, y_max - dh],
        [x_min + dw, y_max - dh]
    ], dtype=np.int32)


def _robust_center_pixels(patch_bgr, sample_count):
    """
    仅用于 normal：保持你原本“鲁棒均值”的逻辑（这是你原版就有的）
    """
    if patch_bgr.size == 0:
        return np.zeros((sample_count, 3), dtype=np.float32)
    px = patch_bgr.reshape(-1, 3)
    if px.shape[0] >= sample_count:
        idx = np.random.choice(px.shape[0], sample_count, replace=False)
        sel = px[idx]
    else:
        reps = sample_count // max(px.shape[0], 1) + 1
        sel = np.tile(px, (reps, 1))[:sample_count]
    med = np.median(sel, axis=0)
    mad = np.median(np.abs(sel - med), axis=0) + 1e-6
    mask = np.all(np.abs(sel - med) <= 2.5 * mad, axis=1)
    sel = sel[mask]
    if sel.shape[0] < sample_count:
        pad_reps = sample_count // max(sel.shape[0], 1) + 1
        sel = np.tile(sel, (pad_reps, 1))[:sample_count]
    return sel.astype(np.float32)


def _random_center_pixels_no_processing(patch_bgr, k):
    """
    用于 aberration：严格按你的要求 —— 多点采样、不做均值、不做鲁棒剔除等任何额外处理
    返回 (k,3) 的 BGR float32
    """
    if patch_bgr.size == 0:
        return np.zeros((k, 3), dtype=np.float32)

    px = patch_bgr.reshape(-1, 3)
    if px.shape[0] >= k:
        idx = np.random.choice(px.shape[0], k, replace=False)
        sel = px[idx]
    else:
        # 像素不够则允许重复抽样补齐（不做其它处理）
        idx = np.random.choice(px.shape[0], k, replace=True)
        sel = px[idx]
    return sel.astype(np.float32)


def extract_card_means(image_bgr, mapped_box, cfg: PipelineConfig, draw_grid=True):
    """
    兼容两种提取模式（与你最开始描述一致）：
    - normal：每个色块采样 sample_count 个点，取均值 -> (3, rows, cols)
    - aberration：每个色块采样 aberration_points 个点，不取均值 -> (3, rows, cols, K)

    ✅ 仅修复“先采样后绘制”的顺序：
    - 采样一律在 draw 前的干净副本上进行，避免黄色框污染
    - 可视化仍然画在传入的 image_bgr 上（保持你原本可视化方式不变）
    """
    # ---- 1) 关键修复：先准备一个“干净图像”用于采样（不含任何绘制）
    sample_src = image_bgr.copy() if draw_grid else image_bgr

    # ---- 2) 参数与网格
    box = shrink_quad(np.asarray(mapped_box, dtype=np.int32), cfg.card_crop_long, cfg.card_crop_short)
    x_min, y_min = np.min(box, axis=0)
    x_max, y_max = np.max(box, axis=0)
    W, H = x_max - x_min, y_max - y_min

    rows, cols = int(cfg.grid_rows), int(cfg.grid_cols)
    cw, ch = W // cols, H // rows

    # 中心区域（按你原本 sample_center_area 推出 side_ratio）
    s = float(cfg.sample_center_side_ratio)
    margin_ratio = (1.0 - s) / 2.0
    dx = int(cw * margin_ratio)
    dy = int(ch * margin_ratio)

    # ---- 3) 模式选择（不改你原本行为：normal=均值；aberration=多点）
    extract_mode = getattr(cfg, "extract_mode", "normal")
    if extract_mode is None:
        extract_mode = "normal"

    if str(extract_mode).lower() == "aberration":
        K = int(getattr(cfg, "aberration_points", cfg.sample_count))
        out = np.zeros((3, rows, cols, K), dtype=np.float32)
    else:
        out = np.zeros((3, rows, cols), dtype=np.float32)

    # ---- 4) 遍历色块：先采样（基于 sample_src），再绘制（基于 image_bgr）
    for r in range(rows):
        for c in range(cols):
            x1 = int(x_min + c * cw)
            y1 = int(y_min + r * ch)
            x2 = int(x1 + cw)
            y2 = int(y1 + ch)

            cx1 = int(x1 + dx)
            cy1 = int(y1 + dy)
            cx2 = int(x2 - dx)
            cy2 = int(y2 - dy)
            if cx2 <= cx1 or cy2 <= cy1:
                cx1, cy1 = x1 + cw // 4, y1 + ch // 4
                cx2, cy2 = x2 - cw // 4, y2 - ch // 4

            # ===== 采样：严格在 sample_src 上完成，保证不会采到黄色框 =====
            patch = sample_src[cy1:cy2, cx1:cx2]

            if str(extract_mode).lower() == "aberration":
                sel = _random_center_pixels_no_processing(patch, K)  # (K,3) BGR
                rgb = sel[:, ::-1]  # -> RGB
                # 输出 shape: (3,rows,cols,K)
                out[:, r, c, :] = rgb.T
            else:
                sel = _robust_center_pixels(patch, cfg.sample_count)  # (N,3) BGR
                rgb = sel[:, ::-1]  # -> RGB
                mean_rgb = rgb.mean(axis=0)
                out[:, r, c] = mean_rgb

            # ===== 绘制：保持你原本的可视化方式不变 =====
            if draw_grid:
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 0, 255), 1)       # 小格外框（红）
                cv2.rectangle(image_bgr, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2) # 采样区域（黄）

    return out
