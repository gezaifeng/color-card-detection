import numpy as np
import cv2
from config import PipelineConfig

def shrink_quad(box, crop_long_ratio, crop_short_ratio):
    x_min, y_min = np.min(box, axis=0)
    x_max, y_max = np.max(box, axis=0)
    w, h = x_max - x_min, y_max - y_min
    dw = int(w * crop_long_ratio)
    dh = int(h * crop_short_ratio)
    return np.array([[x_min+dw,y_min+dh],[x_max-dw,y_min+dh],
                     [x_max-dw,y_max-dh],[x_min+dw,y_max-dh]])

def _robust_center_pixels(patch_bgr, sample_count):
    if patch_bgr.size == 0:
        return np.zeros((sample_count, 3), dtype=np.float32)
    px = patch_bgr.reshape(-1, 3)
    if px.shape[0] >= sample_count:
        idx = np.random.choice(px.shape[0], sample_count, replace=False)
        sel = px[idx]
    else:
        reps = sample_count // max(px.shape[0],1) + 1
        sel = np.tile(px, (reps, 1))[:sample_count]
    med = np.median(sel, axis=0)
    mad = np.median(np.abs(sel - med), axis=0) + 1e-6
    mask = np.all(np.abs(sel - med) <= 2.5 * mad, axis=1)
    sel = sel[mask]
    if sel.shape[0] < sample_count:
        pad_reps = sample_count // max(sel.shape[0],1) + 1
        sel = np.tile(sel, (pad_reps, 1))[:sample_count]
    return sel.astype(np.float32)

def extract_card_means(image_bgr, mapped_box, cfg: PipelineConfig, draw_grid=True):
    """
    在原始图像中对 4×6 网格取“中心 area% 面积”，做鲁棒均值，输出 (3,4,6)
    """
    box = shrink_quad(mapped_box, cfg.card_crop_long, cfg.card_crop_short)
    x_min, y_min = np.min(box, axis=0)
    x_max, y_max = np.max(box, axis=0)
    W, H = x_max - x_min, y_max - y_min
    rows, cols = cfg.grid_rows, cfg.grid_cols
    cw, ch = W // cols, H // rows

    # —— 基于面积比例的中心裁剪
    s = cfg.sample_center_side_ratio              # 线性比例
    margin_ratio = (1.0 - s) / 2.0               # 两侧各裁掉的比例
    dx = int(cw * margin_ratio)
    dy = int(ch * margin_ratio)

    means = np.zeros((3, rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            x1 = x_min + c * cw
            y1 = y_min + r * ch
            x2 = x1 + cw
            y2 = y1 + ch

            # 中心区域坐标（保证合法）
            cx1 = x1 + dx
            cy1 = y1 + dy
            cx2 = x2 - dx
            cy2 = y2 - dy
            if cx2 <= cx1 or cy2 <= cy1:
                # 若面积系数极小导致中心区域无效，退回到 1px 安全取值
                cx1, cy1 = x1 + cw//4, y1 + ch//4
                cx2, cy2 = x2 - cw//4, y2 - ch//4

            patch = image_bgr[cy1:cy2, cx1:cx2]
            sel = _robust_center_pixels(patch, cfg.sample_count)  # (N,3) BGR
            rgb = sel[:, ::-1]  # 转 RGB
            mean_rgb = rgb.mean(axis=0)
            means[:, r, c] = mean_rgb

            if draw_grid:
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0,0,255), 1)      # 小格外框
                cv2.rectangle(image_bgr, (cx1, cy1), (cx2, cy2), (0,255,255), 2) # 实际采样区域（黄）

    return means  # (3,4,6)
