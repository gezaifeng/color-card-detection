# detect.py
import os
import numpy as np
import cv2
from PIL import Image

# optional raw support
try:
    import rawpy  # pip install rawpy
    _HAS_RAWPY = True
except Exception:
    rawpy = None
    _HAS_RAWPY = False


def _is_raw(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in [".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".rw2", ".orf"]

def _auto_brighten_gray(gray8: np.ndarray, cfg):
    """
    只用于检测的灰度增强：解决过暗导致边缘弱、连通域不成形的问题
    - gain: 把 p50 拉到 target_median
    - gamma: 暗图适当<1提升暗部
    - clahe: 可选，增强局部对比（必须 uint8 单通道）
    """
    if gray8 is None or gray8.dtype != np.uint8:
        return gray8

    # 不要求 config 一定有这些字段；没有就用默认值（不破坏你的UI）
    enable = bool(getattr(cfg, "auto_brighten", True))
    if not enable:
        return gray8

    target_median = float(getattr(cfg, "bright_target_median", 110.0))  # 目标中位亮度
    max_gain = float(getattr(cfg, "bright_max_gain", 4.0))              # 最大增益，防噪声爆炸
    p_low = float(getattr(cfg, "bright_clip_low", 1.0))                 # 低端裁剪百分位
    p_high = float(getattr(cfg, "bright_clip_high", 99.0))              # 高端裁剪百分位
    use_clahe = bool(getattr(cfg, "bright_use_clahe", False))
    clahe_clip = float(getattr(cfg, "clahe_clip_limit", 2.0))
    clahe_grid = int(getattr(cfg, "clahe_grid", 8))
    gamma_dark = float(getattr(cfg, "bright_gamma_dark", 0.85))         # 暗图gamma(<1提亮暗部)
    dark_thresh = float(getattr(cfg, "bright_dark_thresh", 70.0))       # p50 低于此认为偏暗

    g = gray8.astype(np.float32)

    # 1) 先做百分位裁剪，抑制反光/极端高亮点对增益计算的干扰
    lo = np.percentile(g, p_low)
    hi = np.percentile(g, p_high)
    if hi <= lo + 1e-6:
        return gray8
    g = np.clip(g, lo, hi)
    g = (g - lo) * (255.0 / (hi - lo + 1e-6))

    # 2) 基于中位数的全局增益
    med = float(np.median(g))
    if med < 1e-6:
        med = 1.0
    gain = target_median / med
    gain = float(np.clip(gain, 1.0, max_gain))
    g = np.clip(g * gain, 0, 255)

    # 3) 暗图再加一层 gamma（只在确实偏暗时启用）
    if med < dark_thresh:
        # gamma < 1 提亮暗部
        inv = 1.0 / max(1e-6, gamma_dark)
        g = 255.0 * ((g / 255.0) ** inv)

    out = np.clip(g, 0, 255).astype(np.uint8)

    # 4) 可选 CLAHE（增强局部对比，有时对“边缘框”很有效）
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
        out = clahe.apply(out)

    return out


def _to_uint8_gray(img) -> np.ndarray:
    """
    img: BGR/RGB/Gray, uint8/uint16/float
    return: gray uint8
    """
    if img is None:
        return None

    if img.ndim == 3:
        # assume BGR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    if gray.dtype == np.uint8:
        return gray
    if gray.dtype == np.uint16:
        return (gray / 257).astype(np.uint8)
    g = gray.astype(np.float32)
    g = g - g.min()
    mx = g.max()
    if mx > 1e-6:
        g = g / mx
    return (g * 255.0).clip(0, 255).astype(np.uint8)

def _apply_det_tuning_bgr(bgr: np.ndarray, cfg) -> np.ndarray:
    """
    只用于检测阶段的图像增强，不用于最终提取保存。
    调节：亮度(V)、饱和度(S)、对比度(alpha)、gamma
    """
    if bgr is None:
        return bgr

    enable = bool(getattr(cfg, "det_tune_enable", True))
    if not enable:
        return bgr

    brightness = float(getattr(cfg, "det_brightness", 1.0))   # V倍增
    saturation = float(getattr(cfg, "det_saturation", 1.0))   # S倍增
    contrast = float(getattr(cfg, "det_contrast", 1.0))       # alpha
    gamma = float(getattr(cfg, "det_gamma", 1.0))             # gamma

    img = bgr

    # 统一到 uint8 处理（检测用）
    if img.dtype == np.uint16:
        img = (img / 257).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img.astype(np.float32), 0, 255).astype(np.uint8)

    # HSV 调 V/S（亮度/饱和度）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * brightness, 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 对比度（alpha）+ 轻微亮度偏置（这里不额外加beta，避免和brightness重复）
    if abs(contrast - 1.0) > 1e-6:
        img = cv2.convertScaleAbs(img, alpha=contrast, beta=0)

    # gamma：gamma<1 提亮暗部；gamma>1 压暗
    if abs(gamma - 1.0) > 1e-6:
        g = np.clip(gamma, 0.05, 5.0)
        lut = np.array([((i / 255.0) ** (1.0 / g)) * 255.0 for i in range(256)], dtype=np.uint8)
        img = cv2.LUT(img, lut)

    return img

def _apply_clahe_gray8(gray8: np.ndarray, cfg) -> np.ndarray:
    if gray8 is None or gray8.dtype != np.uint8:
        return gray8
    if not bool(getattr(cfg, "det_use_clahe", False)):
        return gray8
    clip = float(getattr(cfg, "clahe_clip_limit", 2.0))
    grid = int(getattr(cfg, "clahe_grid", 8))
    grid = max(2, min(grid, 32))
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return clahe.apply(gray8)

def resize_keep_h(pil_img: Image.Image, target_h: int):
    """PIL -> PIL, keep aspect ratio, return resized and meta (ow,oh,nw,nh)."""
    ow, oh = pil_img.size
    target_h = int(target_h)
    if target_h <= 0 or oh <= 0 or oh == target_h:
        return pil_img.copy(), (ow, oh, ow, oh)
    scale = target_h / float(oh)
    nw = max(1, int(round(ow * scale)))
    nh = max(1, int(round(oh * scale)))
    return pil_img.resize((nw, nh), resample=Image.BILINEAR), (ow, oh, nw, nh)


def _raw_to_preview_rgb8(path: str, cfg):
    """
    RAW -> preview rgb8 (uint8), and also provide extract bgr (uint8).
    这里保持最初思路：检测/提取都使用8bit（稳定、与原代码一致）。
    """
    if not _HAS_RAWPY:
        raise RuntimeError("rawpy not installed")

    use_wb = bool(getattr(cfg, "raw_use_camera_wb", True))
    out_bps = int(getattr(cfg, "raw_output_bps", 8))

    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=use_wb,
            no_auto_bright=True,
            gamma=(1, 1),          # 线性（更接近真实）
            output_bps=out_bps,
            bright=1.0,
        )

    # 强制转8bit用于cv2
    if rgb.dtype != np.uint8:
        # rawpy 16bit -> 8bit
        rgb8 = (rgb.astype(np.float32) / 256.0).clip(0, 255).astype(np.uint8)
    else:
        rgb8 = rgb

    bgr8 = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
    pil_rgb8 = Image.fromarray(rgb8, mode="RGB")
    meta = {"raw": True, "output_bps": out_bps}
    return pil_rgb8, bgr8, bgr8.copy(), meta


def load_image_ex(image_path: str, cfg):
    """
    返回：
      pil_rgb8: PIL RGB (for display)
      bgr_extract: cv2 BGR used for extraction (uint8)
      bgr_preview: cv2 BGR used for preview/debug (uint8)
      meta: dict
    """
    input_mode = str(getattr(cfg, "input_mode", "auto")).lower()
    prefer_raw = bool(getattr(cfg, "prefer_raw_linear", True))

    try_raw = (input_mode == "cr2") or (_is_raw(image_path) and prefer_raw)

    if try_raw:
        try:
            return _raw_to_preview_rgb8(image_path, cfg)
        except Exception as e:
            # RAW失败兜底
            raw_err = str(e)
    else:
        raw_err = None

    # PIL fallback
    try:
        pil_rgb8 = Image.open(image_path).convert("RGB")
        rgb8 = np.array(pil_rgb8, dtype=np.uint8)
        bgr = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
        meta = {"raw": False, "raw_err": raw_err}
        return pil_rgb8, bgr, bgr.copy(), meta
    except Exception as e:
        raise RuntimeError(f"读取失败：{e}, RAW兜底信息：{raw_err}")


def detect_regions_pair(det_bgr: np.ndarray, cfg):
    """
    【最初算法】Sobel + threshold + connected components
    输入：用于检测的 BGR（通常是缩小后的图）
    输出：
      debug: dict  (edges_u8, binary, boxes绘制等)
      ref_box_s: 4x2 int32 (缩小图坐标，上方)
      sample_box_s: 4x2 int32 (缩小图坐标，下方)
    """



    # 1) 先对检测图做可调增强（仅检测用）
    det_bgr2 = _apply_det_tuning_bgr(det_bgr, cfg)

    # 2) 转灰度
    gray8 = _to_uint8_gray(det_bgr2)
    if gray8 is None:
        return {"gray": None}, None, None

    # 3) 可选 CLAHE（仅检测用）
    gray8 = _apply_clahe_gray8(gray8, cfg)

    # 4) 可选 自动增亮（如果你之前已有 _auto_brighten_gray，继续用）
    if bool(getattr(cfg, "det_auto_brighten", True)):
        gray8 = _auto_brighten_gray(gray8, cfg)


    k = int(getattr(cfg, "sobel_ksize", 3))
    thr = int(getattr(cfg, "edge_thresh", 50))

    gx = cv2.Sobel(gray8, cv2.CV_64F, 1, 0, ksize=k)
    gy = cv2.Sobel(gray8, cv2.CV_64F, 0, 1, ksize=k)
    mag = np.sqrt(gx * gx + gy * gy)
    mag_u8 = (255.0 * mag / (mag.max() + 1e-8)).astype(np.uint8)

    _, binary = cv2.threshold(mag_u8, thr, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    dbg = {"mag": mag_u8, "binary": binary}

    if num_labels < 3:
        return dbg, None, None

    # ===== 工具函数：相交/包含判断 =====
    def box_aabb(box):
        xs = box[:, 0]; ys = box[:, 1]
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    def aabb_intersection_area(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0
        return (ix2 - ix1) * (iy2 - iy1)

    def aabb_area(a):
        x1, y1, x2, y2 = a
        return max(0, x2 - x1) * max(0, y2 - y1)

    def aabb_contains(outer, inner, margin=0):
        ox1, oy1, ox2, oy2 = outer
        ix1, iy1, ix2, iy2 = inner
        return (ox1 <= ix1 - margin and oy1 <= iy1 - margin and
                ox2 >= ix2 + margin and oy2 >= iy2 + margin)

    def rotated_rect_intersect(r1, r2):
        # r: (center(x,y), (w,h), angle)
        # 返回交集类型：0=无交，1=部分交，2=包含/完全交
        inter_type, _ = cv2.rotatedRectangleIntersection(r1, r2)
        return inter_type != 0

    def poly_contains(poly_outer, poly_inner):
        # 如果 inner 的四个角都在 outer 内 => 包含
        for p in poly_inner:
            if cv2.pointPolygonTest(poly_outer.astype(np.float32), (float(p[0]), float(p[1])), False) < 0:
                return False
        return True

    # ===== 1) 取 TopK 候选（而非 Top2）=====
    # 过滤掉太小的连通域
    H, W = gray8.shape[:2]
    min_area = int(getattr(cfg, "min_cc_area", 0.005) * (H * W))  # 默认 0.5% 画面面积
    areas = stats[1:, cv2.CC_STAT_AREA]  # skip bg
    idxs = np.argsort(areas)[::-1] + 1   # label ids sorted by area desc

    candidates = []
    K = int(getattr(cfg, "cc_topk", 10))  # 默认取前10个候选
    for lab in idxs[:K]:
        if stats[lab, cv2.CC_STAT_AREA] < min_area:
            continue
        mask = (labels == lab).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < min_area:
            continue

        rect = cv2.minAreaRect(cnt)                  # (center,(w,h),angle)
        box = cv2.boxPoints(rect).astype(np.int32)   # 4x2

        # 过滤细长噪声
        (cx, cy), (rw, rh), ang = rect
        if min(rw, rh) < 10:
            continue

        candidates.append({
            "lab": lab,
            "area": float(stats[lab, cv2.CC_STAT_AREA]),
            "rect": rect,
            "box": box,
            "aabb": box_aabb(box),
            "cy": float(cy),
            "cx": float(cx),
        })

    if len(candidates) < 2:
        return dbg, None, None

    # ===== 2) 在候选里找一对“互不交叉、互不包含”的最佳组合 =====
    best = None
    best_score = -1e18

    # 可选约束：希望两个框在图像中线附近、上下分布
    use_center_constraint = bool(getattr(cfg, "pair_center_constraint", True))
    center_x = W * 0.5
    center_band = float(getattr(cfg, "pair_center_band", 0.30)) * W  # 默认±30%宽度

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            A = candidates[i]
            B = candidates[j]

            # (a) 快速 AABB 相交检查
            inter = aabb_intersection_area(A["aabb"], B["aabb"])
            if inter > 0:
                # 进一步用旋转矩形精确判断（避免AABB误杀）
                if rotated_rect_intersect(A["rect"], B["rect"]):
                    continue

            # (b) 互不包含（先AABB包含，再多边形contains做确认）
            if aabb_contains(A["aabb"], B["aabb"], margin=2) and poly_contains(A["box"], B["box"]):
                continue
            if aabb_contains(B["aabb"], A["aabb"], margin=2) and poly_contains(B["box"], A["box"]):
                continue

            # (c) 可选：中心线约束（更符合你的拍摄习惯：两块色卡在中线上）
            if use_center_constraint:
                if abs(A["cx"] - center_x) > center_band or abs(B["cx"] - center_x) > center_band:
                    continue

            # (d) 上下分布：y中心差要足够大，避免同一块被拆成两块
            min_dy = float(getattr(cfg, "pair_min_dy", 0.12)) * H  # 默认至少 12% 高度差
            if abs(A["cy"] - B["cy"]) < min_dy:
                continue
            # (e) 尺寸一致性约束：面积比、宽高比接近（上下两块色卡尺寸应基本一致）
            (cax, cay), (aw, ah), _ = A["rect"]
            (cbx, cby), (bw, bh), _ = B["rect"]

            aw, ah = max(aw, 1e-6), max(ah, 1e-6)
            bw, bh = max(bw, 1e-6), max(bh, 1e-6)

            a_area = aw * ah
            b_area = bw * bh
            area_ratio = max(a_area, b_area) / max(min(a_area, b_area), 1e-6)

            a_ar = max(aw, ah) / min(aw, ah)  # aspect ratio >= 1
            b_ar = max(bw, bh) / min(bw, bh)

            # 可调阈值（建议默认就够用）
            max_area_ratio = float(getattr(cfg, "pair_max_area_ratio", 1.35))   # 面积最多差 35%
            max_ar_ratio   = float(getattr(cfg, "pair_max_ar_ratio", 1.25))     # 宽高比最多差 25%

            if area_ratio > max_area_ratio:
                continue
            if max(a_ar, b_ar) / max(min(a_ar, b_ar), 1e-6) > max_ar_ratio:
                continue


            # ===== 评分：优先选面积大、但惩罚离中心线过远、惩罚 y 太靠近 =====
            score = (A["area"] + B["area"])
            score -= 0.5 * (abs(A["cx"] - center_x) + abs(B["cx"] - center_x))
            score -= 3.0 * max(0.0, (min_dy - abs(A["cy"] - B["cy"])))  # 越接近惩罚越大

            if score > best_score:
                best_score = score
                best = (A, B)

    if best is None:
        return dbg, None, None

    A, B = best
    # 上下排序：y小的是上方 ref
    ref_box, sample_box = (A["box"], B["box"]) if A["cy"] < B["cy"] else (B["box"], A["box"])

    vis = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, [ref_box], True, (0, 255, 0), 2)
    cv2.polylines(vis, [sample_box], True, (255, 0, 0), 2)
    dbg["vis"] = vis

    return dbg, ref_box.astype(np.int32), sample_box.astype(np.int32)
