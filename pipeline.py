# colorcard_kit/pipeline.py
import os
import numpy as np
import cv2
from PIL import Image
from config import PipelineConfig
from detect import load_image_ex, resize_keep_h, detect_regions_pair
from extract import extract_card_means
from features import build_features
from visualize import visualize_pair
from io_utils import out_path
from manual_select import select_two_rects


def _pick_edges_for_vis(edges):
    if edges is None:
        return None
    if isinstance(edges, dict):
        for k in ("edges", "edge", "canny", "img", "image", "binary", "gray", "mag", "vis"):
            if k in edges and isinstance(edges[k], np.ndarray):
                return edges[k]
        for v in edges.values():
            if isinstance(v, np.ndarray):
                return v
        return None
    if isinstance(edges, (tuple, list)) and len(edges) > 0:
        return edges[0]
    return edges


def _auto_white_balance_grayworld_bgr(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    if image_bgr.dtype != np.uint8:
        img = np.clip(image_bgr, 0, 255).astype(np.uint8)
    else:
        img = image_bgr

    b, g, r = cv2.split(img.astype(np.float32))
    mb, mg, mr = float(b.mean()), float(g.mean()), float(r.mean())
    mgray = (mb + mg + mr) / 3.0
    sb = mgray / (mb + 1e-6)
    sg = mgray / (mg + 1e-6)
    sr = mgray / (mr + 1e-6)
    out = cv2.merge([b * sb, g * sg, r * sr])
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_det_tuning_bgr_for_preview(bgr: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """
    复用你 UI 的检测增强参数（亮度/饱和度/对比度/gamma）做“手动框选预览/采样”增亮。
    不改 detect.py，只在这里做一份等价实现，避免模块依赖变化。
    """
    if bgr is None:
        return bgr

    enable = bool(getattr(cfg, "det_tune_enable", True))
    if not enable:
        return bgr

    brightness = float(getattr(cfg, "det_brightness", 1.0))
    saturation = float(getattr(cfg, "det_saturation", 1.0))
    contrast = float(getattr(cfg, "det_contrast", 1.0))
    gamma = float(getattr(cfg, "det_gamma", 1.0))

    img = bgr
    if img.dtype == np.uint16:
        img = (img / 257).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img.astype(np.float32), 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * brightness, 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if abs(contrast - 1.0) > 1e-6:
        img = cv2.convertScaleAbs(img, alpha=contrast, beta=0)

    if abs(gamma - 1.0) > 1e-6:
        g = float(np.clip(gamma, 0.05, 5.0))
        lut = np.array([((i / 255.0) ** (1.0 / g)) * 255.0 for i in range(256)], dtype=np.uint8)
        img = cv2.LUT(img, lut)

    return img


def _auto_brighten_bgr_vchannel(image_bgr: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """
    自动提亮：把 V 通道中位数拉到 target_median
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    enable = bool(getattr(cfg, "det_auto_brighten", True))
    if not enable:
        return image_bgr

    target_median = float(getattr(cfg, "bright_target_median", 110.0))
    max_gain = float(getattr(cfg, "bright_max_gain", 4.0))
    max_gain = max(1.0, max_gain)

    img = image_bgr
    if img.dtype != np.uint8:
        img = np.clip(img.astype(np.float32), 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[..., 2]
    med = float(np.median(v))
    if med < 1e-6:
        med = 1.0
    gain = float(np.clip(target_median / med, 1.0, max_gain))
    hsv[..., 2] = np.clip(v * gain, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _build_adjusted_bgr(im_bgr: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """
    统一生成“调节后图像”（用于全局采样或手动采样覆盖）
    顺序：白平衡 -> UI检测增强 -> 自动提亮
    """
    out = _auto_white_balance_grayworld_bgr(im_bgr)
    out = _apply_det_tuning_bgr_for_preview(out, cfg)
    out = _auto_brighten_bgr_vchannel(out, cfg)
    return out


def process_single(image_path, input_dir, output_dir, cfg: PipelineConfig):
    """
    新增功能（不改其它逻辑）：
    - manual_sample_from_adjusted=True 时：
      仅当自动识别失败进入手动框选，框选显示/采样/保存都使用“调节后图像”
    """

    im = load_image_ex(image_path, cfg)
    if isinstance(im, (tuple, list)):
        im = im[0]

    resized, (ow, oh, nw, nh) = resize_keep_h(im, cfg.target_height)
    scale_x, scale_y = ow / nw, oh / nh

    im_small_bgr = cv2.cvtColor(np.array(resized), cv2.COLOR_RGB2BGR)
    im_bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

    # 预先准备调节图（不改变现有路径，只是备着给开关用）
    adjusted_bgr = _build_adjusted_bgr(im_bgr, cfg)

    # ========= 全局采样开关（你原来的功能，保持一致） =========
    use_global_adj = bool(getattr(cfg, "sample_from_adjusted", False))
    sample_src_bgr = adjusted_bgr if use_global_adj else im_bgr

    # ann 用于可视化（跟随当前 sample_src）
    ann = sample_src_bgr.copy()

    # —— 自动检测（除非强制手动）
    ref_box = sample_box = None
    edges = None
    if not cfg.force_manual:
        edges, ref_box_s, sample_box_s = detect_regions_pair(im_small_bgr, cfg)
        if ref_box_s is not None and sample_box_s is not None:
            ref_box = np.array([[int(x * scale_x), int(y * scale_y)] for x, y in ref_box_s], dtype=np.int32)
            sample_box = np.array([[int(x * scale_x), int(y * scale_y)] for x, y in sample_box_s], dtype=np.int32)

    # —— 手动回退（或强制手动）
    manual_used = False
    if ref_box is None or sample_box is None:
        if cfg.allow_manual or cfg.force_manual:
            # NEW: 手动专用开关：自动失败进入手动时，用调节后图像来框选（更亮）
            use_manual_adj = bool(getattr(cfg, "manual_sample_from_adjusted", False))
            manual_view = adjusted_bgr if use_manual_adj else im_bgr

            ref_box, sample_box = select_two_rects(manual_view, max_side=cfg.manual_downscale)
            manual_used = True

            if ref_box is not None:
                ref_box = np.asarray(ref_box, dtype=np.int32)
            if sample_box is not None:
                sample_box = np.asarray(sample_box, dtype=np.int32)

        if ref_box is None or sample_box is None:
            print(f"[Skip] Unable to get two regions (auto/manual): {image_path}")
            return None

    # NEW: 如果确实走了手动，并且手动专用开关打开，则覆盖采样源为调节后图像
    if manual_used and bool(getattr(cfg, "manual_sample_from_adjusted", False)):
        sample_src_bgr = adjusted_bgr
        ann = sample_src_bgr.copy()  # 可视化也对应调节后图像，避免你误解“保存没变”

    # 标注区域框（保持你原来的可视化）
    cv2.polylines(ann, [ref_box], True, (0, 255, 0), 4)
    cv2.polylines(ann, [sample_box], True, (255, 0, 0), 4)

    # 关键：采样必须在“干净副本”上进行
    clean_for_sampling = sample_src_bgr.copy()
    ref_rgb_346 = extract_card_means(clean_for_sampling, ref_box, cfg, draw_grid=False)
    sample_rgb_346 = extract_card_means(clean_for_sampling, sample_box, cfg, draw_grid=False)

    # 再在 ann 上画网格（只为了可视化）
    _ = extract_card_means(ann, ref_box, cfg, draw_grid=True)
    _ = extract_card_means(ann, sample_box, cfg, draw_grid=True)

    # 特征构建与保存（保持原逻辑）
    X, extras = build_features(
        ref_rgb_346, sample_rgb_346,
        mode=cfg.feature_mode,
        per_image_channel_norm=cfg.per_image_channel_norm
    )
    ratio_346 = extras["ratio"]
    log_ratio_346 = extras["log_ratio"]

    feat_tag = cfg.feature_mode
    feat_path = out_path(input_dir, output_dir, image_path, prefix=f"features_{feat_tag}_", ext="npy")
    np.save(feat_path, X.astype(np.float32))

    ref_path = out_path(input_dir, output_dir, image_path, prefix="ref_", suffix="346", ext="npy")
    sample_path = out_path(input_dir, output_dir, image_path, prefix="sample_", suffix="346", ext="npy")
    np.save(ref_path, ref_rgb_346.astype(np.float32))
    np.save(sample_path, sample_rgb_346.astype(np.float32))

    if cfg.save_extras:
        ratio_path = out_path(input_dir, output_dir, image_path, prefix="ratio_", suffix="346", ext="npy")
        lgrt_path = out_path(input_dir, output_dir, image_path, prefix="logratio_", suffix="346", ext="npy")
        np.save(ratio_path, ratio_346.astype(np.float32))
        np.save(lgrt_path, log_ratio_346.astype(np.float32))

    # 可视化（保持原逻辑）
    vis_dir = os.path.join(output_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    vis_path = visualize_pair(
        image_path,
        edges=_pick_edges_for_vis(edges),
        annotated_bgr=ann,
        ref_rgb_346=ref_rgb_346,
        sample_rgb_346=sample_rgb_346,
        ratio_346=ratio_346,
        log_ratio_346=log_ratio_346,
        feature_mode=cfg.feature_mode,
        out_dir=vis_dir
    )

    return {
        "features": feat_path,
        "ref_346": ref_path,
        "sample_346": sample_path,
        "vis": vis_path
    }
