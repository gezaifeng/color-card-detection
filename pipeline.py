# colorcard_kit/pipeline.py
import os
import numpy as np
import cv2
from PIL import Image

from config import PipelineConfig
from detect import load_image, resize_keep_h, detect_regions_pair
from extract import extract_card_means
from features import build_features
from visualize import visualize_pair
from io_utils import out_path
from manual_select import select_two_rects

def process_single(image_path, input_dir, output_dir, cfg: PipelineConfig):
    """
    流程：
      1) 读取（支持 CR2 线性 postprocess）并缩放
      2) 自动检测上下两块；若失败并允许手动/或强制手动 → 交互框选
      3) 提取 (3,4,6)，构建特征（log_ratio/ratio/multi）
      4) 保存 npy 与可视化
    """
    # 读取（自动 RAW → 线性）
    im = load_image(image_path, cfg)  # PIL.Image RGB
    resized, (ow, oh, nw, nh) = resize_keep_h(im, cfg.target_height)
    scale_x, scale_y = ow / nw, oh / nh

    im_gray = np.array(resized.convert("L"))
    im_bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    ann = im_bgr.copy()

    # —— 自动检测（除非强制手动）
    ref_box = sample_box = None
    edges = None
    if not cfg.force_manual:
        edges, ref_box_s, sample_box_s = detect_regions_pair(im_gray, cfg)
        if ref_box_s is not None and sample_box_s is not None:
            ref_box = np.array([[int(x*scale_x), int(y*scale_y)] for x, y in ref_box_s])
            sample_box = np.array([[int(x*scale_x), int(y*scale_y)] for x, y in sample_box_s])

    # —— 手动回退（或强制手动）
    if ref_box is None or sample_box is None:
        if cfg.allow_manual or cfg.force_manual:
            ref_box, sample_box = select_two_rects(im_bgr, max_side=cfg.manual_downscale)
        if ref_box is None or sample_box is None:
            print(f"[Skip] Unable to get two regions (auto/manual): {image_path}")
            return None
        # 手动模式下 edges 为空，用 None 占位
        edges = edges if edges is not None else None

    # 标注区域框
    cv2.polylines(ann, [ref_box], True, (0, 255, 0), 4)     # 上方（绿）
    cv2.polylines(ann, [sample_box], True, (255, 0, 0), 4)  # 下方（蓝）

    # 提取 (3,4,6)；在 ann 上画红格与黄中心框
    ref_rgb_346    = extract_card_means(ann, ref_box, cfg, draw_grid=True)
    sample_rgb_346 = extract_card_means(ann, sample_box, cfg, draw_grid=True)

    # 构建特征
    X, extras = build_features(
        ref_rgb_346, sample_rgb_346,
        mode=cfg.feature_mode,
        per_image_channel_norm=cfg.per_image_channel_norm
    )
    ratio_346 = extras["ratio"]
    log_ratio_346 = extras["log_ratio"]

    # 保存 npy
    feat_tag = cfg.feature_mode
    feat_path = out_path(input_dir, output_dir, image_path, prefix=f"features_{feat_tag}_", ext="npy")
    np.save(feat_path, X.astype(np.float32))

    ref_path     = out_path(input_dir, output_dir, image_path, prefix="ref_",     suffix="346", ext="npy")
    sample_path  = out_path(input_dir, output_dir, image_path, prefix="sample_",  suffix="346", ext="npy")
    np.save(ref_path,     ref_rgb_346.astype(np.float32))
    np.save(sample_path,  sample_rgb_346.astype(np.float32))

    if cfg.save_extras:
        ratio_path = out_path(input_dir, output_dir, image_path, prefix="ratio_",     suffix="346", ext="npy")
        lgrt_path  = out_path(input_dir, output_dir, image_path, prefix="logratio_",  suffix="346", ext="npy")
        np.save(ratio_path, ratio_346.astype(np.float32))
        np.save(lgrt_path,  log_ratio_346.astype(np.float32))

    # 可视化
    vis_dir = os.path.join(output_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    vis_path = visualize_pair(
        image_path,
        edges=edges,
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
