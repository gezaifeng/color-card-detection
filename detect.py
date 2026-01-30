import numpy as np
import cv2
from PIL import Image
from config import PipelineConfig

# 可选 RAW 支持
try:
    import rawpy  # pip install rawpy
    _HAS_RAWPY = True
except Exception:
    _HAS_RAWPY = False

def _read_raw_linear(path, cfg: PipelineConfig):
    """使用 rawpy 以线性（无伽马）方式解码 RAW，返回 PIL.Image RGB。"""
    if not _HAS_RAWPY:
        return None, "rawpy not installed"
    try:
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(
                use_camera_wb=bool(cfg.raw_use_camera_wb),
                no_auto_bright=True,
                gamma=(1, 1),                # 线性，无伽马
                output_bps=int(cfg.raw_output_bps),  # 8 或 16
                bright=1.0,
                user_flip=0
            )
        # rawpy 输出是 np.uint8 或 uint16 的 RGB
        if rgb.dtype != np.uint8:
            # 若为 16-bit，可线性压缩到 8-bit（简单右移或比例缩放）
            rgb = (rgb.astype(np.float32) / 256.0).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(rgb, mode="RGB")
        return pil, None
    except Exception as e:
        return None, str(e)

def load_image(path, cfg: PipelineConfig):
    """
    统一读取接口：
      - 若是 RAW（.cr2/.CR2/.nef 等），且 prefer_raw_linear=True 且 rawpy 可用 → 线性读取
      - 否则用 PIL 常规读取（JPEG/PNG/TIFF 等）
    返回 PIL.Image（RGB）
    """
    suffix = str(path).split(".")[-1].lower()
    is_raw = suffix in {"cr2", "nef", "arw", "dng", "raf", "rw2", "orf", "cr3"}
    if is_raw and cfg.prefer_raw_linear:
        pil, err = _read_raw_linear(path, cfg)
        if pil is not None:
            return pil
        # raw 失败则回退
        print(f"[RAW fallback] {err}")
    return Image.open(path).convert("RGB")

def resize_keep_h(im, target_h):
    orig_w, orig_h = im.size
    new_w = int(target_h * (orig_w / orig_h))
    return im.resize((new_w, target_h)), (orig_w, orig_h, new_w, target_h)

def detect_regions_pair(im_gray, cfg: PipelineConfig):
    """
    基于 Sobel + 阈值 + 连通域：
      - 取面积 top2 的连通域作为两块色卡
      - y 均值更小者为 ref_box，上方；另一者 sample_box
    返回：edges, ref_box(4x2), sample_box(4x2)
    """
    k = cfg.sobel_ksize
    gx = cv2.Sobel(im_gray, cv2.CV_64F, 1, 0, ksize=k)
    gy = cv2.Sobel(im_gray, cv2.CV_64F, 0, 1, ksize=k)
    mag = np.sqrt(gx**2 + gy**2)
    mag = np.uint8(255 * mag / (mag.max() + 1e-8))
    _, binary = cv2.threshold(mag, cfg.edge_thresh, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels < 3:
        return mag, None, None

    # 找面积 Top2
    areas = stats[1:, cv2.CC_STAT_AREA]
    top2_idx = np.argsort(areas)[-2:][::-1] + 1  # 加1跳过背景
    boxes = []
    for idx in top2_idx:
        mask = (labels == idx).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            rect = cv2.minAreaRect(cnts[0])
            box = cv2.boxPoints(rect).astype(int)
            boxes.append(box)

    if len(boxes) != 2:
        return mag, None, None

    y_mean = [np.mean(b[:, 1]) for b in boxes]
    ref_box, sample_box = (boxes[0], boxes[1]) if y_mean[0] < y_mean[1] else (boxes[1], boxes[0])
    return mag, ref_box, sample_box
