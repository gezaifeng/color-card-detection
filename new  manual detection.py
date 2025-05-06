# ✅ colorcard_graybar_extract.py

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
# 1️⃣ 确保 Matplotlib 运行在交互模式
matplotlib.use("TkAgg")  # 或 'Qt5Agg'，保证 GUI 支持
import os
import rawpy  # 如果你使用 CR2 文件，否则可注释掉

# === 全局配置 ===
GRID_ROWS, GRID_COLS = 4, 6
TARGET_HEIGHT = 512
# --- 配置裁剪参数（相对比例） ---
# --- 独立裁剪参数（按需要自定义） ---
# ---灰度裁剪范围----
GRAY_CROP_TOP_RATIO = 0.15
GRAY_CROP_BOTTOM_RATIO = 0.15
GRAY_CROP_LEFT_RATIO = 0.045
GRAY_CROP_RIGHT_RATIO = 0.025
# ---色块裁剪范围----
LONG_EDGE_CROP_RATIO = 0.03
SHORT_EDGE_CROP_RATIO = 0.06
SAMPLE_COUNT = 100
GRAY_SEGMENTS = 16  # 灰度条分段数，可调整

input_dir = "D:\Desktop\picture-4.7\\NEW"
output_dir = "D:\Desktop\picture-4.7\混合浓度1-npy"
os.makedirs(output_dir, exist_ok=True)

# === 工具函数 ===
def find_images(directory):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('jpg', 'png', 'jpeg', 'cr2')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def get_output_path(image_path, file_ext, suffix=""):
    relative_path = os.path.relpath(image_path, input_dir)
    relative_dir = os.path.dirname(relative_path)
    output_folder = os.path.join(output_dir, relative_dir)
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.splitext(os.path.basename(image_path))[0]
    suffix_str = f"_{suffix}" if suffix else ""
    return os.path.join(output_folder, f"rgb_{filename}{suffix_str}.{file_ext}")
def load_image(image_path):
    if image_path.lower().endswith('.cr2'):
        with rawpy.imread(image_path) as raw:
            rgb_image = raw.postprocess(output_bps=8)
            return Image.fromarray(rgb_image)
    else:
        return Image.open(image_path)

def resize_image(im):
    orig_w, orig_h = im.size
    aspect_ratio = orig_w / orig_h
    new_w = int(TARGET_HEIGHT * aspect_ratio)
    return im.resize((new_w, TARGET_HEIGHT), Image.Resampling.LANCZOS), orig_w, orig_h, new_w, TARGET_HEIGHT

def map_coords_to_original(box, scale_x, scale_y):
    if box is None:
        return None
    return np.array([[int(x * scale_x), int(y * scale_y)] for x, y in box])

# === 核心处理函数 ===
def select_region(image, window_name="Select Region", max_width=1280, max_height=720):
    """
    自动缩放图像后进行 ROI 框选，并将选区坐标映射回原图尺寸。
    返回：四点 box，shape (4, 2)
    """
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)))

    roi = cv2.selectROI(window_name, resized)
    cv2.destroyWindow(window_name)

    x, y, rw, rh = [int(val / scale) for val in roi]
    box = np.array([[x, y], [x + rw, y], [x + rw, y + rh], [x, y + rh]])

    return box

def shrink_card_region(mapped_box):
    x_min, y_min = np.min(mapped_box, axis=0)
    x_max, y_max = np.max(mapped_box, axis=0)
    region_w, region_h = x_max - x_min, y_max - y_min
    shrink_w = int(region_w * LONG_EDGE_CROP_RATIO)
    shrink_h = int(region_h * SHORT_EDGE_CROP_RATIO)
    new_x_min = x_min + shrink_w
    new_y_min = y_min + shrink_h
    new_x_max = x_max - shrink_w
    new_y_max = y_max - shrink_h
    return np.array([[new_x_min, new_y_min], [new_x_max, new_y_min], [new_x_max, new_y_max], [new_x_min, new_y_max]])

def extract_rgb_from_mapped_region(image, mapped_box):
    mapped_box = shrink_card_region(mapped_box)
    x_min, y_min = np.min(mapped_box, axis=0)
    x_max, y_max = np.max(mapped_box, axis=0)
    region_w, region_h = x_max - x_min, y_max - y_min
    cell_w, cell_h = region_w // GRID_COLS, region_h // GRID_ROWS
    rgb_samples = np.zeros((GRID_ROWS, GRID_COLS, SAMPLE_COUNT, 3), dtype=np.float32)
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x1 = x_min + col * cell_w
            y1 = y_min + row * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h
            center_patch = image[y1 + cell_h//4:y2 - cell_h//4, x1 + cell_w//4:x2 - cell_w//4]
            pixels = center_patch.reshape(-1, 3)
            if pixels.shape[0] >= SAMPLE_COUNT:
                selected = pixels[np.random.choice(pixels.shape[0], SAMPLE_COUNT, replace=False)]
            else:
                selected = np.tile(pixels, (SAMPLE_COUNT // pixels.shape[0] + 1, 1))[:SAMPLE_COUNT]
            rgb_samples[row, col] = selected
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return rgb_samples

def extract_graybar_from_region(image, mapped_box2, n_segments=4):
    """
    从灰度条区域中等分提取每段的 RGB 平均值，带裁剪缩进。
    返回：
        gray_means: shape (n_segments, 3)
        gray_vis: 灰度条图像带分段红线
    """


    # --- 提取坐标范围 ---
    x_min, y_min = np.min(mapped_box2, axis=0)
    x_max, y_max = np.max(mapped_box2, axis=0)

    # --- 裁剪区域 ---
    width = x_max - x_min
    height = y_max - y_min

    x_min += int(width * GRAY_CROP_LEFT_RATIO)
    x_max -= int(width * GRAY_CROP_RIGHT_RATIO)
    y_min += int(height * GRAY_CROP_TOP_RATIO)
    y_max -= int(height * GRAY_CROP_BOTTOM_RATIO)

    # --- 裁剪区域 ---
    gray_strip = image[y_min:y_max, x_min:x_max]
    region_h, region_w = gray_strip.shape[:2]
    seg_w = region_w // n_segments

    gray_means = np.zeros((n_segments, 3), dtype=np.float32)

    for i in range(n_segments):
        x1 = i * seg_w
        x2 = region_w if i == n_segments - 1 else (i + 1) * seg_w
        region = gray_strip[:, x1:x2]
        h, w = region.shape[:2]

        # 中心 20% 区域范围
        cx1 = int(w * 0.4)
        cx2 = int(w * 0.6)
        cy1 = int(h * 0.4)
        cy2 = int(h * 0.6)

        center_region = region[cy1:cy2, cx1:cx2]
        pixels = center_region.reshape(-1, 3)

        # 选取9个像素点
        if pixels.shape[0] < 9:
            sampled = np.tile(pixels, (9 // pixels.shape[0] + 1, 1))[:9]
        else:
            sampled = pixels[np.random.choice(pixels.shape[0], 9, replace=False)]

        gray_means[i] = sampled.mean(axis=0)

    # 可视化：分段红线
    gray_vis = gray_strip.copy()
    for i in range(1, n_segments):
        cv2.line(gray_vis, (i * seg_w, 0), (i * seg_w, region_h), (0, 0, 255), 2)

    return gray_means, gray_vis



def apply_gray_correction(rgb_array, gray_targets, gray_means):
    """
    根据灰度条拟合结果矫正RGB数组。
    参数：
        rgb_array: shape (4,6,N,3)
        gray_targets: shape (n_segments,)，理想灰度值，如 [0, 19.6, 39.2, 58.8]
        gray_means: shape (n_segments, 3)，每段的 RGB 平均值
    返回：
        rgb_array_corrected: shape 与 rgb_array 相同，值为矫正后RGB
    """
    rgb_array = rgb_array.astype(np.float32)
    corrected = rgb_array.copy()

    # 分别拟合 R/G/B 三个通道
    for ch in range(3):
        y = gray_means[:, ch]
        x = gray_targets
        if len(x) != len(y):
            raise ValueError("灰度目标值与实际样本数量不一致")
        slope, intercept = np.polyfit(x, y, 1)
        if slope == 0:
            slope = 1e-5  # 避免除以0
        # 反向校正：将样本从“测量值”映射回“理想灰度值”
        corrected[..., ch] = (rgb_array[..., ch] - intercept) / slope

    # 限制范围
    corrected = np.clip(corrected, 0, 255)
    return corrected.astype(np.float32)

def visualize_results(image_path, edges, annotated_image, npy_file,
                      gray_samples=None, gray_vis=None,
                      corrected_colors=None):
    """
    可视化六张图：
    0 - 边缘图
    1 - 标注图
    2 - 原始色块
    3 - 灰度条色块（灰度样本块）
    4 - 灰度拟合曲线（替代灰度图像）
    5 - 矫正后色块（可选）
    """
    colors = np.load(npy_file)
    colors = np.clip(colors / 255.0, 0, 1)

    show_gray = gray_samples is not None
    show_corrected = corrected_colors is not None

    n_plots = 3
    if show_gray:
        n_plots += 2
    if show_corrected:
        n_plots += 1

    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols
    fig, ax = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    ax = ax.flatten()

    idx = 0

    # ① 边缘图
    ax[idx].imshow(edges, cmap='gray')
    ax[idx].set_title("Edge Detection")
    ax[idx].axis("off")
    idx += 1

    # ② 标注图
    ax[idx].imshow(annotated_image)
    ax[idx].set_title("Annotated Detection")
    ax[idx].axis("off")
    idx += 1

    # ③ 原始色块图
    ax[idx].set_xticks([])
    ax[idx].set_yticks([])
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            sample = colors[row, col][:100].reshape((10, 10, 3))
            ax[idx].imshow(sample, extent=(col, col + 1, GRID_ROWS - row - 1, GRID_ROWS - row))
    ax[idx].set_xlim(0, GRID_COLS)
    ax[idx].set_ylim(0, GRID_ROWS)
    ax[idx].set_title("Original RGB Grid")
    idx += 1

    if show_gray:
        # ④ 灰度条图像 + 实际采样区域可视化（绿色框）
        gray_vis_copy = gray_vis.copy()
        region_h, region_w = gray_vis_copy.shape[:2]
        n_segments = gray_samples.shape[0]
        seg_w = region_w // n_segments

        for i in range(n_segments):
            x1 = i * seg_w
            x2 = region_w if i == n_segments - 1 else (i + 1) * seg_w

            cx1 = x1 + int((x2 - x1) * 0.4)
            cx2 = x1 + int((x2 - x1) * 0.6)
            cy1 = int(region_h * 0.4)
            cy2 = int(region_h * 0.6)

            cv2.rectangle(gray_vis_copy, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)

        ax[idx].imshow(gray_vis_copy)
        ax[idx].set_title("Gray Strip Sampling Range")
        ax[idx].axis("off")
        idx += 1

        # ⑤ 拟合曲线图（替代灰度图像）
        x_fit = np.linspace(0, 255, 100)
        colors_line = ['r', 'g', 'b']
        channel_names = ['Red', 'Green', 'Blue']
        gray_targets = np.linspace(0, len(gray_samples) - 1, len(gray_samples)) * (255 / 13)

        for ch in range(3):
            y = gray_samples[:, ch]
            x = gray_targets
            slope, intercept = np.polyfit(x, y, 1)
            y_fit = slope * x_fit + intercept
            ax[idx].plot(x_fit, y_fit, color=colors_line[ch], label=f"{channel_names[ch]} Fit")
            ax[idx].scatter(x, y, color=colors_line[ch], edgecolors='k', s=50, label=f"{channel_names[ch]} Sample")
            ax[idx].text(10, 250 - ch * 30,
                         f"{channel_names[ch]}: y={slope:.2f}x+{intercept:.2f}",
                         color=colors_line[ch])
        ax[idx].set_xlim(0, 255)
        ax[idx].set_ylim(0, 260)
        ax[idx].set_xlabel("Ideal Gray Value")
        ax[idx].set_ylabel("Measured Channel Value")
        ax[idx].set_title("Gray Bar Fit Curve")
        ax[idx].legend()
        ax[idx].grid(True)
        idx += 1

    # ⑥ 矫正后色块图
    if show_corrected:
        ax[idx].set_xticks([]); ax[idx].set_yticks([])
        colors_corr = np.clip(np.array(corrected_colors) / 255.0, 0, 1)
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                sample = colors_corr[row, col][:100].reshape((10, 10, 3))
                ax[idx].imshow(sample, extent=(col, col + 1, GRID_ROWS - row - 1, GRID_ROWS - row))
        ax[idx].set_xlim(0, GRID_COLS)
        ax[idx].set_ylim(0, GRID_ROWS)
        ax[idx].set_title("Corrected RGB Grid")
        idx += 1

    # 清理多余子图
    for i in range(idx, len(ax)):
        fig.delaxes(ax[i])

    plt.tight_layout()
    save_path = os.path.join(output_dir, "vis", os.path.splitext(os.path.basename(image_path))[0] + "_vis.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()



def apply_gray_correction(rgb_array, gray_targets, gray_means):
    """
    使用灰度条的拟合结果对色卡 RGB 数组进行通道线性矫正。
    参数：
        rgb_array: 原始提取色卡样本，shape (4, 6, N, 3)
        gray_targets: shape (n_segments,)，理想灰度值
        gray_means: shape (n_segments, 3)，实测RGB均值
    返回：
        rgb_array_corrected: 同 shape，矫正后的 RGB 数组
    """
    rgb_array = rgb_array.astype(np.float32)
    corrected = rgb_array.copy()

    for ch in range(3):  # 分别处理 R/G/B 通道
        y = gray_means[:, ch]
        x = gray_targets
        if len(x) != len(y):
            raise ValueError("灰度目标值与通道样本数量不匹配")
        slope, intercept = np.polyfit(x, y, 1)
        if abs(slope) < 1e-5:
            slope = 1e-5  # 防止除以0
        corrected[..., ch] = (rgb_array[..., ch] - intercept) / slope

    corrected = np.clip(corrected, 0, 255)
    return corrected.astype(np.float32)


def process_image(image_path):
    im = load_image(image_path)
    im_array = np.array(im)

    # ① 手动选取色卡区域
    print("请用鼠标框选【色卡区域】（4x6 色块）")
    mapped_box = select_region(im_array, "Select Color Card Region")


    # ② 手动选取灰度条区域
    print("请用鼠标框选【灰度条区域】")
    mapped_box2 = select_region(im_array, "Select Gray Bar Region")
    

    # === 绘图区域 ===
    annotated_image = im_array.copy()
    cv2.polylines(annotated_image, [mapped_box], isClosed=True, color=(0, 255, 0), thickness=3)
    cv2.polylines(annotated_image, [mapped_box2], isClosed=True, color=(255, 0, 0), thickness=3)

    # === 色块提取 ===
    rgb_array = extract_rgb_from_mapped_region(annotated_image, mapped_box)
    npy_path = get_output_path(image_path, "npy", suffix="origin")
    np.save(npy_path, rgb_array)

    # === 灰度条提取 + 校正 ===
    gray_means, gray_vis = extract_graybar_from_region(im_array, mapped_box2, GRAY_SEGMENTS)
    gray_targets = np.linspace(0, GRAY_SEGMENTS - 1, GRAY_SEGMENTS) * (255 / 13)
    rgb_array_corrected = apply_gray_correction(rgb_array, gray_targets, gray_means)
    final_rgb_array = rgb_array_corrected
    npy_path_corrected = get_output_path(image_path, "npy", suffix="corrected")
    np.save(npy_path_corrected, final_rgb_array)
    filename = os.path.basename(image_path)
    print(f"✅ [{filename}] 已完成灰度矫正")

    # === 可视化输出 ===
    edges = np.zeros_like(cv2.cvtColor(im_array, cv2.COLOR_RGB2GRAY))  # 占位图，已不用边缘图
    visualize_results(
        image_path,
        edges,
        annotated_image,
        npy_path_corrected,
        gray_samples=gray_means,
        gray_vis=gray_vis,
        corrected_colors=rgb_array_corrected
    )





def batch_process_images():
    images = find_images(input_dir)
    for img_path in images:
        process_image(img_path)

if __name__ == "__main__":
    batch_process_images()
