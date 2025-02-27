import numpy as np
import cv2
import rawpy  # 处理 CR2
from PIL import Image
import matplotlib
import numpy as np
import cv2
matplotlib.use('TkAgg')  # 让 plt.show() 正常显示
import matplotlib.pyplot as plt
import os
import math
# ✅ 输入和输出文件夹
input_dir = "D:\Desktop\pictures-1.20\G1\deep pink"
output_dir = "D:\\Desktop\\extracted"
os.makedirs(output_dir, exist_ok=True)

# ✅ 颜色矩阵网格大小
GRID_ROWS, GRID_COLS = 4, 6
TARGET_HEIGHT = 512  # 目标压缩高度，保持宽高比缩放
LONG_EDGE_CROP_RATIO = 0.03
SHORT_EDGE_CROP_RATIO = 0.06
SAMPLE_COUNT = 100  # ✅ 可自定义的采样点数量

def find_images(directory):
    """ ✅ 递归搜索所有图片，包括 CR2 """
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('jpg', 'png', 'jpeg', 'cr2')):
                image_paths.append(os.path.join(root, file))
    return image_paths


def get_output_path(image_path, file_ext):
    """ ✅ 生成输出路径，保持子目录结构 """
    relative_path = os.path.relpath(image_path, input_dir)
    relative_dir = os.path.dirname(relative_path)
    output_folder = os.path.join(output_dir, relative_dir)
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(output_folder, f"rgb_{filename}.{file_ext}")


def load_image(image_path):
    """ ✅ 兼容 JPG/PNG 和 CR2 格式 """
    if image_path.lower().endswith('.cr2'):
        with rawpy.imread(image_path) as raw:
            rgb_image = raw.postprocess(output_bps=8)
            return Image.fromarray(rgb_image)
    else:
        return Image.open(image_path)


def resize_image(im):
    """ ✅ 按目标高度缩放图片，保持原始宽高比 """
    orig_w, orig_h = im.size
    aspect_ratio = orig_w / orig_h
    new_w = int(TARGET_HEIGHT * aspect_ratio)
    return im.resize((new_w, TARGET_HEIGHT), Image.LANCZOS), orig_w, orig_h, new_w, TARGET_HEIGHT


def detect_edges(im_gray):
    """ ✅ 计算 Sobel 梯度 + 二值化处理 + 连通区域分析 """
    grad_x = cv2.Sobel(im_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(im_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_magnitude = np.uint8(255 * grad_magnitude / grad_magnitude.max())

    _, binary = cv2.threshold(grad_magnitude, 50, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_region = (labels == largest_label).astype(np.uint8) * 255

    contours, _ = cv2.findContours(largest_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rect).astype(int)

    return grad_magnitude, box


def map_coords_to_original(box, scale_x, scale_y):
    """ ✅ 将缩小图片上的坐标映射回原始图片 """
    return np.array([[int(x * scale_x), int(y * scale_y)] for x, y in box])


def shrink_card_region(mapped_box):
    """ ✅ 在识别到的色卡范围内缩进长边 5%，短边 3% """
    x_min, y_min = np.min(mapped_box, axis=0)
    x_max, y_max = np.max(mapped_box, axis=0)

    region_w, region_h = x_max - x_min, y_max - y_min

    # **计算缩进量**
    shrink_w = int(region_w * LONG_EDGE_CROP_RATIO)
    shrink_h = int(region_h * SHORT_EDGE_CROP_RATIO)

    # **缩进后的坐标**
    new_x_min = x_min + shrink_w
    new_y_min = y_min + shrink_h
    new_x_max = x_max - shrink_w
    new_y_max = y_max - shrink_h

    return np.array([[new_x_min, new_y_min], [new_x_max, new_y_min],
                     [new_x_max, new_y_max], [new_x_min, new_y_max]])


def extract_rgb_from_mapped_region(image, mapped_box):
    """ ✅ 在缩进后的色卡区域内进行 4×6 分割，并在每个色块内随机采样 SAMPLE_COUNT 个像素 """
    mapped_box = shrink_card_region(mapped_box)

    x_min, y_min = np.min(mapped_box, axis=0)
    x_max, y_max = np.max(mapped_box, axis=0)

    region_w, region_h = x_max - x_min, y_max - y_min
    cell_w, cell_h = region_w // GRID_COLS, region_h // GRID_ROWS

    rgb_samples = np.zeros((GRID_ROWS, GRID_COLS, SAMPLE_COUNT, 3), dtype=np.float32)

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x1, y1 = x_min + col * cell_w, y_min + row * cell_h
            x2, y2 = x1 + cell_w, y1 + cell_h

            # **计算中心区域**
            center_x1 = int(x1 + 0.25 * cell_w)
            center_x2 = int(x1 + 0.75 * cell_w)
            center_y1 = int(y1 + 0.25 * cell_h)
            center_y2 = int(y1 + 0.75 * cell_h)

            center_patch = image[center_y1:center_y2, center_x1:center_x2]

            # **展开像素点**
            pixels = center_patch.reshape(-1, 3)

            # **随机采样 `SAMPLE_COUNT` 个像素**
            if pixels.shape[0] >= SAMPLE_COUNT:
                selected_pixels = pixels[np.random.choice(pixels.shape[0], SAMPLE_COUNT, replace=False)]
            else:
                selected_pixels = np.tile(pixels, (SAMPLE_COUNT // pixels.shape[0] + 1, 1))[:SAMPLE_COUNT]

            rgb_samples[row, col] = selected_pixels

            # **绘制分割线**
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(image, (center_x1, center_y1), (center_x2, center_y2), (255, 0, 0), 2)

    return rgb_samples


def process_image(image_path):
    """ ✅ 处理单张图片 """
    im = load_image(image_path)
    resized_im, orig_w, orig_h, new_w, new_h = resize_image(im)
    scale_x, scale_y = orig_w / new_w, orig_h / new_h

    im_gray = np.array(resized_im.convert('L'))
    im_array = np.array(im)

    edges, box = detect_edges(im_gray)
    mapped_box = map_coords_to_original(box, scale_x, scale_y)

    annotated_image = im_array.copy()
    cv2.polylines(annotated_image, [mapped_box], isClosed=True, color=(0, 255, 0), thickness=5)

    rgb_array = extract_rgb_from_mapped_region(annotated_image, mapped_box)

    npy_path = get_output_path(image_path, "npy")
    np.save(npy_path, rgb_array.astype(np.float32))

    print(f"✅ 颜色数据已保存至: {npy_path}")

    visualize_results(image_path, edges, annotated_image, npy_path)


def visualize_results(image_path, edges, annotated_image, npy_file):
    """ ✅ 可视化四张图 """
    colors = np.load(npy_file)  # 读取 RGB 采样数据 (4,6,SAMPLE_COUNT,3)
    colors = np.clip(colors / 255.0, 0, 1)  # 归一化

    original_image = np.array(load_image(image_path))  # 确保 RGB 正确显示

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    # **第一张图：原始图像**
    ax[0].imshow(original_image)
    image_filename = os.path.basename(image_path)
    ax[0].set_title(f"Original Image: {image_filename}")
    ax[0].axis("off")

    # **第二张图：边缘检测**
    ax[1].imshow(edges, cmap='gray')
    ax[1].set_title("Edge Detection")
    ax[1].axis("off")

    # **第三张图：色卡分割 & 颜色标记**
    ax[2].imshow(annotated_image)
    ax[2].set_title("Color Grid on Original Image")
    ax[2].axis("off")

    # **第四张图：显示 4×6 色块，每个色块由 10×10 采样点组成**
    ax[3].set_xticks([])
    ax[3].set_yticks([])

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            sample_colors = colors[row, col]  # 100 组 RGB 值 (100,3)
            grid_size = 10  # 10x10 网格
            reshaped_grid = sample_colors.reshape((grid_size, grid_size, 3))  # 变为 (10,10,3)

            # **绘制色块**
            ax[3].imshow(reshaped_grid, extent=(col, col + 1, GRID_ROWS - row - 1, GRID_ROWS - row))

    ax[3].set_xlim(0, GRID_COLS)
    ax[3].set_ylim(0, GRID_ROWS)
    ax[3].set_title("Extracted RGB Grid (10×10 per Block)")

    plt.tight_layout()
    plt.show()



def batch_process_images():
    """ ✅ 递归处理所有子文件夹中的图片 """
    images = find_images(input_dir)
    for img_path in images:
        process_image(img_path)


batch_process_images()
