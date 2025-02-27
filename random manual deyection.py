import numpy as np
import cv2
import rawpy  # 处理 CR2
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')  # 让 plt.show() 正常显示
import matplotlib.pyplot as plt
import os

# ✅ 颜色矩阵网格大小
GRID_ROWS, GRID_COLS = 4, 6
DISPLAY_MAX_SIZE = 1000  # ✅ 限制显示窗口最大宽/高
SAMPLE_COUNT = 100  # ✅ 采样数量，可自定义
# ✅ 变量存储鼠标绘制矩形区域
CENTER_AREA_RATIO = 0.50  # ✅ 选取 50% 面积的范围
rect_start = None
rect_end = None
drawing = False


def convert_cr2_to_jpg(cr2_path):
    """ ✅ 将 CR2 转换为 JPG 并返回 JPG 路径 """
    jpg_path = cr2_path.replace(".cr2", ".jpg").replace(".CR2", ".jpg")
    with rawpy.imread(cr2_path) as raw:
        rgb_image = raw.postprocess(output_bps=8)
        img = Image.fromarray(rgb_image)
        img.save(jpg_path, "JPEG", quality=100)
    return jpg_path


def resize_for_display(image):
    """ ✅ 自适应缩放图片，保证不会超出窗口 """
    h, w = image.shape[:2]
    scale = min(DISPLAY_MAX_SIZE / max(h, w), 1.0)  # 计算缩放比例
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h)), scale


def select_roi(event, x, y, flags, param):
    """ ✅ 鼠标事件回调函数：用于绘制矩形 """
    global rect_start, rect_end, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        rect_start = (x, y)
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        rect_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        rect_end = (x, y)
        drawing = False


def get_manual_selection(image):
    """ ✅ 让用户在图片上绘制矩形选取色卡区域 """
    global rect_start, rect_end

    temp_image, scale = resize_for_display(image)  # ✅ 先缩放图片
    cv2.namedWindow("Select Region")
    cv2.setMouseCallback("Select Region", select_roi)

    while True:
        img_display = temp_image.copy()

        if rect_start and rect_end:
            cv2.rectangle(img_display, rect_start, rect_end, (0, 0, 255), 2)

        cv2.imshow("Select Region", img_display)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # 按 Enter 确认选区
            break

    cv2.destroyAllWindows()

    if rect_start and rect_end:
        # ✅ 把选区放大到原始尺寸
        x1, y1 = int(rect_start[0] / scale), int(rect_start[1] / scale)
        x2, y2 = int(rect_end[0] / scale), int(rect_end[1] / scale)
        return x1, y1, x2, y2
    else:
        return None


def extract_rgb_from_grid(image, x1, y1, x2, y2, sample_count=SAMPLE_COUNT):
    """ ✅ 在手动选择的区域内进行 4×6 分割，并在中心 10% 面积范围内随机采样 `n` 个 RGB 点 """
    region = image[y1:y2, x1:x2]  # **BGR 格式**
    region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)  # **转换为 RGB**

    h, w = region.shape[:2]
    cell_h, cell_w = h // GRID_ROWS, w // GRID_COLS

    rgb_values = np.zeros((GRID_ROWS, GRID_COLS, sample_count, 3), dtype=np.float32)

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x_start, y_start = col * cell_w, row * cell_h
            x_end, y_end = x_start + cell_w, y_start + cell_h

            # **计算中心 10% 面积的边界**
            center_x1 = int(x_start + (1 - CENTER_AREA_RATIO) / 2 * cell_w)
            center_x2 = int(x_start + (1 + CENTER_AREA_RATIO) / 2 * cell_w)
            center_y1 = int(y_start + (1 - CENTER_AREA_RATIO) / 2 * cell_h)
            center_y2 = int(y_start + (1 + CENTER_AREA_RATIO) / 2 * cell_h)

            # **从中心区域随机选取 `n` 个像素点**
            for i in range(sample_count):
                rand_x = np.random.randint(center_x1, center_x2)
                rand_y = np.random.randint(center_y1, center_y2)
                rgb_values[row, col, i] = region[rand_y, rand_x]  # 读取 RGB 值

            # **绘制分割线**
            cv2.rectangle(image, (x1 + x_start, y1 + y_start), (x1 + x_end, y1 + y_end), (0, 0, 255), 2)
            # **绘制 10% 选取区域**
            cv2.rectangle(image, (x1 + center_x1, y1 + center_y1), (x1 + center_x2, y1 + center_y2), (255, 0, 0), 2)

    return rgb_values, image

def save_rgb_values(image_path, output_dir,rgb_values):
    """ ✅ 保存提取的 RGB 颜色数组到自定义输出路径 """
    filename = os.path.splitext(os.path.basename(image_path))[0]
    npy_path = os.path.join(output_dir, f"rgb_{filename}.npy")
    np.save(npy_path, rgb_values.astype(np.float32))

    print(f"✅ RGB 颜色数据已保存至: {npy_path}")
    return npy_path


def visualize_results(image_path, image, x1, y1, x2, y2, rgb_values):
    """ ✅ 可视化结果，包括选区标记、色卡分割、RGB 采样展示 """
    sample_count = rgb_values.shape[2]
    grid_size = int(np.sqrt(sample_count))  # **计算网格大小**

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # **第一张图：PIL 读取原图，确保颜色正确**
    original_image = Image.open(image_path)  # **PIL 读取 RGB**
    original_image = np.array(original_image)  # **保持 RGB 颜色顺序**
    image_filename = os.path.basename(image_path)  # **获取文件名**

    ax[0].imshow(original_image)
    ax[0].set_title(f"Original Image: {image_filename}")  # ✅ **第一张图标题**
    ax[0].axis("off")

    # **第二张图：色卡区域 4×6 分割 + 取样范围**
    ax[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Color Grid with 10% Sampling Area")
    ax[1].axis("off")

    # **第三张图：提取的 24 色块 RGB 采样可视化**
    ax[2].set_xticks([])
    ax[2].set_yticks([])

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            sample_colors = rgb_values[row, col]/255
            reshaped_grid = sample_colors.reshape((grid_size, grid_size, 3))

            # **绘制色块**
            ax[2].imshow(reshaped_grid, extent=(col, col + 1, GRID_ROWS - row - 1, GRID_ROWS - row))

    ax[2].set_xlim(0, GRID_COLS)
    ax[2].set_ylim(0, GRID_ROWS)
    ax[2].set_title("Extracted RGB Grid (√n × √n per Block)")

    plt.tight_layout()
    plt.show()

def process_image(image_path):
    """ ✅ 处理单张图片：手动选择区域、色卡分割、颜色提取 """
    if image_path.lower().endswith('.cr2'):
        print(f"🔄 转换 CR2: {image_path}")
        image_path = convert_cr2_to_jpg(image_path)  # **转换 CR2**

    image = cv2.imread(image_path)  # **OpenCV 读取 BGR**

    selected_region = get_manual_selection(image)
    if selected_region is None:
        print("❌ 未选择区域，跳过此图片！")
        return

    x1, y1, x2, y2 = selected_region
    rgb_values, annotated_image = extract_rgb_from_grid(image, x1, y1, x2, y2)

    npy_path = save_rgb_values(image_path,output_path, rgb_values)  # ✅ 保存 RGB 颜色数组
    visualize_results(image_path, annotated_image, x1, y1, x2, y2, rgb_values)


# **运行处理**
input_path = "D:\Desktop\pictures-1.20\G1\orange\IMG_1043.CR2"  # 可以是单张图片
output_path="D:\Desktop\pictures-1.20"
process_image(input_path)
