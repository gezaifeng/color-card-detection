import numpy as np
import cv2
from PIL import Image
import matplotlib

matplotlib.use('TkAgg')  # 让 plt.show() 正常显示
import matplotlib.pyplot as plt
import os

# ✅ 颜色矩阵网格大小
GRID_ROWS, GRID_COLS = 4, 6

# ✅ 变量存储鼠标绘制矩形区域
rect_start = None
rect_end = None
drawing = False


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

    temp_image = image.copy()
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
        x1, y1 = min(rect_start[0], rect_end[0]), min(rect_start[1], rect_end[1])
        x2, y2 = max(rect_start[0], rect_end[0]), max(rect_start[1], rect_end[1])
        return x1, y1, x2, y2
    else:
        return None


def extract_rgb_from_grid(image, x1, y1, x2, y2):
    """ ✅ 在手动选择的区域内进行 4×6 分割，并计算颜色均值 """
    region = image[y1:y2, x1:x2]  # **BGR 格式**
    region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)  # **转换为 RGB**

    h, w = region.shape[:2]
    cell_h, cell_w = h // GRID_ROWS, w // GRID_COLS

    rgb_values = np.zeros((GRID_ROWS, GRID_COLS, 3), dtype=np.float32)

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x_start, y_start = col * cell_w, row * cell_h
            x_end, y_end = x_start + cell_w, y_start + cell_h

            cell_patch = region[y_start:y_end, x_start:x_end]

            # **计算中心 50% 面积的均值**
            center_x1, center_x2 = int(x_start + 0.25 * cell_w), int(x_start + 0.75 * cell_w)
            center_y1, center_y2 = int(y_start + 0.25 * cell_h), int(y_start + 0.75 * cell_h)
            center_patch = region[center_y1:center_y2, center_x1:center_x2]

            # **存储 RGB 平均值**
            rgb_values[row, col] = np.mean(center_patch, axis=(0, 1))

            # **绘制分割线**
            cv2.rectangle(image, (x1 + x_start, y1 + y_start), (x1 + x_end, y1 + y_end), (0, 0, 255), 2)
            cv2.rectangle(image, (x1 + center_x1, y1 + center_y1), (x1 + center_x2, y1 + center_y2), (255, 0, 0), 2)

    return rgb_values, image


def visualize_results(image_path, image, x1, y1, x2, y2, rgb_values):
    """ ✅ 可视化结果，包括选区标记、色卡分割、RGB 色块 """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # **第一张图：PIL 读取原图，确保颜色正确**
    original_image = Image.open(image_path)  # **PIL 读取 RGB**
    original_image = np.array(original_image)  # **保持 RGB 颜色顺序**
    ax[0].imshow(original_image)  # **确保原图颜色不变**
    ax[0].set_title("Original Image (Corrected RGB)")
    ax[0].axis("off")

    # **第二张图：色卡区域 4×6 分割 + 颜色提取**
    ax[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # **转换 RGB**
    ax[1].set_title("Color Grid in Selection")
    ax[1].axis("off")

    # **第三张图：提取的 24 色块 RGB 可视化**
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            rect = plt.Rectangle((col, GRID_ROWS - row - 1), 1, 1, facecolor=rgb_values[row, col] / 255.0)
            ax[2].add_patch(rect)
    ax[2].set_xlim(0, GRID_COLS)
    ax[2].set_ylim(0, GRID_ROWS)
    ax[2].set_title("Extracted RGB Colors")

    plt.tight_layout()
    plt.show()


def process_image(image_path):
    """ ✅ 处理单张图片：手动选择区域、色卡分割、颜色提取 """
    image = cv2.imread(image_path)  # **OpenCV 读取 BGR**

    # **手动绘制选区**
    selected_region = get_manual_selection(image)
    if selected_region is None:
        print("❌ 未选择区域，跳过此图片！")
        return

    x1, y1, x2, y2 = selected_region

    # **提取色卡区域的颜色信息**
    rgb_values, annotated_image = extract_rgb_from_grid(image, x1, y1, x2, y2)

    # **可视化结果**
    visualize_results(image_path, annotated_image, x1, y1, x2, y2, rgb_values)


def batch_process_images(input_dir):
    """ ✅ 处理文件夹内的所有图片 """
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('jpg', 'png', 'jpeg'))]

    if not images:
        print("❌ 未找到图片！")
        return

    print(f"🔍 发现 {len(images)} 张图片，开始处理...")

    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        print(f"📷 处理图片: {img_name} ...")
        process_image(img_path)

    print("✅ 所有图片处理完成！")


# **运行批量处理**
input_dir = "D:\\Desktop\\pngs2"
batch_process_images(input_dir)
