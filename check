import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 让 plt.show() 正常显示
import matplotlib.pyplot as plt


def load_npy_files(folder_path, display_mode='array', max_images=100):
    """
    读取文件夹中的所有 .npy 文件，并根据展示模式进行处理。

    参数:
        folder_path (str): 目标文件夹路径
        display_mode (str): 'array'（数组展示）或 'rgb'（RGB 图像展示）
        max_images (int): 最多同时展示的图片数量，避免过多导致窗口太大
    """
    if not os.path.isdir(folder_path):
        print(f"错误：{folder_path} 不是有效的文件夹路径。")
        return

    npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

    if not npy_files:
        print("文件夹内没有 .npy 文件。")
        return

    rgb_images = []
    file_names = []

    for file_name in npy_files:
        file_path = os.path.join(folder_path, file_name)
        try:
            data = np.load(file_path)
            print(f"\n加载文件: {file_name}, 形状: {data.shape}, 数据类型: {data.dtype}")

            if display_mode == 'array':
                print(data)  # 直接打印数组内容

            elif display_mode == 'rgb':
                if len(data.shape) == 3 and data.shape[-1] in [3, 4]:  # RGB 或 RGBA 图像
                    rgb_images.append(data)
                    file_names.append(file_name)
                    if len(rgb_images) >= max_images:  # 限制最大展示数量
                        break
                else:
                    print(f"警告: {file_name} 可能不是 RGB 图像（形状: {data.shape}），无法显示。")

        except Exception as e:
            print(f"读取 {file_name} 失败: {e}")

    # 一次性显示多张 RGB 图片
    if display_mode == 'rgb' and rgb_images:
        fig, axes = plt.subplots(1, len(rgb_images), figsize=(5 * len(rgb_images), 5))
        if len(rgb_images) == 1:
            axes = [axes]  # 适配单张图片情况

        for img, ax, title in zip(rgb_images, axes, file_names):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')

        plt.tight_layout()
        plt.show()


def load_single_npy(file_path, display_mode='array'):
    """
    读取单个 .npy 文件，并根据展示模式显示。

    参数:
        file_path (str): .npy 文件路径
        display_mode (str): 'array'（数组展示）或 'rgb'（RGB 图像展示）
    """
    if not os.path.isfile(file_path):
        print(f"错误：{file_path} 不是有效的文件路径。")
        return

    try:
        data = np.load(file_path)
        print(f"\n加载文件: {file_path}, 形状: {data.shape}, 数据类型: {data.dtype}")

        if display_mode == 'array':
            print(data)  # 直接打印数组内容

        elif display_mode == 'rgb':
            if len(data.shape) == 3 and data.shape[-1] in [3, 4]:  # RGB 或 RGBA 图像
                plt.imshow(data/255)
                plt.title(os.path.basename(file_path))
                plt.axis('off')
                plt.show()
            else:
                print(f"警告: {file_path} 可能不是 RGB 图像（形状: {data.shape}），无法显示。")

    except Exception as e:
        print(f"读取 {file_path} 失败: {e}")

# 使用示例：
# 1. 读取并展示整个文件夹的所有 .npy 文件
# load_npy_files("your/folder/path", display_mode='rgb')

# 2. 读取并展示单张 .npy 图片
# load_single_npy("your/folder/path/sample.npy", display_mode='rgb')
data="D:\Desktop\extracted\\rgb_IMG_1049.npy"
load_single_npy(data, display_mode='rgb')
