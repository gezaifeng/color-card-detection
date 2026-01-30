import os

def find_images(directory):
    """递归查找目录下的图像文件"""
    image_paths = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff",".cr2")):
                image_paths.append(os.path.join(root, f))
    return image_paths

def out_path(input_dir, output_dir, image_path, prefix="rgb_", suffix="", ext="npy"):
    """保持与输入目录相对层级一致地生成输出路径"""
    rel = os.path.relpath(image_path, input_dir)
    rel_dir = os.path.dirname(rel)
    folder = os.path.join(output_dir, rel_dir)
    os.makedirs(folder, exist_ok=True)
    name = os.path.splitext(os.path.basename(image_path))[0]
    if suffix: suffix = "_" + suffix
    return os.path.join(folder, f"{prefix}{name}{suffix}.{ext}")
