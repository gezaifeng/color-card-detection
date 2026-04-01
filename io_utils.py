import os

def find_images(directory):
    """递归查找目录下的图像文件"""
    image_paths = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff",".cr2")):
                image_paths.append(os.path.join(root, f))
    return image_paths


def out_path(input_dir, output_dir, image_path, prefix="", suffix="", ext="npy"):
    rel = os.path.relpath(image_path, start=input_dir)
    base = os.path.splitext(os.path.basename(rel))[0]
    if suffix:
        name = f"{prefix}{base}_{suffix}.{ext}"
    else:
        name = f"{prefix}{base}.{ext}"
    return os.path.join(output_dir, name)
