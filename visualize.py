# colorcard_kit/visualize.py
import os
import numpy as np
import matplotlib

# 仅用于保存图片，不依赖交互式后端；Agg 更稳健（避免 TkAgg 在无显示/多线程下报错）
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _as_rcx3_samples(rgb):
    """统一色块 RGB 的形状为 samples: (rows, cols, x, 3)，值域 0~255(float32)。

    支持以下常见形状：
      - (3, rows, cols)                 : 每格一个 RGB
      - (rows, cols, 3)
      - (3, rows, cols, x)              : 每格 x 个采样点（aberration 模式常见）
      - (rows, cols, x, 3)
    """
    arr = np.asarray(rgb)

    if arr.ndim == 3:
        # (3,R,C) or (R,C,3)
        if arr.shape[0] == 3 and arr.shape[-1] != 3:
            # (3,R,C) -> (R,C,3)
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] != 3:
            raise ValueError(f"Unsupported RGB shape: {arr.shape}")
        arr = arr[:, :, None, :]  # (R,C,1,3)

    elif arr.ndim == 4:
        # (3,R,C,X) or (R,C,X,3)
        if arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (1, 2, 3, 0))  # (R,C,X,3)
        if arr.shape[-1] != 3:
            raise ValueError(f"Unsupported RGB shape: {arr.shape}")

    else:
        raise ValueError(f"Unsupported RGB ndim: {arr.ndim}, shape={arr.shape}")

    arr = arr.astype(np.float32)
    return np.clip(arr, 0, 255)


def _samples_to_mosaic(samples_rcx3):
    """将 (R,C,X,3) 的采样点拼成“每格内部有纹理”的马赛克图像。

    每个色块被展开成 side×side 的小方块（side=ceil(sqrt(X))），把 X 个采样点按顺序填入。
    这样第三张图就能直观看到每格内部的 RGB 分布（aberration 的细粒度编码）。
    """
    s = samples_rcx3
    rows, cols, x, _ = s.shape
    side = int(np.ceil(np.sqrt(max(int(x), 1))))

    # 若 x < side*side，用均值补齐
    mean_rc3 = s.mean(axis=2, keepdims=True)  # (R,C,1,3)
    if x < side * side:
        pad = np.repeat(mean_rc3, side * side - x, axis=2)
        s = np.concatenate([s, pad], axis=2)

    s = s[:, :, : side * side, :]                 # (R,C,side*side,3)
    s = s.reshape(rows, cols, side, side, 3)      # (R,C,side,side,3)

    # 拼成大图： (R*side, C*side, 3)
    mosaic = s.transpose(0, 2, 1, 3, 4).reshape(rows * side, cols * side, 3)
    mosaic = np.clip(mosaic, 0, 255) / 255.0
    return mosaic, side


def _show_edge(ax, edges, title="Edge"):
    if edges is None:
        ax.text(0.5, 0.5, "No Edge Image", ha="center", va="center")
        ax.axis("off"); ax.set_title(title); return
    ax.imshow(edges, cmap="gray")
    ax.set_title(title)
    ax.axis("off")


def _show_annotated(ax, annotated_bgr, title="Annotated"):
    if annotated_bgr is None:
        ax.text(0.5, 0.5, "No Annotated Image", ha="center", va="center")
        ax.axis("off"); ax.set_title(title); return
    ax.imshow(annotated_bgr[..., ::-1])
    ax.set_title(title)
    ax.axis("off")


def _show_stacked_rgb_matrices(ax, ref_rgb, sample_rgb, title="RGB (Top=Ref, Bottom=Sample)"):
    """第三张图：
    - normal: (3,R,C) / (R,C,3) → 每格一个颜色块
    - aberration: (3,R,C,X) / (R,C,X,3) → 每格内部显示 X 个采样点分布
    """
    ref_samples = _as_rcx3_samples(ref_rgb)      # (R,C,X,3)
    sam_samples = _as_rcx3_samples(sample_rgb)   # (R,C,X,3)

    ref_mosaic, side = _samples_to_mosaic(ref_samples)
    sam_mosaic, _ = _samples_to_mosaic(sam_samples)

    rows, cols = ref_samples.shape[:2]
    stacked = np.vstack([ref_mosaic, sam_mosaic])

    ax.imshow(
        stacked,
        aspect="equal",
        interpolation="nearest",
        extent=(0, cols, 2 * rows, 0)  # 用 extent 把“像素马赛克”映射回“格子坐标”
    )
    ax.set_title(title + f"  (each cell shows {side}×{side} samples)")
    ax.set_xlim(0, cols)
    ax.set_ylim(2 * rows, 0)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(0, 2 * rows + 1))
    ax.set_xlabel("col")
    ax.set_ylabel("row")

    # 画格子网格线（按色块维度 m×n，不按内部采样点）
    for c in range(cols + 1):
        ax.axvline(c, color="k", linewidth=0.5)
    for r in range(2 * rows + 1):
        ax.axhline(r, color="k", linewidth=0.5)


def _heatmap(ax, mat, title, cmap="viridis", vmin=None, vmax=None, with_cbar=True):
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_yticks(range(mat.shape[0]))
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.grid(color="k", linestyle="-", linewidth=0.5)
    if with_cbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def visualize_pair(
    image_path,
    edges,
    annotated_bgr,
    ref_rgb_346,
    sample_rgb_346,
    ratio_346,
    log_ratio_346,   # 保留入参以兼容旧调用，但不再用于显示
    feature_mode="log_ratio",
    out_dir="./vis"
):
    """
    2x3 布局（按你的新需求统一为“RGB直观可视化”）：
      [1] 灰度/边缘图（维持原逻辑）
      [2] 识别区域标注图（维持原逻辑）
      [3] RGB 三维数据可视化（Top=Ref, Bottom=Sample）
          - normal: 每格一个颜色块
          - aberration: 每格内展开为 side×side 小方块，显示该格 x 个采样点的 RGB 分布
      [4-6] R/G/B 三通道热力图（Top=Ref, Bottom=Sample）
            不再展示 log/ratio 等处理结果，所有模式统一为 RGB 通道可视化。
    """
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    _show_edge(ax1, edges, "Edge")
    _show_annotated(ax2, annotated_bgr, "Annotated")
    _show_stacked_rgb_matrices(ax3, ref_rgb_346, sample_rgb_346, "RGB Samples (Top=Ref, Bottom=Sample)")

    # 下三张：统一展示 RGB 三通道热力图（按色块维度 m×n，aberration 会先对 x 求均值）
    ref_samples = _as_rcx3_samples(ref_rgb_346)   # (R,C,X,3)
    sam_samples = _as_rcx3_samples(sample_rgb_346)

    ref_mean = ref_samples.mean(axis=2)  # (R,C,3)
    sam_mean = sam_samples.mean(axis=2)
    stacked_mean = np.vstack([ref_mean, sam_mean])  # (2R,C,3)

    _heatmap(ax4, stacked_mean[..., 0], "RGB Heatmap · R (Top=Ref, Bottom=Sample)", vmin=0, vmax=255)
    _heatmap(ax5, stacked_mean[..., 1], "RGB Heatmap · G (Top=Ref, Bottom=Sample)", vmin=0, vmax=255)
    _heatmap(ax6, stacked_mean[..., 2], "RGB Heatmap · B (Top=Ref, Bottom=Sample)", vmin=0, vmax=255)

    plt.tight_layout()
    save_path = os.path.join(
        out_dir,
        os.path.splitext(os.path.basename(image_path))[0] + "_pair_vis.png"
    )
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
