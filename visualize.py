# colorcard_kit/visualize.py
import os
import numpy as np
import matplotlib.pyplot as plt

def _rgb346_to_cellcolor(rgb_346):
    arr = np.transpose(rgb_346, (1, 2, 0))  # (4,6,3)
    arr = np.clip(arr, 0, 255) / 255.0
    return arr

def _show_edge(ax, edges, title="Edge"):
    if edges is None:
        ax.text(0.5, 0.5, "No Edge Image", ha="center", va="center")
        ax.axis("off"); ax.set_title(title); return
    ax.imshow(edges, cmap="gray"); ax.set_title(title); ax.axis("off")

def _show_annotated(ax, annotated_bgr, title="Annotated"):
    if annotated_bgr is None:
        ax.text(0.5, 0.5, "No Annotated Image", ha="center", va="center")
        ax.axis("off"); ax.set_title(title); return
    ax.imshow(annotated_bgr[..., ::-1]); ax.set_title(title); ax.axis("off")

def _show_stacked_rgb_matrices(ax, ref_rgb_346, sample_rgb_346, title="RGB Matrices (Top=Ref, Bottom=Sample)"):
    ref_img = _rgb346_to_cellcolor(ref_rgb_346)       # (4,6,3)
    samp_img = _rgb346_to_cellcolor(sample_rgb_346)   # (4,6,3)
    rows, cols = ref_img.shape[:2]
    stacked = np.vstack([ref_img, samp_img])
    ax.imshow(stacked, aspect="equal", interpolation="nearest",
              extent=(0, cols, 2*rows, 0))
    ax.set_title(title)
    ax.set_xlim(0, cols); ax.set_ylim(2*rows, 0)
    ax.set_xticks(range(cols)); ax.set_yticks(range(0, 2*rows+1))
    ax.set_xlabel("col"); ax.set_ylabel("row")
    for c in range(cols+1): ax.axvline(c, color="k", linewidth=0.5)
    for r in range(2*rows+1): ax.axhline(r, color="k", linewidth=0.5)

def _heatmap(ax, mat, title, cmap="viridis", vmin=None, vmax=None, with_cbar=True):
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(range(mat.shape[1])); ax.set_yticks(range(mat.shape[0]))
    ax.set_xlabel("col"); ax.set_ylabel("row")
    ax.grid(color="k", linestyle="-", linewidth=0.5)
    if with_cbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im

def visualize_pair(image_path,
                   edges, annotated_bgr,
                   ref_rgb_346, sample_rgb_346,
                   ratio_346, log_ratio_346,  # 两个都传入，便于不同模式展示
                   feature_mode="log_ratio",
                   out_dir="./vis"):
    """
    2x3 布局：
      [1] Edge
      [2] Annotated
      [3] RGB 堆叠矩阵（Top=Ref, Bottom=Sample）
      [4-6] R/G/B 热力图 —— 若 mode 为 'log_ratio' 则显示 log_ratio 的 R/G/B；
                            若 mode 为 'ratio' 或 'multi' 则显示 ratio 的 R/G/B。
    """
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    _show_edge(ax1, edges, "Edge")
    _show_annotated(ax2, annotated_bgr, "Annotated")
    _show_stacked_rgb_matrices(ax3, ref_rgb_346, sample_rgb_346,
                               "RGB Matrices (Top=Ref, Bottom=Sample)")

    if feature_mode == "log_ratio":
        show_mat = log_ratio_346
        title_suffix = "Log-Ratio"
    else:
        show_mat = ratio_346
        title_suffix = "Ratio"

    _heatmap(ax4, show_mat[0], f"{title_suffix} R")
    _heatmap(ax5, show_mat[1], f"{title_suffix} G")
    _heatmap(ax6, show_mat[2], f"{title_suffix} B")

    plt.tight_layout()
    save_path = os.path.join(out_dir,
        os.path.splitext(os.path.basename(image_path))[0] + "_pair_vis.png")
    plt.savefig(save_path, dpi=150); plt.close(fig)
    return save_path
