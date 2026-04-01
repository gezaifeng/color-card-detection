# ui_main.py
import os
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import inspect  # NEW

from config import PipelineConfig
from io_utils import find_images
from pipeline import process_single


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("色卡识别与特征导出（批处理）")
        self.geometry("980x780")
        self.minsize(920, 700)
        self.resizable(True, True)

        self._stop_flag = threading.Event()
        self._worker = None

        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        # ===== 输入输出 =====
        io = ttk.LabelFrame(self, text="输入与输出")
        io.pack(fill="x", **pad)

        ttk.Label(io, text="输入目录：").grid(row=0, column=0, sticky="e")
        self.in_entry = ttk.Entry(io, width=72)
        self.in_entry.grid(row=0, column=1, sticky="we", padx=(0, 6))
        ttk.Button(io, text="选择...", command=self._choose_in).grid(row=0, column=2)

        ttk.Label(io, text="输出目录：").grid(row=1, column=0, sticky="e")
        self.out_entry = ttk.Entry(io, width=72)
        self.out_entry.grid(row=1, column=1, sticky="we", padx=(0, 6))
        ttk.Button(io, text="选择...", command=self._choose_out).grid(row=1, column=2)

        io.columnconfigure(1, weight=1)

        # ===== Notebook 参数 =====
        nb = ttk.Notebook(self)
        nb.pack(fill="x", **pad)

        tab_basic = ttk.Frame(nb)
        tab_extract = ttk.Frame(nb)
        tab_detect = ttk.Frame(nb)
        tab_feat = ttk.Frame(nb)
        tab_raw = ttk.Frame(nb)
        nb.add(tab_basic, text="基础参数")
        nb.add(tab_extract, text="提取方式")
        nb.add(tab_detect, text="检测增强")
        nb.add(tab_feat, text="特征输出")
        nb.add(tab_raw, text="RAW/手动")

        # ===== 基础参数 =====
        frm = ttk.LabelFrame(tab_basic, text="色卡与检测基本参数")
        frm.pack(fill="x", padx=8, pady=8)

        self.var_grid_rows = tk.IntVar(value=6)
        self.var_grid_cols = tk.IntVar(value=12)
        self.var_target_height = tk.IntVar(value=512)
        self.var_sobel_ksize = tk.IntVar(value=3)
        self.var_edge_thresh = tk.IntVar(value=50)

        self.var_card_crop_long = tk.DoubleVar(value=0.01)
        self.var_card_crop_short = tk.DoubleVar(value=0.02)
        self.var_sample_center_area = tk.DoubleVar(value=0.15)
        self.var_sample_count = tk.IntVar(value=100)

        # 两列网格
        r = 0
        ttk.Label(frm, text="色块行数 m：").grid(row=r, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.var_grid_rows, width=8).grid(row=r, column=1, sticky="w")
        ttk.Label(frm, text="色块列数 n：").grid(row=r, column=2, sticky="e")
        ttk.Entry(frm, textvariable=self.var_grid_cols, width=8).grid(row=r, column=3, sticky="w")
        r += 1

        ttk.Label(frm, text="检测缩放高度：").grid(row=r, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.var_target_height, width=8).grid(row=r, column=1, sticky="w")
        ttk.Label(frm, text="Sobel 核大小：").grid(row=r, column=2, sticky="e")
        ttk.Entry(frm, textvariable=self.var_sobel_ksize, width=8).grid(row=r, column=3, sticky="w")
        r += 1

        ttk.Label(frm, text="边缘阈值：").grid(row=r, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.var_edge_thresh, width=8).grid(row=r, column=1, sticky="w")
        ttk.Label(frm, text="中心采样面积(0~1)：").grid(row=r, column=2, sticky="e")
        ttk.Entry(frm, textvariable=self.var_sample_center_area, width=8).grid(row=r, column=3, sticky="w")
        r += 1

        ttk.Label(frm, text="每块采样点数(常规)：").grid(row=r, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.var_sample_count, width=8).grid(row=r, column=1, sticky="w")
        ttk.Label(frm, text="内缩裁剪(长边/短边)：").grid(row=r, column=2, sticky="e")
        inner = ttk.Frame(frm)
        inner.grid(row=r, column=3, sticky="w")
        ttk.Entry(inner, textvariable=self.var_card_crop_long, width=7).pack(side="left")
        ttk.Label(inner, text=" / ").pack(side="left")
        ttk.Entry(inner, textvariable=self.var_card_crop_short, width=7).pack(side="left")
        r += 1

        for c in range(4):
            frm.columnconfigure(c, weight=1)

        # ===== 提取方式（常规/像差 + 输入模式）=====
        frm2 = ttk.LabelFrame(tab_extract, text="输入模式与提取模式")
        frm2.pack(fill="x", padx=8, pady=8)

        self.var_input_mode = tk.StringVar(value="auto")   # auto/cr2/jpg
        self.var_extract_mode = tk.StringVar(value="normal")  # normal/aberration
        self.var_ab_points = tk.IntVar(value=9)
        self.var_ab_seed = tk.IntVar(value=0)

        rr = 0
        ttk.Label(frm2, text="输入格式模式：").grid(row=rr, column=0, sticky="e")
        ttk.OptionMenu(frm2, self.var_input_mode, "auto", "auto", "cr2", "jpg").grid(row=rr, column=1, sticky="w")
        ttk.Label(frm2, text="（auto=自动判断；cr2=强制RAW；jpg=强制普通图）").grid(row=rr, column=2, sticky="w")
        rr += 1

        ttk.Label(frm2, text="提取方式：").grid(row=rr, column=0, sticky="e")
        ttk.OptionMenu(frm2, self.var_extract_mode, "normal", "normal", "aberration").grid(row=rr, column=1, sticky="w")
        ttk.Label(frm2, text="normal=每块取均值；aberration=每块采样多个点不求均值").grid(row=rr, column=2, sticky="w")
        rr += 1

        ttk.Label(frm2, text="像差采样点数 x：").grid(row=rr, column=0, sticky="e")
        ttk.Entry(frm2, textvariable=self.var_ab_points, width=10).grid(row=rr, column=1, sticky="w")
        ttk.Label(frm2, text="输出形状为 3*m*n*x").grid(row=rr, column=2, sticky="w")
        rr += 1

        ttk.Label(frm2, text="像差采样随机种子：").grid(row=rr, column=0, sticky="e")
        ttk.Entry(frm2, textvariable=self.var_ab_seed, width=10).grid(row=rr, column=1, sticky="w")
        ttk.Label(frm2, text="（固定种子可复现每次采样）").grid(row=rr, column=2, sticky="w")
        rr += 1

        frm2.columnconfigure(2, weight=1)

        # ===== 检测增强（亮度/对比度/饱和度/gamma/CLAHE/自动增亮）=====
        frm3 = ttk.LabelFrame(tab_detect, text="检测增强（仅用于识别找框，不影响最终提取保存）")
        frm3.pack(fill="x", padx=8, pady=8)

        self.var_det_enable = tk.BooleanVar(value=True)
        self.var_det_brightness = tk.DoubleVar(value=1.0)
        self.var_det_contrast = tk.DoubleVar(value=1.0)
        self.var_det_saturation = tk.DoubleVar(value=1.0)
        self.var_det_gamma = tk.DoubleVar(value=1.0)

        self.var_det_use_clahe = tk.BooleanVar(value=False)
        self.var_clahe_clip = tk.DoubleVar(value=2.0)
        self.var_clahe_grid = tk.IntVar(value=8)

        self.var_det_auto_bright = tk.BooleanVar(value=True)
        self.var_bright_target_median = tk.DoubleVar(value=110.0)
        self.var_bright_max_gain = tk.DoubleVar(value=4.0)

        ttk.Checkbutton(frm3, text="启用检测增强", variable=self.var_det_enable).pack(anchor="w")

        def add_slider(parent, title, var, frm, to, digits=2):
            rowf = ttk.Frame(parent)
            rowf.pack(fill="x", pady=2)
            ttk.Label(rowf, text=title, width=18).pack(side="left")
            s = ttk.Scale(rowf, variable=var, from_=frm, to=to)
            s.pack(side="left", fill="x", expand=True, padx=8)
            val = ttk.Label(rowf, textvariable=var, width=8)
            val.pack(side="right")

        add_slider(frm3, "亮度倍率(V)", self.var_det_brightness, 0.3, 3.0)
        add_slider(frm3, "对比度(alpha)", self.var_det_contrast, 0.3, 3.0)
        add_slider(frm3, "饱和度倍率(S)", self.var_det_saturation, 0.0, 3.0)
        add_slider(frm3, "Gamma", self.var_det_gamma, 0.3, 2.5)

        ttk.Separator(frm3).pack(fill="x", pady=6)

        rowc = ttk.Frame(frm3)
        rowc.pack(fill="x")
        ttk.Checkbutton(rowc, text="灰度 CLAHE", variable=self.var_det_use_clahe).pack(side="left")
        ttk.Label(rowc, text="clip").pack(side="left", padx=(12, 4))
        ttk.Entry(rowc, textvariable=self.var_clahe_clip, width=8).pack(side="left")
        ttk.Label(rowc, text="grid").pack(side="left", padx=(12, 4))
        ttk.Spinbox(rowc, from_=2, to=32, textvariable=self.var_clahe_grid, width=6).pack(side="left")

        ttk.Separator(frm3).pack(fill="x", pady=6)

        rowa = ttk.Frame(frm3)
        rowa.pack(fill="x")
        ttk.Checkbutton(rowa, text="过暗自动增亮（建议开启）", variable=self.var_det_auto_bright).pack(side="left")
        ttk.Label(rowa, text="目标中位亮度").pack(side="left", padx=(12, 4))
        ttk.Entry(rowa, textvariable=self.var_bright_target_median, width=8).pack(side="left")
        ttk.Label(rowa, text="最大增益").pack(side="left", padx=(12, 4))
        ttk.Entry(rowa, textvariable=self.var_bright_max_gain, width=8).pack(side="left")

        # ===== 特征输出 =====
        frm4 = ttk.LabelFrame(tab_feat, text="特征输出设置")
        frm4.pack(fill="x", padx=8, pady=8)

        self.var_feature_mode = tk.StringVar(value="log_ratio")
        self.var_norm = tk.BooleanVar(value=True)
        self.var_save_extras = tk.BooleanVar(value=True)

        rr = 0
        ttk.Label(frm4, text="特征模式：").grid(row=rr, column=0, sticky="e")
        ttk.OptionMenu(frm4, self.var_feature_mode, "log_ratio", "log_ratio", "ratio", "multi").grid(row=rr, column=1, sticky="w")
        ttk.Checkbutton(frm4, text="单张按通道归一化", variable=self.var_norm).grid(row=rr, column=2, sticky="w")
        ttk.Checkbutton(frm4, text="额外保存 ratio/logratio", variable=self.var_save_extras).grid(row=rr, column=3, sticky="w")
        rr += 1

        frm4.columnconfigure(3, weight=1)

        # ===== RAW 与手动回退 ====
        frm5 = ttk.LabelFrame(tab_raw, text="RAW读取与手动回退")
        frm5.pack(fill="x", padx=8, pady=8)

        self.var_allow_manual = tk.BooleanVar(value=True)
        self.var_force_manual = tk.BooleanVar(value=False)
        self.var_manual_downscale = tk.IntVar(value=1200)

        self.var_prefer_raw = tk.BooleanVar(value=True)
        self.var_raw_wb = tk.BooleanVar(value=True)
        self.var_raw_bps = tk.IntVar(value=16)

        # NEW 1: 全局开关：采样/保存使用调节后图像
        self.var_sample_from_adjusted = tk.BooleanVar(value=False)

        # NEW 2: 手动专用开关：仅当自动失败进入手动框选时，采样/保存使用调节后图像
        self.var_manual_sample_from_adjusted = tk.BooleanVar(value=True)

        rr = 0
        ttk.Checkbutton(frm5, text="允许手动框选回退", variable=self.var_allow_manual).grid(row=rr, column=0, sticky="w")
        ttk.Checkbutton(frm5, text="强制手动框选", variable=self.var_force_manual).grid(row=rr, column=1, sticky="w")
        ttk.Label(frm5, text="手动显示缩放(最大边)：").grid(row=rr, column=2, sticky="e")
        ttk.Entry(frm5, textvariable=self.var_manual_downscale, width=10).grid(row=rr, column=3, sticky="w")
        rr += 1

        ttk.Checkbutton(frm5, text="优先RAW线性解码（CR2等）", variable=self.var_prefer_raw).grid(row=rr, column=0, sticky="w")
        ttk.Checkbutton(frm5, text="RAW使用相机白平衡", variable=self.var_raw_wb).grid(row=rr, column=1, sticky="w")
        ttk.Label(frm5, text="RAW输出位深(检测/预览)：").grid(row=rr, column=2, sticky="e")
        ttk.Entry(frm5, textvariable=self.var_raw_bps, width=10).grid(row=rr, column=3, sticky="w")
        rr += 1

        # NEW 1: 全局采样源
        ttk.Checkbutton(
            frm5,
            text="采样使用调节后图像（提取/保存用，全局）",
            variable=self.var_sample_from_adjusted
        ).grid(row=rr, column=0, columnspan=4, sticky="w")
        rr += 1

        # NEW 2: 手动专用（只在自动失败→手动框选时生效）
        ttk.Checkbutton(
            frm5,
            text="自动失败进入手动框选时：采样/保存使用调节后图像（推荐）",
            variable=self.var_manual_sample_from_adjusted
        ).grid(row=rr, column=0, columnspan=4, sticky="w")
        rr += 1

        for c in range(4):
            frm5.columnconfigure(c, weight=1)

        # ===== 控制区 =====
        ctrl = ttk.Frame(self)
        ctrl.pack(fill="x", **pad)

        self.run_btn = ttk.Button(ctrl, text="开始批处理", command=self._on_start)
        self.run_btn.pack(side="left")

        ttk.Button(ctrl, text="停止", command=self._on_stop).pack(side="left", padx=8)

        self.open_btn = ttk.Button(ctrl, text="打开输出目录", command=self._open_out, state="disabled")
        self.open_btn.pack(side="left")

        self.progress = ttk.Progressbar(ctrl, mode="determinate")
        self.progress.pack(side="right", fill="x", expand=True)

        # ===== 日志 =====
        logf = ttk.LabelFrame(self, text="运行日志")
        logf.pack(fill="both", expand=True, **pad)

        self.log = tk.Text(logf, height=18, wrap="word")
        self.log.pack(side="left", fill="both", expand=True)

        scroll = ttk.Scrollbar(logf, command=self.log.yview)
        scroll.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=scroll.set)

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(fill="x", padx=8, pady=(0, 8))

    # ---------------- events ----------------
    def _choose_in(self):
        d = filedialog.askdirectory(title="选择输入目录")
        if d:
            self.in_entry.delete(0, tk.END)
            self.in_entry.insert(0, d)

    def _choose_out(self):
        d = filedialog.askdirectory(title="选择输出目录")
        if d:
            self.out_entry.delete(0, tk.END)
            self.out_entry.insert(0, d)

    def _open_out(self):
        outp = self.out_entry.get().strip()
        if outp and os.path.isdir(outp):
            try:
                if os.name == "nt":
                    os.startfile(outp)  # type: ignore
                elif hasattr(os, "uname") and os.uname().sysname.lower().startswith("darwin"):  # type: ignore
                    os.system(f'open "{outp}"')
                else:
                    os.system(f'xdg-open "{outp}"')
            except Exception as e:
                messagebox.showerror("错误", f"无法打开目录：\n{e}")

    def _on_start(self):
        if self._worker and self._worker.is_alive():
            messagebox.showwarning("提示", "任务正在进行中")
            return

        inp, outp = self.in_entry.get().strip(), self.out_entry.get().strip()
        if not inp or not os.path.isdir(inp):
            messagebox.showerror("错误", "请输入有效的【输入目录】")
            return
        if not outp:
            messagebox.showerror("错误", "请输入【输出目录】")
            return
        os.makedirs(outp, exist_ok=True)

        try:
            cfg = self._make_config()
        except ValueError as e:
            messagebox.showerror("参数错误", str(e))
            return

        imgs = find_images(inp)
        if not imgs:
            messagebox.showwarning("提示", "未在输入目录找到图像文件")
            return

        self._stop_flag = threading.Event()
        self.progress.configure(maximum=len(imgs), value=0)
        self.log.delete("1.0", tk.END)
        self.status_var.set(f"准备开始：共 {len(imgs)} 张")
        self.open_btn.configure(state="disabled")

        self._worker = threading.Thread(target=self._run_worker, args=(imgs, inp, outp, cfg), daemon=True)
        self._worker.start()

    def _on_stop(self):
        if self._worker and self._worker.is_alive():
            self._stop_flag.set()
            self.status_var.set("请求停止中…")
        else:
            self.status_var.set("当前无运行中的任务")

    # ---------------- config build ----------------
    def _make_config(self) -> PipelineConfig:
        rows = int(self.var_grid_rows.get())
        cols = int(self.var_grid_cols.get())
        th = int(self.var_target_height.get())
        sc = int(self.var_sample_count.get())
        ksz = int(self.var_sobel_ksize.get())
        thr = int(self.var_edge_thresh.get())
        ccl = float(self.var_card_crop_long.get())
        ccs = float(self.var_card_crop_short.get())
        area = float(self.var_sample_center_area.get())

        ab_pts = int(self.var_ab_points.get())
        ab_seed = int(self.var_ab_seed.get())

        # 基本校验
        if rows <= 0 or cols <= 0:
            raise ValueError("色块行列数必须为正整数")
        if th < 64:
            raise ValueError("检测缩放高度过小（建议 ≥ 256）")
        if sc <= 0:
            raise ValueError("常规采样点数必须为正整数")
        if ksz not in (1, 3, 5, 7):
            raise ValueError("Sobel 核大小需为 1/3/5/7")
        if thr <= 0:
            raise ValueError("边缘阈值必须为正整数")
        if not (0.0 <= ccl < 0.5) or not (0.0 <= ccs < 0.5):
            raise ValueError("内缩裁剪比例建议在 [0,0.5) 内")
        if not (0.0 < area <= 1.0):
            raise ValueError("中心采样面积需在 (0,1] 内")
        if ab_pts <= 0:
            raise ValueError("像差采样点数必须为正整数")

        # NEW: 兼容注入两个字段（构造参数支持就传入，否则 setattr）
        want_sample_from_adjusted = bool(self.var_sample_from_adjusted.get())
        want_manual_sample_from_adjusted = bool(self.var_manual_sample_from_adjusted.get())

        pc_sig = None
        try:
            pc_sig = inspect.signature(PipelineConfig)
        except Exception:
            pc_sig = None

        extra_kwargs = {}
        if pc_sig is not None and ("sample_from_adjusted" in pc_sig.parameters):
            extra_kwargs["sample_from_adjusted"] = want_sample_from_adjusted
        if pc_sig is not None and ("manual_sample_from_adjusted" in pc_sig.parameters):
            extra_kwargs["manual_sample_from_adjusted"] = want_manual_sample_from_adjusted

        cfg = PipelineConfig(
            # 关键模式
            input_mode=str(self.var_input_mode.get()).lower(),
            extract_mode=str(self.var_extract_mode.get()).lower(),
            aberration_points=ab_pts,
            aberration_seed=ab_seed,

            # 网格/检测
            grid_rows=rows,
            grid_cols=cols,
            target_height=th,
            sobel_ksize=ksz,
            edge_thresh=thr,

            # 提取
            sample_count=sc,
            sample_center_area=area,

            # 特征
            feature_mode=self.var_feature_mode.get(),
            per_image_channel_norm=bool(self.var_norm.get()),
            save_extras=bool(self.var_save_extras.get()),

            # 手动/RAW
            allow_manual=bool(self.var_allow_manual.get()),
            force_manual=bool(self.var_force_manual.get()),
            manual_downscale=int(self.var_manual_downscale.get()),
            prefer_raw_linear=bool(self.var_prefer_raw.get()),
            raw_use_camera_wb=bool(self.var_raw_wb.get()),
            raw_output_bps=int(self.var_raw_bps.get()),

            # 检测增强
            det_tune_enable=bool(self.var_det_enable.get()),
            det_brightness=float(self.var_det_brightness.get()),
            det_contrast=float(self.var_det_contrast.get()),
            det_saturation=float(self.var_det_saturation.get()),
            det_gamma=float(self.var_det_gamma.get()),
            det_use_clahe=bool(self.var_det_use_clahe.get()),
            clahe_clip_limit=float(self.var_clahe_clip.get()),
            clahe_grid=int(self.var_clahe_grid.get()),
            det_auto_brighten=bool(self.var_det_auto_bright.get()),
            bright_target_median=float(self.var_bright_target_median.get()),
            bright_max_gain=float(self.var_bright_max_gain.get()),

            **extra_kwargs
        )

        # 若构造期不支持，则尝试动态挂载（不让 UI 崩）
        if "sample_from_adjusted" not in extra_kwargs:
            try:
                setattr(cfg, "sample_from_adjusted", want_sample_from_adjusted)
            except Exception:
                pass
        if "manual_sample_from_adjusted" not in extra_kwargs:
            try:
                setattr(cfg, "manual_sample_from_adjusted", want_manual_sample_from_adjusted)
            except Exception:
                pass

        # 这两个字段也保留为 UI 可调（pipeline/extract 用）
        cfg.card_crop_long = ccl
        cfg.card_crop_short = ccs
        return cfg

    # ---------------- worker ----------------
    def _run_worker(self, imgs, inp, outp, cfg: PipelineConfig):
        ok = fail = 0
        for i, p in enumerate(imgs, 1):
            if self._stop_flag.is_set():
                self._append_log(f"[停止] 已中断，最后处理到：{i-1}/{len(imgs)}")
                break
            try:
                res = process_single(p, inp, outp, cfg)
                if res:
                    ok += 1
                    vis_path = res.get("vis", None)
                    if isinstance(vis_path, (str, bytes, os.PathLike)) and vis_path:
                        vis_rel = os.path.relpath(vis_path, outp)
                    else:
                        vis_rel = "(无可视化)"
                    self._append_log(f"[成功] {i}/{len(imgs)}  {os.path.basename(p)}  →  {vis_rel}")
                else:
                    fail += 1
                    self._append_log(f"[跳过] {i}/{len(imgs)}  {os.path.basename(p)}  未检测到两块区域")
            except Exception as e:
                fail += 1
                self._append_log(
                    f"[错误] {i}/{len(imgs)}  {os.path.basename(p)}  {e}\n{traceback.format_exc(limit=2)}"
                )

            self.progress.configure(value=i)
            self.status_var.set(f"进度：{i}/{len(imgs)}  成功 {ok}  失败 {fail}")
            self.update_idletasks()

        self.status_var.set("完成" if not self._stop_flag.is_set() else "任务已停止")
        self.open_btn.configure(state="normal")

    def _append_log(self, text: str):
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)


def start_ui():
    App().mainloop()


if __name__ == "__main__":
    start_ui()
