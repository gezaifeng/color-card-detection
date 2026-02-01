# colorcard_kit/ui_main.py
import os
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

from config import PipelineConfig
from io_utils import find_images
from pipeline import process_single

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("色卡识别与特征导出 · 自动/手动 · RAW 支持")
        self.geometry("900x720")
        self.resizable(True, True)
        self._stop_flag = threading.Event()
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 6, "pady": 4}

        io = ttk.LabelFrame(self, text="输入 / 输出")
        io.pack(fill="x", **pad)
        ttk.Label(io, text="输入目录：").grid(row=0, column=0, sticky="e")
        self.in_entry = ttk.Entry(io, width=64); self.in_entry.grid(row=0, column=1, sticky="we")
        ttk.Button(io, text="选择", command=self._choose_in).grid(row=0, column=2)
        ttk.Label(io, text="输出目录：").grid(row=1, column=0, sticky="e")
        self.out_entry = ttk.Entry(io, width=64); self.out_entry.grid(row=1, column=1, sticky="we")
        ttk.Button(io, text="选择", command=self._choose_out).grid(row=1, column=2)
        io.columnconfigure(1, weight=1)

        # 参数
        prm = ttk.LabelFrame(self, text="参数设置")
        prm.pack(fill="x", **pad)

        # 左列
        self.var_grid_rows = tk.IntVar(value=4)
        self.var_grid_cols = tk.IntVar(value=6)
        self.var_target_height = tk.IntVar(value=512)
        self.var_sample_count = tk.IntVar(value=100)
        ttk.Label(prm, text="grid_rows").grid(row=0, column=0, sticky="e")
        ttk.Entry(prm, textvariable=self.var_grid_rows, width=10).grid(row=0, column=1, sticky="w")
        ttk.Label(prm, text="grid_cols").grid(row=1, column=0, sticky="e")
        ttk.Entry(prm, textvariable=self.var_grid_cols, width=10).grid(row=1, column=1, sticky="w")
        ttk.Label(prm, text="target_height").grid(row=2, column=0, sticky="e")
        ttk.Entry(prm, textvariable=self.var_target_height, width=10).grid(row=2, column=1, sticky="w")
        ttk.Label(prm, text="sample_count").grid(row=3, column=0, sticky="e")
        ttk.Entry(prm, textvariable=self.var_sample_count, width=10).grid(row=3, column=1, sticky="w")

        # 中列
        self.var_sobel_ksize = tk.IntVar(value=3)
        self.var_edge_thresh = tk.IntVar(value=50)
        self.var_card_crop_long = tk.DoubleVar(value=0.03)
        self.var_card_crop_short = tk.DoubleVar(value=0.06)
        ttk.Label(prm, text="sobel_ksize").grid(row=0, column=2, sticky="e")
        ttk.Entry(prm, textvariable=self.var_sobel_ksize, width=10).grid(row=0, column=3, sticky="w")
        ttk.Label(prm, text="edge_thresh").grid(row=1, column=2, sticky="e")
        ttk.Entry(prm, textvariable=self.var_edge_thresh, width=10).grid(row=1, column=3, sticky="w")
        ttk.Label(prm, text="card_crop_long").grid(row=2, column=2, sticky="e")
        ttk.Entry(prm, textvariable=self.var_card_crop_long, width=10).grid(row=2, column=3, sticky="w")
        ttk.Label(prm, text="card_crop_short").grid(row=3, column=2, sticky="e")
        ttk.Entry(prm, textvariable=self.var_card_crop_short, width=10).grid(row=3, column=3, sticky="w")

        # 右列
        self.var_sample_center_area = tk.DoubleVar(value=0.40)
        ttk.Label(prm, text="sample_center_area").grid(row=0, column=4, sticky="e")
        ttk.Entry(prm, textvariable=self.var_sample_center_area, width=10).grid(row=0, column=5, sticky="w")
        ttk.Label(prm, text="(0~1，中间采样面积比例)").grid(row=0, column=6, sticky="w")

        # 特征模式
        feat = ttk.LabelFrame(self, text="特征输出")
        feat.pack(fill="x", **pad)
        self.var_feature_mode = tk.StringVar(value="log_ratio")
        ttk.Label(feat, text="feature_mode").grid(row=0, column=0, sticky="e")
        ttk.OptionMenu(feat, self.var_feature_mode, "log_ratio", "log_ratio", "ratio", "multi").grid(row=0, column=1, sticky="w")
        self.var_norm = tk.BooleanVar(value=True)
        ttk.Checkbutton(feat, text="per_image_channel_norm", variable=self.var_norm).grid(row=0, column=2, sticky="w")
        self.var_save_extras = tk.BooleanVar(value=True)
        ttk.Checkbutton(feat, text="save_extras (ratio/logratio)", variable=self.var_save_extras).grid(row=0, column=3, sticky="w")

        # 手动回退 & RAW
        extf = ttk.LabelFrame(self, text="扩展功能")
        extf.pack(fill="x", **pad)
        self.var_allow_manual = tk.BooleanVar(value=True)
        self.var_force_manual = tk.BooleanVar(value=False)
        self.var_manual_downscale = tk.IntVar(value=1200)
        ttk.Checkbutton(extf, text="allow_manual_fallback", variable=self.var_allow_manual).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(extf, text="force_manual", variable=self.var_force_manual).grid(row=0, column=1, sticky="w")
        ttk.Label(extf, text="manual_downscale").grid(row=0, column=2, sticky="e")
        ttk.Entry(extf, textvariable=self.var_manual_downscale, width=10).grid(row=0, column=3, sticky="w")

        self.var_prefer_raw = tk.BooleanVar(value=True)
        self.var_raw_wb = tk.BooleanVar(value=True)
        self.var_raw_bps = tk.IntVar(value=8)
        ttk.Checkbutton(extf, text="prefer_raw_linear", variable=self.var_prefer_raw).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(extf, text="raw_use_camera_wb", variable=self.var_raw_wb).grid(row=1, column=1, sticky="w")
        ttk.Label(extf, text="raw_output_bps").grid(row=1, column=2, sticky="e")
        ttk.Entry(extf, textvariable=self.var_raw_bps, width=10).grid(row=1, column=3, sticky="w")

        # 控制区
        ctrl = ttk.Frame(self); ctrl.pack(fill="x", **pad)
        self.run_btn = ttk.Button(ctrl, text="开始批处理", command=self._on_start); self.run_btn.pack(side="left")
        ttk.Button(ctrl, text="停止", command=self._on_stop).pack(side="left", padx=6)
        self.open_btn = ttk.Button(ctrl, text="打开输出目录", command=self._open_out, state="disabled"); self.open_btn.pack(side="left")
        self.progress = ttk.Progressbar(ctrl, mode="determinate"); self.progress.pack(side="right", fill="x", expand=True)

        # 日志
        logf = ttk.LabelFrame(self, text="日志"); logf.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(logf, height=18, wrap="word"); self.log.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(logf, command=self.log.yview); scroll.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=scroll.set)

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(fill="x")

    # 事件
    def _choose_in(self):
        d = filedialog.askdirectory(title="选择输入目录")
        if d: self.in_entry.delete(0, tk.END); self.in_entry.insert(0, d)

    def _choose_out(self):
        d = filedialog.askdirectory(title="选择输出目录")
        if d: self.out_entry.delete(0, tk.END); self.out_entry.insert(0, d)

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
            messagebox.showwarning("提示", "任务正在进行中"); return
        inp, outp = self.in_entry.get().strip(), self.out_entry.get().strip()
        if not inp or not os.path.isdir(inp): messagebox.showerror("错误", "请输入有效的【输入目录】"); return
        if not outp: messagebox.showerror("错误", "请输入【输出目录】"); return
        os.makedirs(outp, exist_ok=True)
        try:
            cfg = self._make_config()
        except ValueError as e:
            messagebox.showerror("参数错误", str(e)); return

        imgs = find_images(inp)
        if not imgs: messagebox.showwarning("提示", "未在输入目录找到图像文件"); return

        self._stop_flag = threading.Event()
        self.progress.configure(maximum=len(imgs), value=0)
        self.log.delete("1.0", tk.END)
        self.status_var.set(f"准备开始：共 {len(imgs)} 张")
        self.open_btn.configure(state="disabled")

        self._worker = threading.Thread(target=self._run_worker, args=(imgs, inp, outp, cfg), daemon=True)
        self._worker.start()

    def _on_stop(self):
        if self._worker and self._worker.is_alive():
            self._stop_flag.set(); self.status_var.set("请求停止中…")
        else:
            self.status_var.set("当前无运行中的任务")

    def _make_config(self) -> PipelineConfig:
        rows = int(self.var_grid_rows.get()); cols = int(self.var_grid_cols.get())
        th = int(self.var_target_height.get()); sc = int(self.var_sample_count.get())
        ksz = int(self.var_sobel_ksize.get()); thr = int(self.var_edge_thresh.get())
        ccl = float(self.var_card_crop_long.get()); ccs = float(self.var_card_crop_short.get())
        area = float(self.var_sample_center_area.get())
        if rows <= 0 or cols <= 0: raise ValueError("grid_rows / grid_cols 必须为正整数")
        if th < 64: raise ValueError("target_height 过小（建议 ≥ 256）")
        if sc <= 0: raise ValueError("sample_count 必须为正整数")
        if ksz not in (1,3,5,7): raise ValueError("sobel_ksize 需为奇数（1/3/5/7）")
        if thr <= 0: raise ValueError("edge_thresh 必须为正整数")
        if not (0.0 <= ccl < 0.5) or not (0.0 <= ccs < 0.5): raise ValueError("card_crop_* 建议在 [0,0.5) 内")
        if not (0.0 < area <= 1.0): raise ValueError("sample_center_area 需在 (0,1] 内")

        cfg = PipelineConfig(
            grid_rows=rows, grid_cols=cols, target_height=th, sample_count=sc,
            sobel_ksize=ksz, edge_thresh=thr,
            sample_center_area=area,
            feature_mode=self.var_feature_mode.get(),
            per_image_channel_norm=bool(self.var_norm.get()),
            save_extras=bool(self.var_save_extras.get()),
            allow_manual=bool(self.var_allow_manual.get()),
            force_manual=bool(self.var_force_manual.get()),
            manual_downscale=int(self.var_manual_downscale.get()),
            prefer_raw_linear=bool(self.var_prefer_raw.get()),
            raw_use_camera_wb=bool(self.var_raw_wb.get()),
            raw_output_bps=int(self.var_raw_bps.get()),
        )
        cfg.card_crop_long = ccl; cfg.card_crop_short = ccs
        return cfg

    def _run_worker(self, imgs, inp, outp, cfg: PipelineConfig):
        ok = fail = 0
        for i, p in enumerate(imgs, 1):
            if self._stop_flag.is_set():
                self._append_log(f"[停止] 已中断，最后处理到：{i-1}/{len(imgs)}"); break
            try:
                res = process_single(p, inp, outp, cfg)
                if res:
                    ok += 1
                    vis_rel = os.path.relpath(res["vis"], outp) if "vis" in res else "(no vis)"
                    self._append_log(f"[OK] {i}/{len(imgs)}  {os.path.basename(p)}  →  {vis_rel}")
                else:
                    fail += 1
                    self._append_log(f"[SKIP] {i}/{len(imgs)}  {os.path.basename(p)}  未检测到两块区域")
            except Exception as e:
                fail += 1
                self._append_log(f"[ERR] {i}/{len(imgs)}  {os.path.basename(p)}  {e}\n{traceback.format_exc(limit=2)}")
            self.progress.configure(value=i)
            self.status_var.set(f"进度：{i}/{len(imgs)}  成功 {ok}  失败 {fail}")
            self.update_idletasks()

        self.status_var.set("完成" if not self._stop_flag.is_set() else "任务已停止")
        self.open_btn.configure(state="normal")

    def _append_log(self, text: str):
        self.log.insert(tk.END, text + "\n"); self.log.see(tk.END)

def start_ui():
    App().mainloop()

if __name__ == "__main__":
    start_ui()
