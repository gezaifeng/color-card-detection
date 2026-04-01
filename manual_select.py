# manual_select.py
import cv2
import numpy as np

# 用 PIL 画中文（解决 OpenCV putText 中文乱码）
from PIL import Image, ImageDraw, ImageFont


def _find_cn_font():
    """
    尝试在 Windows 常见路径中找到中文字体。
    找不到则返回 None（会退化为英文提示）。
    """
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",     # 微软雅黑
        r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\simhei.ttf",   # 黑体
        r"C:\Windows\Fonts\simsun.ttc",   # 宋体
    ]
    for p in candidates:
        try:
            f = ImageFont.truetype(p, 24)
            return p
        except Exception:
            pass
    return None


def _draw_text_cn(bgr, lines, org=(10, 10), font_path=None, font_size=24,
                  color=(255, 255, 255), stroke=(0, 0, 0), stroke_w=2):
    """
    在图像上绘制多行中文文本（PIL渲染），避免OpenCV中文乱码。
    lines: list[str]
    org: 左上角起点
    """
    if bgr is None:
        return bgr
    if not lines:
        return bgr

    # 若找不到字体就不画中文，避免报错
    if font_path is None:
        font_path = _find_cn_font()
    if font_path is None:
        return bgr  # 退化：不画中文（或你也可以改成画英文）

    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)

    font = ImageFont.truetype(font_path, font_size)

    x, y = org
    for s in lines:
        # 画描边
        if stroke_w > 0:
            draw.text((x, y), s, font=font, fill=color, stroke_width=stroke_w, stroke_fill=stroke)
        else:
            draw.text((x, y), s, font=font, fill=color)
        y += int(font_size * 1.35)

    out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return out


def _apply_tuning_bgr(
    bgr: np.ndarray,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    gamma: float = 1.0,
    use_clahe: bool = False,
    clahe_clip: float = 2.0,
    clahe_grid: int = 8,
) -> np.ndarray:
    """
    仅用于“显示增强”，不影响后续数据提取（提取仍走 pipeline/detect 的逻辑）。
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # contrast around 128
    v = (v - 128.0) * float(contrast) + 128.0
    # brightness gain
    v = v * float(brightness)
    v = np.clip(v, 0, 255)

    # gamma
    g = float(gamma)
    if abs(g - 1.0) > 1e-6:
        vv = (v / 255.0)
        vv = np.power(vv, g)
        v = np.clip(vv * 255.0, 0, 255)

    # CLAHE on V
    if use_clahe:
        v8 = v.astype(np.uint8)
        grid = int(max(2, min(32, clahe_grid)))
        clip = float(max(0.1, min(10.0, clahe_clip)))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
        v = clahe.apply(v8).astype(np.float32)

    # saturation
    s = np.clip(s * float(saturation), 0, 255)

    hsv2 = np.stack([h, s, v], axis=-1).astype(np.uint8)
    out = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
    return out


def _auto_tuning_from_gray(gray8: np.ndarray, target_median: float = 120.0, max_gain: float = 4.0):
    med = float(np.median(gray8.astype(np.float32)))
    med = max(med, 1.0)
    gain = float(target_median / med)
    gain = max(0.5, min(max_gain, gain))
    gamma = 0.85 if gain > 1.2 else 1.0
    return gain, gamma


class TwoRectSelectorTunableZoom:
    """
    两次框选：上 Ref 下 Sample
    - 显示增强（亮度/对比度/饱和度/伽马/CLAHE/自动矫正）
    - 鼠标滚轮缩放
    - 右键拖动平移（zoom后很有用）
    """

    def __init__(self, image_bgr, window_name="Manual Select", ctrl_name="Controls",
                 font_path=None):
        self.src = image_bgr
        self.win = window_name
        self.ctrl = ctrl_name
        self.font_path = font_path  # 可传入字体路径，不传就自动找

        self.rects_img = []  # 存在“原图坐标系”的矩形 (x0,y0,x1,y1)

        # drawing state（左键画框）
        self.drawing = False
        self.x0 = self.y0 = 0

        # pan state（右键拖动）
        self.panning = False
        self.pan_start = (0, 0)
        self.view_start = None

        # tuning params
        self.brightness = 1.0
        self.contrast = 1.0
        self.saturation = 1.0
        self.gamma = 1.0
        self.use_clahe = False
        self.clahe_clip = 2.0
        self.clahe_grid = 8

        # view ROI in tuned image coords
        h, w = self.src.shape[:2]
        self.view_x0 = 0.0
        self.view_y0 = 0.0
        self.view_w = float(w)
        self.view_h = float(h)

        # zoom config
        self.min_view_ratio = 0.15   # 最小可缩到原图的15%区域
        self.max_view_ratio = 1.0    # 最大为全图
        self.zoom_step = 0.85        # 每次滚轮缩放比例（<1缩小视野=放大）

        self._need_redraw = True
        self._tuned_cache = None

        # window size (display)
        self.disp_w = None
        self.disp_h = None

    # ---------- ctrl panel ----------
    def _create_ctrl_panel(self):
        cv2.namedWindow(self.ctrl, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.ctrl, 520, 420)

        def _noop(_):
            pass

        # 注意：OpenCV trackbar 名称用中文会乱码，所以这里全部用英文
        cv2.createTrackbar("Brightness x100", self.ctrl, 100, 300, _noop)   # 0.5~3.0
        cv2.createTrackbar("Contrast x100", self.ctrl, 100, 300, _noop)     # 0.5~3.0
        cv2.createTrackbar("Saturation x100", self.ctrl, 100, 300, _noop)   # 0.0~3.0
        cv2.createTrackbar("Gamma x100", self.ctrl, 100, 250, _noop)        # 0.3~2.5

        cv2.createTrackbar("CLAHE 0/1", self.ctrl, 0, 1, _noop)
        cv2.createTrackbar("CLAHE clip x10", self.ctrl, 20, 60, _noop)      # 0.1~6.0
        cv2.createTrackbar("CLAHE grid", self.ctrl, 8, 16, _noop)           # 2~16

        cv2.createTrackbar("Auto(1=run)", self.ctrl, 0, 1, _noop)
        cv2.createTrackbar("Target median", self.ctrl, 120, 220, _noop)
        cv2.createTrackbar("Max gain x100", self.ctrl, 400, 800, _noop)

    def _sync_params_from_trackbars(self):
        b = max(50, cv2.getTrackbarPos("Brightness x100", self.ctrl))
        c = max(50, cv2.getTrackbarPos("Contrast x100", self.ctrl))
        s = max(0,  cv2.getTrackbarPos("Saturation x100", self.ctrl))
        g = max(30, cv2.getTrackbarPos("Gamma x100", self.ctrl))

        use_clahe = cv2.getTrackbarPos("CLAHE 0/1", self.ctrl)
        clip10 = max(1, cv2.getTrackbarPos("CLAHE clip x10", self.ctrl))
        grid = max(2, cv2.getTrackbarPos("CLAHE grid", self.ctrl))

        new_b = b / 100.0
        new_c = c / 100.0
        new_s = s / 100.0
        new_g = g / 100.0

        new_use = bool(use_clahe)
        new_clip = max(0.1, clip10 / 10.0)
        new_grid = int(grid)

        changed = (
            abs(self.brightness - new_b) > 1e-3 or
            abs(self.contrast - new_c) > 1e-3 or
            abs(self.saturation - new_s) > 1e-3 or
            abs(self.gamma - new_g) > 1e-3 or
            self.use_clahe != new_use or
            abs(self.clahe_clip - new_clip) > 1e-3 or
            self.clahe_grid != new_grid
        )
        if changed:
            self.brightness = new_b
            self.contrast = new_c
            self.saturation = new_s
            self.gamma = new_g
            self.use_clahe = new_use
            self.clahe_clip = new_clip
            self.clahe_grid = new_grid
            self._need_redraw = True
            self._tuned_cache = None  # invalidate

    def _trigger_auto(self):
        # auto基于当前tuning后的灰度
        tuned = self._get_tuned()
        gray = cv2.cvtColor(tuned, cv2.COLOR_BGR2GRAY)

        target = float(cv2.getTrackbarPos("Target median", self.ctrl))
        max_gain100 = float(cv2.getTrackbarPos("Max gain x100", self.ctrl))
        max_gain = max(1.0, max_gain100 / 100.0)

        gain, gamma = _auto_tuning_from_gray(gray, target_median=target, max_gain=max_gain)

        new_b = int(np.clip(self.brightness * gain * 100.0, 50, 300))
        new_g = int(np.clip(gamma * 100.0, 30, 250))

        cv2.setTrackbarPos("Brightness x100", self.ctrl, new_b)
        cv2.setTrackbarPos("Gamma x100", self.ctrl, new_g)

        # 暗图建议开CLAHE
        if gain > 1.6:
            cv2.setTrackbarPos("CLAHE 0/1", self.ctrl, 1)
            cv2.setTrackbarPos("CLAHE clip x10", self.ctrl, 25)
            cv2.setTrackbarPos("CLAHE grid", self.ctrl, 8)

        self._need_redraw = True

    # ---------- view mapping ----------
    def _get_tuned(self):
        if self._tuned_cache is None:
            self._tuned_cache = _apply_tuning_bgr(
                self.src,
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                gamma=self.gamma,
                use_clahe=self.use_clahe,
                clahe_clip=self.clahe_clip,
                clahe_grid=self.clahe_grid,
            )
        return self._tuned_cache

    def _clamp_view(self):
        h, w = self.src.shape[:2]
        self.view_w = float(np.clip(self.view_w, w * self.min_view_ratio, w * self.max_view_ratio))
        self.view_h = float(np.clip(self.view_h, h * self.min_view_ratio, h * self.max_view_ratio))

        self.view_x0 = float(np.clip(self.view_x0, 0, max(0.0, w - self.view_w)))
        self.view_y0 = float(np.clip(self.view_y0, 0, max(0.0, h - self.view_h)))

    def _disp_to_img(self, x_disp, y_disp):
        # display -> image coords (in tuned/src coords)
        x_img = self.view_x0 + (x_disp / max(1, self.disp_w)) * self.view_w
        y_img = self.view_y0 + (y_disp / max(1, self.disp_h)) * self.view_h
        return int(round(x_img)), int(round(y_img))

    def _img_to_disp(self, x_img, y_img):
        x_disp = int(round((x_img - self.view_x0) / max(1e-6, self.view_w) * self.disp_w))
        y_disp = int(round((y_img - self.view_y0) / max(1e-6, self.view_h) * self.disp_h))
        return x_disp, y_disp

    def _render_view(self):
        tuned = self._get_tuned()
        h, w = tuned.shape[:2]
        self._clamp_view()

        x0 = int(round(self.view_x0))
        y0 = int(round(self.view_y0))
        x1 = int(round(self.view_x0 + self.view_w))
        y1 = int(round(self.view_y0 + self.view_h))
        x1 = min(x1, w)
        y1 = min(y1, h)

        roi = tuned[y0:y1, x0:x1]
        view = cv2.resize(roi, (self.disp_w, self.disp_h), interpolation=cv2.INTER_LINEAR)
        return view

    def _compose_frame(self, temp_rect_img=None):
        frame = self._render_view()

        # draw confirmed rects (convert img->disp)
        for k, r in enumerate(self.rects_img):
            x0, y0, x1, y1 = r
            c = (0, 255, 0) if k == 0 else (255, 0, 0)
            p0 = self._img_to_disp(x0, y0)
            p1 = self._img_to_disp(x1, y1)
            cv2.rectangle(frame, p0, p1, c, 2)

        # draw temp rect
        if temp_rect_img is not None:
            x0, y0, x1, y1 = temp_rect_img
            c = (0, 255, 0) if len(self.rects_img) == 0 else (255, 0, 0)
            p0 = self._img_to_disp(x0, y0)
            p1 = self._img_to_disp(x1, y1)
            cv2.rectangle(frame, p0, p1, c, 2)

        # Chinese overlay (PIL)
        lines = [
            "手动框选：先框选上方 Ref，再框选下方 Sample",
            "操作：左键拖拽画框 / 右键拖拽平移 / 滚轮缩放 / Enter确认 / r重置 / Esc取消 / a自动矫正",
            f"参数：亮度{self.brightness:.2f} 对比度{self.contrast:.2f} 饱和度{self.saturation:.2f} 伽马{self.gamma:.2f}  CLAHE:{'开' if self.use_clahe else '关'}",
        ]
        frame = _draw_text_cn(frame, lines, org=(10, 10), font_path=self.font_path, font_size=24)

        return frame

    # ---------- mouse events ----------
    def _mouse(self, event, x, y, flags, param):
        # zoom by wheel
        if event == cv2.EVENT_MOUSEWHEEL:
            # OpenCV: flags >0 means forward, <0 means backward (platform dependent but OK on Win)
            delta = 1 if flags > 0 else -1

            # zoom around cursor position
            cx_img, cy_img = self._disp_to_img(x, y)
            old_w, old_h = self.view_w, self.view_h

            if delta > 0:
                # zoom in (view smaller)
                new_w = old_w * self.zoom_step
                new_h = old_h * self.zoom_step
            else:
                # zoom out (view larger)
                new_w = old_w / self.zoom_step
                new_h = old_h / self.zoom_step

            # keep cursor position stable: adjust view_x0/view_y0
            # ratio in view
            rx = (cx_img - self.view_x0) / max(1e-6, old_w)
            ry = (cy_img - self.view_y0) / max(1e-6, old_h)

            self.view_w, self.view_h = new_w, new_h
            self.view_x0 = cx_img - rx * self.view_w
            self.view_y0 = cy_img - ry * self.view_h
            self._clamp_view()
            self._need_redraw = True
            return

        # right button pan
        if event == cv2.EVENT_RBUTTONDOWN:
            self.panning = True
            self.pan_start = (x, y)
            self.view_start = (self.view_x0, self.view_y0)
            return

        if event == cv2.EVENT_MOUSEMOVE and self.panning:
            dx = x - self.pan_start[0]
            dy = y - self.pan_start[1]
            # disp delta -> img delta
            self.view_x0 = self.view_start[0] - dx / max(1, self.disp_w) * self.view_w
            self.view_y0 = self.view_start[1] - dy / max(1, self.disp_h) * self.view_h
            self._clamp_view()
            self._need_redraw = True
            return

        if event == cv2.EVENT_RBUTTONUP:
            self.panning = False
            return

        # left button draw rect
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.x0, self.y0 = self._disp_to_img(x, y)
            return

        if event == cv2.EVENT_MOUSEMOVE and self.drawing:
            x1, y1 = self._disp_to_img(x, y)
            x_min, y_min = min(self.x0, x1), min(self.y0, y1)
            x_max, y_max = max(self.x0, x1), max(self.y0, y1)
            frame = self._compose_frame(temp_rect_img=(x_min, y_min, x_max, y_max))
            cv2.imshow(self.win, frame)
            return

        if event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = self._disp_to_img(x, y)
            x_min, y_min = min(self.x0, x1), min(self.y0, y1)
            x_max, y_max = max(self.x0, x1), max(self.y0, y1)

            if (x_max - x_min) < 5 or (y_max - y_min) < 5:
                self._need_redraw = True
                return

            self.rects_img.append((x_min, y_min, x_max, y_max))
            self._need_redraw = True
            return

    # ---------- main loop ----------
    def run(self):
        # window sizes
        h, w = self.src.shape[:2]
        self.disp_w = min(1600, w)
        self.disp_h = min(1000, h)

        # OpenCV窗口标题也可能乱码，所以用英文标题最稳
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, self.disp_w, self.disp_h)
        cv2.setMouseCallback(self.win, self._mouse)

        self._create_ctrl_panel()

        cv2.imshow(self.win, self._compose_frame())

        while True:
            self._sync_params_from_trackbars()

            # auto button
            if cv2.getTrackbarPos("Auto(1=run)", self.ctrl) == 1:
                self._trigger_auto()
                cv2.setTrackbarPos("Auto(1=run)", self.ctrl, 0)

            if self._need_redraw and (not self.drawing) and (not self.panning):
                cv2.imshow(self.win, self._compose_frame())
                self._need_redraw = False

            key = cv2.waitKey(20) & 0xFF
            if key == 27:  # ESC
                self.rects_img = []
                cv2.destroyWindow(self.win)
                cv2.destroyWindow(self.ctrl)
                return None, None

            elif key == ord('r'):
                self.rects_img = []
                self._need_redraw = True

            elif key == ord('a'):
                self._trigger_auto()
                self._need_redraw = True

            elif key in (13, 10):  # Enter
                if len(self.rects_img) >= 2:
                    break

        cv2.destroyWindow(self.win)
        cv2.destroyWindow(self.ctrl)

        def rect_to_box(r):
            x0, y0, x1, y1 = r
            return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=int)

        ref_box = rect_to_box(self.rects_img[0])
        sample_box = rect_to_box(self.rects_img[1])
        return ref_box, sample_box


def select_two_rects(image_bgr, max_side=1200, font_path=None):
    """
    入口：给定 BGR 图，缩放显示后让用户依次框选两块区域（上：Ref，下：Sample）
    返回：ref_box(4x2), sample_box(4x2)（坐标在原图尺度）
    """
    h, w = image_bgr.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        disp = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
    else:
        disp = image_bgr.copy()

    selector = TwoRectSelectorTunableZoom(
        disp,
        window_name="ManualSelect",
        ctrl_name="Controls",
        font_path=font_path
    )
    ref_box_s, sample_box_s = selector.run()
    if ref_box_s is None or sample_box_s is None:
        return None, None

    # map back to original scale
    if scale != 1.0:
        ref_box = (ref_box_s.astype(np.float32) / scale).round().astype(int)
        sample_box = (sample_box_s.astype(np.float32) / scale).round().astype(int)
    else:
        ref_box, sample_box = ref_box_s, sample_box_s

    return ref_box, sample_box
