# colorcard_kit/manual_select.py
import cv2
import numpy as np

class TwoRectSelector:
    def __init__(self, image_bgr, window_name="Select Ref (Top) then Sample (Bottom)"):
        self.img = image_bgr
        self.clone = image_bgr.copy()
        self.win = window_name
        self.rects = []  # [(x0,y0,x1,y1), ...]
        self.drawing = False
        self.x0 = self.y0 = 0

    def _mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.x0, self.y0 = x, y
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            img2 = self.clone.copy()
            cv2.rectangle(img2, (self.x0, self.y0), (x, y), (0, 255, 0) if len(self.rects)==0 else (255, 0, 0), 2)
            cv2.imshow(self.win, img2)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = x, y
            x0, y0 = self.x0, self.y0
            x_min, y_min = min(x0, x1), min(y0, y1)
            x_max, y_max = max(x0, x1), max(y0, y1)
            self.rects.append((x_min, y_min, x_max, y_max))
            color = (0, 255, 0) if len(self.rects)==1 else (255, 0, 0)
            cv2.rectangle(self.clone, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.imshow(self.win, self.clone)

    def run(self):
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, min(1600, self.img.shape[1]), min(1000, self.img.shape[0]))
        cv2.setMouseCallback(self.win, self._mouse)
        cv2.imshow(self.win, self.clone)

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 27:  # ESC
                self.rects = []
                cv2.destroyWindow(self.win)
                return None, None
            elif key == ord('r'):
                self.clone = self.img.copy()
                self.rects = []
                cv2.imshow(self.win, self.clone)
            elif key in (13, 10):  # Enter
                if len(self.rects) >= 2:
                    break

        cv2.destroyWindow(self.win)
        # 输出两个矩形的四点框
        def rect_to_box(r):
            x0, y0, x1, y1 = r
            return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=int)
        ref_box = rect_to_box(self.rects[0])
        sample_box = rect_to_box(self.rects[1])
        return ref_box, sample_box


def select_two_rects(image_bgr, max_side=1200):
    """
    入口：给定 BGR 图，缩放显示后让用户依次框选两块区域（上：Ref，下：Sample）
    返回：ref_box(4x2), sample_box(4x2)（坐标在原图尺度）
    """
    h, w = image_bgr.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        disp = cv2.resize(image_bgr, (int(w*scale), int(h*scale)))
    else:
        disp = image_bgr.copy()

    selector = TwoRectSelector(disp, "Select Ref (Top) then Sample (Bottom)  [Enter:OK / r:reset / Esc:cancel]")
    ref_box_s, sample_box_s = selector.run()
    if ref_box_s is None or sample_box_s is None:
        return None, None

    # 映射回原图坐标
    if scale != 1.0:
        ref_box = (ref_box_s.astype(np.float32) / scale).round().astype(int)
        sample_box = (sample_box_s.astype(np.float32) / scale).round().astype(int)
    else:
        ref_box, sample_box = ref_box_s, sample_box_s
    return ref_box, sample_box
