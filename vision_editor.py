# ============================================================
#  Vision Editor – Advanced Image Enhancement & Analysis
#  Project: GUI Image Editor (Task 1) + CV Accuracy Challenge (Task 2)
#  Tools: Python, OpenCV, Tkinter, NumPy, Matplotlib, Pillow
# ============================================================

# ─── Imports ─────────────────────────────────────────────────────────────────
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import pandas as pd

# ============================================================
#  IMAGE PROCESSING FUNCTIONS
# ============================================================

def adjust_brightness_contrast(image, brightness=0, contrast=1.0):
    """Point Operation: Adjust brightness (beta) and contrast (alpha)."""
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

def zoom_nearest(image, scale=2.0):
    """Geometric: Zoom using Nearest Neighbor interpolation."""
    h, w = image.shape[:2]
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    if scale > 1.0:
        start_y, start_x = (new_h - h) // 2, (new_w - w) // 2
        return resized[start_y:start_y+h, start_x:start_x+w]
    elif scale < 1.0:
        top, left = (h - new_h) // 2, (w - new_w) // 2
        bottom, right = h - new_h - top, w - new_w - left
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return resized

def zoom_bilinear(image, scale=2.0):
    """Geometric: Zoom using Bilinear interpolation."""
    h, w = image.shape[:2]
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if scale > 1.0:
        start_y, start_x = (new_h - h) // 2, (new_w - w) // 2
        return resized[start_y:start_y+h, start_x:start_x+w]
    elif scale < 1.0:
        top, left = (h - new_h) // 2, (w - new_w) // 2
        bottom, right = h - new_h - top, w - new_w - left
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return resized

def rotate_image(image, angle=45):
    """Geometric: Rotate image by given angle."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def histogram_equalization(image):
    """Enhancement: Equalize histogram via YCrCb (avoids color shift)."""
    if len(image.shape) == 2:
        return cv2.equalizeHist(image)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def gamma_correction(image, gamma=1.5):
    """Enhancement: Apply gamma correction via LUT."""
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)

def gaussian_blur(image, ksize=15):
    """Filter: Gaussian Blur for smoothing."""
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def sobel_edge(image):
    """Filter: Sobel edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobelx, sobely)
    result = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

def canny_edge(image, low=50, high=150):
    """Filter: Canny edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, low, high)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# ============================================================
#  TASK 2 – CV ACCURACY CHALLENGE HELPERS
# ============================================================

def create_synthetic_face(size=(400, 400)):
    """Draw a simple face using OpenCV shapes as a test image."""
    img = np.ones((*size, 3), dtype=np.uint8) * 200
    cx, cy = size[1] // 2, size[0] // 2
    cv2.ellipse(img, (cx, cy),      (100, 130), 0, 0, 360, (220, 185, 150), -1)
    cv2.ellipse(img, (cx-40, cy-30), (18, 12),  0, 0, 360, (50, 40, 30),    -1)
    cv2.ellipse(img, (cx+40, cy-30), (18, 12),  0, 0, 360, (50, 40, 30),    -1)
    cv2.circle(img,  (cx-40, cy-30), 6,                    (10, 10, 10),     -1)
    cv2.circle(img,  (cx+40, cy-30), 6,                    (10, 10, 10),     -1)
    cv2.ellipse(img, (cx, cy+10),    (12, 16),  0, 0, 360, (190, 155, 120),  -1)
    cv2.ellipse(img, (cx, cy+55),    (35, 15),  0, 0, 180, (160, 80, 80),     2)
    cv2.ellipse(img, (cx-40, cy-55), (22, 6),  -10, 0, 180, (60, 40, 20),    3)
    cv2.ellipse(img, (cx+40, cy-55), (22, 6),   10, 0, 180, (60, 40, 20),    3)
    return img

def degrade_image(image, darkness=0.35, noise_level=40):
    """Simulate a low-quality image: darken + add Gaussian noise."""
    darkened = (image.astype(np.float32) * darkness).clip(0, 255).astype(np.uint8)
    noise    = np.random.normal(0, noise_level, darkened.shape).astype(np.float32)
    noisy    = (darkened.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)
    return noisy

def enhance_image_task2(image):
    """Strong pipeline: Bilateral Filter → Gamma → CLAHE → Unsharp Mask."""
    # Step 1: Bilateral filter — removes noise while PRESERVING edges (unlike NlMeans which blurs faces)
    denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 2: Strong gamma correction to fully recover brightness from darkness=0.40 degradation
    # 0.40 darkening needs ~1/0.40 = 2.5x brightness boost → gamma ≈ 2.2
    gamma = 2.2
    inv_gamma = 1.0 / gamma
    lut = np.array([(i / 255.0) ** inv_gamma * 255
                    for i in range(256)], dtype=np.uint8)
    brightened = cv2.LUT(denoised, lut)

    # Step 3: CLAHE with higher clipLimit for stronger local contrast enhancement
    # This restores the luminance differences between facial features (eyes, nose, mouth)
    ycrcb = cv2.cvtColor(brightened, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # Step 4: Unsharp mask — sharpens facial edges so Haar Cascade can find them
    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2)
    sharpened = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)

    return sharpened

def detect_faces_task2(image, cascade, scale=1.1, min_neighbors=4, min_size=(30, 30)):
    """Run face detection. Returns (faces_array, inference_time_ms)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    t0   = time.perf_counter()
    faces = cascade.detectMultiScale(
        gray, scaleFactor=scale, minNeighbors=min_neighbors, minSize=min_size
    )
    elapsed = (time.perf_counter() - t0) * 1000
    return faces, elapsed

def draw_faces(image, faces, color=(0, 255, 100), thickness=2):
    """Draw bounding boxes around detected faces."""
    img_copy = image.copy()
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, thickness)
            cv2.putText(img_copy, 'Face', (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img_copy

# ============================================================
#  MAIN APPLICATION CLASS
# ============================================================

class VisionEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Vision Editor – Advanced Image Enhancement & Analysis')
        self.root.configure(bg='#1e1e2e')
        self.root.state('zoomed')

        # State
        self.original_image  = None
        self.processed_image = None

        # Style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Segoe UI', 10, 'bold'), padding=6,
                         background='#7c3aed', foreground='white')
        style.map('TButton', background=[('active', '#6d28d9')])
        style.configure('TScale', background='#1e1e2e')

        self._build_ui()

    # ── UI Construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg='#13131f', pady=10)
        header.pack(fill=tk.X)
        tk.Label(header, text='🔬 VISION EDITOR',
                 font=('Segoe UI', 20, 'bold'), fg='#a78bfa', bg='#13131f').pack()
        tk.Label(header, text='Advanced Image Enhancement & Computer Vision Analysis',
                 font=('Segoe UI', 10), fg='#6b7280', bg='#13131f').pack()

        # Main container
        main = tk.Frame(self.root, bg='#1e1e2e')
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # ── Left Panel: Controls ─────────────────────────────────────────────
        ctrl_outer = tk.Frame(main, bg='#13131f', width=270, relief=tk.RAISED, bd=1)
        ctrl_outer.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        ctrl_outer.pack_propagate(False)

        ctrl_canvas    = tk.Canvas(ctrl_outer, bg='#13131f', highlightthickness=0)
        ctrl_scrollbar = tk.Scrollbar(ctrl_outer, orient='vertical', command=ctrl_canvas.yview)
        ctrl_canvas.configure(yscrollcommand=ctrl_scrollbar.set)
        ctrl_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        ctrl_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ctrl = tk.Frame(ctrl_canvas, bg='#13131f')
        ctrl_canvas.create_window((0, 0), window=ctrl, anchor='nw')
        ctrl.bind('<Configure>', lambda e: ctrl_canvas.configure(
            scrollregion=ctrl_canvas.bbox('all')))
        ctrl_canvas.bind_all('<MouseWheel>',
            lambda e: ctrl_canvas.yview_scroll(int(-1 * (e.delta / 120)), 'units'))

        # ── Helpers for control widgets ──────────────────────────────────────

        def section_label(text):
            tk.Label(ctrl, text=text, font=('Segoe UI', 9, 'bold'),
                     bg='#2d2d4e', fg='#a78bfa', anchor='w', padx=8
                     ).pack(fill=tk.X, pady=(10, 2))

        def tool_btn(text, cmd):
            tk.Button(ctrl, text=text, command=cmd,
                      font=('Segoe UI', 9), bg='#2d2d4e', fg='#e2e8f0',
                      activebackground='#7c3aed', activeforeground='white',
                      relief=tk.FLAT, anchor=tk.W, padx=8, pady=3, cursor='hand2'
                      ).pack(fill=tk.X, padx=8, pady=1)

        def slider_row(label, from_, to, default, resolution=1):
            tk.Label(ctrl, text=label, font=('Segoe UI', 8),
                     fg='#9ca3af', bg='#13131f'
                     ).pack(anchor=tk.W, padx=12)
            var = tk.DoubleVar(value=default)
            ttk.Scale(ctrl, from_=from_, to=to, variable=var,
                      orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12, pady=(0, 6))
            return var

        # ── File ─────────────────────────────────────────────────────────────
        section_label('📂  FILE')
        tool_btn('📂 Load Image',    self.load_image)
        tool_btn('💾 Save Output',   self.save_image)
        tool_btn('↩ Reset to Original', self.reset_image)

        # ── Point Operations ─────────────────────────────────────────────────
        section_label('☀  BRIGHTNESS  &  CONTRAST')
        self.brightness_var = slider_row('Brightness  (-100 → +100)', -100, 100, 0)
        self.contrast_var   = slider_row('Contrast  (0.1 → 3.0)',      0.1,  3.0, 1.0)
        tool_btn('Apply Brightness / Contrast', self.apply_brightness_contrast)

        # ── Zoom (slider-based — no text entry parsing) ───────────────────────
        section_label('🔍  ZOOM')
        self.zoom_var = slider_row('Zoom Scale  (0.5 → 4.0)', 0.5, 4.0, 1.0)
        tool_btn('🔍 Zoom – Nearest Neighbor', self.apply_zoom_nearest)
        tool_btn('🔍 Zoom – Bilinear',         self.apply_zoom_bilinear)

        # ── Rotation ─────────────────────────────────────────────────────────
        section_label('↻  ROTATION')
        self.angle_var = slider_row('Angle  (0 → 360°)', 0, 360, 45)
        tool_btn('↻ Rotate Image', self.apply_rotate)

        # ── Enhancement ──────────────────────────────────────────────────────
        section_label('✨  ENHANCEMENT')
        self.gamma_var = slider_row('Gamma  (0.1 → 4.0)', 0.1, 4.0, 1.0)
        tool_btn('📊 Histogram Equalization', self.apply_hist_eq)
        tool_btn('γ  Gamma Correction',       self.apply_gamma)

        # ── Filters & Edges ──────────────────────────────────────────────────
        section_label('🌊  FILTERS  &  EDGES')
        self.blur_var = slider_row('Blur Kernel Size  (1 → 51)', 1, 51, 15)
        tool_btn('🌀 Gaussian Blur', self.apply_gaussian)
        tool_btn('〰 Sobel Edge',    self.apply_sobel)
        tool_btn('◈ Canny Edge',     self.apply_canny)

        # ── Task 2 ────────────────────────────────────────────────────────────
        section_label('🤖  TASK 2 — CV ACCURACY CHALLENGE')
        tk.Label(ctrl, text='Proves "Garbage In = Garbage Out"\nvia Haar Cascade face detection.',
                 font=('Segoe UI', 8), fg='#9ca3af', bg='#13131f', justify=tk.LEFT
                 ).pack(anchor=tk.W, padx=12, pady=(2, 4))
        tool_btn('▶ Run Face Detection Test', self.run_task2)

        # ── Center: Image Panels ─────────────────────────────────────────────
        center = tk.Frame(main, bg='#1e1e2e')
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        top_panels = tk.Frame(center, bg='#1e1e2e')
        top_panels.pack(fill=tk.BOTH, expand=True)

        orig_frame = tk.LabelFrame(top_panels, text=' Original Image ',
            font=('Segoe UI', 10, 'bold'), fg='#60a5fa', bg='#1e1e2e',
            labelanchor='n', bd=2, relief=tk.GROOVE)
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.orig_canvas = tk.Label(orig_frame, bg='#0f0f1a',
            text='Load an image to begin…', font=('Segoe UI', 12), fg='#4b5563')
        self.orig_canvas.pack(fill=tk.BOTH, expand=True)

        out_frame = tk.LabelFrame(top_panels, text=' Processed Image ',
            font=('Segoe UI', 10, 'bold'), fg='#34d399', bg='#1e1e2e',
            labelanchor='n', bd=2, relief=tk.GROOVE)
        out_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.out_canvas = tk.Label(out_frame, bg='#0f0f1a',
            text='Apply a tool to see results…', font=('Segoe UI', 12), fg='#4b5563')
        self.out_canvas.pack(fill=tk.BOTH, expand=True)

        # Histogram panel
        hist_frame = tk.Frame(center, bg='#13131f', height=200)
        hist_frame.pack(fill=tk.X, padx=5, pady=5)
        hist_frame.pack_propagate(False)

        self.fig, (self.ax_in, self.ax_out) = plt.subplots(
            1, 2, figsize=(10, 2), facecolor='#13131f')
        for ax in (self.ax_in, self.ax_out):
            ax.set_facecolor('#0f0f1a')
            ax.tick_params(colors='#6b7280', labelsize=7)
            for sp in ax.spines.values():
                sp.set_color('#374151')
        self.ax_in.set_title('Input Histogram',  color='#60a5fa', fontsize=9)
        self.ax_out.set_title('Output Histogram', color='#34d399', fontsize=9)
        self.fig.tight_layout(pad=0.5)

        self.hist_canvas = FigureCanvasTkAgg(self.fig, master=hist_frame)
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status = tk.StringVar(value='Ready. Load an image to begin.')
        tk.Label(self.root, textvariable=self.status,
                 font=('Segoe UI', 9), fg='#9ca3af', bg='#0a0a14',
                 anchor=tk.W, padx=10).pack(fill=tk.X, side=tk.BOTTOM)

    # ── Internal Helpers ─────────────────────────────────────────────────────

    def _cv_to_tk(self, img_bgr, max_w=500, max_h=380):
        h, w   = img_bgr.shape[:2]
        scale  = min(max_w / w, max_h / h, 1.0)
        rw, rh = int(w * scale), int(h * scale)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil     = Image.fromarray(img_rgb).resize((rw, rh), Image.LANCZOS)
        return ImageTk.PhotoImage(pil)

    def _draw_hist(self, ax, img, title):
        ax.clear()
        ax.set_facecolor('#0f0f1a')
        colors = ('#3b82f6', '#10b981', '#ef4444')
        if len(img.shape) == 3:
            for i, (ch, col) in enumerate(zip(cv2.split(img), colors)):
                hist = cv2.calcHist([ch], [0], None, [256], [0, 256])
                ax.plot(hist, color=col, linewidth=0.8, alpha=0.8)
        else:
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            ax.plot(hist, color='#a78bfa', linewidth=0.8)
        ax.set_title(title, color='#9ca3af', fontsize=8)
        ax.tick_params(colors='#4b5563', labelsize=6)
        for sp in ax.spines.values():
            sp.set_color('#374151')

    def _show_images(self):
        if self.original_image is None:
            return
        tk_orig = self._cv_to_tk(self.original_image)
        self.orig_canvas.config(image=tk_orig, text='')
        self.orig_canvas.image = tk_orig

        if self.processed_image is not None:
            tk_proc = self._cv_to_tk(self.processed_image)
            self.out_canvas.config(image=tk_proc, text='')
            self.out_canvas.image = tk_proc
            self._draw_hist(self.ax_out, self.processed_image, 'Output Histogram')

        self._draw_hist(self.ax_in, self.original_image, 'Input Histogram')
        self.hist_canvas.draw()

    def _apply(self, fn, desc):
        if self.original_image is None:
            messagebox.showwarning('No Image', 'Please load an image first!')
            return
        t0 = time.time()
        self.processed_image = fn(self.original_image.copy())
        elapsed = (time.time() - t0) * 1000
        self._show_images()
        h, w = self.processed_image.shape[:2]
        self.status.set(f'✔ Applied: {desc}  |  Size: {w}×{h}  |  Time: {elapsed:.1f} ms')

    # ── File Actions ─────────────────────────────────────────────────────────

    def load_image(self):
        path = filedialog.askopenfilename(
            title='Select Image',
            filetypes=[('Images', '*.jpg *.jpeg *.png *.bmp *.tiff *.webp'), ('All', '*.*')])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror('Error', 'Cannot read image file.')
            return
        self.original_image  = img
        self.processed_image = img.copy()
        self._show_images()
        h, w = img.shape[:2]
        self.status.set(f'Loaded: {path}  |  Size: {w}×{h}')

    def save_image(self):
        if self.processed_image is None:
            messagebox.showwarning('No Output', 'Nothing to save yet.')
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=[('PNG', '*.png'), ('JPEG', '*.jpg'), ('BMP', '*.bmp')])
        if path:
            cv2.imwrite(path, self.processed_image)
            self.status.set(f'Saved to: {path}')

    def reset_image(self):
        if self.original_image is None:
            return
        self.processed_image = self.original_image.copy()
        self._show_images()
        self.status.set('Image reset to original.')

    # ── Tool Callbacks ───────────────────────────────────────────────────────

    def apply_brightness_contrast(self):
        b = int(self.brightness_var.get())
        c = float(self.contrast_var.get())
        self._apply(lambda img: adjust_brightness_contrast(img, b, c),
                    f'Brightness={b}, Contrast={c:.2f}')

    def apply_zoom_nearest(self):
        s = float(self.zoom_var.get())
        self._apply(lambda img: zoom_nearest(img, s), f'Zoom Nearest ×{s:.1f}')

    def apply_zoom_bilinear(self):
        s = float(self.zoom_var.get())
        self._apply(lambda img: zoom_bilinear(img, s), f'Zoom Bilinear ×{s:.1f}')

    def apply_rotate(self):
        a = float(self.angle_var.get())
        self._apply(lambda img: rotate_image(img, a), f'Rotate {a:.0f}°')

    def apply_hist_eq(self):
        self._apply(histogram_equalization, 'Histogram Equalization')

    def apply_gamma(self):
        g = float(self.gamma_var.get())
        self._apply(lambda img: gamma_correction(img, g), f'Gamma={g:.2f}')

    def apply_gaussian(self):
        k = int(self.blur_var.get())
        self._apply(lambda img: gaussian_blur(img, k), f'Gaussian Blur k={k}')

    def apply_sobel(self):
        self._apply(sobel_edge, 'Sobel Edge Detection')

    def apply_canny(self):
        self._apply(canny_edge, 'Canny Edge Detection')

    # ── Task 2: CV Accuracy Challenge ────────────────────────────────────────

    def run_task2(self):
        # Load Haar Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            messagebox.showerror('Error', 'Haar Cascade failed to load!')
            return

        # Use loaded image if available, otherwise use synthetic face
        if self.original_image is not None:
            original_clean = self.original_image.copy()
        else:
            original_clean = create_synthetic_face()
            messagebox.showinfo(
                'No Image Loaded',
                'No image loaded — using synthetic face for demo.\n'
                'Load a real face photo for better results.'
            )

        # Run experiment
        degraded_image = degrade_image(original_clean, darkness=0.40, noise_level=40)
        enhanced_image = enhance_image_task2(degraded_image)

        faces_clean,    t_clean    = detect_faces_task2(original_clean, face_cascade)
        faces_degraded, t_degraded = detect_faces_task2(degraded_image, face_cascade)
        faces_enhanced, t_enhanced = detect_faces_task2(enhanced_image, face_cascade)

        gt_faces     = max(len(faces_clean), 1)
        acc_clean    = min(len(faces_clean)    / gt_faces * 100, 100)
        acc_degraded = min(len(faces_degraded) / gt_faces * 100, 100)
        acc_enhanced = min(len(faces_enhanced) / gt_faces * 100, 100)

        # Visualize
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle('CV Accuracy Challenge – Face Detection Results',
                     fontsize=16, fontweight='bold', color='#1e1e2e')
        fig.patch.set_facecolor('#f8fafc')

        images_info = [
            (original_clean, faces_clean,    f'Clean Image\nFaces: {len(faces_clean)} | Acc: {acc_clean:.0f}%',     '#16a34a'),
            (degraded_image, faces_degraded, f'Degraded Image\nFaces: {len(faces_degraded)} | Acc: {acc_degraded:.0f}%', '#dc2626'),
            (enhanced_image, faces_enhanced, f'Enhanced Image\nFaces: {len(faces_enhanced)} | Acc: {acc_enhanced:.0f}%', '#7c3aed'),
        ]

        for col, (img, faces, title, color) in enumerate(images_info):
            drawn = draw_faces(img, faces)
            axes[0, col].imshow(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))
            axes[0, col].set_title(title, fontsize=11, color=color, fontweight='bold')
            axes[0, col].axis('off')

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            axes[1, col].fill_between(range(256), hist.flatten(), color=color, alpha=0.4)
            axes[1, col].plot(hist, color=color, linewidth=1.2)
            axes[1, col].set_facecolor('#f1f5f9')
            axes[1, col].set_title('Histogram', fontsize=9, color='#374151')
            axes[1, col].set_xlim([0, 256])
            axes[1, col].tick_params(labelsize=7)

        plt.tight_layout()
        plt.show()

        # Comparison table
        results = pd.DataFrame({
            'Image Version':  ['Clean (Ground Truth)', 'Degraded (Low Quality)', 'Enhanced (Post-Processing)'],
            'Faces Detected': [len(faces_clean), len(faces_degraded), len(faces_enhanced)],
            'Accuracy (%)':   [f'{acc_clean:.1f}%', f'{acc_degraded:.1f}%', f'{acc_enhanced:.1f}%'],
            'Inference (ms)': [f'{t_clean:.1f}', f'{t_degraded:.1f}', f'{t_enhanced:.1f}'],
            'Improvement':    ['Baseline',
                               f'{acc_degraded - acc_clean:+.1f}%',
                               f'{acc_enhanced - acc_degraded:+.1f}% over degraded'],
        })

        print('\n' + '=' * 70)
        print(' COMPARISON TABLE – FACE DETECTION ACCURACY')
        print('=' * 70)
        print(results.to_string(index=False))
        print('=' * 70)
        print(f'\nConclusion: Enhancement improved accuracy by {acc_enhanced - acc_degraded:+.1f}%')
        print(f'Preprocessing overhead: {t_enhanced - t_degraded:.1f} ms (typically acceptable)')

        messagebox.showinfo(
            'Task 2 — Results',
            f'━━━ Clean (GT) ━━━━━━━━━━━━\n'
            f'  Faces: {len(faces_clean)}  |  Accuracy: {acc_clean:.0f}%  |  Time: {t_clean:.1f} ms\n\n'
            f'━━━ Degraded ━━━━━━━━━━━━━\n'
            f'  Faces: {len(faces_degraded)}  |  Accuracy: {acc_degraded:.0f}%  |  Time: {t_degraded:.1f} ms\n\n'
            f'━━━ Enhanced ━━━━━━━━━━━━━\n'
            f'  Faces: {len(faces_enhanced)}  |  Accuracy: {acc_enhanced:.0f}%  |  Time: {t_enhanced:.1f} ms\n\n'
            f'  Accuracy gain : {acc_enhanced - acc_degraded:+.1f}%\n'
            f'  See console for full table.'
        )
        self.status.set(
            f'Task 2 done — Degraded: {acc_degraded:.0f}%  →  Enhanced: {acc_enhanced:.0f}%'
        )


# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == '__main__':
    root = tk.Tk()
    app  = VisionEditorApp(root)
    root.mainloop()