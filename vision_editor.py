# ============================================================
#  Vision Editor – Advanced Image Enhancement & Analysis
#  Project: GUI Image Editor (Task 1) + CV Accuracy Challenge (Task 2)
#  Tools: Python, OpenCV, Tkinter, NumPy, Matplotlib, Pillow
# ============================================================

# ─── Imports ─────────────────────────────────────────────────────────────────
import cv2  # OpenCV: core image processing library (read, resize, filter, detect)
import numpy as np  # NumPy: fast array/matrix math — images are just numpy arrays
from PIL import Image, ImageTk  # Pillow: converts OpenCV images to Tkinter-compatible PhotoImage
import tkinter as tk  # Tkinter: Python's built-in GUI framework
from tkinter import ttk, filedialog, \
    messagebox  # ttk: themed widgets | filedialog: open/save dialogs | messagebox: popups
import matplotlib  # Matplotlib: plotting library for histograms

matplotlib.use('TkAgg')  # TkAgg backend: lets matplotlib render inside a Tkinter window
import matplotlib.pyplot as plt  # plt: high-level API for creating figures and plots
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Bridge: embed matplotlib figure inside Tkinter widget
import time  # time: measure processing and inference duration in milliseconds
import pandas as pd  # pandas: build and print the comparison table in Task 2


# ============================================================
#  IMAGE PROCESSING FUNCTIONS
#  All functions: take an OpenCV BGR image (numpy array), return a processed image.
#  They are pure/standalone — no GUI dependency, no side effects.
# ============================================================

def adjust_brightness_contrast(image, brightness=0, contrast=1.0):
    """Point Operation: Adjust brightness (beta) and contrast (alpha).

    Formula per pixel:  output = alpha * pixel + beta
        alpha (contrast)  > 1 → more contrast  |  < 1 → less contrast
        beta  (brightness)> 0 → brighter        |  < 0 → darker
    cv2.convertScaleAbs applies the formula and clips result to [0, 255].
    """
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)


def zoom_nearest(image, scale=2.0):
    """Geometric Transformation: Zoom using Nearest Neighbor interpolation.

    Nearest Neighbor: each new pixel copies the value of the closest original pixel.
    - Fastest method, no blending between pixels.
    - Produces blocky/pixelated result at high zoom (aliasing visible).

    After resize:
    - scale > 1 (zoom in):  crop the center region back to original size.
    - scale < 1 (zoom out): pad edges with black to keep original canvas size.
    max(1, ...) prevents zero-dimension crash for very small scale values.
    """
    h, w = image.shape[:2]
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))  # Compute new dimensions, minimum 1px
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)  # Nearest neighbor resize
    if scale > 1.0:
        # Zoom in: the resized image is larger — crop center to match original size
        start_y, start_x = (new_h - h) // 2, (new_w - w) // 2
        return resized[start_y:start_y + h, start_x:start_x + w]
    elif scale < 1.0:
        # Zoom out: the resized image is smaller — pad with black border to restore original size
        top, left = (h - new_h) // 2, (w - new_w) // 2
        bottom, right = h - new_h - top, w - new_w - left
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return resized  # scale == 1.0: no change needed


def zoom_bilinear(image, scale=2.0):
    """Geometric Transformation: Zoom using Bilinear interpolation.

    Bilinear: each new pixel is a weighted average of the 4 nearest original pixels.
    - Smoother result than Nearest Neighbor — no blockiness.
    - Slightly slower due to interpolation math.
    - Standard choice for photo upscaling in real-world applications.

    Same center-crop / black-padding logic as zoom_nearest to keep canvas size consistent.
    """
    h, w = image.shape[:2]
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))  # New dimensions with safety floor
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)  # Bilinear resize
    if scale > 1.0:
        # Crop center back to original canvas size
        start_y, start_x = (new_h - h) // 2, (new_w - w) // 2
        return resized[start_y:start_y + h, start_x:start_x + w]
    elif scale < 1.0:
        # Pad with black to restore original canvas size
        top, left = (h - new_h) // 2, (w - new_w) // 2
        bottom, right = h - new_h - top, w - new_w - left
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return resized


def rotate_image(image, angle=45):
    """Geometric Transformation: Rotate image around its center by a given angle.

    Steps:
    1. Find the center pixel (pivot point for rotation).
    2. Build a 2×3 rotation matrix M:
       - angle: degrees counter-clockwise
       - scale=1.0: no zoom/shrink applied during rotation
    3. Apply affine transformation (warpAffine) using matrix M.
       - Output canvas size stays same as input (w, h) → corners may be cropped.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)  # Center pixel = rotation pivot
    M = cv2.getRotationMatrix2D(center, angle, 1.0)  # 2×3 affine rotation matrix
    return cv2.warpAffine(image, M, (w, h))  # Apply rotation, keep original canvas size


def histogram_equalization(image):
    """Enhancement: Redistribute pixel intensities to improve global contrast.

    Why YCrCb instead of direct RGB?
    - Equalizing R, G, B channels separately shifts the color balance → unnatural hue.
    - YCrCb separates luminance (Y) from chrominance (Cr, Cb).
    - We equalize ONLY the Y channel (brightness) → contrast improves, colors stay natural.

    Grayscale path: equalize directly (single channel, no color conversion needed).
    """
    if len(image.shape) == 2:
        # Grayscale image: equalize the single intensity channel directly
        return cv2.equalizeHist(image)
    # Color image: work in YCrCb space to preserve hue
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)  # Convert BGR → YCrCb
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])  # Equalize Y channel (index 0) only
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)  # Convert back to BGR


def gamma_correction(image, gamma=1.5):
    """Enhancement: Non-linear brightness adjustment using the power-law formula.

    Formula:  output = (input / 255) ^ (1/gamma) * 255

    Behavior:
    - gamma > 1 → image gets BRIGHTER (lifts dark/shadow areas)
    - gamma < 1 → image gets DARKER
    - gamma = 1 → no change

    LUT (Look-Up Table) approach:
    - Pre-compute the corrected value for all 256 possible input intensities [0..255].
    - cv2.LUT replaces every pixel value instantly using the table.
    - Much faster than computing the power function per pixel at runtime.
    """
    inv_gamma = 1.0 / gamma
    # Build a 256-entry lookup table: index = input intensity, value = corrected intensity
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)  # Apply the lookup table to every pixel in the image


def gaussian_blur(image, ksize=15):
    """Spatial Filter: Smooth the image using a Gaussian-weighted kernel.

    Gaussian kernel: center pixel has the highest weight; weight decreases with distance.
    - Reduces high-frequency noise and fine detail.
    - Kernel size (ksize) MUST be an odd number — enforced by the line below.
    - Larger ksize → stronger blur / more smoothing.
    - sigma=0 → OpenCV auto-calculates sigma from ksize.

    Real-world uses: noise reduction before edge detection, background blurring.
    """
    ksize = ksize if ksize % 2 == 1 else ksize + 1  # Enforce odd kernel size
    return cv2.GaussianBlur(image, (ksize, ksize), 0)  # Apply Gaussian blur


def sobel_edge(image):
    """Spatial Filter / Edge Detection: Detect edges using the Sobel gradient operator.

    Sobel computes the image gradient (rate of intensity change):
    - sobelx: horizontal derivative → detects VERTICAL edges
    - sobely: vertical derivative   → detects HORIZONTAL edges
    - magnitude = sqrt(sobelx² + sobely²) → combined edge strength at each pixel

    CV_64F: 64-bit float captures both positive and negative gradient values.
    normalize: scales the result back to displayable range [0, 255].
    Converted to BGR at the end to stay compatible with the rest of the pipeline.
    """
    # Sobel requires single-channel input → convert color image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in X direction (vertical edges)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in Y direction (horizontal edges)
    magnitude = cv2.magnitude(sobelx, sobely)  # Combined edge magnitude map
    # Normalize float magnitude to uint8 [0, 255] for display
    result = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)  # Return as 3-channel BGR


def canny_edge(image, low=50, high=150):
    """Spatial Filter / Edge Detection: Multi-stage Canny edge detector.

    Canny internal pipeline (all handled by cv2.Canny):
    1. Gaussian blur        — suppresses noise before gradient computation
    2. Sobel gradients      — compute magnitude and direction of intensity change
    3. Non-max suppression  — thin edges to exactly 1 pixel wide
    4. Double thresholding:
         > high threshold   → strong edge (always kept)
         low < x < high     → weak edge (kept only if connected to a strong edge)
         < low threshold    → noise, discarded
    5. Edge tracking by hysteresis — connects weak edges to strong ones

    Result: clean, thin, well-connected edges — superior to Sobel alone.
    low=50, high=150 are the hysteresis thresholds (ratio 1:3 is a common choice).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, low, high)  # Apply full Canny pipeline
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Return as 3-channel BGR


# ============================================================
#  TASK 2 – CV ACCURACY CHALLENGE HELPERS
#  Goal: prove "Garbage In = Garbage Out" — image quality directly impacts CV accuracy.
#  Pipeline: Clean image → Degrade → Enhance → Compare face detection accuracy across all 3.
# ============================================================

def create_synthetic_face(size=(400, 400)):
    """Draw a simple synthetic face using basic OpenCV drawing primitives.

    Used as a fallback test image when no real photo is loaded.
    Draws: face oval, eyes, pupils, eyebrows, nose, mouth — all via ellipses/circles.
    Returns a 400×400 BGR numpy array (light gray background).
    """
    img = np.ones((*size, 3), dtype=np.uint8) * 200  # Light gray background (200, 200, 200)
    cx, cy = size[1] // 2, size[0] // 2  # Center of image = center of face
    # Face outline: large vertical ellipse, skin tone color (BGR)
    cv2.ellipse(img, (cx, cy), (100, 130), 0, 0, 360, (220, 185, 150), -1)
    # Eyes: dark brown filled ellipses (left and right)
    cv2.ellipse(img, (cx - 40, cy - 30), (18, 12), 0, 0, 360, (50, 40, 30), -1)
    cv2.ellipse(img, (cx + 40, cy - 30), (18, 12), 0, 0, 360, (50, 40, 30), -1)
    # Pupils: small dark circles inside each eye
    cv2.circle(img, (cx - 40, cy - 30), 6, (10, 10, 10), -1)
    cv2.circle(img, (cx + 40, cy - 30), 6, (10, 10, 10), -1)
    # Nose: small ellipse below center of face
    cv2.ellipse(img, (cx, cy + 10), (12, 16), 0, 0, 360, (190, 155, 120), -1)
    # Mouth: lower arc (0°→180°) drawn as an outline (thickness=2)
    cv2.ellipse(img, (cx, cy + 55), (35, 15), 0, 0, 180, (160, 80, 80), 2)
    # Eyebrows: thin arcs above each eye
    cv2.ellipse(img, (cx - 40, cy - 55), (22, 6), -10, 0, 180, (60, 40, 20), 3)
    cv2.ellipse(img, (cx + 40, cy - 55), (22, 6), 10, 0, 180, (60, 40, 20), 3)
    return img


def degrade_image(image, darkness=0.35, noise_level=40):
    """Simulate a low-quality / poorly captured image using two degradation steps.

    Step 1 — Darken:
        Multiply every pixel by 'darkness' factor (e.g., 0.35 → only 35% brightness remains).
        Simulates underexposure, low-light camera, or night-time capture.

    Step 2 — Add Gaussian noise:
        Add random values sampled from N(0, noise_level) to each pixel.
        Simulates sensor noise, JPEG compression artifacts, or transmission errors.

    .clip(0, 255) after each step ensures values stay in valid uint8 range.
    """
    darkened = (image.astype(np.float32) * darkness).clip(0, 255).astype(np.uint8)  # Darken
    noise = np.random.normal(0, noise_level, darkened.shape).astype(np.float32)  # Generate Gaussian noise
    noisy = (darkened.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)  # Add noise + clip
    return noisy


def enhance_image_task2(image):
    """4-step enhancement pipeline to recover a darkened + noisy image for face detection.

    ── Step 1: Bilateral Filter ───────────────────────────────────────────────
        Removes noise WHILE preserving edges (unlike Gaussian blur which blurs edges too).
        d=9: pixel neighborhood diameter.
        sigmaColor=75: pixels with color difference < 75 are averaged together.
        sigmaSpace=75: pixels up to 75px apart spatially can influence each other.
        Critical for keeping face contours sharp so Haar Cascade can find them.

    ── Step 2: Gamma Correction (gamma=2.2) ───────────────────────────────────
        Recovers brightness lost by darkness=0.40 degradation.
        Math rationale: 0.40 darkening → need ~1/0.40 = 2.5x boost → gamma ≈ 2.2.
        Lifts shadows non-linearly: more aggressive in dark areas, gentler in highlights.

    ── Step 3: CLAHE (Contrast Limited Adaptive Histogram Equalization) ────────
        Applied only to Y (luminance) channel in YCrCb space to avoid color shift.
        Unlike global histogram equalization, CLAHE works on small local tiles (8×8),
        which restores LOCAL contrast — makes eyes, nose, and mouth distinguishable.
        clipLimit=3.5: controls max contrast amplification (higher = more contrast + more noise).

    ── Step 4: Unsharp Mask ────────────────────────────────────────────────────
        Sharpens edges so the Haar Cascade can detect face boundaries clearly.
        Formula: sharpened = 1.5 * original - 0.5 * blurred
        Subtracting a blurred copy amplifies high-frequency detail (edges and textures).
    """
    # Step 1: Bilateral filter — denoise without blurring face edges
    denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 2: Strong gamma correction — recover brightness from heavy darkening
    gamma = 2.2
    inv_gamma = 1.0 / gamma
    lut = np.array([(i / 255.0) ** inv_gamma * 255
                    for i in range(256)], dtype=np.uint8)
    brightened = cv2.LUT(denoised, lut)  # Apply gamma via lookup table

    # Step 3: CLAHE — restore local contrast in luminance channel only
    ycrcb = cv2.cvtColor(brightened, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))  # 8×8 tile grid for local enhancement
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])  # Apply CLAHE only to Y channel
    enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # Step 4: Unsharp mask — sharpen edges to improve Haar Cascade detection
    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2)  # Slightly blurred version
    sharpened = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)  # Amplify difference = sharper

    return sharpened


def detect_faces_task2(image, cascade, scale=1.1, min_neighbors=4, min_size=(30, 30)):
    """Run Haar Cascade face detection and measure inference time.

    Haar Cascade: a trained sliding-window classifier that scans the image at
    multiple scales looking for learned patterns (Haar features like eye regions,
    nose bridge, etc.).

    Parameters:
    - scaleFactor=1.1   : shrink image by 10% each pass → detects faces of different sizes
    - minNeighbors=4    : a candidate region is accepted only if confirmed by ≥4 overlapping windows
    - minSize=(30,30)   : ignore detection windows smaller than 30×30 pixels

    Returns:
    - faces   : numpy array of (x, y, w, h) bounding boxes | empty tuple () if none found
    - elapsed : total inference time in milliseconds (measured with high-precision perf_counter)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Haar Cascade requires grayscale input
    t0 = time.perf_counter()  # Start high-precision timer
    faces = cascade.detectMultiScale(
        gray, scaleFactor=scale, minNeighbors=min_neighbors, minSize=min_size
    )
    elapsed = (time.perf_counter() - t0) * 1000  # Convert seconds → milliseconds
    return faces, elapsed


def draw_faces(image, faces, color=(0, 255, 100), thickness=2):
    """Draw bounding boxes and 'Face' labels over each detected face region.

    Draws a rectangle from top-left (x, y) to bottom-right (x+w, y+h).
    Label 'Face' is placed 8 pixels above the top-left corner of the box.
    Operates on a copy so the original image is NOT modified.
    """
    img_copy = image.copy()  # Work on a copy — never modify the original
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, thickness)  # Bounding box
            cv2.putText(img_copy, 'Face', (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)  # Label above box
    return img_copy


# ============================================================
#  MAIN APPLICATION CLASS
#  VisionEditorApp: owns all GUI state and wires every button to its processing function.
#  Pattern: single-class — state (images) + view (Tkinter widgets) + control (methods).
# ============================================================

class VisionEditorApp:
    def __init__(self, root):
        # Store reference to the main Tkinter window
        self.root = root
        self.root.title('Vision Editor – Advanced Image Enhancement & Analysis')
        self.root.configure(bg='#1e1e2e')  # Dark purple background for entire window
        self.root.state('zoomed')  # Start maximized (Windows: 'zoomed' = fullscreen)

        # ── Application State ────────────────────────────────────────────────
        # original_image: loaded once and NEVER modified — always safe to reset back to it
        # processed_image: result of the last applied operation, shown in the right panel
        self.original_image = None
        self.processed_image = None

        # ── Widget Styling ───────────────────────────────────────────────────
        style = ttk.Style()
        style.theme_use('clam')  # 'clam' theme supports custom background colors
        style.configure('TButton', font=('Segoe UI', 10, 'bold'), padding=6,
                        background='#7c3aed', foreground='white')
        style.map('TButton', background=[('active', '#6d28d9')])  # Darker purple on hover/click
        style.configure('TScale', background='#1e1e2e')  # Slider track matches window bg

        self._build_ui()  # Delegate all widget creation to _build_ui()

    # ── UI Construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Header bar ───────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg='#13131f', pady=10)
        header.pack(fill=tk.X)  # Stretch full width across the top
        tk.Label(header, text='🔬 VISION EDITOR',
                 font=('Segoe UI', 20, 'bold'), fg='#a78bfa', bg='#13131f').pack()
        tk.Label(header, text='Advanced Image Enhancement & Computer Vision Analysis',
                 font=('Segoe UI', 10), fg='#6b7280', bg='#13131f').pack()

        # ── Main container: left panel + center area side by side ────────────
        main = tk.Frame(self.root, bg='#1e1e2e')
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # ── Left Panel: Scrollable Controls Sidebar ──────────────────────────
        # Fixed 270px wide, fills full height, scrollable vertically
        ctrl_outer = tk.Frame(main, bg='#13131f', width=270, relief=tk.RAISED, bd=1)
        ctrl_outer.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        ctrl_outer.pack_propagate(False)  # Prevent child widgets from resizing this frame

        # Canvas + vertical scrollbar inside ctrl_outer to enable scrolling
        ctrl_canvas = tk.Canvas(ctrl_outer, bg='#13131f', highlightthickness=0)
        ctrl_scrollbar = tk.Scrollbar(ctrl_outer, orient='vertical', command=ctrl_canvas.yview)
        ctrl_canvas.configure(yscrollcommand=ctrl_scrollbar.set)
        ctrl_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        ctrl_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Actual frame that holds all control widgets — placed as a window inside the canvas
        ctrl = tk.Frame(ctrl_canvas, bg='#13131f')
        ctrl_canvas.create_window((0, 0), window=ctrl, anchor='nw')
        # Auto-update scroll region whenever ctrl's size changes (new widgets added)
        ctrl.bind('<Configure>', lambda e: ctrl_canvas.configure(
            scrollregion=ctrl_canvas.bbox('all')))
        # Enable mouse wheel scrolling on the sidebar
        ctrl_canvas.bind_all('<MouseWheel>',
                             lambda e: ctrl_canvas.yview_scroll(int(-1 * (e.delta / 120)), 'units'))

        # ── Helpers for control widgets ──────────────────────────────────────
        # Defined as local functions so they can access 'ctrl' without passing it

        def section_label(text):
            # Purple section header to visually group related buttons/sliders
            tk.Label(ctrl, text=text, font=('Segoe UI', 9, 'bold'),
                     bg='#2d2d4e', fg='#a78bfa', anchor='w', padx=8
                     ).pack(fill=tk.X, pady=(10, 2))

        def tool_btn(text, cmd):
            # Flat-style button linked to a callback; highlights purple on hover
            tk.Button(ctrl, text=text, command=cmd,
                      font=('Segoe UI', 9), bg='#2d2d4e', fg='#e2e8f0',
                      activebackground='#7c3aed', activeforeground='white',
                      relief=tk.FLAT, anchor=tk.W, padx=8, pady=3, cursor='hand2'
                      ).pack(fill=tk.X, padx=8, pady=1)

        def slider_row(label, from_, to, default, resolution=1):
            # Label + horizontal slider; returns DoubleVar so callbacks can read the value
            tk.Label(ctrl, text=label, font=('Segoe UI', 8),
                     fg='#9ca3af', bg='#13131f'
                     ).pack(anchor=tk.W, padx=12)
            var = tk.DoubleVar(value=default)  # Holds the slider's current numeric value
            ttk.Scale(ctrl, from_=from_, to=to, variable=var,
                      orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12, pady=(0, 6))
            return var  # Stored as self.xxx_var so button callbacks can call .get()

        # ── File section ─────────────────────────────────────────────────────
        section_label('📂  FILE')
        tool_btn('📂 Load Image', self.load_image)  # Open file dialog → load image
        tool_btn('💾 Save Output', self.save_image)  # Save processed_image to disk
        tool_btn('↩ Reset to Original', self.reset_image)  # Revert to original_image

        # ── Point Operations section ─────────────────────────────────────────
        section_label('☀  BRIGHTNESS  &  CONTRAST')
        self.brightness_var = slider_row('Brightness  (-100 → +100)', -100, 100, 0)
        self.contrast_var = slider_row('Contrast  (0.1 → 3.0)', 0.1, 3.0, 1.0)
        tool_btn('Apply Brightness / Contrast', self.apply_brightness_contrast)

        # ── Zoom section (slider-based — no text entry parsing) ──────────────
        section_label('🔍  ZOOM')
        self.zoom_var = slider_row('Zoom Scale  (0.5 → 4.0)', 0.5, 4.0, 1.0)
        tool_btn('🔍 Zoom – Nearest Neighbor', self.apply_zoom_nearest)  # Blocky but fast
        tool_btn('🔍 Zoom – Bilinear', self.apply_zoom_bilinear)  # Smooth interpolation

        # ── Rotation section ─────────────────────────────────────────────────
        section_label('↻  ROTATION')
        self.angle_var = slider_row('Angle  (0 → 360°)', 0, 360, 45)
        tool_btn('↻ Rotate Image', self.apply_rotate)

        # ── Enhancement section ──────────────────────────────────────────────
        section_label('✨  ENHANCEMENT')
        self.gamma_var = slider_row('Gamma  (0.1 → 4.0)', 0.1, 4.0, 1.0)
        tool_btn('📊 Histogram Equalization', self.apply_hist_eq)  # Global contrast redistribution
        tool_btn('γ  Gamma Correction', self.apply_gamma)  # Non-linear brightness curve

        # ── Filters & Edges section ──────────────────────────────────────────
        section_label('🌊  FILTERS  &  EDGES')
        self.blur_var = slider_row('Blur Kernel Size  (1 → 51)', 1, 51, 15)
        tool_btn('🌀 Gaussian Blur', self.apply_gaussian)  # Weighted-average smoothing
        tool_btn('〰 Sobel Edge', self.apply_sobel)  # Gradient-based edge detection
        tool_btn('◈ Canny Edge', self.apply_canny)  # Multi-stage accurate edge detection

        # ── Task 2 section ───────────────────────────────────────────────────
        section_label('🤖  TASK 2 — CV ACCURACY CHALLENGE')
        tk.Label(ctrl, text='Proves "Garbage In = Garbage Out"\nvia Haar Cascade face detection.',
                 font=('Segoe UI', 8), fg='#9ca3af', bg='#13131f', justify=tk.LEFT
                 ).pack(anchor=tk.W, padx=12, pady=(2, 4))
        tool_btn('▶ Run Face Detection Test', self.run_task2)

        # ── Center area: image panels + histogram ────────────────────────────
        center = tk.Frame(main, bg='#1e1e2e')
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Top half: Original image (left) and Processed image (right), side by side
        top_panels = tk.Frame(center, bg='#1e1e2e')
        top_panels.pack(fill=tk.BOTH, expand=True)

        # Left image panel: displays original_image
        orig_frame = tk.LabelFrame(top_panels, text=' Original Image ',
                                   font=('Segoe UI', 10, 'bold'), fg='#60a5fa', bg='#1e1e2e',
                                   labelanchor='n', bd=2, relief=tk.GROOVE)
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.orig_canvas = tk.Label(orig_frame, bg='#0f0f1a',
                                    text='Load an image to begin…', font=('Segoe UI', 12), fg='#4b5563')
        self.orig_canvas.pack(fill=tk.BOTH, expand=True)

        # Right image panel: displays processed_image after any operation
        out_frame = tk.LabelFrame(top_panels, text=' Processed Image ',
                                  font=('Segoe UI', 10, 'bold'), fg='#34d399', bg='#1e1e2e',
                                  labelanchor='n', bd=2, relief=tk.GROOVE)
        out_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.out_canvas = tk.Label(out_frame, bg='#0f0f1a',
                                   text='Apply a tool to see results…', font=('Segoe UI', 12), fg='#4b5563')
        self.out_canvas.pack(fill=tk.BOTH, expand=True)

        # Histogram panel: shows RGB channel histograms for both images
        hist_frame = tk.Frame(center, bg='#13131f', height=200)
        hist_frame.pack(fill=tk.X, padx=5, pady=5)
        hist_frame.pack_propagate(False)  # Keep the fixed 200px height

        # 1 row, 2 columns: left = input histogram, right = output histogram
        self.fig, (self.ax_in, self.ax_out) = plt.subplots(
            1, 2, figsize=(10, 2), facecolor='#13131f')
        for ax in (self.ax_in, self.ax_out):
            ax.set_facecolor('#0f0f1a')
            ax.tick_params(colors='#6b7280', labelsize=7)
            for sp in ax.spines.values():
                sp.set_color('#374151')
        self.ax_in.set_title('Input Histogram', color='#60a5fa', fontsize=9)
        self.ax_out.set_title('Output Histogram', color='#34d399', fontsize=9)
        self.fig.tight_layout(pad=0.5)

        # FigureCanvasTkAgg: bridge that embeds the matplotlib figure inside Tkinter
        self.hist_canvas = FigureCanvasTkAgg(self.fig, master=hist_frame)
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Status bar at the very bottom of the window
        # Displays: last operation name | output dimensions | processing time
        self.status = tk.StringVar(value='Ready. Load an image to begin.')
        tk.Label(self.root, textvariable=self.status,
                 font=('Segoe UI', 9), fg='#9ca3af', bg='#0a0a14',
                 anchor=tk.W, padx=10).pack(fill=tk.X, side=tk.BOTTOM)

    # ── Internal Helpers ─────────────────────────────────────────────────────

    def _cv_to_tk(self, img_bgr, max_w=500, max_h=380):
        """Convert OpenCV BGR numpy array → Tkinter PhotoImage for display.

        Steps:
        1. Compute scale factor to fit within max_w × max_h while keeping aspect ratio.
           min(..., 1.0) ensures we only shrink, never enlarge beyond original size.
        2. Convert color order: BGR (OpenCV) → RGB (PIL/Tkinter).
        3. numpy array → PIL Image → resize with LANCZOS (high-quality) → PhotoImage.
        """
        h, w = img_bgr.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)  # Fit-to-panel, never upscale
        rw, rh = int(w * scale), int(h * scale)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # BGR → RGB
        pil = Image.fromarray(img_rgb).resize((rw, rh), Image.LANCZOS)  # High-quality resize
        return ImageTk.PhotoImage(pil)  # Tkinter-compatible image

    def _draw_hist(self, ax, img, title):
        """Draw pixel intensity histogram on a matplotlib axis.

        Color image: plot separate histograms for B (blue), G (green), R (red) channels.
        Grayscale: single histogram plotted in purple.
        cv2.calcHist returns a 256-element array — count of pixels at each intensity [0..255].
        """
        ax.clear()
        ax.set_facecolor('#0f0f1a')
        colors = ('#3b82f6', '#10b981', '#ef4444')  # Blue=B channel, Green=G channel, Red=R channel
        if len(img.shape) == 3:
            # Color image: split into channels and plot each separately
            for i, (ch, col) in enumerate(zip(cv2.split(img), colors)):
                hist = cv2.calcHist([ch], [0], None, [256], [0, 256])  # 256 bins, full intensity range
                ax.plot(hist, color=col, linewidth=0.8, alpha=0.8)
        else:
            # Grayscale image: single histogram
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            ax.plot(hist, color='#a78bfa', linewidth=0.8)
        ax.set_title(title, color='#9ca3af', fontsize=8)
        ax.tick_params(colors='#4b5563', labelsize=6)
        for sp in ax.spines.values():
            sp.set_color('#374151')

    def _show_images(self):
        """Refresh both image display panels and both histograms.

        Called after every operation and after loading a new image.
        Stores the PhotoImage reference on the Label widget (.image attribute) —
        CRITICAL: without this, Python's garbage collector deletes the image and the panel goes blank.
        """
        if self.original_image is None:
            return
        # Refresh left panel: original image
        tk_orig = self._cv_to_tk(self.original_image)
        self.orig_canvas.config(image=tk_orig, text='')
        self.orig_canvas.image = tk_orig  # Keep reference to prevent garbage collection

        # Refresh right panel: processed image (if one exists)
        if self.processed_image is not None:
            tk_proc = self._cv_to_tk(self.processed_image)
            self.out_canvas.config(image=tk_proc, text='')
            self.out_canvas.image = tk_proc  # Keep reference
            self._draw_hist(self.ax_out, self.processed_image, 'Output Histogram')

        # Always redraw the input histogram from the original image
        self._draw_hist(self.ax_in, self.original_image, 'Input Histogram')
        self.hist_canvas.draw()  # Trigger matplotlib canvas re-render

    def _apply(self, fn, desc):
        """Generic operation runner used by all tool callbacks.

        1. Guard: warn and exit if no image is loaded.
        2. Apply fn() to a COPY of the original image (non-destructive — reset always works).
        3. Measure and report processing time.
        4. Refresh UI image panels and histograms.
        5. Update status bar with operation name, output size, and elapsed time.
        """
        if self.original_image is None:
            messagebox.showwarning('No Image', 'Please load an image first!')
            return
        t0 = time.time()
        self.processed_image = fn(self.original_image.copy())  # Operate on copy, never on original
        elapsed = (time.time() - t0) * 1000  # Processing time in ms
        self._show_images()
        h, w = self.processed_image.shape[:2]
        self.status.set(f'✔ Applied: {desc}  |  Size: {w}×{h}  |  Time: {elapsed:.1f} ms')

    # ── File Actions ─────────────────────────────────────────────────────────

    def load_image(self):
        """Open file dialog → read selected image with OpenCV → store as original.

        Supports: jpg, jpeg, png, bmp, tiff, webp.
        On success: both original_image and processed_image point to the loaded image.
        processed_image starts as a copy of original — no operation applied yet.
        """
        path = filedialog.askopenfilename(
            title='Select Image',
            filetypes=[('Images', '*.jpg *.jpeg *.png *.bmp *.tiff *.webp'), ('All', '*.*')])
        if not path:
            return  # User cancelled the dialog — do nothing
        img = cv2.imread(path)  # Read image as BGR numpy array
        if img is None:
            messagebox.showerror('Error', 'Cannot read image file.')
            return
        self.original_image = img
        self.processed_image = img.copy()  # Initial state: processed = original (no change yet)
        self._show_images()
        h, w = img.shape[:2]
        self.status.set(f'Loaded: {path}  |  Size: {w}×{h}')

    def save_image(self):
        """Open save dialog → write processed_image to disk.

        cv2.imwrite selects format automatically from the file extension (.png, .jpg, .bmp).
        """
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
        """Revert processed_image back to original_image and refresh the display.

        Safe because all operations use .copy() — original_image is always intact.
        """
        if self.original_image is None:
            return
        self.processed_image = self.original_image.copy()
        self._show_images()
        self.status.set('Image reset to original.')

    # ── Tool Callbacks ───────────────────────────────────────────────────────
    # Each method reads the relevant slider value(s) then delegates to _apply()
    # with a lambda wrapping the processing function + those parameter values.

    def apply_brightness_contrast(self):
        b = int(self.brightness_var.get())  # beta  = brightness offset
        c = float(self.contrast_var.get())  # alpha = contrast multiplier
        self._apply(lambda img: adjust_brightness_contrast(img, b, c),
                    f'Brightness={b}, Contrast={c:.2f}')

    def apply_zoom_nearest(self):
        s = float(self.zoom_var.get())  # Scale factor from slider
        self._apply(lambda img: zoom_nearest(img, s), f'Zoom Nearest ×{s:.1f}')

    def apply_zoom_bilinear(self):
        s = float(self.zoom_var.get())  # Scale factor from slider
        self._apply(lambda img: zoom_bilinear(img, s), f'Zoom Bilinear ×{s:.1f}')

    def apply_rotate(self):
        a = float(self.angle_var.get())  # Rotation angle in degrees
        self._apply(lambda img: rotate_image(img, a), f'Rotate {a:.0f}°')

    def apply_hist_eq(self):
        # No parameters needed — histogram equalization is fully automatic
        self._apply(histogram_equalization, 'Histogram Equalization')

    def apply_gamma(self):
        g = float(self.gamma_var.get())  # Gamma value: >1 = brighter, <1 = darker
        self._apply(lambda img: gamma_correction(img, g), f'Gamma={g:.2f}')

    def apply_gaussian(self):
        k = int(self.blur_var.get())  # Kernel size (forced odd inside the function)
        self._apply(lambda img: gaussian_blur(img, k), f'Gaussian Blur k={k}')

    def apply_sobel(self):
        # No slider — uses fixed ksize=3 (standard for Sobel)
        self._apply(sobel_edge, 'Sobel Edge Detection')

    def apply_canny(self):
        # No slider — uses fixed thresholds: low=50, high=150
        self._apply(canny_edge, 'Canny Edge Detection')

    # ── Task 2: CV Accuracy Challenge ────────────────────────────────────────

    def run_task2(self):
        """Full Task 2 experiment demonstrating 'Garbage In = Garbage Out'.

        Pipeline:
        1. Load Haar Cascade classifier from OpenCV's built-in data path.
        2. Get test image (user's loaded photo, or synthetic face as fallback).
        3. Create degraded version (dark + noisy) and enhanced version.
        4. Run face detection on all 3 versions; measure accuracy vs. clean image.
        5. Plot 2×3 grid: top row = images with boxes | bottom row = histograms.
        6. Print full comparison table to console.
        7. Show summary popup with key numbers.
        """
        # Load the pre-trained Haar Cascade XML from OpenCV's built-in data folder
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            messagebox.showerror('Error', 'Haar Cascade failed to load!')
            return

        # Use the loaded image as ground truth, or generate a synthetic face as fallback
        if self.original_image is not None:
            original_clean = self.original_image.copy()
        else:
            original_clean = create_synthetic_face()
            messagebox.showinfo(
                'No Image Loaded',
                'No image loaded — using synthetic face for demo.\n'
                'Load a real face photo for better results.'
            )

        # Generate all 3 image versions for the experiment
        degraded_image = degrade_image(original_clean, darkness=0.40, noise_level=40)  # Simulate bad quality
        enhanced_image = enhance_image_task2(degraded_image)  # Attempt to recover

        # Run face detection on each version and time each inference
        faces_clean, t_clean = detect_faces_task2(original_clean, face_cascade)
        faces_degraded, t_degraded = detect_faces_task2(degraded_image, face_cascade)
        faces_enhanced, t_enhanced = detect_faces_task2(enhanced_image, face_cascade)

        # Compute accuracy: clean image detections = ground truth (100%)
        gt_faces = max(len(faces_clean), 1)  # Avoid division by zero
        acc_clean = min(len(faces_clean) / gt_faces * 100, 100)  # Should be 100%
        acc_degraded = min(len(faces_degraded) / gt_faces * 100, 100)  # Expected to drop
        acc_enhanced = min(len(faces_enhanced) / gt_faces * 100, 100)  # Should recover

        # Build 2×3 visualization grid
        # Row 0: face detection result images (with bounding boxes drawn)
        # Row 1: grayscale histograms showing intensity distribution shift
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle('CV Accuracy Challenge – Face Detection Results',
                     fontsize=16, fontweight='bold', color='#1e1e2e')
        fig.patch.set_facecolor('#f8fafc')

        images_info = [
            (original_clean, faces_clean, f'Clean Image\nFaces: {len(faces_clean)} | Acc: {acc_clean:.0f}%', '#16a34a'),
            (degraded_image, faces_degraded, f'Degraded Image\nFaces: {len(faces_degraded)} | Acc: {acc_degraded:.0f}%',
             '#dc2626'),
            (enhanced_image, faces_enhanced, f'Enhanced Image\nFaces: {len(faces_enhanced)} | Acc: {acc_enhanced:.0f}%',
             '#7c3aed'),
        ]

        for col, (img, faces, title, color) in enumerate(images_info):
            # Row 0: draw bounding boxes on image and display it
            drawn = draw_faces(img, faces)
            axes[0, col].imshow(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))  # Matplotlib needs RGB, not BGR
            axes[0, col].set_title(title, fontsize=11, color=color, fontweight='bold')
            axes[0, col].axis('off')  # Hide axes ticks/labels for clean image display

            # Row 1: grayscale histogram for this version
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            axes[1, col].fill_between(range(256), hist.flatten(), color=color, alpha=0.4)  # Shaded area
            axes[1, col].plot(hist, color=color, linewidth=1.2)
            axes[1, col].set_facecolor('#f1f5f9')
            axes[1, col].set_title('Histogram', fontsize=9, color='#374151')
            axes[1, col].set_xlim([0, 256])
            axes[1, col].tick_params(labelsize=7)

        plt.tight_layout()
        plt.show()

        # Print comparison table to console using pandas DataFrame
        results = pd.DataFrame({
            'Image Version': ['Clean (Ground Truth)', 'Degraded (Low Quality)', 'Enhanced (Post-Processing)'],
            'Faces Detected': [len(faces_clean), len(faces_degraded), len(faces_enhanced)],
            'Accuracy (%)': [f'{acc_clean:.1f}%', f'{acc_degraded:.1f}%', f'{acc_enhanced:.1f}%'],
            'Inference (ms)': [f'{t_clean:.1f}', f'{t_degraded:.1f}', f'{t_enhanced:.1f}'],
            'Improvement': ['Baseline',
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

        # Show summary popup with key results
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
        # Update status bar with quick summary of Task 2 results
        self.status.set(
            f'Task 2 done — Degraded: {acc_degraded:.0f}%  →  Enhanced: {acc_enhanced:.0f}%'
        )


# ============================================================
#  ENTRY POINT
#  if __name__ == '__main__': runs only when this file is executed directly,
#  not when it's imported as a module.
#  mainloop() blocks here and processes GUI events (clicks, redraws, key presses)
#  until the user closes the window.
# ============================================================

if __name__ == '__main__':
    root = tk.Tk()  # Create the main Tkinter window
    app = VisionEditorApp(root)  # Build the full UI and initialize state
    root.mainloop()  # Start the event loop — app lives here until window is closed