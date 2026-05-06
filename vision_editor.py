# ============================================================
#  Vision Editor – Advanced Image Enhancement & Analysis
#  Tools: OpenCV, Tkinter, Matplotlib
# ============================================================

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time     

# ────────────────────────────────────────────────────────────
#  GLOBAL STATE
#  We keep two copies: original (never changed) + processed
# ────────────────────────────────────────────────────────────
original_image  = None   # The image as loaded from disk (BGR)
processed_image = None   # The result after any operation (BGR)


# ============================================================
#  SECTION 1 – FILE OPERATIONS
# ============================================================

def load_image():
    """Open a file dialog and load the chosen image."""
    global original_image, processed_image

    path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    if not path:
        return  # User cancelled

    original_image  = cv2.imread(path)          # Load as BGR
    processed_image = original_image.copy()     # Fresh copy
    update_display()                             # Refresh GUI


def reset_image():
    """Restore processed image back to the original."""
    global processed_image
    if original_image is None:
        return
    processed_image = original_image.copy()
    update_display()


# ============================================================
#  SECTION 2 – DISPLAY (Images + Histograms)
# ============================================================

def update_display():
    """Refresh both the image panel and histogram panel."""
    show_images()
    show_histograms()


def show_images():
    """Draw original and processed images in the GUI canvas."""
    if original_image is None:
        return

    # OpenCV uses BGR; Matplotlib needs RGB — flip channels
    orig_rgb = cv2.cvtColor(original_image,  cv2.COLOR_BGR2RGB)
    proc_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

    ax_orig.clear()
    ax_proc.clear()

    ax_orig.imshow(orig_rgb);      ax_orig.set_title("Original",  color="white", fontsize=10)
    ax_proc.imshow(proc_rgb);      ax_proc.set_title("Processed", color="white", fontsize=10)
    ax_orig.axis("off");           ax_proc.axis("off")

    canvas_images.draw()


def show_histograms():
    """Plot RGB histograms for both original and processed images."""
    if original_image is None:
        return

    ax_hist_orig.clear()
    ax_hist_proc.clear()

    colors = ("b", "g", "r")  # Blue, Green, Red channels

    for i, color in enumerate(colors):
        # cv2.calcHist(images, channels, mask, histSize, ranges)
        h_orig = cv2.calcHist([original_image],  [i], None, [256], [0, 256])
        h_proc = cv2.calcHist([processed_image], [i], None, [256], [0, 256])
        ax_hist_orig.plot(h_orig, color=color)
        ax_hist_proc.plot(h_proc, color=color)

    ax_hist_orig.set_title("Original Histogram",  color="white", fontsize=9)
    ax_hist_proc.set_title("Processed Histogram", color="white", fontsize=9)
    ax_hist_orig.tick_params(colors="white")
    ax_hist_proc.tick_params(colors="white")

    canvas_hist.draw()


# ============================================================
#  SECTION 3 – POINT OPERATIONS
# ============================================================

def adjust_brightness():
    """
    Brightness: add/subtract a constant from every pixel.
    cv2.convertScaleAbs(src, alpha=scale, beta=shift)
    alpha=1 keeps contrast; beta shifts brightness.
    """
    global processed_image
    if original_image is None: return

    value = brightness_slider.get()          # Range: -100 to +100
    processed_image = cv2.convertScaleAbs(original_image, alpha=1, beta=value)
    update_display()


def adjust_contrast():
    """
    Contrast: multiply every pixel by a scale factor.
    alpha > 1  → more contrast
    alpha < 1  → less contrast
    """
    global processed_image
    if original_image is None: return

    value = contrast_slider.get()            # Range: 0.5 to 3.0
    processed_image = cv2.convertScaleAbs(original_image, alpha=value, beta=0)
    update_display()


# ============================================================
#  SECTION 4 – GEOMETRIC TRANSFORMATIONS
# ============================================================

def zoom_nearest():
    """
    Zoom using Nearest Neighbor interpolation.
    Fast but blocky — good for understanding how zoom works.
    """
    global processed_image
    if original_image is None: return

    factor = float(zoom_entry.get())
    h, w   = original_image.shape[:2]
    new_w, new_h = int(w * factor), int(h * factor)

    processed_image = cv2.resize(
        original_image, (new_w, new_h),
        interpolation=cv2.INTER_NEAREST      # Pick closest pixel
    )
    update_display()


def zoom_bilinear():
    """
    Zoom using Bilinear Interpolation.
    Smoother result — averages surrounding pixels.
    """
    global processed_image
    if original_image is None: return

    factor = float(zoom_entry.get())
    h, w   = original_image.shape[:2]
    new_w, new_h = int(w * factor), int(h * factor)

    processed_image = cv2.resize(
        original_image, (new_w, new_h),
        interpolation=cv2.INTER_LINEAR       # Weighted average of 4 neighbors
    )
    update_display()


def rotate_image():
    """
    Rotate image by a user-defined angle (degrees).
    getRotationMatrix2D builds a 2x3 affine matrix around the center.
    """
    global processed_image
    if original_image is None: return

    angle  = float(angle_entry.get())
    h, w   = original_image.shape[:2]
    center = (w // 2, h // 2)              # Rotate around image center

    matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    processed_image = cv2.warpAffine(original_image, matrix, (w, h))
    update_display()


# ============================================================
#  SECTION 5 – ENHANCEMENT
# ============================================================

def histogram_equalization():
    """
    Spread pixel intensities more evenly across [0, 255].
    Works on the Y (luminance) channel in YUV space to avoid
    color distortion.
    """
    global processed_image
    if original_image is None: return

    yuv = cv2.cvtColor(original_image, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])   # Only equalize brightness
    processed_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    update_display()


def gamma_correction():
    """
    Gamma correction: out = (in / 255) ^ (1/gamma) * 255
    gamma > 1 → brighter image (useful for dark images)
    gamma < 1 → darker image
    We use a lookup table (LUT) for speed — compute once, apply to all pixels.
    """
    global processed_image
    if original_image is None: return

    gamma     = float(gamma_entry.get())
    inv_gamma = 1.0 / gamma

    # Build a 256-entry lookup table
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ]).astype("uint8")

    processed_image = cv2.LUT(original_image, table)  # Apply table to all pixels
    update_display()


# ============================================================
#  SECTION 6 – FILTERING & EDGE DETECTION
# ============================================================

def gaussian_blur():
    """
    Smoothing filter. Kernel (11x11) averages nearby pixels
    with Gaussian weights — reduces noise and detail.
    """
    global processed_image
    if original_image is None: return

    processed_image = cv2.GaussianBlur(original_image, (11, 11), sigmaX=0)
    update_display()


def edge_sobel():
    """
    Sobel edge detection: computes gradient in X and Y directions,
    then combines them with cv2.magnitude() to get edge strength.
    """
    global processed_image
    if original_image is None: return

    gray    = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)   # Horizontal edges
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)   # Vertical edges
    edges   = cv2.magnitude(sobel_x, sobel_y)               # Combine
    edges   = np.clip(edges, 0, 255).astype(np.uint8)

    processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    update_display()


def edge_canny():
    """
    Canny edge detection: multi-step algorithm (noise reduction →
    gradient → non-max suppression → hysteresis thresholding).
    threshold1=100, threshold2=200 are the hysteresis bounds.
    """
    global processed_image
    if original_image is None: return

    gray  = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)

    processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    update_display()


# ============================================================
#  SECTION 7 – TASK 2: FACE DETECTION (Haar Cascade)
# ============================================================

def run_face_detection():
    """
    TASK 2 — CV Accuracy Challenge
    Steps:
      1. Run Haar Cascade on the ORIGINAL (possibly low-quality) image
      2. Enhance with Histogram Equalization
      3. Run again on the ENHANCED image
      4. Compare face count and inference time
    """
    if original_image is None:
        messagebox.showwarning("No Image", "Please load an image first.")
        return

    # Load the built-in Haar Cascade model for frontal faces
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade  = cv2.CascadeClassifier(cascade_path)

    # ── Step 1: Baseline detection on ORIGINAL image ──────────────
    gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    t_start    = time.time()
    faces_orig = face_cascade.detectMultiScale(
        gray_original, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    time_orig  = round(time.time() - t_start, 4)

    # ── Step 2: Enhance image using Histogram Equalization ────────
    yuv     = cv2.cvtColor(original_image, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    # ── Step 3: Detection on ENHANCED image ───────────────────────
    gray_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    t_start    = time.time()
    faces_enh  = face_cascade.detectMultiScale(
        gray_enhanced, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    time_enh   = round(time.time() - t_start, 4)

    # ── Step 4: Draw bounding boxes on both images ────────────────
    result_orig = original_image.copy()
    result_enh  = enhanced.copy()

    for (x, y, w, h) in faces_orig:
        cv2.rectangle(result_orig, (x, y), (x+w, y+h), (0, 255, 0), 2)
    for (x, y, w, h) in faces_enh:
        cv2.rectangle(result_enh,  (x, y), (x+w, y+h), (0, 255, 0), 2)

    # ── Step 5: Show visual comparison ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#1e1e2e")

    axes[0].imshow(cv2.cvtColor(result_orig, cv2.COLOR_BGR2RGB))
    axes[0].set_title(
        f"BEFORE Enhancement\nFaces Found: {len(faces_orig)}  |  Time: {time_orig}s",
        color="white", fontsize=11
    )
    axes[0].axis("off")
    axes[0].set_facecolor("#313244")

    axes[1].imshow(cv2.cvtColor(result_enh, cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        f"AFTER Enhancement\nFaces Found: {len(faces_enh)}  |  Time: {time_enh}s",
        color="white", fontsize=11
    )
    axes[1].axis("off")
    axes[1].set_facecolor("#313244")

    fig.suptitle("Face Detection: Before vs After Image Enhancement",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # ── Step 6: Print comparison table to console ─────────────────
    sep = "=" * 52
    print(f"\n{sep}")
    print("       FACE DETECTION ACCURACY COMPARISON")
    print(sep)
    print(f"{'Metric':<28} {'Before':>10} {'After':>10}")
    print("-" * 52)
    print(f"{'Faces Detected':<28} {len(faces_orig):>10} {len(faces_enh):>10}")
    print(f"{'Inference Time (s)':<28} {time_orig:>10} {time_enh:>10}")
    print(f"{'Detection Success':<28} {'Yes' if len(faces_orig)>0 else 'No':>10} "
                                    f"{'Yes' if len(faces_enh)>0 else 'No':>10}")
    print(sep)

    # Also show a popup summary
    messagebox.showinfo(
        "Detection Results",
        f"BEFORE: {len(faces_orig)} face(s) detected  [{time_orig}s]\n"
        f"AFTER:  {len(faces_enh)} face(s) detected  [{time_enh}s]\n\n"
        f"Check the console for the full comparison table."
    )


# ============================================================
#  SECTION 8 – BUILD THE GUI
# ============================================================

# ── Root Window ───────────────────────────────────────────────────
root = tk.Tk()
root.title("Vision Editor – Advanced Image Enhancement & Analysis")
root.geometry("1350x820")
root.configure(bg="#1e1e2e")

# ── App Title ─────────────────────────────────────────────────────
tk.Label(
    root, text="🖼  Vision Editor",
    font=("Arial", 18, "bold"), bg="#1e1e2e", fg="#cdd6f4"
).pack(pady=8)

# ── Main Layout: left controls | right display ────────────────────
main_frame = tk.Frame(root, bg="#1e1e2e")
main_frame.pack(fill="both", expand=True, padx=10, pady=5)

# ─────────────────────────────────────────────────────────────────
#  LEFT PANEL — Buttons and Sliders
# ─────────────────────────────────────────────────────────────────
ctrl = tk.Frame(main_frame, bg="#313244", width=240, relief="sunken", bd=2)
ctrl.pack(side="left", fill="y", padx=(0, 8))
ctrl.pack_propagate(False)   # Keep fixed width

def section_label(text):
    """Helper: styled section separator label."""
    tk.Label(ctrl, text=text, font=("Arial", 9, "bold"),
             bg="#45475a", fg="#cba6f7", anchor="w", padx=6
             ).pack(fill="x", pady=(10, 2))

def btn(text, cmd, color="#89b4fa"):
    """Helper: create a styled flat button."""
    tk.Button(ctrl, text=text, command=cmd,
              bg=color, fg="#1e1e2e", font=("Arial", 8, "bold"),
              relief="flat", cursor="hand2"
              ).pack(fill="x", padx=10, pady=2)

def lbl(text):
    """Helper: small label for an input field."""
    tk.Label(ctrl, text=text, bg="#313244", fg="#a6adc8",
             font=("Arial", 8), anchor="w"
             ).pack(fill="x", padx=10)

def entry_field(default):
    """Helper: create a styled entry box with a default value."""
    e = tk.Entry(ctrl, bg="#45475a", fg="#cdd6f4",
                 font=("Arial", 9), insertbackground="white", relief="flat")
    e.insert(0, default)
    e.pack(fill="x", padx=10, pady=2)
    return e

# ── File ────────────────────────────────────────────────────────
section_label("📂  FILE")
btn("Load Image", load_image, color="#89b4fa")
btn("Reset to Original", reset_image, color="#f38ba8")

# ── Brightness ──────────────────────────────────────────────────
section_label("🔆  BRIGHTNESS  (-100 → +100)")
brightness_slider = tk.Scale(
    ctrl, from_=-100, to=100, orient="horizontal",
    bg="#313244", fg="#cdd6f4", troughcolor="#45475a",
    highlightbackground="#313244"
)
brightness_slider.pack(fill="x", padx=10)
btn("Apply Brightness", adjust_brightness, "#a6e3a1")

# ── Contrast ────────────────────────────────────────────────────
section_label("🌗  CONTRAST  (0.5 → 3.0)")
contrast_slider = tk.Scale(
    ctrl, from_=0.5, to=3.0, resolution=0.1, orient="horizontal",
    bg="#313244", fg="#cdd6f4", troughcolor="#45475a",
    highlightbackground="#313244"
)
contrast_slider.set(1.0)
contrast_slider.pack(fill="x", padx=10)
btn("Apply Contrast", adjust_contrast, "#a6e3a1")

# ── Zoom ────────────────────────────────────────────────────────
section_label("🔍  ZOOM  (factor, e.g. 2.0)")
zoom_entry = entry_field("2.0")
btn("Zoom – Nearest Neighbor", zoom_nearest, "#fab387")
btn("Zoom – Bilinear",         zoom_bilinear, "#fab387")

# ── Rotation ────────────────────────────────────────────────────
section_label("🔄  ROTATION  (angle in degrees)")
angle_entry = entry_field("45")
btn("Rotate Image", rotate_image, "#fab387")

# ── Enhancement ─────────────────────────────────────────────────
section_label("✨  ENHANCEMENT")
btn("Histogram Equalization", histogram_equalization, "#cba6f7")
lbl("Gamma Value (e.g. 1.5 = brighten):")
gamma_entry = entry_field("1.5")
btn("Gamma Correction", gamma_correction, "#cba6f7")

# ── Filters ─────────────────────────────────────────────────────
section_label("🌊  FILTERS & EDGES")
btn("Gaussian Blur",   gaussian_blur, "#89dceb")
btn("Edge: Sobel",     edge_sobel,    "#89dceb")
btn("Edge: Canny",     edge_canny,    "#89dceb")

# ── CV Task 2 ───────────────────────────────────────────────────
section_label("🤖  CV TASK 2 — FACE DETECTION")
btn("▶ Run Face Detection Test", run_face_detection, "#f9e2af")


# ─────────────────────────────────────────────────────────────────
#  RIGHT PANEL — Image Canvas + Histogram Canvas
# ─────────────────────────────────────────────────────────────────
right = tk.Frame(main_frame, bg="#1e1e2e")
right.pack(side="left", fill="both", expand=True)

# ── Image display (top) ─────────────────────────────────────────
fig_img, (ax_orig, ax_proc) = plt.subplots(1, 2, figsize=(9, 4.2),
                                            facecolor="#1e1e2e")
for ax in (ax_orig, ax_proc):
    ax.set_facecolor("#313244")
    ax.tick_params(colors="white")

# Placeholder text before any image is loaded
ax_orig.text(0.5, 0.5, "← Load an image to begin",
             ha="center", va="center", color="gray",
             transform=ax_orig.transAxes, fontsize=11)
ax_orig.set_title("Original Image",  color="white", fontsize=10)
ax_proc.set_title("Processed Image", color="white", fontsize=10)
ax_orig.axis("off"); ax_proc.axis("off")
fig_img.tight_layout(pad=2)

canvas_images = FigureCanvasTkAgg(fig_img, master=right)
canvas_images.get_tk_widget().pack(fill="both", expand=True)
canvas_images.draw()

# ── Histogram display (bottom) ──────────────────────────────────
fig_hist, (ax_hist_orig, ax_hist_proc) = plt.subplots(1, 2, figsize=(9, 2.2),
                                                        facecolor="#1e1e2e")
for ax in (ax_hist_orig, ax_hist_proc):
    ax.set_facecolor("#313244")
    ax.tick_params(colors="white")

ax_hist_orig.set_title("Original Histogram",  color="white", fontsize=9)
ax_hist_proc.set_title("Processed Histogram", color="white", fontsize=9)
fig_hist.tight_layout(pad=2)

canvas_hist = FigureCanvasTkAgg(fig_hist, master=right)
canvas_hist.get_tk_widget().pack(fill="x")
canvas_hist.draw()


# ============================================================
#  LAUNCH THE APP
# ============================================================
root.mainloop()