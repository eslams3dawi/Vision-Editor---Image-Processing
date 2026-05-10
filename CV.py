# # ============================================================
# #  Vision Editor – Advanced Image Enhancement & Analysis
# #  Tools: OpenCV, Tkinter, Matplotlib
# #
# #  Task 2 face detection is handled by CV.py (imported below).
# #  apply_face_detection()  →  single-image DNN detection
# #  apply_comparison()      →  before-vs-after comparison
# # ============================================================
#
# import cv2
# import tkinter as tk
# from tkinter import filedialog, messagebox
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import time
#
# # ── Link CV.py ───────────────────────────────────────────────
# import CV as cv_tasks   # <-- face detection module
#
# # ────────────────────────────────────────────────────────────
# #  GLOBAL STATE
# # ────────────────────────────────────────────────────────────
# original_image  = None   # Original image (never changed)
# processed_image = None   # Current processed result
#
# # ============================================================
# #  SECTION 1 – FILE OPERATIONS
# # ============================================================
#
# def load_image():
#     """Open file dialog and load the chosen image."""
#     global original_image, processed_image
#     path = filedialog.askopenfilename(
#         title="Select an Image",
#         filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
#     )
#     if not path:
#         return
#     original_image  = cv2.imread(path)
#     processed_image = original_image.copy()
#     update_display()
#
# def reset_image():
#     """Restore processed image back to original."""
#     global processed_image
#     if original_image is None:
#         return
#     processed_image = original_image.copy()
#     update_display()
#
# # ============================================================
# #  SECTION 2 – DISPLAY
# # ============================================================
#
# def update_display():
#     show_images()
#     show_histograms()
#
# def show_images():
#     if original_image is None:
#         return
#     orig_rgb = cv2.cvtColor(original_image,  cv2.COLOR_BGR2RGB)
#     proc_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
#     ax_orig.clear(); ax_proc.clear()
#     ax_orig.imshow(orig_rgb);  ax_orig.set_title("Original",  color="white", fontsize=10)
#     ax_proc.imshow(proc_rgb);  ax_proc.set_title("Processed", color="white", fontsize=10)
#     ax_orig.axis("off");       ax_proc.axis("off")
#     canvas_images.draw()
#
# def show_histograms():
#     if original_image is None:
#         return
#     ax_hist_orig.clear(); ax_hist_proc.clear()
#     for i, color in enumerate(("b", "g", "r")):
#         h_o = cv2.calcHist([original_image],  [i], None, [256], [0, 256])
#         h_p = cv2.calcHist([processed_image], [i], None, [256], [0, 256])
#         ax_hist_orig.plot(h_o, color=color)
#         ax_hist_proc.plot(h_p, color=color)
#     ax_hist_orig.set_title("Original Histogram",  color="white", fontsize=9)
#     ax_hist_proc.set_title("Processed Histogram", color="white", fontsize=9)
#     ax_hist_orig.tick_params(colors="white")
#     ax_hist_proc.tick_params(colors="white")
#     canvas_hist.draw()
#
# # ============================================================
# #  SECTION 3 – POINT OPERATIONS
# # ============================================================
#
# def adjust_brightness():
#     global processed_image
#     if original_image is None: return
#     value = brightness_slider.get()
#     processed_image = cv2.convertScaleAbs(original_image, alpha=1, beta=value)
#     update_display()
#
# def adjust_contrast():
#     global processed_image
#     if original_image is None: return
#     value = contrast_slider.get()
#     processed_image = cv2.convertScaleAbs(original_image, alpha=value, beta=0)
#     update_display()
#
# # ============================================================
# #  SECTION 4 – GEOMETRIC TRANSFORMATIONS
# # ============================================================
#
# def _get_zoom_factor():
#     try:
#         factor = float(zoom_entry.get())
#         if factor <= 0: raise ValueError
#         return factor
#     except ValueError:
#         messagebox.showerror("Invalid Input", "Zoom factor must be a positive number (e.g. 2.0)")
#         return None
#
# def zoom_nearest():
#     global processed_image
#     if original_image is None: return
#     factor = _get_zoom_factor()
#     if factor is None: return
#     h, w = original_image.shape[:2]
#     processed_image = cv2.resize(original_image,
#                                   (max(1, int(w * factor)), max(1, int(h * factor))),
#                                   interpolation=cv2.INTER_NEAREST)
#     update_display()
#
# def zoom_bilinear():
#     global processed_image
#     if original_image is None: return
#     factor = _get_zoom_factor()
#     if factor is None: return
#     h, w = original_image.shape[:2]
#     processed_image = cv2.resize(original_image,
#                                   (max(1, int(w * factor)), max(1, int(h * factor))),
#                                   interpolation=cv2.INTER_LINEAR)
#     update_display()
#
# def rotate_image():
#     global processed_image
#     if original_image is None: return
#     try:
#         angle = float(angle_entry.get())
#     except ValueError:
#         messagebox.showerror("Invalid Input", "Angle must be a number (e.g. 45)")
#         return
#     h, w   = original_image.shape[:2]
#     matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale=1.0)
#     processed_image = cv2.warpAffine(original_image, matrix, (w, h))
#     update_display()
#
# # ============================================================
# #  SECTION 5 – ENHANCEMENT
# # ============================================================
#
# def histogram_equalization():
#     global processed_image
#     if original_image is None: return
#     yuv = cv2.cvtColor(original_image, cv2.COLOR_BGR2YUV)
#     yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
#     processed_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
#     update_display()
#
# def gamma_correction():
#     global processed_image
#     if original_image is None: return
#     try:
#         gamma = float(gamma_entry.get())
#         if gamma <= 0: raise ValueError
#     except ValueError:
#         messagebox.showerror("Invalid Input", "Gamma must be a positive number (e.g. 1.5)")
#         return
#     inv_gamma = 1.0 / gamma
#     table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
#     processed_image = cv2.LUT(original_image, table)
#     update_display()
#
# # ============================================================
# #  SECTION 6 – FILTERING & EDGE DETECTION
# # ============================================================
#
# def gaussian_blur():
#     global processed_image
#     if original_image is None: return
#     processed_image = cv2.GaussianBlur(original_image, (11, 11), sigmaX=0)
#     update_display()
#
# def edge_sobel():
#     global processed_image
#     if original_image is None: return
#     gray    = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
#     sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
#     sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
#     edges   = np.clip(cv2.magnitude(sobel_x, sobel_y), 0, 255).astype(np.uint8)
#     processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
#     update_display()
#
# def edge_canny():
#     global processed_image
#     if original_image is None: return
#     gray  = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, threshold1=100, threshold2=200)
#     processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
#     update_display()
#
# # ============================================================
# #  SECTION 7 – TASK 2: FACE DETECTION  (calls CV.py)
# # ============================================================
#
# def apply_face_detection():
#     """
#     Run DNN face detection on the current image using CV.py.
#     Updates the processed panel and shows a results popup.
#     """
#     global processed_image
#     if original_image is None:
#         messagebox.showwarning("No Image", "Please load an image first.")
#         return
#
#     # ── Call CV.py detect_faces() ────────────────────────────
#     out, faces, metrics = cv_tasks.detect_faces(original_image)
#
#     # Show result in the processed panel
#     processed_image = out
#     update_display()
#
#     # Build popup message
#     q = metrics['image_quality']
#     msg = (
#         f"Face Detection Results\n"
#         f"──────────────────────\n"
#         f"Faces detected:  {metrics['num_faces']}\n"
#         f"Inference time:  {metrics['inference_time_ms']} ms\n\n"
#         f"Image Quality:\n"
#         f"  Brightness:  {q['brightness']}\n"
#         f"  Contrast:    {q['contrast']}\n"
#         f"  Sharpness:   {q['sharpness']}"
#     )
#     messagebox.showinfo("CV Model Results", msg)
#
#
# def apply_comparison():
#     """
#     Compare face detection before vs after enhancement using CV.py.
#     Shows a matplotlib side-by-side window + console report + popup.
#     """
#     global processed_image
#     if original_image is None:
#         messagebox.showwarning("No Image", "Please load an image first.")
#         return
#
#     # ── Call CV.py compare_before_after() ───────────────────
#     comparison_img, stats = cv_tasks.compare_before_after(original_image)
#
#     # Show the enhanced (processed) side in the main panel
#     processed_image = cv_tasks.enhance_image(original_image)
#     update_display()
#
#     # Print full report to console
#     cv_tasks.print_comparison_report(stats)
#
#     # Show matplotlib comparison window
#     orig_result, _, _ = cv_tasks.detect_faces(original_image)
#     proc_result, _, _ = cv_tasks.detect_faces(processed_image)
#
#     fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#1e1e2e")
#
#     axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
#     axes[0].set_title("Original Image\n(Reference)", color="#a6e3a1", fontsize=11)
#     axes[0].axis("off"); axes[0].set_facecolor("#313244")
#
#     axes[1].imshow(cv2.cvtColor(orig_result, cv2.COLOR_BGR2RGB))
#     axes[1].set_title(
#         f"BEFORE – Original\n"
#         f"Faces: {stats['original']['faces_detected']}  |  "
#         f"Time: {stats['original']['inference_time_ms']}ms",
#         color="#f38ba8", fontsize=10
#     )
#     axes[1].axis("off"); axes[1].set_facecolor("#313244")
#
#     axes[2].imshow(cv2.cvtColor(proc_result, cv2.COLOR_BGR2RGB))
#     axes[2].set_title(
#         f"AFTER – Enhanced\n"
#         f"Faces: {stats['processed']['faces_detected']}  |  "
#         f"Time: {stats['processed']['inference_time_ms']}ms",
#         color="#a6e3a1", fontsize=10
#     )
#     axes[2].axis("off"); axes[2].set_facecolor("#313244")
#
#     fig.suptitle(
#         "TASK 2 — CV Accuracy Challenge: Garbage In = Garbage Out",
#         color="#f9e2af", fontsize=13, fontweight="bold"
#     )
#     plt.tight_layout()
#     plt.show()
#
#     # Popup summary table
#     s = stats
#     msg = (
#         f"{'':─<44}\n"
#         f"{'ORIGINAL':^22}{'PROCESSED':^22}\n"
#         f"{'':─<44}\n"
#         f"Faces:       {s['original']['faces_detected']:<18}{s['processed']['faces_detected']}\n"
#         f"Time (ms):   {s['original']['inference_time_ms']:<18}{s['processed']['inference_time_ms']}\n"
#         f"Brightness:  {s['original']['brightness']:<18}{s['processed']['brightness']}\n"
#         f"Contrast:    {s['original']['contrast']:<18}{s['processed']['contrast']}\n"
#         f"Sharpness:   {s['original']['sharpness']:<18}{s['processed']['sharpness']}\n"
#         f"{'':─<44}\n"
#         f"Detection change:   {s['improvement']['detection_change']:+d} faces\n"
#         f"Improvement:        {s['improvement']['detection_improvement_pct']}%\n"
#         f"Time overhead:      {s['improvement']['time_overhead_ms']} ms"
#     )
#     messagebox.showinfo("Before vs After Comparison", msg)
#
# # ============================================================
# #  SECTION 8 – BUILD THE GUI
# # ============================================================
#
# root = tk.Tk()
# root.title("Vision Editor – Advanced Image Enhancement & Analysis")
# root.geometry("1380x860")
# root.configure(bg="#1e1e2e")
#
# tk.Label(root, text="🖼  Vision Editor",
#          font=("Arial", 18, "bold"), bg="#1e1e2e", fg="#cdd6f4"
#          ).pack(pady=8)
#
# main_frame = tk.Frame(root, bg="#1e1e2e")
# main_frame.pack(fill="both", expand=True, padx=10, pady=5)
#
# # ─────────────────────────────────────────────────────────────
# #  LEFT PANEL – Scrollable Controls
# # ─────────────────────────────────────────────────────────────
# ctrl_outer = tk.Frame(main_frame, bg="#313244", width=255, relief="sunken", bd=2)
# ctrl_outer.pack(side="left", fill="y", padx=(0, 8))
# ctrl_outer.pack_propagate(False)
#
# ctrl_canvas    = tk.Canvas(ctrl_outer, bg="#313244", highlightthickness=0, width=240)
# ctrl_scrollbar = tk.Scrollbar(ctrl_outer, orient="vertical", command=ctrl_canvas.yview)
# ctrl_canvas.configure(yscrollcommand=ctrl_scrollbar.set)
# ctrl_scrollbar.pack(side="right", fill="y")
# ctrl_canvas.pack(side="left", fill="both", expand=True)
#
# ctrl        = tk.Frame(ctrl_canvas, bg="#313244")
# ctrl_window = ctrl_canvas.create_window((0, 0), window=ctrl, anchor="nw")
#
# def _on_ctrl_configure(event):
#     ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox("all"))
# ctrl.bind("<Configure>", _on_ctrl_configure)
#
# def _on_mousewheel(event):
#     ctrl_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
# ctrl_canvas.bind_all("<MouseWheel>", _on_mousewheel)
#
# # ── Helper widgets ───────────────────────────────────────────
#
# def section_label(text):
#     tk.Label(ctrl, text=text, font=("Arial", 9, "bold"),
#              bg="#45475a", fg="#cba6f7", anchor="w", padx=6
#              ).pack(fill="x", pady=(10, 2))
#
# def btn(text, cmd, color="#89b4fa"):
#     tk.Button(ctrl, text=text, command=cmd,
#               bg=color, fg="#1e1e2e", font=("Arial", 8, "bold"),
#               relief="flat", cursor="hand2"
#               ).pack(fill="x", padx=10, pady=2)
#
# def reset_btn():
#     tk.Button(ctrl, text="↺ Reset", command=reset_image,
#               bg="#f38ba8", fg="#1e1e2e", font=("Arial", 8),
#               relief="flat", cursor="hand2"
#               ).pack(fill="x", padx=10, pady=(0, 4))
#
# def lbl(text):
#     tk.Label(ctrl, text=text, bg="#313244", fg="#a6adc8",
#              font=("Arial", 8), anchor="w"
#              ).pack(fill="x", padx=10)
#
# def entry_field(default):
#     e = tk.Entry(ctrl, bg="#45475a", fg="#cdd6f4",
#                  font=("Arial", 9), insertbackground="white", relief="flat")
#     e.insert(0, default)
#     e.pack(fill="x", padx=10, pady=2)
#     return e
#
# # ── FILE ─────────────────────────────────────────────────────
# section_label("📂  FILE")
# btn("Load Image",       load_image,   "#89b4fa")
# btn("Reset to Original", reset_image, "#f38ba8")
#
# # ── BRIGHTNESS ───────────────────────────────────────────────
# section_label("🔆  BRIGHTNESS  (-100 → +100)")
# brightness_slider = tk.Scale(ctrl, from_=-100, to=100, orient="horizontal",
#                               bg="#313244", fg="#cdd6f4", troughcolor="#45475a",
#                               highlightbackground="#313244")
# brightness_slider.pack(fill="x", padx=10)
# btn("Apply Brightness", adjust_brightness, "#a6e3a1")
# reset_btn()
#
# # ── CONTRAST ─────────────────────────────────────────────────
# section_label("🌗  CONTRAST  (0.5 → 3.0)")
# contrast_slider = tk.Scale(ctrl, from_=0.5, to=3.0, resolution=0.1, orient="horizontal",
#                             bg="#313244", fg="#cdd6f4", troughcolor="#45475a",
#                             highlightbackground="#313244")
# contrast_slider.set(1.0)
# contrast_slider.pack(fill="x", padx=10)
# btn("Apply Contrast", adjust_contrast, "#a6e3a1")
# reset_btn()
#
# # ── ZOOM ─────────────────────────────────────────────────────
# section_label("🔍  ZOOM  (factor, e.g. 2.0 = double size)")
# lbl("Zoom Factor:")
# zoom_entry = entry_field("2.0")
# btn("Zoom – Nearest Neighbor", zoom_nearest,  "#fab387")
# btn("Zoom – Bilinear",         zoom_bilinear, "#fab387")
# reset_btn()
#
# # ── ROTATION ─────────────────────────────────────────────────
# section_label("🔄  ROTATION  (angle in degrees)")
# lbl("Angle (e.g. 45):")
# angle_entry = entry_field("45")
# btn("Rotate Image", rotate_image, "#fab387")
# reset_btn()
#
# # ── ENHANCEMENT ──────────────────────────────────────────────
# section_label("✨  ENHANCEMENT")
# btn("Histogram Equalization", histogram_equalization, "#cba6f7")
# lbl("Gamma Value (>1 = brighten, <1 = darken):")
# gamma_entry = entry_field("1.5")
# btn("Gamma Correction", gamma_correction, "#cba6f7")
# reset_btn()
#
# # ── FILTERS & EDGES ──────────────────────────────────────────
# section_label("🌊  FILTERS & EDGES")
# btn("Gaussian Blur", gaussian_blur, "#89dceb")
# btn("Edge: Sobel",   edge_sobel,    "#89dceb")
# btn("Edge: Canny",   edge_canny,    "#89dceb")
# reset_btn()
#
# # ── TASK 2 – FACE DETECTION (CV.py) ─────────────────────────
# section_label("🤖  TASK 2 — FACE DETECTION  (CV.py)")
# tk.Label(ctrl,
#          text="DNN SSD face detector\nimported from CV.py",
#          bg="#313244", fg="#a6adc8", font=("Arial", 8), justify="left"
#          ).pack(fill="x", padx=10, pady=2)
#
# btn("▶ Detect Faces (single image)",    apply_face_detection, "#f9e2af")
# btn("▶ Compare Before vs After",        apply_comparison,     "#f9e2af")
#
# # ─────────────────────────────────────────────────────────────
# #  RIGHT PANEL – Images + Histograms
# # ─────────────────────────────────────────────────────────────
# right = tk.Frame(main_frame, bg="#1e1e2e")
# right.pack(side="left", fill="both", expand=True)
#
# fig_img, (ax_orig, ax_proc) = plt.subplots(1, 2, figsize=(9, 4.2), facecolor="#1e1e2e")
# for ax in (ax_orig, ax_proc):
#     ax.set_facecolor("#313244")
#     ax.tick_params(colors="white")
# ax_orig.text(0.5, 0.5, "← Load an image to begin",
#              ha="center", va="center", color="gray",
#              transform=ax_orig.transAxes, fontsize=11)
# ax_orig.set_title("Original Image",  color="white", fontsize=10)
# ax_proc.set_title("Processed Image", color="white", fontsize=10)
# ax_orig.axis("off"); ax_proc.axis("off")
# fig_img.tight_layout(pad=2)
#
# canvas_images = FigureCanvasTkAgg(fig_img, master=right)
# canvas_images.get_tk_widget().pack(fill="both", expand=True)
# canvas_images.draw()
#
# fig_hist, (ax_hist_orig, ax_hist_proc) = plt.subplots(1, 2, figsize=(9, 2.2), facecolor="#1e1e2e")
# for ax in (ax_hist_orig, ax_hist_proc):
#     ax.set_facecolor("#313244")
#     ax.tick_params(colors="white")
# ax_hist_orig.set_title("Original Histogram",  color="white", fontsize=9)
# ax_hist_proc.set_title("Processed Histogram", color="white", fontsize=9)
# fig_hist.tight_layout(pad=2)
#
# canvas_hist = FigureCanvasTkAgg(fig_hist, master=right)
# canvas_hist.get_tk_widget().pack(fill="x")
# canvas_hist.draw()
#
# # ============================================================
# #  LAUNCH
# # ============================================================
# root.mainloop()