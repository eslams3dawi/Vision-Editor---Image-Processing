import sys
import cv2
import numpy as np
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Processing functions ──────────────────────────────────────────────────────

def degrade_image(img, darkness=0.30, noise_std=45):
    dark = (img.astype(np.float32) * darkness).clip(0, 255).astype(np.uint8)
    noise = np.random.normal(0, noise_std, dark.shape).astype(np.float32)
    return (dark.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)

def enhance_image(img):
    lut = np.array([(i / 255.0) ** 0.4 * 255 for i in range(256)], dtype=np.uint8)
    b = cv2.LUT(img, lut)
    y = cv2.cvtColor(b, cv2.COLOR_BGR2YCrCb)
    y[:, :, 0] = cv2.equalizeHist(y[:, :, 0])
    return cv2.cvtColor(y, cv2.COLOR_YCrCb2BGR)

def detect_faces(img, cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t0 = time.perf_counter()
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1,
                                     minNeighbors=5, minSize=(50, 50))
    ms = (time.perf_counter() - t0) * 1000
    n = len(faces) if isinstance(faces, np.ndarray) else 0
    return faces, n, ms

def draw_boxes(img, faces, color):
    out = img.copy()
    if isinstance(faces, np.ndarray) and len(faces):
        for (x, y, w, h) in faces:
            cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(out, (x, y-26), (x+w, y), color, -1)
            cv2.putText(out, "Face", (x+4, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
    return out

# ── Generate comparison report ────────────────────────────────────────────────

def make_report(shots, cascade):
    """
    shots = list of dicts:
        { "label": str, "mode": str, "img": np.ndarray }
    Runs detection on each, builds comparison figure + prints table.
    """
    if not shots:
        print("No screenshots to compare.")
        return

    print("")
    print("=" * 62)
    print("  CV ACCURACY CHALLENGE - COMPARISON REPORT")
    print("=" * 62)
    print("  {:<22} {:>8} {:>12} {:>10}".format(
          "Screenshot", "Faces", "Accuracy(%)", "Time(ms)"))
    print("-" * 62)

    # Use first NORMAL shot as ground truth if available
    gt = 1
    for s in shots:
        if s["mode"] == "NORMAL":
            _, n, _ = detect_faces(s["img"], cascade)
            gt = max(n, 1)
            break

    results = []
    for s in shots:
        faces, n, ms = detect_faces(s["img"], cascade)
        acc = min(n / gt * 100, 100)
        results.append({**s, "faces": faces, "n": n, "acc": acc, "ms": ms})
        print("  {:<22} {:>8} {:>11.1f}% {:>9.1f}".format(
              s["label"], n, acc, ms))

    print("=" * 62)
    print("")

    # ── Build comparison figure ───────────────────────────────────────────────
    cols  = len(results)
    fig, axes = plt.subplots(2, cols, figsize=(5 * cols, 9))
    if cols == 1:
        axes = np.array(axes).reshape(2, 1)

    fig.suptitle("CV Accuracy Challenge - Face Detection Comparison",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.patch.set_facecolor("#f8fafc")

    COLORS = {
        "NORMAL":   "#f59e0b",
        "DEGRADED": "#dc2626",
        "ENHANCED": "#16a34a",
    }

    for col, r in enumerate(results):
        color = COLORS.get(r["mode"], "#6366f1")

        # Top row: image with detection boxes
        box_bgr = tuple(int(c) for c in plt.matplotlib.colors.to_rgb(color))[::-1]
        box_bgr = (int(color[1:3], 16) if color.startswith("#") else 0,
                   0, 0)
        # simpler: just use fixed BGR per mode
        BGR = {"NORMAL": (0,180,255), "DEGRADED": (50,80,220), "ENHANCED": (0,180,60)}
        bgr_col = BGR.get(r["mode"], (200,200,0))

        drawn = draw_boxes(r["img"], r["faces"], bgr_col)
        rgb   = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)
        axes[0, col].imshow(rgb)
        axes[0, col].set_title(
            "{}\nMode: {} | Faces: {} | Acc: {:.0f}%".format(
                r["label"], r["mode"], r["n"], r["acc"]),
            color=color, fontweight="bold", fontsize=10)
        axes[0, col].axis("off")

        # Bottom row: histogram
        gray_img = cv2.cvtColor(r["img"], cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256]).flatten()
        axes[1, col].fill_between(range(256), hist, color=color, alpha=0.3)
        axes[1, col].plot(hist, color=color, linewidth=1.2)
        axes[1, col].set_facecolor("#f1f5f9")
        axes[1, col].set_title("Histogram", fontsize=9)
        axes[1, col].set_xlim([0, 256])
        axes[1, col].tick_params(labelsize=7)

    plt.tight_layout()
    out_path = "comparison_report.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print("Comparison chart saved --> " + out_path)
    print("Open it to see the full report!")
    print("")

# ── Load Haar Cascade ─────────────────────────────────────────────────────────

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    print("ERROR: Haar Cascade failed to load!")
    sys.exit(1)
print("Haar Cascade loaded OK")

# ── Open camera ───────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# ── State ─────────────────────────────────────────────────────────────────────

mode      = "NORMAL"
shot_n    = 0
frame_n   = 0
fps       = 0.0
fps_timer = time.time()
shots     = []        # list of saved screenshots for comparison

COLORS = {"NORMAL": (255,180,0), "DEGRADED": (50,80,220), "ENHANCED": (0,210,100)}
BG     = {"NORMAL": (60,40,0),   "DEGRADED": (80,20,20),  "ENHANCED": (0,60,30)}

print("")
print("==================================================")
print("  Vision Editor - Live Camera Face Detection")
print("==================================================")
print("  CLICK on the camera window first, then:")
print("  D --> Degrade ON/OFF  (darken + noise)")
print("  E --> Enhance ON/OFF  (Gamma + Hist.Eq)")
print("  S --> Save screenshot for comparison")
print("  Q --> Quit + generate comparison report")
print("==================================================")
print("  TIP: Press S once in each mode to get the best")
print("  comparison: NORMAL, DEGRADED, then ENHANCED.")
print("==================================================")
print("")

# ── Main loop ─────────────────────────────────────────────────────────────────

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    frame_n += 1
    if frame_n % 20 == 0:
        fps = 20.0 / max(time.time() - fps_timer, 0.001)
        fps_timer = time.time()

    # Apply selected mode
    if mode == "DEGRADED":
        disp = degrade_image(frame)
    elif mode == "ENHANCED":
        disp = enhance_image(frame)
    else:
        disp = frame.copy()

    # Detect faces on current frame
    faces, n, _ = detect_faces(disp, face_cascade)
    col = COLORS[mode]

    # Draw bounding boxes
    if n > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(disp, (x, y), (x+w, y+h), col, 2)
            cv2.rectangle(disp, (x, y-26), (x+w, y), col, -1)
            cv2.putText(disp, "Face " + str(w) + "x" + str(h),
                        (x+4, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    # Top HUD
    cv2.rectangle(disp, (0, 0), (640, 46), BG[mode], -1)
    cv2.rectangle(disp, (0, 46), (640, 47), col, -1)
    cv2.putText(disp, "Mode: " + mode,
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    cv2.putText(disp, "Faces: " + str(n) + "   FPS: " + str(int(fps)),
                (260, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220,220,220), 2)

    # Screenshots counter (top right)
    cv2.rectangle(disp, (520, 8), (632, 38), col, -1)
    cv2.putText(disp, "Shots:" + str(len(shots)),
                (526, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,0,0), 2)

    # Bottom hint bar
    cv2.rectangle(disp, (0, 455), (640, 480), (20,20,20), -1)
    cv2.putText(disp, "[D] Degrade  [E] Enhance  [S] Screenshot  [Q] Quit+Report",
                (6, 472), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160,160,160), 1)

    cv2.imshow("Vision Editor - Live Camera  (click here first!)", disp)

    key = cv2.waitKey(1) & 0xFF

    if key in (ord("q"), ord("Q"), 27):
        print("Closing camera...")
        break

    elif key in (ord("d"), ord("D")):
        mode = "NORMAL" if mode == "DEGRADED" else "DEGRADED"
        print("Mode: " + mode)

    elif key in (ord("e"), ord("E")):
        mode = "NORMAL" if mode == "ENHANCED" else "ENHANCED"
        print("Mode: " + mode)

    elif key in (ord("s"), ord("S")):
        shot_n += 1
        label = "Shot" + str(shot_n).zfill(2) + "_" + mode
        # Save a clean copy (without HUD) for the report
        if mode == "DEGRADED":
            clean = degrade_image(frame)
        elif mode == "ENHANCED":
            clean = enhance_image(frame)
        else:
            clean = frame.copy()
        shots.append({"label": label, "mode": mode, "img": clean})
        # Also save to disk
        fname = label + ".png"
        cv2.imwrite(fname, disp)
        print("Screenshot saved: " + fname + "  (total shots: " + str(len(shots)) + ")")

# ── Cleanup + generate report ─────────────────────────────────────────────────

cap.release()
cv2.destroyAllWindows()
print("")
print("Generating comparison report from " + str(len(shots)) + " screenshots...")
make_report(shots, face_cascade)
