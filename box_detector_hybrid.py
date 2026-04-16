"""
box_detector_hybrid.py
======================

Hybrid box detector for drone club use.

Primary method  : OWL-ViT (Google zero-shot AI) — detects a "cardboard box"
                  using a plain English text prompt, no training required.
Fallback method : OpenCV contour analysis — detects large rectangular shapes
                  when OWL-ViT finds nothing above the confidence threshold.

The program captures ONE frame per second from a USB camera, annotates it,
and shows the live result in an OpenCV window.

Colour coding
-------------
  GREEN  box  = detected by OWL-ViT (AI confidence shown as %)
  YELLOW box  = detected by OpenCV contour fallback

Requirements
------------
    pip install opencv-python transformers torch torchvision pillow

Usage
-----
    python box_detector_hybrid.py

Controls
--------
    q  - quit
    s  - save current annotated frame as PNG
    +  - raise OWL-ViT confidence threshold by 5 %
    -  - lower OWL-ViT confidence threshold by 5 %
"""

import sys
import time

import cv2
import numpy as np
from PIL import Image

# ── Configuration ──────────────────────────────────────────────────────────────

# USB camera device index (0 = default/first camera)
CAMERA_INDEX = 1

# Seconds between captures
CAPTURE_INTERVAL_S = 1.0

# OWL-ViT: minimum confidence to accept an AI detection (0.0 - 1.0)
# If nothing clears this bar the OpenCV fallback runs automatically.
OWLVIT_THRESHOLD = 0.10

# Text prompts fed to OWL-ViT — add synonyms to improve recall
TEXT_PROMPTS = [
    "a cardboard box",
    "a box on the ground",
    "a package",
]

# OpenCV fallback: minimum fraction of frame area a contour must cover
# to be reported (prevents tiny rectangles from spamming the screen)
MIN_CONTOUR_AREA_FRACTION = 0.01   # 1 % of frame area
MAX_CONTOUR_AREA_FRACTION = 0.90   # ignore if it almost fills the whole frame

# How many of the largest OpenCV rectangles to display at most
MAX_OPENCV_RESULTS = 3

# Display window
WINDOW_NAME = "Hybrid Box Detector  |  q=quit  s=save  +/-=threshold"

# ── Colours (BGR) ──────────────────────────────────────────────────────────────
COLOR_AI       = (0,  220,  80)   # green  - OWL-ViT detection
COLOR_FALLBACK = (0,  220, 220)   # yellow - OpenCV fallback
COLOR_TEXT_BG  = (20,  20,  20)
COLOR_TEXT     = (255, 255, 255)


# ══════════════════════════════════════════════════════════════════════════════
# OWL-ViT helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_owlvit():
    """Download (first run) and return the OWL-ViT processor + model."""
    try:
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
    except ImportError:
        print("[ERROR] transformers not installed.")
        print("        Run:  pip install transformers torch torchvision pillow")
        sys.exit(1)

    import torch
    model_name = "google/owlvit-base-patch32"
    print(f"[INFO] Loading OWL-ViT model ({model_name}) ...")
    print("       First run will download ~400 MB - please wait.")
    processor = OwlViTProcessor.from_pretrained(model_name)
    model     = OwlViTForObjectDetection.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"[INFO] OWL-ViT ready on device: {device}\n")
    return processor, model, device


def owlvit_detect(frame_bgr, processor, model, device, threshold):
    """
    Run OWL-ViT on a BGR frame.

    Returns a list of dicts:
        [{'x1', 'y1', 'x2', 'y2', 'score', 'label'}, ...]
    sorted by descending score, filtered by threshold.
    """
    import torch

    h, w = frame_bgr.shape[:2]
    pil_image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    inputs = processor(
        text=TEXT_PROMPTS,
        images=pil_image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process: map boxes back to original image dimensions
    target_sizes = torch.tensor([[h, w]], device=device)
    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=target_sizes
    )[0]

    detections = []
    for score, label_idx, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        score_val = float(score)
        box_vals  = box.tolist()   # [x1, y1, x2, y2] in pixel coords
        x1, y1, x2, y2 = (
            max(0, int(box_vals[0])),
            max(0, int(box_vals[1])),
            min(w, int(box_vals[2])),
            min(h, int(box_vals[3])),
        )
        prompt_label = TEXT_PROMPTS[label_idx % len(TEXT_PROMPTS)]
        detections.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "score": score_val,
            "label": prompt_label,
            "method": "OWL-ViT",
        })

    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections


# ══════════════════════════════════════════════════════════════════════════════
# OpenCV contour fallback
# ══════════════════════════════════════════════════════════════════════════════

def opencv_detect_boxes(frame_bgr):
    """
    Detect large rectangular contours in the frame.

    Pipeline:
      BGR -> grayscale -> Gaussian blur -> Canny edges
      -> dilate -> find contours -> approx polygon -> filter quads

    Returns a list of dicts compatible with owlvit_detect output.
    """
    h, w     = frame_bgr.shape[:2]
    frame_area = h * w
    min_area   = MIN_CONTOUR_AREA_FRACTION * frame_area
    max_area   = MAX_CONTOUR_AREA_FRACTION * frame_area

    gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Adaptive threshold works better across lighting conditions
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=21, C=4
    )

    # Also run Canny and merge - catches boxes with subtle edges
    edges  = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    merged = cv2.dilate(cv2.bitwise_or(thresh, edges), kernel, iterations=2)

    contours, _ = cv2.findContours(
        merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area < area < max_area):
            continue

        # Approximate the contour to a polygon
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # We want quadrilaterals (4 corners) — box-like shapes
        if len(approx) < 4 or len(approx) > 8:
            continue

        bx, by, bw, bh = cv2.boundingRect(approx)

        # Aspect ratio guard — extremely thin slivers are probably not boxes
        aspect = bw / bh if bh > 0 else 0
        if aspect < 0.15 or aspect > 8.0:
            continue

        # Convexity check — a box should be mostly convex
        hull       = cv2.convexHull(cnt)
        hull_area  = cv2.contourArea(hull)
        solidity   = area / hull_area if hull_area > 0 else 0
        if solidity < 0.60:
            continue

        boxes.append({
            "x1": bx, "y1": by,
            "x2": bx + bw, "y2": by + bh,
            "score": solidity,        # use solidity as a proxy confidence
            "label": "rectangular shape",
            "method": "OpenCV",
            "area": area,
        })

    # Sort by area descending, keep top N
    boxes.sort(key=lambda b: b["area"], reverse=True)
    return boxes[:MAX_OPENCV_RESULTS]


# ══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ══════════════════════════════════════════════════════════════════════════════

def draw_detection(frame, det):
    """Draw one bounding box + label on the frame."""
    color = COLOR_AI if det["method"] == "OWL-ViT" else COLOR_FALLBACK
    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]

    # Box outline (thick)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Corner accents for a cleaner look
    corner_len = max(10, min(30, (x2 - x1) // 6, (y2 - y1) // 6))
    thickness  = 3
    for cx, cy, sx, sy in [
        (x1, y1,  1,  1), (x2, y1, -1,  1),
        (x1, y2,  1, -1), (x2, y2, -1, -1),
    ]:
        cv2.line(frame, (cx, cy), (cx + sx * corner_len, cy), color, thickness)
        cv2.line(frame, (cx, cy), (cx, cy + sy * corner_len), color, thickness)

    # Label text
    if det["method"] == "OWL-ViT":
        tag = f"  {det['label']}  {det['score']:.0%}  [AI]"
    else:
        tag = f"  {det['label']}  [CV fallback]"

    (tw, th), bl = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    ty = max(y1 - 4, th + 6)
    cv2.rectangle(frame,
                  (x1,      ty - th - bl - 4),
                  (x1 + tw, ty + 2),
                  color, cv2.FILLED)
    cv2.putText(frame, tag,
                (x1, ty - bl - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (20, 20, 20), 1, cv2.LINE_AA)
    return frame


def draw_hud(frame, method_used, n_dets, elapsed_ms, frame_idx, threshold):
    """Semi-transparent HUD bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_h = 34

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    if n_dets:
        indicator_color = COLOR_AI if method_used == "OWL-ViT" else COLOR_FALLBACK
        status_text = (
            f"  #{frame_idx}  |  "
            f"BOX DETECTED via {method_used}  ({n_dets} result{'s' if n_dets>1 else ''})  |  "
            f"{elapsed_ms:.0f} ms  |  threshold: {threshold:.0%}  (+/-)"
        )
    else:
        indicator_color = (80, 80, 80)
        status_text = (
            f"  #{frame_idx}  |  No box detected  |  "
            f"{elapsed_ms:.0f} ms  |  threshold: {threshold:.0%}  (+/-)"
        )

    # Coloured left edge bar
    cv2.rectangle(frame, (0, h - bar_h), (5, h), indicator_color, cv2.FILLED)

    cv2.putText(frame, status_text,
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                (210, 210, 210), 1, cv2.LINE_AA)
    return frame


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Load OWL-ViT ──────────────────────────────────────────────────────────
    processor, owlvit_model, device = load_owlvit()

    # ── Open camera ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}. "
              "Change CAMERA_INDEX at the top of the file.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"[INFO] Camera opened  "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} × "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} px")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 740)

    print(f"[INFO] Text prompts   : {TEXT_PROMPTS}")
    print(f"[INFO] AI threshold   : {OWLVIT_THRESHOLD:.0%}  (press +/- to adjust live)")
    print(f"[INFO] Capture rate   : 1 frame every {CAPTURE_INTERVAL_S}s")
    print("[INFO] Starting loop ... Press  q  to quit,  s  to save.\n")

    threshold        = OWLVIT_THRESHOLD
    last_capture     = 0.0
    frame_idx        = 0
    annotated_frame  = None

    try:
        while True:
            # Keep draining the camera buffer so we always get the freshest frame
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            now = time.monotonic()
            if now - last_capture >= CAPTURE_INTERVAL_S:
                last_capture = now
                frame_idx   += 1
                t0           = time.monotonic()

                # ── Step 1: OWL-ViT ───────────────────────────────────────────
                print(f"[Frame #{frame_idx}] OWL-ViT ...", end=" ", flush=True)
                ai_dets = owlvit_detect(
                    frame, processor, owlvit_model, device, threshold
                )

                method_used = "OWL-ViT"
                detections  = ai_dets

                # ── Step 2: OpenCV fallback if AI found nothing ────────────────
                if not ai_dets:
                    print("no AI hit -> OpenCV fallback ...", end=" ", flush=True)
                    cv_dets     = opencv_detect_boxes(frame)
                    method_used = "OpenCV" if cv_dets else "none"
                    detections  = cv_dets

                elapsed_ms = (time.monotonic() - t0) * 1000

                # ── Annotate frame ────────────────────────────────────────────
                annotated = frame.copy()
                for det in detections:
                    draw_detection(annotated, det)
                draw_hud(annotated, method_used, len(detections),
                         elapsed_ms, frame_idx, threshold)

                annotated_frame = annotated

                # ── Console log ───────────────────────────────────────────────
                if detections:
                    parts = [
                        f"{d['label']} ({d['score']:.0%}) via {d['method']}"
                        for d in detections
                    ]
                    print(f"{elapsed_ms:.0f} ms  ->  {' | '.join(parts)}")
                else:
                    print(f"{elapsed_ms:.0f} ms  ->  nothing detected")

            # Show newest annotated frame (show live feed until first inference)
            display = annotated_frame if annotated_frame is not None else frame
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] Quit.")
                break
            elif key == ord('s') and annotated_frame is not None:
                fname = f"detection_{frame_idx:04d}.png"
                cv2.imwrite(fname, annotated_frame)
                print(f"[INFO] Saved -> {fname}")
            elif key == ord('+') or key == ord('='):
                threshold = min(0.95, round(threshold + 0.05, 2))
                print(f"[INFO] Threshold -> {threshold:.0%}")
            elif key == ord('-'):
                threshold = max(0.01, round(threshold - 0.05, 2))
                print(f"[INFO] Threshold -> {threshold:.0%}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released. Done.")


if __name__ == "__main__":
    main()
