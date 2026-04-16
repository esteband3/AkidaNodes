"""
camera_box_detector.py
======================

Live USB camera object detection using the Akida YOLOv2 model pretrained
on PASCAL-VOC 2007 (20 classes).

Captures one frame per second from a USB camera, runs inference on the
Akida neuromorphic model, and draws bounding boxes + class labels on the
live display window.

NOTE ON "BOX" DETECTION
-----------------------
The PASCAL-VOC dataset does not include a generic "box" or "cardboard box"
class. The 20 VOC classes are listed in VOC_LABELS below. If you need to
detect a specific box type you will need a custom-trained model.

For quick experimentation, TARGET_CLASSES below can be set to a subset of
VOC_LABELS (e.g. ['bottle'] or left empty [] to show ALL detections).

Requirements
------------
    pip install opencv-python akida akida-models cnn2snn tensorflow tf-keras

Usage
-----
    python camera_box_detector.py

Controls
--------
    q  – quit the detection loop
    s  – save the current annotated frame as a PNG
"""

import time
import sys

import cv2
import numpy as np

# ── Akida / model imports ──────────────────────────────────────────────────────
from akida_models import yolo_voc_pretrained
from akida_models.detection.processing import preprocess_image, decode_output
from cnn2snn import convert
from tf_keras import Model

# ── Configuration ──────────────────────────────────────────────────────────────

# Index of the USB camera device (0 = first/default camera, 1 = second, …)
CAMERA_INDEX = 0

# Seconds between captures (set to 1 for one frame per second)
CAPTURE_INTERVAL_S = 1.0

# Minimum confidence score to display a detection (0.0 – 1.0)
CONFIDENCE_THRESHOLD = 0.30

# Leave empty [] to show ALL detected classes, or list specific ones to filter.
# Must be valid names from VOC_LABELS.
# Example: TARGET_CLASSES = ['bottle', 'chair']
TARGET_CLASSES = []

# Display window name
WINDOW_NAME = "Akida YOLO – Box Detector  |  q=quit  s=save"

# ── VOC class labels ───────────────────────────────────────────────────────────
VOC_LABELS = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# ── Colour palette (BGR) – one colour per VOC class ───────────────────────────
PALETTE = [
    (255,  56,  56), (255, 157,  151), (255, 112,  31), (255, 178,  29),
    (207, 210,  49), ( 72, 249,  10), (146, 204,  23), ( 61, 219, 134),
    ( 26, 147, 52),  ( 0, 212, 187), ( 44, 153, 168), (0,   194, 255),
    ( 52,  69, 147), (100,  115, 255), (0,   24, 236), (132,  56, 255),
    (82,   0, 133),  (203,  56, 255), (255, 149, 200), (255,  55, 198),
]


def load_akida_model():
    """Load the pretrained VOC YOLO Keras model, convert it to an Akida model,
    and return (akida_model, anchors, grid_size)."""

    print("[INFO] Loading pretrained YOLO-VOC Keras model …")
    model_keras, anchors = yolo_voc_pretrained()

    grid_h, grid_w = 7, 7
    num_anchors    = len(anchors)
    num_classes    = len(VOC_LABELS)

    # Strip the detection head so cnn2snn can convert the model
    compatible_model = Model(model_keras.input, model_keras.layers[-1].output)

    print("[INFO] Converting Keras model → Akida model …")
    model_akida = convert(compatible_model)
    model_akida.summary()

    print("[INFO] Model ready.\n")
    return model_akida, anchors, (grid_h, grid_w)


def open_camera(index: int) -> cv2.VideoCapture:
    """Open a USB camera and raise if it cannot be opened."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera at index {index}. "
            "Try a different CAMERA_INDEX value."
        )
    # Prefer 1280×720 if the camera supports it (falls back gracefully)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"[INFO] Camera opened  →  "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} × "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} px")
    return cap


def run_inference(frame_bgr, model_akida, anchors, grid_size):
    """
    Run YOLO inference on a single BGR OpenCV frame.

    Returns a list of dicts:
        [{'label': str, 'score': float, 'x1': int, 'y1': int,
          'x2': int, 'y2': int, 'class_id': int}, …]
    """
    h_orig, w_orig = frame_bgr.shape[:2]

    # Convert BGR → RGB and wrap in a tf-compatible array
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Determine the input spatial dimensions expected by the model
    input_dims = model_akida.layers[0].input_dims  # (H, W, C)
    input_shape = (input_dims[0], input_dims[1])

    # Preprocess: resize + normalise to uint8
    image_pre   = preprocess_image(frame_rgb, input_shape)
    input_batch = image_pre[np.newaxis, :].astype(np.uint8)

    # Forward pass
    raw_pots = model_akida.predict(input_batch)[0]

    # Reshape potentials → (grid_h, grid_w, num_anchors, 5 + num_classes)
    gh, gw = grid_size
    na     = len(anchors)
    nc     = len(VOC_LABELS)
    pots   = raw_pots.reshape((gh, gw, na, 4 + 1 + nc))

    # Decode into BoundingBox objects
    raw_boxes = decode_output(pots, anchors, nc)

    # Filter by confidence and build result list
    results = []
    target_ids = (
        [VOC_LABELS.index(c) for c in TARGET_CLASSES if c in VOC_LABELS]
        if TARGET_CLASSES else list(range(nc))
    )

    for box in raw_boxes:
        score    = box.get_score()
        class_id = box.get_label()

        if score < CONFIDENCE_THRESHOLD:
            continue
        if class_id not in target_ids:
            continue

        results.append({
            'label':    VOC_LABELS[class_id],
            'score':    score,
            'class_id': class_id,
            'x1': int(box.x1 * w_orig),
            'y1': int(box.y1 * h_orig),
            'x2': int(box.x2 * w_orig),
            'y2': int(box.y2 * h_orig),
        })

    return results


def draw_detections(frame_bgr, detections):
    """Draw bounding boxes and labels onto the frame in-place."""
    annotated = frame_bgr.copy()

    for det in detections:
        colour = PALETTE[det['class_id'] % len(PALETTE)]
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        label_text = f"{det['label']}  {det['score']:.0%}"

        # Bounding box rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

        # Label background pill
        (tw, th), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        top = max(y1 - th - baseline - 4, 0)
        cv2.rectangle(annotated,
                      (x1, top),
                      (x1 + tw + 4, top + th + baseline + 4),
                      colour, cv2.FILLED)

        # Label text
        cv2.putText(annotated, label_text,
                    (x1 + 2, top + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

    return annotated


def draw_status_bar(frame, n_detections, elapsed_ms, capture_idx):
    """Overlay a status bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_h = 30
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (30, 30, 30), cv2.FILLED)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    status = (f"  Frame #{capture_idx}  |  "
              f"Detections: {n_detections}  |  "
              f"Inference: {elapsed_ms:.0f} ms  |  "
              f"Interval: {CAPTURE_INTERVAL_S}s")
    cv2.putText(frame, status,
                (6, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA)
    return frame


# ── Main detection loop ────────────────────────────────────────────────────────

def main():
    # ── Load model ────────────────────────────────────────────────────────────
    model_akida, anchors, grid_size = load_akida_model()

    # ── Open camera ───────────────────────────────────────────────────────────
    try:
        cap = open_camera(CAMERA_INDEX)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    if TARGET_CLASSES:
        print(f"[INFO] Filtering for classes: {TARGET_CLASSES}")
    else:
        print("[INFO] Showing ALL detected classes.")

    print(f"[INFO] Confidence threshold : {CONFIDENCE_THRESHOLD:.0%}")
    print(f"[INFO] Capture interval     : {CAPTURE_INTERVAL_S}s")
    print("[INFO] Starting detection loop … press  q  to quit,  s  to save frame.\n")

    last_capture_time = 0.0
    capture_idx       = 0
    annotated_frame   = None
    last_raw_frame    = None

    try:
        while True:
            now = time.monotonic()

            # Always grab + decode from the buffer to keep it fresh, but only
            # run inference once per interval.
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame from camera – retrying …")
                time.sleep(0.1)
                continue

            if now - last_capture_time >= CAPTURE_INTERVAL_S:
                last_capture_time = now
                capture_idx      += 1
                last_raw_frame    = frame.copy()

                print(f"[Frame #{capture_idx}] Running inference …", end=" ", flush=True)
                t0 = time.monotonic()
                detections = run_inference(frame, model_akida, anchors, grid_size)
                elapsed_ms = (time.monotonic() - t0) * 1000

                annotated_frame = draw_detections(frame, detections)
                draw_status_bar(annotated_frame, len(detections), elapsed_ms, capture_idx)

                if detections:
                    labels_found = ", ".join(
                        f"{d['label']} ({d['score']:.0%})" for d in detections
                    )
                    print(f"{elapsed_ms:.0f} ms  →  {labels_found}")
                else:
                    print(f"{elapsed_ms:.0f} ms  →  no detections above threshold")

            # Show the most recently annotated frame (or raw if not yet analysed)
            display = annotated_frame if annotated_frame is not None else frame
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] Quit requested – exiting.")
                break
            elif key == ord('s') and last_raw_frame is not None:
                fname = f"detection_frame_{capture_idx:04d}.png"
                cv2.imwrite(fname, annotated_frame)
                print(f"[INFO] Saved annotated frame → {fname}")

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt – exiting.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released. Done.")


if __name__ == "__main__":
    main()
