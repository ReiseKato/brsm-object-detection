from ultralytics import YOLO
import time
import json
import os

# Path to your test image
IMAGE_PATH = "test.jpg"

# Paths to the different model formats
MODELS = [
    "models/best-yolov8s.pt",            # PyTorch
    "models/best-yolov8s.onnx",          # ONNX
    "models/best-yolov8s.engine",        # TensorRT
    "models/best-yolov8s-fp16.engine",   # TensorRT FP16
]

# Common inference settings
IMGSZ = 1920
CONF = 0.1
IOU = 0.6

def boxes_to_list(boxes_obj, names_map=None):
    """Convert ultralytics Boxes-like object to a list of dicts.

    Each detection dict contains: xyxy, conf, cls_id, cls_name (if available).
    """
    detections = []
    if boxes_obj is None:
        return detections

    # Try to extract arrays/lists for xyxy, conf, cls
    try:
        xyxy = boxes_obj.xyxy.cpu().numpy().tolist()
    except Exception:
        try:
            xyxy = boxes_obj.xyxy.tolist()
        except Exception:
            xyxy = []

    try:
        confs = boxes_obj.conf.cpu().numpy().tolist()
    except Exception:
        try:
            confs = boxes_obj.conf.tolist()
        except Exception:
            confs = []

    try:
        cls_ids = boxes_obj.cls.cpu().numpy().tolist()
    except Exception:
        try:
            cls_ids = boxes_obj.cls.tolist()
        except Exception:
            cls_ids = []

    # Build detection dicts
    n = max(len(xyxy), len(confs), len(cls_ids))
    for i in range(n):
        box = xyxy[i] if i < len(xyxy) else []
        conf = float(confs[i]) if i < len(confs) else None
        cls_id = int(cls_ids[i]) if i < len(cls_ids) else None
        cls_name = None
        if names_map is not None and cls_id is not None:
            try:
                cls_name = names_map.get(cls_id, None) if isinstance(names_map, dict) else names_map[cls_id]
            except Exception:
                cls_name = None

        detections.append({
            "xyxy": box,
            "conf": conf,
            "cls_id": cls_id,
            "cls_name": cls_name,
        })

    return detections


for model_path in MODELS:
    print(f"\n--- Testing model: {model_path} ---")
    model = YOLO(model_path)

    # Warmup (especially important for TensorRT and ONNX)
    _ = model(IMAGE_PATH, imgsz=IMGSZ, conf=CONF, iou=IOU)

    # Timed inference
    start = time.time()
    results = model(IMAGE_PATH, imgsz=IMGSZ, conf=CONF, iou=IOU)
    end = time.time()

    total_time = end - start

    # Detections and per-detection info
    boxes_obj = getattr(results[0], "boxes", None)
    # names mapping: try results then model
    names_map = getattr(results[0], "names", None) or getattr(model, "names", None)
    detections = boxes_to_list(boxes_obj, names_map=names_map)
    det_count = len(detections)

    # Per-step timings (if available)
    per_step = None
    try:
        per_step = getattr(results[0], "speed", None)
    except Exception:
        per_step = None

    # Print summary
    print(f"Detections: {det_count}")
    print(f"Speed (total): {total_time:.3f}s")
    print(f"Per-step timings: {per_step}")

    # Save the annotated result (most versions save into runs/detect/...)
    save_result = None
    try:
        save_result = results[0].save()  # may return path or list
    except Exception:
        save_result = None

    print(f"Saved labeled image to: {save_result}")

    # Build JSON result
    model_name = os.path.basename(model_path)
    # Create a friendly prefix based on file extension so outputs are unique
    _, ext = os.path.splitext(model_name)
    ext = ext.lower()
    prefix_map = {
        ".engine": "TensorRT",
        ".onnx": "ONNX",
        ".pt": "PyTorch",
        ".pth": "PyTorch",
    }
    prefix = prefix_map.get(ext, ext.replace('.', '').upper())

    json_obj = {
        "model_path": model_path,
        "model_name": model_name,
        "image": IMAGE_PATH,
        "detections": detections,
        "detection_count": det_count,
        "timings": {
            "total": total_time,
            "per_step": per_step,
        },
        "annotated_image": save_result,
    }

    # Save JSON file next to script with model-based name and prefix by format
    base = os.path.splitext(model_name)[0]
    out_json = f"results/{prefix}_{base}_results.json"
    try:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(json_obj, f, indent=2)
        print(f"Saved JSON results to: {os.path.abspath(out_json)}")
    except Exception as e:
        print("Failed to write JSON results:", e)
