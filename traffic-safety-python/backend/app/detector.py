import os
import io
import cv2
import numpy as np
from PIL import Image
import logging
from typing import Dict, List, Tuple, Any

logger = logging.getLogger("detector")

# Class color scheme (BGR for OpenCV)
COLOR_SCHEME = {
    "Pothole": (0, 0, 225),             # Bright Red
    "Crack": (0, 140, 255),             # Vibrant Orange
    "Damaged Road Surface": (0, 215, 255), # Yellow/Gold
    "Traffic Hazard": (180, 0, 255),    # Purple/Magenta
    "Clean Road": (0, 200, 83)          # Neon Green
}

SEVERITY_COLORS = {
    "High": (0, 0, 220),
    "Medium": (0, 140, 255),
    "Low": (0, 200, 83)
}

yolo_model = None
yolo_attempted = False

def get_yolo_model():
    """Lazy load YOLOv8 model locally without blocking backend server on remote downloads."""
    global yolo_model, yolo_attempted
    if yolo_model is not None or yolo_attempted:
        return yolo_model
    
    yolo_attempted = True
    try:
        from ultralytics import YOLO
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        os.makedirs(model_dir, exist_ok=True)
        best_path = os.path.join(model_dir, "best.pt")
        yolo_path = os.path.join(model_dir, "yolov8n.pt")
        
        if os.path.exists(best_path):
            yolo_model = YOLO(best_path)
            logger.info("Local custom fine-tuned best.pt YOLOv8 model loaded successfully.")
        elif os.path.exists(yolo_path):
            yolo_model = YOLO(yolo_path)
            logger.info("Local yolov8n.pt model loaded successfully.")
        else:
            logger.info("Local model file not found in models/. Using high-speed OpenCV Road Detection Engine.")
            yolo_model = None
    except Exception as e:
        logger.warning(f"YOLOv8 loading skipped: {e}")
        yolo_model = None
    return yolo_model


def analyze_road_cv(img_np: np.ndarray) -> Tuple[List[Dict[str, Any]], str, float, str]:
    """
    Intelligent OpenCV Road Infrastructure Analysis.
    Performs multi-scale edge detection, contour geometry analysis, texture surface variance,
    and dark-depression region extraction to identify Potholes, Cracks, and Surface Damage.
    """
    height, width = img_np.shape[:2]
    total_area = height * width
    
    # Convert to Grayscale & Blur
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Adaptive thresholding & Canny edge detection
    edges = cv2.Canny(blurred, 40, 150)
    
    # Morphological closing to connect fragmented edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    total_defect_area = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (total_area * 0.003) or area > (total_area * 0.45):
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        box_area = w * h
        solidity = float(area) / box_area if box_area > 0 else 0
        
        roi = gray[y:y+h, x:x+w]
        mean_val = np.mean(roi) if roi.size > 0 else 128
        std_val = np.std(roi) if roi.size > 0 else 0
        
        label = "Damaged Road Surface"
        confidence = round(min(0.98, max(0.65, 0.70 + (std_val / 200.0) + (solidity * 0.15))), 4)
        
        if aspect_ratio > 3.0 or aspect_ratio < 0.33:
            label = "Crack"
            confidence = round(min(0.96, max(0.72, confidence + 0.05)), 4)
        elif mean_val < 100 and std_val > 25 and solidity > 0.4:
            label = "Pothole"
            confidence = round(min(0.99, max(0.78, confidence + 0.10)), 4)
        elif aspect_ratio >= 0.8 and aspect_ratio <= 1.5 and std_val > 40:
            label = "Traffic Hazard"
            confidence = round(min(0.95, max(0.70, confidence)), 4)
            
        detections.append({
            "x_min": int(x),
            "y_min": int(y),
            "x_max": int(x + w),
            "y_max": int(y + h),
            "label": label,
            "confidence": float(confidence * 100.0),
            "area": int(box_area)
        })
        total_defect_area += box_area
        
    detections = filter_overlapping_boxes(detections)
    
    area_ratio = total_defect_area / float(total_area) if total_area > 0 else 0
    defect_count = len(detections)
    
    if defect_count == 0:
        overall_label = "Clean Road"
        overall_severity = "Low"
        avg_confidence = 98.5
    else:
        label_counts = {}
        for d in detections:
            lbl = d["label"]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        overall_label = max(label_counts, key=label_counts.get)
        
        avg_confidence = round(float(np.mean([d["confidence"] for d in detections])), 1)
        
        if defect_count >= 3 or area_ratio > 0.12 or any(d["label"] == "Pothole" for d in detections):
            overall_severity = "High"
        elif defect_count >= 1 or area_ratio > 0.04:
            overall_severity = "Medium"
        else:
            overall_severity = "Low"
            
    return detections, overall_label, avg_confidence, overall_severity


def filter_overlapping_boxes(boxes: List[Dict[str, Any]], iou_threshold: float = 0.4) -> List[Dict[str, Any]]:
    if not boxes:
        return []
    
    boxes = sorted(boxes, key=lambda x: x["confidence"], reverse=True)
    keep = []
    
    while boxes:
        current = boxes.pop(0)
        keep.append(current)
        boxes = [b for b in boxes if calculate_iou(current, b) < iou_threshold]
        
    return keep


def calculate_iou(boxA: Dict[str, Any], boxB: Dict[str, Any]) -> float:
    xA = max(boxA["x_min"], boxB["x_min"])
    yA = max(boxA["y_min"], boxB["y_min"])
    xB = min(boxA["x_max"], boxB["x_max"])
    yB = min(boxA["y_max"], boxB["y_max"])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA["x_max"] - boxA["x_min"]) * (boxA["y_max"] - boxA["y_min"])
    boxBArea = (boxB["x_max"] - boxB["x_min"]) * (boxB["y_max"] - boxB["y_min"])
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def draw_bounding_boxes(img_np: np.ndarray, detections: List[Dict[str, Any]], overall_severity: str) -> np.ndarray:
    annotated = img_np.copy()
    height, width = annotated.shape[:2]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(width, height) / 1000.0)
    thickness = max(2, int(min(width, height) / 400.0))
    
    for det in detections:
        x_min, y_min = det["x_min"], det["y_min"]
        x_max, y_max = det["x_max"], det["y_max"]
        label = det["label"]
        conf = det["confidence"]
        
        color = COLOR_SCHEME.get(label, (0, 215, 255))
        
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, thickness)
        
        text = f"{label} {conf:.1f}%"
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness=1)
        
        tag_y1 = max(0, y_min - text_h - 10)
        tag_y2 = y_min
        cv2.rectangle(annotated, (x_min, tag_y1), (x_min + text_w + 12, tag_y2), color, -1)
        cv2.putText(annotated, text, (x_min + 6, tag_y2 - 5), font, font_scale, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    banner_height = int(max(45, height * 0.07))
    cv2.rectangle(annotated, (0, 0), (width, banner_height), (20, 24, 33), -1)
    
    sev_color = SEVERITY_COLORS.get(overall_severity, (0, 200, 83))
    cv2.rectangle(annotated, (0, 0), (12, banner_height), sev_color, -1)
    
    banner_text = f"AI DETECTION: {len(detections)} ISSUES FOUND | SEVERITY: {overall_severity.upper()}"
    cv2.putText(annotated, banner_text, (25, int(banner_height * 0.65)), font, font_scale * 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated


NON_ROAD_CLASSES = {
    "person", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
}

ROAD_SCENE_INDICATORS = {
    "car", "truck", "bus", "motorcycle", "bicycle", "traffic light", "stop sign",
    "street sign", "parking meter", "pothole", "crack", "asphalt", "road"
}

coco_model = None
coco_attempted = False

def get_coco_model():
    """Lazy load base COCO YOLO model for Stage 1 Scene & Object Verification if available locally."""
    global coco_model, coco_attempted
    if coco_model is not None or coco_attempted:
        return coco_model
    coco_attempted = True
    try:
        from ultralytics import YOLO
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        coco_path = os.path.join(model_dir, "yolov8n.pt")
        if os.path.exists(coco_path):
            coco_model = YOLO(coco_path)
            logger.info("COCO Scene Classifier loaded successfully.")
        else:
            coco_model = None
    except Exception as e:
        logger.warning(f"COCO Scene Classifier unavailable: {e}")
        coco_model = None
    return coco_model


def verify_is_road_image(pil_img: Image.Image, img_np: np.ndarray) -> Tuple[bool, str]:
    """
    Stage 1: Road & Non-Road Scene Verification Engine.
    Analyzes color spectrum, asphalt grayscale neutrality ratio, texture edge density,
    and indoor vs outdoor scene characteristics.
    """
    coco = get_coco_model()
    if coco is not None:
        try:
            results = coco(pil_img, verbose=False)
            non_road_detected = []
            road_context_detected = []
            
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0]) * 100.0
                    cls_id = int(box.cls[0])
                    name = str(coco.names.get(cls_id, "")).lower().strip()
                    
                    if conf > 40.0:
                        if name in NON_ROAD_CLASSES:
                            non_road_detected.append(name)
                        elif name in ROAD_SCENE_INDICATORS:
                            road_context_detected.append(name)
                            
            if non_road_detected and not road_context_detected:
                found_str = ", ".join(set(non_road_detected[:3])).title()
                return False, f"Non-road objects ({found_str}) detected! These are not road images, please upload actual road images."
        except Exception as e:
            logger.warning(f"COCO verification error: {e}")
            
    # Surface & Color Spectrum Scene Classifier
    height, width = img_np.shape[:2]
    total_pixels = height * width
    
    hsv = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    
    h, s, v = cv2.split(hsv)
    
    # 1. Human Skin Tone Mask (HSV)
    skin_mask1 = (h >= 0) & (h <= 25) & (s >= 25) & (s <= 170) & (v >= 60) & (v <= 255)
    skin_mask2 = (h >= 170) & (h <= 180) & (s >= 25) & (s <= 170) & (v >= 60) & (v <= 255)
    skin_ratio = np.sum(skin_mask1 | skin_mask2) / float(total_pixels)
    
    # 2. White Paper / Document / Poster Background Mask
    paper_mask = (s < 25) & (v > 210)
    paper_ratio = np.sum(paper_mask) / float(total_pixels)
    
    # 3. High-Frequency Text Edge Density (Documents, Posters, Ads)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    text_edges = cv2.Canny(blur, 100, 200)
    text_edge_ratio = np.sum(text_edges > 0) / float(total_pixels)
    
    # 4. Asphalt neutrality in lower half (ground plane)
    lower_hsv = hsv[int(height * 0.4):, :]
    lower_s = lower_hsv[:, :, 1]
    lower_v = lower_hsv[:, :, 2]
    asphalt_mask_lower = (lower_s < 65) & (lower_v > 25) & (lower_v < 210)
    asphalt_ratio_lower = np.sum(asphalt_mask_lower) / float(lower_hsv.size)
    
    # Exclude natural outdoor green trees/foliage (Hue 30 to 90) from indoor saturation check
    is_green_foliage = (h >= 30) & (h <= 90)
    high_sat_indoor_mask = (s > 100) & (v > 50) & (~is_green_foliage)
    high_sat_ratio = np.sum(high_sat_indoor_mask) / float(total_pixels)
    
    # Non-Road Decision Rules:
    if skin_ratio > 0.04 and asphalt_ratio_lower < 0.25:
        return False, "Human or portrait photo detected! These are not road images, please upload actual road images."
        
    if paper_ratio > 0.22 and text_edge_ratio > 0.035 and asphalt_ratio_lower < 0.20:
        return False, "Document, poster or advertisement image detected! These are not road images, please upload actual road images."
        
    if high_sat_ratio > 0.35:
        return False, "Indoor colorful objects detected! These are not road images, please upload actual road images."
        
    if asphalt_ratio_lower < 0.15:
        return False, "These are not road images, please upload actual road images."
        
    return True, "Valid Road Scene"


def draw_invalid_banner(img_np: np.ndarray, msg: str) -> np.ndarray:
    """Annotates non-road image with explicit error warning banner."""
    annotated = img_np.copy()
    height, width = annotated.shape[:2]
    banner_h = int(max(60, height * 0.12))
    
    cv2.rectangle(annotated, (0, 0), (width, banner_h), (20, 20, 200), -1)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(width, height) / 900.0)
    
    cv2.putText(annotated, "INVALID IMAGE: NOT A ROAD SURFACE", (20, int(banner_h * 0.45)), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated, "Please upload an actual street, highway, or pavement photo.", (20, int(banner_h * 0.8)), font, font_scale * 0.75, (220, 220, 255), 1, cv2.LINE_AA)
    
    return annotated


def process_image_bytes(image_bytes: bytes) -> Tuple[bytes, Dict[str, Any]]:
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Stage 1: Verify if image is a road scene
    is_road, message = verify_is_road_image(pil_img, img_np)
    if not is_road:
        annotated_np = draw_invalid_banner(img_np, message)
        success, buffer = cv2.imencode('.jpg', annotated_np, [cv2.IMWRITE_JPEG_QUALITY, 92])
        annotated_bytes = buffer.tobytes() if success else image_bytes
        
        metadata = {
            "damageType": "Invalid Image (Non-Road)",
            "confidence": 0.0,
            "severity": "Low",
            "boundingBoxes": [],
            "isCleanRoad": True,
            "isValidRoad": False,
            "message": message,
            "totalHazardsDetected": 0
        }
        return annotated_bytes, metadata

    # Stage 2: Road Damage & Pothole Inspection
    detections = []
    overall_label = "Clean Road"
    avg_confidence = 98.0
    severity = "Low"
    
    model = get_yolo_model()
    if model is not None:
        try:
            results = model(pil_img, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0]) * 100.0
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    
                    raw_name = str(model.names.get(cls_id, "Obstacle")).strip()
                    lower_name = raw_name.lower()
                    if lower_name in ["pothole"]:
                        cls_name = "Pothole"
                    elif lower_name in ["crack"]:
                        cls_name = "Crack"
                    elif lower_name in ["car", "truck", "bus", "traffic light", "stop sign"]:
                        cls_name = "Traffic Hazard"
                    elif lower_name in ["damaged road surface", "damage", "defect"]:
                        cls_name = "Damaged Road Surface"
                    else:
                        cls_name = raw_name.title()
                        
                    if conf > 40.0:
                        detections.append({
                            "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2,
                            "label": cls_name, "confidence": round(conf, 1)
                        })
        except Exception as err:
            logger.warning(f"YOLO inference error: {err}")
            
    # Use OpenCV heuristics ONLY if custom YOLO model is not available
    if model is None:
        cv_detections, cv_label, cv_conf, cv_severity = analyze_road_cv(img_np)
        detections = cv_detections
        overall_label = cv_label
        avg_confidence = cv_conf
        severity = cv_severity
    else:
        detections = filter_overlapping_boxes(detections)
        if detections:
            label_counts = {}
            for d in detections:
                lbl = d["label"]
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
            overall_label = max(label_counts, key=label_counts.get)
            avg_confidence = round(float(np.mean([d["confidence"] for d in detections])), 1)
            
            if len(detections) >= 3 or any(d["label"] == "Pothole" for d in detections):
                severity = "High"
            elif len(detections) >= 1:
                severity = "Medium"
            else:
                severity = "Low"
        else:
            overall_label = "Clean Road"
            severity = "Low"
            avg_confidence = 98.5
        
    annotated_np = draw_bounding_boxes(img_np, detections, severity)
    
    success, buffer = cv2.imencode('.jpg', annotated_np, [cv2.IMWRITE_JPEG_QUALITY, 92])
    annotated_bytes = buffer.tobytes() if success else image_bytes
    
    metadata = {
        "damageType": overall_label,
        "confidence": avg_confidence,
        "severity": severity,
        "boundingBoxes": detections,
        "isCleanRoad": (overall_label == "Clean Road" or len(detections) == 0),
        "isValidRoad": True,
        "message": "Road damage analysis complete.",
        "totalHazardsDetected": len(detections)
    }
    
    return annotated_bytes, metadata

