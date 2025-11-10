

import argparse
import cv2
import numpy as np
import imutils
from ultralytics import YOLO
import easyocr
import pytesseract
import re
import time
from pathlib import Path
from PIL import Image
import os
from pathlib import Path


DESKTOP = Path.home() / "Desktop"
DEFAULT_IMAGE_PATH = DESKTOP / "car.jpg"  


# ---------------------------
# Configuration
# ---------------------------
YOLO_MODEL = "yolov12s.pt"   
DETECTION_CONFIDENCE = 0.25
OCR_LANGS = ["en"]          
USE_PYTESSERACT_FALLBACK = False


PLATE_REGEX = r"[A-Z0-9\-]{4,12}"


# ---------------------------
# Helper functions
# ---------------------------
def load_detector(model_path=YOLO_MODEL):
    """Load YOLO model (Ultralytics)."""
    print(f"[INFO] Loading detector model: {model_path}")
    model = YOLO(model_path)
    return model


def detect_plates(model, image: np.ndarray, conf_thresh=DETECTION_CONFIDENCE):
    """
    Run detector on image and return list of bounding boxes (x1,y1,x2,y2) and confidences.
    Assumes model returns boxes in xyxy format.
    """
    # ultralytics YOLO model returns results with .boxes.xyxy numpy array
    results = model.predict(source=image, conf=conf_thresh, verbose=False)  # single image
    bboxes = []
    # model.predict returns list-like results (for each image)
    if len(results) == 0:
        return bboxes
    res = results[0]
    boxes = getattr(res, "boxes", None)
    if boxes is None:
        return bboxes
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2]
        conf = float(box.conf.cpu().numpy())
        cls = int(box.cls.cpu().numpy()) if hasattr(box, "cls") else None
        # NOTE: If using a general model, classes may vary. If you trained a plate-class-only model, fine.
        bboxes.append((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), conf))
    return bboxes


def preprocess_plate(crop: np.ndarray):
    """Preprocess cropped plate image for OCR: grayscale, adaptive thresholding, resize, denoise."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = max(1, 400 // max(h, w))
    if scale > 1:
        gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
    return th


def ocr_easyocr(reader, image):
    """Run EasyOCR on image and return best text candidate."""
    result = reader.readtext(image, detail=0) 
    if not result:
        return ""

    combined = " ".join(result).upper()
    m = re.search(PLATE_REGEX, combined)
    if m:
        return m.group(0)
    tokens = sorted(result, key=lambda s: len(s), reverse=True)
    return tokens[0].upper() if tokens else combined.upper()


def ocr_tesseract(image):
    """Fallback OCR via pytesseract (image should be grayscale or PIL)."""
    if isinstance(image, np.ndarray):
        pil = Image.fromarray(image)
    else:
        pil = image
    text = pytesseract.image_to_string(pil, config="--psm 7")
    text = text.strip().upper()
    m = re.search(PLATE_REGEX, text)
    return m.group(0) if m else text


def postprocess_text(text: str):
    """Clean OCR text: remove unwanted chars and keep alnum + dash."""
    if not text:
        return ""
    cleaned = re.sub(r"[^A-Z0-9\-]", "", text.upper())
    return cleaned


# ---------------------------
# Main detection + OCR pipeline
# ---------------------------
def process_image_file(model, reader, img_path: str, visualize=True):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not open image {img_path}")
    return process_frame(model, reader, img, visualize=visualize)


def process_frame(model, reader, frame: np.ndarray, visualize=True):
    """
    Detect plates in frame, OCR them, and optionally annotate and return annotated frame and results.
    Returns:
      annotated_frame, results_list
    where results_list is list of dicts: {'bbox':(x1,y1,x2,y2), 'conf':float, 'text':str}
    """
    orig = frame.copy()
    # resize for faster detection (optional)
    frame_small = imutils.resize(frame, width=1280)
    bboxes = detect_plates(model, frame_small)
    results = []
    for (x1, y1, x2, y2, conf) in bboxes:
        # crop with padding
        pad = int(0.03 * (x2 - x1 + y2 - y1) / 2)
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(frame_small.shape[1] - 1, x2 + pad)
        y2p = min(frame_small.shape[0] - 1, y2 + pad)
        crop = frame_small[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            continue
        prep = preprocess_plate(crop)
        # Try easyocr
        text = ocr_easyocr(reader, prep)
        if (not text or len(text) < 2) and USE_PYTESSERACT_FALLBACK:
            text = ocr_tesseract(prep)
        text = postprocess_text(text)
        results.append({"bbox": (x1p, y1p, x2p, y2p), "conf": conf, "text": text})
        # annotate
        if visualize:
            cv2.rectangle(frame_small, (x1p, y1p), (x2p, y2p), (0, 255, 0), 2)
            label = f"{text} {conf:.2f}"
            cv2.putText(frame_small, label, (x1p, max(0, y1p-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return frame_small, results


# ---------------------------
# CLI and video loop
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(DEFAULT_IMAGE_PATH), help="Path to input image")
    ap.add_argument("--video", help="Path to input video file")
    ap.add_argument("--webcam", type=int, help="Webcam device index (0 etc.)")
    ap.add_argument("--weights", default=YOLO_MODEL, help="YOLO weights path")
    args = ap.parse_args()

    model = load_detector(args.weights)
    reader = easyocr.Reader(OCR_LANGS, gpu=False) 

    if args.image:
        out_img, results = process_image_file(model, reader, args.image, visualize=True)
        print("[RESULTS]")
        for r in results:
            print(r)
        cv2.imshow("Plates", out_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        out_path = Path(args.image).with_name(Path(args.image).stem + "_annotated.jpg")
        cv2.imwrite(str(out_path), out_img)
        print(f"Annotated image saved to {out_path}")

    elif args.video or args.webcam is not None:
        if args.video:
            cap = cv2.VideoCapture(args.video)
        else:
            cap = cv2.VideoCapture(args.webcam)
        if not cap.isOpened():
            print("[ERROR] Could not open video source")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        print("[INFO] Starting video loop. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated, results = process_frame(model, reader, frame, visualize=True)
            if results:
                for r in results:
                    print(f"Detected: {r['text']}, conf={r['conf']:.2f}, bbox={r['bbox']}")
            cv2.imshow("Plate Reader", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Provide --image or --video or --webcam. See --help.")


if __name__ == "__main__":
    main()
