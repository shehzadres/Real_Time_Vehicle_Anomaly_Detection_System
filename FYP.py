"""
FYP.py
State-based traffic anomaly detection pipeline
Supports SPEEDING, STALLED, WRONG WAY, LANE VIOL
Optimized for ~1 minute videos
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
import supervision as sv

# --------------------------------------------------
# INITIALIZATION
# --------------------------------------------------
def initialize_pipeline(config: dict):
    # ---------------- CONFIG ----------------
    fps = config.get("fps", 25)

    vehicle_model_path = config.get("vehicle_model_path", "yolo11s.pt")
    vehicle_classes = config.get("vehicle_classes", [2, 3, 5, 7])

    road_length_m = config.get("road_length_m", 20.0)
    road_width_m = config.get("road_width_m", 10.0)
    lane_divisions = config.get("lane_divisions", 3)

    speed_limit_kmh = config.get("speed_limit_kmh", 30)
    smooth_window = config.get("speed_smoothing_window", 10)
    consec_speeding_frames = config.get("speeding_consec_frames", 6)

    stall_seconds = config.get("stall_seconds", 3)
    stall_speed_thresh = config.get("stall_speed_thresh", 0.5)

    lane_violation_seconds = config.get("lane_violation_seconds", 0.5)
    lane_direction = config.get(
        "lane_direction",
        {lane: +1 for lane in range(lane_divisions)}
    )

    homography_src = config.get("homography_src", None)

    # ---------------- MODELS ----------------
    vehicle_model = YOLO(vehicle_model_path)

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(thickness=1)
    label_annotator = sv.LabelAnnotator(text_scale=0.3)
    trace_annotator = sv.TraceAnnotator(thickness=1)

    # ---------------- STATE ----------------
    state = {
        # Models & tools
        "vehicle_model": vehicle_model,
        "vehicle_classes": vehicle_classes,
        "tracker": tracker,
        "box_annotator": box_annotator,
        "label_annotator": label_annotator,
        "trace_annotator": trace_annotator,

        # Geometry
        "homography": None,
        "homography_src": homography_src,
        "road_length": road_length_m,
        "road_width": road_width_m,
        "lane_divisions": lane_divisions,

        # Timing
        "fps": fps,
        "frame_idx": 0,

        # Motion
        "prev_positions": {},

        # Speed logic
        "speed_limit": speed_limit_kmh,
        "speed_history": defaultdict(lambda: deque(maxlen=smooth_window)),
        "consec_speeding": defaultdict(int),
        "consec_speeding_frames": consec_speeding_frames,

        # Stall logic
        "stall_frames": int(stall_seconds * fps),
        "stall_speed_thresh": stall_speed_thresh,
        "stalled_counter": defaultdict(int),

        # Lane logic
        "vehicle_lane": {},
        "lane_change_counter": defaultdict(int),
        "lane_violation_frames": int(lane_violation_seconds * fps),

        # Direction
        "lane_direction": lane_direction
    }

    return state



# --------------------------------------------------
# FRAME PROCESSING
# --------------------------------------------------
def process_frame(frame, state):
    model = state["vehicle_model"]
    tracker = state["tracker"]

    # -------- Homography init (once) --------
    if state["homography"] is None:
        h, w = frame.shape[:2]
        src = np.array([[100, h-50], [w-100, h-50],
                        [w-100, 50], [100, 50]], dtype=np.float32)
        dst = np.array([[0, 0],
                        [state["road_width"], 0],
                        [state["road_width"], state["road_length"]],
                        [0, state["road_length"]]], dtype=np.float32)
        state["homography"] = cv2.getPerspectiveTransform(src, dst)

    # -------- Detection & Tracking --------
    results = model.track(frame, persist=True, classes=[2,3,5,7], verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    anomalies = []
    labels = []
    curr_positions = {}

    lane_width = state["road_width"] / state["lane_divisions"]

    if detections.tracker_id is not None:
        for i, tid in enumerate(detections.tracker_id):
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            cx, cy = int((x1+x2)/2), int(y2)

            pt = np.array([[[cx, cy]]], dtype=np.float32)
            world = cv2.perspectiveTransform(pt, state["homography"])[0][0]
            curr_positions[tid] = world

            # ---------------- SPEED ----------------
            speed = 0.0
            if tid in state["prev_positions"]:
                dist = np.linalg.norm(world - state["prev_positions"][tid])
                speed = dist * state["fps"] * 3.6

            state["speed_history"][tid].append(speed)
            avg_speed = float(np.mean(state["speed_history"][tid]))

            if avg_speed > state["speed_limit"]:
                state["consec_speeding"][tid] += 1
            else:
                state["consec_speeding"][tid] = 0

            is_speeding = state["consec_speeding"][tid] >= state["consec_speeding_frames"]

            # ---------------- STALLED ----------------
            if avg_speed < 0.5:
                state["stalled_counter"][tid] += 1
            else:
                state["stalled_counter"][tid] = 0

            is_stalled = state["stalled_counter"][tid] >= state["stall_frames"]

            # ---------------- LANE ----------------
            lane_num = int(world[0] // lane_width)
            if lane_num < 0 or lane_num >= state["lane_divisions"]:
                lane_num = None

            prev_lane = state["vehicle_lane"].get(tid)
            is_lane_violation = False

            if prev_lane is None:
                state["vehicle_lane"][tid] = lane_num
            else:
                if lane_num != prev_lane and lane_num is not None:
                    state["lane_change_counter"][tid] += 1
                else:
                    state["lane_change_counter"][tid] = 0

                if state["lane_change_counter"][tid] >= state["lane_violation_frames"]:
                    is_lane_violation = True
                    state["vehicle_lane"][tid] = lane_num
                    state["lane_change_counter"][tid] = 0

            # ---------------- WRONG WAY ----------------
            is_wrong_way = False
            if tid in state["prev_positions"] and lane_num is not None:
                dy = world[1] - state["prev_positions"][tid][1]
                correct_dir = state["lane_direction"].get(lane_num, +1)
                if dy * correct_dir < 0:
                    is_wrong_way = True

            # ---------------- LABEL ----------------
            label = f"#{tid} {avg_speed:.1f} km/h"
            if is_speeding: label += " [SPEEDING]"
            if is_stalled: label += " [STALLED]"
            if is_wrong_way: label += " [WRONG WAY]"
            if is_lane_violation: label += " [LANE VIOL]"
            labels.append(label)

            # ---------------- LOG ANOMALIES ----------------
            def log(event):
                anomalies.append({
                    "timestamp_s": round(state["frame_idx"] / state["fps"], 2),
                    "frame_idx": state["frame_idx"],
                    "tracker_id": tid,
                    "event": event,
                    "speed_kmh": round(avg_speed, 1),
                    "plate_text": "",
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2
                })

            if is_speeding: log("SPEEDING")
            if is_stalled: log("STALLED")
            if is_wrong_way: log("WRONG WAY")
            if is_lane_violation: log("LANE VIOL")

    # -------- Update state --------
    state["prev_positions"] = curr_positions
    state["frame_idx"] += 1

    # -------- Annotate --------
    frame = state["trace_annotator"].annotate(frame, detections)
    frame = state["box_annotator"].annotate(frame, detections)
    frame = state["label_annotator"].annotate(frame, detections, labels)

    return frame, anomalies, state
