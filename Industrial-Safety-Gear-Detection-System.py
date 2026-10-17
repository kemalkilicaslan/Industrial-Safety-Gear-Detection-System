# Industrial Safety Gear Detection System

# Import the necessary libraries
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import config

# Model and class configuration (loaded from config.py)
MODEL_PATH = config.MODEL_PATH
EQUIPMENT_CLASSES = config.EQUIPMENT_CLASSES
PERSON_CLASS = config.PERSON_CLASS

# Detection thresholds (loaded from config.py)
CONF_THRESH = config.CONF_THRESH
IOU_THRESH  = config.IOU_THRESH
MAX_LOST    = config.MAX_LOST
DIST_THRESH = config.DIST_THRESH

# Colors (BGR) (loaded from config.py)
COLOR_GREEN  = config.COLOR_GREEN
COLOR_RED    = config.COLOR_RED
COLOR_ORANGE = config.COLOR_ORANGE

# Fonts
FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


# Function to calculate Intersection over Union between two bounding boxes
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)


# Function to check if the center of a small box lies inside a large box
def box_center_in(box_small, box_large, margin=0.0):
    cx = (box_small[0] + box_small[2]) / 2
    cy = (box_small[1] + box_small[3]) / 2
    x1, y1, x2, y2 = box_large
    w = x2 - x1; h = y2 - y1
    return (x1 - margin * w <= cx <= x2 + margin * w and
            y1 - margin * h <= cy <= y2 + margin * h)


# Function to assign detected equipment to the closest person
def assign_equipment_to_persons(persons, equipments):
    assignment = defaultdict(set)
    for eq_cls, eq_box, eq_conf in equipments:
        best_idx, best_score = -1, -1.0
        for i, (_, p_box, _) in enumerate(persons):
            if box_center_in(eq_box, p_box, margin=0.05):
                score = iou(eq_box, p_box) + 1.0
            else:
                score = iou(eq_box, p_box)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx >= 0 and best_score >= IOU_THRESH:
            assignment[best_idx].add(eq_cls)
    return assignment


# Function to determine compliance color and present/missing equipment sets
def compliance_status(eq_set):
    present = EQUIPMENT_CLASSES & eq_set
    missing = EQUIPMENT_CLASSES - eq_set
    if not missing:
        return COLOR_GREEN, present, missing
    elif present:
        return COLOR_ORANGE, present, missing
    else:
        return COLOR_RED, present, missing


# Function to draw bounding box, ID, equipment labels and warning on a person
def draw_person_overlay(frame, person_box, eq_set, track_id, conf):
    x1, y1, x2, y2 = [int(v) for v in person_box]
    box_color, present, _ = compliance_status(eq_set)

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # Draw equipment labels inside the bounding box
    label_y = y1 + 18
    for eq in sorted(EQUIPMENT_CLASSES):
        if label_y > y2 - 4:
            break
        col = COLOR_GREEN if eq in present else COLOR_RED
        (tw2, th2), _ = cv2.getTextSize(eq, FONT, 0.38, 1)
        cv2.rectangle(frame, (x1 + 2, label_y - th2 - 2), (x1 + tw2 + 10, label_y + 3),
                      (20, 20, 20), -1)
        cv2.putText(frame, eq, (x1 + 5, label_y),
                    FONT, 0.38, col, 1, cv2.LINE_AA)
        label_y += 17

    # Draw warning banner if no equipment is detected
    if not present:
        warn_text = "EQUIPMENT: NONE"
        (tw3, th3), _ = cv2.getTextSize(warn_text, FONT_BOLD, 0.5, 1)
        wx = x1 + (x2 - x1 - tw3) // 2
        wy = y2 - 10
        cv2.rectangle(frame, (wx - 4, wy - th3 - 4), (wx + tw3 + 4, wy + 4), (0, 0, 180), -1)
        cv2.putText(frame, warn_text, (wx, wy),
                    FONT_BOLD, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# Function to get the centroid of a bounding box
def get_centroid(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


# Function to calculate Euclidean distance between two centroids
def centroid_distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


# Function to update tracker and return {det_index: track_id} mapping
def update_tracker(detections):
    global track_states, next_track_id

    if not detections:
        for tid in list(track_states):
            track_states[tid]["lost"] += 1
            if track_states[tid]["lost"] > MAX_LOST:
                del track_states[tid]
        return {}

    det_centroids = [get_centroid(d[1]) for d in detections]
    track_ids = list(track_states.keys())

    assignment = {}
    used_tracks = set()

    if track_ids:
        # Build cost matrix using centroid distance weighted by IoU bonus
        cost = np.full((len(detections), len(track_ids)), 1e9)
        for di, dc in enumerate(det_centroids):
            for ti, tid in enumerate(track_ids):
                dist  = centroid_distance(dc, track_states[tid]["centroid"])
                iou_v = iou(detections[di][1], track_states[tid]["box"])
                cost[di, ti] = dist * (1.0 - 0.5 * iou_v)

        # Greedy nearest-neighbour matching — skip pairs that exceed distance threshold
        for idx in np.argsort(cost, axis=None):
            di, ti = divmod(int(idx), len(track_ids))
            if di in assignment or track_ids[ti] in used_tracks:
                continue
            if cost[di, ti] > DIST_THRESH:
                continue
            assignment[di] = track_ids[ti]
            used_tracks.add(track_ids[ti])

    # Open new tracks for unmatched detections
    for di in range(len(detections)):
        if di not in assignment:
            assignment[di] = next_track_id
            next_track_id += 1

    # Age out unmatched existing tracks
    for tid in list(track_states):
        if tid not in set(assignment.values()):
            track_states[tid]["lost"] += 1
            if track_states[tid]["lost"] > MAX_LOST:
                del track_states[tid]

    # Update matched tracks with new positions
    for di, tid in assignment.items():
        track_states[tid] = {
            "box":      detections[di][1],
            "centroid": det_centroids[di],
            "lost":     0,
        }

    return assignment


# Centroid-based person tracker state
track_states  = {}  # track_id -> {"box": xyxy, "centroid": (cx, cy), "lost": int}
next_track_id = 1

# Load YOLO model
model = YOLO(MODEL_PATH)
names = model.names

# Open video (paths loaded from config.py)
video_capture = cv2.VideoCapture(config.INPUT_VIDEO)  # write the name of the video file here
output_file = config.OUTPUT_VIDEO  # write the name and extension of the video file to be recorded

# Video features
frame_width  = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps          = int(video_capture.get(cv2.CAP_PROP_FPS))

# Create the VideoWriter object to save the video file
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_video = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = video_capture.read()  # read video frame
    if not ret:
        break

    # Run YOLO inference
    results = model(frame, conf=CONF_THRESH, verbose=False)[0]

    # Separate person and equipment detections
    persons    = []
    equipments = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_nm = names[cls_id]
        conf_v = float(box.conf[0])
        xyxy   = box.xyxy[0].cpu().numpy().astype(int).tolist()
        if cls_nm == PERSON_CLASS:
            persons.append((cls_nm, xyxy, conf_v))
        elif cls_nm in EQUIPMENT_CLASSES:
            equipments.append((cls_nm, xyxy, conf_v))

    # Assign detected equipment to their corresponding person
    assignment = assign_equipment_to_persons(persons, equipments)

    # Update tracker to obtain persistent IDs for each person
    track_map = update_tracker(persons)

    # Draw overlay for each detected person
    for i, (_, p_box, p_conf) in enumerate(persons):
        eq_set   = assignment[i]
        track_id = track_map.get(i, i + 1)
        draw_person_overlay(frame, p_box, eq_set, track_id, p_conf)

    # Write the drawn frame to the video file to be saved
    output_video.write(frame)

    # Show video with industrial safety gear detection
    cv2.imshow("Industrial Safety Gear Detection", frame)

    # Switch off video when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release all open windows
video_capture.release()
output_video.release()
cv2.destroyAllWindows()