# Industrial Safety Gear Detection System — Configuration

# Model and class configuration
MODEL_PATH = "industrial-safety-gear-detection.pt"
EQUIPMENT_CLASSES = {"helmet", "safety-vest", "gloves", "face-mask"}
PERSON_CLASS = "person"

# Detection and tracking constraints
CONF_THRESH = 0.15   # detection confidence threshold
IOU_THRESH  = 0.15   # equipment-to-person assignment IoU threshold
MAX_LOST    = 30     # frames a track may survive undetected before being dropped
DIST_THRESH = 80.0   # maximum centroid distance for track matching

# Colors (BGR)
COLOR_GREEN  = (50, 220, 50)
COLOR_RED    = (40, 40, 220)
COLOR_ORANGE = (30, 165, 255)

# Video files
INPUT_VIDEO  = "Industrial-Safety-Gear.mp4"            # input video file name
OUTPUT_VIDEO = "Industrial-Safety-Gear-Detection.mp4"  # output video file name
