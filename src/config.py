"""Project-wide constants. Class names match the brief exactly (25 classes, no 'train')."""

from pathlib import Path

# ---------------------------------------------------------------------------
# 25 classes from the SmartVision AI brief (order is the YOLO / softmax index)
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "cow",
    "elephant",
    "bottle",
    "cup",
    "bowl",
    "pizza",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
]

NUM_CLASSES = len(CLASS_NAMES)
assert NUM_CLASSES == 25, f"Expected 25 classes, got {NUM_CLASSES}"
assert "train" not in CLASS_NAMES, "Class 'train' is not in the required 25-class subset"

NAME_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_NAME = {i: name for i, name in enumerate(CLASS_NAMES)}

# Hugging Face source
HF_DATASET = "detection-datasets/coco"
HF_SPLIT = "train"

IMAGES_PER_CLASS = 200
COLLECT_BUFFER = 240  # extra so the quality crop filter can still leave 200
RANDOM_SEED = 42

# Splits
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Classification
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Detection / YOLO
YOLO_IMAGE_SIZE = 640
DEFAULT_CONFIDENCE = 0.50
DEFAULT_IOU = 0.45

# Paths — resolve relative to the project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "smartvision_dataset"
CLASSIFICATION_DIR = DATASET_DIR / "classification"
DETECTION_DIR = DATASET_DIR / "detection"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"
METADATA_PATH = DATASET_DIR / "dataset_metadata.json"

CLASSIFICATION_MODEL_FILES = {
    "VGG16": "vgg16.keras",
    "ResNet50": "resnet50.keras",
    "MobileNetV2": "mobilenetv2.keras",
    "EfficientNetB0": "efficientnetb0.keras",
}
YOLO_WEIGHTS = "yolov8_best.pt"

# Training defaults (Colab T4-friendly)
CLASSIFICATION_EPOCHS = 25
CLASSIFICATION_BATCH = 32
YOLO_EPOCHS = 50
YOLO_BATCH = 16
YOLO_MODEL = "yolov8m.pt"  # freeze-10 fine-tune used for the submitted detector


def classification_split_dir(split: str) -> Path:
    return CLASSIFICATION_DIR / split


def model_path(name: str) -> Path:
    return MODELS_DIR / CLASSIFICATION_MODEL_FILES[name]


def yolo_path() -> Path:
    return MODELS_DIR / YOLO_WEIGHTS
