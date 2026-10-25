from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models


# ============================================================
# PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT
    / "models_saved"
    / "normal_skin_detector.pth"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# MODEL
# ============================================================

def create_feature_extractor():

    weights = models.ResNet18_Weights.DEFAULT

    model = models.resnet18(weights=weights)

    model.fc = nn.Identity()

    model.eval()

    for parameter in model.parameters():

        parameter.requires_grad = False

    return model.to(DEVICE)


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Normal skin detector model not found:\n"
        f"{MODEL_PATH}\n\n"
        f"Run:\n"
        f"python ai\\normal_skin_detector"
        f"\\train_normal_detector.py"
    )


checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)


NORMAL_FEATURES = checkpoint[
    "normal_features"
].float()


THRESHOLD = float(
    checkpoint["threshold"]
)


FEATURE_MODEL = create_feature_extractor()


print("=" * 60)
print("NORMAL SKIN DETECTOR")
print("=" * 60)

print(
    f"Device             : {DEVICE}"
)

print(
    f"Normal database    : "
    f"{len(NORMAL_FEATURES)} vectors"
)

print(
    f"Threshold          : "
    f"{THRESHOLD:.6f}"
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_feature(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        feature = FEATURE_MODEL(
            tensor
        )

    feature = torch.nn.functional.normalize(
        feature,
        p=2,
        dim=1
    )

    return feature.cpu()


# ============================================================
# NORMAL SKIN DETECTION
# ============================================================

def detect_normal_skin(image_path):

    query_feature = extract_feature(
        image_path
    )

    distances = torch.cdist(
        query_feature,
        NORMAL_FEATURES,
        p=2
    )

    min_distance, nearest_index = torch.min(
        distances,
        dim=1
    )

    distance = float(
        min_distance.item()
    )

    nearest_index = int(
        nearest_index.item()
    )

    # ========================================================
    # DECISION
    # ========================================================

    is_normal = (
        distance <= THRESHOLD
    )

    # ========================================================
    # SIMPLE RELATIVE SCORE
    # ========================================================
    #
    # This is NOT a medical probability.
    #

    normality_score = max(
        0.0,
        min(
            1.0,
            1.0 - (
                distance / THRESHOLD
            )
        )
    )

    normality_percent = (
        normality_score * 100
    )

    # ========================================================
    # RESULT
    # ========================================================

    if is_normal:

        result = {

            "is_normal": True,

            "class":
                "normal_skin",

            "decision":
                "NORMAL",

            "message":
                "No clear abnormality was detected "
                "by the research model.",

            "distance":
                round(distance, 6),

            "threshold":
                round(THRESHOLD, 6),

            "normality_percent":
                round(normality_percent, 2),

            "nearest_normal_index":
                nearest_index,

            "detector_type":
                "normal_skin_nearest_neighbour"
        }

    else:

        result = {

            "is_normal": False,

            "class":
                "not_normal_like",

            "decision":
                "CONTINUE_TO_DISEASE_MODEL",

            "message":
                "Image is not sufficiently similar "
                "to the normal-skin reference dataset.",

            "distance":
                round(distance, 6),

            "threshold":
                round(THRESHOLD, 6),

            "normality_percent":
                round(normality_percent, 2),

            "nearest_normal_index":
                nearest_index,

            "detector_type":
                "normal_skin_nearest_neighbour"
        }

    return result