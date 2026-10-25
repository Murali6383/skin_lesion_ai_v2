from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models, transforms

from PIL import Image


# ============================================================
# PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT
    / "models_saved"
    / "skin_validator.pth"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Validator model not found:\n{MODEL_PATH}"
    )


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = checkpoint.get(
    "image_size",
    224
)

THRESHOLD = float(
    checkpoint["threshold"]
)

NON_SKIN_FEATURES = checkpoint[
    "non_skin_features"
].float().to(DEVICE)


# ============================================================
# RESNET18 FEATURE EXTRACTOR
# ============================================================

weights = models.ResNet18_Weights.DEFAULT

base_model = models.resnet18(
    weights=weights
)

feature_extractor = nn.Sequential(
    *list(base_model.children())[:-1]
)

feature_extractor = feature_extractor.to(
    DEVICE
)

feature_extractor.load_state_dict(
    checkpoint["model_state_dict"]
)

feature_extractor.eval()


# Freeze model

for parameter in feature_extractor.parameters():
    parameter.requires_grad = False


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# STARTUP LOG
# ============================================================

print("=" * 60)
print("NON-SKIN NEAREST-NEIGHBOUR VALIDATOR")
print("=" * 60)

print(
    f"Model       : {MODEL_PATH}"
)

print(
    f"Device      : {DEVICE}"
)

print(
    f"Threshold   : {THRESHOLD:.6f}"
)

print(
    f"Non-skin DB : {len(NON_SKIN_FEATURES)} vectors"
)

print("=" * 60)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(image_path: str):

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        image
    ).unsqueeze(0)

    image_tensor = image_tensor.to(
        DEVICE
    )

    with torch.no_grad():

        features = feature_extractor(
            image_tensor
        )

        features = features.view(
            features.size(0),
            -1
        )

        features = F.normalize(
            features,
            p=2,
            dim=1
        )

    return features


# ============================================================
# VALIDATE IMAGE
# ============================================================

def validate_skin_image(
    image_path: str
):

    try:

        # ----------------------------------------------------
        # Extract image features
        # ----------------------------------------------------

        query_features = extract_features(
            image_path
        )


        # ----------------------------------------------------
        # Compare with ALL non-skin vectors
        # ----------------------------------------------------

        distances = torch.cdist(
            query_features,
            NON_SKIN_FEATURES,
            p=2
        )


        # Closest non-skin distance

        min_distance = (
            distances.min().item()
        )


        # Which non-skin image was closest

        nearest_index = (
            distances.argmin().item()
        )


        # ----------------------------------------------------
        # DECISION
        # ----------------------------------------------------

        if min_distance <= THRESHOLD:

            result_class = "non_skin"

            is_skin = False

            decision = "REJECT"

            message = (
                "Image is similar to "
                "known non-skin images."
            )

        else:

            result_class = "possible_skin"

            is_skin = True

            decision = "PASS"

            message = (
                "Image passed the non-skin "
                "validation gate."
            )


        # ----------------------------------------------------
        # DISTANCE SCORE
        # ----------------------------------------------------

        if min_distance <= THRESHOLD:

            score = (
                1.0
                - (
                    min_distance
                    / THRESHOLD
                )
            )

            score = max(
                0.0,
                min(1.0, score)
            )

        else:

            score = (
                min_distance
                - THRESHOLD
            ) / max(
                THRESHOLD,
                1e-8
            )

            score = min(
                1.0,
                score
            )


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        result = {

            "valid_image": True,

            "is_skin": is_skin,

            "class": result_class,

            "decision": decision,

            "message": message,

            "distance": round(
                min_distance,
                6
            ),

            "threshold": round(
                THRESHOLD,
                6
            ),

            "confidence": round(
                score,
                4
            ),

            "nearest_non_skin_index":
                nearest_index,

            "validator_type":
                "non_skin_nearest_neighbour"
        }


        # ----------------------------------------------------
        # LOG
        # ----------------------------------------------------

        print("\nValidator Result")

        print(
            f"Image     : "
            f"{Path(image_path).name}"
        )

        print(
            f"Distance  : "
            f"{min_distance:.6f}"
        )

        print(
            f"Threshold : "
            f"{THRESHOLD:.6f}"
        )

        print(
            f"Decision  : "
            f"{decision}"
        )

        print(
            f"Class     : "
            f"{result_class}"
        )

        return result


    except Exception as e:

        print(
            f"\nValidator error: {e}"
        )

        return {

            "valid_image": False,

            "is_skin": False,

            "class": "error",

            "decision": "ERROR",

            "message":
                "Unable to validate image.",

            "error": str(e)
        }