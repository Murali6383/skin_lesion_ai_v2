from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import numpy as np


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

# IMPORTANT:
# Your non-skin images are inside:
# dataset/validator
NON_SKIN_DIR = ROOT / "dataset" / "validator"

# Saved validator model
MODEL_PATH = ROOT / "models_saved" / "skin_validator.pth"


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# START
# ============================================================

print("=" * 60)
print("NON-SKIN NEAREST-NEIGHBOUR VALIDATOR TRAINING")
print("=" * 60)

print(f"Device       : {DEVICE}")
print(f"Dataset      : {NON_SKIN_DIR}")
print(f"Output model : {MODEL_PATH}")


# ============================================================
# CHECK DATASET
# ============================================================

if not NON_SKIN_DIR.exists():

    raise RuntimeError(
        f"\nDataset folder not found:\n"
        f"{NON_SKIN_DIR}\n\n"
        f"Expected folder:\n"
        f"{ROOT / 'dataset' / 'validator'}"
    )


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
# DATASET
# ============================================================

class NonSkinDataset(Dataset):

    def __init__(self, folder):

        self.folder = Path(folder)

        extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        }

        self.images = [
            p
            for p in self.folder.rglob("*")
            if p.is_file()
            and p.suffix.lower() in extensions
        ]

        if len(self.images) == 0:

            raise RuntimeError(
                f"\nNo images found in:\n"
                f"{self.folder}\n\n"
                f"Supported formats:\n"
                f".jpg .jpeg .png .bmp .webp"
            )

        print(
            f"\nFound {len(self.images)} "
            f"non-skin images"
        )

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        image_path = self.images[index]

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            image = transform(image)

            return image, str(image_path)

        except Exception as e:

            print(
                f"\nWARNING: Could not read:"
                f"\n{image_path}"
                f"\nError: {e}"
            )

            next_index = (
                index + 1
            ) % len(self.images)

            return self.__getitem__(
                next_index
            )


# ============================================================
# LOAD DATASET
# ============================================================

dataset = NonSkinDataset(
    NON_SKIN_DIR
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LOAD RESNET18
# ============================================================

print("\nLoading ResNet18...")

weights = models.ResNet18_Weights.DEFAULT

base_model = models.resnet18(
    weights=weights
)


# ============================================================
# REMOVE CLASSIFICATION HEAD
# ============================================================

feature_extractor = nn.Sequential(
    *list(
        base_model.children()
    )[:-1]
)

feature_extractor = (
    feature_extractor.to(DEVICE)
)

feature_extractor.eval()


# Freeze model

for parameter in (
    feature_extractor.parameters()
):

    parameter.requires_grad = False


print(
    "ResNet18 feature extractor ready"
)


# ============================================================
# EXTRACT NON-SKIN FEATURES
# ============================================================

all_features = []

print(
    "\nExtracting non-skin feature vectors..."
)

with torch.no_grad():

    for batch_index, (
        images,
        paths
    ) in enumerate(loader):

        images = images.to(DEVICE)

        features = feature_extractor(
            images
        )

        # Shape:
        # [batch, 512, 1, 1]

        features = features.view(
            features.size(0),
            -1
        )

        # Normalize feature vectors

        features = torch.nn.functional.normalize(
            features,
            p=2,
            dim=1
        )

        all_features.append(
            features.cpu()
        )

        print(
            f"Processed batch "
            f"{batch_index + 1}/"
            f"{len(loader)}"
        )


# ============================================================
# COMBINE ALL FEATURES
# ============================================================

features = torch.cat(
    all_features,
    dim=0
)


print(
    f"\nFeature matrix shape: "
    f"{features.shape}"
)

print(
    f"Number of non-skin vectors: "
    f"{len(features)}"
)


# ============================================================
# CALCULATE NEAREST-NEIGHBOUR DISTANCE
# ============================================================

print(
    "\nCalculating nearest-neighbour distances..."
)


# Compare every non-skin image with
# every other non-skin image.

distance_matrix = torch.cdist(
    features,
    features,
    p=2
)


# Ignore self-distance.

distance_matrix.fill_diagonal_(
    float("inf")
)


# Find closest other non-skin image.

nearest_distances = torch.min(
    distance_matrix,
    dim=1
).values


nearest_distances_np = (
    nearest_distances.numpy()
)


# ============================================================
# DISTANCE STATISTICS
# ============================================================

print("\nDistance statistics:")

print(
    f"Minimum : "
    f"{nearest_distances_np.min():.6f}"
)

print(
    f"Maximum : "
    f"{nearest_distances_np.max():.6f}"
)

print(
    f"Mean    : "
    f"{nearest_distances_np.mean():.6f}"
)

print(
    f"Median  : "
    f"{np.median(nearest_distances_np):.6f}"
)

print(
    f"P90     : "
    f"{np.percentile(nearest_distances_np, 90):.6f}"
)

print(
    f"P95     : "
    f"{np.percentile(nearest_distances_np, 95):.6f}"
)

print(
    f"P99     : "
    f"{np.percentile(nearest_distances_np, 99):.6f}"
)


# ============================================================
# THRESHOLD
# ============================================================

# 99th percentile:
# almost all known non-skin examples
# should remain inside this boundary.

threshold = float(
    np.percentile(
        nearest_distances_np,
        99
    )
)


# Small safety margin

threshold = threshold * 1.10


print(
    f"\nFinal threshold: "
    f"{threshold:.6f}"
)


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


checkpoint = {

    # ResNet feature extractor
    "model_state_dict":
        feature_extractor.state_dict(),

    # ALL non-skin feature vectors
    "non_skin_features":
        features,

    # Nearest-neighbour threshold
    "threshold":
        threshold,

    # Image size
    "image_size":
        IMAGE_SIZE,

    # Model name
    "model_name":
        "resnet18_non_skin_nearest_neighbour",

    # Feature dimension
    "feature_dimension":
        int(features.shape[1]),

    # Number of training images
    "num_non_skin_images":
        int(features.shape[0])
}


torch.save(
    checkpoint,
    MODEL_PATH
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"\nModel saved:"
    f"\n{MODEL_PATH}"
)

print(
    f"\nNon-skin images:"
    f" {features.shape[0]}"
)

print(
    f"Feature dimension:"
    f" {features.shape[1]}"
)

print(
    f"Threshold:"
    f" {threshold:.6f}"
)

print("\nValidator logic:")

print(
    "\nKnown non-skin-like image"
    "\n        ↓"
    "\nNearest distance <= threshold"
    "\n        ↓"
    "\nREJECT"
)

print(
    "\nUnknown / sufficiently different image"
    "\n        ↓"
    "\nNearest distance > threshold"
    "\n        ↓"
    "\nPASS → Disease Model"
)

print("\n" + "=" * 60)