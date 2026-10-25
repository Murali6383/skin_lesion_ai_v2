from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ============================================================
# PATH
# ============================================================

ROOT = Path(__file__).resolve().parent

DATASET_DIR = ROOT / "dataset" / "validator" / "non_skin"

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

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

centroid = checkpoint["centroid"].to(DEVICE)

old_threshold = checkpoint["threshold"]

image_size = checkpoint["image_size"]


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (image_size, image_size)
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

dataset = datasets.ImageFolder(
    root=DATASET_DIR.parent,
    transform=transform
)


loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

try:

    weights = models.ResNet18_Weights.DEFAULT

    base_model = models.resnet18(
        weights=weights
    )

except Exception:

    base_model = models.resnet18(
        weights=None
    )


model = nn.Sequential(
    *list(base_model.children())[:-1]
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


model = model.to(DEVICE)

model.eval()


# ============================================================
# CALCULATE DISTANCES
# ============================================================

distances = []


with torch.inference_mode():

    for images, _ in loader:

        images = images.to(DEVICE)

        features = model(images)

        features = features.flatten(
            start_dim=1
        )

        batch_distances = torch.norm(
            features - centroid,
            dim=1
        )

        distances.extend(
            batch_distances.cpu().tolist()
        )


# ============================================================
# SORT
# ============================================================

distances.sort()


tensor_distances = torch.tensor(
    distances
)


# ============================================================
# STATISTICS
# ============================================================

print()
print("==========================================")
print("NON-SKIN DISTANCE DISTRIBUTION")
print("==========================================")

print(
    f"Images: {len(distances)}"
)

print(
    f"Old threshold: {old_threshold:.4f}"
)

print(
    f"Minimum: {tensor_distances.min().item():.4f}"
)

print(
    f"Maximum: {tensor_distances.max().item():.4f}"
)

print(
    f"Mean: {tensor_distances.mean().item():.4f}"
)

print(
    f"Median: {tensor_distances.median().item():.4f}"
)

print(
    f"P90: {torch.quantile(tensor_distances, 0.90).item():.4f}"
)

print(
    f"P95: {torch.quantile(tensor_distances, 0.95).item():.4f}"
)

print(
    f"P97: {torch.quantile(tensor_distances, 0.97).item():.4f}"
)

print(
    f"P98: {torch.quantile(tensor_distances, 0.98).item():.4f}"
)

print(
    f"P99: {torch.quantile(tensor_distances, 0.99).item():.4f}"
)


# ============================================================
# SAMPLE DISTANCES
# ============================================================

print()
print("==========================================")
print("HIGHEST NON-SKIN DISTANCES")
print("==========================================")

for value in distances[-20:]:

    print(
        f"{value:.4f}"
    )


print()
print("==========================================")
print("DONE")
print("==========================================")