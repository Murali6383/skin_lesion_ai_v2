from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = ROOT / "dataset" / "normal_skin"
MODEL_PATH = ROOT / "models_saved" / "normal_skin_detector.pth"


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGE TRANSFORM
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
# RESNET18 FEATURE EXTRACTOR
# ============================================================

def create_feature_extractor():

    weights = models.ResNet18_Weights.DEFAULT

    model = models.resnet18(weights=weights)

    # Remove final classification layer
    model.fc = nn.Identity()

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model.to(DEVICE)


# ============================================================
# LOAD IMAGE
# ============================================================

def extract_feature(model, image_path):

    image = Image.open(image_path).convert("RGB")

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        feature = model(tensor)

    # Normalize feature vector
    feature = torch.nn.functional.normalize(
        feature,
        p=2,
        dim=1
    )

    return feature.cpu()


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    print("=" * 60)
    print("NORMAL SKIN DETECTOR TRAINING")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"Dataset folder not found:\n{DATASET_DIR}"
        )

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    image_paths = [
        path
        for path in DATASET_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in image_extensions
    ]

    if len(image_paths) == 0:

        raise RuntimeError(
            "No normal skin images found."
        )

    print(
        f"Found {len(image_paths)} normal skin images"
    )

    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = create_feature_extractor()

    # ========================================================
    # EXTRACT FEATURES
    # ========================================================

    features = []

    valid_images = []

    print("\nExtracting features...\n")

    for index, image_path in enumerate(image_paths):

        try:

            feature = extract_feature(
                model,
                image_path
            )

            features.append(feature)

            valid_images.append(
                str(image_path.relative_to(ROOT))
            )

            print(
                f"[{index + 1}/{len(image_paths)}] "
                f"{image_path.name}"
            )

        except Exception as error:

            print(
                f"Skipping {image_path.name}: {error}"
            )

    if len(features) == 0:

        raise RuntimeError(
            "Could not extract features from images."
        )

    feature_matrix = torch.cat(
        features,
        dim=0
    )

    print("\nFeature matrix:")
    print(feature_matrix.shape)

    # ========================================================
    # CALCULATE NORMAL DISTANCES
    # ========================================================
    #
    # For each normal image:
    #
    # compare it against all other normal images
    # and find nearest normal image.
    #
    # This is Leave-One-Out nearest neighbour distance.
    #

    distance_matrix = torch.cdist(
        feature_matrix,
        feature_matrix,
        p=2
    )

    # Ignore self-distance
    distance_matrix.fill_diagonal_(float("inf"))

    nearest_distances, nearest_indices = torch.min(
        distance_matrix,
        dim=1
    )

    print("\nNormal nearest-neighbour statistics:")

    print(
        f"Minimum : {nearest_distances.min().item():.6f}"
    )

    print(
        f"Maximum : {nearest_distances.max().item():.6f}"
    )

    print(
        f"Mean    : {nearest_distances.mean().item():.6f}"
    )

    print(
        f"Median  : "
        f"{nearest_distances.median().item():.6f}"
    )

    print(
        f"P90     : "
        f"{torch.quantile(nearest_distances, 0.90).item():.6f}"
    )

    print(
        f"P95     : "
        f"{torch.quantile(nearest_distances, 0.95).item():.6f}"
    )

    print(
        f"P99     : "
        f"{torch.quantile(nearest_distances, 0.99).item():.6f}"
    )

    # ========================================================
    # THRESHOLD
    # ========================================================

    p99 = torch.quantile(
        nearest_distances,
        0.99
    ).item()

    # Small safety margin
    threshold = p99 * 1.10

    print(
        f"\nFinal normal threshold: "
        f"{threshold:.6f}"
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {

        "feature_extractor": "resnet18",

        "image_size": IMAGE_SIZE,

        "feature_dimension":
            feature_matrix.shape[1],

        "normal_features":
            feature_matrix,

        "threshold":
            threshold,

        "num_normal_images":
            len(valid_images),

        "image_paths":
            valid_images
    }

    torch.save(
        checkpoint,
        MODEL_PATH
    )

    print("\nModel saved:")
    print(MODEL_PATH)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()