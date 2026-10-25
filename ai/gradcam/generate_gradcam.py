from pathlib import Path
import sys
import uuid


# ============================================================
# PROJECT ROOT / PYTHON PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# IMPORTS
# ============================================================

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms


# ============================================================
# PATHS
# ============================================================

MODELS_DIR = ROOT / "models_saved"

RESULTS_DIR = ROOT / "results" / "gradcam"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL FILES
# ============================================================

EFFICIENTNET_PATH = (
    MODELS_DIR / "efficientnet_b0.pth"
)

EGA_PATH = (
    MODELS_DIR / "ega_selector.joblib"
)

CLASSIFIER_PATH = (
    MODELS_DIR / "classifier.joblib"
)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose(
    [
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
        ),
    ]
)


# ============================================================
# CHECK MODEL FILES
# ============================================================

def check_model_files():

    required_files = [
        EFFICIENTNET_PATH,
        EGA_PATH,
        CLASSIFIER_PATH,
    ]

    print()
    print("==========================================")
    print("Checking Grad-CAM model files")
    print("==========================================")

    for file_path in required_files:

        print(
            f"{file_path.name:<30} : ",
            end=""
        )

        if file_path.exists():

            print("FOUND")

        else:

            print("MISSING")

            raise FileNotFoundError(
                f"Required model file not found:\n"
                f"{file_path}"
            )

    print()


# ============================================================
# LOAD EFFICIENTNET-B0
# ============================================================

def load_efficientnet():

    print(
        f"[Grad-CAM] Loading EfficientNet-B0 "
        f"on {DEVICE}"
    )

    model = models.efficientnet_b0(
        weights=None
    )

    # --------------------------------------------------------
    # IMPORTANT
    # Existing disease model uses EfficientNet
    # as feature extractor.
    # --------------------------------------------------------

    model.classifier = nn.Identity()

    checkpoint = torch.load(
        EFFICIENTNET_PATH,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Support checkpoint dictionary
    # --------------------------------------------------------

    if (
        isinstance(
            checkpoint,
            dict
        )
        and "state_dict" in checkpoint
    ):

        checkpoint = checkpoint[
            "state_dict"
        ]

    model.load_state_dict(
        checkpoint,
        strict=True
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model


# ============================================================
# LOAD EGA + CLASSIFIER
# ============================================================

def load_ml_models():

    print(
        "[Grad-CAM] Loading EGA selector"
    )

    ega_selector = joblib.load(
        EGA_PATH
    )

    print(
        "[Grad-CAM] Loading classifier"
    )

    classifier = joblib.load(
        CLASSIFIER_PATH
    )

    return (
        ega_selector,
        classifier
    )


# ============================================================
# GET CLASSIFIER COEFFICIENTS
# ============================================================

def get_classifier_weights(
    classifier,
    class_index
):

    if not hasattr(
        classifier,
        "coef_"
    ):

        raise RuntimeError(
            "The classifier does not expose "
            "'coef_'. Cannot generate Grad-CAM "
            "from the current classifier."
        )

    coefficients = np.asarray(
        classifier.coef_,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Binary classifier
    # --------------------------------------------------------

    if coefficients.ndim == 1:

        return coefficients

    # --------------------------------------------------------
    # Multi-class classifier
    # --------------------------------------------------------

    if class_index < 0:

        class_index = 0

    if class_index >= coefficients.shape[0]:

        class_index = (
            coefficients.shape[0] - 1
        )

    return coefficients[
        class_index
    ]


# ============================================================
# GET EGA FEATURE INDICES
# ============================================================

def get_selected_indices(
    ega_selector
):

    # --------------------------------------------------------
    # sklearn selector
    # --------------------------------------------------------

    if hasattr(
        ega_selector,
        "get_support"
    ):

        try:

            indices = (
                ega_selector.get_support(
                    indices=True
                )
            )

            return np.asarray(
                indices,
                dtype=np.int64
            )

        except Exception:

            pass

    # --------------------------------------------------------
    # support_
    # --------------------------------------------------------

    if hasattr(
        ega_selector,
        "support_"
    ):

        try:

            support = np.asarray(
                ega_selector.support_
            )

            if support.dtype == bool:

                return np.where(
                    support
                )[0]

        except Exception:

            pass

    # --------------------------------------------------------
    # selected_features_
    # --------------------------------------------------------

    if hasattr(
        ega_selector,
        "selected_features_"
    ):

        try:

            return np.asarray(
                ega_selector.selected_features_,
                dtype=np.int64
            )

        except Exception:

            pass

    # --------------------------------------------------------
    # selected_features
    # --------------------------------------------------------

    if hasattr(
        ega_selector,
        "selected_features"
    ):

        try:

            return np.asarray(
                ega_selector.selected_features,
                dtype=np.int64
            )

        except Exception:

            pass

    # --------------------------------------------------------
    # indices_
    # --------------------------------------------------------

    if hasattr(
        ega_selector,
        "indices_"
    ):

        try:

            return np.asarray(
                ega_selector.indices_,
                dtype=np.int64
            )

        except Exception:

            pass

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    print(
        "[Grad-CAM] WARNING:"
    )

    print(
        "[Grad-CAM] Could not automatically "
        "read EGA selected feature indices."
    )

    return None


# ============================================================
# CREATE EFFICIENTNET CHANNEL WEIGHTS
# ============================================================

def create_efficientnet_channel_weights(
    ega_selector,
    classifier,
    class_index
):

    # EfficientNet-B0 feature size
    EFFICIENTNET_FEATURES = 1280

    # Inception-ResNet-v2 feature size
    INCEPTION_FEATURES = 1536

    # Combined feature size
    TOTAL_FEATURES = (
        EFFICIENTNET_FEATURES
        +
        INCEPTION_FEATURES
    )

    classifier_weights = (
        get_classifier_weights(
            classifier,
            class_index
        )
    )

    selected_indices = (
        get_selected_indices(
            ega_selector
        )
    )

    print(
        "[Grad-CAM] Classifier feature count:",
        len(classifier_weights)
    )

    if selected_indices is not None:

        print(
            "[Grad-CAM] EGA selected feature count:",
            len(selected_indices)
        )

    else:

        print(
            "[Grad-CAM] EGA feature indices: unavailable"
        )

    channel_weights = np.zeros(
        EFFICIENTNET_FEATURES,
        dtype=np.float32
    )

    # ========================================================
    # CASE 1
    # EGA exposes selected indices
    # ========================================================

    if selected_indices is not None:

        usable_count = min(
            len(
                classifier_weights
            ),
            len(
                selected_indices
            )
        )

        for i in range(
            usable_count
        ):

            original_index = int(
                selected_indices[i]
            )

            # ------------------------------------------------
            # EfficientNet portion only
            # ------------------------------------------------

            if (
                0 <= original_index
                < EFFICIENTNET_FEATURES
            ):

                channel_weights[
                    original_index
                ] = (
                    classifier_weights[i]
                )

    # ========================================================
    # CASE 2
    # EGA indices unavailable
    # ========================================================

    else:

        print(
            "[Grad-CAM] Using fallback "
            "EfficientNet feature mapping."
        )

        count = min(
            EFFICIENTNET_FEATURES,
            len(classifier_weights)
        )

        channel_weights[
            :count
        ] = classifier_weights[
            :count
        ]

    # ========================================================
    # Safety fallback
    # ========================================================

    if np.allclose(
        channel_weights,
        0
    ):

        print(
            "[Grad-CAM] WARNING:"
        )

        print(
            "[Grad-CAM] EfficientNet channel "
            "weights are zero."
        )

        count = min(
            EFFICIENTNET_FEATURES,
            len(classifier_weights)
        )

        channel_weights[
            :count
        ] = classifier_weights[
            :count
        ]

    return channel_weights


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(
    image_path
):

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    tensor = transform(
        image
    )

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.to(
        DEVICE
    )

    return (
        image,
        tensor
    )


# ============================================================
# CREATE GRAD-CAM
# ============================================================

def generate_gradcam(
    image_path,
    class_index=0,
    output_path=None
):

    image_path = Path(
        image_path
    )

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n"
            f"{image_path}"
        )

    # --------------------------------------------------------
    # Check model files
    # --------------------------------------------------------

    check_model_files()

    # --------------------------------------------------------
    # Load EfficientNet
    # --------------------------------------------------------

    model = load_efficientnet()

    # --------------------------------------------------------
    # Load EGA + classifier
    # --------------------------------------------------------

    (
        ega_selector,
        classifier
    ) = load_ml_models()

    # --------------------------------------------------------
    # Target layer
    #
    # EfficientNet-B0:
    #
    # model.features[-1]
    #
    # is the final convolutional feature block.
    # --------------------------------------------------------

    target_layer = (
        model.features[-1]
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    (
        original_image,
        input_tensor
    ) = preprocess_image(
        image_path
    )

    # --------------------------------------------------------
    # Generate classifier-derived
    # EfficientNet channel weights
    # --------------------------------------------------------

    channel_weights = (
        create_efficientnet_channel_weights(
            ega_selector,
            classifier,
            class_index
        )
    )

    channel_weights_tensor = torch.tensor(
        channel_weights,
        dtype=torch.float32,
        device=DEVICE
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    activations = []

    gradients = []

    # ========================================================
    # FORWARD HOOK
    # ========================================================

    def forward_hook(
        module,
        inputs,
        output
    ):

        activations.append(
            output
        )

    # ========================================================
    # BACKWARD HOOK
    # ========================================================

    def backward_hook(
        module,
        grad_input,
        grad_output
    ):

        gradients.append(
            grad_output[0]
        )

    # --------------------------------------------------------
    # Register hooks
    # --------------------------------------------------------

    forward_handle = (
        target_layer.register_forward_hook(
            forward_hook
        )
    )

    backward_handle = (
        target_layer.register_full_backward_hook(
            backward_hook
        )
    )

    try:

        # ----------------------------------------------------
        # Clear previous gradients
        # ----------------------------------------------------

        model.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        features = model(
            input_tensor
        )

        # ----------------------------------------------------
        # Expected:
        #
        # [1, 1280]
        # ----------------------------------------------------

        if features.ndim != 2:

            features = (
                features.flatten(
                    start_dim=1
                )
            )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if (
            features.shape[1]
            != 1280
        ):

            raise RuntimeError(
                "Unexpected EfficientNet feature "
                f"dimension: {features.shape}"
            )

        # ----------------------------------------------------
        # Create target score
        #
        # This uses classifier-derived
        # EfficientNet channel weights.
        # ----------------------------------------------------

        score = torch.sum(
            features
            *
            channel_weights_tensor.unsqueeze(
                0
            )
        )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        score.backward()

        # ----------------------------------------------------
        # Validate hooks
        # ----------------------------------------------------

        if not activations:

            raise RuntimeError(
                "Grad-CAM activation was not captured."
            )

        if not gradients:

            raise RuntimeError(
                "Grad-CAM gradient was not captured."
            )

        # ----------------------------------------------------
        # Read tensors
        # ----------------------------------------------------

        activation = (
            activations[0]
        )

        gradient = (
            gradients[0]
        )

        # ----------------------------------------------------
        # Gradient global average pooling
        # ----------------------------------------------------

        weights = gradient.mean(
            dim=(2, 3),
            keepdim=True
        )

        # ----------------------------------------------------
        # Weighted activation maps
        # ----------------------------------------------------

        cam = (
            weights
            *
            activation
        ).sum(
            dim=1
        )

        # ----------------------------------------------------
        # ReLU
        # ----------------------------------------------------

        cam = torch.relu(
            cam
        )

        # ----------------------------------------------------
        # Convert to NumPy
        # ----------------------------------------------------

        cam = cam[
            0
        ].detach().cpu().numpy()

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        cam -= cam.min()

        maximum = cam.max()

        if maximum > 0:

            cam /= maximum

        # ----------------------------------------------------
        # Original dimensions
        # ----------------------------------------------------

        original_width, original_height = (
            original_image.size
        )

        # ----------------------------------------------------
        # Resize CAM
        # ----------------------------------------------------

        cam = cv2.resize(
            cam,
            (
                original_width,
                original_height
            ),
            interpolation=cv2.INTER_LINEAR
        )

        # ----------------------------------------------------
        # Convert CAM to heatmap
        # ----------------------------------------------------

        heatmap = np.uint8(
            cam * 255
        )

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        # ----------------------------------------------------
        # Original image
        # ----------------------------------------------------

        original_cv = cv2.cvtColor(
            np.asarray(
                original_image
            ),
            cv2.COLOR_RGB2BGR
        )

        # ----------------------------------------------------
        # Overlay
        # ----------------------------------------------------

        overlay = cv2.addWeighted(
            original_cv,
            0.60,
            heatmap,
            0.40,
            0
        )

        # ----------------------------------------------------
        # Output filename
        # ----------------------------------------------------

        if output_path is None:

            filename = (
                f"{image_path.stem}_"
                f"gradcam_"
                f"{uuid.uuid4().hex[:8]}.jpg"
            )

            output_path = (
                RESULTS_DIR /
                filename
            )

        else:

            output_path = Path(
                output_path
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Save image
        # ----------------------------------------------------

        saved = cv2.imwrite(
            str(output_path),
            overlay
        )

        if not saved:

            raise RuntimeError(
                f"Could not save Grad-CAM image:\n"
                f"{output_path}"
            )

        print()
        print(
            "=========================================="
        )
        print(
            "Grad-CAM Generated Successfully"
        )
        print(
            "=========================================="
        )

        print(
            f"Input     : {image_path}"
        )

        print(
            f"Class     : {class_index}"
        )

        print(
            f"Device    : {DEVICE}"
        )

        print(
            f"Output    : {output_path}"
        )

        print(
            "=========================================="
        )

        return {
            "success": True,
            "path": str(
                output_path
            ),
            "filename": output_path.name,
            "class_index": int(
                class_index
            ),
            "device": str(
                DEVICE
            ),
        }

    finally:

        # ----------------------------------------------------
        # Always remove hooks
        # ----------------------------------------------------

        forward_handle.remove()

        backward_handle.remove()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "=========================================="
    )

    print(
        "SkinGuardian AI - Grad-CAM"
    )

    print(
        "=========================================="
    )

    # --------------------------------------------------------
    # Argument validation
    # --------------------------------------------------------

    if len(sys.argv) < 2:

        print()
        print(
            "Usage:"
        )

        print(
            "python "
            "ai\\gradcam\\generate_gradcam.py "
            "<image_path> [class_index]"
        )

        print()

        print(
            "Example:"
        )

        print(
            "python "
            "ai\\gradcam\\generate_gradcam.py "
            "ai\\test_image.jpg 0"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    image_path = sys.argv[1]

    # --------------------------------------------------------
    # Class index
    # --------------------------------------------------------

    class_index = 0

    if len(sys.argv) >= 3:

        try:

            class_index = int(
                sys.argv[2]
            )

        except ValueError:

            print(
                "ERROR: class_index must be an integer."
            )

            sys.exit(1)

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    try:

        result = generate_gradcam(
            image_path,
            class_index
        )

        print()

        print(
            "Grad-CAM file:"
        )

        print(
            result["path"]
        )

    except Exception as error:

        print()

        print(
            "=========================================="
        )

        print(
            "Grad-CAM ERROR"
        )

        print(
            "=========================================="
        )

        print(
            str(error)
        )

        print(
            "=========================================="
        )

        sys.exit(1)