import argparse
import joblib
import torch

from PIL import Image
from torchvision import transforms, models
from torch import nn
import timm

from ai.config import IMAGE_SIZE, CLASS_NAMES, MODELS_DIR
from ai.utils import device


class SkinLesionPredictor:

    def __init__(self):

        self.dev = device()

        print("Loading EfficientNet-B0...")

        self.eff = models.efficientnet_b0(
            weights=None
        )

        self.eff.classifier = nn.Identity()

        self.eff.load_state_dict(
            torch.load(
                MODELS_DIR / "efficientnet_b0.pth",
                map_location=self.dev
            )
        )

        print("Loading Inception-ResNet-v2...")

        self.inc = timm.create_model(
            "inception_resnet_v2",
            pretrained=False,
            num_classes=0
        )

        self.inc.load_state_dict(
            torch.load(
                MODELS_DIR / "inception_resnet_v2.pth",
                map_location=self.dev
            )
        )

        self.eff.to(self.dev).eval()
        self.inc.to(self.dev).eval()

        print("Loading EGA selector...")

        self.ega = joblib.load(
            MODELS_DIR / "ega_selector.joblib"
        )

        print("Loading Logistic Regression...")

        self.clf = joblib.load(
            MODELS_DIR / "classifier.joblib"
        )

        self.tfm = transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                [.485, .456, .406],
                [.229, .224, .225]
            )
        ])

        print("All disease models loaded successfully.")

    # =====================================================
    # DISEASE PREDICTION
    # =====================================================

    def predict(self, image_path):

        image = Image.open(
            image_path
        ).convert("RGB")

        x = self.tfm(
            image
        ).unsqueeze(0).to(self.dev)

        # =================================================
        # HYBRID CNN FEATURE EXTRACTION
        # =================================================

        with torch.inference_mode():

            eff_features = self.eff(x)

            inc_features = self.inc(x)

            features = torch.cat(
                [
                    eff_features,
                    inc_features
                ],
                dim=1
            ).cpu().numpy()

        # =================================================
        # EGA FEATURE SELECTION
        # =================================================

        selected_features = self.ega.transform(
            features
        )

        # =================================================
        # LOGISTIC REGRESSION
        # =================================================

        pred = int(
            self.clf.predict(
                selected_features
            )[0]
        )

        probabilities = self.clf.predict_proba(
            selected_features
        )[0]

        confidence = float(
            probabilities[pred] * 100
        )

        disease = CLASS_NAMES[pred]

        return {
            "prediction": disease,
            "confidence": round(
                confidence,
                2
            )
        }


# =========================================================
# GLOBAL PREDICTOR
# =========================================================
#
# Loaded only once.
# FastAPI can reuse this object.
#
# =========================================================

predictor = SkinLesionPredictor()


# =========================================================
# CLI FUNCTION
# =========================================================

def main(image_path):

    result = predictor.predict(
        image_path
    )

    print()
    print("======================================")
    print("       SKIN LESION PREDICTION")
    print("======================================")

    print(
        "Predicted class:",
        result["prediction"]
    )

    print(
        f'Confidence: {result["confidence"]:.2f}%'
    )

    print("======================================")

    return {
        "success": True,

        "predicted_class":
            result["prediction"],

        "confidence_percent":
            result["confidence"],

        "note": (
            "Research classification output; "
            "not a medical diagnosis."
        )
    }


# =========================================================
# COMMAND LINE TESTING
# =========================================================

if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--image",
        required=True
    )

    args = ap.parse_args()

    main(
        args.image
    )