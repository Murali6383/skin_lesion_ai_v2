from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models_saved"
RESULTS_DIR = ROOT / "results"
IMAGE_SIZE = 224
RANDOM_STATE = 42

CLASS_NAMES = [
    "Actinic Keratosis", "Atopic Dermatitis", "Benign Keratosis",
    "Dermatofibroma", "Melanocytic Nevus", "Melanoma",
    "Squamous Cell Carcinoma", "Tinea Ringworm Candidiasis",
    "Vascular Lesion"
]

CLASS_FOLDER_ALIASES = {
    "actinic_keratosis": 0, "actinic keratosis": 0,
    "atopic_dermatitis": 1, "atopic dermatitis": 1,
    "benign_keratosis": 2, "benign keratosis": 2,
    "dermatofibroma": 3,
    "melanocytic_nevus": 4, "melanocytic nevus": 4,
    "melanoma": 5,
    "squamous_cell_carcinoma": 6, "squamous cell carcinoma": 6,
    "tinea_ringworm_candidiasis": 7, "tinea ringworm candidiasis": 7,
    "vascular_lesion": 8, "vascular lesion": 8,
}

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
