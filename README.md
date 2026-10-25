# Skin Lesion Classification — Hybrid CNN + EGA

Pipeline:
Image -> preprocessing -> EfficientNet-B0 + Inception-ResNet-v2 -> feature fusion -> EGA feature selection -> classifier -> prediction.

## 9 classes
0 Actinic Keratosis
1 Atopic Dermatitis
2 Benign Keratosis
3 Dermatofibroma
4 Melanocytic Nevus
5 Melanoma
6 Squamous Cell Carcinoma
7 Tinea Ringworm Candidiasis
8 Vascular Lesion

## Dataset
Put images under:
dataset/train/<class>/
dataset/val/<class>/
dataset/test/<class>/

Run:
python -m ai.check_dataset --data dataset
python -m ai.training.train_hybrid --data dataset
python -m ai.training.train_ega_classifier
python -m ai.predict --image path/to/image.jpg

API:
uvicorn backend.main:app --reload
Open http://127.0.0.1:8000/docs

Metrics are calculated from the supplied dataset; no accuracy is hard-coded.
Output is a research classification result, not a medical diagnosis.

STEPS MODEL TRAIN

1.python -m ai.training.train_hybrid
2.python -m ai.training.train_ega_classifier

STEPS :

1.//pip install -r requirements.txt

2.//python -m uvicorn backend.main:app --reload

to open another terminal :

1. cd frontend
2. npm install
3. npm run dev



python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8001


