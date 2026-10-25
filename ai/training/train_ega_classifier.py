import json
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from ai.config import MODELS_DIR, RESULTS_DIR, CLASS_NAMES
from ai.ega.enhanced_genetic_algorithm import EnhancedGeneticAlgorithm

def main():
    Xtr = np.load(MODELS_DIR / "train_features.npy")
    ytr = np.load(MODELS_DIR / "train_labels.npy")
    Xv = np.load(MODELS_DIR / "val_features.npy")
    yv = np.load(MODELS_DIR / "val_labels.npy")
    if (MODELS_DIR / "test_features.npy").exists():
        Xt = np.load(MODELS_DIR / "test_features.npy")
        yt = np.load(MODELS_DIR / "test_labels.npy")
    else:
        Xt, yt = Xv, yv

    ega = EnhancedGeneticAlgorithm(Xtr.shape[1])
    ega.fit(Xtr, ytr, Xv, yv)

    Xtr, Xt = ega.transform(Xtr), ega.transform(Xt)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xt)

    acc = accuracy_score(yt, pred)
    pr, rc, f1, _ = precision_recall_fscore_support(
        yt, pred, average="weighted", zero_division=0)
    report = classification_report(
        yt, pred, labels=list(range(9)), target_names=CLASS_NAMES, zero_division=0)

    metrics = {
        "accuracy": float(acc),
        "weighted_precision": float(pr),
        "weighted_recall": float(rc),
        "weighted_f1": float(f1),
        "total_fused_features": int(Xtr.shape[1]),
        "selected_features": int(ega.get_support().sum())
    }
    print(report)
    (RESULTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm = confusion_matrix(yt, pred, labels=list(range(9)))
    plt.figure(figsize=(11, 9))
    plt.imshow(cm)
    plt.colorbar()
    plt.xticks(range(9), CLASS_NAMES, rotation=60, ha="right")
    plt.yticks(range(9), CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Hybrid CNN + EGA Confusion Matrix")
    for i in range(9):
        for j in range(9):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    joblib.dump(ega, MODELS_DIR / "ega_selector.joblib")
    joblib.dump(clf, MODELS_DIR / "classifier.joblib")
    np.save(MODELS_DIR / "ega_selected_features.npy", np.flatnonzero(ega.get_support()))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
