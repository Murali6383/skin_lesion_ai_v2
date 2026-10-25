from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Example comparison values for manuscript visualization.
# Replace these numbers with your exact validation/test results if you have different baselines.
MODEL_COMPARISON = {
    "EfficientNet-B0": {"Accuracy": 84.0, "Macro-F1": 82.0},
    "Inception-ResNet-v2": {"Accuracy": 86.0, "Macro-F1": 84.0},
    "Proposed Hybrid + EGA": {"Accuracy": 88.4, "Macro-F1": 90.0},
}

labels = list(MODEL_COMPARISON.keys())
accuracy = [MODEL_COMPARISON[m]["Accuracy"] for m in labels]
macro_f1 = [MODEL_COMPARISON[m]["Macro-F1"] for m in labels]

# Figure 1: performance comparison
fig, ax = plt.subplots(figsize=(8, 5.2))
bar_width = 0.35
x = range(len(labels))
ax.bar([i - bar_width / 2 for i in x], accuracy, width=bar_width, label="Accuracy (%)", color="#4C78A8")
ax.bar([i + bar_width / 2 for i in x], macro_f1, width=bar_width, label="Macro-F1 (%)", color="#F58518")
ax.set_xticks(list(x))
ax.set_xticklabels(labels, rotation=18, ha="right")
ax.set_ylabel("Performance (%)")
ax.set_title("Performance Comparison of CNN Backbones and Proposed Hybrid Model")
ax.set_ylim(0, 100)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.legend(frameon=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for i, v in enumerate(accuracy):
    ax.text(i - bar_width / 2, v + 1.8, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
for i, v in enumerate(macro_f1):
    ax.text(i + bar_width / 2, v + 1.8, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
fig.tight_layout()
fig.savefig(OUT_DIR / "model_comparison.png", dpi=300)
fig.savefig(OUT_DIR / "model_comparison.svg", dpi=300)
plt.close(fig)

# Figure 2: model architecture diagram
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("off")

blocks = [
    ("Input Image\n224x224x3", 0.04, 0.72, 0.16, 0.18),
    ("Preprocessing\nResize + Normalize", 0.22, 0.72, 0.16, 0.18),
    ("EfficientNet-B0\nBackbone", 0.40, 0.72, 0.18, 0.18),
    ("Inception-ResNet-v2\nBackbone", 0.60, 0.72, 0.18, 0.18),
    ("Feature Fusion\nConcat Features", 0.80, 0.72, 0.18, 0.18),
    ("EGA Selection\nDimensionality Reduction", 0.32, 0.36, 0.24, 0.18),
    ("Classifier\nSoftmax / Decision Layer", 0.58, 0.36, 0.22, 0.18),
    ("Prediction\n9 Skin Classes", 0.82, 0.36, 0.16, 0.18),
]

for label, x, y, w, h in blocks:
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        edgecolor="#2F4F4F",
        facecolor="#EAF2F8"
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10, color="#1F2937")

for i in range(len(blocks) - 1):
    x1, y1, w1, h1 = blocks[i][1:]
    x2, y2, w2, h2 = blocks[i + 1][1:]
    start = (x1 + w1, y1 + h1 / 2)
    end = (x2, y2 + h2 / 2)
    arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=14, linewidth=1.5, color="#3B82F6")
    ax.add_patch(arrow)

ax.text(0.5, 0.92, "Proposed Hybrid Skin Lesion Classification Framework", ha="center", va="center",
        fontsize=15, fontweight="bold", color="#111827")
fig.tight_layout()
fig.savefig(OUT_DIR / "proposed_model_architecture.png", dpi=300)
fig.savefig(OUT_DIR / "proposed_model_architecture.svg", dpi=300)
plt.close(fig)

print(f"Graphs saved to: {OUT_DIR}")
print("- model_comparison.png / .svg")
print("- proposed_model_architecture.png / .svg")
