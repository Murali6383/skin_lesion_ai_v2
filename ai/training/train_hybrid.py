import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torchvision import models, transforms
import timm

from ai.config import IMAGE_SIZE, CLASS_FOLDER_ALIASES, MODELS_DIR
from ai.utils import seed_everything, device

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def folder_to_id(name):
    key = name.lower().replace("-", "_")
    if key not in CLASS_FOLDER_ALIASES:
        raise ValueError(f"Unknown class folder: {name}")
    return CLASS_FOLDER_ALIASES[key]

def collect(folder):
    rows = []
    for d in sorted(Path(folder).iterdir()):
        if d.is_dir():
            y = folder_to_id(d.name)
            for p in d.rglob("*"):
                if p.is_file() and p.suffix.lower() in EXTS:
                    rows.append((str(p), y))
    return rows

def build_models(dev):
    eff = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    eff.classifier = nn.Identity()
    inc = timm.create_model("inception_resnet_v2", pretrained=True, num_classes=0)
    return eff.to(dev).eval(), inc.to(dev).eval()

def extract(items, eff, inc, dev, batch_size=16):
    tfm = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406], [.229,.224,.225])
    ])
    X, y, paths = [], [], []
    for s in tqdm(range(0, len(items), batch_size), desc="Extracting"):
        batch = items[s:s+batch_size]
        ims, by = [], []
        bp = []
        for p, label in batch:
            try:
                ims.append(tfm(Image.open(p).convert("RGB")))
                by.append(label)
                bp.append(p)
            except Exception as e:
                print("Skipping", p, e)
        if not ims:
            continue
        x = torch.stack(ims).to(dev)
        with torch.inference_mode():
            a = eff(x)
            b = inc(x)
            f = torch.cat([a, b], 1).cpu().numpy()
        X.append(f); y.extend(by); paths.extend(bp)
    return np.concatenate(X), np.asarray(y), paths

def save(name, X, y, paths):
    np.save(MODELS_DIR / f"{name}_features.npy", X)
    np.save(MODELS_DIR / f"{name}_labels.npy", y)
    pd.DataFrame({"path": paths, "label": y}).to_csv(
        MODELS_DIR / f"{name}_metadata.csv", index=False)

def main(data):
    seed_everything()
    dev = device()
    root = Path(data)
    train = collect(root / "train")
    val = collect(root / "val") if (root / "val").exists() else []
    test = collect(root / "test") if (root / "test").exists() else []
    if not train:
        raise RuntimeError("No training images found.")
    print("Device:", dev)
    eff, inc = build_models(dev)
    torch.save(eff.state_dict(), MODELS_DIR / "efficientnet_b0.pth")
    torch.save(inc.state_dict(), MODELS_DIR / "inception_resnet_v2.pth")
    for name, items in (("train", train), ("val", val), ("test", test)):
        if items:
            X, y, paths = extract(items, eff, inc, dev)
            save(name, X, y, paths)
            print(name, X.shape)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    main(ap.parse_args().data)
