import argparse
from pathlib import Path
from PIL import Image
from ai.config import CLASS_FOLDER_ALIASES

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def scan(root):
    root = Path(root)
    total = bad = 0
    print("=" * 60)
    print("SKIN LESION DATASET CHECK")
    print("=" * 60)
    for split in ("train", "val", "test"):
        d = root / split
        if not d.exists():
            continue
        print(f"\n[{split.upper()}]")
        for folder in sorted(d.iterdir()):
            if not folder.is_dir():
                continue
            key = folder.name.lower().replace("-", "_")
            files = [p for p in folder.rglob("*") if p.suffix.lower() in EXTS]
            print(f"{folder.name:35s} {len(files):6d}  class_id={CLASS_FOLDER_ALIASES.get(key)}")
            total += len(files)
            for f in files:
                try:
                    with Image.open(f) as im:
                        im.verify()
                except Exception:
                    bad += 1
                    print("  BAD:", f)
    print("\nTotal images:", total)
    print("Unreadable images:", bad)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    scan(ap.parse_args().data)
