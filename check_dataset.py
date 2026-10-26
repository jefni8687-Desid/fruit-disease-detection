
from pathlib import Path
from collections import Counter

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
PROJECT_ROOT = Path(__file__).resolve().parent

CANDIDATE_ROOTS = [
    PROJECT_ROOT / "dataset" / "disease_detection" / "Mango",
    PROJECT_ROOT / "dataset" / "fruit_identification" / "Mango",
    PROJECT_ROOT / "dataset" / "Mango",
    PROJECT_ROOT / "Mango",
]

def image_files(folder: Path):
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

def print_folder_summary(root: Path):
    print(f"\nDataset folder: {root}")
    class_folders = sorted(p for p in root.iterdir() if p.is_dir())
    if not class_folders:
        imgs = image_files(root)
        print(f"  Images directly inside folder: {len(imgs)}")
        return

    total = 0
    for class_dir in class_folders:
        files = image_files(class_dir)
        total += len(files)
        print(f"  {class_dir.name}: {len(files)} images")
    print(f"  TOTAL: {total} images across {len(class_folders)} class folders")

def main():
    print("=" * 60)
    print("FruitCare AI — Dataset Checker")
    print(f"Project root: {PROJECT_ROOT}")
    print("=" * 60)

    found = []
    seen = set()
    for root in CANDIDATE_ROOTS:
        if root.exists() and root.is_dir() and root.resolve() not in seen:
            found.append(root)
            seen.add(root.resolve())
            print_folder_summary(root)

    if not found:
        print("\nNo expected Mango dataset folder was found.")
        print("Please check that the dataset is inside your project folder.")
        print("\nExpected example:")
        print("  dataset/disease_detection/Mango/Alternaria/")
        print("  dataset/disease_detection/Mango/Anthracnose/")
        print("  dataset/disease_detection/Mango/Black Mould Rot/")
        print("  dataset/disease_detection/Mango/Healthy/")
        print("  dataset/disease_detection/Mango/Stem and Rot/")
        print("\nIf your folder names differ, edit CANDIDATE_ROOTS in this script.")

    print("\nNote: these counts describe the folders found. They do not prove")
    print("that the images are correctly labelled or that a model is trained.")

if __name__ == "__main__":
    main()
