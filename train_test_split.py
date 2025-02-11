from collections import Counter
from pathlib import Path

# ROOT = "E:/datasets/A2D2/2d_bbox_from_3d/VOC2007"
ROOT = r"E:\datasets\A2D2\2d_bbox_from_map"
ROOT = Path(ROOT)

if __name__ == "__main__":
    anotation_dir = ROOT / "Annotations"

    files = []
    for file in anotation_dir.glob("*.xml"):
        files.append(file.stem)

    dates = [name.split("_")[0] for name in files]

    counter = Counter(dates)
    min_date = min(counter, key=counter.get)

    train = ROOT / "ImageSets" / "Main" / "train.txt"
    test = ROOT / "ImageSets" / "Main" / "test.txt"
    train.parent.mkdir(parents=True, exist_ok=True)

    with open(train, "w") as f, open(test, "w") as f2:
        for name in files:
            if min_date in name:
                f2.write(f"{name}\n")
            else:
                f.write(f"{name}\n")
