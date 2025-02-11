# import packages
import json
from pathlib import Path
from collections import Counter

from joblib import Parallel, delayed
from tqdm import tqdm

A2D2_ROOT = "E:/datasets/A2D2/camera_lidar_semantic_bboxes/camera_lidar_semantic_bboxes"
A2D2_ROOT = Path(A2D2_ROOT)
assert A2D2_ROOT.exists()

def process(image_file, classes, camera_params):
    camera_path = image_file.parent.name
    bbox_name = image_file.stem.replace("_camera_", "_label3D_")
    bbox_path = (
        image_file.parent.parent.parent / "label3D" / camera_path / f"{bbox_name}.json"
    )

    with bbox_path.open("r") as f:
        bbox = json.load(f)

    classes = []
    for _, value in bbox.items():
        classes.append(value["class"])
    return classes


def main():
    camera_params_path = A2D2_ROOT / "cams_lidars.json"
    assert camera_params_path.exists()
    assert camera_params_path.is_file

    with open(camera_params_path, "r") as f:
        camera_params = json.load(f)["cameras"]

    classes = A2D2_ROOT / "class_list.json"
    assert classes.exists()
    assert classes.is_file

    with open(classes, "r") as f:
        classes = json.load(f)

    image_files = A2D2_ROOT.glob("**/camera/**/*.png")

    # joblib parralel
    classes = Parallel(n_jobs=-4, verbose=0)(
        delayed(process)(image_file, classes, camera_params)
        for image_file in tqdm(image_files)
    )
    
    # flatten list in list
    classes = [item for sublist in classes for item in sublist]
    classes = Counter(classes)

    # sort by num of classes
    classes = sorted(classes.items(), key=lambda x: x[1], reverse=True)
    print(json.dumps([class_name for class_name, _ in classes]))
    print(len(classes))



if __name__ == "__main__":
    main()
