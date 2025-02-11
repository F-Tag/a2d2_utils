# single process settings
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# import packages
import json
from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed
from pascal_voc_writer import Writer
from tqdm import tqdm

A2D2_ROOT = "E:/datasets/A2D2/camera_lidar_semantic_bboxes/camera_lidar_semantic_bboxes"
A2D2_ROOT = Path(A2D2_ROOT)
assert A2D2_ROOT.exists()

UNDISTORT = False

output_dir = "E:/datasets/A2D2/2d_bbox_from_3d"
if UNDISTORT:
    output_dir += "_undistorted"
output_dir = Path(output_dir)


def process(image_file, classes, camera_params):
    out_image_dir = output_dir / "JPEGImages"
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_image_path = out_image_dir / f"{image_file.stem}.jpg"

    out_label_dir = output_dir / "Annotations"
    out_label_dir.mkdir(parents=True, exist_ok=True)
    out_label_path = out_label_dir / f"{image_file.stem}.xml"

    if out_image_path.exists() and out_label_path.exists():
        return

    image_json = image_file.with_suffix(".json")
    if image_json.exists():
        with open(image_json, "r") as f:
            image_json = json.load(f)
        camera_name = image_json["cam_name"]
    else:
        camera_name = image_file.parent.name

    camera_param = camera_params[camera_name]
    camera_matrix = np.array(camera_param["CamMatrix"])
    dist_coeffs = np.array(camera_param["Distortion"])
    lens_type = camera_param["Lens"]
    resolution = camera_param["Resolution"]

    camera_path = image_file.parent.name
    bbox_name = image_file.stem.replace("_camera_", "_label3D_")
    bbox_path = (
        image_file.parent.parent.parent / "label3D" / camera_path / f"{bbox_name}.json"
    )

    image = cv2.imread(image_file)
    with bbox_path.open("r") as f:
        bbox = json.load(f)

    bboxes, classes = [], []
    for _, value in bbox.items():
        bboxes.append(value["2d_bbox"])
        classes.append(value["class"])
    bboxes = np.array(bboxes)

    if UNDISTORT:
        bboxes = bboxes.reshape(1, -1, 2)
        if lens_type == "Telecam":
            image = cv2.undistort(
                image, camera_matrix, dist_coeffs, None, camera_matrix
            )
            bboxes = cv2.undistortPoints(
                bboxes, camera_matrix, dist_coeffs, None, camera_matrix
            )
        else:
            image = cv2.fisheye.undistortImage(
                image, camera_matrix, dist_coeffs, None, camera_matrix
            )
            bboxes = cv2.fisheye.undistortPoints(
                bboxes, camera_matrix, dist_coeffs, None, camera_matrix
            )
        bboxes = bboxes.squeeze(1).reshape(-1, 4)

    cv2.imwrite(out_image_path, image)

    relative_path = str(out_image_path.relative_to(output_dir))
    writer = Writer(str(relative_path), resolution[0], resolution[1])
    for class_name, bbox in zip(classes, bboxes):
        x1, y1, x2, y2 = bbox
        writer.addObject(class_name, x1, y1, x2, y2)

    writer.save(out_label_path)


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
    Parallel(n_jobs=-2, verbose=0)(
        delayed(process)(image_file, classes, camera_params)
        for image_file in tqdm(image_files)
    )


if __name__ == "__main__":
    main()
