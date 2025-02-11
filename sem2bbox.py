# single process settings
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENCV_FOR_THREADS_NUM"] = "1"
os.environ["OPENCV_FFMPEG_THREADS"] = "1"

# import packages
import json
from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed
from pascal_voc_writer import Writer
from tqdm import tqdm

A2D2_ROOT = "E:/datasets/A2D2/camera_lidar_semantic/camera_lidar_semantic"
A2D2_ROOT = Path(A2D2_ROOT)
assert A2D2_ROOT.exists()

USE_CLASS = [
    "Car",
    "Bicycle",
    "Pedestrian",
    "Truck",
    "Small vehicles",
    "Traffic sign",
    "Traffic signal",
    "Utility vehicle",
    "Sidebars",
    "Irrelevant signs",
    "Road blocks",
    "Tractor",
    "Zebra crossing",
    "Obstacles / trash",
    "Poles",
    "Animals",
    "Signal corpus",
    "Electronic traffic",
    "Painted driv. instr.",
    "Traffic guide obj.",
]

output_dir = "E:/datasets/A2D2/2d_bbox_from_map"
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

    try:
        camera_param = camera_params[camera_name]
    except KeyError:
        camera_param = camera_params[camera_name.replace("cam_", "")]
    camera_matrix = np.array(camera_param["CamMatrix"])
    dist_coeffs = np.array(camera_param["Distortion"])
    lens_type = camera_param["Lens"]
    resolution = camera_param["Resolution"]

    camera_path = image_file.parent.name
    segmap_name = image_file.stem.replace("_camera_", "_label_")

    segmap_path = (
        image_file.parent.parent.parent / "label" / camera_path / f"{segmap_name}.png"
    )
    assert segmap_path.exists(), segmap_path
    assert segmap_path.is_file

    image = cv2.imread(image_file)
    segmap = cv2.imread(str(segmap_path))
    if lens_type == "Telecam":
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs,
            None,
            camera_matrix,
            resolution,
            cv2.CV_32FC1,
        )
    else:
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs,
            None,
            camera_matrix,
            resolution,
            cv2.CV_32FC1,
        )
    image_undistort = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
    segmap_undistort = cv2.remap(segmap, map1, map2, cv2.INTER_NEAREST)

    cv2.imwrite(out_image_path, image_undistort)
    relative_path = str(out_image_path.relative_to(output_dir))
    writer = Writer(str(relative_path), resolution[0], resolution[1])
    for key, value in classes.items():
        last_char_is_number = True
        try:
            int(value[-1])
        except ValueError:
            last_char_is_number = False
        class_name = value.rsplit(" ", 1)[0] if last_char_is_number else value
        if class_name not in USE_CLASS:
            continue

        colorcode = np.array([int(key[1:3], 16), int(key[3:5], 16), int(key[5:7], 16)])
        colorcode_bgr = colorcode[::-1].reshape(1, 1, 3)
        class_map = segmap_undistort == colorcode_bgr
        class_map = class_map.all(axis=-1)
        if class_map.sum() == 0:
            continue
        class_map = (class_map * 255).astype(np.uint8)

        contour, _ = cv2.findContours(class_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            x, y, w, h = cv2.boundingRect(cnt)
            writer.addObject(class_name, x, y, x + w, y + h)

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

    image_files = sorted(A2D2_ROOT.glob("**/camera/**/*.png"))

    # joblib parralel
    Parallel(n_jobs=-4, verbose=0)(
        delayed(process)(image_file, classes, camera_params)
        for image_file in tqdm(image_files)
    )


if __name__ == "__main__":
    main()
