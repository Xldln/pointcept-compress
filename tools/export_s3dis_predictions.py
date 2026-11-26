#!/usr/bin/env python3
"""Export Pointcept S3DIS room predictions as colored PLY files."""

import argparse
from pathlib import Path

import numpy as np


CLASS_NAMES = (
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "table",
    "chair",
    "sofa",
    "bookcase",
    "board",
    "clutter",
)

# Distinct RGB colors indexed by Pointcept's S3DIS semantic class IDs.
CLASS_COLORS = np.array(
    [
        [174, 199, 232],
        [152, 223, 138],
        [31, 119, 180],
        [255, 187, 120],
        [188, 189, 34],
        [140, 86, 75],
        [255, 152, 150],
        [214, 39, 40],
        [197, 176, 213],
        [148, 103, 189],
        [196, 156, 148],
        [23, 190, 207],
        [247, 182, 210],
    ],
    dtype=np.uint8,
)


def write_ply(path, coord, prediction):
    colors = CLASS_COLORS[prediction]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(coord)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property int pred_label\n"
        "end_header\n"
    )
    vertices = np.empty(
        len(coord),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("pred_label", "<i4"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = coord.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    vertices["pred_label"] = prediction
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        vertices.tofile(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path, help="Directory containing *_pred.npy files")
    parser.add_argument("--data-root", type=Path, default=Path("data/s3dis"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    result_dir = args.result_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else result_dir / "cloudcompare_ply"
    )
    predictions = sorted(result_dir.glob("Area_5-*_pred.npy"))
    if not predictions:
        parser.error(f"No Area_5 prediction files found in {result_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for prediction_path in predictions:
        room_name = prediction_path.stem.removeprefix("Area_5-").removesuffix("_pred")
        room_dir = data_root / "Area_5" / room_name
        coord_path = room_dir / "coord.npy"
        if not coord_path.is_file():
            raise FileNotFoundError(f"Coordinates not found for {room_name}: {coord_path}")

        coord = np.load(coord_path, allow_pickle=False)
        prediction = np.load(prediction_path, allow_pickle=False).reshape(-1)
        if coord.ndim != 2 or coord.shape[1] != 3:
            raise ValueError(f"Expected N x 3 coordinates in {coord_path}, got {coord.shape}")
        if len(coord) != len(prediction):
            raise ValueError(
                f"Point count mismatch for {room_name}: "
                f"{len(coord)} coordinates, {len(prediction)} predictions"
            )
        if not np.issubdtype(prediction.dtype, np.integer):
            raise TypeError(f"Expected integer class IDs in {prediction_path}")
        if prediction.size and (prediction.min() < 0 or prediction.max() >= len(CLASS_NAMES)):
            raise ValueError(f"Prediction class IDs out of range in {prediction_path}")

        ply_path = output_dir / f"{room_name}_pred.ply"
        write_ply(ply_path, coord, prediction)
        print(f"{room_name}: {len(prediction):,} points -> {ply_path}")

    print("Class IDs:")
    for class_id, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_id}: {class_name}")


if __name__ == "__main__":
    main()
