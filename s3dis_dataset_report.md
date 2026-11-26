# Evaluation Dataset

## S3DIS Overview

The Stanford Large-Scale 3D Indoor Spaces Dataset (S3DIS), released by Armeni et al. in 2016, is widely used for point cloud semantic segmentation. It covers six areas across three buildings, including offices, meeting rooms, and hallways. Each point has 3D coordinates, an RGB color, and one of 13 semantic labels covering building elements such as floors and walls, furniture such as tables and chairs, and clutter. Pointcept's PTv3 Area 5 protocol uses `Area_5` for testing and the other five areas for training. [Stanford dataset documentation](https://gradientspaces.stanford.edu/research#datasets) · [Pointcept PTv3 evaluation protocol](https://github.com/Pointcept/Pointcept#point-transformer-v3)

## Area 5 Evaluation Subset

The evaluation uses `data/s3dis_le_1m`, a subset of S3DIS `Area_5` preprocessed with Pointcept. We counted points by the number of rows in each room's `coord.npy` and kept rooms with **fewer than one million points**. Each selected room retains its original coordinate, RGB, normal, and annotation files.

| Statistic | Full Area 5 | Evaluation subset |
|---|---:|---:|
| Rooms | 68 | **39** |
| Total points | 78,719,063 | **29,359,083** |
| Points per room (range) | 175,538–4,481,021 | **175,538–928,712** |

The subset excludes 29 rooms and retains **57.4% of the rooms** and **37.3% of the points** in the full Area 5 split. The reported segmentation accuracy and inference performance for the PTv3 baseline and GitMerge3D use the same 39 rooms.
