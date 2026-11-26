# 实验数据集

## S3DIS 概况

S3DIS（Stanford Large-Scale 3D Indoor Spaces Dataset）是 Armeni 等人于 2016 年发布的室内三维场景数据集，常用于点云语义分割。数据覆盖 3 栋建筑中的 6 个区域，包含办公室、会议室和走廊等室内空间。点云提供三维坐标和 RGB 颜色，并带有 13 类逐点语义标注，涵盖地板、墙面等建筑构件，桌椅等家具以及杂物。Pointcept 的 PTv3 Area 5 评估协议使用 `Area_5` 作为测试集，其余五个区域用于训练。 [Stanford 数据集说明](https://gradientspaces.stanford.edu/research#datasets) · [Pointcept PTv3 评估说明](https://github.com/Pointcept/Pointcept#point-transformer-v3)

## Area 5 评估子集

本实验使用 `Pointcept/data/s3dis_le_1m`，由 Pointcept 预处理后的 S3DIS `Area_5` 构建。依据每个房间 `coord.npy` 的行数统计点数，保留实际点数 **少于 100 万**的房间。筛选以整个房间为单位，保留房间的坐标、RGB、法线及标注均沿用原文件。

| 统计项 | 完整 Area 5 | 实验子集 |
|---|---:|---:|
| 房间数 | 68 | **39** |
| 总点数 | 78,719,063 | **29,359,083** |
| 单房间点数范围 | 175,538–4,481,021 | **175,538–928,712** |

子集排除了 29 个房间，保留了完整 Area 5 的 **57.4% 房间**和 **37.3% 点数**。本实验在同一子集上对比 PTv3 基线与 GitMerge3D 的分割精度和推理性能，报告指标均对应这 39 个房间。
