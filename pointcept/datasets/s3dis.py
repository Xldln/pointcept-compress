"""
S3DIS Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import zlib

import numpy as np
from scipy.spatial import cKDTree

from .transform import index_operator
from .defaults import DefaultDataset
from .builder import DATASETS


@DATASETS.register_module()
class S3DISDataset(DefaultDataset):
    def get_data(self, idx):
        data = super().get_data(idx)
        if not self.test_mode or "randomdrop_ratio" not in self.test_cfg:
            return data

        ratio = self.test_cfg.randomdrop_ratio
        if type(ratio) not in (int, float) or not 0 <= ratio <= 1:
            raise ValueError("RandomDrop --r must be between 0 and 1")
        coords = data["coord"]
        count = len(coords)
        keep_count = max(1, count - int(count * ratio))
        if keep_count >= count:
            return data

        # One reproducible sample per room, shared by every test augmentation.
        seed = [self.test_cfg.randomdrop_seed, zlib.crc32(data["name"].encode())]
        rng = np.random.default_rng(seed)
        kept = np.sort(rng.choice(count, keep_count, replace=False))
        labels = data["segment"]
        data = index_operator(data.copy(), kept)
        _, inverse = cKDTree(data["coord"]).query(coords, k=1)
        inverse[kept] = np.arange(keep_count)
        data["origin_segment"] = labels
        data["inverse"] = inverse
        return data

    def get_data_name(self, idx):
        remain, room_name = os.path.split(self.data_list[idx % len(self.data_list)])
        remain, area_name = os.path.split(remain)
        return f"{area_name}-{room_name}"
