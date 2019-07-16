from __future__ import print_function, absolute_import

import numpy as np
import torch
from torch.utils.data import Dataset


class PoseGenerator(Dataset):
    def __init__(self, poses_3d, poses_2d, feature_mutual):
        assert poses_3d is not None
        self._poses_3d = poses_3d
        self._poses_2d = poses_2d
        self._feature_mutual = feature_mutual
        assert self._poses_3d.shape[0] == self._poses_2d.shape[0]

    def __getitem__(self, index):
        out_pose_3d = self._poses_3d[index]
        out_pose_2d = self._poses_2d[index]
        out_feature_mutual = self._feature_mutual[index]

        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        out_pose_2d = torch.from_numpy(out_pose_2d).float()
        out_feature_mutual = torch.from_numpy(out_feature_mutual).float()

        return out_pose_3d, out_pose_2d, out_feature_mutual

    def __len__(self):
        return len(self._poses_2d)
