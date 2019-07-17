from __future__ import absolute_import, division

import numpy as np

from .camera import world_to_camera, normalize_screen_coordinates
from scipy.io import loadmat
import os

def create_2d_data(data_path, dataset):
    keypoints = np.load(data_path, allow_pickle=True)
    keypoints = keypoints['positions_2d'].item()

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

    return keypoints


def read_3d_data(dataset):
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, :] -= pos_3d[:, :1]  # Remove global offset
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

    return dataset


def fetch(subjects, dataset, keypoints, action_filter=None, stride=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_actions = []

    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    # if action.startswith(a):
                    if action.split(' ')[0] == a:
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])
                out_actions.append([action.split(' ')[0]] * poses_2d[i].shape[0])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_poses_3d) == 0:
        out_poses_3d = None

    if stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            out_actions[i] = out_actions[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_poses_3d, out_poses_2d, out_actions


def MUCO3DHP_data_filter(data_2d, data_3d, img_name):
    _data_2d = data_2d.transpose((0,2,1,3))
    _data_3d = data_3d.transpose((0, 2, 1, 3))
    img_width = 2048
    img_height = 2048
    # filt out the frame which one person is out of the image.
    filters = []
    for i in range(len(_data_2d)):
        discard = 0
        for j in range(_data_2d.shape[1]):
            total = 0
            for k in range(_data_2d.shape[2]):
                if(_data_2d[i, j, k, 0]<0 or _data_2d[i, j, k, 0]>img_width-1 or _data_2d[i, j, k, 1]<0 or _data_2d[i, j, k, 1]>img_height-1):
                    total = total + 1
            if(total > 0):  #0--only save full joint frame...>=data_2d.shape[2]--it is acceptable that some joints is out of image
                discard = 1
        if(discard==0):
            filters.append(i)
    _data_2d = _data_2d[filters,:,:,:]
    _data_3d = _data_3d[filters, :, :, :]
    img_name = img_name[filters]

    _data_2d = _data_2d.transpose((0, 2, 1, 3))
    _data_3d = _data_3d.transpose((0, 2, 1, 3))

    return _data_2d, _data_3d, img_name



def get_MUCO3DHP_data(data_path, args):
    mat_files = os.listdir(data_path)
    mat_files = sorted([filename for filename in mat_files if filename.endswith(".mat")],
                            key=lambda d: int((d.split('_')[1])))
    data_2d = []
    data_3d = []
    img_name = []
    person_num = []
    pose_num = []
    for ind, mat_file in enumerate(mat_files):
        ## num=500
        mat_file_path = os.path.join(data_path, mat_file)
        data = loadmat(mat_file_path)
        _data_3d = data["joint_loc3"]
        person_num = _data_3d.shape[2]
        pose_num = _data_3d.shape[1]
        _data_3d = list(_data_3d.transpose((3, 1, 2, 0)))  #framenum 17 numperson 3
        _data_2d = data["joint_loc2"]
        _data_2d = list(_data_2d.transpose((3, 1, 2, 0)))
        _img_name = data["img_names"]
        _img_name = list(_img_name.transpose((1, 0)))
        data_2d.append(_data_2d)
        data_3d.append(_data_3d)
        img_name.append(_img_name)
        # for i in range(len())
        # _data_2d = list(data["joint_loc2"])
        # _data_3d = list(data["joint_loc3"])
        # _img_name = data["img_names"]
        # data_2d.append(_data_2d)
        # data_3d.append(_data_3d)
        # img_name.append(_img_name)
    ## should be N * (M*17) * 3, N images, M persons per image
    data_2d = np.concatenate(data_2d)
    data_3d = np.concatenate(data_3d)
    img_name = np.concatenate(img_name)

    data_2d, data_3d, img_name = MUCO3DHP_data_filter(data_2d, data_3d, img_name)
    a = np.max(data_2d)
    b = np.min(data_2d)
    ## align the data
    frame_num = len(data_2d)
    data_2d = np.reshape(data_2d, (frame_num, -1, 2))
    data_3d = np.reshape(data_3d, (frame_num, -1, 3))

    # align data for calculating adj_mutual
    feature_mutual = np.zeros((frame_num, person_num*person_num, pose_num*2*2))
    for frame in range(frame_num):
        for i in range(person_num):
            for j in range(person_num):
                src = j
                target = i
                tmp1 = data_2d[frame, src::person_num, :]
                tmp2 = data_2d[frame, target::person_num, :]
                tmp = np.concatenate((tmp1, tmp2))
                feature_mutual[frame, target*person_num+src, :] = np.reshape(tmp, (1,-1))

    return data_2d, data_3d, img_name, feature_mutual




