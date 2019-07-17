#coding:utf-8
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import viz

from common.log import Logger, savefig
from torch.utils.data import DataLoader
from common.utils import AverageMeter, lr_decay, save_ckpt
import time
from progress.bar import Bar
from common.generators import PoseGenerator
from common.loss import mpjpe, p_mpjpe
from tensorboardX import SummaryWriter
import argparse

from common.h36m_dataset import CMUPanoDataset
from common.data_utils import fetch, read_3d_data, create_2d_data, get_MUCO3DHP_data
from common.graph_utils import adj_mx_from_skeleton
from models.sem_gcn import MultiSemGCN
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import datetime
import os
import numpy as np
from scipy.io import loadmat
human36m_data_path = os.path.join('data', 'data_3d_' + "h36m" + '.npz')
MUCO3DHP_path = "/home/lgh/data1/multi3Dpose/mpi_inf_3dhp/mpi_inf_3dhp_train_set/S1/Seq1/annot.mat"
data = loadmat(MUCO3DHP_path)
joint3d1 = np.array(data['annot2']).squeeze()[0]
joint3d2 = np.array(data['annot2']).squeeze()[5]
joint3d1 = joint3d1.reshape(6416,28,2)
joint3d2 = joint3d2.reshape(6416,28,2)

joint1 = joint3d1[0,:,:]
joint2 = joint3d2[1000,:,:]

ax = plt.subplot(111, projection='3d')
ax.scatter(joint1[:,0],joint1[:,1], joint1[:,2], c='r')
ax.scatter(joint2[:,0],joint2[:,1], joint2[:,2], c='g')
#viz.show3Dpose(data_3d[0,:,:], ax)
plt.show()