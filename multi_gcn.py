#coding:utf-8
import path
from common.h36m_dataset import CMUPanoDataset
from common.data_utils import fetch, read_3d_data, create_2d_data
from common.graph_utils import adj_mx_from_skeleton
from models.sem_gcn import MultiSemGCN
import torch.backends.cudnn as cudnn
import torch
def main():
    data_path = ""
    keypoints_path = ""
    hid_dim = 128
    num_layers = 4
    non_local = True
    print('==> Loading multi-person dataset...')
    dataset_path = path.join(data_path)
    dataset = CMUPanoDataset(dataset_path)
    dataset = read_3d_data(dataset)
    keypoints = create_2d_data(keypoints_path, dataset) #dataset用来读相机参数

    cudnn.benchmark = True
    device = torch.device("cuda")

    adj, adj_mutual = adj_mx_from_skeleton(dataset.skeleton()) #ok
    model_pos = MultiSemGCN(adj, adj_mutual, hid_dim, num_layers=num_layers,
                       nodes_group=dataset.skeleton().joints_group() if non_local else None).to(device)  #ok