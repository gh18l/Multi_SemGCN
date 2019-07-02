import path
from common.h36m_dataset import CMUPanoDataset
from common.data_utils import fetch, read_3d_data, create_2d_data
def main()
    data_path = ""
    print('==> Loading multi-person dataset...')
    dataset_path = path.join(data_path)
    dataset = CMUPanoDataset(dataset_path)
    dataset = read_3d_data(dataset)
    keypoints = create_2d_data(path.join('data', 'data_2d_' + args.dataset + '_' + args.keypoints + '.npz'), dataset)
