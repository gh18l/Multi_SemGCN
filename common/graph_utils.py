#coding:utf-8
from __future__ import absolute_import

import torch
import numpy as np
import scipy.sparse as sp


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_mx_from_edges(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1] #全1矩阵（15条边）、子节点、父节点
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32) #17*17的连接矩阵

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0])) #权重指的是每个节点
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    return adj_mx


def adj_mx_from_skeleton(skeleton, person_num, sparse=False):
    num_joints = skeleton.num_joints()
    num_person = person_num
    #edge:zip([子节点,父节点]) 除去根节点, 只取17个点
    edges = list(filter(lambda x: x[1] >= 0, zip(list(range(0, num_joints)), skeleton.parents())))

    edges = np.array(edges, dtype=np.int32)
    data = np.ones(edges.shape[0] * num_person)
    r = []
    c = []
    for i in range(num_person): #第i个人，第edges[:, 0]个joint
        tmp_r = i + edges[:, 0]*num_person
        tmp_c = i + edges[:, 1]*num_person
        r.extend(tmp_r)
        c.extend(tmp_c)
    adj_mx = sp.coo_matrix((data, (r, c)), shape=(num_joints*num_person, num_joints*num_person), dtype=np.float32)
    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))  # 权重指的是每个节点
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)

    ##mutual matrix alone
    ## 检查下是否有重复的边，使得稀疏矩阵累加
    r = []
    c = []
    data = np.ones(num_person*num_person*num_joints)

    for i in range(num_person*num_joints):
        for j in range(num_person):
            r.append(i)
            c.append(i/num_person*num_person+j)
    # for i in range(num_person):
    #     for j in range(num_joints):
    #         for k in range(num_person):
    #             if i*num_joints+j != k*num_joints+j:
    #                 r.append(i*num_joints+j)
    #                 c.append(k*num_joints+j)
    adj_mx_mutual = sp.coo_matrix((data, (r, c)), shape=(num_joints * num_person, num_joints * num_person), dtype=np.float32)
    adj_mx_mutual = normalize(adj_mx_mutual)  # 权重指的是每个节点
    if sparse:
        adj_mx_mutual = sparse_mx_to_torch_sparse_tensor(adj_mx_mutual)
    else:
        adj_mx_mutual = torch.tensor(adj_mx_mutual.todense(), dtype=torch.float)
    return adj_mx, adj_mx_mutual
