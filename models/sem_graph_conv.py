#coding:utf-8
from __future__ import absolute_import, division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(SemGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.adj = adj
        self.m = (self.adj > 0)
        self.e = nn.Parameter(torch.zeros(1, len(self.m.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e.data, 1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])  ##34*128
        h1 = torch.matmul(input, self.W[1])  ##34*128

        adj = -9e15 * torch.ones_like(self.adj).to(input.device)
        adj[self.m] = self.e  ##给连接矩阵的非0值赋参数，即每一个非0值位置都是一个参数
        adj = F.softmax(adj, dim=1)  ##对这个矩阵softmax一下

        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)

        ##对于每个图节点，自己对自己的增益是一套参数，邻接点对自己的增益是另外一套参数。
        # 先把2D joint拉到128维，在根据连接矩阵作图卷积。图权重根据节点的邻居数量决定
        output = torch.matmul(adj * M, h0) + torch.matmul(adj * (1 - M), h1)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class MutualSemGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj_mutual, bias=True):
        '''
        :param in_features: dim of input feature
        :param out_features: dim of output feature
        :param adj: self-adj matrix
        :param adj_mutual: mutual-adj matrix
        :param feature_mutual: temporarily it is NN * (17*2*2) matrix, such as
        [x11,y11,...,x172,y172
         x11,y11,...,x173,y173
        ...]
        :param bias:
        '''
        super(MutualSemGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        ### mutual connection
        self.adj_mutual = adj_mutual
        self.m_mutual = (self.adj_mutual > 0)
        self.m_mutual = self.m_mutual.T ## for calculating

        ## 设连接矩阵为xJN, 在这里我们把两个mutual点之间的权重设置为[x11,y11,x21,y21,...,x172,y172]*W，W为可学习的参数。
        ## 当然，权重也可以是[xa1,ya1, xa2, ya2].
        ## above is nosense
        self.adj_mutual_W = nn.Parameter(torch.zeros(size=(self.adj.size(0)*2*2, self.adj.size(0)), dtype=torch.float))
        nn.init.xavier_uniform_(self.adj_mutual_W.data, gain=1.414)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        """
        :param input:[0] is feature, [1] is feature_mutual
        :return:
        """
        h0 = torch.matmul(input[0], self.W[0])
        h1 = torch.matmul(input[0], self.W[1])

        e = torch.matmul(input[1], self.adj_mutual_W)  ## N(N) * 17
        e = e.t()
        e = e.reshape([-1]) ##-->这样来
        adj_mutual = -9e15 * torch.ones_like(self.adj_mutual).to(input.device)

        adj_mutual[self.m_mutual] = e  ## 给连接矩阵的非0值赋参数，即每一个非0值位置都是一个参数，顺序是从上往下来
        adj_mutual = adj_mutual.t()

        adj_mutual = F.softmax(adj_mutual, dim=1)  ## 对这个矩阵softmax一下

        M = torch.eye(adj_mutual.size(0), dtype=torch.float).to(input.device)

        ##对于每个图节点，自己对自己的增益是一套参数，邻接点对自己的增益是另外一套参数。
        # 先把2D joint拉到128维，在根据连接矩阵作图卷积。图权重根据节点的邻居数量决定
        output = torch.matmul(adj_mutual * M, h0) + torch.matmul(adj_mutual * (1 - M), h1)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
