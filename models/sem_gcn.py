#coding:utf-8
from __future__ import absolute_import
import torch
import torch.nn as nn
from models.sem_graph_conv import SemGraphConv, MultiSemGraphConv
from models.graph_non_local import GraphNonLocal


class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = SemGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x

class _MultiGraphConv(nn.Module):
    def __init__(self, person_num, adj_mutual, input_dim, output_dim, p_dropout=None):
        super(_MultiGraphConv, self).__init__()

        self.gconv = MultiSemGraphConv(input_dim, output_dim, person_num, adj_mutual)  #NN * output_dim
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x): #注意这里，输入是input[0]：joint特征; [1]：用来计算adj_mutual中的权重；输出是NN * dim的矩阵
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x)
        x = x.transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x

class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out


class _GraphNonLocal(nn.Module):
    def __init__(self, hid_dim, grouped_order, restored_order, group_size):
        super(_GraphNonLocal, self).__init__()

        self.nonlocal = GraphNonLocal(hid_dim, sub_sample=group_size)
        self.grouped_order = grouped_order
        self.restored_order = restored_order

    def forward(self, x):
        out = x[:, self.grouped_order, :]
        out = self.nonlocal(out.transpose(1, 2)).transpose(1, 2)
        out = out[:, self.restored_order, :]
        return out

class _MutualGraphNonLocal(nn.Module):
    def __init__(self, hid_dim, grouped_order, restored_order, group_size, num_person):
        super(_MutualGraphNonLocal, self).__init__()
        self.nonlocal = GraphNonLocal(hid_dim, sub_sample=group_size)
        self.grouped_order = grouped_order
        self.restored_order = restored_order
        self.num_person = num_person

    def forward(self, x): # batch 17N hid_dim
        ## only test, most lowest mode
        out = torch.zeros_like(x)
        for i in range(self.num_person):
            x_n = x[:, i::self.num_person, :]
            out_n = x_n[:, self.grouped_order, :]
            out_n = self.nonlocal(out_n.transpose(1,2)).transpose(1,2)
            out_n = out_n[:, self.restored_order, :]
            out[:, i::self.num_person, :] = out_n
        return out


class SemGCN(nn.Module):
    def __init__(self, adj, adj_mutual, hid_dim, coords_dim=(2, 3), num_layers=4, nodes_group=None):
        super(SemGCN, self).__init__()

        _gconv_input = [_GraphConv(adj, coords_dim[0], hid_dim, p_dropout=None)]  #出来是17*hid_dim的矩阵
        _gconv_layers = []

        if nodes_group is None:
            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=None))
        else:
            group_size = len(nodes_group[0])
            assert group_size > 1

            grouped_order = list(reduce(lambda x, y: x + y, nodes_group))  ##list展开
            restored_order = [0] * len(grouped_order)
            for i in range(len(restored_order)):
                for j in range(len(grouped_order)):
                    if grouped_order[j] == i:
                        restored_order[i] = j
                        break
            ## 相当于先转换为grouped_order的顺序into nonlocal layer，然后再用restored_order恢复原来的顺序
            _gconv_input.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))
            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=None))
                _gconv_layers.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))

        self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_layers = nn.Sequential(*_gconv_layers)
        self.gconv_output = SemGraphConv(hid_dim, coords_dim[1], adj)

    def forward(self, x):
        out = self.gconv_input(x)
        out = self.gconv_layers(out)
        out = self.gconv_output(out)
        return out

class MultiSemGCN(nn.Module):
    def __init__(self, adj, adj_mutual, person_num, hid_dim, coords_dim=(2, 3), num_layers=4, nodes_group=None):
        super(MultiSemGCN, self).__init__()
        _gconv_input = [_GraphConv(adj, coords_dim[0], hid_dim, p_dropout=None)]  #出来是17*hid_dim的矩阵
        _gconv_layers = []
        _gconv_mutual = []
        self.num_layers = num_layers
        if nodes_group is None:
            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=None))
        else:
            group_size = len(nodes_group[0])
            assert group_size > 1

            grouped_order = list(reduce(lambda x, y: x + y, nodes_group))  ##list展开
            restored_order = [0] * len(grouped_order)
            for i in range(len(restored_order)):
                for j in range(len(grouped_order)):
                    if grouped_order[j] == i:
                        restored_order[i] = j
                        break
            ## 相当于先转换为grouped_order的顺序into nonlocal layer，然后再用restored_order恢复原来的顺序
            _gconv_input.append(_MutualGraphNonLocal(hid_dim, grouped_order, restored_order, group_size, person_num))
            _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=None))
            _gconv_layers.append(_MutualGraphNonLocal(hid_dim, grouped_order, restored_order, group_size, person_num))
            _gconv_mutual.append(_MultiGraphConv(person_num, adj_mutual, hid_dim, hid_dim))
        self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_layers = nn.Sequential(*_gconv_layers)
        self.gconv_mutual = nn.Sequential(*_gconv_mutual)
        self.gconv_output = SemGraphConv(hid_dim, coords_dim[1], adj)

    def forward(self, x):
        out = self.gconv_input(x[0])
        for i in range(self.num_layers):
            out = self.gconv_layers(out)
            out = self.gconv_mutual([out, x[1]])
        out = self.gconv_output(out)
        return out
