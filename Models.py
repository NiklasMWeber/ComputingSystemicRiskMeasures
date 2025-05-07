from typing import List

import numpy as np
import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import SAGEConv

from EisenbergNoe import get_clearing_vector_iter_from_batch


class GNN(nn.Module):
    def __init__(self, n_in_feats: int, n_zeros: int, n_h_feats: int, n_e_feats: int, n_out_feats: int,
                 n_GNN_lay: int):
        self.n_in_feats = n_in_feats
        self.n_zeros = n_zeros
        self.n_h_feats = n_h_feats
        self.n_e_feats = n_e_feats
        self.n_out_feats = n_out_feats
        self.n_GNN_lay = n_GNN_lay
        super(GNN, self).__init__()

        # SAGEConv
        tmp_n_h_feats = 1*(n_in_feats + n_zeros)
        self.GNN_Layers = nn.ModuleList()
        for i in range(n_GNN_lay-1):
            self.GNN_Layers.append(SAGEConv(tmp_n_h_feats, n_h_feats, 'mean'))
            tmp_n_h_feats = n_h_feats
        self.GNN_Layers.append(SAGEConv(tmp_n_h_feats, n_out_feats, 'mean'))
        # self.activation = torch.nn.LeakyReLU()
        self.activation = torch.nn.Sigmoid()
        # if we want readout FNN
        self.Readout = nn.Linear(n_out_feats+n_in_feats, 1, bias=True)

    def reset_params(self):
        self.__init__(self.n_in_feats,self.n_zeros, self.n_h_feats, self.n_e_featsself, self.n_out_feats,
                      self.n_GNN_lay)

    def forward(self, g, features, e_feat):
        ones = torch.ones((len(features), 7))
        h = torch.concat((features, ones), dim=1)
        for conv in self.GNN_Layers[:-1]:
            h = conv(g, h, e_feat)
            h = self.activation(h)
        h = self.GNN_Layers[-1](g, h, e_feat)
        # if we want to use linear readout with original features and gnn features
        h = self.Readout(torch.concat((features, h), dim=1))
        return h

    def predict(self, graph, list_of_graphs, batch_of_liab, batch_of_assets, batch_of_outgoing):
        features_list = [graph.ndata['assets'], graph.ndata['outgoingLiabilities'], graph.ndata['incomingLiabilities']]
        features = torch.stack(features_list, dim=1)
        efeat = torch.reshape(graph.edata['weight'], (-1, 1))
        bailout_score = self.forward(graph, features, efeat)
        n_graphs_in_batch = len(list_of_graphs)
        bailout_score = bailout_score.split(list_of_graphs[0].num_nodes())
        bailout_score = torch.stack(bailout_score, dim=0).reshape((n_graphs_in_batch, -1))
        bailout_score = torch.softmax(bailout_score, dim=1)
        return bailout_score

class ConstantBailout(nn.Module):
    def __init__(self, n_dimension: int):
        self.n_dimension = n_dimension
        super(ConstantBailout, self).__init__()
        self.params = nn.Linear(1, self.n_dimension)  # only use bias term by feeding in zero later

    def reset_params(self):
        self.__init__(self.n_dimension)

    def forward(self, n_graphs_in_batch):
        a = torch.zeros((n_graphs_in_batch, 1))
        return self.params(a)

    def predict(self, graph, list_of_graphs, batch_of_liab, batch_of_assets, batch_of_outgoing):
        # list_graphs = list_of_graphs
        n_graphs_in_batch = len(list_of_graphs)
        bailout_score = self.forward(n_graphs_in_batch)
        # bailout_score = bailout_score.split(list_of_graphs[0].num_nodes())
        # bailout_score = torch.stack(bailout_score, dim=0).reshape((n_graphs_in_batch, -1))
        bailout_score = torch.softmax(bailout_score, dim=1)
        return bailout_score

class LinMod(nn.Module):
    def __init__(self, n_node_feat: int, bias: bool):
        super(LinMod, self).__init__()
        self.layer = nn.Linear(n_node_feat, 1, bias=bias)

    def forward(self, original_node_feat):
        h = self.layer(original_node_feat)
        return h

    def predict(self, graph, list_of_graphs, batch_of_liab, batch_of_assets, batch_of_outgoing):
        n_graphs_in_batch = len(list_of_graphs)
        n_nodes = list_of_graphs[0].num_nodes()
        features_list = [graph.ndata['assets'], graph.ndata['outgoingLiabilities'], graph.ndata['incomingLiabilities']]
        features = torch.stack(features_list, dim=1)
        features_per_node = features.reshape(n_graphs_in_batch, n_nodes, -1)
        bailout_score = self.forward(features_per_node)
        bailout_score = bailout_score.squeeze(dim=-1)
        bailout_score = torch.softmax(bailout_score, dim=1)
        return bailout_score

class NN(nn.Module):
    def __init__(self, h_feat_dims , n_out_feats: int):
        super(NN, self).__init__()
        self.activation = torch.nn.LeakyReLU()
        self.NN_Layers = nn.ModuleList()
        for i in range(len(h_feat_dims) - 1):
            self.NN_Layers.append(nn.Linear(h_feat_dims[i], h_feat_dims[i + 1]))
        self.Out = nn.Linear(h_feat_dims[-1], n_out_feats)

    def forward(self, in_feat):
        h = 1 * in_feat
        for lay in self.NN_Layers:
            h = lay(h)
            h = self.activation(h)
        h = self.Out(h)
        return h

    def predict(self, graph, list_of_graphs, batch_of_liab, batch_of_assets, batch_of_outgoing):
        features_list = [graph.ndata['assets'], graph.ndata['outgoingLiabilities'], graph.ndata['incomingLiabilities']]
        n_graphs_in_batch = len(list_of_graphs)
        features = torch.stack(features_list, dim=1)
        features_per_graph = features.reshape(n_graphs_in_batch, -1)
        bailout_score = self.forward(features_per_graph)
        bailout_score = torch.softmax(bailout_score, dim=1)
        return bailout_score

class NNL(nn.Module):
    def __init__(self, h_feat_dims, n_out_feats: int):
        super(NNL, self).__init__()
        self.n_out_feats = n_out_feats
        self.h_feat_dims = h_feat_dims
        self.activation = torch.nn.LeakyReLU()


        self.NN_Layers = nn.ModuleList()
        for i in range(len(h_feat_dims) - 1):
            self.NN_Layers.append(nn.Linear(h_feat_dims[i], h_feat_dims[i + 1]))
        self.Out = nn.Linear(h_feat_dims[-1], self.n_out_feats)

    def forward(self, in_feat):
        h = 1 * in_feat
        for lay in self.NN_Layers:
            h = lay(h)
            h = self.activation(h)
        h = self.Out(h)
        return h

    def predict(self, graph, list_of_graphs, batch_of_liab, batch_of_assets, batch_of_outgoing):
        n_graphs_in_batch = len(list_of_graphs)
        liab_vec = batch_of_liab.reshape(n_graphs_in_batch, -1)

        features_list = [graph.ndata['assets'], graph.ndata['outgoingLiabilities'], graph.ndata['incomingLiabilities']]
        features = torch.stack(features_list, dim=1)
        features_per_graph = features.reshape(n_graphs_in_batch, -1)
        features_with_liab = torch.concat([features_per_graph, liab_vec], dim=1)
        bailout_score = self.forward(features_with_liab)
        bailout_score = torch.softmax(bailout_score, dim=1)
        return bailout_score


class PENN(nn.Module):
    def __init__(self, n_node_feat: int, n_edge_feat: int, layer_size_list: [List, List, List, List], ex: bool,
                 id: bool):
        if id is True:
            n_node_feat = n_node_feat +1
        self.id = id
        self.ex = ex
        if layer_size_list[0] == [] and self.ex == True:
            raise ValueError("layer_size_list[0] must not be empty if PENN.ex is set to True")  # ' but '

        self.layer_sizes_phi = layer_size_list[1]
        self.layer_sizes_alpha = layer_size_list[2]
        self.layer_sizes_rho = layer_size_list[3]
        self.layer_sizes_psi = layer_size_list[0]

        super(PENN, self).__init__()
        self.activate = torch.nn.LeakyReLU()

        tmp_size = 2 * n_node_feat + n_edge_feat
        tmp_size_psi = 2 * n_node_feat + 2 * n_edge_feat
        self.layers = nn.ModuleList()

        if self.ex is True:
            for size in self.layer_sizes_psi:
                self.layers.append(nn.Linear(tmp_size_psi, size))
                tmp_size_psi = size

        for size in self.layer_sizes_phi:
            self.layers.append(nn.Linear(tmp_size, size))
            tmp_size = size

        tmp_size = tmp_size + n_node_feat
        for size in self.layer_sizes_alpha:
            self.layers.append(nn.Linear(tmp_size, size))
            tmp_size = size

        if self.ex is True:
            tmp_size = tmp_size + n_node_feat + tmp_size_psi
        else:
            tmp_size = tmp_size + n_node_feat

        for size in self.layer_sizes_rho:
            self.layers.append(nn.Linear(tmp_size, size))
            tmp_size = size


    def forward(self, node_feat, liab_matrix_tensor):

        node_feat_size = node_feat.size()
        number_of_nodes = node_feat_size[-2]
        repeat_dim = [1 for dim in node_feat_size]  # torch.ones(len(original_node_feat_size))
        repeat_dim.append(number_of_nodes)
        reshape_dim = list(node_feat_size)
        reshape_dim.insert(-2, -1)
        node_feat_src = node_feat.repeat(repeat_dim).reshape(reshape_dim)
        node_feat_dst = node_feat_src.transpose(dim0=-3, dim1=-2)
        h = torch.cat([node_feat_src, liab_matrix_tensor.unsqueeze(dim=-1), node_feat_dst], dim=-1)

        layer_count = 0
        if self.ex is True:
            h_psi = torch.cat([node_feat_src, liab_matrix_tensor.unsqueeze(dim=-1),
                                                  torch.transpose(liab_matrix_tensor, dim0=-2, dim1=-1).unsqueeze(dim=-1),
                                                  node_feat_dst], dim=-1)
            if len(self.layer_sizes_psi) > 1:  # last layer will be followed by activation --> need to distinguish
                for i in range(len(self.layer_sizes_psi) - 1):
                    h_psi = self.layers[layer_count](h_psi)
                    h_psi = self.activate(h_psi)
                    layer_count += 1
                h_psi = self.layers[layer_count](h_psi)
                layer_count += 1
            else:
                h_psi = self.layers[layer_count](h_psi)
                layer_count += 1
            h_psi = torch.mean(h_psi, dim=-2)


        if len(self.layer_sizes_phi) > 1:  # last layer will be followed by activation --> need to distinguish
            for i in range(len(self.layer_sizes_phi) - 1):
                h = self.layers[layer_count](h)
                h = self.activate(h)
                layer_count += 1

            h = self.layers[layer_count](h)
            layer_count += 1

        else:
            h = self.layers[layer_count](h)
            layer_count += 1

        h = torch.mean(h, dim=-2)
        h = torch.cat([node_feat, h], dim=-1)  # manipulate h according to PENN model from phi to alpha

        if len(self.layer_sizes_alpha) > 1:
            for i in range(len(self.layer_sizes_alpha) - 1):
                h = self.layers[layer_count](h)
                h = self.activate(h)
                layer_count += 1
        h = self.layers[layer_count](h)
        layer_count += 1

        h = h.mean(dim=-2, keepdim=True)  # manipulate h according to PENN from alpha to rho
        repeat_dim = [1 for _ in h.size()]  # torch.ones(len(original_node_feat_size))
        repeat_dim[-2] = number_of_nodes
        h = h.repeat(repeat_dim)
        if self.ex is True:
            h = torch.cat([node_feat, h, h_psi], dim=-1)
        else:
            h = torch.cat([node_feat, h], dim=-1)

        if len(self.layer_sizes_rho) > 1:
            for i in range(len(self.layer_sizes_rho) - 1):
                h = self.layers[layer_count](h)
                h = self.activate(h)
                layer_count += 1
        h = self.layers[layer_count](h)
        return h

    def predict(self, graph, list_of_graphs, batch_of_liab, batch_of_assets, batch_of_outgoing):
        n_graphs_in_batch = len(list_of_graphs)
        liab_vec = batch_of_liab
        features_list = [graph.ndata['assets'], graph.ndata['outgoingLiabilities'], graph.ndata['incomingLiabilities']]
        n_nodes = list_of_graphs[0].num_nodes()
        if self.id is True:
            random_node_ID_list = [np.arange(n_nodes) for _ in range(n_graphs_in_batch)]
            # for x in random_node_ID_list:
            #     np.random.shuffle(x)
            random_node_IDs = torch.tensor(np.concatenate(random_node_ID_list, axis=0))
            features_list.append(random_node_IDs)

        features = torch.stack(features_list, dim=1)
        features_per_node = features.reshape(n_graphs_in_batch, n_nodes, -1)
        bailout_score = self.forward(features_per_node, liab_vec)
        bailout_score = bailout_score.squeeze(dim=-1)
        bailout_score = torch.softmax(bailout_score, dim=1)
        return bailout_score

class UniformBailoutPredictor:
    @staticmethod
    def predict(graph, list_of_graphs, batch_of_liab, batch_of_assets, batch_of_outgoing):
        bailout_score = torch.ones(size=batch_of_assets.size()) / list_of_graphs[0].num_nodes()
        bailout_ratio = bailout_score.squeeze(-1)
        return bailout_ratio


class DefaultOnlyBailoutPredictor:
    @staticmethod
    def predict(graph, list_of_graphs, batch_of_liab, batch_of_assets, batch_of_outgoing):
        batch_of_cv = get_clearing_vector_iter_from_batch(batch_of_liab, batch_of_assets, batch_of_outgoing, n_iter = 50)
        cv_shortfall = batch_of_outgoing - batch_of_cv
        batch_of_defaults = (cv_shortfall > 1e-10)
        n_defaults = batch_of_defaults.sum(dim=-2)
        bailout_score = batch_of_defaults.squeeze(-1)/n_defaults
        bailout_ratio = bailout_score
        return bailout_ratio


class FirstLevelBailoutPredictor:
    @staticmethod
    def predict(graph, list_of_graphs, batch_of_liab, batch_of_assets, batch_of_outgoing):
        batch_of_shortfalls = batch_of_outgoing - batch_of_assets - batch_of_liab.sum(-2).unsqueeze(-1)
        batch_of_defaults = (batch_of_shortfalls > 1e-10)
        n_defaults = batch_of_defaults.sum(dim=-2)
        bailout_score = batch_of_defaults.squeeze(-1) / n_defaults
        bailout_ratio = bailout_score
        return bailout_ratio
